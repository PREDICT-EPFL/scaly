"""Typed entry points: the signature follows the leaf dtypes, and every leaf crosses the JIT, the C
header and the C++ header as its own C type, checked against NumPy."""

from __future__ import annotations

import ctypes
import re
import subprocess
from typing import Any

import numpy as np
import pytest

import scaly as sc
from scaly.ad.rules import ELEMENTWISE
from scaly.codegen import render_c_api_header, render_c_module
from scaly.codegen.abi import buffer_idents, c_api_signature, c_ident
from scaly.function.model import as_concrete
from scaly.ir.expr import OP_INFO
from scaly.ir.program import ProgramNode, ProgramOp, walk_program
from scaly.ir.types import CanonicalPattern
from scaly.passes.lowering import LoweringError, lower_function
from tests.codegen.test_float32 import _elementwise, _reference
from tests.compilers import c_compiler, cxx_compiler

DOUBLE = "int {}(const double** arg, double** res, int* iw, double* w, int mem)"
FLOAT = "int {}(const float** arg, float** res, int* iw, float* w, int mem)"
VOID = "int {}(const void** arg, void** res, int* iw, void* w, int mem)"
ITEMSIZE = {"float64": 8, "float32": 4, "int32": 4, "int64": 8, "bool": 1}
# A frame budget small enough that every large temporary spills, while scalars stay local.
SPILL_BUDGET = 4200


def leaf(name: str, shape: int | tuple[int, ...], dtype: str) -> Any:
  return sc.arg(name, sc.TensorType(shape if isinstance(shape, tuple) else (shape,), dtype=sc.dtypes.from_name(dtype)))


def _unary(dtype: str, n: int = 5) -> sc.Function:
  @sc.function(leaf("x", n, dtype), outputs=sc.arg("y"), name=f"typed_{dtype}")
  def fn(x: sc.Expr) -> sc.Expr:
    if dtype == "bool":
      return ~x
    if dtype in ("int32", "int64"):
      return (x * 3 - 7) // 2 + x % 4
    return (x.sin() * 2.0 + 1.0).with_lowering("block")

  return fn


def _unary_reference(dtype: str, x: np.ndarray) -> np.ndarray:
  if dtype == "bool":
    return ~x
  if dtype in ("int32", "int64"):
    return (x * 3 - 7) // 2 + x % 4
  return np.sin(x) * x.dtype.type(2) + x.dtype.type(1)


def _mixed() -> sc.Function:
  @sc.function(
    sc.group(leaf("x", 4, "float32"), leaf("i", 4, "int64"), leaf("m", 4, "bool")),
    outputs=sc.group(sc.arg("y"), sc.arg("k"), sc.arg("b")),
    name="typed_mixed",
  )
  def fn(inputs: tuple[sc.Expr, sc.Expr, sc.Expr]) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    x, i, m = inputs
    y = sc.where(m, sc.cast(x, "float64") * 2.0, sc.cast(i, "float64"))
    return y, sc.cast(i // 3, "int32"), m | (x > 0.5)

  return fn


def _mixed_inputs() -> list[np.ndarray]:
  return [np.array([0.25, 0.75, -1.5, 2.0], np.float32), np.array([-7, 5, 9, -2], np.int64), np.array([True, False, True, False])]


def _mixed_reference(x: np.ndarray, i: np.ndarray, m: np.ndarray) -> list[np.ndarray]:
  return [np.where(m, x.astype(np.float64) * 2.0, i.astype(np.float64)), (i // 3).astype(np.int32), m | (x > 0.5)]


def _values(dtype: str, n: int = 5) -> np.ndarray:
  if dtype == "bool":
    return np.array([True, False, True, True, False][:n])
  if dtype in ("int32", "int64"):
    return np.array([-9, -2, 0, 5, 2**30 if dtype == "int64" else 11][:n], dtype=dtype)
  return np.linspace(-1.25, 1.5, n, dtype=dtype)


def _assert_match(got: np.ndarray, expected: np.ndarray) -> None:
  assert got.dtype == expected.dtype
  if expected.dtype.kind == "f":
    np.testing.assert_array_max_ulp(got, expected, maxulp=4)
  else:
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize(
  "fn,expected",
  [
    (_unary("float64"), DOUBLE),
    (_unary("float32"), FLOAT),
    (_unary("int32"), VOID),
    (_unary("int64"), VOID),
    (_unary("bool"), VOID),
    (_mixed(), VOID),
  ],
)
def test_signature_follows_the_leaf_dtypes(fn, expected):
  symbol = c_ident(as_concrete(fn).name)
  assert c_api_signature(fn) == expected.format(symbol)
  assert expected.format(symbol) + ";" in render_c_api_header(fn)
  assert expected.format(symbol) + ";" in render_c_api_header(fn, lang="cpp")
  assert expected.format(symbol) + " {" in render_c_module(fn).body


@pytest.mark.parametrize("dtype", ["float64", "float32", "int32", "int64", "bool"])
def test_each_leaf_dtype_matches_numpy_through_the_jit(dtype):
  x = _values(dtype)
  _assert_match(np.asarray(_unary(dtype)(x)), _unary_reference(dtype, x))


def test_mixed_leaves_match_numpy_through_the_jit():
  for got, expected in zip(_mixed()(tuple(_mixed_inputs())), _mixed_reference(*_mixed_inputs()), strict=True):
    _assert_match(np.asarray(got), expected)


def test_jit_casts_inputs_within_their_kind_and_refuses_across_kinds():
  np.testing.assert_array_equal(_unary("float32")(np.zeros(5)), np.ones(5, np.float32))
  with pytest.raises(TypeError, match="float64.*int64"):
    _unary("int64")(np.zeros(5))


_C_DRIVER = """
#include <stdlib.h>
#include <string.h>
#include "{header}"

int run({params}) {{
{copy_in}
  {symbol}_workspace_t* ws = malloc(sizeof *ws);
  int status = {symbol}_call({call});
{copy_out}
  free(ws);
  return status;
}}
"""

_CPP_DRIVER = """
#include <cstring>
#include <memory>
#include "{header}"

extern "C" int run({params}) {{
{copy_in}
  auto ws = std::make_unique<{symbol}::workspace_t>();
  int status = {symbol}::call({call});
{copy_out}
  return status;
}}
"""


def _build_driver(tmp_path, fun: sc.Function, lang: str, flags: tuple[str, ...] = (), frame_budget: int = 65536) -> ctypes.CDLL:
  """A shared library whose ``run`` copies flat arrays into the typed buffers, calls the typed
  wrapper with a heap workspace, and copies the outputs back."""
  module = render_c_module(fun, lang=lang, frame_budget=frame_budget)
  concrete = as_concrete(fun)
  symbol = c_ident(concrete.name)
  (tmp_path / module.header_name).write_text(module.header)
  (tmp_path / module.source_name).write_text(module.source)
  inputs, outputs = buffer_idents(concrete)
  prefix = f"{symbol}::" if lang == "cpp" else f"{symbol}_"
  params = ", ".join(f"const void* p_{n}" for n in inputs) + ", " + ", ".join(f"void* p_{n}" for n in outputs)
  copy_in = "\n".join(f"  {prefix}{n}_t {n}; memcpy({n}.data, p_{n}, sizeof {n}.data);" for n in inputs)
  copy_in += "\n" + "\n".join(f"  {prefix}{n}_t {n};" for n in outputs)
  copy_out = "\n".join(f"  memcpy(p_{n}, {n}.data, sizeof {n}.data);" for n in outputs)
  if lang == "cpp":
    call = ", ".join([*inputs, *outputs, "*ws"])
  else:
    call = ", ".join([*(f"&{n}" for n in (*inputs, *outputs)), "ws"])
  driver = (_CPP_DRIVER if lang == "cpp" else _C_DRIVER).format(
    header=module.header_name, params=params, copy_in=copy_in, copy_out=copy_out, symbol=symbol, call=call
  )
  driver_path = tmp_path / f"driver.{'cpp' if lang == 'cpp' else 'c'}"
  driver_path.write_text(driver)
  kernel, library = tmp_path / "kernel.o", tmp_path / "driver.so"
  subprocess.run([*c_compiler(), "-O2", *flags, "-fPIC", "-c", str(tmp_path / module.source_name), "-o", str(kernel)], check=True)
  compiler = [*cxx_compiler(), "-std=c++17"] if lang == "cpp" else [*c_compiler(), "-std=c11"]
  subprocess.run(
    [*compiler, "-O2", *flags, "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared", str(driver_path), str(kernel), "-lm", "-o", str(library)],
    check=True,
    capture_output=True,
    text=True,
  )
  return ctypes.CDLL(str(library))


def _run_driver(library: ctypes.CDLL, fun: sc.Function, inputs: list[np.ndarray]) -> list[np.ndarray]:
  concrete = as_concrete(fun)
  outputs = [np.empty(e.type.storage_shape, dtype=e.type.dtype.numpy()) for e in concrete.outputs]
  arrays = [np.ascontiguousarray(a) for a in inputs]
  assert library.run(*(ctypes.c_void_p(a.ctypes.data) for a in (*arrays, *outputs))) == 0
  return outputs


@pytest.mark.parametrize("lang", ["c", "cpp"])
@pytest.mark.parametrize("dtype", ["float64", "float32", "int32", "int64", "bool"])
def test_each_leaf_dtype_matches_numpy_through_the_typed_headers(tmp_path, lang, dtype):
  fun = _unary(dtype)
  x = _values(dtype)
  header = render_c_api_header(fun, lang=lang)
  c_type = {"float64": "double", "float32": "float", "int32": "int32_t", "int64": "int64_t", "bool": "bool"}[dtype]
  assert (f"Buffer<{c_type}, 5>" if lang == "cpp" else f"{c_type} data[5];") in header
  (got,) = _run_driver(_build_driver(tmp_path, fun, lang), fun, [x])
  _assert_match(got, _unary_reference(dtype, x))


@pytest.mark.parametrize("lang", ["c", "cpp"])
def test_mixed_leaves_match_numpy_through_the_typed_headers(tmp_path, lang):
  fun = _mixed()
  got = _run_driver(_build_driver(tmp_path, fun, lang), fun, _mixed_inputs())
  for value, expected in zip(got, _mixed_reference(*_mixed_inputs()), strict=True):
    _assert_match(value, expected)


def _spilling(n: int = 1100) -> sc.Function:
  """Large float32, int64 and bool temporaries, each read twice in different orders so each owns a slot."""
  perm = np.arange(n)[::-1].copy()

  @sc.function(sc.group(leaf("x", n, "float32"), leaf("i", n, "int64")), outputs=sc.group(sc.arg("y"), sc.arg("k"), sc.arg("c")), name="typed_spill")
  def fn(inputs: tuple[sc.Expr, sc.Expr]) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    x, i = inputs
    a, j = x.sin(), (i * 3) // 7
    m = a > 0.25
    count = sc.cast(m, "int32")
    return a.sum() + a.gather(perm).sum(), j.sum() + j.gather(perm).sum(), count.sum() + sc.cast(m.gather(perm), "int32").sum()

  return fn


def _slots(prog) -> list:
  return [n for n in walk_program(prog) if n.op == ProgramOp.BUFFER and "workspace_offset" in n.attrs]


def test_spilled_slots_of_mixed_dtypes_start_at_64_byte_offsets():
  prog = lower_function(_spilling(), lanes=1, frame_budget=SPILL_BUDGET)
  slots = {(n.attrs["name"], n.attrs["workspace_offset"], n.dtype.name, n.attrs["shape"][0]) for n in _slots(prog)}
  assert {dtype for _, _, dtype, _ in slots} >= {"float32", "int64", "bool"}
  for _, offset, _, _ in slots:
    assert offset % 64 == 0
  sz_w = int(prog.args[-1].attrs["sz_w"])
  assert 4096 < int(prog.args[-1].attrs["frame_bytes"]) <= SPILL_BUDGET
  assert sz_w >= max(offset + size * ITEMSIZE[dtype] for _, offset, dtype, size in slots)
  assert sz_w % 64 == 0
  ends = sorted((offset, offset + size * ITEMSIZE[dtype]) for _, offset, dtype, size in slots)
  assert all(a_end <= b_start for (_, a_end), (b_start, _) in zip(ends, ends[1:], strict=False))
  x, i = np.linspace(0.1, 2, 1100, dtype=np.float32), np.arange(-500, 600, dtype=np.int64)
  y, k, c = _spilling()((x, i))
  assert y.dtype == np.float32 and k.dtype == np.int64 and c.dtype == np.int32
  np.testing.assert_allclose(y, 2 * np.sin(x).sum(dtype=np.float32), rtol=1e-5)
  assert k == 2 * ((i * 3) // 7).sum() and c == 2 * (np.sin(x) > 0.25).sum()


def test_float32_workspace_counts_floats_at_64_byte_offsets():
  @sc.function(leaf("x", 1030, "float32"), outputs=sc.group(sc.arg("s"), sc.arg("t")), name="float32_spill")
  def fn(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    a = x.sin().block()
    b = x[:1025].cos().block()
    return a.sum() + b.sum(), (a[:1025] * b).sum()

  prog = lower_function(fn, lanes=1, frame_budget=SPILL_BUDGET)
  slots = _slots(prog)
  assert len(slots) == 2 and all(n.dtype == sc.dtypes.float32 for n in slots)
  assert sorted(n.attrs["workspace_offset"] for n in slots) == [0, 1040]
  assert int(prog.args[-1].attrs["sz_w"]) == 1040 + 1040
  body = render_c_module(fn, frame_budget=SPILL_BUDGET).body
  assert FLOAT.format("float32_spill") + " {" in body and re.search(r"float\* s\d+ = w \+ 1040;", body)
  x = np.linspace(0.1, 1, 1030, dtype=np.float32)
  s, t = fn(x)
  np.testing.assert_allclose([s, t], [np.sin(x).sum() + np.cos(x[:1025]).sum(), (np.sin(x[:1025]) * np.cos(x[:1025])).sum()], rtol=1e-4)


def test_float64_spills_are_padded_to_64_bytes():
  @sc.function(sc.arg("x", 1027), outputs=sc.group(sc.arg("s"), sc.arg("t")), name="padded_spill")
  def fn(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    a = x.sin().block()
    b = x[:1025].cos().block()
    return (a.sum() + b.sum()), (a[:1025] * b).sum()

  prog = lower_function(fn, lanes=1, frame_budget=SPILL_BUDGET)
  offsets = sorted(n.attrs["workspace_offset"] for n in _slots(prog))
  assert len(offsets) == 2 and all(offset % 8 == 0 for offset in offsets)
  assert int(prog.args[-1].attrs["sz_w"]) % 8 == 0
  assert DOUBLE.format("padded_spill") + " {" in render_c_module(fn, frame_budget=SPILL_BUDGET).body


def _siblings() -> sc.Function:
  """Two noinline-free block callees of different dtypes, each spilling, called one after the other."""
  n = 1100

  @sc.function(leaf("x", n, "float32"), outputs=sc.arg("y"), name="float_stage")
  def float_stage(x: sc.Expr) -> sc.Expr:
    a = x.cos().block()
    return (a * a + a).block()

  @sc.function(leaf("i", n, "int64"), outputs=sc.arg("j"), name="int_stage")
  def int_stage(i: sc.Expr) -> sc.Expr:
    a = (i * 5).block()
    return (a // 3 + a).block()

  @sc.function(sc.group(leaf("x", n, "float32"), leaf("i", n, "int64")), outputs=sc.group(sc.arg("y"), sc.arg("j")), name="typed_siblings")
  def fn(inputs: tuple[sc.Expr, sc.Expr]) -> tuple[sc.Expr, sc.Expr]:
    x, i = inputs
    y = float_stage(x)
    j = int_stage(i + sc.cast(y, "int64"))
    return y + float_stage(y), j

  return fn


def _siblings_reference(x: np.ndarray, i: np.ndarray) -> list[np.ndarray]:
  def float_stage(v):
    a = np.cos(v)
    return a * a + a

  def int_stage(v):
    a = v * 5
    return a // 3 + a

  y = float_stage(x)
  return [y + float_stage(y), int_stage(i + y.astype(np.int64))]


def test_byte_workspace_gives_sibling_callees_disjoint_windows():
  prog = lower_function(_siblings(), lanes=1, frame_budget=SPILL_BUDGET)
  calls = [n for n in walk_program(prog.args[-1]) if n.op == ProgramOp.CALL and n.attrs.get("callee_needs_w")]
  windows = {n.attrs["callee"]: int(n.attrs["w_self"]) for n in calls}
  assert len(windows) == 2 and len(set(windows.values())) == 2
  sizes = {pr.attrs["name"]: int(pr.attrs["sz_w"]) for pr in prog.args}
  (first, start), (second, end) = sorted(windows.items(), key=lambda item: item[1])
  assert start % 64 == 0 and end % 64 == 0 and start + sizes[first] <= end
  assert end + sizes[second] <= int(prog.args[-1].attrs["sz_w"])


@pytest.mark.parametrize("lang", ["c", "cpp"])
def test_byte_workspace_is_correct_under_strict_aliasing(tmp_path, lang):
  fun = _siblings()
  x = np.linspace(-3, 3, 1100, dtype=np.float32)
  i = np.arange(-550, 550, dtype=np.int64)
  library = _build_driver(tmp_path, fun, lang, flags=("-O3", "-fstrict-aliasing", "-Wstrict-aliasing"), frame_budget=SPILL_BUDGET)
  for got, expected in zip(_run_driver(library, fun, [x, i]), _siblings_reference(x, i), strict=True):
    _assert_match(got, expected)
  for got, expected in zip(fun((x, i)), _siblings_reference(x, i), strict=True):
    _assert_match(np.asarray(got), expected)


_OPS = list(ELEMENTWISE)


@pytest.mark.parametrize("mode", ["block", "scalar"])
def test_every_elementwise_op_runs_with_float32_leaves(mode):
  """The float32 parity gate: every row of the elementwise table, with float32 leaves and a
  float32 output, through the JIT."""
  values = np.linspace(0.125, 0.875, 17, dtype=np.float32)
  other = values[::-1].copy()

  @sc.function(leaf("x", 17, "float32"), leaf("y", 17, "float32"), outputs=sc.arg("out"), name=f"float32_leaf_ops_{mode}")
  def fn(x: sc.Expr, y: sc.Expr) -> sc.Expr:
    rows = [_elementwise(op, x, y) for op in _OPS]
    return sc.stack([r if r.type.dtype == sc.dtypes.float32 else sc.cast(r, "float32") for r in rows]).with_lowering(mode)

  assert c_api_signature(as_concrete(fn)) == FLOAT.format(c_ident(as_concrete(fn).name))
  results = np.asarray(fn(values, other))
  assert results.dtype == np.float32
  for op, got in zip(_OPS, results, strict=True):
    expected = np.asarray(_reference(op, values, other), dtype=np.float32)
    np.testing.assert_array_max_ulp(got, expected, maxulp=4)


_SMOOTH = [op for op in _OPS if OP_INFO[op].differentiable and op not in {sc.ExprOp.CAST}]


@pytest.mark.parametrize("op", _SMOOTH, ids=lambda op: op.name)
def test_float32_derivatives_match_float64_with_float32_leaves(op):
  values = np.linspace(0.125, 0.875, 6, dtype=np.float32)
  other = values[::-1].copy()

  def build(dtype: str) -> sc.Function:
    @sc.function(leaf("x", 6, dtype), leaf("y", 6, dtype), outputs=sc.arg("out"), name=f"derivative_{op.name.lower()}_{dtype}")
    def fn(x: sc.Expr, y: sc.Expr) -> sc.Expr:
      return _elementwise(op, x, y).sum()

    return fn

  jac32 = sc.jacobian(build("float32"), "out", "x")
  jac64 = sc.jacobian(build("float64"), "out", "x")
  assert c_api_signature(as_concrete(jac32)).startswith("int ") and "const float**" in c_api_signature(as_concrete(jac32))
  got = np.asarray(jac32(values, other))
  assert got.dtype == np.float32
  np.testing.assert_allclose(got, np.asarray(jac64(values.astype(float), other.astype(float))), rtol=1e-5, atol=1e-6)


def test_casadi_layer_stays_float64():
  with pytest.raises(ValueError, match="casadi=True.*float32"):
    render_c_module(_unary("float32"), casadi=True)


@pytest.mark.solver("piqp")
def test_solver_graphs_stay_float64():
  from tests.solvers.problem_helpers import build_qp

  qp = build_qp(P=sc.const(np.eye(2)), c=-sc.Expr.sym("mu", (2,)), name="float64_qp")

  @sc.function(leaf("mu", 2, "float32"), outputs=sc.arg("cost"), name="float32_qp_host")
  def host(mu: sc.Expr) -> sc.Expr:
    return sc.cast(sc.sumsqr(qp(sc.cast(mu, "float64"))[0]), "float32")

  with pytest.raises(LoweringError, match="solver.*float64"):
    render_c_module(host)


def test_no_float64_constant_remains_public():
  assert not hasattr(sc, "C_API_SIGNATURE")


@pytest.mark.parametrize("layout", ["csc", "csr"])
def test_sparse_float32_leaves_keep_their_dtype_through_the_jit(layout):
  from scipy import sparse

  dense = np.array([[1.5, 0.0, -2.0], [0.0, 3.0, 0.0]], dtype=np.float32)
  pattern = CanonicalPattern((2, 3), (0, 0, 1), (0, 2, 1))
  type_ = sc.TensorType((2, 3), dtype=sc.dtypes.float32, pattern=pattern)

  @sc.function(sc.arg("a", type_, layout=layout), outputs=sc.arg("b", type_, layout=layout), name=f"sparse_float32_{layout}")
  def fn(a: sc.Expr) -> sc.Expr:
    return a * 2.0

  out = sparse.coo_array(fn(sparse.csc_array(dense.astype(np.float64))))
  assert out.data.dtype == np.float32
  np.testing.assert_array_equal(out.toarray(), 2 * dense)


@pytest.mark.parametrize(
  "dtype,name", [("int32", "int32_t"), ("int64", "int64_t"), ("bool", "uint8_t"), ("int64", "scaly_floordiv"), ("int32", "scaly_floormod")]
)
def test_leaf_names_never_shadow_c_names(dtype, name):
  @sc.function(leaf(name, 2, dtype), outputs=sc.arg("y"), name=f"shadow_{dtype}_{name}")
  def fn(x: sc.Expr) -> sc.Expr:
    return x if dtype == "bool" else x // 2 + x % 3

  values = np.array([0, 1], dtype=dtype) if dtype == "bool" else np.array([-7, 5], dtype=dtype)
  np.testing.assert_array_equal(fn(values), values if dtype == "bool" else values // 2 + values % 3)


@pytest.mark.parametrize("dtype", ["float32", "int32", "int64", "bool"])
def test_mapped_calls_keep_the_leaf_dtype(dtype):
  pattern = CanonicalPattern((1, 1), (0,), (0,))
  type_ = sc.TensorType((1, 1), dtype=sc.dtypes.from_name(dtype), pattern=pattern)

  @sc.function(sc.arg("x", type_), outputs=sc.arg("y", type_), name=f"sparse_map_{dtype}")
  def fn(x: sc.Expr) -> sc.Expr:
    return x

  from scipy import sparse

  # An integer input rounds once to float32, as in an unmapped call.
  value = (1 << 62) + (1 << 38) + 1 if dtype == "float32" else 1
  values = sparse.csc_array(np.array([[value]], dtype="int64" if dtype == "float32" else dtype))
  direct = sparse.coo_array(fn(values))
  mapped = sparse.coo_array(sc.vmap(fn, 2)(sc.broadcast(values)))
  np.testing.assert_array_equal(mapped.data, np.repeat(direct.data, 2))
  assert mapped.data.dtype == direct.data.dtype == np.dtype(dtype)

  @sc.function(leaf("x", 2, dtype), outputs=sc.arg("y"), name=f"empty_map_{dtype}")
  def dense(x: sc.Expr) -> sc.Expr:
    return x

  empty = sc.vmap(dense, 0)(np.empty((0, 2), dtype=dtype))
  assert empty.shape == (0, 2) and empty.dtype == np.dtype(dtype)


def test_float32_broadcast_map_gradient():
  @sc.function(leaf("x", 2, "float32"), leaf("p", 2, "float32"), outputs=sc.arg("y"), name="broadcast_float32_stage")
  def stage(x: sc.Expr, p: sc.Expr) -> sc.Expr:
    return x * p

  @sc.function(leaf("x", (3, 2), "float32"), leaf("p", 2, "float32"), outputs=sc.arg("out"), name="broadcast_float32_host")
  def host(x: sc.Expr, p: sc.Expr) -> sc.Expr:
    return sc.vmap(stage, 3)(x, sc.broadcast(p)).sum()

  values = np.arange(6, dtype=np.float32).reshape(3, 2)
  got = np.asarray(sc.gradient(host, "out", "p")(values, np.ones(2, np.float32)))
  assert got.dtype == np.float32
  np.testing.assert_array_equal(got, values.sum(axis=0))


def test_mapped_gradients_of_mixed_dtype_formals():
  @sc.function(leaf("a", 2, "float32"), sc.arg("b", 2), outputs=sc.arg("y"), name="mixed_active_stage")
  def stage(a: sc.Expr, b: sc.Expr) -> sc.Expr:
    return sc.cast(a, "float64") * b * b

  @sc.function(sc.arg("x", (3, 2)), outputs=sc.arg("out"), name="mixed_active_host")
  def host(x: sc.Expr) -> sc.Expr:
    return sc.vmap(stage, 3)(sc.cast(x, "float32"), x).sum()

  # The float32 formal's cotangent rounds to float32, the float64 one must not.
  x = np.arange(6, dtype=np.float64).reshape(3, 2) / 3 + 0.1
  expected = (x * x).astype(np.float32).astype(np.float64) + 2 * x.astype(np.float32).astype(np.float64) * x
  np.testing.assert_allclose(sc.gradient(host, "out", "x")(x), expected, rtol=1e-14)


def test_verifier_rejects_a_call_whose_argument_dtype_differs_from_the_callee():
  from scaly.ir import program as p
  from scaly.ir.program_spec import verify_program
  from scaly.ir.spec import VerifyError
  from scaly.ir.types import dtypes

  formal = p.buffer("x", dtypes.float32, (2,))
  callee = p.proc("stage", [formal], [])
  actual = p.buffer("y", dtypes.float64, (2,))
  call = ProgramNode(ProgramOp.CALL, (actual,), {"callee": "stage", "n_in": 1, "n_out": 0}, dtype=dtypes.float64)
  with pytest.raises(VerifyError, match="passes float64 for its float32 parameter 'x'"):
    verify_program(p.program([callee, p.proc("host", [actual], [call])]))


def test_float32_column_major_and_pattern_hole_leaves_round_once():
  from scipy import sparse

  value = (1 << 62) + (1 << 38) + 1
  rounded = np.asarray(value).astype(np.float32).item()

  @sc.function(sc.arg("a", leaf("a", (2, 3), "float32").decls[0], layout="column-major"), outputs=sc.arg("b"), name="float32_column_major")
  def columns(a: sc.Expr) -> sc.Expr:
    return a.T * 2.0

  dense = np.arange(6, dtype=np.int64).reshape(2, 3) + value
  out = np.asarray(columns(dense))
  assert out.dtype == np.float32
  np.testing.assert_array_equal(out, (dense.astype(np.float32) * np.float32(2)).T)

  hole: Any = sc.function(sc.arg("x", sc.TensorType((2, 2), dtype=sc.dtypes.float32), pattern=...), name="float32_hole")(lambda x: x * 2.0)
  source = sparse.coo_array((np.array([value], dtype=np.int64), ([0], [1])), shape=(2, 2))
  got = sparse.coo_array(hole(source))
  assert got.data.dtype == np.float32 and got.data[0] == np.float32(rounded) * 2


def test_external_functions_take_a_double_workspace_inside_a_byte_one():
  from tests.function.test_extern import pair

  external = pair()

  @sc.function(
    sc.arg("x", 3), leaf("k", 3, "int64"), outputs=sc.group(sc.arg("squared"), sc.arg("total"), sc.arg("doubled")), name="typed_extern_host"
  )
  def host(x: sc.Expr, k: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    squared, total = external(x)
    return squared, total, k * 2

  assert c_api_signature(host) == VOID.format("typed_extern_host")
  assert "(double*)" in render_c_module(host).body
  x, k = np.array([1.0, 2.0, 3.0]), np.array([4, 5, 6])
  squared, total, doubled = host(x, k)
  np.testing.assert_array_equal(squared, x * x)
  assert total == x.sum() + 0.75
  _assert_match(np.asarray(doubled), 2 * k)

  @sc.function(leaf("x", 3, "float32"), outputs=sc.arg("total"), name="float32_extern_host")
  def float_host(x: sc.Expr) -> sc.Expr:
    return sc.cast(external(sc.cast(x, "float64"))[1], "float32")

  with pytest.raises(LoweringError, match="scaly_ext_pair.*double workspace"):
    lower_function(float_host)

  narrow = sc.extern(external.library, "scaly_ext_narrow", inputs=(leaf("x", 3, "float32"),), outputs=sc.arg("y", 3))

  @sc.function(leaf("x", 3, "float32"), outputs=sc.arg("y"), name="narrow_extern_host")
  def narrow_host(x: sc.Expr) -> sc.Expr:
    return narrow(x)

  with pytest.raises(LoweringError, match="external C function needs float64 leaves"):
    lower_function(narrow_host)
