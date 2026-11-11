"""Dtype-faithful scalar operations and lowering, before typed public entry points."""

from __future__ import annotations

import ctypes
import math
import subprocess

import numpy as np
import pytest

import scaly as sc
from scaly.ad.rules import ELEMENTWISE
from scaly.codegen.c import _emit_scalar, _includes, _render_raw_callee
from scaly.ir import program as p
from scaly.ir.expr import OP_INFO, _reduce
from scaly.ir.program import ProgramNode, ProgramOp, walk_program
from scaly.ir.program_spec import verify_program
from scaly.ir.spec import VerifyError
from scaly.ir.types import CanonicalPattern, TensorType, dtypes
from scaly.passes.arith import constant, fold_program
from scaly.passes.lowering import lower_function, main_proc
from scaly.passes.program.coalesce_stores import coalesce_stores
from scaly.passes.program.fold_arith import fold_arith
from scaly.passes.program.pack_workspace import pack_workspace
from scaly.passes.program.widen_ranges import widen_ranges
from tests.compilers import c_compiler


@pytest.mark.parametrize("op", [ProgramOp.ADD, ProgramOp.SUB, ProgramOp.MUL, ProgramOp.DIV, ProgramOp.POW])
def test_folding_rounds_every_float32_operation(op):
  a, b = constant(16777216, dtypes.float32), constant(1.5, dtypes.float32)
  if op in (ProgramOp.MUL, ProgramOp.POW):
    a, b = constant(1.0000001192092896, dtypes.float32), constant(1.5, dtypes.float32)
  row = next(row for row in ELEMENTWISE.values() if row.program == op)
  expected = row.numpy(np.float32(a.attrs["value"]), np.float32(b.attrs["value"]))
  folded = fold_program(ProgramNode(op, (a, b), dtype=dtypes.float32))
  assert folded.attrs["value"] == float(expected)


@pytest.mark.parametrize("value", [(1 << 62) + (1 << 38) + 1, -((1 << 62) + (1 << 38) + 1)])
def test_int64_to_float32_folding_rounds_once(value):
  expected = np.array(value, dtype=np.int64).astype(np.float32).item()
  cast = ProgramNode(ProgramOp.CAST, (p.const_int(value),), dtype=dtypes.float32)
  assert fold_program(cast).attrs["value"] == expected
  assert constant(value, dtypes.float32).attrs["value"] == expected
  assert p.const_float(value, dtypes.float32).attrs["value"] == expected
  expr = sc.const(value, dtype="float32")
  assert expr.value is not None and expr.value.item() == expected


@pytest.mark.parametrize("value,dtype", [(-1, "uint8"), (1 << 31, "int32"), (1 << 63, "int64")])
def test_float32_literal_conversion_keeps_integer_range_checks(value, dtype):
  with pytest.raises(OverflowError):
    sc.const(value, dtype=dtype)


@pytest.mark.parametrize("mode", ["block", "scalar"])
def test_python_integer_literals_in_float32_expressions_round_once(mode):
  value = (1 << 62) + (1 << 38) + 1

  @sc.function(sc.arg("x", ()), outputs=sc.arg("out"), name=f"python_midpoint_{mode}")
  def fn(x):
    return sc.cast(sc.cast(x, "float32") * 0 + value, "float64").with_lowering(mode)

  assert fn(np.array(1.0)) == np.asarray(value).astype(np.float32).item()


@pytest.mark.parametrize("mode", ["block", "scalar"])
def test_scalarizing_a_call_keeps_int64_to_float32_rounding(mode):
  value = (1 << 62) + (1 << 38) + 1

  @sc.function(sc.arg("x", ()), outputs=sc.arg("zero"), name="midpoint_zero")
  def hidden_zero(x):
    return (x * 0).scalar()

  @sc.function(sc.arg("x", ()), outputs=sc.arg("out"), name=f"midpoint_{mode}")
  def fn(x):
    integer = sc.cast(hidden_zero(x), "int64") + sc.const(np.array(value, dtype=np.int64))
    return sc.cast(sc.cast(integer, "float32"), "float64").with_lowering(mode)

  assert fn(np.array(1.0)) == np.array(value, dtype=np.int64).astype(np.float32).astype(np.float64).item()


def _run_raw(prog, tmp_path, inputs, outputs, *, dialect="gnu"):
  verify_program(prog)
  params = prog.args[-1].args[: prog.args[-1].attrs["param_count"]]
  declarations = ", ".join(f"{param.dtype.c_type}* a{i}" for i, param in enumerate(params))
  arguments = ", ".join(f"a{i}" for i in range(len(params)))
  source = "\n".join(
    [
      *_includes(prog=prog, dialect=dialect),
      *(line for proc in prog.args for line in _render_raw_callee(proc, dialect=dialect)),
      f"void run({declarations}) {{ {prog.args[-1].attrs['name']}_raw({arguments}, NULL); }}",
    ]
  )
  path, library = tmp_path / "kernel.c", tmp_path / "kernel.so"
  path.write_text(source)
  subprocess.run(
    [*c_compiler(), "-std=c99", "-O2", "-ffp-contract=off", "-shared", "-fPIC", str(path), "-lm", "-o", str(library)], check=True, capture_output=True
  )
  function = ctypes.CDLL(str(library)).run
  function.argtypes = [ctypes.c_void_p] * len(params)
  function(*(v.ctypes.data for v in [*inputs, *outputs]))
  return source


def _elementwise(op, x, y):
  if op in {sc.ExprOp.AND, sc.ExprOp.OR}:
    args = (x < y, x > 0.5)
  elif op == sc.ExprOp.NOT:
    args = (x < y,)
  elif op == sc.ExprOp.SELECT:
    args = (x < y, x, y)
  elif op in {sc.ExprOp.FLOORDIV, sc.ExprOp.MOD}:
    args = (sc.cast(x * 10, "int64"), sc.cast(y * 3, "int64"))
  elif op == sc.ExprOp.CAST:
    return sc.cast(sc.cast(x, "float64"), "float32")
  else:
    args = (x, y) if OP_INFO[op].arity == 2 else (x,)
  dtype = (
    dtypes.bool_
    if op in {sc.ExprOp.LT, sc.ExprOp.LE, sc.ExprOp.EQ, sc.ExprOp.NE, sc.ExprOp.AND, sc.ExprOp.OR, sc.ExprOp.NOT, sc.ExprOp.ISFINITE}
    else args[-1].type.dtype
  )
  return sc.Expr(op, args, TensorType(x.shape, dtype))


def _reference(op, x, y):
  if op in {sc.ExprOp.AND, sc.ExprOp.OR}:
    return ELEMENTWISE[op].numpy(x < y, x > 0.5)
  if op == sc.ExprOp.NOT:
    return ~(x < y)
  if op == sc.ExprOp.SELECT:
    return np.where(x < y, x, y)
  if op in {sc.ExprOp.FLOORDIV, sc.ExprOp.MOD}:
    with np.errstate(divide="ignore"):
      return ELEMENTWISE[op].numpy((x * 10).astype(np.int64), (y * 3).astype(np.int64))
  if op == sc.ExprOp.CAST:
    return x
  if op == sc.ExprOp.ERF:
    return np.array([math.erf(float(v)) for v in x], dtype=np.float32)
  return ELEMENTWISE[op].numpy(x, y) if OP_INFO[op].arity == 2 else ELEMENTWISE[op].numpy(x)


@pytest.mark.parametrize("mode", ["block", "scalar", "folded"])
def test_every_elementwise_op_has_a_float32_twin(tmp_path, mode):
  values = np.linspace(0.125, 0.875, 17, dtype=np.float32)
  other = values[::-1].copy()
  ops = list(ELEMENTWISE)

  @sc.function(sc.arg("x", 17), sc.arg("y", 17), outputs=sc.arg("out"), name="float32_ops")
  def fn(x, y):
    a, b = (sc.const(values), sc.const(other)) if mode == "folded" else (sc.cast(x, "float32"), sc.cast(y, "float32"))
    return sc.stack([sc.cast(_elementwise(op, a, b), "float64") for op in ops]).with_lowering("scalar" if mode == "scalar" else "block")

  prog = lower_function(fn)
  assert bool(main_proc(prog).attrs.get("scalarized")) == (mode == "scalar")
  results = np.empty((len(ops), 17))
  source = _run_raw(prog, tmp_path, [values.astype(float), other.astype(float)], [results])
  if mode != "folded":
    for row in ELEMENTWISE.values():
      if row.c.isidentifier():
        assert f"{row.c}f(" in source
  for op, got in zip(ops, results, strict=True):
    expected = np.asarray(_reference(op, values, other), dtype=np.float32)
    np.testing.assert_array_max_ulp(got.astype(np.float32), expected, maxulp=4)


@pytest.mark.parametrize(
  "op",
  [
    op for op, row in ELEMENTWISE.items() if row.c.isidentifier() or op in {sc.ExprOp.NEG, sc.ExprOp.ADD, sc.ExprOp.SUB, sc.ExprOp.MUL, sc.ExprOp.DIV}
  ],
)
def test_program_math_folding_matches_numpy_float32(op):
  values = [np.float32(0.31234567), np.float32(0.71234567)][: OP_INFO[op].arity]
  node = ProgramNode(ELEMENTWISE[op].program, tuple(constant(float(v), dtypes.float32) for v in values), dtype=dtypes.float32)
  folded = fold_program(node)
  expected = np.asarray(ELEMENTWISE[op].numpy(*values), dtype=np.float32)
  np.testing.assert_array_max_ulp(np.float32(folded.attrs["value"]), expected, maxulp=1)


@pytest.mark.parametrize("mode", ["block", "scalar"])
def test_float32_intermediate_stores_keep_rounding(mode):
  @sc.function(sc.arg("x", 3), outputs=sc.arg("out"), name=f"float32_round_{mode}")
  def fn(x):
    a = sc.cast(x, "float32")
    return sc.cast((a + 1.0) - a, "float64").with_lowering(mode)

  values = np.array([16777216.0, 16777218.0, -16777216.0])
  expected = (values.astype(np.float32) + np.float32(1)) - values.astype(np.float32)
  np.testing.assert_array_equal(fn(values), expected)


@pytest.mark.parametrize("mode", ["block", "scalar"])
@pytest.mark.parametrize("dtype,bits", [("int64", 53), ("int32", 24)])
def test_integer_constant_tables_and_sums_stay_exact(mode, dtype, bits):
  values = np.array([2**bits + 1, -(2**bits), 2**bits + 3, -(2**bits)], dtype=dtype)

  @sc.function(sc.arg("x", 4), outputs=sc.group(sc.arg("difference"), sc.arg("total")), name=f"exact_int64_{mode}")
  def fn(x):
    ints = sc.const(values) + sc.cast(x, dtype)
    return sc.cast(ints - sc.const(values - 1), "float64").with_lowering(mode), sc.cast(ints.sum(), "float64").with_lowering(mode)

  difference, total = fn(np.zeros(4))
  np.testing.assert_array_equal(difference, np.ones(4))
  assert total == 4


@pytest.mark.parametrize("op", [ProgramOp.ABS, ProgramOp.MINIMUM, ProgramOp.MAXIMUM])
def test_integer_libm_operations_never_convert_to_float(tmp_path, op):
  values = np.array([-(2**53 + 1), -(2**53 + 3), 2**53 + 1, 2**53 + 3], dtype=np.int64)
  x, y = (p.buffer(n, dtypes.int64, (4,)) for n in ("x", "y"))
  i = p.var("i")
  a = p.load(p.view(x, [i]))
  args = (a,) if op == ProgramOp.ABS else (a, p.const_int(2**53 + 2))
  value = ProgramNode(op, args, dtype=dtypes.int64)
  prog = p.program([p.proc("kernel", [x, y], [p.for_(p.range_("i", 0, 4), [p.store(p.view(y, [i]), value)])])])
  got = np.empty_like(values)
  _run_raw(prog, tmp_path, [values], [got])
  expected = np.abs(values) if op == ProgramOp.ABS else (np.minimum if op == ProgramOp.MINIMUM else np.maximum)(values, 2**53 + 2)
  np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("dtype,bits", [("int32", 24), ("int64", 53)])
@pytest.mark.parametrize("mode", ["block", "scalar", "folded"])
def test_integer_abs_is_available_through_the_expression_api(dtype, bits, mode):
  values = np.array([-4, -2, 0, 2, 4])
  base = (1 << bits) + 1

  @sc.function(sc.arg("x", 5), outputs=sc.arg("out"), name=f"integer_abs_{dtype}_{mode}")
  def fn(x):
    i = sc.const(values.astype(dtype)) if mode == "folded" else sc.cast(x, dtype)
    large = sc.where(i < 0, i - base, i + base)
    return sc.cast(large.abs() - base, "float64").with_lowering("scalar" if mode == "scalar" else "block")

  np.testing.assert_array_equal(fn(values.astype(float)), np.abs(values))


@pytest.mark.parametrize("dialect", ["gnu", "c"])
@pytest.mark.parametrize("lanes", [2, 4, 8])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "gather"])
def test_float32_lanes_and_tails_match_numpy(tmp_path, dialect, lanes, layout):
  x = p.buffer("x", dtypes.float32, (38,))
  y = p.buffer("y", dtypes.float32, (19,))
  i = p.var("i")
  index = i if layout == "contiguous" else p.mul(i, p.const_int(2)) if layout == "strided" else p.mod(p.mul(i, i), p.const_int(38))
  a = p.load(p.view(x, [index]))
  value = p.add(ProgramNode(ProgramOp.SIN, (a,), dtype=dtypes.float32), constant(0.25, dtypes.float32))
  prog = p.program([p.proc("kernel", [x, y], [p.for_(p.range_("i", 0, 19), [p.store(p.view(y, [i]), value)])])])
  widened = widen_ranges(prog, lanes=lanes)
  assert any(n.op == ProgramOp.RANGE and "lanes" in n.attrs for n in walk_program(widened))
  values = np.linspace(-2, 2, 38, dtype=np.float32)
  got = np.full(23, np.float32(8765))
  _run_raw(widened, tmp_path, [values], [got], dialect=dialect)
  indices = np.arange(19) if layout == "contiguous" else np.arange(19) * 2 if layout == "strided" else np.arange(19) ** 2 % 38
  np.testing.assert_array_max_ulp(got[:19], np.sin(values[indices]) + np.float32(0.25), maxulp=4)
  np.testing.assert_array_equal(got[19:], np.full(4, np.float32(8765)))


@pytest.mark.parametrize("dialect", ["gnu", "c"])
@pytest.mark.parametrize("form", ["store", "assign"])
def test_float32_lane_reduction_keeps_sequential_rounding(tmp_path, dialect, form):
  x = p.buffer("x", dtypes.float32, (19,))
  y = p.buffer("y", dtypes.float32, (1,))
  i, z = p.var("i"), p.const_int(0)
  target = p.view(y, [z])
  acc = p.var("acc", dtypes.float32)
  value = p.add(p.load(target) if form == "store" else acc, p.load(p.view(x, [i])))
  update = p.store(target, value) if form == "store" else p.assign("acc", value)
  loop = p.for_(p.range_("i", 0, 19, kind=p.RangeKind.REDUCE), [update])
  body = (
    [p.store(target, constant(0, dtypes.float32)), loop]
    if form == "store"
    else [p.assign("acc", constant(0, dtypes.float32), declare=True), loop, p.store(target, acc)]
  )
  prog = p.program([p.proc("kernel", [x, y], body)])
  widened = widen_ranges(prog, lanes=4)
  assert any(n.attrs.get("ordered_reduction") for n in walk_program(widened))
  values = np.resize(np.array([16777216, 1, -16777216], dtype=np.float32), 19)
  got = np.empty(1, dtype=np.float32)
  _run_raw(widened, tmp_path, [values], [got], dialect=dialect)
  assert got[0] == np.add.accumulate(values)[-1]


@pytest.mark.parametrize("dtype", ["float32", "int64"])
@pytest.mark.parametrize("mode", ["block", "scalar"])
def test_sparse_plans_use_zeros_of_the_value_dtype(dtype, mode):
  diagonal = sc.SparsityPattern((2, 2), (0, 1), (0, 1))
  off_diagonal = sc.SparsityPattern((2, 2), (0, 1), (1, 0))

  @sc.function(sc.arg("x", 2), outputs=sc.group(sc.arg("union"), sc.arg("scattered")), name=f"typed_sparse_{dtype}_{mode}")
  def fn(x):
    a = sc.pack(sc.cast(x, dtype), diagonal)
    b = sc.pack(sc.const(np.array([3, 4], dtype=dtype)), off_diagonal)
    scattered = sc.scatter(a, [0, 1, 1, 3], (2, 2))
    return tuple(sc.cast(value.to_dense(), "float64").with_lowering(mode) for value in (a + b, scattered))

  lower_function(fn)
  union, scattered = fn(np.array([2.0, 5.0]))
  np.testing.assert_array_equal(union, [[2, 3], [4, 5]])
  np.testing.assert_array_equal(scattered, [[2, 0], [0, 5]])


@pytest.mark.parametrize("mode", ["block", "scalar"])
def test_sparse_float32_products_keep_ordered_typed_accumulators(mode):
  pattern = sc.SparsityPattern((2, 3), (0, 0, 0, 1), (0, 1, 2, 1))

  @sc.function(sc.arg("x", 4), outputs=sc.arg("out"), name=f"sparse_float32_product_{mode}")
  def fn(x):
    matrix = sc.pack(sc.cast(x, "float32"), pattern)
    product = matrix @ sc.const(np.ones(3, dtype=np.float32))
    return sc.cast(product.to_dense(), "float64").with_lowering(mode)

  values = np.array([16777216, 1, -16777216, 2.5], dtype=np.float32)
  np.testing.assert_array_equal(fn(values.astype(float)), [np.add.accumulate(values[:3])[-1], values[3]])


def test_float32_stores_are_not_coalesced_or_spilled():
  x = p.buffer("x", dtypes.float32, (2048,), address_space="private")
  y = p.buffer("y", dtypes.float32, (2048,))
  stores = [p.store(p.view(x, [p.const_int(i)]), constant(i, dtypes.float32)) for i in (0, 1)]
  prog = p.program([p.proc("kernel", [y], [x, *stores, p.store(p.view(y, [p.const_int(0)]), p.load(p.view(x, [p.const_int(0)])))])])
  assert coalesce_stores(prog) is prog
  packed = pack_workspace(prog)
  assert packed.args[0].attrs["sz_w"] == 0
  assert not any("workspace_offset" in n.attrs for n in walk_program(packed))


@pytest.mark.parametrize("operation", ["view", "load", "store"])
def test_verifier_rejects_buffer_dtype_mismatches(operation):
  out = p.buffer("out", dtypes.float32, (1,))
  target = p.view(out, [p.const_int(0)])
  if operation == "view":
    target = ProgramNode(ProgramOp.VIEW, target.args, target.attrs, dtype=dtypes.float64)
    bad = p.store(target, constant(0, dtypes.float64))
  elif operation == "load":
    bad = p.assign("bad", ProgramNode(ProgramOp.LOAD, (target,), dtype=dtypes.float64), declare=True)
  else:
    bad = p.store(target, constant(0, dtypes.float64))
  with pytest.raises(VerifyError, match="dtype"):
    verify_program(p.program([p.proc("kernel", [out], [bad])]))


def test_float32_constant_reads_round_to_the_buffer_dtype():
  table = p.const_buffer("table", dtypes.float32, (1,), [16777217.0])
  out = p.buffer("out", dtypes.float32, (1,))
  prog = p.program([p.proc("kernel", [out], [table, p.store(p.view(out, [p.const_int(0)]), p.load(p.view(table, [p.const_int(0)])))])])
  folded = fold_arith(prog)
  value = next(n for n in walk_program(folded) if n.op == ProgramOp.STORE).args[1]
  assert value.attrs["value"] == 16777216.0


@pytest.mark.parametrize("value", [1.0 + 2**-24, (1 << 62) + (1 << 38) + 1])
@pytest.mark.parametrize("folded", [False, True])
def test_float32_constant_table_formatting_matches_folded_reads(tmp_path, value, folded):
  table = p.const_buffer("table", dtypes.float32, (1,), [value])
  out = p.buffer("out", dtypes.float32, (1,))
  prog = p.program([p.proc("kernel", [out], [table, p.store(p.view(out, [p.const_int(0)]), p.load(p.view(table, [p.const_int(0)])))])])
  got = np.empty(1, dtype=np.float32)
  _run_raw(fold_arith(prog) if folded else prog, tmp_path, [], [got])
  assert got[0] == np.asarray(value).astype(np.float32).item()


def test_program_nodes_require_an_explicit_dtype():
  with pytest.raises(TypeError):
    ProgramNode(ProgramOp.CONST_FLOAT, attrs={"value": 1.0})  # ty: ignore[missing-argument]


def test_float32_literals_use_nine_digits_and_an_f_suffix():
  assert _emit_scalar(constant(float(np.float32(1.23456789)), dtypes.float32), {}) == "1.23456788f"
  assert _emit_scalar(constant(1, dtypes.float32), {}) == "1.0f"
  assert _emit_scalar(constant(-0.0, dtypes.float32), {}) == "-0.0f"


@pytest.mark.parametrize("mode", ["block", "scalar", "folded"])
def test_float32_structural_operations_match_numpy(mode):
  values = np.arange(12, dtype=np.float32).reshape(3, 4) / np.float32(8)

  @sc.function(
    sc.arg("x", (3, 4)), outputs=sc.group(sc.arg("product"), sc.arg("moved"), sc.arg("scattered"), sc.arg("total")), name=f"float32_structural_{mode}"
  )
  def fn(x):
    a = sc.const(values) if mode == "folded" else sc.cast(x, "float32")
    moved = sc.concat([a.T.reshape(12)[::2], a.gather([0, 3, 7, 11])])
    scattered = sc.scatter(a.vec(), np.repeat(np.arange(6), 2), (6,))
    return tuple(sc.cast(v, "float64").with_lowering("scalar" if mode == "scalar" else "block") for v in (a @ a.T, moved, scattered, a.sum()))

  expected = [
    values @ values.T,
    np.concatenate([values.T.reshape(12)[::2], values.reshape(-1)[[0, 3, 7, 11]]]),
    values.reshape(6, 2).sum(axis=1),
    values.sum(),
  ]
  for got, ref in zip(fn(values.astype(float)), expected, strict=True):
    np.testing.assert_array_max_ulp(np.asarray(got, dtype=np.float32), np.asarray(ref, dtype=np.float32), maxulp=4)


@pytest.mark.parametrize("mode", ["block", "scalar", "folded"])
def test_scatter_and_sum_constants_preserve_each_dtype(mode):
  ints = np.array([2**53 + 1, 0], dtype=np.int64)
  values = np.array([16777216, 1, -16777216], dtype=np.float32)

  @sc.function(sc.arg("x", 3), outputs=sc.group(sc.arg("scatter"), sc.arg("exact")), name=f"typed_scatter_{mode}")
  def fn(x):
    a = sc.const(values) if mode == "folded" else sc.cast(x, "float32")
    scattered = sc.scatter(a, [0, 0, 0], (1,))
    exact = sc.equal(sc.const(ints).sum(), np.int64(2**53 + 1))
    return sc.cast(scattered, "float64").with_lowering("scalar" if mode == "scalar" else "block"), sc.cast(exact, "float64")

  scattered, exact = fn(values.astype(float))
  expected = np.zeros(1, dtype=np.float32)
  np.add.at(expected, [0, 0, 0], values)
  np.testing.assert_array_equal(scattered, expected)
  assert exact == 1


def test_extreme_int64_table_literals_compile_exactly(tmp_path):
  values = np.array([-(1 << 63), (1 << 63) - 1, (1 << 53) + 1], dtype=np.int64)
  table = p.const_buffer("table", dtypes.int64, (3,), values.tolist())
  out = p.buffer("out", dtypes.int64, (3,))
  i = p.var("i")
  prog = p.program([p.proc("kernel", [out], [table, p.for_(p.range_("i", 0, 3), [p.store(p.view(out, [i]), p.load(p.view(table, [i])))])])])
  got = np.empty_like(values)
  _run_raw(prog, tmp_path, [], [got])
  np.testing.assert_array_equal(got, values)


@pytest.mark.parametrize("dialect", ["gnu", "c"])
def test_float32_bit_select_lanes_keep_high_mask_bits(tmp_path, dialect):
  x, y = (p.buffer(n, dtypes.float32, (67,)) for n in ("x", "y"))
  i = p.var("i")
  value = p.bit_select(p.const_int(-(1 << 63)), p.mod(i, p.const_int(64)), p.load(p.view(x, [i])))
  prog = p.program([p.proc("kernel", [x, y], [p.for_(p.range_("i", 0, 67), [p.store(p.view(y, [i]), value)])])])
  widened = widen_ranges(prog, lanes=4)
  values = np.full(67, np.float32(-3))
  expected = np.zeros(67, dtype=np.float32)
  expected[63] = -3
  got = np.empty_like(values)
  _run_raw(widened, tmp_path, [values], [got], dialect=dialect)
  np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("mode", ["block", "scalar"])
def test_float32_calls_and_maps_keep_rounding(mode):
  @sc.function(sc.arg("x", 3), outputs=sc.arg("y"), name=f"float32_body_{mode}")
  def body(x):
    a = sc.cast(x, "float32")
    return sc.cast((a + 1) - a, "float64").with_lowering(mode)

  @sc.function(sc.arg("x", (5, 3)), outputs=sc.arg("y"), name=f"float32_map_{mode}")
  def fn(x):
    return sc.vmap(body, 5)(x).with_lowering(mode)

  values = np.tile([16777216.0, 16777218.0, -16777216.0], (5, 1))
  a = values.astype(np.float32)
  np.testing.assert_array_equal(fn(values), (a + np.float32(1)) - a)


F32_SPECIAL = np.array([-np.inf, -3.4e38, -1.5, -1e-45, -0.0, 0.0, 1e-45, 0.1, 1.5, 3.4e38, np.inf, np.nan], dtype=np.float32)
F32_X, F32_Y = (a.reshape(-1) for a in np.meshgrid(F32_SPECIAL, F32_SPECIAL, indexing="ij"))


def _assert_float32_special(got, expected):
  expected = np.asarray(expected, dtype=np.float32)
  np.testing.assert_array_equal(got, got.astype(np.float32).astype(np.float64))
  np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
  finite = np.isfinite(expected)
  np.testing.assert_array_equal(got[~finite], expected[~finite])
  np.testing.assert_array_max_ulp(got[finite].astype(np.float32), expected[finite], maxulp=4)


@pytest.mark.parametrize("mode", ["block", "scalar"])
def test_float32_special_value_grid_matches_numpy(mode):
  """Adapt devrush's test_float32_ops_round_like_numpy_float32 to internal float32 values."""
  ops = [sc.ExprOp.MINIMUM, sc.ExprOp.MAXIMUM, sc.ExprOp.SELECT, sc.ExprOp.LE, sc.ExprOp.EQ, sc.ExprOp.ISFINITE, sc.ExprOp.ATAN2]

  @sc.function(sc.arg("x", F32_X.size), sc.arg("y", F32_X.size), outputs=sc.arg("out"), name=f"float32_special_{mode}")
  def fn(x, y):
    a, b = sc.cast(x, "float32"), sc.cast(y, "float32")
    return sc.stack([sc.cast(_elementwise(op, a, b), "float64") for op in ops]).with_lowering(mode)

  for op, got in zip(ops, np.asarray(fn(F32_X.astype(float), F32_Y.astype(float))), strict=True):
    with np.errstate(all="ignore"):
      _assert_float32_special(got, _reference(op, F32_X, F32_Y))


@pytest.mark.parametrize("mode", ["block", "scalar"])
def test_float32_special_libm_results_stay_float32(mode):
  """Adapt devrush's float32 libm test, including the arithmetic around each libm result."""
  ops = [op for op, row in ELEMENTWISE.items() if OP_INFO[op].arity == 1 and row.c.isidentifier()]

  @sc.function(sc.arg("x", F32_X.size), sc.arg("y", F32_X.size), outputs=sc.arg("out"), name=f"float32_special_libm_{mode}")
  def fn(x, y):
    a, b = sc.cast(x, "float32"), sc.cast(y, "float32")
    outputs = [_elementwise(op, a, b) for op in ops] + [a**b, a.floor() * 0.1 + a.abs().sqrt()]
    return sc.stack([sc.cast(value, "float64") for value in outputs]).with_lowering(mode)

  with np.errstate(all="ignore"):
    expected = [_reference(op, F32_X, F32_Y) for op in ops] + [F32_X**F32_Y, np.floor(F32_X) * np.float32(0.1) + np.sqrt(np.abs(F32_X))]
  for got, ref in zip(np.asarray(fn(F32_X.astype(float), F32_Y.astype(float))), expected, strict=True):
    _assert_float32_special(got, ref)


@pytest.mark.parametrize("dtype,kind", [("float32", k) for k in ("add", "mean", "max", "min")] + [("int32", "add"), ("int64", "add")])
@pytest.mark.parametrize("mode", ["block", "scalar", "folded"])
@pytest.mark.parametrize("form", ["dense", "supported", "sparse"])
def test_axis_reductions_keep_the_value_dtype(dtype, kind, mode, form):
  axes = (0, 1, (0, 1), ())
  mask = np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0], [0, 1, 1]], dtype=dtype)
  slots = np.flatnonzero(mask)
  rows, cols = np.nonzero(mask)
  pattern = CanonicalPattern(mask.shape, tuple(map(int, rows)), tuple(map(int, cols)))
  if dtype == "float32":
    data = np.random.default_rng(55).uniform(0.2, 1.0, mask.shape).astype(dtype)
  else:
    big = 2 ** (53 if dtype == "int64" else 24) + 1
    data = np.tile(np.array([big, -big + 1, 1], dtype=dtype), (4, 1))
  source = data if form == "dense" else data * mask
  if kind in {"add", "mean"}:
    expected = [(np.sum if kind == "add" else np.mean)(source, axis=axis, keepdims=True, dtype=dtype) for axis in axes]
  else:
    expected = [(np.max if kind == "max" else np.min)(source, axis=axis, keepdims=True) for axis in axes]

  @sc.function(sc.arg("x", mask.shape), outputs=sc.group(*(sc.arg(f"out{i}") for i in range(len(axes)))))
  def fn(x):
    value = sc.const(data) if mode == "folded" else sc.const(data) + sc.cast(x, dtype)
    if form == "supported":
      value = value * sc.const(mask)
    elif form == "sparse":
      value = sc.pack(sc.gather(value, slots), pattern)
    result = []
    for axis, ref in zip(axes, expected, strict=True):
      if kind in {"add", "mean"}:
        reduced = (value.sum if kind == "add" else value.mean)(axis, keepdims=True)
      else:
        reduced = _reduce(value, kind, axis)
        reduced = reduced.reshape(tuple(1 if i in reduced.attrs["axes"] else n for i, n in enumerate(value.shape)))
      reduced = reduced.to_dense()
      if dtype != "float32":
        reduced = sc.equal(reduced, sc.const(ref))
      result.append(sc.cast(reduced, "float64").with_lowering("scalar" if mode == "scalar" else "block"))
    return tuple(result)

  for actual, ref in zip(fn(np.zeros(mask.shape)), expected, strict=True):
    if dtype == "float32":
      np.testing.assert_array_max_ulp(np.asarray(actual, dtype=dtype), ref, maxulp=4)
    else:
      np.testing.assert_array_equal(actual, np.ones_like(ref))


@pytest.mark.parametrize("dtype", ["float32", "int32", "int64"])
@pytest.mark.parametrize("mode", ["block", "scalar", "folded"])
def test_empty_axis_sums_keep_typed_zero(dtype, mode):
  @sc.function(sc.arg("x", (3, 0)), outputs=sc.group(sc.arg("rows"), sc.arg("total")))
  def fn(x):
    value = sc.const(np.empty((3, 0), dtype=dtype)) if mode == "folded" else sc.cast(x, dtype)
    return tuple(sc.cast(value.sum(axis), "float64").with_lowering("scalar" if mode == "scalar" else "block") for axis in (1, None))

  rows, total = fn(np.empty((3, 0)))
  np.testing.assert_array_equal(rows, np.zeros(3))
  assert total == 0
