"""Generated lane helpers compile in GNU and scalar C and preserve numerical results."""

import ctypes
import platform
import re
import shutil
import subprocess

import numpy as np
import pytest

from scaly.codegen.c import _includes, _render_raw_callee
from scaly.ir import program as p
from scaly.ir.program import ProgramNode, ProgramOp
from scaly.ir.types import dtypes
from scaly.utils.names import NameScope
from scaly.passes.program.fold_tiles import fold_tiles
from scaly.passes.program.pack_workspace import pack_workspace
from scaly.passes.program.coalesce_stores import coalesce_stores
from scaly.passes.program.widen_ranges import widen_ranges

pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
_NATIVE = "-mcpu=native" if platform.machine().lower() in ("arm64", "aarch64") else "-march=native"
_GLIBC_X86 = platform.libc_ver()[0] == "glibc" and platform.machine().lower() in ("x86_64", "amd64")


def _compile(prog, tmp_path, *, dialect="gnu", compiler="cc", vector_libm="none", defines=()):
  source = "\n".join([*_includes(dialect=dialect, prog=prog), *_render_raw_callee(prog.args[0], dialect=dialect, vector_libm=vector_libm)])
  workspace = int(prog.args[0].attrs.get("sz_w", 0))
  source += f"\nvoid run(double *x, double *y) {{ double w[{max(workspace, 1)}]; kernel_raw(x, y, w); }}\n"
  return _compile_source(source, tmp_path, dialect=dialect, compiler=compiler, defines=defines)


def _compile_source(source, tmp_path, *, dialect="gnu", compiler="cc", defines=()):
  path = tmp_path / "kernel.c"
  path.write_text(source)
  library = tmp_path / "kernel.so"
  subprocess.run(
    [
      compiler,
      "-shared",
      "-fPIC",
      "-O3",
      _NATIVE,
      "-fno-math-errno",
      "-std=c99",
      *(["-pedantic-errors"] if dialect == "c" else []),
      *defines,
      str(path),
      "-lm",
      "-o",
      str(library),
    ],
    check=True,
    capture_output=True,
  )
  handle = ctypes.CDLL(str(library))
  function = handle.run
  function.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
  return function, source


def _program(count=11):
  x = p.buffer("x", dtypes.float64, (count * 2,))
  y = p.buffer("y", dtypes.float64, (count * 3,))
  i = p.var("stage")
  value = p.load(p.view(x, [p.mul(i, p.const_int(2))]))
  rng = p.range_("stage", 0, count)
  rng = ProgramNode(rng.op, rng.args, {**rng.attrs, "mapped": True}, rng.dtype)
  sin = ProgramNode(ProgramOp.SIN, (value,))
  rhs = p.add(p.mul(sin, value), p.const_float(0.125))
  return p.program([p.proc("kernel", [x, y], [p.for_(rng, [p.store(p.view(y, [p.mul(i, p.const_int(3))]), rhs)])])])


def _evaluate(function, x, count):
  y = np.full(count * 3 + 4, 8765.0, dtype=np.float64)
  function(x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
  return y


@pytest.mark.parametrize("compiler", ["gcc", "clang"])
@pytest.mark.parametrize("dialect", ["gnu", "c"])
@pytest.mark.parametrize("lanes", [1, 2, 4, 8])
@pytest.mark.parametrize("count", [2, 3, 5, 11, 17])
def test_widths_match_scalar_bytes(tmp_path, compiler, dialect, lanes, count):
  if shutil.which(compiler) is None:
    pytest.skip(f"{compiler} unavailable")
  baseline = tmp_path / "baseline"
  baseline.mkdir()
  scalar, _ = _compile(_program(count), baseline, compiler=compiler)
  vector, source = _compile(widen_ranges(_program(count), lanes=lanes), tmp_path, dialect=dialect, compiler=compiler)
  x = np.linspace(-2, 3, count * 2)
  assert _evaluate(vector, x, count).tobytes() == _evaluate(scalar, x, count).tobytes()
  assert source.count("void kernel_lanes_1(") == (count >= 2 * lanes)
  if dialect == "c":
    assert "__attribute__" not in source


@pytest.mark.parametrize("dialect", ["gnu", "c"])
@pytest.mark.parametrize("lanes", [2, 4, 8])
def test_fixed_lanes_keep_short_loops_scalar_and_share_one_vector_type(tmp_path, dialect, lanes):
  counts = (3, 2 * lanes - 1, 2 * lanes, 2 * lanes + 5)
  x, y = p.buffer("x", dtypes.float64, (sum(counts),)), p.buffer("y", dtypes.float64, (sum(counts),))
  loops = []
  for part, count in enumerate(counts):
    i, start = p.var(f"i{part}"), p.const_int(sum(counts[:part]))
    value = p.load(p.view(x, [p.add(start, i)]))
    rng = p.range_(f"i{part}", 0, count)
    rng = ProgramNode(rng.op, rng.args, {**rng.attrs, "mapped": True}, rng.dtype)
    rhs = p.add(p.mul(ProgramNode(ProgramOp.SIN, (value,)), value), p.const_float(0.125))
    loops.append(p.for_(rng, [p.store(p.view(y, [p.add(start, i)]), rhs)]))
  prog = p.program([p.proc("kernel", [x, y], loops)])
  baseline = tmp_path / "baseline"
  baseline.mkdir()
  scalar, _ = _compile(prog, baseline)
  vector, source = _compile(widen_ranges(prog, lanes=lanes), tmp_path, dialect=dialect)
  assert len(re.findall(r"void kernel_lanes_\d+\(", source)) == sum(count >= 2 * lanes for count in counts)
  assert source.count("#define SCALY_WIDTH_") == 1
  assert len(re.findall(r"^typedef double \w+ __attribute__\(\(vector_size\(8 \*", source, re.M)) == (dialect == "gnu")
  values = np.linspace(-2, 3, sum(counts))
  assert _evaluate(vector, values, sum(counts)).tobytes() == _evaluate(scalar, values, sum(counts)).tobytes()


@pytest.mark.parametrize("lanes", [1, 2, 4, 8])
def test_auto_width_override_and_tail(tmp_path, lanes):
  function, _ = _compile(widen_ranges(_program(), lanes="auto"), tmp_path, defines=[f"-DSCALY_LANES={lanes}"])
  x = np.linspace(-2, 3, 22)
  result = _evaluate(function, x, 11)
  np.testing.assert_allclose(result[:33:3], np.sin(x[::2]) * x[::2] + 0.125, atol=1e-15)
  assert np.all(result[33:] == 8765)


@pytest.mark.parametrize("values", [[2.0, 3.0] * 4, [0.0, -0.0] * 4])
def test_compiled_tiles_preserve_dynamic_values_and_zero_sign(tmp_path, values):
  x = p.buffer("x", dtypes.float64, (8,))
  y = p.buffer("y", dtypes.float64, (8,))
  table = p.const_buffer("tile", dtypes.float64, (8,), values)
  i = p.var("i")
  prog = p.program([p.proc("kernel", [x, y], [table, p.for_(p.range_("i", 0, 8), [p.store(p.view(y, [i]), p.load(p.view(table, [i])))])])])
  function, _ = _compile(fold_tiles(prog), tmp_path)
  output = _evaluate(function, np.zeros(8), 8)[:8]
  assert output.tobytes() == np.asarray(values, dtype=np.float64).tobytes()


def _nested_program(count=7, size=3):
  x = p.buffer("x", dtypes.float64, (count * size,))
  y = p.buffer("y", dtypes.float64, (count,))
  a, b = p.buffer("a", dtypes.float64, (size,)), p.buffer("b", dtypes.float64, (1,))
  scratch = p.buffer("scratch", dtypes.float64, (size,), address_space="private")
  j = p.var("j")
  acc = p.var("acc", dtypes.float64)
  stores = p.for_(p.range_("j", 0, size), [p.store(p.view(scratch, [j]), p.mul(p.load(p.view(a, [j])), p.const_float(2)))])
  reduction = p.for_(p.range_("j", 0, size), [p.assign("acc", p.add(acc, p.load(p.view(scratch, [j]))))])
  callee = p.proc(
    "inner", [a, b], [scratch, stores, p.assign("acc", p.const_float(0), declare=True), reduction, p.store(p.view(b, [p.const_int(0)]), acc)]
  )
  rng = p.range_("stage", 0, count)
  rng = ProgramNode(rng.op, rng.args, {**rng.attrs, "mapped": True}, rng.dtype)
  stage = p.var("stage")
  call = ProgramNode(ProgramOp.CALL, (p.view(x, [p.mul(stage, p.const_int(size))]), p.view(y, [stage])), {"callee": "inner", "n_in": 1, "n_out": 1})
  return p.program([callee, p.proc("kernel", [x, y], [p.for_(rng, [call])])])


@pytest.mark.parametrize("dialect", ["gnu", "c"])
@pytest.mark.parametrize("lanes", [2, 4, 8])
def test_mapped_nested_reduction_keeps_each_lane_order(tmp_path, dialect, lanes):
  widened = widen_ranges(_nested_program(count=17), lanes=lanes)
  function, source = _compile(p.program([widened.args[-1]]), tmp_path, dialect=dialect)
  x = np.linspace(-1, 2, 51)
  actual = _evaluate(function, x, 17)[:17]
  expected = np.asarray([sum(row * 2) for row in x.reshape(17, 3)])
  assert actual.tobytes() == expected.tobytes()
  assert "scratch[24]" in source


@pytest.mark.parametrize("compiler", ["gcc", "clang"])
@pytest.mark.parametrize("lanes", [1, 2, 4, 8])
def test_glibc_vector_symbols_are_opt_in(tmp_path, lanes, compiler):
  if not _GLIBC_X86 or shutil.which(compiler) is None:
    pytest.skip("requires x86-64 glibc and the selected compiler")
  macros = subprocess.run([compiler, _NATIVE, "-dM", "-E", "-x", "c", "-"], input="", capture_output=True, text=True, check=True).stdout
  if lanes > 1 and {2: "__SSE2__", 4: "__AVX__", 8: "__AVX512F__"}[lanes] not in macros:
    pytest.skip("host does not support the selected libmvec width")
  off = tmp_path / "off"
  off.mkdir()
  _, _ = _compile(widen_ranges(_program(17), lanes=lanes), off, compiler=compiler)
  function, _ = _compile(widen_ranges(_program(17), lanes=lanes), tmp_path, vector_libm="glibc", compiler=compiler)
  off_symbols = subprocess.run(["nm", "-D", str(off / "kernel.so")], check=True, capture_output=True, text=True).stdout
  on_symbols = subprocess.run(["nm", "-D", str(tmp_path / "kernel.so")], check=True, capture_output=True, text=True).stdout
  assert "_ZGV" not in off_symbols
  assert ("_ZGV" in on_symbols) == (lanes > 1)
  x = np.linspace(-2, 3, 34)
  np.testing.assert_allclose(_evaluate(function, x, 17)[:51:3], np.sin(x[::2]) * x[::2] + 0.125, atol=1e-14)


@pytest.mark.parametrize("compiler", ["gcc", "clang"])
@pytest.mark.parametrize("lanes", [2, 4, 8])
def test_reduction_stages_loads_but_keeps_original_sum_order(tmp_path, compiler, lanes):
  if shutil.which(compiler) is None:
    pytest.skip(f"{compiler} unavailable")
  x, y = p.buffer("x", dtypes.float64, (19,)), p.buffer("y", dtypes.float64, (1,))
  i = p.var("i")
  target = p.view(y, [p.const_int(0)])
  value = p.load(p.view(x, [i]))
  update = p.store(target, p.add(p.load(target), p.mul(value, value)))
  loop = p.for_(p.range_("i", 0, 19, kind=p.RangeKind.REDUCE), [update])
  prog = p.program([p.proc("kernel", [x, y], [p.store(target, p.const_float(0)), loop])])
  baseline = tmp_path / "baseline"
  baseline.mkdir()
  scalar, _ = _compile(prog, baseline, compiler=compiler)
  vector, source = _compile(widen_ranges(prog, lanes=lanes), tmp_path, compiler=compiler)
  assert "kernel_lanes_1" in source
  values = np.random.default_rng(17).normal(size=19)
  assert _evaluate(vector, values, 1).tobytes() == _evaluate(scalar, values, 1).tobytes()


@pytest.mark.parametrize("lanes", [1, 2, 4, 8])
def test_by_value_vector_types_are_not_under_aligned_aliases(lanes):
  # aarch64 GCC 13.1 and 14 to 16.1 crash on a by-value parameter whose type has both attributes.
  prog = widen_ranges(_program(17), lanes=lanes)
  source = "\n".join([*_includes(prog=prog), *_render_raw_callee(prog.args[0], dialect="gnu", vector_libm="glibc")])
  aliases = set(re.findall(r"typedef \w+ (\w+) __attribute__\(\(.*aligned.*may_alias", source))
  parameters = set(re.findall(r"[(,]\s*(?:const\s+)?(\w+)(?:\s+\w+)?\s*(?=[,)])", source))
  assert aliases - {"double2"}
  assert set(re.findall(r"typedef double (\w+) __attribute__\(\(vector_size\(8 \*", source)) & parameters
  assert not aliases & parameters


def test_helper_names_do_not_capture_user_names(tmp_path):
  x = p.buffer("scaly_valid", dtypes.float64, (9,))
  y = p.buffer("scaly_load_1", dtypes.float64, (9,))
  i = p.var("i")
  rng = p.range_("i", 0, 9)
  rng = ProgramNode(rng.op, rng.args, {**rng.attrs, "mapped": True}, rng.dtype)
  scalar = p.var("kernel_lanes_1_valid", dtypes.float64)
  body = [
    p.assign("kernel_lanes_1_valid", p.const_float(2), declare=True),
    p.for_(rng, [p.store(p.view(y, [i]), p.mul(p.load(p.view(x, [i])), scalar))]),
  ]
  prog = p.program([p.proc("kernel", [x, y], body)])
  function, _ = _compile(widen_ranges(prog, lanes=4), tmp_path)
  values = np.arange(9, dtype=float)
  np.testing.assert_array_equal(_evaluate(function, values, 9)[:9], values * 2)


@pytest.mark.parametrize("dialect", ["gnu", "c"])
def test_minimum_maximum_and_negative_zero_broadcast(tmp_path, dialect):
  x, y = p.buffer("x", dtypes.float64, (9,)), p.buffer("y", dtypes.float64, (9,))
  i = p.var("i")
  rng = p.range_("i", 0, 9)
  rng = ProgramNode(rng.op, rng.args, {**rng.attrs, "mapped": True}, rng.dtype)
  zero = p.var("negative_zero", dtypes.float64)
  value = p.load(p.view(x, [i]))
  minimum = ProgramNode(ProgramOp.MINIMUM, (value, zero))
  maximum = ProgramNode(ProgramOp.MAXIMUM, (minimum, value))
  body = [p.assign("negative_zero", p.load(p.view(x, [p.const_int(0)])), declare=True), p.for_(rng, [p.store(p.view(y, [i]), maximum)])]
  prog = p.program([p.proc("kernel", [x, y], body)])
  baseline = tmp_path / "baseline"
  baseline.mkdir()
  scalar, _ = _compile(prog, baseline)
  vector, _ = _compile(widen_ranges(prog, lanes=4), tmp_path, dialect=dialect)
  values = np.asarray([-0.0, 0.0, 1.0, -1.0, np.inf, -np.inf, np.nan, 2.0, -2.0])
  assert _evaluate(vector, values, 9).tobytes() == _evaluate(scalar, values, 9).tobytes()


def test_glibc_wrong_instruction_set_fails_at_compile_time(tmp_path):
  if not _GLIBC_X86:
    pytest.skip("requires x86-64 glibc")
  with pytest.raises(subprocess.CalledProcessError) as error:
    _compile(widen_ranges(_program(17), lanes=8), tmp_path, vector_libm="glibc", defines=["-mno-avx512f"])
  assert b"requires __AVX512F__ and -lmvec" in error.value.stderr


@pytest.mark.parametrize("dialect", ["gnu", "c"])
def test_contiguous_output_axis_retains_inner_reduction(tmp_path, dialect):
  x, y = p.buffer("x", dtypes.float64, (27,)), p.buffer("y", dtypes.float64, (9,))
  i, j = p.var("i"), p.var("j")
  target = p.view(y, [i])
  value = p.load(p.view(x, [p.add(p.mul(j, p.const_int(9)), i)]))
  inner = p.for_(p.range_("j", 0, 3, kind=p.RangeKind.REDUCE), [p.store(target, p.add(p.load(target), value))])
  outer = p.for_(p.range_("i", 0, 9), [p.store(target, p.const_float(0)), inner])
  prog = p.program([p.proc("kernel", [x, y], [outer])])
  widened = widen_ranges(prog, lanes=4)
  function, source = _compile(widened, tmp_path, dialect=dialect)
  values = np.arange(27, dtype=float)
  np.testing.assert_array_equal(_evaluate(function, values, 9)[:9], values.reshape(3, 9).sum(axis=0))
  if dialect == "gnu":
    assert re.search(r"\*\(const \w+_vec_mem\*\)", source)


@pytest.mark.parametrize("dialect", ["gnu", "c"])
@pytest.mark.parametrize("lanes", [2, 4, 8])
def test_lane_scratch_is_packed_for_eight_lanes(tmp_path, dialect, lanes):
  prog = pack_workspace(widen_ranges(_nested_program(count=17, size=1100), lanes=lanes))
  main = prog.args[-1]
  assert main.attrs["sz_w"] == 1100 * 8
  function, source = _compile(p.program([main]), tmp_path, dialect=dialect)
  values = np.arange(17 * 1100, dtype=np.float64)
  np.testing.assert_array_equal(_evaluate(function, values, 17)[:17], values.reshape(17, 1100).sum(axis=1) * 2)
  assert "double* s0 = w + 0" in source


@pytest.mark.parametrize("dialect", ["gnu", "c"])
def test_paired_local_stores_keep_variable_major_layout(tmp_path, dialect):
  original = _nested_program(count=9)
  callee, main = original.args
  a, b, scratch, _stores, *rest = callee.args
  pair = p.store_pair(p.view(scratch, [p.const_int(0)]), *(p.mul(p.load(p.view(a, [p.const_int(i)])), p.const_float(2)) for i in (0, 1)))
  last = p.store(p.view(scratch, [p.const_int(2)]), p.mul(p.load(p.view(a, [p.const_int(2)])), p.const_float(2)))
  prog = p.program([p.proc("inner", [a, b], [scratch, pair, last, *rest]), main])
  widened = coalesce_stores(pack_workspace(widen_ranges(prog, lanes=4)))
  function, _ = _compile(p.program([widened.args[-1]]), tmp_path, dialect=dialect)
  values = np.arange(27, dtype=np.float64)
  np.testing.assert_array_equal(_evaluate(function, values, 9)[:9], values.reshape(9, 3).sum(axis=1) * 2)


@pytest.mark.parametrize("dialect", ["gnu", "c"])
@pytest.mark.parametrize("lanes", [2, 8])
def test_public_render_pipeline_keeps_ordered_reduction(tmp_path, dialect, lanes):
  import scaly as sc
  from scaly.codegen.aot import render_c_module

  @sc.function(sc.arg("x", 9), outputs=sc.arg("y", ()), name="kernel")
  def fun(x: sc.Expr) -> sc.Expr:
    return ((x + 1) ** 2).sum()

  module = render_c_module(fun, lanes=lanes, dialect=dialect, vector_libm="none")
  scalar_dir = tmp_path / "scalar"
  scalar_dir.mkdir()
  scalar_module = render_c_module(fun, lanes=1, vector_libm="none")
  scalar, _ = _compile(scalar_module.program, scalar_dir)
  vector, _ = _compile(module.program, tmp_path, dialect=dialect)
  values = np.asarray([1.0, -0.5, -0.25, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
  assert _evaluate(vector, values, 1).tobytes() == _evaluate(scalar, values, 1).tobytes()


@pytest.mark.parametrize("dialect", ["gnu", "c"])
def test_store_schedule_preserves_compiled_values(tmp_path, dialect):
  x, y = p.buffer("x", dtypes.float64, (17,)), p.buffer("y", dtypes.float64, (340,))
  i = p.var("i")
  base = p.var("base", dtypes.float64)
  body = [p.assign("base", p.load(p.view(x, [i])), declare=True)]
  body += [p.assign(f"value{j}", p.add(p.mul(base, p.const_float(j)), p.const_float(0.25)), declare=True) for j in range(20)]
  body += [p.store(p.view(y, [p.add(p.mul(i, p.const_int(20)), p.const_int(j))]), p.var(f"value{j}", dtypes.float64)) for j in range(20)]
  rng = p.range_("i", 0, 17)
  rng = ProgramNode(rng.op, rng.args, {**rng.attrs, "mapped": True}, rng.dtype)
  prog = p.program([p.proc("kernel", [x, y], [p.for_(rng, body)])])
  baseline = tmp_path / "baseline"
  baseline.mkdir()
  scalar, _ = _compile(prog, baseline)
  vector, source = _compile(widen_ranges(prog, lanes=8), tmp_path, dialect=dialect)
  assert "void kernel_lanes_1(" in source
  values = np.random.default_rng(194).normal(size=17)
  assert _evaluate(vector, values, 114).tobytes() == _evaluate(scalar, values, 114).tobytes()


@pytest.mark.parametrize("lanes", [2, 4, 8])
def test_scheduled_nonaffine_index_uses_gather(tmp_path, lanes):
  x, y = p.buffer("x", dtypes.float64, (41,)), p.buffer("y", dtypes.float64, (17,))
  i = p.var("i")
  offset = p.assign("offset", p.mul(p.div(i, p.const_int(2)), p.const_int(3)), declare=True)
  index = p.add(p.var("offset"), i)
  store = p.store(p.view(y, [i]), p.load(p.view(x, [index])))
  prog = p.program([p.proc("kernel", [x, y], [p.for_(p.range_("i", 0, 17), [offset, store])])])
  function, source = _compile(widen_ranges(prog, lanes=lanes), tmp_path)
  assert "void kernel_lanes_1(" in source
  values = np.arange(41, dtype=np.float64)
  expected = values[(np.arange(17) // 2) * 3 + np.arange(17)]
  np.testing.assert_array_equal(_evaluate(function, values, 17)[:17], expected)


def _compile_module(module, tmp_path, dialect):
  source = (
    module.body
    + f"\nvoid run(double *x, double *y) {{ const double *arg[] = {{x}}; double *res[] = {{y}}; double w[{max(1, module.workspace_size)}]; {module.fun.name}(arg, res, NULL, w, 0); }}\n"
  )
  return _compile_source(source, tmp_path, dialect=dialect)


@pytest.mark.parametrize("dialect", ["gnu", "c"])
def test_call_jacobian_keeps_noinline_frame_in_c99(tmp_path, dialect):
  import scaly as sc
  from scaly.codegen.aot import render_c_module

  @sc.function(sc.arg("x", 2), outputs=sc.arg("y", 2))
  def inner(x: sc.Expr) -> sc.Expr:
    return x.sin() + x * x

  @sc.function(sc.arg("z", 2), outputs=sc.arg("y", 2))
  def outer(z: sc.Expr) -> sc.Expr:
    return inner(z * z)

  jac = outer.factory("kernel", ["z"], [sc.factory.Jac("y", "z")])
  module = render_c_module(jac, lanes=1, dialect=dialect)
  function, source = _compile_module(module, tmp_path, dialect)
  values = np.asarray([0.3, -0.8])
  actual = _evaluate(function, values, 2)[:4].reshape(2, 2)
  expected = np.diag((np.cos(values * values) + 2 * values * values) * 2 * values)
  np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=1e-15)
  if dialect == "c":
    assert "(*volatile inner_fwd" in source
    assert "__attribute__" not in source
  else:
    assert "__attribute__((noinline))" in source


def test_c99_noinline_implementation_name_is_reserved():
  x, y = p.buffer("x", dtypes.float64, (1,)), p.buffer("y", dtypes.float64, (1,))
  proc = p.proc("f_fwd", [x, y], [p.store(p.view(y, [p.const_int(0)]), p.load(p.view(x, [p.const_int(0)])))])
  source = "\n".join(_render_raw_callee(proc, dialect="c", reserved_names=NameScope({"f_fwd_raw_impl"})))
  assert "void f_fwd_raw_impl_2(" in source
  assert "(*volatile f_fwd_raw)" in source


@pytest.mark.parametrize("dialect", ["gnu", "c"])
def test_public_mapped_block_matrix_promotes_lane_scratch(tmp_path, dialect):
  import scaly as sc
  from scaly.codegen.aot import render_c_module
  from scaly.ir.program import walk_program

  size, stages = 140, 8

  @sc.function(sc.arg("stage_x", size), outputs=sc.arg("stage_y", size), name="matrix_stage")
  def stage(x: sc.Expr) -> sc.Expr:
    hidden = (sc.const(np.eye(size)) @ x).block()
    return hidden * hidden

  @sc.function(sc.arg("x", size * stages), outputs=sc.arg("y", size * stages), name="kernel")
  def fn(z: sc.Expr) -> sc.Expr:
    return sc.vmap(stage, stages)(z).vec()

  module = render_c_module(fn, lanes=4, dialect=dialect)
  main = module.program.args[-1]
  assert any(n.attrs.get("vector_mapped") for n in walk_program(main))
  assert module.workspace_size >= size * 8
  function, _ = _compile_module(module, tmp_path, dialect)
  values = np.linspace(-2, 3, size * stages)
  np.testing.assert_allclose(_evaluate(function, values, size * stages)[: size * stages], values * values, rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize("dialect", ["gnu", "c"])
def test_mutable_integer_indices_stay_scalar(tmp_path, dialect):
  x, y = p.buffer("x", dtypes.float64, (20,)), p.buffer("y", dtypes.float64, (9,))
  i, j = p.var("i"), p.var("j")
  offset = p.assign("offset", i, declare=True)
  update = p.assign("offset", p.add(p.mul(p.div(i, p.const_int(2)), p.const_int(3)), j))
  inner = p.for_(p.range_("j", 0, 3), [update])
  store = p.store(p.view(y, [i]), p.load(p.view(x, [p.var("offset")])))
  rng = p.range_("i", 0, 9)
  rng = ProgramNode(rng.op, rng.args, {**rng.attrs, "mapped": True}, rng.dtype)
  prog = p.program([p.proc("kernel", [x, y], [p.for_(rng, [offset, inner, store])])])
  widened = widen_ranges(prog, lanes=4)
  assert widened is prog
  function, _ = _compile(widened, tmp_path, dialect=dialect)
  values = np.arange(20, dtype=np.float64)
  np.testing.assert_array_equal(_evaluate(function, values, 9)[:9], values[(np.arange(9) // 2) * 3 + 2])


@pytest.mark.parametrize("dialect", ["gnu", "c"])
@pytest.mark.parametrize("lanes", [2, 4, 8])
def test_contiguous_access_with_sanitized_range_name(tmp_path, dialect, lanes):
  x, y = p.buffer("x", dtypes.float64, (17,)), p.buffer("y", dtypes.float64, (17,))
  i = p.var("stage.part")
  prog = p.program([p.proc("kernel", [x, y], [p.for_(p.range_("stage.part", 0, 17), [p.store(p.view(y, [i]), p.load(p.view(x, [i])))])])])
  function, source = _compile(widen_ranges(prog, lanes=lanes), tmp_path, dialect=dialect)
  values = np.linspace(-2, 3, 17)
  result = _evaluate(function, values, 17)
  np.testing.assert_array_equal(result[:17], values)
  np.testing.assert_array_equal(result[17:], 8765.0)
  if dialect == "gnu":
    assert re.search(r"\*\(const \w+_vec_mem\*\)", source)


@pytest.mark.parametrize("dialect", ["gnu", "c"])
@pytest.mark.parametrize("lanes", [2, 4, 8, "auto"])
def test_private_aliases_pack_active_lanes_with_tail(tmp_path, dialect, lanes):
  from scaly.ir.program import walk_program

  x, y = p.buffer("x", dtypes.float64, (51,)), p.buffer("y", dtypes.float64, (17,))
  scratch = p.buffer("scratch", dtypes.float64, (1100,), address_space="private")
  alias = ProgramNode(
    ProgramOp.BUFFER, (), {**scratch.attrs, "name": "alias", "shape": (10,), "alias_of": "scratch", "alias_offset": 3}, scratch.dtype
  )
  tail = ProgramNode(ProgramOp.BUFFER, (), {**alias.attrs, "name": "tail", "shape": (3,), "alias_of": "alias", "alias_offset": 2}, scratch.dtype)
  i, j = p.var("i"), p.var("j")
  rng = p.range_("i", 0, 17)
  rng = ProgramNode(rng.op, rng.args, {**rng.attrs, "mapped": True}, rng.dtype)
  fill = p.for_(p.range_("j", 0, 3), [p.store(p.view(tail, [j]), p.load(p.view(x, [p.add(p.mul(i, p.const_int(3)), j)])))])
  value = p.add(p.load(p.view(scratch, [p.const_int(5)])), p.add(p.load(p.view(alias, [p.const_int(3)])), p.load(p.view(tail, [p.const_int(2)]))))
  prog = p.program([p.proc("kernel", [x, y], [p.for_(rng, [scratch, alias, tail, fill, p.store(p.view(y, [i]), value)])])])
  widened = widen_ranges(prog, lanes=lanes)
  if lanes == "auto":
    from scaly.ir.match import Pattern, rewrite
    from scaly.passes.program._common import rebuild_program

    def cap(node):
      return ProgramNode(node.op, node.args, {**node.attrs, "lane_cap": 4, "lane_caps": (4, 4, 4, 4)}, node.dtype)

    widened = rewrite(widened, [Pattern(ProgramOp.RANGE, lambda n: "lane_caps" in n.attrs, cap)], rebuild=rebuild_program, fixpoint=False)
  nodes = list(walk_program(widened))
  width = next(n.attrs["lane_width"] for n in nodes if n.op == ProgramOp.RANGE and "lane_width" in n.attrs)
  assert not any(n.op == ProgramOp.BUFFER and "alias_of" in n.attrs for n in nodes)
  views = [n for n in nodes if n.op == ProgramOp.VIEW and n.attrs.get("lane_local")]
  assert views and all(any(v.op == ProgramOp.VAR and v.attrs["name"] == width for v in walk_program(n.args[0])) for n in views)
  packed = pack_workspace(widened)
  assert packed.args[0].attrs["sz_w"] == 1100 * 8
  function, _ = _compile(packed, tmp_path, dialect=dialect, defines=("-DSCALY_LANES=8",) if lanes == "auto" else ())
  values = np.linspace(-2, 3, 51)
  expected = np.asarray([row[0] + (row[1] + row[2]) for row in values.reshape(17, 3)])
  actual = _evaluate(function, values, 17)
  assert actual[:17].tobytes() == expected.tobytes()
  np.testing.assert_array_equal(actual[17:], 8765.0)


@pytest.mark.parametrize("dialect", ["gnu", "c"])
def test_private_slots_reused_by_helpers_with_different_widths(tmp_path, dialect):
  from scaly.ir.program import walk_program

  counts = (2, 4)
  x, y = p.buffer("x", dtypes.float64, (4,)), p.buffer("y", dtypes.float64, (6,))
  loops = []
  for part, count in enumerate(counts):
    scratch = p.buffer(f"scratch{part}", dtypes.float64, (1100,), address_space="private")
    i = p.var(f"i{part}")
    rng = p.range_(f"i{part}", 0, count)
    rng = ProgramNode(rng.op, rng.args, {**rng.attrs, "mapped": True}, rng.dtype)
    value = p.mul(p.load(p.view(x, [i])), p.const_float(part + 2))
    loops.append(
      p.for_(
        rng,
        [
          scratch,
          p.store(p.view(scratch, [p.const_int(1)]), value),
          p.store(p.view(y, [p.add(i, p.const_int(sum(counts[:part])))]), p.load(p.view(scratch, [p.const_int(1)]))),
        ],
      )
    )
  widened = widen_ranges(p.program([p.proc("kernel", [x, y], loops)]), lanes="auto")
  caps = {n.attrs["lane_width"]: n.attrs["lane_caps"] for n in walk_program(widened) if n.op == ProgramOp.RANGE and "lane_caps" in n.attrs}
  assert sorted(caps.values()) == [(2,) * 4, (4,) * 4]
  packed = pack_workspace(widened)
  assert packed.args[0].attrs["sz_w"] == 1100 * 8
  function, _ = _compile(packed, tmp_path, dialect=dialect, defines=("-DSCALY_LANES=8",))
  values = np.linspace(-2, 3, 4)
  actual = _evaluate(function, values, 6)
  assert actual[:6].tobytes() == np.concatenate([values[:2] * 2, values * 3]).tobytes()
  np.testing.assert_array_equal(actual[6:], 8765.0)
