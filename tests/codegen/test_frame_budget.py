"""Generated frames fit the compiler warning limit and spilling preserves values."""

import ctypes
import subprocess

import numpy as np
import pytest

import scaly as sc
from scaly.codegen.aot import render_c_api_header, render_c_module, render_c_source, workspace_size, write_module
from scaly.codegen.toolchain import BuildRecipe
from scaly.ir.program import ProgramOp, walk_program
from tests.compilers import cxx_compiler, named_compiler


def _function():
  @sc.function(sc.arg("x", 900), outputs=sc.arg("y"), name="stage")
  def stage(x):
    value = x.sin().block()
    return value.sum() + (value * value).sum()

  @sc.function(sc.arg("x", 900), outputs=sc.arg("y"), name="kernel")
  def kernel(x):
    return sum((stage(x + i) for i in range(10)))

  return kernel


def test_zero_frame_budget_spills_bool_mask_with_float32_leaves(tmp_path) -> None:
  n = 4096
  x_type = sc.TensorType((n,), dtype=sc.dtypes.float32)

  @sc.function(sc.arg("x", x_type), outputs=sc.arg("y"))
  def masked(x: sc.Expr) -> sc.Expr:
    mask = (x > 0).block()
    return sc.where(mask, x, -x) + sc.where(mask[::-1], x, -x)

  module = write_module(masked, tmp_path, frame_budget=0)
  assert module.frame_bytes == 0
  assert any(
    slot.op == ProgramOp.BUFFER and slot.dtype == sc.dtypes.bool_ and "workspace_offset" in slot.attrs for slot in walk_program(module.program)
  )
  assert "bool b[4096];" in module.header
  driver = tmp_path / "driver.c"
  driver.write_text(
    '#include "masked.h"\n'
    "#include <string.h>\n"
    "void run(const float* data, float* result) {\n"
    "  static masked_x_t x;\n"
    "  static masked_y_t y;\n"
    "  static masked_workspace_t w;\n"
    "  memcpy(x.data, data, sizeof x.data);\n"
    "  masked(&x, &y, &w);\n"
    "  memcpy(result, y.data, sizeof y.data);\n"
    "}\n"
  )
  lib = tmp_path / "masked.so"
  subprocess.run(
    [*named_compiler("gcc"), "-std=c11", "-O2", "-shared", "-fPIC", str(driver), str(tmp_path / module.source_name), "-lm", "-o", str(lib)],
    check=True,
  )
  x = np.linspace(-2, 3, n, dtype=np.float32)
  result = np.empty_like(x)
  ctypes.CDLL(str(lib)).run(ctypes.c_void_p(x.ctypes.data), ctypes.c_void_p(result.ctypes.data))
  np.testing.assert_array_equal(result, np.abs(x) + np.where(x[::-1] > 0, x, -x))


def test_cond_callee_gets_workspace_at_zero_frame_budget(tmp_path) -> None:
  n = 2000

  @sc.function(sc.arg("x", n), outputs=sc.arg("y"))
  def plain(x: sc.Expr) -> sc.Expr:
    return x.sum()

  @sc.function(sc.arg("x", n), outputs=sc.arg("y"))
  def spilled(x: sc.Expr) -> sc.Expr:
    values = x.sin().block()
    return values.sum() + (values * values).sum()

  @sc.function(sc.arg("choose", sc.TensorType((), dtype=sc.dtypes.bool_)), sc.arg("x", n), outputs=sc.arg("y"))
  def conditional(choose: sc.Expr, x: sc.Expr) -> sc.Expr:
    return sc.cond(choose, (plain, spilled), x)

  module = write_module(conditional, tmp_path, frame_budget=0)
  assert module.frame_bytes == 0
  assert module.workspace_size >= n * sc.dtypes.float64.itemsize
  driver = tmp_path / "driver.c"
  driver.write_text(
    '#include "conditional.h"\n'
    "int run(bool choose, const double* x, double* y) {\n"
    "  const void* arg[] = {&choose, x};\n"
    "  void* res[] = {y};\n"
    "  static conditional_workspace_t w;\n"
    "  return conditional_abi(arg, res, &w);\n"
    "}\n"
  )
  library = tmp_path / "conditional.so"
  subprocess.run(
    [*named_compiler("gcc"), "-std=c11", "-O2", "-shared", "-fPIC", str(driver), str(tmp_path / module.source_name), "-lm", "-o", str(library)],
    check=True,
  )
  x = np.linspace(-0.7, 1.3, n)
  values = np.sin(x)
  y = ctypes.c_double()
  run = ctypes.CDLL(str(library)).run
  run.argtypes = [ctypes.c_bool, ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
  for choose, expected in [(False, x.sum()), (True, values.sum() + (values * values).sum())]:
    assert run(choose, x.ctypes.data, ctypes.byref(y)) == 0
    np.testing.assert_allclose(y.value, expected, rtol=1e-13)


@pytest.mark.slow
@pytest.mark.parametrize("compiler", ["gcc", "clang"])
@pytest.mark.parametrize("lanes,dialect", [(1, "gnu"), (4, "gnu"), ("auto", "c")])
@pytest.mark.parametrize("budget", [8192, 0])
def test_generated_frames_fit_and_match_numpy(tmp_path, compiler, lanes, dialect, budget):
  command = named_compiler(compiler)
  module = write_module(_function(), tmp_path, frame_budget=budget, lanes=lanes, dialect=dialect, cpu="native")
  assert module.workspace_size > 0
  assert module.frame_bytes <= max(budget, 4096)
  assert f"#define kernel_FRAME_BYTES {module.frame_bytes}" in module.header
  if budget == 0:
    assert all(
      "workspace_offset" in n.attrs
      for n in walk_program(module.program)
      if n.op == ProgramOp.BUFFER and n.attrs.get("address_space") == "private" and "alias_of" not in n.attrs
    )
  library = tmp_path / "kernel.so"
  subprocess.run(
    [
      *command,
      "-O3",
      *module.recipe.cpu_flags,
      "-fno-math-errno",
      "-shared",
      "-fPIC",
      f"-Wframe-larger-than={max(module.frame_bytes, 4096)}",
      "-Werror",
      str(tmp_path / module.source_name),
      "-lm",
      "-o",
      str(library),
    ],
    capture_output=True,
    text=True,
    check=True,
  )
  handle = ctypes.CDLL(str(library))
  pointer = ctypes.POINTER(ctypes.c_double)
  handle.kernel_abi.argtypes = [ctypes.POINTER(pointer), ctypes.POINTER(pointer), ctypes.c_void_p]
  data = np.linspace(-2, 3, 900)
  result = np.zeros(1)
  scratch = np.empty(module.workspace_size + 63, dtype=np.uint8)
  work = ctypes.c_void_p((scratch.ctypes.data + 63) & ~63)
  status = handle.kernel_abi((pointer * 1)(data.ctypes.data_as(pointer)), (pointer * 1)(result.ctypes.data_as(pointer)), work)
  assert status == 0
  expected = sum((np.sin(data + i).sum() + np.square(np.sin(data + i)).sum()) for i in range(10))
  np.testing.assert_allclose(result[0], expected, rtol=1e-12)


@pytest.mark.parametrize("compiler", ["gcc", "clang"])
def test_compiler_gate_rejects_unspilled_frame(tmp_path, compiler):
  source = tmp_path / "unspilled.c"
  source.write_text(render_c_source(_function(), frame_budget=1000000, lanes=1))
  result = subprocess.run(
    [
      *named_compiler(compiler),
      "-O3",
      "-fno-math-errno",
      "-Wframe-larger-than=4096",
      "-Werror",
      "-c",
      str(source),
      "-o",
      str(tmp_path / "unspilled.o"),
    ],
    capture_output=True,
    text=True,
  )
  assert result.returncode != 0
  assert "frame-larger-than" in result.stderr


def test_frame_option_reaches_all_render_entry_points(tmp_path):
  fun = _function()
  module = render_c_module(fun, frame_budget=0)
  assert workspace_size(fun, frame_budget=0) == module.workspace_size
  assert render_c_source(fun, frame_budget=0) == module.body
  assert render_c_api_header(fun, frame_budget=0) == module.header
  assert write_module(fun, tmp_path, frame_budget=0).header == module.header
  assert f"#define kernel_FRAME_BYTES {module.frame_bytes}" in render_c_api_header(fun, frame_budget=0, lang="cpp")


def test_frame_macro_does_not_shadow_buffer_names(tmp_path):
  @sc.function(sc.arg("kernel_FRAME_BYTES", 3), outputs=sc.arg("y"), name="inner")
  def inner(x):
    return x.sin().block()

  @sc.function(sc.arg("kernel_FRAME_BYTES", 3), outputs=sc.arg("y"), name="kernel")
  def kernel(x):
    return inner(x)

  module = write_module(kernel, tmp_path, lanes=1)
  subprocess.run(
    [*named_compiler("gcc"), "-Werror", "-c", str(tmp_path / module.source_name), "-o", str(tmp_path / "kernel.o")],
    capture_output=True,
    text=True,
    check=True,
  )
  cpp = render_c_api_header(kernel, lang="cpp", lanes=1)
  (tmp_path / "kernel.hpp").write_text(cpp)
  cpp_path = tmp_path / "use.cpp"
  cpp_path.write_text('#include "kernel.hpp"\n')
  subprocess.run(
    [*cxx_compiler(), "-x", "c++", "-std=c++17", "-Werror", "-c", str(cpp_path), "-o", str(tmp_path / "header.o")],
    capture_output=True,
    text=True,
    check=True,
  )
  assert "* kernel_FRAME_BYTES," not in module.header
  assert "& kernel_FRAME_BYTES," not in cpp


def test_frame_macro_does_not_shadow_sparse_namespace(tmp_path):
  @sc.function(sc.arg("FRAME_BYTES", 2), outputs=sc.arg("y"), name="base")
  def base(x):
    return x * x

  fun = base.factory("spjac_y", ["FRAME_BYTES"], [sc.factory.SpJac("y", "FRAME_BYTES")])
  module = write_module(fun, tmp_path, lang="cpp", lanes=1)
  path = tmp_path / "use.cpp"
  path.write_text(f'#include "{module.header_name}"\n')
  result = subprocess.run(
    [*named_compiler("gcc"), "-x", "c++", "-std=c++17", "-Werror", "-c", str(path), "-o", str(tmp_path / "use.o")],
    capture_output=True,
    text=True,
  )
  assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("budget", [-1, True, 1.5])
def test_invalid_frame_budget(budget):
  with pytest.raises(ValueError, match="frame_budget"):
    BuildRecipe(frame_budget=budget)
