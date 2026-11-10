"""Generated frames fit the compiler warning limit and spilling preserves values."""

import ctypes
import subprocess

import numpy as np
import pytest

import scaly as sc
from scaly.codegen.aot import render_c_api_header, render_c_module, render_c_source, workspace_size, write_module
from scaly.codegen.toolchain import BuildRecipe
from scaly.ir.program import ProgramOp, walk_program
from tests.compilers import named_compiler


def _function():
  @sc.function(sc.arg("x", 900), outputs=sc.arg("y"), name="stage")
  def stage(x):
    value = x.sin().block()
    return value.sum() + (value * value).sum()

  @sc.function(sc.arg("x", 900), outputs=sc.arg("y"), name="kernel")
  def kernel(x):
    return sum((stage(x + i) for i in range(10)))

  return kernel


@pytest.mark.parametrize("compiler", ["gcc", "clang"])
@pytest.mark.parametrize("lanes,dialect", [(1, "gnu"), (4, "gnu"), ("auto", "c")])
@pytest.mark.parametrize("budget", [65536, 8192, 0])
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
      f"-Wframe-larger-than={module.frame_bytes}",
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
  handle.kernel.argtypes = [ctypes.POINTER(pointer), ctypes.POINTER(pointer), ctypes.c_void_p, pointer, ctypes.c_int]
  data = np.linspace(-2, 3, 900)
  result = np.zeros(1)
  work = np.zeros(module.workspace_size)
  status = handle.kernel(
    (pointer * 1)(data.ctypes.data_as(pointer)), (pointer * 1)(result.ctypes.data_as(pointer)), None, work.ctypes.data_as(pointer), 0
  )
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


@pytest.mark.parametrize("budget", [-1, True, 1.5])
def test_invalid_frame_budget(budget):
  with pytest.raises(ValueError, match="frame_budget"):
    BuildRecipe(frame_budget=budget)
