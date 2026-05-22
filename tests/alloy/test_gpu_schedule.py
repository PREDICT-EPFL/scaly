"""Phase 7: device-placed Functions lower to host-driver PROC + KERNEL.

Today no GPU backend renders the kernel — the C renderer probe returns False
for non-host functions and the JIT path raises a clear ``JitError``. These
tests pin the Program IR shape so a future Phase 8 backend has a stable
contract to render against.
"""

from __future__ import annotations

import pytest

import alloy as al
from alloy.lowering import lower_function, main_proc
from alloy.program import POps, format_program, verify_program


def test_device_placed_function_lowers_to_kernel_plus_driver() -> None:
  x = al.sym("x", 8)
  fn = al.Function("f_dev", [x], [x.sin()], ["x"], ["y"], device="cuda:0")
  prog = lower_function(fn)
  assert prog.op == POps.PROGRAM
  assert int(prog.attrs["proc_count"]) == 1
  assert int(prog.attrs["kernel_count"]) == 1
  driver = main_proc(prog)
  assert driver.attrs["name"] == "f_dev"
  # the host driver body should be a single LAUNCH
  body_start = int(driver.attrs["param_count"])
  body = driver.args[body_start:]
  assert len(body) == 1
  assert body[0].op == POps.LAUNCH
  assert body[0].attrs["kernel"] == "f_dev_kernel"
  # find the kernel in prog.args (after the procs)
  pc = int(prog.attrs["proc_count"])
  kernel = prog.args[pc]
  assert kernel.op == POps.KERNEL
  assert kernel.attrs["device"].kind == "cuda"
  verify_program(prog)


def test_device_placed_function_kernel_contains_loop_body() -> None:
  # Use float32 since Metal's BackendSupport entry doesn't advertise float64.
  x = al.sym("x", 16, dtype=al.dtypes.float32)
  fn = al.Function("f_dev_loop", [x], [x.cos()], ["x"], ["y"], device="metal:0")
  prog = lower_function(fn)
  text = format_program(prog)
  assert "kernel f_dev_loop_kernel" in text
  assert "proc f_dev_loop" in text
  assert "launch f_dev_loop_kernel" in text
  assert "for i_y in [0, 16)" in text  # body landed inside the kernel


def test_launch_grid_size_from_first_global_for() -> None:
  """The schedule pass picks the first GLOBAL FOR's stop bound as the launch grid size."""
  x = al.sym("x", 24)
  fn = al.Function("f_grid", [x], [x.sin()], ["x"], ["y"], device="cuda:0")
  prog = lower_function(fn)
  driver = main_proc(prog)
  launch = driver.args[int(driver.attrs["param_count"])]
  grid_dims = int(launch.attrs["grid_dims"])
  grid_arg = launch.args[0]
  assert grid_dims == 1
  assert grid_arg.op == POps.CONST_INT
  assert grid_arg.attrs["value"] == 24


def test_jit_call_on_device_function_raises_clear_error() -> None:
  x = al.sym("x", 4)
  fn = al.Function("f_dev_call", [x], [-x], ["x"], ["y"], device="cuda:0")
  from alloy.jit import JitError

  import numpy as np

  with pytest.raises(JitError, match="only host lowering"):
    fn(np.array([1.0, 2.0, 3.0, 4.0]))


def test_mixed_device_call_lowers_as_external_universal_abi() -> None:
  """A host caller invoking a CUDA-placed callee now lowers to an external CALL
  through the universal ABI (Phase 9). The callee's KERNEL TU is the user's
  responsibility to link in; the host C just forward-declares it."""
  from alloy.codegen.program_c import render_program_c_source

  x = al.sym("x", 3)
  inner_gpu = al.Function("inner_gpu_call", [x], [x.sin()], ["x"], ["y"], device="cuda:0")
  z = al.sym("z", 3)
  (out,) = inner_gpu.call([z])
  outer_host = al.Function("outer_host_mixed", [z], [out], ["z"], ["y"])
  prog = lower_function(outer_host)
  text = format_program(prog)
  assert "call inner_gpu_call(" in text
  src = render_program_c_source(outer_host)
  assert "extern int inner_gpu_call(const double** arg, double** res, int* iw, double* w, void* mem);" in src
  assert "inner_gpu_call(_ext_arg, _ext_res, NULL, NULL, NULL);" in src


def test_program_ir_renderer_falls_back_for_device_function() -> None:
  """``can_render_program_c`` returns False for non-host placements so the
  legacy renderer (also unsupported for non-host) is not invoked either —
  the JIT layer raises the clear ``only host lowering`` diagnostic first."""
  from alloy.codegen.program_c import can_render_program_c

  x = al.sym("x", 4)
  fn = al.Function("f_dev_probe", [x], [x.sin()], ["x"], ["y"], device="cuda:0")
  assert not can_render_program_c(fn)
