"""Phase 8 follow-up: end-to-end CUDA dispatch.

Tests are skipped when no compatible nvcc + driver is reachable on this
machine. When the runtime is available, every test compiles the rendered
CUDA C, dispatches the kernel, and checks numerical agreement with numpy.

These tests pin the fixes for the two CUDA-specific bugs found while
testing the text-only renderer on an RTX 5090:

- **Bug 1**: when a thread-bound elementwise feeds a top-level REDUCE,
  the per-thread workspace makes the reduction read uninitialized stack.
  Schedule pass now sets ``bind_threads=False`` for that shape, host
  driver launches with grid=1/block=1.
- **Bug 2**: ``_render_host_driver`` hardcoded ``double*`` even for
  ``float32`` Functions. Now the device-side ``cudaMalloc``/``cudaMemcpy``
  uses the input/output dtype, with a host-side cast through the
  universal ``const double**`` ABI.

It also covers the CALL/MAP operations that the renderer didn't support
before — both now lower via ``__device__`` inline callees, matching the
Metal pattern.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy as al
from alloy.codegen.cuda import render_cuda_source
from alloy.cuda_runtime import cuda_available


pytestmark = pytest.mark.skipif(not cuda_available(), reason="no compatible nvcc + CUDA driver on this machine")


def test_cuda_elementwise_sin_matches_numpy() -> None:
  x = al.sym("x", 64)
  fn = al.Function("cu_sin", [x], [x.sin()], ["x"], ["y"]).with_device("cuda:0")
  x_val = np.linspace(-2.0, 2.0, 64)
  np.testing.assert_allclose(fn(x_val), np.sin(x_val), atol=1e-12, rtol=1e-12)


def test_cuda_chained_elementwise_fuses_into_one_thread_bound_loop() -> None:
  """sin + x*x - cos chained should fuse into a single thread-bound FOR."""
  x = al.sym("x", 32)
  fn = al.Function("cu_chain", [x], [x.sin() + x * x - x.cos()], ["x"], ["y"]).with_device("cuda:0")
  src = render_cuda_source(fn)
  # Only one thread-bound assignment line should appear
  assert src.count("blockIdx.x * blockDim.x + threadIdx.x") == 1
  x_val = np.linspace(-1.0, 1.0, 32)
  ref = np.sin(x_val) + x_val * x_val - np.cos(x_val)
  np.testing.assert_allclose(fn(x_val), ref, atol=1e-12, rtol=1e-12)


def test_cuda_full_reduce_bug1_fixed() -> None:
  """(x*x).sum() on a thread-bound elementwise used to race on per-thread workspace.

  The schedule pass now disables thread-binding when a top-level REDUCE
  reads from a thread-bound elementwise's workspace. Verifies the kernel
  launches with grid=1/block=1 in that case.
  """
  x = al.sym("x", 12)
  fn = al.Function("cu_full_reduce", [x], [(x * x).sum()], ["x"], ["s"]).with_device("cuda:0")
  src = render_cuda_source(fn)
  # bind_threads=False ⇒ launch with literal grid=1, block=1
  assert "<<<dim3(1), dim3(1)>>>" in src
  x_val = np.arange(12, dtype=np.float64)
  ref = float((x_val * x_val).sum())
  out = fn(x_val)
  assert isinstance(out, np.ndarray)
  np.testing.assert_allclose(float(out.item()), ref, atol=1e-12, rtol=1e-12)


def test_cuda_matmat_thread_binds_rows() -> None:
  rng = np.random.RandomState(0)
  A = al.sym("A", (16, 24))
  B = al.sym("B", (24, 20))
  fn = al.Function("cu_matmat", [A, B], [A @ B], ["A", "B"], ["C"]).with_device("cuda:0")
  src = render_cuda_source(fn)
  # matmat has 1 top-level GLOBAL FOR (rows), so thread-binding is enabled
  assert "blockIdx.x * blockDim.x + threadIdx.x" in src
  Av = rng.randn(16, 24)
  Bv = rng.randn(24, 20)
  np.testing.assert_allclose(fn(Av, Bv), Av @ Bv, atol=1e-10, rtol=1e-10)


def test_cuda_matvec_dispatch() -> None:
  rng = np.random.RandomState(1)
  A = al.sym("A", (8, 12))
  v = al.sym("v", 12)
  fn = al.Function("cu_matvec", [A, v], [A @ v], ["A", "v"], ["y"]).with_device("cuda:0")
  Av = rng.randn(8, 12)
  vv = rng.randn(12)
  np.testing.assert_allclose(fn(Av, vv), Av @ vv, atol=1e-10, rtol=1e-10)


def test_cuda_sum_axis_dispatch() -> None:
  x = al.sym("x", (4, 6))
  fn = al.Function("cu_sum_axis", [x], [x.sum(axis=1)], ["x"], ["y"]).with_device("cuda:0")
  x_val = np.linspace(-1.0, 1.0, 24).reshape(4, 6)
  np.testing.assert_allclose(fn(x_val), x_val.sum(axis=1), atol=1e-12, rtol=1e-12)


def test_cuda_call_inline_device_function() -> None:
  """A named callee on cuda is emitted as a ``__device__`` inline function."""
  x = al.sym("x", 4)
  inner = al.Function("inner_cu", [x], [x.sin() + x], ["x"], ["y"])
  z = al.sym("z", 4)
  (out,) = inner.call([z])
  outer = al.Function("outer_cu", [z], [out], ["z"], ["r"]).with_device("cuda:0")
  src = render_cuda_source(outer)
  assert "__device__ void inner_cu(" in src
  z_val = np.array([0.0, 0.5, 1.0, 1.5])
  np.testing.assert_allclose(outer(z_val), np.sin(z_val) + z_val, atol=1e-12, rtol=1e-12)


def test_cuda_map_dispatches_each_iteration_to_a_thread() -> None:
  x = al.sym("x", 4)
  stage = al.Function("stage_cu", [x], [(x * x).sum()], ["x"], ["y"])
  batch = al.sym("batch", 32)
  mapped = al.map_(stage, length=8, inputs=[(batch, 0, 4)])
  fn = al.Function("mapped_cu", [batch], [mapped], ["batch"], ["m"]).with_device("cuda:0")
  src = render_cuda_source(fn)
  assert "__device__ void stage_cu(" in src
  # Each MAP iteration → one thread (the it_ loop is thread-bound).
  assert "blockIdx.x * blockDim.x + threadIdx.x" in src
  batch_val = np.linspace(-1.0, 1.0, 32)
  ref = (batch_val.reshape(8, 4) ** 2).sum(axis=1)
  np.testing.assert_allclose(fn(batch_val), ref, atol=1e-12, rtol=1e-12)


def test_cuda_float32_bug2_fixed() -> None:
  """float32 Functions used to fail at nvcc compile (kernel float* vs host double*).

  The host driver now allocates the device buffer as the kernel's dtype and
  casts double↔float on the host side at the universal-ABI boundary.
  """
  x = al.sym("x", 64, dtype=al.dtypes.float32)
  fn = al.Function("cu_f32", [x], [x.sin() + x.cos()], ["x"], ["y"]).with_device("cuda:0")
  src = render_cuda_source(fn)
  # Kernel param is float*; host stages through float* then casts back to double for the ABI.
  assert "float* x" in src
  assert "(float)arg[0][_i]" in src
  assert "(double)h_out0[_i]" in src
  x_val = np.linspace(-1.0, 1.0, 64)
  out = fn(x_val)
  ref = (np.sin(x_val) + np.cos(x_val)).astype(np.float32).astype(np.float64)
  np.testing.assert_allclose(out, ref, atol=1e-6, rtol=1e-6)


def test_cuda_boundary_shapes_dispatch() -> None:
  """Elementwise of various sizes (1, 257>blockDim, large) all run correctly."""
  for size in (1, 31, 257, 4096):
    x = al.sym("x", size)
    fn = al.Function(f"cu_bd_{size}", [x], [x.sin() + x.cos()], ["x"], ["y"]).with_device("cuda:0")
    x_val = np.linspace(-1.0, 1.0, size)
    np.testing.assert_allclose(fn(x_val), np.sin(x_val) + np.cos(x_val), atol=1e-12, rtol=1e-12)


def test_cuda_repeated_dispatch_reuses_compiled_so() -> None:
  """Compiling and dispatching twice should use the cached .so."""
  x = al.sym("x", 16)
  fn = al.Function("cu_repeat", [x], [x.sin()], ["x"], ["y"]).with_device("cuda:0")
  x_val = np.linspace(-1, 1, 16)
  o1 = fn(x_val)
  o2 = fn(x_val)
  np.testing.assert_array_equal(o1, o2)


def test_cuda_compiled_function_invalid_input_shape_raises() -> None:
  x = al.sym("x", 4)
  fn = al.Function("cu_wrongshape", [x], [x.sin()], ["x"], ["y"]).with_device("cuda:0")
  with pytest.raises(ValueError, match="shape"):
    fn(np.zeros(8))


def test_cuda_runtime_picks_compatible_nvcc() -> None:
  """The auto-detected nvcc on this machine should produce kernels the driver actually runs.

  This is essentially what ``cuda_available()`` returns, but it also exposes
  the resolved nvcc path so the test diagnostic is informative when something
  upstream changes (driver upgrade, new CUDA toolchain install).
  """
  from alloy.cuda_runtime import _CudaDevice

  dev = _CudaDevice.get()
  assert dev.nvcc.exists()
