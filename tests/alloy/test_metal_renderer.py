"""Phase 8 follow-up: Metal Shading Language kernel source rendering.

Tests pin the rendered MSL shape. When ``xcrun metal`` is available with the
Metal Toolchain installed, an optional test compiles the MSL to an ``.air``
object to prove it's syntactically valid; otherwise it skips.

Host-side Metal dispatch (PyObjC or metal-cpp) is not generated yet.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

import alloy as al
from alloy.codegen.metal import can_render_metal, render_metal_source
from alloy.metal_runtime import compile_msl_to_metallib, metal_available


def _xcrun_metal_available() -> bool:
  if shutil.which("xcrun") is None:
    return False
  try:
    res = subprocess.run(["xcrun", "metal", "--version"], capture_output=True, text=True)
    return res.returncode == 0
  except (FileNotFoundError, OSError):
    return False


def test_can_render_metal_only_for_metal_device() -> None:
  x = al.sym("x", 4, dtype=al.dtypes.float32)
  host_fn = al.Function("f_metal_host", [x], [x.sin()], ["x"], ["y"])
  metal_fn = host_fn.with_device("metal:0")
  assert not can_render_metal(host_fn)
  assert can_render_metal(metal_fn)


def test_render_metal_kernel_shape() -> None:
  x = al.sym("x", 8, dtype=al.dtypes.float32)
  fn = al.Function("f_metal_sin", [x], [x.sin()], ["x"], ["y"]).with_device("metal:0")
  src = render_metal_source(fn)
  assert "#include <metal_stdlib>" in src
  assert "using namespace metal;" in src
  assert "kernel void f_metal_sin_kernel(" in src
  # buffer bindings in declaration order
  assert "device float* x [[buffer(0)]]" in src
  assert "device float* y [[buffer(1)]]" in src
  # thread index
  assert "uint tid [[thread_position_in_grid]]" in src
  # top-level GLOBAL FOR bound to tid (loop var is named after the output buffer)
  assert "(long)tid" in src
  assert "if (i_y >= 8) return;" in src
  # math function via metal:: namespace
  assert "metal::sin(x[i_y])" in src


def test_render_metal_batched_matmul() -> None:
  # Text-only: renders MSL for (B,M,K)@(B,K,N); the batch loop binds to tid.
  a = al.sym("a", (3, 2, 4), dtype=al.dtypes.float32)
  b = al.sym("b", (3, 4, 2), dtype=al.dtypes.float32)
  fn = al.Function("f_metal_bmm", [a, b], [a @ b], ["a", "b"], ["c"]).with_device("metal:0")
  assert can_render_metal(fn)
  src = render_metal_source(fn)
  assert "kernel void f_metal_bmm_kernel(" in src
  assert "uint tid [[thread_position_in_grid]]" in src
  # batch is the top-level GLOBAL FOR -> bound to tid, capped at B=3
  assert "(long)tid" in src
  assert "if (b_c >= 3) return;" in src


def test_render_metal_supports_binary_elementwise_and_const() -> None:
  x = al.sym("x", 4, dtype=al.dtypes.float32)
  y = al.sym("y", 4, dtype=al.dtypes.float32)
  fn = al.Function(
    "f_metal_bin",
    [x, y],
    [x * y + al.const([1.0, 2.0, 3.0, 4.0], dtype=al.dtypes.float32)],
    ["x", "y"],
    ["z"],
  ).with_device("metal:0")
  src = render_metal_source(fn)
  assert "device float* x [[buffer(0)]]" in src
  assert "device float* y [[buffer(1)]]" in src
  assert "device float* z [[buffer(2)]]" in src
  # MSL float literal with 'f' suffix
  assert "1f" in src or "1.0f" in src or "1.00000000" in src or "1.00000000f" in src


def test_render_metal_rejects_host_function() -> None:
  x = al.sym("x", 4, dtype=al.dtypes.float32)
  fn = al.Function("f_metal_host_only", [x], [x.sin()], ["x"], ["y"])
  with pytest.raises(ValueError, match="needs device='metal:N'"):
    render_metal_source(fn)


def test_render_metal_rejects_float64_at_construction() -> None:
  # Metal's BackendSupport doesn't advertise float64; with_device should fail.
  x = al.sym("x", 4)  # default float64
  fn = al.Function("f_metal_f64", [x], [x.sin()], ["x"], ["y"])
  with pytest.raises(ValueError, match="cannot lower dtype float64"):
    fn.with_device("metal:0")


@pytest.mark.skipif(
  not _xcrun_metal_available(),
  reason="xcrun metal toolchain not installed (run `xcodebuild -downloadComponent MetalToolchain`)",
)
def test_metal_msl_compiles_with_xcrun() -> None:
  """Compile the generated MSL to an ``.air`` object to prove it's syntactically valid."""
  x = al.sym("x", 8, dtype=al.dtypes.float32)
  fn = al.Function("f_metal_compile", [x], [x.sin() + x], ["x"], ["y"]).with_device("metal:0")
  src = render_metal_source(fn)
  with tempfile.TemporaryDirectory() as td:
    metal_path = Path(td) / "kernel.metal"
    air_path = Path(td) / "kernel.air"
    metal_path.write_text(src)
    res = subprocess.run(
      ["xcrun", "metal", "-c", str(metal_path), "-o", str(air_path)],
      capture_output=True,
      text=True,
    )
    assert res.returncode == 0, f"xcrun metal failed:\n{res.stderr}"
    assert air_path.exists()


@pytest.mark.skipif(
  not metal_available(),
  reason="Metal runtime unavailable (need xcrun metal + Metal framework on macOS)",
)
def test_metal_kernel_dispatch_matches_numpy_elementwise() -> None:
  """End-to-end: render MSL, compile via xcrun, dispatch via ctypes/libobjc on the system Metal device."""
  import numpy as np
  from alloy.metal_runtime import MetalRuntime

  x = al.sym("x", 64, dtype=al.dtypes.float32)
  fn = al.Function("sin_dispatch", [x], [x.sin()], ["x"], ["y"]).with_device("metal:0")
  msl = render_metal_source(fn)
  metallib = compile_msl_to_metallib(msl, name="sin_dispatch")
  rt = MetalRuntime()
  x_val = np.linspace(-2.0, 2.0, 64, dtype=np.float32)
  y_buf = np.zeros_like(x_val)
  outs = rt.run_kernel(metallib, "sin_dispatch_kernel", [x_val, y_buf], grid=64)
  np.testing.assert_allclose(outs[1], np.sin(x_val), atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(
  not metal_available(),
  reason="Metal runtime unavailable",
)
def test_metal_function_call_dispatches_through_jit() -> None:
  """User-facing ``fn(x)`` on a Metal-placed Function routes through the Metal runtime."""
  import numpy as np

  x = al.sym("x", 32, dtype=al.dtypes.float32)
  fn = al.Function("call_dispatch", [x], [x.sin() + x * x], ["x"], ["y"]).with_device("metal:0")
  x_val = np.linspace(-1.5, 1.5, 32, dtype=np.float32)
  out = fn(x_val)
  ref = np.sin(x_val) + x_val * x_val
  np.testing.assert_allclose(out, ref, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(
  not metal_available(),
  reason="Metal runtime unavailable",
)
def test_metal_matvec_dispatch_matches_numpy() -> None:
  import numpy as np

  A = al.sym("A", (8, 12), dtype=al.dtypes.float32)
  v = al.sym("v", 12, dtype=al.dtypes.float32)
  fn = al.Function("matvec_metal", [A, v], [A @ v], ["A", "v"], ["y"]).with_device("metal:0")
  rng = np.random.RandomState(0)
  A_val = rng.randn(8, 12).astype(np.float32)
  v_val = rng.randn(12).astype(np.float32)
  out = fn(A_val, v_val)
  np.testing.assert_allclose(out, A_val @ v_val, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
  not metal_available(),
  reason="Metal runtime unavailable",
)
def test_metal_matmat_dispatch_matches_numpy() -> None:
  import numpy as np

  A = al.sym("A", (4, 6), dtype=al.dtypes.float32)
  B = al.sym("B", (6, 5), dtype=al.dtypes.float32)
  fn = al.Function("matmat_metal", [A, B], [A @ B], ["A", "B"], ["C"]).with_device("metal:0")
  rng = np.random.RandomState(1)
  A_val = rng.randn(4, 6).astype(np.float32)
  B_val = rng.randn(6, 5).astype(np.float32)
  out = fn(A_val, B_val)
  np.testing.assert_allclose(out, A_val @ B_val, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(
  not metal_available(),
  reason="Metal runtime unavailable",
)
def test_metal_sum_axis_dispatch_matches_numpy() -> None:
  import numpy as np

  x = al.sym("x", (4, 6), dtype=al.dtypes.float32)
  fn = al.Function("sumax_metal", [x], [x.sum(axis=1)], ["x"], ["y"]).with_device("metal:0")
  x_val = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(4, 6)
  out = fn(x_val)
  np.testing.assert_allclose(out, x_val.sum(axis=1), atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(
  not metal_available(),
  reason="Metal runtime unavailable",
)
def test_metal_map_dispatch_matches_numpy() -> None:
  """MAP-shaped workloads (per-stage callee + outer loop) run on Metal via inline callee + per-thread invocation."""
  import numpy as np

  x = al.sym("x", 4, dtype=al.dtypes.float32)
  stage = al.Function("stage_map_metal", [x], [(x * x).sum()], ["x"], ["y"])
  batch = al.sym("batch", 32, dtype=al.dtypes.float32)
  mapped = al.map_(stage, length=8, inputs=[(batch, 0, 4)])
  fn = al.Function("mapped_dispatch", [batch], [mapped], ["batch"], ["m"]).with_device("metal:0")
  batch_val = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
  out = fn(batch_val)
  ref = (batch_val.reshape(8, 4) ** 2).sum(axis=1)
  np.testing.assert_allclose(out, ref, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(
  not metal_available(),
  reason="Metal runtime unavailable",
)
def test_metal_kernel_dispatch_handles_chained_elementwise() -> None:
  """Chained ops should fuse into one thread-bound loop in MSL (one element per thread)."""
  import numpy as np
  from alloy.metal_runtime import MetalRuntime

  x = al.sym("x", 32, dtype=al.dtypes.float32)
  fn = al.Function("chained_dispatch", [x], [x.sin() + x * x], ["x"], ["y"]).with_device("metal:0")
  msl = render_metal_source(fn)
  # Sanity: the chain should produce no inner serial FORs after fusion.
  assert "for (long i_" not in msl, f"expected fully-fused MSL, got:\n{msl}"
  metallib = compile_msl_to_metallib(msl, name="chained_dispatch")
  rt = MetalRuntime()
  x_val = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
  y_buf = np.zeros_like(x_val)
  outs = rt.run_kernel(metallib, "chained_dispatch_kernel", [x_val, y_buf], grid=32)
  ref = np.sin(x_val) + x_val * x_val
  np.testing.assert_allclose(outs[1], ref, atol=1e-6, rtol=1e-6)
