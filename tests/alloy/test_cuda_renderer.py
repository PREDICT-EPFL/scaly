"""Phase 8 first slice: CUDA source generation (no nvcc compile).

Tests pin the rendered text shape: kernel `__global__`, host driver does
``cudaMalloc`` / ``cudaMemcpy`` / launch / ``cudaDeviceSynchronize`` / copy
back / ``cudaFree``. A future Phase 8 follow-up will add nvcc integration,
range-to-thread binding, and numerical equivalence tests on actual hardware.
"""

from __future__ import annotations

import pytest

import alloy as al
from alloy.codegen.cuda import can_render_cuda, render_cuda_source


def test_can_render_cuda_only_for_cuda_device() -> None:
  x = al.sym("x", 4)
  host_fn = al.Function("f_host", [x], [x.sin()], ["x"], ["y"])
  cuda_fn = host_fn.with_device("cuda:0")
  assert not can_render_cuda(host_fn)
  assert can_render_cuda(cuda_fn)


def test_render_cuda_kernel_and_driver() -> None:
  x = al.sym("x", 8)
  fn = al.Function("f_cu", [x], [x.sin()], ["x"], ["y"]).with_device("cuda:0")
  src = render_cuda_source(fn)
  # kernel block
  assert "__global__ void f_cu_kernel(double* x, double* y)" in src
  assert "for (long long i_t0 = 0; i_t0 < 8;" in src
  assert "= sin(x[i_t0])" in src
  # host driver block
  assert 'extern "C" int f_cu(const double** arg, double** res' in src
  assert "cudaMalloc((void**)&d_in0" in src
  assert "cudaMemcpy(d_in0, arg[0]" in src
  assert "f_cu_kernel<<<dim3(8), dim3(1)>>>(d_in0, d_out0);" in src
  assert "cudaDeviceSynchronize()" in src
  assert "cudaMemcpy(res[0], d_out0" in src
  assert "cudaFree(d_in0);" in src


def test_render_cuda_supports_binary_elementwise() -> None:
  x = al.sym("x", 4)
  y = al.sym("y", 4)
  fn = al.Function("f_bin_cu", [x, y], [x * y + x], ["x", "y"], ["z"]).with_device("cuda:0")
  src = render_cuda_source(fn)
  assert "__global__ void f_bin_cu_kernel(double* x, double* y, double* z)" in src
  assert "(x[" in src and " * y[" in src


def test_render_cuda_rejects_host_function() -> None:
  x = al.sym("x", 4)
  fn = al.Function("f_host", [x], [x.sin()], ["x"], ["y"])
  with pytest.raises(ValueError, match="needs device='cuda:N'"):
    render_cuda_source(fn)


def test_cuda_source_includes_runtime_header_and_status_codes() -> None:
  x = al.sym("x", 2)
  fn = al.Function("f_status", [x], [-x], ["x"], ["y"]).with_device("cuda:0")
  src = render_cuda_source(fn)
  assert "#include <cuda_runtime.h>" in src
  assert "#define ALLOY_SUCCESS 0" in src
  assert "#define ALLOY_ERR_CUDA 5" in src
