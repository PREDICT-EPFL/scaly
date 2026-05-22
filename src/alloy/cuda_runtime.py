"""CUDA runtime: dispatch generated CUDA C kernels via nvcc + ctypes.

The CUDA renderer emits a complete translation unit (kernel + ``extern "C"``
host driver implementing the universal Alloy ABI). The runtime's job:

1. Pick an nvcc binary compatible with the installed driver (some toolchains
   emit PTX that the driver rejects with CUDA error 222 ``unsupported
   toolchain``). Probes ``/usr/local/cuda-*/bin/nvcc`` in descending version
   order until a smoke-test kernel actually launches.
2. Compile the rendered CUDA C source to a shared object that exports the
   function symbol.
3. Open the ``.so`` via ``ctypes`` and call the entry point with the same
   ABI shape as the host C path (``const double**`` / ``double**`` / ``int*``
   / ``double*`` / ``void*``).

We deliberately do *not* link against ``libcuda``/``libcudart`` from Python:
the generated TU does its own ``cudaMalloc``/``cudaMemcpy``/launch/
``cudaDeviceSynchronize`` inside the host wrapper, so Python only sees a plain
shared-object symbol with the universal ABI. This keeps the runtime small and
mirrors the structure of ``codegen.c`` → ``jit.py``.

Caching: rendered source is hashed → ``<cache_root>/cuda/<hash>/lib<sym>.so``.
The shared object survives across processes; the ``CUDADevice`` singleton
holds the resolved nvcc path so the toolchain probe runs at most once per
process.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
  from .function import Function


_C_DOUBLE_P = ctypes.POINTER(ctypes.c_double)
_C_INT_P = ctypes.POINTER(ctypes.c_int)


class CudaUnavailable(RuntimeError):
  """nvcc is not present, or no available nvcc is compatible with the driver."""


def _cache_root() -> Path:
  override = os.environ.get("ALLOY_CACHE_DIR")
  if override:
    return Path(override).expanduser() / "cuda"
  xdg = os.environ.get("XDG_CACHE_HOME")
  base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
  return base / "alloy" / "cuda"


def _candidate_nvccs() -> list[Path]:
  """Return nvcc paths to try, newest first. Honors ``ALLOY_NVCC`` override."""
  override = os.environ.get("ALLOY_NVCC")
  if override:
    p = Path(override)
    return [p] if p.exists() else []
  found: list[Path] = []
  # Versioned dirs: /usr/local/cuda-13.2, /usr/local/cuda-13.0, …
  for p in sorted(Path("/usr/local").glob("cuda-*"), reverse=True):
    nvcc = p / "bin" / "nvcc"
    if nvcc.exists():
      found.append(nvcc)
  # Symlink fallback: /usr/local/cuda/bin/nvcc
  default = Path("/usr/local/cuda/bin/nvcc")
  if default.exists() and default.resolve() not in {f.resolve() for f in found}:
    found.append(default)
  # PATH fallback
  path_nvcc = shutil.which("nvcc")
  if path_nvcc:
    p = Path(path_nvcc)
    if p.resolve() not in {f.resolve() for f in found}:
      found.append(p)
  return found


_SMOKE_SRC = r"""
#include <cuda_runtime.h>
extern "C" __global__ void __alloy_smoke(int* out) { *out = 42; }
extern "C" int __alloy_run_smoke() {
  int* dev = nullptr;
  if (cudaMalloc((void**)&dev, sizeof(int)) != cudaSuccess) return 1;
  __alloy_smoke<<<1, 1>>>(dev);
  if (cudaGetLastError() != cudaSuccess) { cudaFree(dev); return 2; }
  if (cudaDeviceSynchronize() != cudaSuccess) { cudaFree(dev); return 3; }
  int host = 0;
  if (cudaMemcpy(&host, dev, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) { cudaFree(dev); return 4; }
  cudaFree(dev);
  return host == 42 ? 0 : 5;
}
"""


def _smoke_test(nvcc: Path) -> bool:
  """Compile a minimal kernel with ``nvcc`` and launch it. True iff the driver accepts the PTX."""
  with tempfile.TemporaryDirectory(prefix="alloy_cuda_smoke_") as td_str:
    td = Path(td_str)
    cu = td / "smoke.cu"
    cu.write_text(_SMOKE_SRC)
    so = td / "libsmoke.so"
    res = subprocess.run(
      [str(nvcc), "-O0", "--shared", "-Xcompiler", "-fPIC", str(cu), "-o", str(so)],
      capture_output=True,
      text=True,
    )
    if res.returncode != 0:
      return False
    try:
      lib = ctypes.CDLL(str(so))
    except OSError:
      return False
    fn = getattr(lib, "__alloy_run_smoke")
    fn.restype = ctypes.c_int
    return int(fn()) == 0


def cuda_available() -> bool:
  """Return True iff the host has a working nvcc + driver combination."""
  try:
    _CudaDevice.get()
    return True
  except CudaUnavailable:
    return False


@dataclass(frozen=True, slots=True)
class _CudaDevice:
  nvcc: Path

  @staticmethod
  def get() -> "_CudaDevice":
    return _resolve_device()


_device_lock = threading.Lock()
_device_cached: "_CudaDevice | None" = None


def _resolve_device() -> _CudaDevice:
  global _device_cached
  with _device_lock:
    if _device_cached is not None:
      return _device_cached
    candidates = _candidate_nvccs()
    if not candidates:
      raise CudaUnavailable("no nvcc found (set ALLOY_NVCC or install CUDA)")
    last_err: str | None = None
    for nvcc in candidates:
      if _smoke_test(nvcc):
        _device_cached = _CudaDevice(nvcc=nvcc)
        return _device_cached
      last_err = f"{nvcc} compiled but its PTX failed to launch on this driver"
    raise CudaUnavailable(f"no nvcc compatible with installed driver. Tried: {[str(p) for p in candidates]} ({last_err})")


def _shared_lib_ext() -> str:
  return ".dylib" if sys.platform == "darwin" else ".so"


def _compile_cuda(source: str, *, symbol: str) -> Path:
  """Compile ``source`` to a shared object via the resolved nvcc. Cached by source hash."""
  dev = _CudaDevice.get()
  key = hashlib.sha256(source.encode()).hexdigest()[:32]
  cache_dir = _cache_root() / key
  cache_dir.mkdir(parents=True, exist_ok=True)
  cu_path = cache_dir / f"{symbol}.cu"
  so_path = cache_dir / f"lib{symbol}{_shared_lib_ext()}"
  if so_path.exists():
    return so_path
  tmp = cu_path.with_suffix(cu_path.suffix + ".tmp")
  tmp.write_text(source)
  tmp.replace(cu_path)
  cmd = [str(dev.nvcc), "-O2", "--shared", "-Xcompiler", "-fPIC", str(cu_path), "-o", str(so_path)]
  res = subprocess.run(cmd, capture_output=True, text=True)
  if res.returncode != 0:
    raise CudaUnavailable(f"nvcc failed for {symbol!r}: {res.stderr.strip()}")
  return so_path


class CudaCompiledFunction:
  """Mirror of ``alloy.jit.CompiledFunction`` for ``device='cuda:N'`` Functions."""

  __slots__ = (
    "fun",
    "so_path",
    "_lib",
    "_entry",
    "_input_shapes",
    "_input_sizes",
    "_output_shapes",
    "_output_sizes",
  )

  def __init__(self, fun: "Function") -> None:
    if fun.device.kind != "cuda":
      raise ValueError(f"CudaCompiledFunction needs cuda device, got {fun.device}")
    from .codegen.cuda import render_cuda_source

    source = render_cuda_source(fun)
    self.so_path = _compile_cuda(source, symbol=fun.name)
    self._lib = ctypes.CDLL(str(self.so_path))
    self._entry = getattr(self._lib, fun.name)
    self._entry.argtypes = [
      ctypes.POINTER(_C_DOUBLE_P),
      ctypes.POINTER(_C_DOUBLE_P),
      _C_INT_P,
      _C_DOUBLE_P,
      ctypes.c_void_p,
    ]
    self._entry.restype = ctypes.c_int
    self.fun = fun
    self._input_shapes = tuple(e.shape for e in fun.inputs)
    self._input_sizes = tuple(int(e.size) for e in fun.inputs)
    self._output_shapes = tuple(e.shape for e in fun.outputs)
    self._output_sizes = tuple(int(e.size) for e in fun.outputs)

  def run(self, args: list[np.ndarray]) -> list[np.ndarray]:
    from .jit import JitError

    if len(args) != len(self.fun.inputs):
      raise TypeError(f"expected {len(self.fun.inputs)} inputs, got {len(args)}")
    arg_array = (_C_DOUBLE_P * max(len(args), 1))()
    arg_buffers: list[np.ndarray] = []
    for i, value in enumerate(args):
      expected = self._input_shapes[i]
      arr = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
      if arr.shape != expected:
        raise ValueError(f"input {i} shape {arr.shape} != expected {expected}")
      arg_buffers.append(arr)
      arg_array[i] = arr.ctypes.data_as(_C_DOUBLE_P)
    outputs: list[np.ndarray] = []
    res_array = (_C_DOUBLE_P * max(len(self.fun.outputs), 1))()
    for i in range(len(self.fun.outputs)):
      size = self._output_sizes[i] or 1
      out = np.zeros(size, dtype=np.float64)
      outputs.append(out)
      res_array[i] = out.ctypes.data_as(_C_DOUBLE_P)
    status = self._entry(arg_array, res_array, _C_INT_P(), _C_DOUBLE_P(), None)
    if status != 0:
      raise JitError(f"{self.fun.name} returned ABI status {status}")
    return [out.reshape(self._output_shapes[i] or (1,)) for i, out in enumerate(outputs)]


__all__ = ["CudaCompiledFunction", "CudaUnavailable", "cuda_available"]
