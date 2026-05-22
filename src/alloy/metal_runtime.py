"""Ad-hoc Metal runtime: dispatch generated MSL kernels via ctypes + libobjc.

No PyObjC dependency — we go through ``ctypes.util.find_library`` to locate
``libobjc``, ``Metal``, ``Foundation`` and message Metal objects directly with
``objc_msgSend``. This mirrors tinygrad's approach and lets the test suite
exercise the full MSL render → compile → dispatch → readback round-trip on
Apple Silicon without a heavy framework dependency.

The flow:

1. ``compile_msl_to_metallib(msl)`` writes ``kernel.metal``, invokes
   ``xcrun -sdk macosx metal -c`` then ``xcrun -sdk macosx metallib`` to
   produce a ``.metallib``.
2. ``MetalRuntime`` lazily binds the libobjc + Metal entry points it needs.
3. ``MetalRuntime.run_kernel(metallib, name, buffers, grid)`` loads the
   metallib, builds a compute pipeline, allocates ``MTLBuffer``s for each
   numpy array, dispatches threads, syncs, and copies results back.

Limitations of this first slice:

- ``float32`` only (matches the ``metal`` ``BackendSupport`` capability set).
- 1D dispatch — multi-axis binding is a future schedule pass.
- One ``MTLDevice`` (the system default).
- Synchronous dispatch (no command queue reuse / overlap).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def _find(name: str) -> str:
  path = ctypes.util.find_library(name)
  if path is None:  # pragma: no cover - Apple platforms always have these
    raise RuntimeError(f"could not locate {name} framework/library")
  return path


def metal_available() -> bool:
  """Return True iff every piece the runtime needs is reachable on this machine."""
  if shutil.which("xcrun") is None:
    return False
  try:
    res = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "--version"], capture_output=True, text=True)
    if res.returncode != 0:
      return False
  except (FileNotFoundError, OSError):
    return False
  try:
    _find("objc")
    _find("Metal")
    _find("Foundation")
  except RuntimeError:
    return False
  return True


def compile_msl_to_metallib(msl_source: str, *, name: str = "kernel") -> Path:
  """Compile ``msl_source`` to a ``.metallib`` using ``xcrun metal`` + ``metallib``.

  Returns the path to the resulting ``.metallib`` inside a fresh tempdir. The
  caller owns the tempdir's lifetime (we leak it intentionally for the duration
  of the test; for production use, wrap in a ``TemporaryDirectory``).
  """
  td = Path(tempfile.mkdtemp(prefix="alloy_metal_"))
  metal_path = td / f"{name}.metal"
  air_path = td / f"{name}.air"
  lib_path = td / f"{name}.metallib"
  metal_path.write_text(msl_source)
  subprocess.run(
    ["xcrun", "-sdk", "macosx", "metal", "-c", str(metal_path), "-o", str(air_path)],
    check=True,
    capture_output=True,
    text=True,
  )
  subprocess.run(
    ["xcrun", "-sdk", "macosx", "metallib", str(air_path), "-o", str(lib_path)],
    check=True,
    capture_output=True,
    text=True,
  )
  return lib_path


class _Objc:
  """Thin wrapper around libobjc's ``objc_msgSend`` + selector/class lookup.

  Sending messages goes through:

      result = objc_msgSend(receiver, sel_registerName(b"name:"), arg1, ...)

  We re-prototype the function for each return/arg type combination we need.
  """

  def __init__(self) -> None:
    self.objc = ctypes.CDLL(_find("objc"))
    self.objc.sel_registerName.restype = ctypes.c_void_p
    self.objc.sel_registerName.argtypes = [ctypes.c_char_p]
    self.objc.objc_getClass.restype = ctypes.c_void_p
    self.objc.objc_getClass.argtypes = [ctypes.c_char_p]
    # objc_msgSend is variadic; we instantiate it per-signature via _msg.
    self._msgsend_addr = ctypes.cast(self.objc.objc_msgSend, ctypes.c_void_p).value

  def sel(self, name: bytes) -> int:
    return self.objc.sel_registerName(name)

  def cls(self, name: bytes) -> int:
    return self.objc.objc_getClass(name)

  def _msg(self, restype, argtypes):
    proto = ctypes.CFUNCTYPE(restype, *argtypes)
    return proto(self._msgsend_addr)  # ty: ignore[no-matching-overload]


class _MetalLibs:
  """Hold references to the Metal/Foundation framework images.

  Loading the framework dylibs registers the Metal Objective-C classes with the
  runtime so ``objc_getClass(b"MTLDevice")`` etc. resolve. We don't need any
  symbol from the dylibs directly — the C entry points we use
  (``MTLCreateSystemDefaultDevice``, ``NSStringFromString`` analogues) sit on
  the framework or on libobjc.
  """

  def __init__(self) -> None:
    self.metal = ctypes.CDLL(_find("Metal"))
    self.foundation = ctypes.CDLL(_find("Foundation"))
    self.metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    self.metal.MTLCreateSystemDefaultDevice.argtypes = []


def _nsstring(objc: _Objc, value: str) -> int:
  cls = objc.cls(b"NSString")
  alloc_sel = objc.sel(b"alloc")
  init_sel = objc.sel(b"initWithUTF8String:")
  alloc_fn = objc._msg(ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p])
  init_fn = objc._msg(ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p])
  s = alloc_fn(cls, alloc_sel)
  return init_fn(s, init_sel, value.encode("utf-8"))


def _nsurl(objc: _Objc, path: str) -> int:
  cls = objc.cls(b"NSURL")
  sel = objc.sel(b"fileURLWithPath:")
  fn = objc._msg(ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p])
  return fn(cls, sel, _nsstring(objc, path))


class MetalRuntime:
  """One-shot Metal runtime: open device, compile metallib, run a kernel."""

  def __init__(self) -> None:
    if not metal_available():
      raise RuntimeError("Metal runtime unavailable — install the Xcode Metal Toolchain")
    self.objc = _Objc()
    self.libs = _MetalLibs()
    self.device = self.libs.metal.MTLCreateSystemDefaultDevice()
    if not self.device:  # pragma: no cover - macOS Apple Silicon always has one
      raise RuntimeError("MTLCreateSystemDefaultDevice returned NULL")

  def run_kernel(
    self,
    metallib_path: Path,
    kernel_name: str,
    buffers: list[np.ndarray],
    *,
    grid: int,
    threadgroup: int = 64,
  ) -> list[np.ndarray]:
    """Load ``metallib_path``, dispatch ``kernel_name`` with ``grid`` threads.

    ``buffers`` is the kernel's argument list in declaration order. Numpy
    arrays are copied into ``MTLBuffer``s via ``newBufferWithBytes:length:options:``
    (storage mode shared so the GPU can see the host bytes directly) and copied
    back into newly-allocated ``np.ndarray``s of the same shape after sync.
    """
    objc = self.objc
    msg_ptr_ptr = objc._msg(ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p])
    msg_ptr_ptr_ptr = objc._msg(ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p])
    msg_void = objc._msg(None, [ctypes.c_void_p, ctypes.c_void_p])
    msg_void_void = objc._msg(None, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p])
    msg_buf_bytes = objc._msg(ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong])
    msg_set_buffer = objc._msg(None, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong])
    msg_contents = objc._msg(ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p])

    # Load the metallib.
    url = _nsurl(objc, str(metallib_path))
    library = msg_ptr_ptr_ptr(self.device, objc.sel(b"newLibraryWithURL:error:"), url, ctypes.c_void_p(0))
    if not library:
      raise RuntimeError(f"failed to load metallib at {metallib_path}")

    # Get the function.
    func_name = _nsstring(objc, kernel_name)
    fn = msg_ptr_ptr(library, objc.sel(b"newFunctionWithName:"), func_name)
    if not fn:
      raise RuntimeError(f"kernel {kernel_name!r} not found in {metallib_path}")

    # Build the compute pipeline state.
    pipeline = msg_ptr_ptr_ptr(
      self.device,
      objc.sel(b"newComputePipelineStateWithFunction:error:"),
      fn,
      ctypes.c_void_p(0),
    )
    if not pipeline:
      raise RuntimeError("failed to create compute pipeline")

    # Create MTLBuffers, host-shared storage (option = 0 = MTLResourceStorageModeShared).
    mtl_buffers: list[int] = []
    for arr in buffers:
      contig = np.ascontiguousarray(arr)
      mb = msg_buf_bytes(
        self.device,
        objc.sel(b"newBufferWithBytes:length:options:"),
        contig.ctypes.data,
        ctypes.c_size_t(contig.nbytes),
        ctypes.c_ulong(0),  # MTLResourceStorageModeShared
      )
      if not mb:
        raise RuntimeError("MTLDevice newBufferWithBytes returned NULL")
      mtl_buffers.append(mb)

    # Command queue + buffer + compute encoder.
    queue = msg_ptr_ptr(self.device, objc.sel(b"newCommandQueue"), ctypes.c_void_p(0))
    if not queue:
      raise RuntimeError("newCommandQueue returned NULL")
    cmd_buf = msg_ptr_ptr(queue, objc.sel(b"commandBuffer"), ctypes.c_void_p(0))
    encoder = msg_ptr_ptr(cmd_buf, objc.sel(b"computeCommandEncoder"), ctypes.c_void_p(0))
    msg_void_void(encoder, objc.sel(b"setComputePipelineState:"), pipeline)
    for i, mb in enumerate(mtl_buffers):
      msg_set_buffer(encoder, objc.sel(b"setBuffer:offset:atIndex:"), mb, 0, i)

    # Dispatch with grid as the *total* thread count (1D).
    # MTLSize is passed by value — three NSUInteger fields packed in a struct.
    class MTLSize(ctypes.Structure):
      _fields_ = [("width", ctypes.c_ulong), ("height", ctypes.c_ulong), ("depth", ctypes.c_ulong)]

    grid_size = MTLSize(grid, 1, 1)
    tg_size = MTLSize(min(threadgroup, grid), 1, 1)
    # dispatchThreads:threadsPerThreadgroup: takes MTLSize by value — we
    # re-prototype to accept the struct.
    dispatch_proto = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, MTLSize, MTLSize)
    dispatch_fn = dispatch_proto(self.objc._msgsend_addr)  # ty: ignore[no-matching-overload]
    dispatch_fn(encoder, objc.sel(b"dispatchThreads:threadsPerThreadgroup:"), grid_size, tg_size)
    msg_void(encoder, objc.sel(b"endEncoding"))
    msg_void(cmd_buf, objc.sel(b"commit"))
    msg_void(cmd_buf, objc.sel(b"waitUntilCompleted"))

    # Copy results back from each MTLBuffer.
    out: list[np.ndarray] = []
    for arr, mb in zip(buffers, mtl_buffers, strict=True):
      ptr = msg_contents(mb, objc.sel(b"contents"))
      n_bytes = arr.nbytes
      copy = (ctypes.c_uint8 * n_bytes).from_address(ptr)
      result = np.frombuffer(bytes(copy), dtype=arr.dtype).reshape(arr.shape)
      out.append(result)
    return out


class MetalCompiledFunction:
  """Lazily-built per-Function Metal kernel handle.

  Mirrors ``alloy.jit.CompiledFunction`` for ``device='metal:N'`` Functions.
  Caches the compiled metallib path + a shared ``MetalRuntime`` so repeated
  calls reuse the same pipeline state implicitly via ``newComputePipelineState``
  re-creation (cheap on hot paths today; a follow-up can memoize per-name).
  """

  _runtime: "MetalRuntime | None" = None

  def __init__(self, fun) -> None:  # ``fun`` is alloy.Function; avoid an import cycle.
    from .codegen.metal import render_metal_source

    msl = render_metal_source(fun)
    self.metallib = compile_msl_to_metallib(msl, name=fun.name)
    self.kernel_name = f"{fun.name}_kernel"
    self.fun = fun
    if MetalCompiledFunction._runtime is None:
      MetalCompiledFunction._runtime = MetalRuntime()
    self.runtime = MetalCompiledFunction._runtime

  def run(self, args: list[np.ndarray]) -> list[np.ndarray]:
    # Build numpy arrays for each kernel parameter (inputs + outputs).
    np_args: list[np.ndarray] = []
    for actual, expected in zip(args, self.fun.inputs, strict=True):
      arr = np.ascontiguousarray(actual, dtype=expected.type.dtype.numpy())
      if arr.shape != expected.type.shape:
        raise ValueError(f"input {expected.name!r} shape {arr.shape} != expected {expected.type.shape}")
      np_args.append(arr)
    out_buffers: list[np.ndarray] = []
    for out_expr in self.fun.outputs:
      buf = np.zeros(out_expr.type.shape or (1,), dtype=out_expr.type.dtype.numpy())
      out_buffers.append(buf)
    grid = max((int(np.prod(a.shape, dtype=int)) for a in np_args + out_buffers), default=1)
    # Sensible 1D launch: trip count of the largest IO buffer.
    raw = self.runtime.run_kernel(
      self.metallib,
      self.kernel_name,
      [*np_args, *out_buffers],
      grid=grid,
    )
    # Return only the output buffers.
    return [raw[len(np_args) + i] for i in range(len(out_buffers))]


__all__: list[str] = ["MetalCompiledFunction", "MetalRuntime", "compile_msl_to_metallib", "metal_available"]

# Silence unused-imports
_ = Any
