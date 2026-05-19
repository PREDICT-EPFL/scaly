"""JIT pipeline: render -> compile -> cache -> dispatch through the universal ABI.

`Function.__call__` routes here by default; the tape interpreter remains the reference path
via `Function.eval_interpreter`. Environment variables:

- ``ALLOY_DISABLE_JIT=1`` skips JIT entirely and uses the interpreter.
- ``ALLOY_CACHE_DIR`` overrides the on-disk cache root (default: ``$XDG_CACHE_HOME/alloy/jit``
  or ``~/.cache/alloy/jit``).
- ``ALLOY_CC`` overrides the C compiler binary (default: ``cc`` from ``$PATH``).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .abi import C_API_SIGNATURE
from .codegen.c import _c_ident, render_c_source
from .codegen.solver_c import solver_compile_flags

if TYPE_CHECKING:
  from .function import Function


# Bump when the ABI, codegen output, or JIT cache layout changes incompatibly so
# that previously cached `.so` files are not reused by a newer Alloy version.
_JIT_CACHE_VERSION = "1"

_C_DOUBLE_P = ctypes.POINTER(ctypes.c_double)
_C_INT_P = ctypes.POINTER(ctypes.c_int)


class JitUnavailable(RuntimeError):
  """Raised when JIT compilation cannot proceed and the caller should fall back."""


class JitError(RuntimeError):
  """Raised when a compiled function returns a non-zero ABI status code."""


def jit_disabled() -> bool:
  """Return True iff ``ALLOY_DISABLE_JIT`` is set to a truthy value."""
  return os.environ.get("ALLOY_DISABLE_JIT", "0") not in ("0", "", "false", "False")


def _cache_root() -> Path:
  override = os.environ.get("ALLOY_CACHE_DIR")
  if override:
    return Path(override).expanduser()
  xdg = os.environ.get("XDG_CACHE_HOME")
  base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
  return base / "alloy" / "jit"


def _shared_lib_ext() -> str:
  return ".dylib" if sys.platform == "darwin" else ".so"


def _shared_lib_flag() -> str:
  return "-dynamiclib" if sys.platform == "darwin" else "-shared"


def _find_compiler() -> str | None:
  override = os.environ.get("ALLOY_CC")
  if override:
    return override if shutil.which(override) else None
  return shutil.which("cc")


def _compute_cache_key(source: str, *, fun_name: str) -> str:
  """SHA-256 over the rendered C source plus the cache-version and ABI signature.

  Any change to the codegen output, the ABI surface, or `_JIT_CACHE_VERSION` invalidates
  previously cached artifacts. Function names are included so two functions that happen
  to share a source skeleton (different symbols) still get distinct entries.
  """
  h = hashlib.sha256()
  h.update(_JIT_CACHE_VERSION.encode())
  h.update(b"\0")
  h.update(C_API_SIGNATURE.encode())
  h.update(b"\0")
  h.update(fun_name.encode())
  h.update(b"\0")
  h.update(source.encode())
  return h.hexdigest()


@dataclass(frozen=True, slots=True)
class _Artifact:
  lib_path: Path
  key: str


_artifact_cache: dict[str, _Artifact] = {}
_artifact_lock = threading.Lock()


def _build_artifact(fun: Function) -> _Artifact:
  """Render, compile (if needed), and return a path to ``fun``'s cached shared object.

  Raises ``JitUnavailable`` if there is no usable compiler or codegen does not support
  ``fun``; raises ``JitError`` if the compiler itself returns a non-zero status.
  """
  cc = _find_compiler()
  if cc is None:
    raise JitUnavailable("no C compiler found (set ALLOY_CC or install cc)")

  try:
    source = render_c_source(fun)
  except NotImplementedError as exc:
    raise JitUnavailable(f"codegen does not support function {fun.name!r}: {exc}") from exc

  key = _compute_cache_key(source, fun_name=fun.name)
  with _artifact_lock:
    cached = _artifact_cache.get(key)
  if cached is not None and cached.lib_path.exists():
    return cached

  symbol = _c_ident(fun.name)
  cache_dir = _cache_root() / key
  cache_dir.mkdir(parents=True, exist_ok=True)
  source_path = cache_dir / f"{symbol}.c"
  lib_path = cache_dir / f"lib{symbol}{_shared_lib_ext()}"

  if not lib_path.exists():
    # Write to a temp file then rename to avoid partially-written sources on concurrent builds.
    tmp_source = source_path.with_suffix(source_path.suffix + ".tmp")
    tmp_source.write_text(source)
    tmp_source.replace(source_path)
    extra_flags = solver_compile_flags(fun)
    cmd = [cc, "-O2", "-fPIC", _shared_lib_flag(), *extra_flags, str(source_path), "-lm", "-o", str(lib_path)]
    try:
      subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
      raise JitError(f"failed to compile {fun.name!r}: {exc.stderr or exc.stdout}") from exc

  artifact = _Artifact(lib_path=lib_path, key=key)
  with _artifact_lock:
    _artifact_cache[key] = artifact
  return artifact


class CompiledFunction:
  """Handle around a JIT-compiled `Function`.

  Holds the ``ctypes.CDLL`` for the cached shared object, the resolved entry point with
  ``argtypes``/``restype`` set up for the universal ABI, and the workspace size queried
  from the compiled ``<symbol>_sz_w`` helper.
  """

  __slots__ = (
    "_fun",
    "_artifact",
    "_lib",
    "_symbol",
    "_entry",
    "_sz_w",
    "_input_sizes",
    "_input_shapes",
    "_input_names",
    "_output_shapes",
    "_output_sizes",
    "_n_args",
    "_n_res",
  )

  def __init__(self, fun: Function):
    self._fun = fun
    self._artifact = _build_artifact(fun)
    self._lib = ctypes.CDLL(str(self._artifact.lib_path))
    symbol = _c_ident(fun.name)
    self._symbol = symbol
    entry = getattr(self._lib, symbol)
    entry.argtypes = [
      ctypes.POINTER(_C_DOUBLE_P),
      ctypes.POINTER(_C_DOUBLE_P),
      _C_INT_P,
      _C_DOUBLE_P,
      ctypes.c_void_p,
    ]
    entry.restype = ctypes.c_int
    self._entry = entry
    sz_w_fn = getattr(self._lib, f"{symbol}_sz_w")
    sz_w_fn.restype = ctypes.c_int
    self._sz_w = int(sz_w_fn())
    self._input_names = tuple(fun.input_names)
    self._input_shapes = tuple(e.shape for e in fun.inputs)
    self._input_sizes = tuple(int(e.size) for e in fun.inputs)
    self._output_shapes = tuple(e.shape for e in fun.outputs)
    self._output_sizes = tuple(int(e.size) for e in fun.outputs)
    self._n_args = len(fun.inputs)
    self._n_res = len(fun.outputs)

  @property
  def lib_path(self) -> Path:
    return self._artifact.lib_path

  @property
  def cache_key(self) -> str:
    return self._artifact.key

  def run(self, args: list[np.ndarray]) -> list[np.ndarray]:
    """Dispatch the compiled entry point with positional NumPy inputs.

    Inputs are converted to contiguous float64 buffers with shapes validated against the
    declared input shapes. Outputs are freshly-allocated NumPy arrays reshaped to the
    Function's declared output shapes (the compact buffer shape for sparse outputs).
    Raises ``JitError`` if the C function returns a non-zero ABI status.
    """
    if len(args) != self._n_args:
      raise TypeError(f"expected {self._n_args} inputs, got {len(args)}")

    arg_buffers: list[np.ndarray] = []
    arg_array = (_C_DOUBLE_P * max(self._n_args, 1))()
    for i, value in enumerate(args):
      name = self._input_names[i]
      expected_shape = self._input_shapes[i]
      arr = np.asarray(value, dtype=np.float64)
      if arr.shape != expected_shape:
        raise ValueError(f"input {name!r} has shape {arr.shape}, expected {expected_shape}")
      if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
      arg_buffers.append(arr)
      arg_array[i] = arr.ctypes.data_as(_C_DOUBLE_P)

    outputs: list[np.ndarray] = []
    res_array = (_C_DOUBLE_P * max(self._n_res, 1))()
    for i in range(self._n_res):
      out = np.empty(self._output_sizes[i], dtype=np.float64)
      outputs.append(out)
      res_array[i] = out.ctypes.data_as(_C_DOUBLE_P)

    if self._sz_w:
      w_buf = (ctypes.c_double * self._sz_w)()
      w_ptr = ctypes.cast(w_buf, _C_DOUBLE_P)
    else:
      w_buf = None  # noqa: F841 -- keep lifetime explicit even when unused
      w_ptr = _C_DOUBLE_P()

    status = self._entry(arg_array, res_array, _C_INT_P(), w_ptr, None)
    if status != 0:
      raise JitError(f"{self._fun.name} returned ABI status {status}")

    return [out.reshape(self._output_shapes[i]) for i, out in enumerate(outputs)]


def get_compiled(fun: Function) -> CompiledFunction:
  """Compile ``fun`` (or reuse a cached `.so`) and return a `CompiledFunction` handle."""
  return CompiledFunction(fun)


def invalidate_cache(fun: Function) -> None:
  """Drop both the in-memory artifact entry and the on-disk cache directory for ``fun``.

  Safe to call when nothing is cached yet; codegen failures (``NotImplementedError``) are
  swallowed since there cannot be a corresponding cache entry to remove.
  """
  try:
    source = render_c_source(fun)
  except NotImplementedError:
    return
  key = _compute_cache_key(source, fun_name=fun.name)
  with _artifact_lock:
    _artifact_cache.pop(key, None)
  cache_dir = _cache_root() / key
  if cache_dir.exists():
    shutil.rmtree(cache_dir, ignore_errors=True)
