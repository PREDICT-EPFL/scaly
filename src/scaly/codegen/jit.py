"""JIT: compile, cache, load and dispatch a ``codegen.aot.CModule`` through the universal ABI — no
rendering decisions of its own.

`Function.__call__` routes here by default. Environment variables:

- ``SCALY_CACHE_DIR`` overrides the on-disk cache root (default: ``$XDG_CACHE_HOME/scaly/jit``
  or ``~/.cache/scaly/jit``).
- ``SCALY_CC`` overrides the C compiler binary (default: ``cc`` from ``$PATH``).
- ``SCALY_CC_OPT`` overrides the optimization flag (default: ``-O2``). Benchmark harnesses that
  compile a baseline at ``-O3`` should set it, so both sides of a comparison get the same level.

The JIT compiles for the machine it runs on, so it also passes the host CPU target and
``-fno-math-errno``; the distributed solver plugin wheels stay at the portable baseline.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .abi import C_API_SIGNATURE, c_ident
from .aot import render_c_module
from .solver import solver_stats_symbols
from ..solvers.stats import SCALY_SOLVER_STATS_VERSION, CSolverStats, SolverStats
from .toolchain import cache_root, find_c_compiler
from ..utils.env import shared_lib_ext, shared_lib_flag

if TYPE_CHECKING:
  from ..function import Function


# Bump when the ABI, codegen output, or JIT cache layout changes incompatibly so
# that previously cached `.so` files are not reused by a newer Scaly version.
_JIT_CACHE_VERSION = "4"

_C_DOUBLE_P = ctypes.POINTER(ctypes.c_double)
_C_INT_P = ctypes.POINTER(ctypes.c_int)
_LM_ID_NEWLM = -1
_RTLD_DI_LMID = 1
_SOLVER_NAMESPACE: int | None = None
_SOLVER_NAMESPACE_ANCHOR: ctypes.CDLL | None = None
_SOLVER_NAMESPACE_LOCK = threading.Lock()


def _load_library(path: Path, *, isolated: bool) -> ctypes.CDLL:
  """Keep Linux solver dependencies out of the host process linker namespace."""
  global _SOLVER_NAMESPACE, _SOLVER_NAMESPACE_ANCHOR
  if not isolated or sys.platform != "linux":
    return ctypes.CDLL(str(path))
  libc = ctypes.CDLL(None)
  dlmopen = libc.dlmopen
  dlmopen.argtypes = [ctypes.c_long, ctypes.c_char_p, ctypes.c_int]
  dlmopen.restype = ctypes.c_void_p
  with _SOLVER_NAMESPACE_LOCK:
    handle = dlmopen(_LM_ID_NEWLM if _SOLVER_NAMESPACE is None else _SOLVER_NAMESPACE, os.fsencode(path), os.RTLD_NOW | os.RTLD_LOCAL)
    if not handle:
      raise OSError(f"dlmopen failed for {path}")
    # Python 3.14 reopens a path even when CDLL receives a handle, so bind it directly.
    lib = ctypes.CDLL(None)
    lib._handle = handle
    lib._name = str(path)
    if _SOLVER_NAMESPACE is None:
      dlinfo = libc.dlinfo
      dlinfo.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
      dlinfo.restype = ctypes.c_int
      namespace = ctypes.c_long()
      if dlinfo(handle, _RTLD_DI_LMID, ctypes.byref(namespace)) != 0:
        raise OSError(f"dlinfo failed for {path}")
      _SOLVER_NAMESPACE = namespace.value
      _SOLVER_NAMESPACE_ANCHOR = lib
  return lib


class JitUnavailable(RuntimeError):
  """Raised when JIT compilation cannot proceed and the caller should fall back."""


class JitError(RuntimeError):
  """Raised when a compiled function returns a non-zero ABI status code."""


def _find_compiler() -> str | None:
  """Compatibility wrapper for tests; use ``scaly.codegen.toolchain.find_c_compiler`` in new code."""
  compiler = find_c_compiler()
  return compiler.cc if compiler is not None else None


def opt_flag() -> str:
  """Optimization flag for JIT compilation. ``-O2`` unless ``SCALY_CC_OPT`` says otherwise."""
  return os.environ.get("SCALY_CC_OPT") or "-O2"


# gcc and clang both spell the host target ``-march=native`` on x86. On AArch64 (Apple silicon,
# Linux arm64) clang rejects ``-march=native``; both compilers accept ``-mcpu=native`` there.
_NATIVE_CPU_FLAG = "-mcpu=native" if platform.machine().lower() in {"arm64", "aarch64"} else "-march=native"
HOST_CFLAGS: tuple[str, ...] = (_NATIVE_CPU_FLAG, "-fno-math-errno")
"""The host CPU target (so FMA and wider vectors are available) and ``-fno-math-errno`` (so ``sqrt``
and friends inline). The benchmark harness compiles both providers with the same two flags."""


def compile_flags() -> tuple[str, ...]:
  """Flags the JIT passes to every compile: the optimization level plus ``HOST_CFLAGS``."""
  return (opt_flag(), *HOST_CFLAGS)


def _compute_cache_key(source: str, *, fun_name: str, compile_flags: tuple[str, ...] = ()) -> str:
  """SHA-256 over the rendered C source plus the cache-version and ABI signature.

  Any change to the codegen output, the ABI surface, or `_JIT_CACHE_VERSION` invalidates
  previously cached artifacts. Function names and compile/link flags are included so two
  functions that happen to share a source skeleton (different symbols or solver rpaths) still
  get distinct entries.
  """
  h = hashlib.sha256()
  h.update(_JIT_CACHE_VERSION.encode())
  h.update(b"\0")
  h.update(C_API_SIGNATURE.encode())
  h.update(b"\0")
  h.update(fun_name.encode())
  h.update(b"\0")
  h.update(source.encode())
  for flag in compile_flags:
    h.update(b"\0")
    h.update(flag.encode())
  return h.hexdigest()


@dataclass(frozen=True, slots=True)
class _Artifact:
  lib_path: Path
  key: str
  flags: tuple[str, ...]
  workspace_size: int


_artifact_cache: dict[str, _Artifact] = {}
_artifact_lock = threading.Lock()


def _build_artifact(fun: Function) -> _Artifact:
  """Render, compile (if needed), and return a path to ``fun``'s cached shared object.

  Raises ``JitUnavailable`` if there is no usable compiler or codegen does not support
  ``fun``; callers let it propagate (there is no interpreter fallback). Raises ``JitError``
  if the compiler itself returns a non-zero status.
  """
  compiler = find_c_compiler()
  if compiler is None:
    raise JitUnavailable("no C compiler found (set SCALY_CC or install cc)")
  cc = compiler.cc

  try:
    module = render_c_module(fun)
  except NotImplementedError as exc:
    raise JitUnavailable(f"codegen does not support function {fun.name!r}: {exc}") from exc

  extra_flags = module.link_flags
  # The cache compiles ``body`` — the translation unit without the header include a written-out
  # ``.c`` carries — so the key is a hash of exactly the text handed to the compiler.
  # The compile flags are part of the key: two flag sets produce different machine code from the
  # same source, so they must not share a cache entry.
  flags = compile_flags()
  key = _compute_cache_key(module.body, fun_name=fun.name, compile_flags=(*flags, *extra_flags))
  with _artifact_lock:
    cached = _artifact_cache.get(key)
  if cached is not None and cached.lib_path.exists():
    return cached

  symbol = c_ident(fun.name)
  cache_dir = cache_root() / key
  cache_dir.mkdir(parents=True, exist_ok=True)
  source_path = cache_dir / f"{symbol}.c"
  lib_path = cache_dir / f"lib{symbol}{shared_lib_ext()}"

  if not lib_path.exists():
    # Write to a process-unique temp file then rename, so concurrent builds neither read a
    # half-written source nor race each other on the rename.
    tmp_source = source_path.with_suffix(source_path.suffix + f".{os.getpid()}.tmp")
    tmp_source.write_text(module.body)
    tmp_source.replace(source_path)
    # Compile to a process-unique temp lib then atomically rename, so concurrent builds of the same
    # function (e.g. pytest-xdist workers on a cold cache) never observe a half-written .so.
    tmp_lib = lib_path.with_suffix(lib_path.suffix + f".{os.getpid()}.tmp")
    # Link libraries (-l in extra_flags) MUST come after the source: ld defaults to --as-needed on
    # Linux, so a -lpiqpc/-lipopt placed before the object that references it is dropped (no
    # DT_NEEDED -> "undefined symbol" at dlopen of solver functions).
    cmd = [cc, *flags, "-fPIC", shared_lib_flag(), str(source_path), *extra_flags, "-lm", "-o", str(tmp_lib)]
    try:
      subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
      tmp_lib.unlink(missing_ok=True)
      raise JitError(f"failed to compile {fun.name!r}: {exc.stderr or exc.stdout}") from exc
    tmp_lib.replace(lib_path)

  artifact = _Artifact(lib_path=lib_path, key=key, flags=extra_flags, workspace_size=module.workspace_size)
  with _artifact_lock:
    _artifact_cache[key] = artifact
  return artifact


class CompiledFunction:
  """Handle around a JIT-compiled `Function`.

  Holds the ``ctypes.CDLL`` for the cached shared object, the resolved entry point with
  ``argtypes``/``restype`` set up for the pointer ABI, and the workspace size the rendered module
  reported (the header's ``SZ_W``; the library exports no size query of its own).
  """

  __slots__ = (
    "_fun",
    "_artifact",
    "_lib",
    "_symbol",
    "_entry",
    "_stats_entries",
    "_sz_w",
    "_input_sizes",
    "_input_shapes",
    "_input_names",
    "_output_shapes",
    "_output_sizes",
    "_output_bool",
    "_n_args",
    "_n_res",
  )

  def __init__(self, fun: Function):
    self._fun = fun
    self._artifact = _build_artifact(fun)
    self._lib = _load_library(self._artifact.lib_path, isolated=bool(self._artifact.flags))
    symbol = c_ident(fun.name)
    self._symbol = symbol
    entry = getattr(self._lib, symbol)
    entry.argtypes = [
      ctypes.POINTER(_C_DOUBLE_P),
      ctypes.POINTER(_C_DOUBLE_P),
      _C_INT_P,
      _C_DOUBLE_P,
      ctypes.c_int,
    ]
    entry.restype = ctypes.c_int
    self._entry = entry
    self._stats_entries: dict[str, Any] = {}
    for stats_symbol in solver_stats_symbols(fun):
      try:
        stats_entry = getattr(self._lib, f"{stats_symbol}_stats")
      except AttributeError as exc:
        raise JitError(f"compiled artifact is missing solver stats symbol {stats_symbol}_stats") from exc
      stats_entry.argtypes = [ctypes.POINTER(CSolverStats)]
      stats_entry.restype = ctypes.c_int
      self._stats_entries[stats_symbol] = stats_entry
    self._sz_w = self._artifact.workspace_size
    self._input_names = tuple(fun.input_names)
    self._input_shapes = tuple(e.shape for e in fun.inputs)
    self._input_sizes = tuple(int(e.size) for e in fun.inputs)
    self._output_shapes = tuple(e.shape for e in fun.outputs)
    self._output_sizes = tuple(int(e.size) for e in fun.outputs)
    self._output_bool = tuple(e.type.dtype.is_bool for e in fun.outputs)
    self._n_args = len(fun.inputs)
    self._n_res = len(fun.outputs)

  @property
  def lib_path(self) -> Path:
    return self._artifact.lib_path

  @property
  def cache_key(self) -> str:
    return self._artifact.key

  def solver_stats(self, name: str | None = None) -> SolverStats:
    if name is None:
      if len(self._stats_entries) != 1:
        raise JitError(f"solver name is required when an artifact has {len(self._stats_entries)} solver stats entries")
      name = next(iter(self._stats_entries))
    symbol = c_ident(name)
    if symbol not in self._stats_entries:
      raise JitError(f"no solver stats entry for {name!r}")
    raw = CSolverStats()
    status = self._stats_entries[symbol](ctypes.byref(raw))
    if status != 0:
      raise JitError(f"{symbol}_stats returned status {status}")
    if raw.version == 0:
      raise JitError(f"solver {name!r} has not run yet (stats version is 0)")
    if raw.version != SCALY_SOLVER_STATS_VERSION:
      raise JitError(f"solver stats ABI mismatch for {name!r}: artifact version {raw.version}, expected {SCALY_SOLVER_STATS_VERSION}")
    return SolverStats.from_c(raw)

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

    status = self._entry(arg_array, res_array, _C_INT_P(), w_ptr, 0)
    if status != 0:
      raise JitError(f"{self._fun.name} returned ABI status {status}")

    # A bool output crosses the ABI as 0.0 or 1.0 in its double array.
    return [(out != 0.0 if self._output_bool[i] else out).reshape(self._output_shapes[i]) for i, out in enumerate(outputs)]


def get_compiled(fun: Function) -> CompiledFunction:
  """Compile ``fun`` (or reuse a cached `.so`) and return a `CompiledFunction` handle."""
  return CompiledFunction(fun)


def invalidate_cache(fun: Function) -> None:
  """Drop both the in-memory artifact entry and the on-disk cache directory for ``fun``.

  Safe to call when nothing is cached yet; codegen failures (``NotImplementedError``) are
  swallowed since there cannot be a corresponding cache entry to remove.
  """
  try:
    module = render_c_module(fun)
  except NotImplementedError:
    return
  key = _compute_cache_key(module.body, fun_name=fun.name, compile_flags=(*compile_flags(), *module.link_flags))
  with _artifact_lock:
    _artifact_cache.pop(key, None)
  cache_dir = cache_root() / key
  if cache_dir.exists():
    shutil.rmtree(cache_dir, ignore_errors=True)
