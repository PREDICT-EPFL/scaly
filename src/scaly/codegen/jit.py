"""JIT: compile, cache, load and dispatch a ``codegen.aot.CModule`` through the universal ABI — no
rendering decisions of its own.

`Function.__call__` routes here by default. Environment variables:

- ``SCALY_CACHE_DIR`` overrides the on-disk cache root (default: ``$XDG_CACHE_HOME/scaly/jit``
  or ``~/.cache/scaly/jit``).
- ``SCALY_CC`` overrides the C compiler binary (default: ``cc`` from ``$PATH``). Its path and
  version are part of the cache key, so switching compilers never reuses the other's library.
- ``SCALY_CC_OPT`` overrides the optimization flag (default: ``-O2``). Benchmark harnesses that
  compile a baseline at ``-O3`` should set it, so both sides of a comparison get the same level.
  At ``-O2`` GCC before version 12 also gets ``-ftree-vectorize``: from 12 on GCC vectorizes at
  ``-O2`` by itself (with its cheapest cost model, which the flag would change), and clang always
  has.

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
from ..function.extern import ExternState, extern_functions
from .toolchain import cache_root, compiler_identity, find_c_compiler, gcc_major, is_gcc
from ..utils.env import shared_lib_ext, shared_lib_flag

if TYPE_CHECKING:
  from ..function import ConcreteFunction


# Bump when the ABI, codegen output, or JIT cache layout changes incompatibly so
# that previously cached `.so` files are not reused by a newer Scaly version.
_JIT_CACHE_VERSION = "4"

_LM_ID_NEWLM = -1
_RTLD_DI_LMID = 1
_SOLVER_NAMESPACE: int | None = None
_SOLVER_NAMESPACE_ANCHOR: ctypes.CDLL | None = None
_SOLVER_NAMESPACE_LOCK = threading.Lock()


def _load_library(path: Path, *, isolated: bool) -> ctypes.CDLL:
  """Keep the native dependencies of extern callees (solver libraries) out of the host process
  linker namespace on Linux."""
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


def vectorize_flags(opt: str, cc: str) -> tuple[str, ...]:
  """``-ftree-vectorize`` when GCC before version 12 compiles at ``-O2``, which it vectorizes only
  from 12 on; nothing otherwise, since the flag also replaces the cost model GCC 12 and later use
  at ``-O2``. A benchmark baseline compiled beside the JIT takes the same flags from here."""
  if opt != "-O2" or not is_gcc(cc):
    return ()
  major = gcc_major(cc)
  return ("-ftree-vectorize",) if major is not None and major < 12 else ()


def compile_flags() -> tuple[str, ...]:
  """Flags the JIT passes to every compile: the optimization level, ``vectorize_flags`` and
  ``HOST_CFLAGS``."""
  opt = opt_flag()
  compiler = find_c_compiler()
  return (opt, *(vectorize_flags(opt, compiler.cc) if compiler is not None else ()), *HOST_CFLAGS)


def _compute_cache_key(source: str, *, fun_name: str, compiler: tuple[str, ...], compile_flags: tuple[str, ...] = ()) -> str:
  """SHA-256 over the rendered C source plus the cache-version and ABI signature.

  Any change to the codegen output, the ABI surface, or `_JIT_CACHE_VERSION` invalidates
  previously cached artifacts. Function names and compile/link flags are included so two
  functions that happen to share a source skeleton (different symbols or solver rpaths) still
  get distinct entries. ``compiler`` is ``toolchain.compiler_identity``, since the flags alone do
  not tell clang from GCC 12 or later.
  """
  h = hashlib.sha256()
  h.update(_JIT_CACHE_VERSION.encode())
  h.update(b"\0")
  h.update(C_API_SIGNATURE.encode())
  h.update(b"\0")
  h.update(fun_name.encode())
  h.update(b"\0")
  h.update(source.encode())
  for part in (*compiler, *compile_flags):
    h.update(b"\0")
    h.update(part.encode())
  return h.hexdigest()


@dataclass(frozen=True, slots=True)
class _Artifact:
  lib_path: Path
  key: str
  flags: tuple[str, ...]
  workspace_size: int
  isolated: bool


_artifact_cache: dict[str, _Artifact] = {}
_artifact_lock = threading.Lock()


def _build_artifact(fun: ConcreteFunction) -> _Artifact:
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
  # The compiler and the compile flags are part of the key: changing either changes the machine
  # code built from the same source, so the two builds must not share a cache entry.
  flags = compile_flags()
  key = _compute_cache_key(module.body, fun_name=fun.name, compiler=compiler_identity(cc), compile_flags=(*flags, *extra_flags))
  with _artifact_lock:
    cached = _artifact_cache.get(key)
  if cached is not None and cached.lib_path.exists():
    return cached

  symbol = c_ident(fun.name)
  cache_dir = cache_root() / key
  source_path = cache_dir / f"{symbol}.c"
  lib_path = cache_dir / f"lib{symbol}{shared_lib_ext()}"
  # Another process's ``recompile()`` (or someone clearing the cache) can remove the directory while
  # this build writes into it; build again once rather than fail on the vanished temp file.
  for attempt in range(2):
    try:
      _compile_into(cache_dir, source_path, lib_path, module.body, [cc, *flags, "-fPIC", shared_lib_flag()], extra_flags, fun.name)
      break
    except (FileNotFoundError, JitError):
      if attempt or cache_dir.exists():
        raise

  artifact = _Artifact(lib_path=lib_path, key=key, flags=extra_flags, workspace_size=module.workspace_size, isolated=module.requirements.isolated)
  with _artifact_lock:
    _artifact_cache[key] = artifact
  return artifact


def _compile_into(cache_dir: Path, source_path: Path, lib_path: Path, body: str, cc_cmd: list[str], extra_flags: tuple[str, ...], name: str) -> None:
  """Write ``body`` and compile it to ``lib_path`` inside ``cache_dir``, unless the library is there."""
  cache_dir.mkdir(parents=True, exist_ok=True)
  if lib_path.exists():
    return
  # Write to a thread-unique temp file then rename, so concurrent builds neither read a
  # half-written source nor race each other on the rename.
  unique = f".{os.getpid()}.{threading.get_ident()}.tmp"  # per thread: two threads may build one function
  tmp_source = source_path.with_suffix(source_path.suffix + unique)
  tmp_source.write_text(body)
  tmp_source.replace(source_path)
  # Compile to a thread-unique temp lib then atomically rename, so concurrent builds of the same
  # function (e.g. pytest-xdist workers on a cold cache) never observe a half-written .so.
  tmp_lib = lib_path.with_suffix(lib_path.suffix + unique)
  # Link libraries (-l in extra_flags) MUST come after the source: ld defaults to --as-needed on
  # Linux, so a -lpiqpc/-lipopt placed before the object that references it is dropped (no
  # DT_NEEDED -> "undefined symbol" at dlopen of solver functions).
  cmd = [*cc_cmd, str(source_path), *extra_flags, "-lm", "-o", str(tmp_lib)]
  try:
    subprocess.run(cmd, check=True, capture_output=True, text=True)
  except subprocess.CalledProcessError as exc:
    tmp_lib.unlink(missing_ok=True)
    raise JitError(f"failed to compile {name!r}: {exc.stderr or exc.stdout}") from exc
  tmp_lib.replace(lib_path)


class CompiledFunction:
  """Handle around a JIT-compiled `Function`.

  Holds the ``ctypes.CDLL`` for the cached shared object, the resolved entry point with
  ``argtypes``/``restype`` set up for the pointer ABI, and the workspace size the rendered module
  reported (the header's ``SZ_W``; the library exports no size query of its own).
  """

  __slots__ = (
    "_name",
    "_artifact",
    "_lib",
    "_symbol",
    "_entry",
    "_state_entries",
    "_sz_w",
    "_input_sizes",
    "_input_shapes",
    "_input_names",
    "_output_shapes",
    "_output_sizes",
    "_output_bool",
    "_n_args",
    "_n_res",
    "_arg_array_type",
    "_res_array_type",
    "_workspaces",
  )

  def __init__(self, fun: ConcreteFunction):
    self._name = fun.name  # not the Function, which holds this handle: no cycle keeps the workspaces
    self._artifact = _build_artifact(fun)
    self._lib = _load_library(self._artifact.lib_path, isolated=self._artifact.isolated)
    symbol = c_ident(fun.name)
    self._symbol = symbol
    entry = getattr(self._lib, symbol)
    # Raw addresses: building typed pointer objects per argument cost more than the C call itself
    # for small functions, and numpy's ``data_as`` leaves a reference cycle per buffer for the
    # garbage collector.
    entry.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    entry.restype = ctypes.c_int
    self._entry = entry
    # The state each extern callee exposes (a solver's statistics), keyed by its C symbol.
    self._state_entries: dict[str, tuple[Any, ExternState]] = {}
    for callee in extern_functions(fun):
      state = callee.extern.state(callee) if callee.extern is not None else None
      if state is None:
        continue
      try:
        accessor = getattr(self._lib, state.accessor)
      except AttributeError as exc:
        raise JitError(f"compiled artifact is missing the state accessor {state.accessor}") from exc
      accessor.argtypes = [ctypes.POINTER(state.ctype)]
      accessor.restype = ctypes.c_int
      self._state_entries[c_ident(callee.name)] = (accessor, state)
    self._sz_w = self._artifact.workspace_size
    self._input_names = tuple(fun.input_names)
    self._input_shapes = tuple(e.shape for e in fun.inputs)
    self._input_sizes = tuple(int(e.size) for e in fun.inputs)
    self._output_shapes = tuple(e.shape for e in fun.outputs)
    self._output_sizes = tuple(int(e.size) for e in fun.outputs)
    self._output_bool = tuple(e.type.dtype.is_bool for e in fun.outputs)
    self._n_args = len(fun.inputs)
    self._n_res = len(fun.outputs)
    # Built once: a ctypes array type is a class, and making one per call leaves a cycle to collect.
    self._arg_array_type = ctypes.c_void_p * max(self._n_args, 1)
    self._res_array_type = ctypes.c_void_p * max(self._n_res, 1)
    self._workspaces = threading.local()

  @property
  def lib_path(self) -> Path:
    return self._artifact.lib_path

  @property
  def cache_key(self) -> str:
    return self._artifact.key

  def callee_state(self, name: str | None = None) -> Any:
    """The state the extern callee ``name`` exposes after the latest call (a solver's statistics);
    ``name`` may be left out when the library holds one. A decoder that finds the state invalid
    (never run, another layout) raises ``ValueError``, reported as a ``JitError``."""
    if name is None:
      if len(self._state_entries) != 1:
        raise JitError(f"a callee name is required when an artifact has {len(self._state_entries)} extern callees with state")
      name = next(iter(self._state_entries))
    symbol = c_ident(name)
    if symbol not in self._state_entries:
      raise JitError(f"no extern callee with state named {name!r}")
    accessor, state = self._state_entries[symbol]
    raw = state.ctype()
    status = accessor(ctypes.byref(raw))
    if status != 0:
      raise JitError(f"{state.accessor} returned status {status}")
    try:
      return state.decode(raw)
    except ValueError as exc:
      raise JitError(str(exc)) from exc

  def _workspace(self) -> np.ndarray | None:
    """This thread's workspace: allocated on its first call and reused, not zeroed. The kernel
    assumes nothing about its contents (it holds the buffers that would otherwise be uninitialized
    C locals), and a fresh allocation per call paid a page fault for every page of a large one. A
    NumPy buffer rather than ``ctypes.cast`` of a ctypes array, which would tie it into a
    reference cycle."""
    if not self._sz_w:
      return None
    buf = getattr(self._workspaces, "w", None)
    if buf is None:
      buf = self._workspaces.w = np.empty(self._sz_w, dtype=np.float64)
    return buf

  def run(self, args: list[np.ndarray]) -> list[np.ndarray]:
    """Dispatch the compiled entry point with positional NumPy inputs.

    Inputs are converted to contiguous float64 buffers with shapes validated against the
    declared input shapes. Outputs are freshly-allocated NumPy arrays reshaped to the
    Function's declared output shapes (the compact buffer shape for sparse outputs).
    Raises ``JitError`` if the C function returns a non-zero ABI status.
    """
    if len(args) != self._n_args:
      raise TypeError(f"expected {self._n_args} inputs, got {len(args)}")

    # ``arg_buffers`` and ``outputs`` keep the arrays alive until the C call returns: the pointer
    # arrays below hold raw addresses only.
    arg_buffers: list[np.ndarray] = []
    arg_array = self._arg_array_type()
    for i, value in enumerate(args):
      expected_shape = self._input_shapes[i]
      if type(value) is np.ndarray and value.dtype == _F64 and value.shape == expected_shape and value.flags.c_contiguous:
        arr = value  # already what the ABI takes
      else:
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != expected_shape:
          raise ValueError(f"input {self._input_names[i]!r} has shape {arr.shape}, expected {expected_shape}")
        if not arr.flags["C_CONTIGUOUS"]:
          arr = np.ascontiguousarray(arr)
      arg_buffers.append(arr)
      arg_array[i] = _address(arr)

    outputs: list[np.ndarray] = []
    res_array = self._res_array_type()
    for i in range(self._n_res):
      out = np.empty(self._output_sizes[i], dtype=np.float64)
      outputs.append(out)
      res_array[i] = _address(out)

    w_buf = self._workspace()
    status = self._entry(arg_array, res_array, None, None if w_buf is None else _address(w_buf), 0)
    if status != 0:
      raise JitError(f"{self._name} returned ABI status {status}")

    # A bool output crosses the ABI as 0.0 or 1.0 in its double array.
    return [(out != 0.0 if self._output_bool[i] else out).reshape(self._output_shapes[i]) for i, out in enumerate(outputs)]


_F64 = np.dtype(np.float64)


def _address(arr: np.ndarray) -> int:
  """The address of a contiguous array's data. ``arr.ctypes.data`` builds a ctypes helper object
  on each access, which costs about three times as much as reading the address through the buffer
  protocol; that path needs a writable, non-empty buffer, so anything else takes the slow one."""
  try:
    return ctypes.addressof(ctypes.c_char.from_buffer(arr))
  except (TypeError, ValueError, BufferError):
    return arr.ctypes.data


def get_compiled(fun: ConcreteFunction) -> CompiledFunction:
  """Compile ``fun`` (or reuse a cached `.so`) and return a `CompiledFunction` handle."""
  return CompiledFunction(fun)


def invalidate_cache(fun: ConcreteFunction) -> None:
  """Drop both the in-memory artifact entry and the on-disk cache directory for ``fun``.

  Safe to call when nothing is cached yet; codegen failures (``NotImplementedError``) are
  swallowed since there cannot be a corresponding cache entry to remove. Only the current
  compiler's entry goes; libraries other compilers built for ``fun`` stay.
  """
  compiler = find_c_compiler()
  if compiler is None:
    return  # nothing can have been built without one
  try:
    module = render_c_module(fun)
  except NotImplementedError:
    return
  key = _compute_cache_key(
    module.body, fun_name=fun.name, compiler=compiler_identity(compiler.cc), compile_flags=(*compile_flags(), *module.link_flags)
  )
  with _artifact_lock:
    _artifact_cache.pop(key, None)
  cache_dir = cache_root() / key
  if cache_dir.exists():
    shutil.rmtree(cache_dir, ignore_errors=True)
