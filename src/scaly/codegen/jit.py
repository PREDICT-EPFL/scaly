"""JIT: compile, cache, load and dispatch a ``codegen.aot.CModule`` through the universal ABI — no
rendering decisions of its own.

`ConcreteFunction.__call__` routes here by default. Environment variables:

- ``SCALY_CACHE_DIR`` overrides the on-disk cache root (default: ``$XDG_CACHE_HOME/scaly/jit``
  or ``~/.cache/scaly/jit``).
- ``SCALY_CC`` overrides the C compiler command (default: ``zig cc`` from the ``zig-bin`` package,
  then ``CC``, then ``cc`` from ``$PATH``).
- ``SCALY_CC_OPT`` overrides the optimization flag (default: ``-O2``). Benchmark harnesses that
  compile a baseline at ``-O3`` should set it, so both sides of a comparison get the same level.
- ``SCALY_VECTOR_LIBM`` selects ``none`` or ``glibc``; unset uses the native recipe.
- ``SCALY_JIT_KEY=verify`` renders a Function even when the index names its library, and raises
  when that library was built from other C, workspace size or link flags than the render gives.

A warm process finds a library without lowering: an index in the cache maps a key over the
Function's graph (``function/digest.py``), the installed source and the toolchain to the library
built for it. A Function the digest cannot encode is rendered to find its library.

The JIT compiles for the machine it runs on, so it also passes the host CPU target and
``-fno-math-errno``; the distributed solver plugin wheels stay at the portable baseline.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import struct
import tempfile
import sys
import threading
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .abi import C_API_SIGNATURE, c_ident
from .external import LibraryBuild, compile_library, libraries_of, prepare_library
from .aot import _RENDER_OBSERVERS, CModule, render_c_module
from .solver import solver_stats_symbols, solver_functions
from ..ir.layout import DenseLayout, SparseLayout
from ..passes import program
from ..passes.program._common import prints
from ..solvers.model import CSolverOption
from ..solvers.graph import solver_backends_used
from ..solvers.paths import SolverLibraryError, backend_compile_flags, solver_paths
from ..solvers.stats import SCALY_SOLVER_STATS_VERSION, CSolverStats, SolverStats
from .toolchain import (
  BuildRecipe,
  Compiler,
  cache_root,
  directory_fingerprint,
  compiler_fingerprint,
  find_c_compiler,
  native_recipe,
  no_gather_flags,
  vector_libm_flags,
)
from ..utils.env import ToolchainError, env, shared_lib_ext, shared_lib_flag

if TYPE_CHECKING:
  from ..function.concrete import ConcreteFunction
from ..function.digest import graph_digest, source_digest
from ..function.model import Function, as_concrete


# Bump when the ABI, codegen output, or JIT cache layout changes incompatibly.
# Solver modules also key on the contents of their include and library search
# directories, each read once per process. Keep native files fixed while Python
# runs, and stop it before replacing solver headers or libraries at the same paths.
_JIT_CACHE_VERSION = "8"

_C_DOUBLE_P = ctypes.POINTER(ctypes.c_double)
_C_INT_P = ctypes.POINTER(ctypes.c_int)
_LM_ID_NEWLM = -1
_RTLD_DI_LMID = 1


@dataclass
class _LinkerNamespace:
  id: int
  external: bool
  libraries: list[ctypes.CDLL]
  dependencies: dict[str, str]
  environ: ctypes.c_void_p | None = None


_LINKER_NAMESPACES: list[_LinkerNamespace] = []
_LINKER_NAMESPACE_LOCK = threading.Lock()


def _elf_identity(path: Path) -> tuple[str | None, str]:
  data = path.read_bytes()
  endian = "<" if data[5] == 1 else ">"
  wide = data[4] == 2
  header = struct.unpack_from(endian + ("16sHHIQQQIHHHHHH" if wide else "16sHHIIIIIHHHHHH"), data)
  segments = []
  dynamic = {}
  for i in range(header[10]):
    segment = struct.unpack_from(endian + ("IIQQQQQQ" if wide else "IIIIIIII"), data, header[5] + i * header[9])
    kind = segment[0]
    offset, address, size = (segment[2], segment[3], segment[5]) if wide else (segment[1], segment[2], segment[4])
    if kind == 1:  # PT_LOAD translates the dynamic string table's virtual address to a file offset.
      segments.append((offset, address, size))
    if kind == 2:  # PT_DYNAMIC, including stripped libraries without section headers.
      step = 16 if wide else 8
      for pos in range(offset, offset + size, step):
        tag, value = struct.unpack_from(endian + ("qQ" if wide else "iI"), data, pos)
        if tag == 0:
          break
        dynamic[tag] = value
  soname = None
  if 14 in dynamic:  # DT_SONAME, relative to DT_STRTAB.
    strings = dynamic[5]
    offset, address, _ = next(segment for segment in segments if segment[1] <= strings < segment[1] + segment[2])
    start = offset + strings - address + dynamic[14]
    soname = os.fsdecode(data[start : data.index(b"\0", start)])
  return soname, hashlib.sha256(data).hexdigest()


def _namespace_dependencies(path: Path) -> dict[str, str]:
  # ldd asks the platform loader to resolve the full closure without running initializers.
  result = subprocess.run(["ldd", str(path)], capture_output=True, text=True, env={**os.environ, "LC_ALL": "C"})
  if result.returncode and "statically linked" not in result.stdout:
    raise OSError(f"cannot inspect dependencies of {path}: {result.stderr or result.stdout}")
  paths = [(str(path), path)]
  for line in result.stdout.splitlines():
    name, _, resolved = line.strip().rpartition(" => ")
    resolved = resolved.rsplit(" (", 1)[0].strip()
    if resolved.startswith("/"):
      paths.append((name or resolved, Path(resolved)))
  dependencies = {}
  for requested, library in paths:
    soname, identity = _elf_identity(library.resolve())
    for name in (requested, str(library), str(library.resolve()), *((soname,) if soname else ())):
      if name in dependencies and dependencies[name] != identity:
        raise OSError(f"conflicting dependencies for {name} in {path}")
      dependencies[name] = identity
  return dependencies


def load_library(path: Path, *, isolated: bool, namespace: tuple[str, ...] = ()) -> ctypes.CDLL:
  """Load a shared library, isolating incompatible Linux dependency builds from one another.

  External kernels reuse a namespace when all shared paths and SONAMEs have identical content,
  including transitive dependencies. The solver namespace stays separate. Loaded handles and
  namespaces live until process exit, so old compiled handles remain callable after a rebuild.
  Environment changes after the last load remain subject to the limitation tracked in #159.
  """
  if isolated and sys.platform == "win32":
    with ExitStack() as stack:
      for directory in solver_paths(required=False).lib_dirs:
        stack.enter_context(os.add_dll_directory(str(directory.resolve())))
      return ctypes.CDLL(str(path))
  if not isolated or sys.platform != "linux":
    return ctypes.CDLL(str(path))
  libc = ctypes.CDLL(None)
  dlmopen = libc.dlmopen
  dlmopen.argtypes = [ctypes.c_long, ctypes.c_char_p, ctypes.c_int]
  dlmopen.restype = ctypes.c_void_p
  dlerror = libc.dlerror
  dlerror.argtypes = []
  dlerror.restype = ctypes.c_char_p
  with _LINKER_NAMESPACE_LOCK:
    dependencies = _namespace_dependencies(path) if namespace else {}
    previous = next(
      (
        candidate
        for candidate in _LINKER_NAMESPACES
        if candidate.external == bool(namespace)
        and all(name not in candidate.dependencies or candidate.dependencies[name] == identity for name, identity in dependencies.items())
      ),
      None,
    )
    namespace_id = previous.id if previous is not None else _LM_ID_NEWLM
    environ = previous.environ if previous is not None else None
    # A namespace's libc retains the process environment array from its first load. Refresh it
    # before another initializer calls getenv, since Python may have replaced that array.
    if environ is not None:
      environ.value = ctypes.c_void_p.in_dll(libc, "environ").value
    dlerror()
    # Prefer this kernel's own dependencies over the namespace anchor's exported symbols.
    mode = os.RTLD_NOW | os.RTLD_LOCAL | (os.RTLD_DEEPBIND if namespace else 0)
    handle = dlmopen(namespace_id, os.fsencode(path), mode)
    if not handle:
      error = dlerror()
      reason = error.decode(errors="replace") if error else "unknown loader error"
      advice = (
        ". Use a fresh process with fewer incompatible dependency versions." if "no more namespaces" in reason or "static TLS" in reason else ""
      )
      raise OSError(f"dlmopen failed for {path}: {reason}{advice}")
    # Python 3.14 reopens a path even when CDLL receives a handle, so bind it directly.
    lib = ctypes.CDLL(None)
    lib._handle = handle
    lib._name = str(path)
    if previous is None:
      dlinfo = libc.dlinfo
      dlinfo.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
      dlinfo.restype = ctypes.c_int
      resolved = ctypes.c_long()
      if dlinfo(handle, _RTLD_DI_LMID, ctypes.byref(resolved)) != 0:
        raise OSError(f"dlinfo failed for {path}")
      previous = _LinkerNamespace(resolved.value, bool(namespace), [], {})
      _LINKER_NAMESPACES.append(previous)
    if environ is None:
      try:
        environ = ctypes.c_void_p.in_dll(lib, "environ")
      except ValueError:
        pass  # A self-contained external kernel need not link libc.
    previous.libraries.append(lib)
    previous.dependencies.update(dependencies)
    previous.environ = environ
  return lib


@functools.cache
def _process_libc() -> ctypes.CDLL:
  return ctypes.CDLL("ucrtbase.dll" if sys.platform == "win32" else None)


class JitUnavailable(RuntimeError):
  """Raised when JIT compilation cannot proceed and the caller should fall back."""


class JitError(RuntimeError):
  """Raised when a compiled function returns a non-zero ABI status code."""


def opt_flag() -> str:
  """Optimization flag for JIT compilation. ``-O2`` unless ``SCALY_CC_OPT`` says otherwise."""
  return os.environ.get("SCALY_CC_OPT") or "-O2"


HOST_CFLAGS: tuple[str, ...] = (*BuildRecipe(cpu="native").cpu_flags, "-fno-math-errno")
"""Host CPU target and math flags shared by both benchmark providers."""


def compile_flags(recipe: BuildRecipe | None = None) -> tuple[str, ...]:
  """Optimization and target flags matching the module recipe, or the native benchmark defaults."""
  return (opt_flag(), *(HOST_CFLAGS if recipe is None else (*recipe.cpu_flags, "-fno-math-errno")))


def _host_recipe(compiler: tuple[str, ...]) -> BuildRecipe:
  try:
    recipe = native_recipe(compiler)
  except (OSError, subprocess.CalledProcessError) as exc:
    raise JitError(f"failed to probe native C compiler {compiler!r}: {exc}") from exc
  override = env("SCALY_VECTOR_LIBM")
  if override is not None:
    if override not in {"none", "glibc"}:
      raise ToolchainError(f"SCALY_VECTOR_LIBM must be 'none' or 'glibc', got {override!r}")
    recipe = replace(recipe, vector_libm=override)
  return recipe


def _render_native(fun: ConcreteFunction, compiler: tuple[str, ...]) -> CModule:
  recipe = _host_recipe(compiler)
  return render_c_module(
    fun, cpu=recipe.cpu, lanes=recipe.lanes, dialect=recipe.dialect, vector_libm=recipe.vector_libm, reciprocal=recipe.reciprocal
  )


def _compile_command(
  compiler: tuple[str, ...],
  recipe: BuildRecipe,
  link_flags: tuple[str, ...],
  source: str,
  output: str,
  objects: tuple[str, ...] = (),
  *,
  external: bool = False,
) -> list[str]:
  # Link libraries (-l in link_flags) MUST come after the source: ld defaults to --as-needed on
  # Linux, so a -lpiqpc/-lipopt placed before the object that references it is dropped (no
  # DT_NEEDED -> "undefined symbol" at dlopen of solver functions).
  exports = (Path(source).with_suffix(".def").as_posix(),) if sys.platform == "win32" else ()
  return [
    *compiler,
    *compile_flags(recipe),
    *no_gather_flags(compiler),
    *vector_libm_flags(compiler, recipe.vector_libm),
    "-fPIC",
    shared_lib_flag(),
    source,
    *exports,
    *objects,
    *(("-Wl,-Bsymbolic",) if external and sys.platform == "linux" else ()),
    *link_flags,
    "-lm",
    "-o",
    output,
  ]


def solver_inputs_fingerprint(link_flags: tuple[str, ...]) -> str:
  """Content digest of solver include/library trees, each read once per process.

  The digest covers indirect headers and bundled library dependencies, even when a plugin's
  version stays unchanged. Native files must stay fixed during the process, as loaded libraries do.
  """
  h = hashlib.sha256()
  directories = dict.fromkeys(Path(flag[2:]) for flag in link_flags if flag.startswith(("-I", "-L")))
  for directory in directories:
    h.update(_solver_directory_fingerprint(directory))
  return h.hexdigest()


@functools.cache
def _solver_directory_fingerprint(directory: Path) -> bytes:
  return directory_fingerprint(directory)


def _hash_parts(*parts: str) -> str:
  h = hashlib.sha256()
  for part in parts:
    h.update(part.encode() + b"\0")
  return h.hexdigest()


def _compute_cache_key(source: str, *, fun_name: str, command: list[str], toolchain: str, solver_inputs: str = "") -> str:
  """SHA-256 over the rendered C source plus the cache-version and ABI signature.

  Any change to the codegen output, the ABI surface, or `_JIT_CACHE_VERSION` invalidates
  previously cached artifacts. ConcreteFunction names and the full compile command are included so
  two functions that happen to share a source skeleton (different symbols or solver rpaths) still
  get distinct entries. ``toolchain`` is the compiler's ``compiler_fingerprint``, so another
  compiler, a changed one at the same path, or another CPU misses the cache. ``solver_inputs``
  identifies the native headers and libraries used to build a solver module.
  """
  return _hash_parts(_JIT_CACHE_VERSION, C_API_SIGNATURE, fun_name, source, toolchain, solver_inputs, *command)


def _toolchain(fun: ConcreteFunction, compiler: Compiler, libraries: tuple[LibraryBuild, ...] = ()) -> str:
  try:
    toolchain = compiler_fingerprint(compiler.command)
  except (OSError, subprocess.CalledProcessError) as exc:
    raise JitError(f"failed to identify C compiler {shlex.join(compiler.command)!r}: {exc}") from exc
  if libraries:
    toolchain += ":extern:" + ":".join(build.key for build in libraries)
  if epoch := _cache_epochs.get(fun.name, 0):
    toolchain += f":{os.getpid()}:{epoch}"
  return toolchain


def _render_and_key(fun: ConcreteFunction, compiler: Compiler) -> tuple[CModule, str]:
  """Render ``fun`` for the host and derive its cache key, which hashes exactly the text handed to the compiler."""
  module = _render_native(fun, compiler.command)
  try:
    toolchain = _toolchain(fun, compiler, module._library_builds)
  except (OSError, subprocess.CalledProcessError) as exc:
    detail = (exc.stderr or exc.stdout) if isinstance(exc, subprocess.CalledProcessError) else str(exc)
    raise JitError(f"failed to prepare external libraries for {fun.name!r}: {detail}") from exc
  command = _compile_command(
    compiler.command, module.recipe, module.link_flags, "module.c", "module" + shared_lib_ext(), external=bool(module._libraries)
  )
  solver_inputs = solver_inputs_fingerprint(module.link_flags) if module.backends else ""
  return module, _compute_cache_key(module.body, fun_name=fun.name, command=command, toolchain=toolchain, solver_inputs=solver_inputs)


@functools.cache
def _scaly_version() -> str:
  return importlib.metadata.version("scaly")


def _index_key(fun: ConcreteFunction, compiler: Compiler) -> str | None:
  """The key of ``fun``'s index entry, from everything rendering and compiling read, without lowering.

  None when there is none to trust: the digest cannot encode the graph, the source on disk may not
  be the code this process runs, a render observer watches ``fun``, or this process
  invalidated ``fun`` and builds it apart from the entry other processes use. A pass put in the
  pipeline is code the source digest does not see."""
  if _cache_epochs.get(fun.name, 0) or program.PASS_PIPELINE is not program.SHIPPED_PIPELINE:
    return None
  if any(begin(fun) is not None for begin in _RENDER_OBSERVERS):
    return None
  if (graph := graph_digest(fun)) is None or (source := source_digest()) is None:
    return None
  recipe = _host_recipe(compiler.command)
  backends = solver_backends_used(fun)
  try:
    libraries = tuple(prepare_library(library, compiler.command) for library in libraries_of(fun))
    link_flags = (*backend_compile_flags(backends), *recipe.link_flags, *(flag for library in libraries for flag in library.link_flags))
  except (SolverLibraryError, OSError, subprocess.CalledProcessError):
    return None  # rendering says what is missing
  command = _compile_command(compiler.command, recipe, link_flags, "module.c", "module" + shared_lib_ext(), external=bool(libraries))
  solver_inputs = solver_inputs_fingerprint(link_flags) if backends else ""
  libc = platform.libc_ver() if recipe.vector_libm == "glibc" else ()
  toolchain = _toolchain(fun, compiler, libraries)
  return _hash_parts(_JIT_CACHE_VERSION, C_API_SIGNATURE, _scaly_version(), source, graph, repr(recipe), toolchain, solver_inputs, *libc, *command)


@dataclass(frozen=True, slots=True)
class _Artifact:
  lib_path: Path
  key: str
  flags: tuple[str, ...]
  workspace_size: int
  solver_library: bool
  prints: bool
  namespace: tuple[str, ...] = ()


_artifact_cache: dict[str, _Artifact] = {}
_cache_epochs: dict[str, int] = {}
_artifact_lock = threading.Lock()


def _publish(temporary: Path, path: Path) -> None:
  """Rename a finished file into the cache, where entries are content-addressed.

  Windows refuses to replace a file another process has open. That process published the same
  content first and is compiling or running it, so its file stays.
  """
  try:
    temporary.replace(path)
  except PermissionError:
    if not path.exists():
      raise
    temporary.unlink()


def _index_path(index: str) -> Path:
  return cache_root() / "index" / f"{index}.json"


def _index_read(index: str) -> _Artifact | None:
  """The artifact the index names for ``index``, or None when it has no entry."""
  try:
    entry = json.loads(_index_path(index).read_text())
    return _Artifact(
      lib_path=cache_root() / entry.pop("library"),
      flags=tuple(entry.pop("flags")),
      namespace=tuple(entry.pop("namespace")),
      **entry,
    )
  except (OSError, ValueError, KeyError, TypeError, AttributeError):
    return None


def _index_write(index: str, artifact: _Artifact) -> None:
  """Record ``artifact`` under ``index``, its library by its path inside the cache so the cache can move.
  A library this process built under another cache root is not recorded in this one."""
  root = cache_root()
  if not artifact.lib_path.is_relative_to(root):
    return
  entry = asdict(artifact)
  entry["library"] = entry.pop("lib_path").relative_to(root).as_posix()
  path = _index_path(index)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
  temporary.write_text(json.dumps(entry))
  _publish(temporary, path)


def _write_cached_text(path: Path, text: str) -> None:
  if path.exists():
    return
  temporary = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
  temporary.write_text(text)
  _publish(temporary, path)


def _build_artifact(fun: ConcreteFunction) -> _Artifact:
  """Render, compile (if needed), and return a path to ``fun``'s cached shared object.

  Raises ``JitUnavailable`` if there is no usable compiler or codegen does not support
  ``fun``; callers let it propagate (there is no interpreter fallback). Raises ``JitError``
  if the compiler itself returns a non-zero status.
  """
  compiler = find_c_compiler()
  if compiler is None:
    raise JitUnavailable("no C compiler found (set SCALY_CC or install cc)")
  mode = env("SCALY_JIT_KEY") or "index"
  if mode not in ("index", "verify"):
    raise JitError(f"SCALY_JIT_KEY must be unset or 'verify', got {mode!r}")
  verify = mode == "verify"
  index = _index_key(fun, compiler)
  indexed = None if index is None else _index_read(index)
  if indexed is not None and not verify and indexed.lib_path.exists():
    return indexed

  try:
    module, key = _render_and_key(fun, compiler)
  except NotImplementedError as exc:
    raise JitUnavailable(f"codegen does not support function {fun.name!r}: {exc}") from exc
  symbol = c_ident(fun.name)
  cache_dir = cache_root() / key
  source_path = cache_dir / f"{symbol}.c"
  lib_path = cache_dir / f"lib{symbol}{shared_lib_ext()}"
  artifact = _Artifact(
    lib_path=lib_path,
    key=key,
    flags=module.link_flags,
    workspace_size=module.workspace_size,
    solver_library=bool(module.backends or module._libraries),
    prints=prints(module.program) or bool(module._libraries),
    namespace=("extern", *(f"{link.path}:{link.identity}" for build in module._library_builds for link in build.links)) if module._libraries else (),
  )
  if verify and index is not None and indexed is not None and replace(indexed, lib_path=lib_path) != artifact:
    raise JitError(f"the index entry {_index_path(index)} for {fun.name!r} names a library built from other C than {fun.name!r} renders")
  with _artifact_lock:
    cached = _artifact_cache.get(key)
  if cached is not None and cached.lib_path.exists():
    if index is not None and indexed != cached:
      _index_write(index, cached)
    return cached

  cache_dir.mkdir(parents=True, exist_ok=True)
  if not lib_path.exists():
    _write_cached_text(source_path, module.body)
    if sys.platform == "win32":
      # Explicit CasADi oracle exports disable MinGW's automatic exports of our entry points.
      # A module definition file also works with Zig, which ignores --export-all-symbols.
      exports = [symbol]
      if module.backends:
        exports += [f"{symbol}_with_options"]
        for solver_symbol in solver_stats_symbols(fun):
          exports += [f"{solver_symbol}_stats", f"{solver_symbol}_default_options"]
      _write_cached_text(source_path.with_suffix(".def"), "EXPORTS\n" + "\n".join(exports) + "\n")
    # Compile to a temp lib unique to this process and thread, then atomically rename, so concurrent
    # builds of the same function (pytest-xdist workers or threads on a cold cache) never observe a half-written .so.
    tmp_lib = lib_path.with_suffix(lib_path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    # zig keeps a separate build of parts of its C runtime for every working directory that contains
    # its installation, such as a checkout holding the venv, so compile from the temporary directory.
    # It is also short, which the function's own directory may not be: Windows refuses a working
    # directory longer than 260 characters.
    try:
      objects = tuple(obj for build in module._library_builds for obj in compile_library(build, compiler.command))
      subprocess.run(
        _compile_command(
          compiler.command, module.recipe, module.link_flags, source_path.as_posix(), tmp_lib.as_posix(), objects, external=bool(module._libraries)
        ),
        cwd=tempfile.gettempdir(),
        check=True,
        capture_output=True,
        text=True,
      )
    except subprocess.CalledProcessError as exc:
      tmp_lib.unlink(missing_ok=True)
      raise JitError(f"failed to compile {fun.name!r}: {exc.stderr or exc.stdout}") from exc
    _publish(tmp_lib, lib_path)

  with _artifact_lock:
    _artifact_cache[key] = artifact
  if index is not None and indexed != artifact:
    _index_write(index, artifact)
  return artifact


class CompiledFunction:
  """Handle around a JIT-compiled `ConcreteFunction`.

  Holds the ``ctypes.CDLL`` for the cached shared object, the resolved entry point with
  ``argtypes``/``restype`` set up for the pointer ABI, and the workspace size the rendered module
  reported. That is the header's ``SZ_W``, since the library exports no size query of its own.
  """

  __slots__ = (
    "_fun",
    "_artifact",
    "_lib",
    "_symbol",
    "_entry",
    "_stats_entries",
    "_options",
    "_sz_w",
    "_input_sizes",
    "_input_shapes",
    "_input_names",
    "_output_shapes",
    "_output_sizes",
    "_n_args",
    "_n_res",
  )

  def __init__(self, fun: Function | ConcreteFunction):
    fun = as_concrete(fun)
    self._fun = fun
    self._artifact = _build_artifact(fun)
    self._lib = load_library(
      self._artifact.lib_path, isolated=self._artifact.solver_library, **({"namespace": self._artifact.namespace} if self._artifact.namespace else {})
    )
    symbol = c_ident(fun.name)
    self._symbol = symbol
    solvers = solver_functions(fun)
    entry = getattr(self._lib, f"{symbol}_with_options" if solvers else symbol)
    entry.argtypes = [
      ctypes.POINTER(_C_DOUBLE_P),
      ctypes.POINTER(_C_DOUBLE_P),
      _C_INT_P,
      _C_DOUBLE_P,
      ctypes.c_int,
    ]
    self._options = (ctypes.POINTER(CSolverOption) * len(solvers))(*(fn.descriptor.runtime_options for fn in solvers))
    if solvers:
      entry.argtypes = [*entry.argtypes, ctypes.POINTER(ctypes.POINTER(CSolverOption))]
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
    self._input_shapes = tuple(e.type.storage_shape for e in fun.inputs)
    self._input_sizes = tuple(e.type.storage_size for e in fun.inputs)
    self._output_shapes = tuple(e.type.storage_shape for e in fun.outputs)
    self._output_sizes = tuple(e.type.storage_size for e in fun.outputs)
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

  def run(self, args: list[Any]) -> list[Any]:
    """Dispatch the compiled entry point with positional NumPy inputs.

    Inputs are converted to contiguous float64 buffers with shapes validated against the
    declared input shapes. Outputs are freshly-allocated NumPy arrays reshaped to the
    ConcreteFunction's declared output shapes (the compact buffer shape for sparse outputs).
    Raises ``JitError`` if the C function returns a non-zero ABI status.
    """
    if len(args) != self._n_args:
      raise TypeError(f"expected {self._n_args} inputs, got {len(args)}")

    arg_buffers: list[np.ndarray] = []
    arg_array = (_C_DOUBLE_P * max(self._n_args, 1))()
    for i, value in enumerate(args):
      name = self._input_names[i]
      expected_shape = self._input_shapes[i]
      layout = self._fun.input_tree.layouts[i]
      arr = np.asarray(value, dtype=np.float64) if layout is None else layout.numerical_values(value, self._fun.inputs[i].shape)
      if arr.shape != expected_shape:
        raise ValueError(f"input {name!r} has shape {arr.shape}, expected {expected_shape}")
      if isinstance(layout, DenseLayout):
        arr = arr.ravel(order="K")
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

    # C stdout buffers apart from sys.stdout, and holds its lines until exit when it is not a terminal.
    # Flushing both around a call that prints keeps the two in the order they were written.
    if self._artifact.prints:
      sys.stdout.flush()
    status = self._entry(arg_array, res_array, _C_INT_P(), w_ptr, 0, *((self._options,) if self._options else ()))
    if self._artifact.prints:
      libc = self._lib if self._artifact.solver_library and sys.platform == "linux" else _process_libc()
      if flush := getattr(libc, "fflush", None):
        flush(None)
    if status != 0:
      raise JitError(f"{self._fun.name} returned ABI status {status}")

    return [
      layout.sparse_array(out, self._fun.outputs[i].shape)
      if isinstance(layout, SparseLayout)
      else layout.dense_array(out, self._output_shapes[i])
      if isinstance(layout, DenseLayout)
      else out.reshape(self._output_shapes[i])
      for i, (out, layout) in enumerate(zip(outputs, self._fun.output_tree.layouts, strict=True))
    ]


def get_compiled(fun: Function | ConcreteFunction) -> CompiledFunction:
  """Compile ``fun`` (or reuse a cached `.so`) and return a `CompiledFunction` handle."""
  return CompiledFunction(fun)


def invalidate_cache(fun: ConcreteFunction) -> None:
  """Make ``fun``'s next build in this process use another cache path.

  Existing source files and libraries stay on disk for processes still building or loading them.
  Loaded handles remain callable, including on Windows, where loaded DLLs cannot be deleted or
  replaced. After ``recompile()`` in one process, a fresh process still reuses the original shared
  library instead of rebuilding it. Scaly writes each library under a temporary name and renames
  it into place only when complete, so no process finds a partial library.

  Scaly does not prune old libraries after process exit. To reclaim the space, delete the cache
  manually while no process is using it. Safe to call before the first build.
  """
  with _artifact_lock:
    _cache_epochs[fun.name] = _cache_epochs.get(fun.name, 0) + 1
