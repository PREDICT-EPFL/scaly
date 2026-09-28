"""Vendored-solver library and header discovery.

Which solvers exist is not hardcoded here: every installed external method (a ``scaly.methods``
entry point under ``opt.``, an ``opt.external.External``) declares its shared-library stem, C header, and link flags, and its
vendored ``lib`` / ``include`` package directories join the search path. Explicit override
variables and legacy core package paths remain as debugging and migration fallbacks.

The compiler side of the native toolchain lives in ``scaly/codegen/toolchain.py``, which reads this module
for its diagnostics report; the dependency only points that way.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import re
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ...utils.env import EnvVar, ToolchainError, env_bool, env_path, shared_lib_ext
from ..method import external_method, external_methods

if TYPE_CHECKING:
  from collections.abc import Sequence


class SolverLibraryError(ToolchainError):
  """Raised when a requested solver shared library cannot be located or loaded."""


@dataclass(frozen=True, slots=True)
class SolverPaths:
  include_dirs: tuple[Path, ...]
  lib_dirs: tuple[Path, ...]
  # backend name -> dlopen-able path (or system library name under the
  # experimental fallback); None when the backend's library was not found.
  loads: dict[str, str | None]
  source: str


def _package_root() -> Path:
  return Path(__file__).resolve().parents[1]  # the scaly package directory, not solvers/


def _dedup_paths(paths: list[Path]) -> tuple[Path, ...]:
  out: list[Path] = []
  seen: set[str] = set()
  for p in paths:
    ep = p.expanduser()
    key = str(ep)
    if key not in seen:
      seen.add(key)
      out.append(ep)
  return tuple(out)


def _lib_filename(stem: str) -> str:
  return f"lib{stem}{shared_lib_ext()}" if sys.platform != "win32" else f"{stem}.dll"


def _existing_dir(path: Path | None) -> Path | None:
  return path if path is not None and path.exists() and path.is_dir() else None


def _library_from_dirs(stem: str, dirs: tuple[Path, ...]) -> Path | None:
  name = _lib_filename(stem)
  for d in dirs:
    p = d / name
    if p.exists():
      return p
  if shared_lib_ext() == ".so":
    for d in dirs:
      matches = sorted(d.glob(f"lib{stem}.so*"))
      if matches:
        return matches[0]
  return None


def _system_library(stem: str) -> str | None:
  if not env_bool("SCALY_SOLVER_SYSTEM_FALLBACK", False):
    return None
  return ctypes.util.find_library(stem)


def _plugin_solver_paths() -> list[tuple[Path, Path]]:
  out: list[tuple[Path, Path]] = []
  for name, method in _backends().items():
    try:
      out.append((method.include_dir(), method.lib_dir()))
    except Exception as exc:  # noqa: BLE001 - one broken plugin must not hide the others
      warnings.warn(f"could not inspect opt.{name}: {exc}", RuntimeWarning, stacklevel=2)
  return out


def _backends() -> dict[str, Any]:
  return external_methods()


def _lib_env_var(name: str) -> str:
  return f"SCALY_{re.sub(r'\W', '_', name.upper())}_LIB"


def solver_paths(required: bool = False) -> SolverPaths:
  backends = _backends()
  include_dirs: list[Path] = []
  lib_dirs: list[Path] = []
  source = "unconfigured"

  exact: dict[str, Path] = {}
  for name in backends:
    exact_lib = env_path(_lib_env_var(name))
    if exact_lib is not None:
      exact[name] = exact_lib
      lib_dirs.append(exact_lib.parent)
      source = _lib_env_var(name) if source == "unconfigured" else source

  explicit_include = _existing_dir(env_path("SCALY_SOLVER_INCLUDE_DIR"))
  explicit_lib = _existing_dir(env_path("SCALY_SOLVER_LIB_DIR"))
  if explicit_include is not None:
    include_dirs.append(explicit_include)
    source = "SCALY_SOLVER_INCLUDE_DIR" if source == "unconfigured" else source
  if explicit_lib is not None:
    lib_dirs.append(explicit_lib)
    source = "SCALY_SOLVER_LIB_DIR" if source == "unconfigured" else source

  plugin_paths = _plugin_solver_paths()
  for include_dir, lib_dir in plugin_paths:
    if _existing_dir(include_dir) is not None:
      include_dirs.append(include_dir)
    if _existing_dir(lib_dir) is not None:
      lib_dirs.append(lib_dir)
  if source == "unconfigured" and plugin_paths:
    source = "plugin"

  pkg = _package_root()
  pkg_inc = _existing_dir(pkg / "include")
  pkg_lib = _existing_dir(pkg / "lib")
  if pkg_inc is not None:
    include_dirs.append(pkg_inc)
  if pkg_lib is not None:
    lib_dirs.append(pkg_lib)
  if source == "unconfigured" and (pkg_inc is not None or pkg_lib is not None):
    source = "vendored"

  include_tuple = _dedup_paths(include_dirs)
  lib_tuple = _dedup_paths(lib_dirs)
  loads: dict[str, str | None] = {}
  for name, backend in backends.items():
    lib = exact[name] if name in exact and exact[name].exists() else _library_from_dirs(backend.lib_stem, lib_tuple)
    loads[name] = str(lib) if lib is not None else _system_library(backend.lib_stem)
  if source == "unconfigured" and any(load is not None for load in loads.values()):
    source = "system"

  if required and (not loads or any(load is None for load in loads.values())):
    raise SolverLibraryError(solver_diagnostic())

  return SolverPaths(include_tuple, lib_tuple, loads, source)


def _backend_header(name: str) -> Path | None:
  backend = _backends().get(name)
  return Path(backend.header) if backend is not None else None


def solver_header_include(name: str) -> str:
  paths = solver_paths()
  rel = _backend_header(name)
  if rel is not None:
    for inc in paths.include_dirs:
      if (inc / rel).exists():
        return str(rel)
  raise SolverLibraryError(solver_diagnostic(name))


def _header_available(paths: SolverPaths, name: str, extra_include_dirs: tuple[Path, ...] = ()) -> bool:
  rel = _backend_header(name)
  return rel is not None and any((inc / rel).exists() for inc in (*paths.include_dirs, *extra_include_dirs))


def solver_discoverable(name: str) -> bool:
  """Return whether Scaly can find the named solver plugin's library and C headers."""
  paths = solver_paths()
  return paths.loads.get(name) is not None and _header_available(paths, name)


def solver_loadable(name: str) -> bool:
  """Return whether the discovered solver can be dlopened and has headers for JIT/AOT codegen."""
  if not solver_discoverable(name):
    return False
  load_name = solver_paths().loads[name]
  assert load_name is not None
  if sys.platform == "linux":
    return (
      subprocess.run(
        [sys.executable, "-c", "import ctypes, sys; ctypes.CDLL(sys.argv[1])", load_name],
        check=False,
        capture_output=True,
      ).returncode
      == 0
    )
  try:
    ctypes.CDLL(load_name)
    return True
  except OSError:
    return False


def _pkg_config_flags(packages: tuple[str, ...], flag: Literal["--cflags", "--libs"]) -> list[str]:
  if not env_bool("SCALY_SOLVER_SYSTEM_FALLBACK", False) or shutil.which("pkg-config") is None:
    return []
  out: list[str] = []
  for package in packages:
    proc = subprocess.run(["pkg-config", flag, package], check=False, capture_output=True, text=True)
    if proc.returncode == 0:
      out.extend(proc.stdout.strip().split())
  return out


def _include_dirs_from_cflags(cflags: list[str]) -> tuple[Path, ...]:
  return _dedup_paths([Path(flag[2:]) for flag in cflags if flag.startswith("-I") and len(flag) > 2])


def backend_compile_flags(names: Sequence[str], *, rpath: bool = True) -> list[str]:
  """Include/lib/rpath/link flags for the named solver plugins, in order.

  ``graph.solver_compile_flags`` is the same question asked of a ``Function``."""
  if not names:
    return []
  backends = [external_method(name) for name in names]
  paths = solver_paths()
  pkg_cflags = _pkg_config_flags(tuple(names), "--cflags")
  pkg_include_dirs = _include_dirs_from_cflags(pkg_cflags)
  missing: list[str] = []
  for name in names:
    if paths.loads.get(name) is None:
      missing.append(f"{name} library")
    if not _header_available(paths, name, pkg_include_dirs):
      missing.append(f"{name} headers")
  if missing:
    raise SolverLibraryError(f"missing native solver pieces for JIT/AOT codegen: {', '.join(missing)}\n\n{solver_diagnostic()}")

  flags: list[str] = []
  flags.extend(f"-I{p}" for p in paths.include_dirs)
  flags.extend(pkg_cflags)
  flags.extend(f"-L{p}" for p in paths.lib_dirs)
  if rpath:
    flags.extend(f"-Wl,-rpath,{p}" for p in paths.lib_dirs)
  flags.extend(_pkg_config_flags(tuple(names), "--libs"))
  for backend in backends:
    flags.extend(backend.link_flags)
  return flags


def solver_diagnostic(name: str | None = None) -> str:
  paths = solver_paths(required=False)
  want = f" for {name}" if name else ""
  lines = [f"Could not find usable vendored solver libraries/headers{want}.", ""]
  lines += [
    "Build the source-vendored solvers:",
    "  SCALY_BUILD_SOLVERS=required uv sync",
    "",
    "Discovery state:",
    f"  source: {paths.source}",
    f"  include dirs: {', '.join(str(p) for p in paths.include_dirs) or '<none>'}",
    f"  lib dirs: {', '.join(str(p) for p in paths.lib_dirs) or '<none>'}",
    *(f"  {backend} lib: {load or '<missing>'}" for backend, load in sorted(paths.loads.items())),
  ]
  if not paths.loads:
    lines.append("  installed solver plugins: <none>")
  return "\n".join(lines)


SOLVER_ENV_VARS: tuple[EnvVar, ...] = (
  EnvVar("SCALY_SOLVER_INCLUDE_DIR", None, "Override the vendored solver C header directory."),
  EnvVar("SCALY_SOLVER_LIB_DIR", None, "Override the vendored solver shared-library directory."),
  EnvVar("SCALY_<NAME>_LIB", None, "Exact path to an installed solver plugin's shared library (e.g. SCALY_PIQP_LIB, SCALY_IPOPT_LIB)."),
  EnvVar("SCALY_SOLVER_SYSTEM_FALLBACK", "0", "Experimental: allow ctypes/pkg-config/default-linker system solver fallback."),
  EnvVar("SCALY_BUILD_SOLVERS", "auto", "Build-hook solver mode: auto, skip, or required/1/true."),
)
"""The environment variables solver discovery and the vendored builds read, listed by
``scaly.utils.env.scaly_env_vars`` through the ``scaly.env_vars`` entry points."""


def toolchain_report() -> list[str]:
  """The solver section of ``scaly_toolchain``: where libraries and headers were found, whether each
  plugin loads, and the flags the JIT would link with."""
  paths = solver_paths(required=False)
  lines = [
    f"  solver source: {paths.source}",
    f"  include dirs: {', '.join(str(p) for p in paths.include_dirs) or '<none>'}",
    f"  lib dirs: {', '.join(str(p) for p in paths.lib_dirs) or '<none>'}",
  ]
  for name in sorted(paths.loads):
    lines += [
      f"  {name}: {paths.loads[name] or '<missing>'}",
      f"  {name} discoverable: {solver_discoverable(name)}",
      f"  {name} loadable: {solver_loadable(name)}",
    ]
  try:
    lines.append("  JIT solver flags: " + " ".join(backend_compile_flags(tuple(sorted(paths.loads)))))
  except SolverLibraryError as exc:
    lines.append("  JIT solver flags: <unavailable>")
    lines.append("  reason: " + str(exc).splitlines()[0])
  return lines
