"""Centralized native-toolchain and vendored-solver discovery.

Alloy's normal Python workflow is controlled by a small set of environment
variables. Keep their names and defaults registered here so JIT compilation,
solver ctypes loading, tests, and diagnostics agree on one source of truth.

PIQP/IPOPT are built from source by ``hatch_build.py`` into the package layout
``alloy/lib`` and ``alloy/include``. A few explicit override variables are kept
for debugging or downstream packaging, but the supported release path is the
vendored source build.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class EnvVar:
  name: str
  default: str | None
  help: str


ENV_VARS: tuple[EnvVar, ...] = (
  EnvVar("ALLOY_CACHE_DIR", None, "Override the JIT cache root."),
  EnvVar("ALLOY_CC", None, "Override the C compiler used by the JIT."),
  EnvVar("ALLOY_SOLVER_INCLUDE_DIR", None, "Override the vendored solver C header directory."),
  EnvVar("ALLOY_SOLVER_LIB_DIR", None, "Override the vendored solver shared-library directory."),
  EnvVar("ALLOY_PIQP_LIB", None, "Exact path to libpiqpc."),
  EnvVar("ALLOY_IPOPT_LIB", None, "Exact path to libipopt."),
  EnvVar("ALLOY_SOLVER_SYSTEM_FALLBACK", "0", "Experimental: allow ctypes/pkg-config/default-linker system solver fallback."),
  EnvVar("ALLOY_BUILD_SOLVERS", "auto", "Build-hook solver mode: auto, skip, or required/1/true."),
  EnvVar("ALLOY_VIZ_DIR", None, "Visualization recording directory."),
  EnvVar("ALLOY_TRACKING_SWEEP", "0", "Run the opt-in tracking sparse-Jacobian sweep."),
  EnvVar("ALLOY_GBENCH", "0", "Run the opt-in Google Benchmark Python-dispatch microbenchmark."),
)

SolverName = Literal["piqp", "ipopt"]
SolverStem = Literal["piqpc", "ipopt"]


class ToolchainError(RuntimeError):
  """Raised when native compiler/header/library discovery fails."""


class SolverLibraryError(ToolchainError):
  """Raised when a requested solver shared library cannot be located or loaded."""


@dataclass(frozen=True, slots=True)
class Compiler:
  cc: str
  source: str


@dataclass(frozen=True, slots=True)
class SolverPaths:
  include_dirs: tuple[Path, ...]
  lib_dirs: tuple[Path, ...]
  piqp_lib: Path | None
  ipopt_lib: Path | None
  piqp_load: str | None
  ipopt_load: str | None
  source: str


def _package_root() -> Path:
  return Path(__file__).resolve().parent


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


def env(name: str, default: str | None = None) -> str | None:
  return os.environ.get(name, default)


def env_bool(name: str, default: bool = False) -> bool:
  raw = os.environ.get(name)
  if raw is None or raw == "":
    return default
  val = raw.strip().lower()
  if val in {"1", "true", "yes", "on"}:
    return True
  if val in {"0", "false", "no", "off"}:
    return False
  raise ToolchainError(f"{name} must be a boolean (1/0, true/false, yes/no, on/off), got {raw!r}")


def env_path(name: str) -> Path | None:
  raw = os.environ.get(name)
  return Path(raw).expanduser() if raw else None


def alloy_env_vars() -> tuple[EnvVar, ...]:
  return ENV_VARS


def cache_root() -> Path:
  override = env_path("ALLOY_CACHE_DIR")
  if override is not None:
    return override
  xdg = env_path("XDG_CACHE_HOME")
  base = xdg if xdg is not None else Path.home() / ".cache"
  return base / "alloy" / "jit"


def shared_lib_ext() -> str:
  if sys.platform == "darwin":
    return ".dylib"
  if sys.platform == "win32":
    return ".dll"
  return ".so"


def shared_lib_flag() -> str:
  return "-dynamiclib" if sys.platform == "darwin" else "-shared"


def find_c_compiler() -> Compiler | None:
  for key in ("ALLOY_CC", "CC"):
    override = os.environ.get(key)
    if override:
      found = shutil.which(override)
      return Compiler(found, key) if found is not None else None
  found = shutil.which("cc")
  return Compiler(found, "PATH") if found is not None else None


def _lib_filename(stem: SolverStem) -> str:
  return f"lib{stem}{shared_lib_ext()}" if sys.platform != "win32" else f"{stem}.dll"


def _existing_dir(path: Path | None) -> Path | None:
  return path if path is not None and path.exists() and path.is_dir() else None


def _library_from_dirs(stem: SolverStem, dirs: tuple[Path, ...]) -> Path | None:
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


def _system_library(stem: SolverStem) -> str | None:
  if not env_bool("ALLOY_SOLVER_SYSTEM_FALLBACK", False):
    return None
  return ctypes.util.find_library(stem)


def solver_paths(required: bool = False) -> SolverPaths:
  include_dirs: list[Path] = []
  lib_dirs: list[Path] = []
  source = "unconfigured"

  explicit_include = _existing_dir(env_path("ALLOY_SOLVER_INCLUDE_DIR"))
  explicit_lib = _existing_dir(env_path("ALLOY_SOLVER_LIB_DIR"))
  if explicit_include is not None:
    include_dirs.append(explicit_include)
    source = "ALLOY_SOLVER_INCLUDE_DIR"
  if explicit_lib is not None:
    lib_dirs.append(explicit_lib)
    source = "ALLOY_SOLVER_LIB_DIR" if source == "unconfigured" else source

  pkg = _package_root()
  pkg_inc = _existing_dir(pkg / "include")
  pkg_lib = _existing_dir(pkg / "lib")
  if pkg_inc is not None:
    include_dirs.append(pkg_inc)
  if pkg_lib is not None:
    lib_dirs.append(pkg_lib)
  if source == "unconfigured" and (pkg_inc is not None or pkg_lib is not None):
    source = "vendored"

  piqp_exact = env_path("ALLOY_PIQP_LIB")
  ipopt_exact = env_path("ALLOY_IPOPT_LIB")
  if piqp_exact is not None:
    lib_dirs.append(piqp_exact.parent)
    source = "ALLOY_PIQP_LIB"
  if ipopt_exact is not None:
    lib_dirs.append(ipopt_exact.parent)
    source = "ALLOY_IPOPT_LIB" if source == "unconfigured" else source

  include_tuple = _dedup_paths(include_dirs)
  lib_tuple = _dedup_paths(lib_dirs)
  piqp_lib = piqp_exact if piqp_exact is not None and piqp_exact.exists() else _library_from_dirs("piqpc", lib_tuple)
  ipopt_lib = ipopt_exact if ipopt_exact is not None and ipopt_exact.exists() else _library_from_dirs("ipopt", lib_tuple)
  piqp_load = str(piqp_lib) if piqp_lib is not None else _system_library("piqpc")
  ipopt_load = str(ipopt_lib) if ipopt_lib is not None else _system_library("ipopt")
  if source == "unconfigured" and (piqp_load is not None or ipopt_load is not None):
    source = "system"

  if required and (piqp_load is None or ipopt_load is None):
    raise SolverLibraryError(solver_diagnostic())

  return SolverPaths(include_tuple, lib_tuple, piqp_lib, ipopt_lib, piqp_load, ipopt_load, source)


def _header_candidates(solver: SolverName) -> tuple[Path, ...]:
  if solver == "piqp":
    return (Path("piqp/piqp.h"),)
  return (Path("coin-or/IpStdCInterface.h"),)


def solver_header_include(solver: SolverName) -> str:
  paths = solver_paths()
  for inc in paths.include_dirs:
    for rel in _header_candidates(solver):
      if (inc / rel).exists():
        return str(rel)
  raise SolverLibraryError(solver_diagnostic("piqpc" if solver == "piqp" else "ipopt"))


def _header_available(paths: SolverPaths, solver: SolverName, extra_include_dirs: tuple[Path, ...] = ()) -> bool:
  return any((inc / rel).exists() for inc in (*paths.include_dirs, *extra_include_dirs) for rel in _header_candidates(solver))


def _load_name(paths: SolverPaths, solver: SolverName) -> str | None:
  return paths.piqp_load if solver == "piqp" else paths.ipopt_load


def _solver_to_stem(solver: SolverName) -> SolverStem:
  return "piqpc" if solver == "piqp" else "ipopt"


def solver_library_path(stem: SolverStem) -> Path | None:
  paths = solver_paths()
  return paths.piqp_lib if stem == "piqpc" else paths.ipopt_lib


def solver_library_available(stem: SolverStem) -> bool:
  paths = solver_paths()
  return (paths.piqp_load if stem == "piqpc" else paths.ipopt_load) is not None


def solver_library_loadable(stem: SolverStem) -> bool:
  try:
    load_solver_library(stem)
    return True
  except SolverLibraryError:
    return False


def solver_discoverable(solver: SolverName) -> bool:
  """Return whether Alloy can find both the solver library name/path and C headers."""
  paths = solver_paths()
  return _load_name(paths, solver) is not None and _header_available(paths, solver)


def solver_loadable(solver: SolverName) -> bool:
  """Return whether the discovered solver can be dlopened and has headers for JIT/AOT codegen."""
  return solver_discoverable(solver) and solver_library_loadable(_solver_to_stem(solver))


def solver_available(solver: SolverName) -> bool:
  """Compatibility alias for older callers; prefer ``solver_loadable`` or ``solver_discoverable``."""
  return solver_loadable(solver)


def load_solver_library(stem: SolverStem) -> ctypes.CDLL:
  paths = solver_paths()
  load_name = paths.piqp_load if stem == "piqpc" else paths.ipopt_load
  if load_name is None:
    raise SolverLibraryError(solver_diagnostic(stem))
  try:
    return ctypes.CDLL(load_name)
  except OSError as exc:
    raise SolverLibraryError(f"failed to load {stem!r} from {load_name!r}: {exc}\n\n{solver_diagnostic(stem)}") from exc


def _pkg_config_flags(packages: tuple[str, ...], flag: Literal["--cflags", "--libs"]) -> list[str]:
  if not env_bool("ALLOY_SOLVER_SYSTEM_FALLBACK", False) or shutil.which("pkg-config") is None:
    return []
  out: list[str] = []
  for package in packages:
    proc = subprocess.run(["pkg-config", flag, package], check=False, capture_output=True, text=True)
    if proc.returncode == 0:
      out.extend(proc.stdout.strip().split())
  return out


def _include_dirs_from_cflags(cflags: list[str]) -> tuple[Path, ...]:
  return _dedup_paths([Path(flag[2:]) for flag in cflags if flag.startswith("-I") and len(flag) > 2])


def solver_compile_flags(needs_piqp: bool, needs_ipopt: bool, *, rpath: bool = True) -> list[str]:
  if not (needs_piqp or needs_ipopt):
    return []

  paths = solver_paths()
  packages = tuple(x for x, need in (("piqp", needs_piqp), ("ipopt", needs_ipopt)) if need)
  pkg_cflags = _pkg_config_flags(packages, "--cflags")
  pkg_include_dirs = _include_dirs_from_cflags(pkg_cflags)
  missing: list[str] = []
  if needs_piqp and paths.piqp_load is None:
    missing.append("PIQP library")
  if needs_ipopt and paths.ipopt_load is None:
    missing.append("IPOPT library")
  if needs_piqp and not _header_available(paths, "piqp", pkg_include_dirs):
    missing.append("PIQP headers")
  if needs_ipopt and not _header_available(paths, "ipopt", pkg_include_dirs):
    missing.append("IPOPT headers")
  if missing:
    raise SolverLibraryError(f"missing native solver pieces for JIT/AOT codegen: {', '.join(missing)}\n\n{solver_diagnostic()}")

  flags: list[str] = []
  flags.extend(f"-I{p}" for p in paths.include_dirs)
  flags.extend(pkg_cflags)
  flags.extend(f"-L{p}" for p in paths.lib_dirs)
  if rpath:
    flags.extend(f"-Wl,-rpath,{p}" for p in paths.lib_dirs)
  flags.extend(_pkg_config_flags(packages, "--libs"))
  if needs_piqp:
    flags.append("-lpiqpc")
  if needs_ipopt:
    flags.append("-lipopt")
  return flags


def solver_diagnostic(stem: SolverStem | None = None) -> str:
  paths = solver_paths(required=False)
  want = f" for {stem}" if stem else ""
  lines = [f"Could not find usable vendored solver libraries/headers{want}.", ""]
  lines += [
    "Build the source-vendored solvers:",
    "  ALLOY_BUILD_SOLVERS=required uv sync",
    "",
    "Discovery state:",
    f"  source: {paths.source}",
    f"  include dirs: {', '.join(str(p) for p in paths.include_dirs) or '<none>'}",
    f"  lib dirs: {', '.join(str(p) for p in paths.lib_dirs) or '<none>'}",
    f"  piqp lib: {paths.piqp_load or '<missing>'}",
    f"  ipopt lib: {paths.ipopt_load or '<missing>'}",
  ]
  return "\n".join(lines)


def _format_report() -> str:
  paths = solver_paths(required=False)
  compiler = find_c_compiler()
  lines = ["Alloy native toolchain", f"  cache root: {cache_root()}"]
  lines.append(f"  cc: {compiler.cc} ({compiler.source})" if compiler is not None else "  cc: <missing>")
  lines += [
    f"  solver source: {paths.source}",
    f"  include dirs: {', '.join(str(p) for p in paths.include_dirs) or '<none>'}",
    f"  lib dirs: {', '.join(str(p) for p in paths.lib_dirs) or '<none>'}",
    f"  piqp: {paths.piqp_load or '<missing>'}",
    f"  ipopt: {paths.ipopt_load or '<missing>'}",
    f"  piqp discoverable: {solver_discoverable('piqp')}",
    f"  ipopt discoverable: {solver_discoverable('ipopt')}",
    f"  piqp loadable: {solver_loadable('piqp')}",
    f"  ipopt loadable: {solver_loadable('ipopt')}",
  ]
  try:
    lines.append("  JIT solver flags: " + " ".join(solver_compile_flags(True, True)))
  except SolverLibraryError as exc:
    lines.append("  JIT solver flags: <unavailable>")
    lines.append("  reason: " + str(exc).splitlines()[0])
  return "\n".join(lines)


def main() -> None:
  print(_format_report())


if __name__ == "__main__":
  main()
