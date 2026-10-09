"""Native C compiler discovery, JIT cache location, and the toolchain diagnostics report.

Scaly's normal Python workflow is controlled by a small set of environment variables, registered
in ``utils/env.py``. Solver library and header discovery lives in ``solvers/paths.py``; this
module reads it for the report and nothing else depends on that direction.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import shlex
import subprocess
from functools import lru_cache
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..solvers.paths import SolverLibraryError, backend_compile_flags, solver_discoverable, solver_loadable, solver_paths
from ..utils.env import env_path


@dataclass(frozen=True, slots=True)
class Compiler:
  """The command the JIT runs as its C compiler, and the setting that supplied it."""

  command: tuple[str, ...]
  source: str


CPU_LEVELS = ("generic", "native", "x86-64-v3", "x86-64-v4", "apple-m4")
type CpuLevel = Literal["generic", "native", "x86-64-v3", "x86-64-v4", "apple-m4"]
type LaneCount = Literal["auto"] | Literal[1, 2, 4, 8]
type CDialect = Literal["gnu", "c"]
type VectorLibm = Literal["none", "glibc"]


@dataclass(frozen=True, slots=True)
class BuildRecipe:
  """CPU baseline, render policy, and compiler flags for one generated module."""

  cpu: CpuLevel = "generic"
  lanes: LaneCount = "auto"
  dialect: CDialect = "gnu"
  vector_libm: VectorLibm = "none"
  reciprocal: bool = False

  def __post_init__(self) -> None:
    if self.cpu not in CPU_LEVELS:
      raise ValueError(f"cpu must be one of {CPU_LEVELS}, got {self.cpu!r}")
    if self.lanes != "auto" and (type(self.lanes) is not int or self.lanes not in (1, 2, 4, 8)):
      raise ValueError("lanes must be 'auto', 1, 2, 4, or 8")
    if self.dialect not in ("gnu", "c"):
      raise ValueError("dialect must be 'gnu' or 'c'")
    if self.vector_libm not in ("none", "glibc"):
      raise ValueError("vector_libm must be 'none' or 'glibc'")
    if self.dialect == "c" and self.vector_libm != "none":
      raise ValueError("vector_libm='glibc' requires dialect='gnu'")

  @property
  def cpu_flags(self) -> tuple[str, ...]:
    """Target flags shared by GCC and Clang for the selected CPU baseline."""
    native = "-mcpu=native" if platform.machine().lower() in ("arm64", "aarch64") else "-march=native"
    return {
      "generic": (),
      "native": (native,),
      "x86-64-v3": ("-march=x86-64-v3",),
      "x86-64-v4": ("-march=x86-64-v4",),
      "apple-m4": ("-mcpu=apple-m4",),
    }[self.cpu]

  @property
  def link_flags(self) -> tuple[str, ...]:
    """Additional libraries required by the selected math implementation."""
    return ("-lmvec",) if self.vector_libm == "glibc" else ()

  def comment(self, source_name: str) -> str:
    """Describe the CPU contract and exact object-build commands for generated source."""
    baseline = "host-local native CPU" if self.cpu == "native" else self.cpu
    libc = "glibc x86-64; tanh requires glibc >= 2.35" if self.vector_libm == "glibc" else "scalar libm"
    flags = ("-O3", *self.cpu_flags, "-fno-math-errno", "-c", source_name)
    commands = [shlex.join((cc, *flags)) for cc in ("gcc", "clang")]
    return "\n".join(
      (
        "/* Scaly build recipe",
        f" * CPU baseline: {baseline}",
        f" * lanes={self.lanes}, dialect={self.dialect}, vector_libm={self.vector_libm}, reciprocal={self.reciprocal}",
        f" * Math library: {libc}",
        *(f" * {command}" for command in commands),
        " * Link with: " + " ".join((*self.link_flags, "-lm")),
        " */",
        "",
      )
    )


def _native_macros(command: tuple[str, ...]) -> str:
  """The compiler's predefined macros for the native CPU: its target features and its version."""
  return subprocess.run(
    [*command, *BuildRecipe(cpu="native").cpu_flags, "-dM", "-E", "-x", "c", "-"],
    input="",
    text=True,
    capture_output=True,
    check=True,
  ).stdout


@lru_cache(maxsize=8)
def native_recipe(command: tuple[str, ...]) -> BuildRecipe:
  """Resolve fixed host lanes and the available native math library for JIT compilation."""
  macros = {
    parts[1]: parts[2] if len(parts) > 2 else "" for line in _native_macros(command).splitlines() if (parts := line.split())[:1] == ["#define"]
  }
  sve256 = macros.get("__ARM_FEATURE_SVE_BITS", "0").isdigit() and int(macros.get("__ARM_FEATURE_SVE_BITS", "0")) >= 256
  lanes = 8 if "__AVX512F__" in macros else 4 if "__AVX__" in macros or sve256 else 2 if {"__SSE2__", "__aarch64__"} & macros.keys() else 1
  libc, version = platform.libc_ver()
  version_parts = tuple(int(part) for part in version.split(".")[:2]) if version and all(p.isdigit() for p in version.split(".")[:2]) else ()
  vector_libm = "glibc" if "__x86_64__" in macros and libc == "glibc" and version_parts >= (2, 35) and lanes > 1 else "none"
  return BuildRecipe(cpu="native", lanes=lanes, vector_libm=vector_libm)


def compiler_fingerprint(command: tuple[str, ...]) -> str:
  """Hash what decides the machine code ``command`` emits on this host, for the JIT cache key.

  Covers the command, its executable's real path, size and modification time, its ``--version``
  output, and its native target macros, so a wrapper, a patched compiler at the same path or another
  CPU misses the cache. The compiler runs again only when its executable changes.
  """
  executable = Path(shutil.which(command[0]) or command[0]).resolve()
  stat = executable.stat()
  return _executable_fingerprint(command, str(executable), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=8)
def _executable_fingerprint(command: tuple[str, ...], executable: str, size: int, mtime_ns: int) -> str:
  version = subprocess.run([*command, "--version"], text=True, capture_output=True, check=True).stdout
  parts = [command, executable, size, mtime_ns, version, _native_macros(command)]
  return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def cache_root() -> Path:
  """Return the JIT cache directory: ``SCALY_CACHE_DIR``, else ``$XDG_CACHE_HOME/scaly/jit``, else ``~/.cache/scaly/jit``."""
  override = env_path("SCALY_CACHE_DIR")
  if override is not None:
    return override
  xdg = env_path("XDG_CACHE_HOME")
  base = xdg if xdg is not None else Path.home() / ".cache"
  return base / "scaly" / "jit"


def find_c_compiler() -> Compiler | None:
  """Find the C compiler the JIT uses: ``SCALY_CC``, else ``zig cc`` from the ``ziglang`` package, else ``CC``, else ``cc`` on ``PATH``.

  Returns the compiler command and which of those four supplied it, or ``None`` if nothing is found.
  """
  override = os.environ.get("SCALY_CC")
  if override:
    found = shutil.which(override)
    return Compiler((found,), "SCALY_CC") if found is not None else None
  spec = importlib.util.find_spec("ziglang")
  if spec is not None and spec.origin is not None and (zig := shutil.which("zig", path=str(Path(spec.origin).parent))) is not None:
    return Compiler((zig, "cc"), "ziglang")
  override = os.environ.get("CC")
  if override:
    found = shutil.which(override)
    return Compiler((found,), "CC") if found is not None else None
  found = shutil.which("cc")
  return Compiler((found,), "PATH") if found is not None else None


def _format_report() -> str:
  paths = solver_paths(required=False)
  compiler = find_c_compiler()
  lines = ["Scaly native toolchain", f"  cache root: {cache_root()}"]
  lines.append(f"  cc: {shlex.join(compiler.command)} ({compiler.source})" if compiler is not None else "  cc: <missing>")
  if compiler is not None:
    lines.extend(("  native build recipe:", native_recipe(compiler.command).comment("module.c").rstrip()))
  lines += [
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
  return "\n".join(lines)


def main() -> None:
  print(_format_report())


if __name__ == "__main__":
  main()
