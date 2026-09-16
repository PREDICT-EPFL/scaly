"""Native C compiler discovery, JIT cache location, and the toolchain diagnostics report.

Scaly's normal Python workflow is controlled by a small set of environment variables, registered
in ``utils/env.py``. Solver library and header discovery lives in ``solvers/paths.py``; this
module reads it for the report and nothing else depends on that direction.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..solvers.paths import SolverLibraryError, backend_compile_flags, solver_discoverable, solver_loadable, solver_paths
from ..utils.env import env_path


@dataclass(frozen=True, slots=True)
class Compiler:
  cc: str
  source: str


def cache_root() -> Path:
  override = env_path("SCALY_CACHE_DIR")
  if override is not None:
    return override
  xdg = env_path("XDG_CACHE_HOME")
  base = xdg if xdg is not None else Path.home() / ".cache"
  return base / "scaly" / "jit"


def find_c_compiler() -> Compiler | None:
  for key in ("SCALY_CC", "CC"):
    override = os.environ.get(key)
    if override:
      found = shutil.which(override)
      return Compiler(found, key) if found is not None else None
  found = shutil.which("cc")
  return Compiler(found, "PATH") if found is not None else None


def _format_report() -> str:
  paths = solver_paths(required=False)
  compiler = find_c_compiler()
  lines = ["Scaly native toolchain", f"  cache root: {cache_root()}"]
  lines.append(f"  cc: {compiler.cc} ({compiler.source})" if compiler is not None else "  cc: <missing>")
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
