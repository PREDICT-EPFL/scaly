"""Native C compiler discovery, JIT cache location, and the toolchain diagnostics report.

Scaly's normal Python workflow is controlled by a small set of environment variables, registered
in ``utils/env.py``. The report's further sections come from the packages that own them (solver
library discovery, say) through the ``scaly.toolchain_report`` entry points, so this module imports
none of them.
"""

from __future__ import annotations

import functools
import re
import os
import shutil
import subprocess
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path

from ..utils.env import env_path

REPORT_ENTRY_POINTS = "scaly.toolchain_report"
"""The entry-point group of the report's further sections: each names a function returning lines."""


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


def compiler_identity(cc: str) -> tuple[str, str]:
  """``cc``'s path and the first line of its ``--version`` banner, which names the compiler and its
  version (``Apple clang version 21.0.0 (clang-2100.1.1.101)``, ``gcc-15 (Homebrew GCC 15.2.0)
  15.2.0``). The JIT keys its cache on both, since clang and GCC 12 or later get the same flags and
  a compiler upgraded in place keeps its path. The banner is read once per process per compiler."""
  banner = _version_banner(cc)
  return cc, banner.splitlines()[0] if banner else ""


def is_gcc(cc: str) -> bool:
  """Whether ``cc`` is GCC, from its ``--version`` banner; ``cc`` is often GCC on Linux and clang on macOS."""
  return _is_gcc_banner(_version_banner(cc))


@functools.lru_cache(maxsize=8)
def _version_banner(cc: str) -> str:
  try:
    return subprocess.run([cc, "--version"], capture_output=True, text=True, timeout=30).stdout
  except (OSError, subprocess.SubprocessError):
    return ""


def _is_gcc_banner(banner: str) -> bool:
  return "free software foundation" in banner.lower()  # GCC's copyright line; clang's banner has none


def gcc_major(cc: str) -> int | None:
  """GCC's major version from its ``--version`` banner, or None for another compiler."""
  banner = _version_banner(cc)
  return _gcc_major_of(banner) if _is_gcc_banner(banner) else None


def _gcc_major_of(banner: str) -> int | None:
  """The major version on a GCC banner's first line: its last ``x.y[.z]`` (``gcc (GCC) 11.4.0``)."""
  found = re.findall(r"(\d+)\.\d+(?:\.\d+)?", banner.splitlines()[0] if banner else "")
  return int(found[-1]) if found else None


def _format_report() -> str:
  compiler = find_c_compiler()
  lines = ["Scaly native toolchain", f"  cache root: {cache_root()}"]
  lines.append(f"  cc: {compiler.cc} ({compiler.source})" if compiler is not None else "  cc: <missing>")
  for ep in sorted(entry_points(group=REPORT_ENTRY_POINTS), key=lambda ep: ep.name):
    lines += ep.load()()
  return "\n".join(lines)


def main() -> None:
  print(_format_report())


if __name__ == "__main__":
  main()
