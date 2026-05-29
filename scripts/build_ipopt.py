"""Standalone driver to build the vendored IPOPT stack (METIS -> MUMPS -> IPOPT).

Reuses the build functions from hatch_build.py with a minimal fake hook so we can
run just the IPOPT stack without a full wheel build. Run with:

    uv run --with hatchling python scripts/build_ipopt.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import hatch_build  # noqa: E402


class _App:
  def display_info(self, msg: str) -> None:
    print(f"[ipopt-build] {msg}", flush=True)


class _Hook:
  app = _App()
  root = str(ROOT)


def main() -> None:
  third_party = ROOT / "third_party"
  lib_dir = ROOT / "src" / "alloy" / "lib"
  include_dir = ROOT / "src" / "alloy" / "include"
  hatch_build._build_ipopt_stack(_Hook(), third_party, lib_dir, include_dir)
  print("[ipopt-build] DONE", flush=True)


if __name__ == "__main__":
  main()
