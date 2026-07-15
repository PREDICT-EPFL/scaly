from __future__ import annotations

from pathlib import Path


def include_dir() -> Path:
  return Path(__file__).resolve().parent / "include"


def lib_dir() -> Path:
  return Path(__file__).resolve().parent / "lib"


class _Backend:
  """Plugin metadata only — solves run through alloy's generated C wrapper."""

  name = "piqp"
  kind = "qp"
  protocol_version = 1
  lib_stem = "piqpc"
  link_flags = ("-lpiqpc",)
  header = "piqp/piqp.h"

  include_dir = staticmethod(include_dir)
  lib_dir = staticmethod(lib_dir)


BACKEND = _Backend()
