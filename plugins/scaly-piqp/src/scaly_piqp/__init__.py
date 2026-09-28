from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from scaly.ext import require_ext_api

require_ext_api(1, "scaly-piqp")

if TYPE_CHECKING:
  from scaly.solvers.wrapper import SolverWrapperCtx
  from scaly.function import ConcreteFunction


def include_dir() -> Path:
  return Path(__file__).resolve().parent / "include"


def lib_dir() -> Path:
  return Path(__file__).resolve().parent / "lib"


class _Backend:
  """PIQP solver plugin: vendored lib/header metadata + the C wrapper
  template (``codegen.py``). Solves run through scaly's generated C wrapper."""

  name = "piqp"
  kind = "qp"
  protocol_version = 8
  lib_stem = "piqpc"
  link_flags = ("-lpiqpc",)
  header = "piqp/piqp.h"

  include_dir = staticmethod(include_dir)
  lib_dir = staticmethod(lib_dir)

  def render_wrapper(self, fun: ConcreteFunction, ctx: SolverWrapperCtx) -> list[str]:
    from .codegen import render_wrapper

    return render_wrapper(fun, ctx)


BACKEND = _Backend()
