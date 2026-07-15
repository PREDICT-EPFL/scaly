from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from alloy.codegen.solver_c import SolverWrapperCtx
  from alloy.function import Function


def include_dir() -> Path:
  return Path(__file__).resolve().parent / "include"


def lib_dir() -> Path:
  return Path(__file__).resolve().parent / "lib"


class _Backend:
  """IPOPT solver plugin: vendored lib/header metadata + the C wrapper
  template (``codegen.py``). Solves run through alloy's generated C wrapper."""

  name = "ipopt"
  kind = "nlp"
  protocol_version = 2
  lib_stem = "ipopt"
  link_flags = ("-lipopt",)
  header = "coin-or/IpStdCInterface.h"

  include_dir = staticmethod(include_dir)
  lib_dir = staticmethod(lib_dir)

  def render_wrapper(self, fun: Function, ctx: SolverWrapperCtx) -> list[str]:
    from .codegen import render_wrapper

    return render_wrapper(fun, ctx)


BACKEND = _Backend()
