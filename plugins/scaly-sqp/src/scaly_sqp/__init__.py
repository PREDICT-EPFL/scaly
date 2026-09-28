from __future__ import annotations

from typing import TYPE_CHECKING

from scaly.ext import require_ext_api
from scaly_piqp import include_dir, lib_dir

from .external import external_nlp

require_ext_api(1, "scaly-sqp")

if TYPE_CHECKING:
  from scaly.solvers.wrapper import SolverWrapperCtx
  from scaly.function import ConcreteFunction


class _Backend:
  name = "sqp"
  kind = "nlp"
  hess_triangle = "upper"
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

__all__ = ["BACKEND", "external_nlp"]
