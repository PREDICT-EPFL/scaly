from __future__ import annotations

from typing import TYPE_CHECKING

from scaly_piqp import include_dir, lib_dir

from .external import external_nlp

if TYPE_CHECKING:
  from scaly.codegen.solver import SolverWrapperCtx
  from scaly.function.concrete import ConcreteFunction


class _Backend:
  name = "sqp"
  kind = "nlp"
  hess_triangle = "upper"
  protocol_version = 7
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
