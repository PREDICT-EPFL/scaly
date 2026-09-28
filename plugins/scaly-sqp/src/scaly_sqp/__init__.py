from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from scaly.ext import require_ext_api
from scaly.opt.external import External
from scaly_piqp import include_dir, lib_dir

from .external import external_nlp

require_ext_api(1, "scaly-sqp")

if TYPE_CHECKING:
  from scaly.opt.external import SolverWrapperCtx
  from scaly.function import ConcreteFunction


@dataclass(frozen=True)
class SQP(External):
  """Scaly's own SQP method: sequential quadratic programming with PIQP solving each QP, generated
  as C (``sqp.c.jinja``). ``options`` are its own (``max_iter``, ``tol``, ``hessian``, ``merit``, ...;
  ``codegen.py`` lists them)."""

  name = "opt.sqp"
  kind = "nlp"
  hess_triangle = "upper"
  lib_stem = "piqpc"
  link_flags = ("-lpiqpc",)
  header = "piqp/piqp.h"

  include_dir = staticmethod(include_dir)
  lib_dir = staticmethod(lib_dir)

  def check_options(self) -> None:
    from .codegen import _option_map

    _option_map(tuple(self.options.items()))

  def render_wrapper(self, fun: ConcreteFunction, ctx: SolverWrapperCtx) -> list[str]:
    from .codegen import render_wrapper

    return render_wrapper(fun, ctx)


__all__ = ["SQP", "external_nlp"]
