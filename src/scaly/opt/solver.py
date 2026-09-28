"""``solver``: the Function that solves an opt problem, built by the method asked for or the first that supports it."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..function import ConcreteFunction
from ..ir.expr import Expr
from .method import REGISTRY, Info, short_name
from .problem import NLP


def solver[SV, NV, SP, NP](
  problem: NLP[SV, NV, SP, NP],
  method: Any = "auto",
  /,
  *,
  name: str | None = None,
) -> ConcreteFunction[
  [SV, SV, Expr, Expr, SP],
  [NV, NV, np.ndarray, np.ndarray, NP],
  tuple[SV, SV, Expr, Expr, Info],
  tuple[NV, NV, np.ndarray, np.ndarray, Info],
]:
  """The Function that solves ``problem``: its warm start (the variables, their bound multipliers,
  the equality and inequality multipliers) and its parameters in, the solution and its multipliers
  out, whichever method builds it.

  ``method`` is a method with its options (``sc.opt.PIQP(options={"eps_abs": 1e-9})``), a method's
  name (``"ipopt"``, with its default options), or ``"auto"``, the first installed method that
  supports the problem: a QP method for a quadratic problem, then the NLP ones. A method that does
  not fit the problem raises: ``NotQuadratic`` from a QP method, for one."""
  chosen = REGISTRY.auto(problem) if method == "auto" else REGISTRY.get(method)() if isinstance(method, str) else method
  if not isinstance(problem, chosen.problem):
    raise TypeError(f"{chosen.name} solves {chosen.problem.__name__}, not {type(problem).__name__}")
  return chosen.build(problem, name=name or f"{problem.name}_{short_name(chosen.name)}")
