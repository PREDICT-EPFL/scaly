"""Select a solver plugin for a typed backend-free problem."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..function import Function
from ..ir.expr import Expr
from .nlp import build_nlp
from .qp import build_qp
from .problem import Problem
from .registry import get_backend, require_backend


def solver[SV, NV, SP, NP](
  problem: Problem[SV, NV, SP, NP],
  backend: str,
  /,
  *,
  name: str | None = None,
  options: dict[str, Any] | None = None,
) -> Function[
  tuple[SV, SV, Expr, Expr, SP],
  tuple[NV, NV, np.ndarray, np.ndarray, NP],
  tuple[SV, SV, Expr, Expr],
  tuple[NV, NV, np.ndarray, np.ndarray],
]:
  """Build a typed plain Function that solves one backend-free problem."""
  selected = get_backend(backend)
  solver_name = name or f"{problem.name}_{backend}"
  if selected.kind == "nlp":
    nlp_backend = require_backend(backend, "nlp")
    return build_nlp(problem, nlp_backend, name=solver_name, options=options)
  if selected.kind == "qp":
    return build_qp(problem, require_backend(backend, "qp"), name=solver_name, options=options)
  raise ValueError(f"solver plugin {backend!r} has unknown kind {selected.kind!r}")
