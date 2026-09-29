"""Ellipsoids ``{x : x'Px <= alpha}``: membership, and the quadratic constraint that keeps a point inside."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..ir.expr import Expr
from .polytope import Constraint

__all__ = ["Ellipsoid"]


@dataclass(frozen=True)
class Ellipsoid:
  """The set ``{x : x'Px <= alpha}``. As an OCP's terminal set it constrains the last state by one
  quadratic inequality, which an NLP solver takes and a QP solver does not: for a QP, use a
  ``Polytope``."""

  P: np.ndarray
  alpha: float

  def contains(self, x: Any, tol: float = 1e-9) -> np.ndarray:
    """Whether ``x`` (one point, or one per row) lies inside: a bool array, 0-d for one point."""
    x = np.asarray(x, dtype=np.float64)
    return np.asarray(np.einsum("...i,ij,...j->...", x, self.P, x) <= self.alpha * (1 + tol))

  def constraints(self, x: Expr) -> tuple[Constraint, ...]:
    """``x`` in the set as constraints: ``x'Px <= alpha``, one row."""
    return (((x @ (Expr.const(self.P) @ x)).reshape((1,)), None, float(self.alpha)),)
