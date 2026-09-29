"""The methods of ``scaly.roots``: their registry over the ``scaly.methods`` entry points, the method API they implement and the ``Info`` they report."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from ..function.method import Info as MethodInfo
from ..function.method import Status, registry
from ..function.tree import L, Record
from ..ir.expr import Expr, isfinite, norm_inf, where
from ..ir.types import TensorType

METHOD_API = 1
"""The version of the method API of ``Root`` and ``LeastSquares`` every roots method implements:
bump it whenever what a method's ``build`` or ``iterate`` receives, or what it must return, changes."""


@dataclass(frozen=True)
class Info(MethodInfo):
  """What every roots solver returns beside the solution: ``status`` (a ``Status`` code), ``iter``,
  the iterations taken, and ``residual``, the largest entry of what the method drives to zero at the
  returned point: ``F`` for a ``Root``, the gradient ``J^T r`` of ``1/2 |r|^2`` for a
  ``LeastSquares``. All three are ``float64`` scalars that carry no derivative."""

  residual: Any

  @classmethod
  def tree(cls, prefix: str = "info:") -> Record:
    return Record(cls, **{f.name: L(prefix + f.name, TensorType((), diff=False)) for f in fields(cls)})


def status(converged: Expr, z: Expr, residual: Expr) -> Expr:
  """``OK`` where the test ``converged`` passed at a finite ``z``, else ``NUMERICS`` for a ``z`` or a
  residual that is not finite and ``MAX_ITER`` otherwise: the status of an iteration that stopped by
  its bound or its test. (A diverging iterate can pass a test relative to its own size.)"""
  finite = isfinite(norm_inf(z)) & isfinite(residual)
  return where(converged & finite, float(Status.OK), where(finite, float(Status.MAX_ITER), float(Status.NUMERICS)))


REGISTRY = registry("roots", preference=("newton", "newton_bisection", "levenberg_marquardt", "gauss_newton"))
"""Every roots method by short name (``"newton"``); ``auto`` tries Newton for a square system, then
the bracketing Newton for a scalar one with bounds, then the least-squares methods."""


__all__ = ["METHOD_API", "REGISTRY", "Info", "status"]
