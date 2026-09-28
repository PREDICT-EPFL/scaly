"""Terminal ingredients: the LQR gain and cost, ellipsoidal and maximal positively invariant sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..ir.expr import Expr
from ..opt.problem import Bounded, bounded
from .polytope import Polytope

__all__ = ["Ellipsoid", "largest_ellipsoid", "lqr", "max_invariant_set"]


def lqr(a: Any, b: Any, q: Any, r: Any) -> tuple[np.ndarray, np.ndarray]:
  """``(K, P)``: the infinite-horizon LQR of ``x+ = A x + B u`` with stage cost ``x'Qx + u'Ru``, the
  control law ``u = K x`` and its cost to go ``x'Px``, from the discrete algebraic Riccati equation
  (SciPy's solver), whose residual is checked. ``x'Px`` is the terminal cost that makes an
  unconstrained MPC of any horizon return ``K x``."""
  from scipy.linalg import solve_discrete_are

  a, b = np.atleast_2d(np.asarray(a, dtype=np.float64)), np.asarray(b, dtype=np.float64).reshape(np.shape(a)[0], -1)
  q, r = np.atleast_2d(np.asarray(q, dtype=np.float64)), np.atleast_2d(np.asarray(r, dtype=np.float64))
  p = solve_discrete_are(a, b, q, r)
  gain = -np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
  residual = a.T @ p @ a - p + q + a.T @ p @ b @ gain
  if not np.abs(residual).max() <= 1e-8 * max(1.0, np.abs(p).max()):
    raise ValueError(f"the Riccati equation's residual is {np.abs(residual).max():.1e}: (A, B) may not be stabilizable, or Q not detectable")
  return gain, (p + p.T) / 2


def max_invariant_set(closed_loop: Any, constraints: Polytope, *, max_iter: int = 200, tol: float = 1e-9) -> Polytope:
  """The maximal positively invariant set of ``x+ = A_K x`` in ``constraints``: the states whose
  whole trajectory stays in the polytope, ``{x : H A_K^t x <= h, t = 0, 1, ...}``.

  Gilbert and Tan (1991): the rows of ``t + 1`` are added until every one of them is redundant given
  those before, one linear program per row, which proves the rest redundant too; the result has its
  redundant rows removed. For a stable ``A_K`` and a bounded set with the origin inside, this ends;
  otherwise ``max_iter`` steps raise. For an LQR terminal set, ``constraints`` is the state set
  intersected with the input set's ``preimage(K)``."""
  a_k = np.atleast_2d(np.asarray(closed_loop, dtype=np.float64))
  if a_k.shape != (constraints.dim, constraints.dim):
    raise ValueError(f"the closed loop must be {constraints.dim}x{constraints.dim}, got {a_k.shape}")
  current, power = constraints, a_k
  for _ in range(max_iter):
    rows = constraints.H @ power
    if all(current.support(row) <= bound + tol * max(1.0, abs(bound)) for row, bound in zip(rows, constraints.h, strict=True)):
      return current.remove_redundancy()
    current = current.intersect(Polytope(rows, constraints.h))
    power = power @ a_k
  raise ValueError(
    f"the set is not finitely determined within {max_iter} steps: is the closed loop stable and the set bounded, with the origin inside?"
  )


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

  def constraints(self, x: Expr) -> tuple[list[Expr], list[Bounded]]:
    """As an OCP's terminal set: ``x'Px <= alpha``."""
    return [], [bounded((x @ (Expr.const(self.P) @ x)).reshape((1,)), hi=float(self.alpha), name="terminal_ellipsoid")]


def largest_ellipsoid(p: Any, polytope: Polytope) -> Ellipsoid:
  """The largest level set ``{x : x'Px <= alpha}`` inside a polytope with the origin in its interior:
  ``alpha = min_i h_i^2 / (H_i P^{-1} H_i')``, where each row's hyperplane touches the level set. With
  ``P`` from ``lqr`` and the LQR constraint polytope, the result is an invariant terminal set: every
  level set of the cost to go is invariant under the closed loop."""
  p = np.atleast_2d(np.asarray(p, dtype=np.float64))
  if np.any(polytope.h <= 0):
    raise ValueError("the origin must lie in the polytope's interior: every h_i > 0")
  inverse = np.linalg.inv(p)
  reach = np.einsum("ij,jk,ik->i", polytope.H, inverse, polytope.H)
  return Ellipsoid(p, float(np.min(polytope.h**2 / reach)))
