"""Polytopes in halfspace form and the linear programs on them (SciPy's HiGHS)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..ir.expr import Expr

__all__ = ["Constraint", "Polytope"]

type Constraint = tuple[Expr, Expr | float | None, Expr | float | None]
"""``(expr, lo, hi)``: the constraint ``lo <= expr <= hi`` entry by entry, a side ``None`` when it is
absent. What a set gives an optimal control problem, which turns it into its own constraint group."""


def _linprog(c: np.ndarray, h_mat: np.ndarray, h: np.ndarray) -> Any:
  """``min c.x`` over ``H x <= h``, ``x`` free, by HiGHS. SciPy's optimizer is imported here, not at
  the top: ``import scaly`` should not pay for it."""
  from scipy.optimize import linprog

  return linprog(c, A_ub=h_mat, b_ub=h, bounds=[(None, None)] * c.size, method="highs")


class Polytope:
  """The set ``{x : H x <= h}``.

  Built from rows (``Polytope(H, h)``) or a box (``Polytope.box(lo, hi)``), combined by
  ``intersect`` and ``preimage``, and trimmed by ``remove_redundancy``. Every question that needs an
  optimization (``support``, ``is_empty``, ``chebyshev_center``, redundancy) is a linear program. As an
  OCP's terminal set it constrains the last state, ``H x_N <= h``, rows a QP solver takes.
  """

  def __init__(self, h_mat: Any, h: Any) -> None:
    h_mat, h = np.atleast_2d(np.asarray(h_mat, dtype=np.float64)), np.ravel(np.asarray(h, dtype=np.float64))
    if h_mat.shape[0] != h.size:
      raise ValueError(f"a polytope needs one bound per row: H is {h_mat.shape}, h has {h.size}")
    self.H, self.h = h_mat, h

  @classmethod
  def box(cls, lo: Any, hi: Any) -> Polytope:
    """``{x : lo <= x <= hi}``; an infinite side gives no row."""
    lo, hi = np.ravel(np.asarray(lo, dtype=np.float64)), np.ravel(np.asarray(hi, dtype=np.float64))
    lo, hi = np.broadcast_arrays(lo, hi)
    eye = np.eye(lo.size)
    upper, lower = np.isfinite(hi), np.isfinite(lo)
    return cls(np.vstack([eye[upper], -eye[lower]]), np.concatenate([hi[upper], -lo[lower]]))

  @property
  def dim(self) -> int:
    """The dimension of the space."""
    return self.H.shape[1]

  def __repr__(self) -> str:
    return f"Polytope({self.H.shape[0]} rows in {self.dim} dimensions)"

  def intersect(self, other: Polytope) -> Polytope:
    """Both sets' rows."""
    if other.dim != self.dim:
      raise ValueError(f"polytopes in {self.dim} and {other.dim} dimensions do not intersect")
    return Polytope(np.vstack([self.H, other.H]), np.concatenate([self.h, other.h]))

  def preimage(self, matrix: Any) -> Polytope:
    """``{x : M x in self}``: the rows ``H M``. The set of states whose image under ``M`` (a
    closed-loop matrix, or a gain ``K`` for control constraints) lies in the polytope."""
    matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
    if matrix.shape[0] != self.dim:
      raise ValueError(f"the preimage of a polytope in {self.dim} dimensions takes a matrix with {self.dim} rows, got {matrix.shape}")
    return Polytope(self.H @ matrix, self.h)

  def contains(self, x: Any, tol: float = 1e-9) -> np.ndarray:
    """Whether ``x`` (one point, or one per row) satisfies every row to ``tol``: a bool array, 0-d for one point."""
    x = np.asarray(x, dtype=np.float64)
    return np.asarray(np.all(x @ self.H.T <= self.h + tol, axis=-1))

  def support(self, direction: Any) -> float:
    """``max d.x`` over the set; ``inf`` when unbounded in that direction."""
    result = _linprog(-np.asarray(direction, dtype=np.float64), self.H, self.h)
    if result.status == 3:
      return np.inf
    if result.status == 2:
      return -np.inf  # empty
    return float(-result.fun)

  def is_empty(self) -> bool:
    """Whether no point satisfies every row."""
    return _linprog(np.zeros(self.dim), self.H, self.h).status == 2

  def chebyshev_center(self) -> tuple[np.ndarray, float]:
    """``(center, radius)`` of the largest ball inside the set: ``max r`` over ``H x + r |H_i| <= h``."""
    norms = np.linalg.norm(self.H, axis=1)
    c = np.zeros(self.dim + 1)
    c[-1] = -1.0
    result = _linprog(c, np.hstack([self.H, norms[:, None]]), self.h)
    if result.status != 0 or result.x[-1] < 0:  # a negative radius: no point satisfies every row
      raise ValueError(f"no Chebyshev center: the polytope is empty or unbounded (HiGHS status {result.status})")
    return result.x[:-1], float(result.x[-1])

  def remove_redundancy(self, tol: float = 1e-9) -> Polytope:
    """The same set with only the rows that bound it: row ``i`` goes when ``max H_i x`` over the other
    kept rows (and row ``i`` itself, to keep the program bounded) is within ``tol`` of ``h_i``, one
    linear program per row. Zero rows go too, and duplicates are kept once."""
    norms = np.linalg.norm(self.H, axis=1)
    scale = np.where(norms > 0, norms, 1.0)
    h_mat, h = self.H / scale[:, None], self.h / scale
    keep = [i for i in range(h.size) if norms[i] > 0]
    if any(norms[i] == 0 and h[i] < -tol for i in range(h.size)):
      return Polytope(np.zeros((1, self.dim)), -np.ones(1))  # 0 <= negative: empty
    i = 0
    while i < len(keep):
      row = keep[i]
      others = [k for k in keep if k != row]
      relaxed_h = np.concatenate([h[others], [h[row] + 1.0]])
      result = _linprog(-h_mat[row], np.vstack([h_mat[others], h_mat[row][None, :]]), relaxed_h)
      if result.status == 0 and -result.fun <= h[row] + tol:
        keep.pop(i)
      else:
        i += 1
    return Polytope(self.H[keep], self.h[keep])

  def vertices(self) -> np.ndarray:
    """The vertices of a bounded, nonempty polytope, for plotting in two or three dimensions."""
    from scipy.spatial import ConvexHull, HalfspaceIntersection

    center, radius = self.chebyshev_center()
    if not radius > 0:
      raise ValueError("the polytope has no interior")
    points = HalfspaceIntersection(np.hstack([self.H, -self.h[:, None]]), center).intersections
    if self.dim == 1:
      return np.unique(points, axis=0)
    return points[ConvexHull(points).vertices]

  def constraints(self, x: Expr) -> tuple[Constraint, ...]:
    """``x`` in the set as constraints: ``H x <= h``, one group of rows."""
    return ((Expr.const(self.H) @ x, None, Expr.const(self.h)),)
