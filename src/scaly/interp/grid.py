"""The axes of an interpolant: knot vectors with their search partitions and extrapolation, the interval searches built as expressions, and the per-axis B-spline tables."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy import sparse

from ..ir.expr import Expr, cast, logical_not, maximum, minimum, stack, take, where
from ..ir.types import dtypes

type Search = Literal["uniform", "count", "binary"]
type Extrap = Literal["extend", "linear", "clamp", "periodic", "fill"]
type Side = Literal["right", "left"]

SEARCHES = ("uniform", "count", "binary")
EXTRAPS = ("extend", "linear", "clamp", "periodic", "fill")


def check_sites(values: object, what: str, *, minimum_points: int = 2) -> np.ndarray:
  """``values`` as a float vector that is finite and strictly increasing, with at least
  ``minimum_points`` entries."""
  sites = np.asarray(values, dtype=np.float64)
  if sites.ndim != 1:
    raise ValueError(f"{what} must be a vector, got shape {sites.shape}")
  if sites.size < minimum_points:
    raise ValueError(f"{what} needs at least {minimum_points} points, got {sites.size}")
  if not np.all(np.isfinite(sites)):
    raise ValueError(f"{what} must be finite")
  if np.any(np.diff(sites) <= 0):
    raise ValueError(f"{what} must be strictly increasing")
  return sites


def derivative_matrix(knots: np.ndarray, degree: int) -> sparse.csr_array:
  """The map from the coefficients of a degree-``k`` spline on ``knots`` to those of its derivative,
  degree ``k - 1`` on ``knots[1:-1]``: ``c'_i = k (c_{i+1} - c_i) / (t_{i+k+1} - t_{i+1})`` (de Boor),
  a zero where the denominator is (a knot of multiplicity ``k + 1``)."""
  n = knots.size - degree - 1
  span = knots[degree + 1 : degree + n] - knots[1:n]
  w = np.divide(degree, span, out=np.zeros(n - 1), where=span > 0)
  rows = np.repeat(np.arange(n - 1), 2)
  cols = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).reshape(-1)
  return sparse.csr_array((np.stack([-w, w], axis=1).reshape(-1), (rows, cols)), shape=(n - 1, n))


def basis_derivatives(knots: np.ndarray, degree: int, points: np.ndarray, order: int) -> sparse.csr_array:
  """The ``order``-th derivative of every B-spline basis function at ``points`` (inside the base
  interval), a ``(points, n)`` matrix: a design matrix of degree ``k - order`` times the differencing
  maps. At a knot the piece to the right is taken."""
  from scipy.interpolate import BSpline

  maps = []
  t, k = knots, degree
  for _ in range(order):
    maps.append(derivative_matrix(t, k))
    t, k = t[1:-1], k - 1
  out = BSpline.design_matrix(points, t, k).tocsr()
  for m in reversed(maps):
    out = out @ m
  return sparse.csr_array(out)


class Axis:
  """One axis of a tensor-product B-spline: its knot vector and degree, the partition searched at
  run time, and what happens outside.

  The partition ``edges`` runs from ``t[k]`` to ``t[n]`` and contains every distinct knot between, so
  each of its cells lies inside one polynomial piece; a fit may refine it with its data sites, which
  keeps a uniform data grid uniform when the knots are not (not-a-knot drops two). Cells are
  ``[e_j, e_{j+1})`` with the last one closed (``side="right"``, as ``searchsorted(side="right") - 1``),
  or ``(e_j, e_{j+1}]`` with the first closed (``side="left"``, for ``nearest``'s midpoints).
  """

  __slots__ = ("knots", "degree", "edges", "side", "extrap", "fill", "search", "_uniform", "_local", "_offsets")

  def __init__(
    self,
    knots: np.ndarray,
    degree: int,
    *,
    edges: np.ndarray | None = None,
    side: Side = "right",
    extrap: Extrap | None = None,
    fill: float = math.nan,
    search: Search | Literal["auto"] = "auto",
  ) -> None:
    knots = np.asarray(knots, dtype=np.float64)
    if not isinstance(degree, (int, np.integer)) or degree < 0:
      raise ValueError(f"degree must be a non-negative integer, got {degree!r}")
    degree = int(degree)
    n = knots.size - degree - 1
    if knots.ndim != 1 or n < 1 or not np.all(np.isfinite(knots)) or np.any(np.diff(knots) < 0):
      raise ValueError(f"knots must be a finite non-decreasing vector of at least {degree + 2} entries for degree {degree}")
    if knots[degree] >= knots[n]:
      raise ValueError("the base interval [t[k], t[n]] of the knots is empty")
    span = knots[degree : n + 1]
    breaks = span[np.concatenate([[True], np.diff(span) > 0])]  # sorted already: no need for unique's sort
    edges = breaks if edges is None else check_sites(edges, "the partition")
    found = np.minimum(np.searchsorted(edges, breaks), edges.size - 1)
    if edges[0] != breaks[0] or edges[-1] != breaks[-1] or not np.array_equal(edges[found], breaks):
      raise ValueError("the partition must span the base interval and contain every distinct knot in it")
    extrap = extrap if extrap is not None else ("clamp" if degree == 0 else "linear")
    if extrap not in EXTRAPS:
      raise ValueError(f"extrap must be one of {EXTRAPS}, got {extrap!r}")
    if extrap == "linear" and degree <= 1:
      extrap = "clamp" if degree == 0 else "extend"  # the same function, without the slope's extra work
    if side not in ("right", "left"):
      raise ValueError(f"side must be 'right' or 'left', got {side!r}")
    self.knots, self.degree, self.edges, self.side, self.extrap, self.fill = knots, degree, edges, side, extrap, float(fill)
    self._uniform = uniform_step(edges)
    self._local: np.ndarray | None = None
    self._offsets: np.ndarray | None = None
    if search == "auto":
      # Measured (perf_2026_09_28_interp/bench_eval.py): the binary search beats the count from two
      # cells up, one point per call; the count stays for an explicit choice.
      search = "uniform" if self._uniform is not None else "binary"
    if search not in SEARCHES:
      raise ValueError(f"search must be 'auto' or one of {SEARCHES}, got {search!r}")
    if search == "uniform" and self._uniform is None:
      raise ValueError("search='uniform' needs a uniform partition")
    self.search: Search = search

  @property
  def n(self) -> int:
    """The number of coefficients along this axis."""
    return self.knots.size - self.degree - 1

  @property
  def cells(self) -> int:
    return self.edges.size - 1

  @property
  def lo(self) -> float:
    return float(self.edges[0])

  @property
  def hi(self) -> float:
    return float(self.edges[-1])

  @property
  def centers(self) -> np.ndarray:
    """The midpoint of each cell: the local polynomials are expanded there, where the powers of
    ``s`` stay below half the cell width, which halves the cancellation a left-edge expansion has."""
    return 0.5 * (self.edges[:-1] + self.edges[1:])

  @property
  def offsets(self) -> np.ndarray:
    """The first non-zero basis function on each cell: ``span - k``, where the span ``mu`` has
    ``t[mu] <= e_j < t[mu + 1]``."""
    if self._offsets is None:
      span = np.searchsorted(self.knots, self.edges[:-1], side="right") - 1
      self._offsets = np.minimum(span, self.n - 1) - self.degree
    return self._offsets

  @property
  def local(self) -> np.ndarray:
    """``(cells, k + 1, k + 1)``: entry ``[j, a, m]`` is the coefficient of ``s^m``, ``s = x - c_j``
    for the cell's center ``c_j``, in the ``a``-th non-zero basis function of cell ``j``."""
    if self._local is None:
      k, centers = self.degree, self.centers
      rows = np.arange(centers.size)[:, None]
      cols = self.offsets[:, None] + np.arange(k + 1)[None, :]
      rows, cols = np.broadcast_arrays(rows, cols)
      self._local = np.stack(
        [
          np.asarray(basis_derivatives(self.knots, k, centers, m)[rows.ravel(), cols.ravel()]).reshape(rows.shape) / math.factorial(m)
          for m in range(k + 1)
        ],
        axis=-1,
      )
    return self._local

  def taylor(self, coeffs: np.ndarray) -> np.ndarray:
    """The piecewise-polynomial form along this axis of ``coeffs`` (coefficients on the first axis):
    ``(cells, k + 1, *rest)``, entry ``[j, m]`` the coefficient of ``s^m``, ``s = x - c_j``, on cell ``j``."""
    flat = coeffs.reshape(coeffs.shape[0], -1)
    out = np.stack([basis_derivatives(self.knots, self.degree, self.centers, m) @ flat / math.factorial(m) for m in range(self.degree + 1)], axis=1)
    return out.reshape(self.cells, self.degree + 1, *coeffs.shape[1:])

  def wrap(self, x: Expr) -> Expr:
    """``x`` moved by whole periods into ``[lo, hi]``, for ``extrap="periodic"``."""
    period = self.hi - self.lo
    return x - period * ((x - self.lo) * (1.0 / period)).floor()

  def cell(self, x: Expr) -> Expr:
    """The cell of ``x`` as a float index in ``[0, cells - 1]``, elementwise: a point left of the
    partition gets the first cell, one right of it the last, and NaN the first (the comparisons fail)."""
    if self.cells == 1:
      return Expr.const(np.zeros(x.shape))
    if self.search == "uniform":
      return self._uniform_cell(x)
    if self.search == "count":
      return self._count_cell(x)
    return self._binary_cell(x)

  def _above(self, x: Expr, edge: Expr) -> Expr:
    """Whether ``x`` lies at or past ``edge`` for the axis's continuity."""
    return x >= edge if self.side == "right" else x > edge

  def _uniform_cell(self, x: Expr) -> Expr:
    # floor((x - e_0) / h) lands within one cell of the answer (see uniform_step); one compare
    # against the cell's own edges on each side settles it. The clamp comes before the cast:
    # fmin/fmax send NaN to a bound and ±inf to an end.
    assert self._uniform is not None
    t0, inv_h = self._uniform
    last = float(self.cells - 1)
    j0 = minimum(maximum(((x - t0) * inv_h).floor(), 0.0), last)
    i0 = cast(j0, dtypes.int64)
    both = gather(self.edges, stack([i0, i0 + 1], axis=len(x.shape)))
    lo, hi = (both[(..., 0)], both[(..., 1)])
    down = logical_not(self._above(x, lo)) & (j0 > 0.0)
    up = self._above(x, hi) & (j0 < last)
    return j0 - cast(down, dtypes.float64) + cast(up, dtypes.float64)

  def _count_cell(self, x: Expr) -> Expr:
    inner = Expr.const(self.edges[1:-1])
    if not x.shape:
      return cast(self._above(x, inner), dtypes.float64).sum()
    flat = x.reshape((x.size, 1))
    hits = cast(self._above(flat, inner.reshape((1, inner.size))), dtypes.float64)
    return (hits @ Expr.const(np.ones(inner.size))).reshape(x.shape)

  def _binary_cell(self, x: Expr) -> Expr:
    # Halvings on the left edges padded to a power of two with NaN, which every compare fails (+inf
    # would let x = +inf through): the index never passes the last cell, and a NaN x stays at the first.
    levels = math.ceil(math.log2(self.cells))
    padded = np.full(1 << levels, np.nan)
    padded[1 : self.cells] = self.edges[1:-1]
    lo = Expr.const(np.zeros(x.shape, dtype=np.int64), dtype=dtypes.int64)
    for m in reversed(range(levels)):
      cand = lo + Expr.const(np.full(x.shape, 1 << m, dtype=np.int64), dtype=dtypes.int64)
      lo = where(self._above(x, gather(padded, cand)), cand, lo)
    return cast(lo, dtypes.float64)


def uniform_step(edges: np.ndarray) -> tuple[float, float] | None:
  """``(e_0, 1 / h)`` when the uniform search suits ``edges``, else ``None``: when every edge lies
  within a quarter cell of ``e_0 + j h``.

  Then the estimate ``floor((x - e_0) / h)`` is within one cell of the true one for every ``x`` (an
  edge off by ``d < h`` moves the quotient by less than one), and the one-step correction against
  the cell's own edges makes the search exact, at either side's ties. That holds for any ``d < h``;
  the quarter is where a partition stops being called uniform.
  """
  cells = edges.size - 1
  if cells < 2:
    return None
  h = (edges[-1] - edges[0]) / cells
  if np.max(np.abs(edges - (edges[0] + h * np.arange(edges.size)))) > 0.25 * h:
    return None
  return float(edges[0]), float(1.0 / h)


def gather(table: np.ndarray, index: Expr) -> Expr:
  """``table[index]`` for a constant vector and an ``int64`` index of any shape, in range."""
  flat = index.reshape((index.size,)) if index.shape != (index.size,) else index
  out = take(Expr.const(np.ascontiguousarray(table, dtype=np.float64)), flat, in_range=True)
  return out.reshape(index.shape) if index.shape != (index.size,) else out
