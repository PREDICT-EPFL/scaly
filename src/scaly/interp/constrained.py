"""Least-squares spline fits under shape constraints (monotone, convex, bounded, pinned, periodic), solved as a quadratic program by PIQP when the graph is built."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from scipy import sparse

from ..function.model import ConcreteFunction
from ..ir.types import DType
from ..solvers.qp import qp_problem
from ..solvers.solver import solver
from ..solvers.stats import ScalySolveStatus
from .grid import Extrap, Search, basis_derivatives, check_sites, derivative_matrix
from .spline import BSpline, Strategy, _per_axis, design_matrix

type Direction = Literal["increasing", "decreasing"] | None
type Curvature = Literal["convex", "concave"] | None

_SOLVERS: dict[tuple[int, int, int], ConcreteFunction[Any, Any, Any, Any]] = {}


def constrained(
  x: np.ndarray | Sequence[float],
  y: np.ndarray | Sequence[float],
  *,
  degree: int | tuple[int, ...] = 3,
  knots: int | Sequence[float] | np.ndarray | tuple[int | Sequence[float] | np.ndarray, ...] = 16,
  weights: np.ndarray | Sequence[float] | None = None,
  monotone: Direction | tuple[Direction, ...] = None,
  convex: Curvature | tuple[Curvature, ...] = None,
  bounds: tuple[float, float] | None = None,
  equal: Sequence[tuple[Any, float, Any]] = (),
  periodic: bool = False,
  penalty: int = 2,
  lam: float = 0.0,
  extrap: Extrap | tuple[Extrap | None, ...] | None = None,
  fill: float = math.nan,
  search: Search | Literal["auto"] | tuple[Search | Literal["auto"], ...] = "auto",
  strategy: Strategy = "auto",
  dtype: DType | str = "float64",
  name: str = "interp",
) -> BSpline:
  """The B-spline closest to scattered data in least squares whose coefficients satisfy linear
  constraints that make its shape what the physics says: a quadratic program, solved by PIQP now.

  The constraints are sufficient conditions on the coefficients, exact for the spline everywhere,
  not only at the data:

  - ``monotone``: the spline's derivative along the axis, itself a B-spline, has coefficients of one
    sign (``"increasing"`` or ``"decreasing"``, per axis);
  - ``convex``: likewise its second derivative (``"convex"`` or ``"concave"``, per axis: convex
    along each line parallel to the axis, not jointly);
  - ``bounds``: ``lo <= c <= hi``, so ``lo <= f <= hi`` by the convex-hull property;
  - ``equal``: ``(point, value, order)`` pins a value or a derivative, ``order`` an integer in 1-D
    (0 for the value) or a tuple of per-axis orders;
  - ``periodic``: 1-D, the value and ``k - 1`` derivatives equal at the two ends.

  Args:
    x: the data sites, ``(m,)`` or ``(m, D)`` points.
    y: ``(m,)`` values.
    degree: ``k``, per axis or one for all.
    knots: per axis, the number of equal intervals spanning the data, or the interior knots; the
      ends are clamped.
    weights: of the data in the least-squares term.
    penalty, lam: ``lam`` times the squared ``penalty``-th differences of the coefficients along
      each axis is added to the objective; needed when the data leave some coefficient free.
    extrap, fill, search, strategy, dtype, name: as for ``BSpline``.

  A constraint the data do not press on is inactive, and the fit is then the plain penalized least
  squares. The solver is generated once per problem size and reused.
  """
  pts = np.asarray(x, dtype=np.float64)
  pts = pts[:, None] if pts.ndim == 1 else pts
  values = np.asarray(y, dtype=np.float64).reshape(-1)
  if pts.ndim != 2 or values.size != pts.shape[0]:
    raise ValueError(f"x gives {pts.shape[0]} points and y {values.size} values")
  if not (np.all(np.isfinite(pts)) and np.all(np.isfinite(values))):
    raise ValueError("x and y must be finite")
  ndim = pts.shape[1]
  degrees = _per_axis(degree, ndim, "degree")
  knot_vectors = []
  for d, (k, spec) in enumerate(zip(degrees, _per_axis(knots, ndim, "knots"), strict=True)):
    lo, hi = float(pts[:, d].min()), float(pts[:, d].max())
    inner = np.linspace(lo, hi, int(spec) + 1)[1:-1] if isinstance(spec, (int, np.integer)) else np.asarray(spec, dtype=np.float64)
    breaks = check_sites(np.concatenate([[lo], inner, [hi]]), f"the knots of axis {d}")
    knot_vectors.append(np.concatenate([np.full(k + 1, lo), breaks[1:-1], np.full(k + 1, hi)]))
  sizes = [t.size - k - 1 for t, k in zip(knot_vectors, degrees, strict=True)]
  n = math.prod(sizes)

  B = design_matrix(knot_vectors, degrees, pts)
  w = np.ones(values.size) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
  if w.size != values.size or np.any(w < 0):
    raise ValueError("weights must be one non-negative number per datum")
  # The objective is half the squared residual of ``M c - target``: the weighted data, then the
  # scaled differences, so the polish below can solve it without squaring its condition number.
  stacked, target = [sparse.diags(np.sqrt(w)) @ B], [np.sqrt(w) * values]
  if lam:
    for d, size in enumerate(sizes):
      diff = sparse.csr_array(np.diff(np.eye(size), n=penalty, axis=0))
      stacked.append(math.sqrt(lam) * _along_axis(diff, d, sizes))
      target.append(np.zeros(stacked[-1].shape[0]))
  M, rhs = sparse.vstack(stacked).toarray(), np.concatenate(target)
  hessian, linear = M.T @ M, -(M.T @ rhs)

  eq_rows: list[np.ndarray] = []
  eq_rhs: list[float] = []
  for point, value, order in equal:
    orders = (order,) if ndim == 1 else tuple(order)
    at = np.asarray(point, dtype=np.float64).reshape(ndim)
    eq_rows.append(_derivative_row(knot_vectors, degrees, at, orders))
    eq_rhs.append(float(value))
  if periodic:
    if ndim != 1:
      raise ValueError("periodic is 1-D")
    t, k = knot_vectors[0], degrees[0]
    for nu in range(k):
      ends = basis_derivatives(t, k, np.array([t[k], t[-k - 1]]), nu).toarray()
      eq_rows.append(ends[0] - ends[1])
      eq_rhs.append(0.0)

  rows, lower, upper = [], [], []
  for d, (direction, curvature) in enumerate(zip(_per_axis(monotone, ndim, "monotone"), _per_axis(convex, ndim, "convex"), strict=True)):
    t, k = knot_vectors[d], degrees[d]
    for want, order, positive in ((direction, 1, "increasing"), (curvature, 2, "convex")):
      if want is None:
        continue
      if want not in (positive, {"increasing": "decreasing", "convex": "concave"}[positive]):
        raise ValueError(f"{'monotone' if order == 1 else 'convex'} takes {positive!r} or its opposite, got {want!r}")
      if k < order:
        raise ValueError(f"a derivative of order {order} needs degree {order} or more on axis {d}")
      diff = derivative_matrix(t, k)
      if order == 2:
        diff = derivative_matrix(t[1:-1], k - 1) @ diff
      block = _along_axis(sparse.csr_array(diff), d, sizes).toarray()
      rows.append(block)
      sign = np.inf if want == positive else 0.0
      lower.append(np.zeros(block.shape[0]) if want == positive else np.full(block.shape[0], -np.inf))
      upper.append(np.full(block.shape[0], sign if want == positive else 0.0))
  if bounds is not None:
    lo, hi = bounds
    rows.append(np.eye(n))
    lower.append(np.full(n, float(lo)))
    upper.append(np.full(n, float(hi)))

  A = np.array(eq_rows).reshape(len(eq_rows), n)
  G = np.vstack(rows) if rows else np.zeros((0, n))
  g_lb = np.concatenate(lower) if lower else np.zeros(0)
  g_ub = np.concatenate(upper) if upper else np.zeros(0)
  solve = _qp_solver(n, A.shape[0], G.shape[0])
  params = ((hessian, linear), (A, np.array(eq_rhs)), (G, g_lb, g_ub))
  result = solve.numerical_call(np.zeros(n), np.zeros(n), np.zeros(A.shape[0]), np.zeros(G.shape[0]), params)
  status = solve.solver_stats().status
  if status not in (ScalySolveStatus.OK, ScalySolveStatus.ACCEPTABLE):
    raise ValueError(f"the constrained fit failed: {status.name.lower()} (infeasible constraints?)")
  coeffs = _polish(M, rhs, A, np.array(eq_rhs), G, g_lb, g_ub, np.asarray(result[0]), np.asarray(result[3])).reshape(sizes)
  return BSpline(
    tuple(knot_vectors) if ndim > 1 else knot_vectors[0],
    coeffs,
    degrees,
    extrap=extrap,
    fill=fill,
    search=search,
    strategy=strategy,
    dtype=dtype,
    name=name,
  )


def _polish(
  M: np.ndarray, target: np.ndarray, A: np.ndarray, b: np.ndarray, G: np.ndarray, lb: np.ndarray, ub: np.ndarray, x: np.ndarray, z: np.ndarray
) -> np.ndarray:
  """The interior point's answer refined to rounding: an interior-point method stops with every
  inactive constraint still pushing by its barrier, a few 1e-10 on a fit, so the least squares is
  solved once more with the constraints it found active held as equalities and the rest dropped.
  Kept only when it is feasible and each active constraint's multiplier has the sign of a bound
  that holds the fit back; otherwise ``x`` stands."""
  gx = G @ x
  slack = np.minimum(gx - lb, ub - gx)
  active = (np.abs(z) > slack) | (slack <= 1e-12 * (1.0 + np.abs(gx)))
  upper = active & (ub - gx < gx - lb)
  C = np.vstack([A, G[active]])
  d = np.concatenate([b, np.where(upper, ub, lb)[active]])
  if C.shape[0]:
    u, sigma, vt = np.linalg.svd(C)
    rank = int(np.sum(sigma > sigma[0] * max(C.shape) * np.finfo(np.float64).eps)) if sigma.size else 0
    base = vt[:rank].T @ ((u[:, :rank].T @ d) / sigma[:rank])
    free = vt[rank:].T
  else:
    base, free = np.zeros(x.size), np.eye(x.size)
  polished = base + free @ np.linalg.lstsq(M @ free, target - M @ base, rcond=None)[0] if free.shape[1] else base
  residual = M.T @ (target - M @ polished)  # the constraints' forces: C^T mult = residual
  mult = np.linalg.lstsq(C.T, residual, rcond=None)[0] if C.shape[0] else np.zeros(0)
  tol = 1e-9 * (1.0 + np.abs(polished).max())
  g = G @ polished
  pushes = mult[A.shape[0] :]
  if (
    np.all(np.abs(A @ polished - b) <= tol)
    and np.all(g >= lb - tol)
    and np.all(g <= ub + tol)
    and np.all(np.where(upper[active], pushes, -pushes) >= -tol * (1.0 + np.abs(mult).max(initial=0.0)))
  ):
    return polished
  return x


def _along_axis(matrix: sparse.csr_array, axis: int, sizes: Sequence[int]) -> sparse.csr_array:
  """``matrix`` acting on one axis of the coefficient tensor (C order), the identity on the rest."""
  before, after = math.prod(sizes[:axis]), math.prod(sizes[axis + 1 :])
  return sparse.csr_array(sparse.kron(sparse.kron(sparse.eye(before), matrix), sparse.eye(after)))


def _derivative_row(knots: Sequence[np.ndarray], degrees: Sequence[int], point: np.ndarray, orders: Sequence[int]) -> np.ndarray:
  """The row of the tensor-product basis's mixed derivative at one point."""
  row = np.ones(1)
  for t, k, x, nu in zip(knots, degrees, point, orders, strict=True):
    if not 0 <= nu <= k:
      raise ValueError(f"a derivative of order {nu} of a degree-{k} axis")
    row = np.kron(row, basis_derivatives(t, k, np.array([x]), int(nu)).toarray()[0])
  return row


def _qp_solver(n: int, n_eq: int, n_ineq: int) -> ConcreteFunction[Any, Any, Any, Any]:
  key = (n, n_eq, n_ineq)
  if key not in _SOLVERS:
    _SOLVERS[key] = solver(
      qp_problem(n, n_eq, n_ineq),
      "piqp",
      name=f"interp_constrained_{n}_{n_eq}_{n_ineq}",
      options={"eps_abs": 1e-12, "eps_rel": 1e-12, "eps_duality_gap_abs": 1e-12, "eps_duality_gap_rel": 1e-12},
    )
  return _SOLVERS[key]
