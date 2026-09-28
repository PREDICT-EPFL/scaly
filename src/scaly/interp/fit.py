"""Interpolants from data: the fits behind ``interpolant``, one axis at a time, done by SciPy when the graph is built for NumPy data."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from scipy import sparse

from ..function.model import ConcreteFunction
from ..function.sugar import scan
from ..ir.expr import Expr, cast, concat, equal, gather, minimum, not_equal, reduce_max, stack, where
from ..ir.types import DType, dtypes
from .grid import Axis, Extrap, Search, Side, check_sites
from .spline import BSpline, Strategy, _along, _per_axis, design_matrix

type Kind = Literal["nearest", "zoh", "linear", "cubic", "spline", "pchip", "akima", "makima", "steffen", "smooth_linear"]
type Boundary = Literal["not-a-knot", "natural", "clamped", "periodic"]

KINDS = ("nearest", "zoh", "linear", "cubic", "spline", "pchip", "akima", "makima", "steffen", "smooth_linear")
BOUNDARIES = ("not-a-knot", "natural", "clamped", "periodic")
HERMITE = ("pchip", "akima", "makima", "steffen")
DENSE_FIT = 40
"""The most sites along an axis whose fit, for ``Expr`` data, is a dense constant map; a C2 cubic
with more solves its tridiagonal system in scans instead. Measured (perf_2026_09_28_interp/
bench_param.py): the scans cost 1.3x the map at 32 sites and 0.38x at 64."""

_SCIPY_BC = {"not-a-knot": None, "natural": "natural", "clamped": "clamped", "periodic": "periodic"}


def interpolant(
  grid: np.ndarray | Sequence[float] | tuple[np.ndarray | Sequence[float], ...],
  values: np.ndarray | Sequence[Any] | Expr,
  kind: Kind | tuple[Kind, ...] = "linear",
  *,
  bc: Boundary | tuple[Boundary, ...] = "not-a-knot",
  degree: int | tuple[int, ...] = 3,
  extrap: Extrap | tuple[Extrap | None, ...] | None = None,
  fill: float = math.nan,
  period: float | tuple[float | None, ...] | None = None,
  frac: float = 0.1,
  search: Search | Literal["auto"] | tuple[Search | Literal["auto"], ...] = "auto",
  strategy: Strategy = "auto",
  dtype: DType | str = "float64",
  name: str = "interp",
) -> BSpline:
  """A lookup table or interpolating spline through ``values`` on a rectilinear grid.

  Args:
    grid: the data sites, strictly increasing: a vector in 1-D, a tuple of vectors in n-D.
    values: shape ``(n_1, ..., n_D, *out_shape)``; trailing axes give a vector- or matrix-valued
      interpolant.
    kind: per axis, or one for all:

      - ``"nearest"``: the value of the nearest site; a point midway takes the lower one, as SciPy's
        ``RegularGridInterpolator(method="nearest")``.
      - ``"zoh"``: the value of the last site at or before the point (a zero-order hold,
        ``searchsorted(side="right") - 1``); the last value holds for one more spacing, or up to
        ``period``.
      - ``"linear"``: piecewise linear, ``np.interp`` in 1-D and multilinear in n-D.
      - ``"cubic"``: the C² cubic spline with boundary condition ``bc``, SciPy's ``CubicSpline``.
      - ``"spline"``: the interpolating spline of ``degree`` 1 to 5 (``make_interp_spline``).
      - ``"pchip"``, ``"akima"``, ``"makima"``, ``"steffen"``: C1 piecewise cubics whose slopes
        at the sites come from the neighbouring data, so a step or a plateau does not ring:
        SciPy's ``PchipInterpolator`` (monotone data stays monotone) and ``Akima1DInterpolator``
        (``method="akima"`` or ``"makima"``), and Steffen's (1990) method, monotone and never
        overshooting a local extremum of the data. 1-D only; they are not linear in the data.
      - ``"smooth_linear"``: CasADi's ``smooth_linear`` algorithm: a cubic B-spline with knots
        at each site and ``frac`` of the smallest spacing either side of it, its coefficients the
        linear interpolant at the knots' Greville points; linear away from the sites and rounded
        within ``frac`` of them. It does not pass through the data.
    bc: for ``"cubic"``: ``"not-a-knot"``, ``"natural"`` (zero second derivative at the ends),
      ``"clamped"`` (zero first derivative) or ``"periodic"``; ``"spline"`` takes ``"not-a-knot"``
      or ``"periodic"``.
    degree: for ``"spline"``.
    extrap, fill, search, strategy, dtype, name: as for ``BSpline``. ``extrap`` defaults to
      ``"periodic"`` on a periodic axis, ``"clamp"`` for ``nearest`` and ``zoh``, ``"linear"``
      otherwise. A periodic axis of any kind but ``nearest`` and ``zoh`` needs its first and last
      values equal.
    period: for ``"zoh"`` with ``extrap="periodic"``, the period; the last value holds until the
      first site plus it.
    frac: for ``"smooth_linear"``, in ``(0, 0.5)``; CasADi's default.

  The fit is done per axis, so an n-D interpolant is the tensor product of the 1-D ones (SciPy's
  ``NdBSpline`` of per-axis ``make_interp_spline`` fits).

  ``values`` may be an ``Expr``: a table read at run time (a calibration changed without
  recompiling) or decision variables (identification). The fit is then part of the graph: a
  constant linear map for the kinds linear in the data (a sparse one for ``smooth_linear``; above
  ``DENSE_FIT`` sites a C2 cubic solves its tridiagonal system for the slopes in two scans instead,
  and is represented on doubled knots), the slope formulas as expressions for the shape-preserving
  kinds, whose derivative then follows ``sc.options(nonsmooth=...)`` where they switch. A periodic
  axis cannot check that such data close up.
  """
  sites = grid if isinstance(grid, tuple) else (grid,)
  ndim = len(sites)
  symbolic = isinstance(values, Expr)
  coeffs: Any = (
    (values if values.type.dtype == dtypes.float64 else cast(values, dtypes.float64))
    if isinstance(values, Expr)
    else np.asarray(values, dtype=np.float64)
  )
  if len(coeffs.shape) < ndim:
    raise ValueError(f"values must have at least {ndim} axes for a {ndim}-D grid, got shape {coeffs.shape}")
  if not symbolic and not np.all(np.isfinite(coeffs)):
    raise ValueError("values must be finite")
  per_axis = zip(
    sites,
    _per_axis(kind, ndim, "kind"),
    _per_axis(bc, ndim, "bc"),
    _per_axis(degree, ndim, "degree"),
    _per_axis(extrap, ndim, "extrap"),
    _per_axis(period, ndim, "period"),
    _per_axis(search, ndim, "search"),
    strict=True,
  )
  axes = []
  for d, (g, kd, bcd, deg, ext, per, srch) in enumerate(per_axis):
    g = check_sites(g, f"grid axis {d}")
    if coeffs.shape[d] != g.size:
      raise ValueError(f"values has {coeffs.shape[d]} entries along axis {d}, the grid {g.size}")
    if kd in HERMITE and ndim > 1:
      raise ValueError(f"kind={kd!r} is 1-D only: it is not a tensor-product spline in n-D")
    if not symbolic and (ext == "periodic" or (ext is None and bcd == "periodic")) and kd not in ("nearest", "zoh"):
      first, last = np.take(coeffs, 0, axis=d), np.take(coeffs, -1, axis=d)
      if not np.allclose(first, last, rtol=1e-14, atol=1e-14 * max(1.0, float(np.max(np.abs(coeffs))))):
        raise ValueError(f"a periodic axis needs its first and last values equal (axis {d})")
    coeffs, knots, k, edges, side = _axis_fit(g, kd, bcd, deg, per, frac, coeffs, d)
    axes.append(
      Axis(knots, k, edges=edges, side=side, extrap=ext if ext is not None else "periodic" if bcd == "periodic" else None, fill=fill, search=srch)
    )
  return BSpline.from_axes(axes, coeffs, strategy=strategy, dtype=dtype, name=name)


def _axis_fit(
  sites: np.ndarray, kind: str, bc: str, degree: int, period: float | None, frac: float, values: Any, axis: int
) -> tuple[Any, np.ndarray, int, np.ndarray, Side]:
  """One axis's fit: the coefficients (``values`` with that axis turned into coefficients), the knot
  vector, the degree, the search partition and the cell continuity."""
  if kind not in KINDS:
    raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
  if bc not in BOUNDARIES:
    raise ValueError(f"bc must be one of {BOUNDARIES}, got {bc!r}")
  if bc != "not-a-knot" and kind not in ("cubic", "spline"):
    raise ValueError(f"bc applies to kind='cubic' and 'spline', not {kind!r}")
  if kind in HERMITE:  # 1-D only, so the data's axis is the first
    knots = np.concatenate([np.full(4, sites[0]), np.repeat(sites[1:-1], 2), np.full(4, sites[-1])])
    return hermite_coefficients(sites, values, kind), knots, 3, sites, "right"
  if kind == "smooth_linear":
    knots, edges, weights = smooth_linear_knots(sites, frac)
    return _along(weights, values, axis), knots, 3, edges, "right"
  if kind == "nearest":
    edges = np.concatenate([sites[:1], 0.5 * (sites[:-1] + sites[1:]), sites[-1:]])
    return values, edges, 0, edges, "left"
  if kind == "zoh":
    end = sites[0] + period if period is not None else sites[-1] + (sites[-1] - sites[-2])
    if end <= sites[-1]:
      raise ValueError(f"period must exceed the grid's span on axis {axis}")
    knots = np.append(sites, end)
    return values, knots, 0, knots, "right"
  if kind == "linear":
    return values, np.concatenate([sites[:1], sites, sites[-1:]]), 1, sites, "right"
  k = 3 if kind == "cubic" else degree
  if kind == "spline" and (not isinstance(k, (int, np.integer)) or not 1 <= k <= 5):
    raise ValueError(f"degree must be 1 to 5 for kind='spline', got {degree!r}")
  if kind == "spline" and bc not in ("not-a-knot", "periodic"):
    raise ValueError(f"kind='spline' takes bc='not-a-knot' or 'periodic', got {bc!r}")
  if sites.size < k + 1 and bc == "not-a-knot":
    raise ValueError(f"a not-a-knot spline of degree {k} needs at least {k + 1} points on axis {axis}, got {sites.size}")
  from scipy.interpolate import make_interp_spline

  if isinstance(values, Expr) and kind == "cubic" and sites.size > DENSE_FIT:
    knots = np.concatenate([np.full(4, sites[0]), np.repeat(sites[1:-1], 2), np.full(4, sites[-1])])
    slopes = cubic_slopes(sites, bc, _moved(values, axis))
    return _unmoved(_hermite_form(sites, _moved(values, axis), slopes), axis), knots, 3, sites, "right"
  if isinstance(values, Expr):  # the fit's constant linear map, from the fit of the identity
    spl = make_interp_spline(sites, np.eye(sites.size), k=int(k), bc_type=_SCIPY_BC[bc])
    fitted = _along(spl.c, values, axis)
  else:
    spl = make_interp_spline(sites, values, k=int(k), bc_type=_SCIPY_BC[bc], axis=axis)
    fitted = np.moveaxis(spl.c, 0, axis)
  n = spl.t.size - k - 1
  return fitted, spl.t, int(k), np.union1d(sites, spl.t[k : n + 1]), "right"


def linear_weights(sites: np.ndarray, points: np.ndarray) -> sparse.csr_array:
  """The sparse ``(points, sites)`` matrix of piecewise-linear interpolation, continued linearly
  outside."""
  j = np.clip(np.searchsorted(sites, points, side="right") - 1, 0, sites.size - 2)
  w = (points - sites[j]) / (sites[j + 1] - sites[j])
  rows = np.repeat(np.arange(points.size), 2)
  return sparse.csr_array(
    (np.stack([1.0 - w, w], axis=1).reshape(-1), (rows, np.stack([j, j + 1], axis=1).reshape(-1))), shape=(points.size, sites.size)
  )


def smooth_linear_knots(sites: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray, sparse.csr_array]:
  """CasADi's ``smooth_linear`` (``casadi/solvers/bspline_interpolant.hpp``, ``construct_graph``): the
  knot vector, the partition, and the map from data to coefficients (the linear interpolant at the
  Greville points)."""
  if not 0.0 < frac < 0.5:
    raise ValueError(f"frac must be in (0, 0.5), got {frac!r}")
  step = frac * float(np.min(np.diff(sites)))
  inner = sites[1:-1]
  edges = np.concatenate(
    [[sites[0], sites[0] + step], np.stack([inner - step, inner, inner + step], axis=1).reshape(-1), [sites[-1] - step, sites[-1]]]
  )
  knots = np.concatenate([np.full(3, sites[0]), edges, np.full(3, sites[-1])])
  greville = (knots[1:-3] + knots[2:-2] + knots[3:-1]) / 3  # summed in CasADi's order
  return knots, edges, linear_weights(sites, greville)


def hermite_slopes(kind: str, sites: np.ndarray, values: Any) -> Any:
  """The slopes at the sites of a shape-preserving kind, for ``values`` along the first axis: SciPy's
  formulas for ``pchip``, ``akima`` and ``makima``, operation for operation, and Steffen's; in
  NumPy, or as expressions for an ``Expr``."""
  ops = _EXPR_OPS if isinstance(values, Expr) else _NUMPY_OPS
  h = np.diff(sites).reshape(-1, *(1,) * (len(values.shape) - 1))
  m = (values[1:] - values[:-1]) / h
  if values.shape[0] == 2:  # two points: the line through them, as SciPy
    return ops.concat([m, m])
  if kind == "pchip":
    return _pchip_slopes(h, m, ops)
  if kind in ("akima", "makima"):
    return _akima_slopes(m, kind, ops)
  return _steffen_slopes(h, m, ops)


class _Ops:
  """The few operations the slope formulas need, for NumPy arrays or for expressions."""

  def __init__(self, where: Any, sign: Any, abs_: Any, minimum: Any, concat: Any, maximum_all: Any, equal: Any, differ: Any) -> None:
    self.where, self.sign, self.abs, self.minimum, self.concat, self.max_all, self.equal, self.differ = (
      where,
      sign,
      abs_,
      minimum,
      concat,
      maximum_all,
      equal,
      differ,
    )


_NUMPY_OPS = _Ops(np.where, np.sign, np.abs, np.minimum, np.concatenate, np.max, lambda a, b: a == b, lambda a, b: a != b)
_EXPR_OPS = _Ops(
  where,
  lambda x: where(x > 0.0, 1.0, where(x < 0.0, -1.0, 0.0)),
  lambda x: x.abs(),
  minimum,
  lambda xs: concat(list(xs)),
  reduce_max,
  equal,
  not_equal,
)


def _pchip_slopes(h: np.ndarray, m: Any, ops: _Ops) -> Any:
  # Fritsch-Carlson: the weighted harmonic mean of the neighbouring secants where they agree in
  # sign, zero where they do not; three-point estimates at the ends, limited (Moler's pchiptx).
  # The divisions read 1 where they are not used, so that no branch divides by zero.
  sign = ops.sign(m)
  flat = ops.differ(sign[1:], sign[:-1]) | ops.equal(m[1:], 0.0) | ops.equal(m[:-1], 0.0)
  w1, w2 = 2 * h[1:] + h[:-1], h[1:] + 2 * h[:-1]
  whmean = (w1 / ops.where(flat, 1.0, m[:-1]) + w2 / ops.where(flat, 1.0, m[1:])) / (w1 + w2)
  inner = ops.where(flat, 0.0, 1.0 / ops.where(flat, 1.0, whmean))
  return ops.concat([_pchip_end(h[0], h[1], m[0:1], m[1:2], ops), inner, _pchip_end(h[-1], h[-2], m[-1:], m[-2:-1], ops)])


def _pchip_end(h0: np.ndarray, h1: np.ndarray, m0: Any, m1: Any, ops: _Ops) -> Any:
  d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
  wrong = ops.differ(ops.sign(d), ops.sign(m0))
  steep = ops.differ(ops.sign(m0), ops.sign(m1)) & (ops.abs(d) > 3.0 * ops.abs(m0))
  return ops.where(wrong, 0.0, ops.where(steep, 3.0 * m0, d))


def _akima_slopes(m: Any, method: str, ops: _Ops) -> Any:
  # Two secants extrapolated at each end, then each slope a weighted mean of the two central
  # secants, weighted by how much the outer ones differ (plus their mean's size for makima).
  first = 2.0 * m[0:1] - m[1:2]
  last = 2.0 * m[-1:] - m[-2:-1]
  mm = ops.concat([2.0 * first - m[0:1], first, m, last, 2.0 * last - m[-1:]])
  t = 0.5 * (mm[3:] + mm[:-3])
  dm = ops.abs(mm[1:] - mm[:-1])
  if method == "makima":
    pm = ops.abs(mm[1:] + mm[:-1])
    f1, f2 = dm[2:] + 0.5 * pm[2:], dm[:-2] + 0.5 * pm[:-2]
  else:
    f1, f2 = dm[2:], dm[:-2]
  f12 = f1 + f2
  defined = f12 > 1e-9 * ops.max_all(f12)
  weighted = mm[1:-2] + (f2 / ops.where(defined, f12, 1.0)) * (mm[2:-1] - mm[1:-2])
  return ops.where(defined, weighted, t)


def _steffen_slopes(h: np.ndarray, m: Any, ops: _Ops) -> Any:
  # M. Steffen, A&A 239:443 (1990), eq. 11 inside; at the ends the parabola through the first (last)
  # three points, limited to keep the end interval monotone (eqs. 26-27).
  p = (m[:-1] * h[1:] + m[1:] * h[:-1]) / (h[:-1] + h[1:])
  inner = (ops.sign(m[:-1]) + ops.sign(m[1:])) * ops.minimum(ops.minimum(ops.abs(m[:-1]), ops.abs(m[1:])), 0.5 * ops.abs(p))
  return ops.concat([_steffen_end(h[0], h[1], m[0:1], m[1:2], ops), inner, _steffen_end(h[-1], h[-2], m[-1:], m[-2:-1], ops)])


def _steffen_end(h0: np.ndarray, h1: np.ndarray, m0: Any, m1: Any, ops: _Ops) -> Any:
  p = m0 * (1.0 + h0 / (h0 + h1)) - m1 * h0 / (h0 + h1)
  return ops.where(p * m0 <= 0.0, 0.0, ops.where(ops.abs(p) > 2.0 * ops.abs(m0), 2.0 * m0, p))


def hermite_coefficients(sites: np.ndarray, values: Any, kind: str) -> Any:
  """The B-spline coefficients, on the knots with every interior site doubled, of the C1 cubic
  through ``values`` with the kind's slopes."""
  return _hermite_form(sites, values, hermite_slopes(kind, sites, values))


def _hermite_form(sites: np.ndarray, values: Any, slopes: Any) -> Any:
  """The B-coefficients on doubled knots of the C1 cubic with ``values`` and ``slopes`` at the sites:
  the Bezier points ``y_i -+ h d_i / 3`` either side of each site, the site's own value implied by C1."""
  h = np.diff(sites).reshape(-1, *(1,) * (len(values.shape) - 1))
  after, before = values[:-1] + h * slopes[:-1] / 3.0, values[1:] - h * slopes[1:] / 3.0
  if isinstance(values, Expr):
    pairs = stack([after, before], axis=1).reshape((2 * (sites.size - 1), *values.shape[1:]))
    return concat([values[0:1], pairs, values[-1:]])
  out = np.empty((2 * sites.size, *values.shape[1:]))
  out[0], out[-1] = values[0], values[-1]
  out[1:-1:2], out[2:-1:2] = after, before
  return out


def cubic_slopes(sites: np.ndarray, bc: str, values: Expr) -> Expr:
  """The slopes at the sites of the C2 cubic spline through ``values`` (along the first axis), as
  SciPy's ``CubicSpline`` sets up the system: tridiagonal for not-a-knot, natural and clamped,
  cyclic for periodic, both solved by elimination in two scans with the pivots and multipliers, all
  constant, found now."""
  n, dx = sites.size, np.diff(sites)
  rest = values.shape[1:]
  shape = lambda v: v.reshape(-1, *(1,) * len(rest))
  slope = (values[1:] - values[:-1]) / shape(dx)
  if bc == "periodic":  # unknowns s_0 .. s_{n-2}; s_{n-1} = s_0
    d_prev, d_next = np.roll(dx, 1), dx  # the spacing before and after each unknown's site
    s_prev = concat([slope[-1:], slope[:-1]])
    rhs = 3.0 * (shape(d_next) * s_prev + shape(d_prev) * slope)
    solved = _cyclic_solve(d_next, 2.0 * (d_prev + d_next), d_prev, rhs)
    return concat([solved, solved[0:1]])
  lower, diag, upper = np.zeros(n), np.zeros(n), np.zeros(n)
  lower[1:-1], diag[1:-1], upper[1:-1] = dx[1:], 2.0 * (dx[:-1] + dx[1:]), dx[:-1]
  inner = 3.0 * (shape(dx[1:]) * slope[:-1] + shape(dx[:-1]) * slope[1:])
  if bc == "not-a-knot":
    d0, d1 = sites[2] - sites[0], sites[-1] - sites[-3]
    diag[0], upper[0] = dx[1], d0
    diag[-1], lower[-1] = dx[-2], d1
    first = ((dx[0] + 2 * d0) * dx[1] * slope[0:1] + dx[0] ** 2 * slope[1:2]) / d0
    last = (dx[-1] ** 2 * slope[-2:-1] + (2 * d1 + dx[-1]) * dx[-2] * slope[-1:]) / d1
  elif bc == "clamped":  # zero first derivative
    diag[0], diag[-1] = 1.0, 1.0
    first, last = 0.0 * slope[0:1], 0.0 * slope[-1:]
  else:  # natural: zero second derivative
    diag[0], upper[0], diag[-1], lower[-1] = 2 * dx[0], dx[0], 2 * dx[-1], dx[-1]
    first, last = 3.0 * dx[0] * slope[0:1], 3.0 * dx[-1] * slope[-1:]
  return _tridiagonal_solve(lower, diag, upper, concat([first, inner, last]))


def _tridiagonal_solve(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: Expr) -> Expr:
  """``A s = rhs`` for a constant tridiagonal ``A`` (``lower[i] = A[i, i-1]``, ``upper[i] = A[i,
  i+1]``) and rows of ``rhs`` along its first axis: the Thomas algorithm, its pivots found now, as a
  forward scan and a backward one."""
  n = diag.size
  width = rhs.size // n
  pivot, ratio = np.empty(n), np.empty(n)
  pivot[0], ratio[0] = diag[0], upper[0] / diag[0]
  for i in range(1, n):
    pivot[i] = diag[i] - lower[i] * ratio[i - 1]
    ratio[i] = upper[i] / pivot[i]
  forward = np.stack([lower, 1.0 / pivot], axis=1).reshape(-1)
  flat = rhs.reshape((rhs.size,))
  (_, reduced) = scan(_sweep(width, "forward"), Expr.const(np.zeros(width)), [(flat, 0, width), (Expr.const(forward), 0, 2)], length=n)
  back = np.arange(n)[::-1]
  rows = (back[:, None] * width + np.arange(width)[None, :]).reshape(-1)
  (_, out) = scan(
    _sweep(width, "backward"), Expr.const(np.zeros(width)), [(gather(reduced, rows), 0, width), (Expr.const(ratio[back]), 0, 1)], length=n
  )
  return gather(out, rows).reshape(rhs.shape)


def _cyclic_solve(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: Expr) -> Expr:
  """A cyclic tridiagonal system (``lower[0]`` and ``upper[-1]`` in the corners) by Sherman and
  Morrison: one tridiagonal solve of the data, and a correction along a vector found now."""
  n = diag.size
  alpha, beta, gamma = upper[-1], lower[0], -diag[0]  # A[n-1, 0], A[0, n-1]
  corrected = diag.copy()
  corrected[0], corrected[-1] = diag[0] - gamma, diag[-1] - alpha * beta / gamma
  inner_lower, inner_upper = lower.copy(), upper.copy()
  inner_lower[0], inner_upper[-1] = 0.0, 0.0
  from scipy.linalg import solve_banded

  u = np.zeros(n)
  u[0], u[-1] = gamma, alpha
  banded = np.stack([np.concatenate([[0.0], inner_upper[:-1]]), corrected, np.concatenate([inner_lower[1:], [0.0]])])
  z = solve_banded((1, 1), banded, u)
  x = _tridiagonal_solve(inner_lower, corrected, inner_upper, rhs)
  factor = z / (1.0 + z[0] + beta * z[-1] / gamma)
  weight = x[0:1] + (beta / gamma) * x[-1:]
  return x - Expr.const(factor.reshape(-1, *(1,) * (len(rhs.shape) - 1))) * weight


_SWEEPS: dict[tuple[int, str], ConcreteFunction[Any, Any, Any, Any]] = {}


def _sweep(width: int, direction: str) -> ConcreteFunction[Any, Any, Any, Any]:
  """One row of the Thomas algorithm, as a scan body: forward ``r_i = (b_i - l_i r_{i-1}) / p_i``,
  backward ``s_i = r_i - c_i s_{i+1}``; the carry is the previous row."""
  key = (width, direction)
  if key not in _SWEEPS:
    prev, row = Expr.sym("prev", (width,)), Expr.sym("row", (width,))
    if direction == "forward":
      coef = Expr.sym("coef", (2,))
      nxt = (row - coef[0] * prev) * coef[1]
    else:
      coef = Expr.sym("coef", (1,))
      nxt = row - coef[0] * prev
    _SWEEPS[key] = ConcreteFunction._from_exprs(
      f"interp_thomas_{direction}_{width}", [prev, row, coef], [nxt, nxt], ["prev", "row", "coef"], ["next", "out"]
    )
  return _SWEEPS[key]


def _moved(values: Expr, axis: int) -> Expr:
  order = (axis, *(d for d in range(len(values.shape)) if d != axis))
  return values.transpose(order) if axis else values


def _unmoved(values: Expr, axis: int) -> Expr:
  if not axis:
    return values
  order = (axis, *(d for d in range(len(values.shape)) if d != axis))
  return values.transpose(tuple(int(i) for i in np.argsort(order)))


def smoothing(
  x: np.ndarray | Sequence[float] | tuple[np.ndarray | Sequence[float], ...],
  y: np.ndarray | Sequence[Any] | Expr,
  *,
  degree: int | tuple[int, ...] = 3,
  segments: int | tuple[int, ...] = 20,
  penalty: int = 2,
  lam: float | Literal["gcv"] = "gcv",
  method: Literal["pspline", "cubic"] = "pspline",
  extrap: Extrap | tuple[Extrap | None, ...] | None = None,
  fill: float = math.nan,
  search: Search | Literal["auto"] | tuple[Search | Literal["auto"], ...] = "auto",
  strategy: Strategy = "auto",
  dtype: DType | str = "float64",
  name: str = "interp",
) -> BSpline:
  """A smoothing spline through noisy data: it trades closeness to ``y`` for smoothness, by ``lam``.

  Args:
    x: the data sites: a vector ``(m,)``, points ``(m, D)``, or a tuple of grid vectors with ``y``
      on the grid.
    y: ``(m, *out_shape)``, or ``(n_1, ..., n_D, *out_shape)`` on a grid.
    method: ``"pspline"`` (Eilers and Marx): a B-spline of ``degree`` on ``segments`` equal
      intervals per axis spanning the data, fitted by least squares with the ``penalty``-th
      differences of its coefficients along each axis penalized, ``|y - B c|^2 + lam sum_d
      |D_d c|^2``; any degree and dimension. ``"cubic"``: SciPy's ``make_smoothing_spline``, the
      cubic with knots at the data minimizing ``|y - f|^2 + lam int f''^2`` (1-D).
    lam: the weight of the penalty, or ``"gcv"`` to choose it by generalized cross-validation.
    extrap, fill, search, strategy, dtype, name: as for ``BSpline``.
  """
  if method not in ("pspline", "cubic"):
    raise ValueError(f"method must be 'pspline' or 'cubic', got {method!r}")
  if lam != "gcv" and not (isinstance(lam, (int, float)) and lam >= 0):
    raise ValueError(f"lam must be 'gcv' or a non-negative number, got {lam!r}")
  values: Any = y if isinstance(y, Expr) else np.asarray(y, dtype=np.float64)
  if isinstance(values, Expr) and lam == "gcv":
    raise ValueError("lam='gcv' needs the data's values; with an Expr for y, give lam")
  if isinstance(x, tuple):
    grids = [check_sites(g, f"grid axis {d}") for d, g in enumerate(x)]
    points = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, len(grids))
    values = values.reshape((points.shape[0], *values.shape[len(grids) :]))
  else:
    points = np.asarray(x, dtype=np.float64)
    points = points[:, None] if points.ndim == 1 else points
  if points.ndim != 2 or values.shape[0] != points.shape[0]:
    raise ValueError(f"x gives {points.shape[0]} points and y {values.shape[0]} values")
  if not (np.all(np.isfinite(points)) and (isinstance(values, Expr) or np.all(np.isfinite(values)))):
    raise ValueError("x and y must be finite")
  ndim = points.shape[1]
  if method == "cubic":
    from scipy.interpolate import make_smoothing_spline

    if ndim != 1:
      raise ValueError("method='cubic' is 1-D")
    sites = check_sites(points[:, 0], "x", minimum_points=5)
    if isinstance(values, Expr):  # linear in y for a given lam: the fit of the identity
      spl = make_smoothing_spline(sites, np.eye(sites.size), lam=float(lam))
      return BSpline(spl.t, _along(spl.c, values, 0), 3, extrap=extrap, fill=fill, search=search, strategy=strategy, dtype=dtype, name=name)
    spl = make_smoothing_spline(sites, values, lam=None if lam == "gcv" else float(lam))
    return BSpline(spl.t, spl.c, 3, extrap=extrap, fill=fill, search=search, strategy=strategy, dtype=dtype, name=name)
  degrees, pieces = _per_axis(degree, ndim, "degree"), _per_axis(segments, ndim, "segments")
  knots = []
  for d, (k, nseg) in enumerate(zip(degrees, pieces, strict=True)):
    lo, hi = float(points[:, d].min()), float(points[:, d].max())
    if not hi > lo or nseg < 1:
      raise ValueError(f"axis {d} needs data spread over an interval and at least one segment")
    step = (hi - lo) / nseg  # the base interval is exactly [lo, hi], whatever lo + nseg * step rounds to
    knots.append(np.concatenate([lo - step * np.arange(k, 0, -1), np.linspace(lo, hi, nseg + 1), hi + step * np.arange(1, k + 1)]))
  coeffs, _ = pspline_fit(knots, degrees, points, values, penalty, lam)
  shape = tuple(t.size - k - 1 for t, k in zip(knots, degrees, strict=True))
  return BSpline(
    tuple(knots), coeffs.reshape((*shape, *values.shape[1:])), degrees, extrap=extrap, fill=fill, search=search, strategy=strategy, name=name
  )


def pspline_fit(
  knots: Sequence[np.ndarray], degrees: Sequence[int], points: np.ndarray, values: Any, penalty: int, lam: float | str
) -> tuple[Any, float]:
  """The penalized least-squares coefficients (flat, C order) and the weight used: the one given, or
  the minimizer of ``m RSS / (m - tr H)^2`` over ``log10 lam``, a coarse grid then a bounded search.
  For an ``Expr`` of values the weight is given, and the fit is its constant linear map."""
  from scipy import linalg, optimize

  B = design_matrix(knots, degrees, points).toarray()
  sizes = [t.size - k - 1 for t, k in zip(knots, degrees, strict=True)]
  if any(n <= penalty for n in sizes):
    raise ValueError(f"a difference penalty of order {penalty} needs more than {penalty} coefficients per axis")
  P = np.zeros((B.shape[1], B.shape[1]))
  for d, n in enumerate(sizes):
    diff = np.diff(np.eye(n), n=penalty, axis=0)
    blocks = [np.eye(m) for m in sizes]
    blocks[d] = diff.T @ diff
    term = blocks[0]
    for block in blocks[1:]:
      term = np.kron(term, block)
    P += term
  if isinstance(values, Expr):
    gram = B.T @ B
    lhs = gram + float(lam) * P
    return _along(linalg.cho_solve(linalg.cho_factor(lhs), B.T), values.reshape((B.shape[0], values.size // B.shape[0])), 0), float(lam)
  flat = values.reshape(values.shape[0], -1)
  gram, rhs, m = B.T @ B, B.T @ flat, B.shape[0]

  def solve(weight: float) -> tuple[np.ndarray, np.ndarray]:
    factor = linalg.cho_factor(gram + weight * P)
    return linalg.cho_solve(factor, rhs), factor

  if lam != "gcv":
    return solve(float(lam))[0], float(lam)
  scale = np.trace(gram) / max(np.trace(P), 1e-300)

  def gcv(log_lam: float) -> float:
    weight = scale * 10.0**log_lam
    coeffs, factor = solve(weight)
    trace = float(np.trace(linalg.cho_solve(factor, gram)))
    rss = float(np.sum((flat - B @ coeffs) ** 2))
    return m * rss / max(m - trace, 1e-12) ** 2

  grid = np.linspace(-9.0, 6.0, 31)
  best = int(np.argmin([gcv(g) for g in grid]))
  lo, hi = grid[max(best - 1, 0)], grid[min(best + 1, grid.size - 1)]
  log_lam = float(optimize.minimize_scalar(gcv, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4}).x)
  weight = scale * 10.0**log_lam
  return solve(weight)[0], weight
