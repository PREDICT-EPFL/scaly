"""Interpolants from data: the fits behind ``interpolant``, one axis at a time, done by SciPy when the graph is built for NumPy data."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np

from .grid import Axis, Extrap, Search, Side, check_sites
from .spline import BSpline, Strategy, _per_axis

type Kind = Literal["nearest", "zoh", "linear", "cubic", "spline"]
type Boundary = Literal["not-a-knot", "natural", "clamped", "periodic"]

KINDS = ("nearest", "zoh", "linear", "cubic", "spline")
BOUNDARIES = ("not-a-knot", "natural", "clamped", "periodic")

_SCIPY_BC = {"not-a-knot": None, "natural": "natural", "clamped": "clamped", "periodic": "periodic"}


def interpolant(
  grid: np.ndarray | Sequence[float] | tuple[np.ndarray | Sequence[float], ...],
  values: np.ndarray | Sequence[Any],
  kind: Kind | tuple[Kind, ...] = "linear",
  *,
  bc: Boundary | tuple[Boundary, ...] = "not-a-knot",
  degree: int | tuple[int, ...] = 3,
  extrap: Extrap | tuple[Extrap | None, ...] | None = None,
  fill: float = math.nan,
  period: float | tuple[float | None, ...] | None = None,
  search: Search | Literal["auto"] | tuple[Search | Literal["auto"], ...] = "auto",
  strategy: Strategy = "auto",
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
    bc: for ``"cubic"``: ``"not-a-knot"``, ``"natural"`` (zero second derivative at the ends),
      ``"clamped"`` (zero first derivative) or ``"periodic"``; ``"spline"`` takes ``"not-a-knot"``
      or ``"periodic"``.
    degree: for ``"spline"``.
    extrap, fill, search, strategy, name: as for ``BSpline``. ``extrap`` defaults to
      ``"periodic"`` on a periodic axis, ``"clamp"`` for ``nearest`` and ``zoh``, ``"linear"``
      otherwise. A periodic axis of any kind but ``nearest`` and ``zoh`` needs its first and last
      values equal.
    period: for ``"zoh"`` with ``extrap="periodic"``, the period; the last value holds until the
      first site plus it.

  The fit is linear in ``values`` and done per axis, so an n-D interpolant is the tensor product of
  the 1-D ones (SciPy's ``NdBSpline`` of per-axis ``make_interp_spline`` fits).
  """
  sites = grid if isinstance(grid, tuple) else (grid,)
  ndim = len(sites)
  coeffs = np.asarray(values, dtype=np.float64)
  if coeffs.ndim < ndim:
    raise ValueError(f"values must have at least {ndim} axes for a {ndim}-D grid, got shape {coeffs.shape}")
  if not np.all(np.isfinite(coeffs)):
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
    knots, k, edges, side, fit = _axis_fit(g, kd, bcd, deg, per, d)
    if (ext == "periodic" or (ext is None and bcd == "periodic")) and kd not in ("nearest", "zoh"):
      first, last = np.take(coeffs, 0, axis=d), np.take(coeffs, -1, axis=d)
      if not np.allclose(first, last, rtol=1e-14, atol=1e-14 * max(1.0, float(np.max(np.abs(coeffs))))):
        raise ValueError(f"a periodic axis needs its first and last values equal (axis {d})")
    coeffs = fit(coeffs, d)
    axes.append(
      Axis(knots, k, edges=edges, side=side, extrap=ext if ext is not None else "periodic" if bcd == "periodic" else None, fill=fill, search=srch)
    )
  return BSpline.from_axes(axes, coeffs, strategy=strategy, name=name)


def _identity(values: np.ndarray, axis: int) -> np.ndarray:
  return values


def _axis_fit(
  sites: np.ndarray, kind: str, bc: str, degree: int, period: float | None, axis: int
) -> tuple[np.ndarray, int, np.ndarray, Side, Callable[[np.ndarray, int], np.ndarray]]:
  """The knot vector, degree, search partition and cell continuity of one axis, and the map from
  its data to its coefficients (along an array axis)."""
  if kind not in KINDS:
    raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
  if bc not in BOUNDARIES:
    raise ValueError(f"bc must be one of {BOUNDARIES}, got {bc!r}")
  if kind == "nearest":
    edges = np.concatenate([sites[:1], 0.5 * (sites[:-1] + sites[1:]), sites[-1:]])
    return edges, 0, edges, "left", _identity
  if kind == "zoh":
    end = sites[0] + period if period is not None else sites[-1] + (sites[-1] - sites[-2])
    if end <= sites[-1]:
      raise ValueError(f"period must exceed the grid's span on axis {axis}")
    knots = np.append(sites, end)
    return knots, 0, knots, "right", _identity
  if kind == "linear":
    return np.concatenate([sites[:1], sites, sites[-1:]]), 1, sites, "right", _identity
  k = 3 if kind == "cubic" else degree
  if kind == "spline" and (not isinstance(k, (int, np.integer)) or not 1 <= k <= 5):
    raise ValueError(f"degree must be 1 to 5 for kind='spline', got {degree!r}")
  if kind == "spline" and bc not in ("not-a-knot", "periodic"):
    raise ValueError(f"kind='spline' takes bc='not-a-knot' or 'periodic', got {bc!r}")
  if sites.size < k + 1 and bc == "not-a-knot":
    raise ValueError(f"a not-a-knot spline of degree {k} needs at least {k + 1} points on axis {axis}, got {sites.size}")
  from scipy.interpolate import make_interp_spline

  knots = make_interp_spline(sites, np.zeros(sites.size), k=int(k), bc_type=_SCIPY_BC[bc]).t
  n = knots.size - k - 1
  edges = np.union1d(sites, knots[k : n + 1])

  def fit(values: np.ndarray, ax: int) -> np.ndarray:
    return np.moveaxis(make_interp_spline(sites, values, k=int(k), bc_type=_SCIPY_BC[bc], axis=ax).c, 0, ax)

  return knots, int(k), edges, "right", fit
