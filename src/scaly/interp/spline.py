"""The tensor-product B-spline every interpolant is: its evaluation as expressions (piecewise polynomials or local bases), its Function, and its SciPy twin."""

from __future__ import annotations

import hashlib
import math
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..function.model import ConcreteFunction
from ..function.tree import L, param_list
from ..function.sugar import vmap
from ..ir.expr import Expr, as_expr, cast, not_equal, stack, take, where
from ..ir.types import dtypes
from .grid import Axis, Extrap, Search

type Strategy = Literal["auto", "pp", "basis"]

PP_BUDGET = 1 << 20
"""The most bytes of piecewise-polynomial table ``strategy="auto"`` generates; a larger spline is
evaluated from its B-spline coefficients and local bases instead."""


@dataclass(frozen=True, eq=False)
class Index:
  """The cell of each point along each axis, from ``BSpline.index``: found once and shared by every
  evaluation at the same points (a spline and its derivative, say). Float ``Expr``s shaped like the
  batch."""

  cells: tuple[Expr, ...]


def _per_axis(value: Any, ndim: int, what: str) -> tuple[Any, ...]:
  if isinstance(value, tuple):
    if len(value) != ndim:
      raise ValueError(f"{what} gives {len(value)} entries for {ndim} axes")
    return value
  return (value,) * ndim


class BSpline:
  """A tensor-product B-spline ``f(x) = sum_a c_a prod_d B_{a_d, k_d}(x_d; t_d)`` evaluated as
  generated code: a lookup table (degree 0 or 1), an interpolating or smoothing spline, or a spline
  with coefficients of your own.

  Args:
    knots: the knot vector ``t`` (non-decreasing, multiplicities allowed), or a tuple of them, one
      per axis.
    coeffs: the B-spline coefficients, shape ``(n_1, ..., n_D, *out_shape)`` with ``n_d =
      len(t_d) - k_d - 1``; trailing axes make a vector- or matrix-valued spline.
    degree: ``k``, one for all axes or a tuple.
    extrap: outside the base interval ``[t[k], t[n]]``, per axis: ``"linear"`` (the default for
      degree 1 and up: continued along the tangent at the end, so the gradient outside is the one at
      the end; in n-D the product of the axes' continuations, which keeps it C1), ``"clamp"`` (the
      default for degree 0: the end value), ``"extend"`` (the end polynomial continued, as SciPy's
      ``extrapolate=True``), ``"periodic"`` (wrapped by the base interval's length) or ``"fill"``
      (the constant ``fill``). At degree 1 ``"linear"`` is ``"extend"``, and at degree 0 ``"clamp"``,
      and the axis records it so. NaN gives NaN in every mode.
    fill: the value outside for ``extrap="fill"``, NaN by default.
    search: how the cell of a point is found, per axis: ``"uniform"`` (a floor and one correction,
      for uniform knots), ``"binary"`` (branch-free halvings) or ``"count"`` (a branch-free count
      of the knots passed); ``"auto"`` picks uniform when it is exact, else binary.
    strategy: ``"pp"`` evaluates per-cell polynomial coefficients by nested Horner, the fastest,
      storing ``prod (k_d + 1)`` values per cell; ``"basis"`` stores the B-spline coefficients and
      combines the ``k + 1`` local basis functions of each axis. ``"auto"`` takes ``pp`` while its
      table fits ``PP_BUDGET``.
    name: the base of the name of ``function()``.

  Calling the spline evaluates it: at a point (a scalar in 1-D, a ``(D,)`` vector) or at a batch
  (``(N,)`` in 1-D, ``(N, D)``), as one vectorized graph. Derivatives with respect to the point come
  from differentiating that graph; at a knot they are the one-sided ones of the cell to the right.
  """

  def __init__(
    self,
    knots: np.ndarray | Sequence[float] | tuple[np.ndarray, ...],
    coeffs: np.ndarray,
    degree: int | tuple[int, ...] = 3,
    *,
    extrap: Extrap | tuple[Extrap | None, ...] | None = None,
    fill: float = math.nan,
    search: Search | Literal["auto"] | tuple[Search | Literal["auto"], ...] = "auto",
    strategy: Strategy = "auto",
    name: str = "interp",
  ) -> None:
    grids = knots if isinstance(knots, tuple) else (knots,)
    ndim = len(grids)
    degrees, extraps, searches = _per_axis(degree, ndim, "degree"), _per_axis(extrap, ndim, "extrap"), _per_axis(search, ndim, "search")
    axes = tuple(
      Axis(np.asarray(t, dtype=np.float64), k, extrap=e, fill=fill, search=s) for t, k, e, s in zip(grids, degrees, extraps, searches, strict=True)
    )
    self._setup(axes, coeffs, strategy, name)

  @classmethod
  def from_axes(cls, axes: Sequence[Axis], coeffs: np.ndarray, *, strategy: Strategy = "auto", name: str = "interp") -> BSpline:
    """A spline over prepared axes (a fit's, whose partitions its data sites refine)."""
    self = cls.__new__(cls)
    self._setup(tuple(axes), coeffs, strategy, name)
    return self

  def _setup(self, axes: tuple[Axis, ...], coeffs: Any, strategy: Strategy, name: str) -> None:
    if not axes:
      raise ValueError("a spline needs at least one axis")
    coeffs = np.asarray(coeffs, dtype=np.float64)
    shape = tuple(ax.n for ax in axes)
    if coeffs.shape[: len(axes)] != shape:
      raise ValueError(f"coeffs must have shape {shape} + out_shape for these knots and degrees, got {coeffs.shape}")
    if not np.all(np.isfinite(coeffs)):
      raise ValueError("coeffs must be finite")
    if strategy not in ("auto", "pp", "basis"):
      raise ValueError(f"strategy must be 'auto', 'pp' or 'basis', got {strategy!r}")
    self.axes, self.coeffs, self.name = axes, coeffs, name
    self.out_shape: tuple[int, ...] = coeffs.shape[len(axes) :]
    pp_bytes = 8 * math.prod(ax.cells * (ax.degree + 1) for ax in axes) * math.prod(self.out_shape)
    self.strategy: Literal["pp", "basis"] = ("pp" if pp_bytes <= PP_BUDGET else "basis") if strategy == "auto" else strategy
    self._pp: np.ndarray | None = None

  @property
  def ndim(self) -> int:
    return len(self.axes)

  @property
  def degree(self) -> tuple[int, ...]:
    return tuple(ax.degree for ax in self.axes)

  @property
  def knots(self) -> tuple[np.ndarray, ...]:
    return tuple(ax.knots for ax in self.axes)

  def __repr__(self) -> str:
    grid = " x ".join(f"{ax.cells}" for ax in self.axes)
    return f"BSpline({self.name!r}, degree={self.degree}, cells={grid}, out_shape={self.out_shape}, strategy={self.strategy!r})"

  def _points(self, x: Any) -> tuple[Expr, bool]:
    """``x`` as float64, and whether it is a batch."""
    x = as_expr(x)
    if x.type.dtype != dtypes.float64:
      x = cast(x, dtypes.float64)
    point = () if self.ndim == 1 else (self.ndim,)
    if x.shape == point:
      return x, False
    if len(x.shape) == len(point) + 1 and x.shape[1:] == point:
      return x, True
    what = "a scalar or a batch (N,)" if self.ndim == 1 else f"a point ({self.ndim},) or a batch (N, {self.ndim})"
    raise ValueError(f"a {self.ndim}-D spline takes {what}, got shape {x.shape}")

  def _coordinates(self, x: Expr) -> list[Expr]:
    """One scalar per axis, wrapped on a periodic axis."""
    pts = [x] if self.ndim == 1 else [x[d] for d in range(self.ndim)]
    return [ax.wrap(p) if ax.extrap == "periodic" else p for ax, p in zip(self.axes, pts, strict=True)]

  def index(self, x: Any) -> Index:
    """The cells of the points ``x``, to pass as ``index=`` to several evaluations at them."""
    x, batch = self._points(x)
    if not batch:
      return Index(tuple(ax.cell(p) for ax, p in zip(self.axes, self._coordinates(x), strict=True)))
    n = x.shape[0]
    cells = vmap(self._search_function(), n, [(x.reshape((x.size,)), 0, x.size // n)]).reshape((n, self.ndim))
    return Index(tuple(cells[:, d] for d in range(self.ndim)))

  def __call__(self, x: Any, *, index: Index | None = None) -> Expr:
    x, batch = self._points(x)
    if not batch:
      if index is not None and (len(index.cells) != self.ndim or any(c.shape for c in index.cells)):
        raise ValueError("index= was found for points of another shape")
      coords = self._coordinates(x)
      cells = index.cells if index is not None else tuple(ax.cell(p) for ax, p in zip(self.axes, coords, strict=True))
      return self._at(coords, cells)
    # A batch is one mapped call per point: its derivatives differentiate the point's graph once, and
    # its code is one loop whatever the batch size.
    n = x.shape[0]
    flat = (x.reshape((x.size,)), 0, x.size // n)
    if index is None:
      out = vmap(self.function(), n, [flat])
    else:
      if len(index.cells) != self.ndim or any(c.shape != (n,) for c in index.cells):
        raise ValueError("index= was found for points of another shape")
      cells = stack(list(index.cells), axis=1).reshape((n * self.ndim,))
      out = vmap(self._cell_function(), n, [flat, (cells, 0, self.ndim)])
    return out.reshape((n, *self.out_shape))

  def _at(self, coords: list[Expr], cells: tuple[Expr, ...]) -> Expr:
    """The value at one point, from its coordinates and cells."""
    local: list[tuple[Expr, Expr, Expr | None]] = []  # per axis: the int cell, s, and the step outside for "linear"
    guards: list[tuple[Expr, float]] = []
    for ax, p, j in zip(self.axes, coords, cells, strict=True):
      if ax.degree == 0:  # a constant never reads its coordinate, so NaN is passed on by hand
        guards.append((not_equal(p, p), math.nan))
      i = cast(j, dtypes.int64)
      delta = None
      if ax.extrap in ("clamp", "fill", "linear"):
        clamped = where(p < ax.lo, ax.lo, where(p > ax.hi, ax.hi, p))  # NaN passes through
        if ax.extrap == "linear":
          delta = p - clamped
        if ax.extrap == "fill":
          guards.append(((p < ax.lo) | (p > ax.hi), ax.fill))
        p = clamped
      local.append((i, p - _gather(ax.centers, i), delta))
    value = self._pp_eval(local) if self.strategy == "pp" else self._basis_eval(local)
    for out, fill in reversed(guards):
      value = where(out, fill, value)
    return value

  def _pp_table(self) -> np.ndarray:
    """Cell-major, then the ``prod (k_d + 1)`` powers, then the output: the Taylor coefficients of
    every cell at its center."""
    if self._pp is None:
      t = self.coeffs
      for d, ax in enumerate(self.axes):
        t = np.moveaxis(ax.taylor(np.moveaxis(t, 2 * d, 0)), [0, 1], [2 * d, 2 * d + 1])
      nd = self.ndim
      self._pp = np.ascontiguousarray(t.transpose([*range(0, 2 * nd, 2), *range(1, 2 * nd, 2), *range(2 * nd, t.ndim)])).reshape(-1)
    return self._pp

  def _pp_eval(self, local: list[tuple[Expr, Expr, Expr | None]]) -> Expr:
    orders = [ax.degree + 1 for ax in self.axes]
    base = local[0][0]
    for (i, _, _), ax in zip(local[1:], self.axes[1:], strict=True):
      base = base * ax.cells + i
    coef = _block(self._pp_table(), base, math.prod(orders) * math.prod(self.out_shape)).reshape((*orders, *self.out_shape))
    for d in reversed(range(self.ndim)):  # nested Horner, the last axis innermost
      _, s, delta = local[d]
      coef = _horner([coef[(*(slice(None),) * d, m)] for m in range(orders[d])], s, delta)
    return coef

  def _basis_eval(self, local: list[tuple[Expr, Expr, Expr | None]]) -> Expr:
    orders = [ax.degree + 1 for ax in self.axes]
    out_size = math.prod(self.out_shape)
    strides = [math.prod(ax.n for ax in self.axes[d + 1 :]) * out_size for d in range(self.ndim)]
    bases, first = [], []
    for (i, s, delta), ax, stride in zip(local, self.axes, strides, strict=True):
      k1 = ax.degree + 1
      mats = _block(ax.local.reshape(-1), i, k1 * k1).reshape((k1, k1))
      bases.append(_horner([mats[:, m] for m in range(k1)], s, delta))
      first.append(cast(_gather(ax.offsets.astype(np.float64), i), dtypes.int64) * stride)
    offsets = sum(
      np.arange(k).reshape((*(1,) * d, k, *(1,) * (self.ndim - d - 1))) * stride for d, (k, stride) in enumerate(zip(orders, strides, strict=True))
    )
    offsets = (np.asarray(offsets).reshape(-1)[:, None] + np.arange(out_size)[None, :]).reshape(-1)
    base = first[0]
    for f in first[1:]:
      base = base + f
    index = stack([base]) + Expr.const(offsets, dtype=dtypes.int64)
    coef = take(Expr.const(self.coeffs.reshape(-1)), index, in_range=True)
    for b, k in zip(bases, orders, strict=True):  # contract one axis at a time
      coef = b @ coef.reshape((k, coef.size // k))
    return coef.reshape(self.out_shape)

  def _search_function(self) -> ConcreteFunction[Any, Any, Any, Any]:
    point = () if self.ndim == 1 else (self.ndim,)
    return _interned(
      f"{self.name}_{self.digest}_cell",
      lambda: ConcreteFunction(
        f"{self.name}_{self.digest}_cell",
        lambda x: stack([ax.cell(p) for ax, p in zip(self.axes, self._coordinates(x), strict=True)]),
        param_list(L("x", point)),
        L("cell", (self.ndim,)),
      ),
    )

  def _cell_function(self) -> ConcreteFunction[Any, Any, Any, Any]:
    point = () if self.ndim == 1 else (self.ndim,)
    return _interned(
      f"{self.name}_{self.digest}_at",
      lambda: ConcreteFunction(
        f"{self.name}_{self.digest}_at",
        lambda x, cell: self._at(self._coordinates(x), tuple(cell[d] for d in range(self.ndim))),
        param_list(L("x", point), L("cell", (self.ndim,))),
        L("y", self.out_shape),
      ),
    )

  @property
  def digest(self) -> str:
    """Eight hex digits naming the spline's content, for ``function()`` names that never collide."""
    h = hashlib.sha1(self.strategy.encode())
    for ax in self.axes:
      h.update(repr((ax.degree, ax.side, ax.extrap, ax.fill, ax.search)).encode())
      h.update(ax.knots.tobytes())
      h.update(ax.edges.tobytes())
    h.update(repr(self.coeffs.shape).encode())
    h.update(np.ascontiguousarray(self.coeffs).tobytes())
    return h.hexdigest()[:8]

  def function(self, name: str | None = None) -> ConcreteFunction[Any, Any, Any, Any]:
    """The spline as a Function of one point, ``x -> f(x)``: calls then share one procedure and one
    copy of the table, where inlining would give each calling procedure its own. Named
    ``{name}_{digest}`` unless ``name`` is given; two splines with the same content share it."""
    point = () if self.ndim == 1 else (self.ndim,)
    fname = name or f"{self.name}_{self.digest}"
    return _interned(fname, lambda: ConcreteFunction(fname, lambda x: self(x), param_list(L("x", point)), L("y", self.out_shape)))

  def to_scipy(self) -> Any:
    """The same spline as a SciPy ``BSpline`` (1-D) or ``NdBSpline``, extrapolating by its end
    polynomials; for tests and plotting."""
    from scipy.interpolate import BSpline as SciBSpline
    from scipy.interpolate import NdBSpline

    if self.ndim == 1:
      return SciBSpline(self.axes[0].knots, self.coeffs, self.axes[0].degree)
    return NdBSpline(self.knots, self.coeffs, self.degree)


_FUNCTIONS: weakref.WeakValueDictionary[str, ConcreteFunction[Any, Any, Any, Any]] = weakref.WeakValueDictionary()


def _interned(name: str, build: Callable[[], ConcreteFunction[Any, Any, Any, Any]]) -> ConcreteFunction[Any, Any, Any, Any]:
  """One Function per name: the names carry the content's digest, and lowering refuses two different
  Functions of one name in a graph."""
  fn = _FUNCTIONS.get(name)
  if fn is None:
    fn = _FUNCTIONS[name] = build()
  return fn


def _gather(table: np.ndarray, i: Expr) -> Expr:
  """``table[i]`` for a constant vector and a scalar ``int64`` index in range."""
  return take(Expr.const(np.ascontiguousarray(table, dtype=np.float64)), stack([i]), in_range=True)[0]


def _block(table: np.ndarray, base: Expr, width: int) -> Expr:
  """``table[base * width + arange(width)]`` for a scalar ``int64`` ``base``."""
  return take(Expr.const(table), stack([base]) * width + Expr.const(np.arange(width), dtype=dtypes.int64), in_range=True)


def _horner(coef: list[Expr], s: Expr, delta: Expr | None) -> Expr:
  """``sum_m coef[m] s^m``, and with ``delta`` its linear continuation ``p(s) + delta p'(s)``."""
  q, dq = coef[-1], None
  for c in reversed(coef[:-1]):
    if delta is not None:
      dq = q if dq is None else dq * s + q
    q = q * s + c
  return q if delta is None or dq is None else q + delta * dq
