"""The tensor-product B-spline every interpolant is: its evaluation as expressions (piecewise polynomials or local bases), its Function, and its SciPy twin."""

from __future__ import annotations

import hashlib
import math
import os
import warnings
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import sparse

from ..function.model import ConcreteFunction
from ..function.tree import L, param_list
from ..function.sugar import custom_derivative, vmap
from ..ir.expr import Expr, as_expr, cast, equal, not_equal, stack, take, where
from ..ir.types import DType, as_dtype, dtypes
from ..roots.newton import NewtonBisection
from .grid import Axis, Extrap, Search, derivative_matrix, num

type Strategy = Literal["auto", "pp", "basis"]

LARGE_TABLE = 1 << 20
"""The most values a spline tabulates in the generated C before it warns: tens of megabytes of
source and seconds of compilation past it."""

PP_BUDGET = 1 << 22
"""The most bytes of piecewise-polynomial table ``strategy="auto"`` generates when the B-spline
coefficients and local bases would be smaller; past it such a spline is evaluated from those.
Per-cell polynomials are the faster to evaluate, the local bases the smaller to compile. Read-only:
the library reads its own copy."""


@dataclass(frozen=True, eq=False)
class Index:
  """The cell of each point along each axis, from ``BSpline.index``: found once and shared by every
  evaluation at the same points (a spline and its derivative, say). Float ``Expr``s shaped like the
  batch, and the partition they were found on: a spline on another refuses them."""

  cells: tuple[Expr, ...]
  partition: str = ""


def _per_axis(value: Any, ndim: int, what: str) -> tuple[Any, ...]:
  if isinstance(value, list) and value and all(v is None or isinstance(v, str) for v in value):
    value = tuple(value)  # a list of names is one per axis as a tuple is
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
      for uniform knots), ``"bucket"`` (a uniform bucket index, the bucket's first cell from a
      table, then as many compares as the fullest bucket needs), ``"binary"`` (branch-free
      halvings) or ``"count"`` (a branch-free count of the knots passed). ``"auto"`` takes binary up
      to 32 cells, then bucket, uniform or binary, the first the partition allows.
    strategy: ``"pp"`` evaluates per-cell polynomial coefficients by nested Horner, the fastest,
      storing ``prod (k_d + 1)`` values per cell; ``"basis"`` stores the B-spline coefficients and
      combines the ``k + 1`` local basis functions of each axis. ``"auto"`` takes ``pp`` while its
      table fits ``PP_BUDGET`` or is the smaller of the two (as in 1-D, where ``basis`` stores
      ``(k + 1)^2`` values per cell).
    dtype: ``float32`` evaluates in single precision with the tables stored so, for embedded
      targets; fits and tables are computed in double precision first.
    name: the base of the name of ``function()``.

  Calling the spline evaluates it: at a point (a scalar in 1-D, a ``(D,)`` vector) or at a batch
  (``(N,)`` in 1-D, ``(N, D)``), as one vectorized graph. Derivatives with respect to the point come
  from differentiating that graph; at a knot they are the one-sided ones of the cell to the right.
  """

  def __init__(
    self,
    knots: np.ndarray | Sequence[float] | tuple[np.ndarray, ...],
    coeffs: np.ndarray | Expr,
    degree: int | tuple[int, ...] = 3,
    *,
    extrap: Extrap | tuple[Extrap | None, ...] | None = None,
    fill: float = math.nan,
    search: Search | Literal["auto"] | tuple[Search | Literal["auto"], ...] = "auto",
    strategy: Strategy = "auto",
    dtype: DType | str = "float64",
    name: str = "interp",
  ) -> None:
    grids = knots if isinstance(knots, tuple) else (knots,)
    ndim = len(grids)
    degrees, extraps, searches = _per_axis(degree, ndim, "degree"), _per_axis(extrap, ndim, "extrap"), _per_axis(search, ndim, "search")
    axes = tuple(
      Axis(np.asarray(t, dtype=np.float64), k, extrap=e, fill=fill, search=s) for t, k, e, s in zip(grids, degrees, extraps, searches, strict=True)
    )
    self._setup(axes, coeffs, strategy, name, dtype)

  @classmethod
  def from_axes(
    cls, axes: Sequence[Axis], coeffs: np.ndarray | Expr, *, strategy: Strategy = "auto", dtype: DType | str = "float64", name: str = "interp"
  ) -> BSpline:
    """A spline over prepared axes (a fit's, whose partitions its data sites refine)."""
    self = cls.__new__(cls)
    self._setup(tuple(axes), coeffs, strategy, name, dtype)
    return self

  def _setup(self, axes: tuple[Axis, ...], coeffs: Any, strategy: Strategy, name: str, dtype: DType | str) -> None:
    if not axes:
      raise ValueError("a spline needs at least one axis")
    self.dtype = as_dtype(dtype)
    if self.dtype not in (dtypes.float32, dtypes.float64):
      raise ValueError(f"dtype must be float32 or float64, got {self.dtype}")
    axes = tuple(ax if ax.dtype == self.dtype else ax.retyped(self.dtype) for ax in axes)  # tables about the centers it holds
    if strategy not in ("auto", "pp", "basis"):
      raise ValueError(f"strategy must be 'auto', 'pp' or 'basis', got {strategy!r}")
    shape = tuple(ax.n for ax in axes)
    if isinstance(coeffs, Expr):
      coeffs = coeffs if coeffs.type.dtype == self.dtype else cast(coeffs, self.dtype)
      if strategy == "pp":
        raise ValueError("strategy='pp' tabulates constant coefficients; an Expr's are evaluated from the local bases")
      strategy = "basis"
    else:
      coeffs = np.asarray(coeffs, dtype=np.float64)
      if not np.all(np.isfinite(coeffs)):
        raise ValueError("coeffs must be finite")
    if coeffs.shape[: len(axes)] != shape:
      raise ValueError(f"coeffs must have shape {shape} + out_shape for these knots and degrees, got {coeffs.shape}")
    self.axes, self.name = axes, name
    self.coeffs: np.ndarray | Expr = coeffs
    self.out_shape: tuple[int, ...] = coeffs.shape[len(axes) :]
    out = math.prod(self.out_shape)
    pp_bytes = 8 * math.prod(ax.table_cells * (ax.degree + 1) for ax in axes) * out
    basis_bytes = 8 * (math.prod(ax.n for ax in axes) * out + sum(ax.table_cells * ((ax.degree + 1) ** 2 + 1) for ax in axes))
    self.strategy: Literal["pp", "basis"] = ("pp" if pp_bytes <= max(PP_BUDGET, basis_bytes) else "basis") if strategy == "auto" else strategy
    self._pp: np.ndarray | None = None
    table = pp_bytes if self.strategy == "pp" else basis_bytes
    if self.constant and table // 8 > LARGE_TABLE:
      warnings.warn(
        f"{name}: {table // 8} tabulated values, some {20 * table // 8 >> 20} to {32 * table // 8 >> 20} MB of C source to compile "
        "(more where several generated functions read the table); an Expr for the data (a Function input) keeps it out of the code",
        skip_file_prefixes=(os.path.dirname(__file__),),
      )

  @property
  def constant(self) -> bool:
    """Whether the coefficients are numbers (tabulated in the C) rather than an ``Expr``."""
    return not isinstance(self.coeffs, Expr)

  def _numeric(self, what: str) -> np.ndarray:
    if not isinstance(self.coeffs, np.ndarray):
      raise ValueError(f"{what} needs constant coefficients")
    return self.coeffs

  def _flat_coeffs(self) -> Expr:
    if isinstance(self.coeffs, np.ndarray):
      return Expr.const(self.coeffs.reshape(-1), dtype=self.dtype)
    return self.coeffs.reshape((self.coeffs.size,))

  @property
  def ndim(self) -> int:
    """The number of axes, the dimension of a point."""
    return len(self.axes)

  @property
  def degree(self) -> tuple[int, ...]:
    """The degree along each axis."""
    return tuple(ax.degree for ax in self.axes)

  @property
  def knots(self) -> tuple[np.ndarray, ...]:
    """The knot vector of each axis."""
    return tuple(ax.knots for ax in self.axes)

  def __repr__(self) -> str:
    grid = " x ".join(f"{ax.cells}" for ax in self.axes)
    kind = "" if self.dtype == dtypes.float64 else f", dtype={self.dtype.name}"
    return f"BSpline({self.name!r}, degree={self.degree}, cells={grid}, out_shape={self.out_shape}, strategy={self.strategy!r}{kind})"

  def _points(self, x: Any) -> tuple[Expr, bool]:
    """``x`` in the spline's float type, and whether it is a batch."""
    x = as_expr(x)
    if x.type.dtype != self.dtype:
      x = cast(x, self.dtype)
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
    """The cells of the points ``x``, to pass as ``index=`` to several evaluations at them by splines
    on the same partition: this one and its derivatives, say."""
    x, batch = self._points(x)
    if not batch:
      return Index(tuple(ax.cell(p) for ax, p in zip(self.axes, self._coordinates(x), strict=True)), self._partition())
    n = x.shape[0]
    cells = vmap(self._search_function(), n, [(x.reshape((x.size,)), 0, x.size // n)]).reshape((n, self.ndim))
    return Index(tuple(cells[:, d] for d in range(self.ndim)), self._partition())

  def _partition(self) -> str:
    """What decides the cells an index holds: each axis's edges, side, search and wrap."""
    h = hashlib.sha1()
    for ax in self.axes:
      h.update(repr((ax.side, ax.search, ax.extrap == "periodic", ax.dtype.name)).encode())
      h.update(ax.edges.tobytes())
    return h.hexdigest()[:12]

  def _check_index(self, index: Index, shape: tuple[int, ...]) -> None:
    if len(index.cells) != self.ndim or any(c.shape != shape for c in index.cells):
      raise ValueError("index= was found for points of another shape")
    if index.partition != self._partition():
      raise ValueError("index= was found on another partition: its cells are not this spline's")

  def __call__(self, x: Any, *, index: Index | None = None) -> Expr:
    x, batch = self._points(x)
    if not batch:
      if index is not None:
        self._check_index(index, ())
      coords = self._coordinates(x)
      cells = index.cells if index is not None else tuple(ax.cell(p) for ax, p in zip(self.axes, coords, strict=True))
      return self._at(coords, cells)
    # A batch is one mapped call per point: its derivatives differentiate the point's graph once, and
    # its code is one loop whatever the batch size.
    n = x.shape[0]
    flat = (x.reshape((x.size,)), 0, x.size // n)
    coeffs = [] if self.constant else [(self._flat_coeffs(), 0, 0)]  # broadcast to every point
    if index is None:
      out = vmap(self.function(), n, [flat, *coeffs])
    else:
      self._check_index(index, (n,))
      cells = stack(list(index.cells), axis=1).reshape((n * self.ndim,))
      out = vmap(self._cell_function(), n, [flat, (cells, 0, self.ndim), *coeffs])
    return out.reshape((n, *self.out_shape))

  def _at(self, coords: list[Expr], cells: tuple[Expr, ...]) -> Expr:
    """The value at one point, from its coordinates and cells."""
    local: list[tuple[Expr, Expr]] = []  # per axis: the tabulated cell, and s
    guards: list[tuple[Expr, float]] = []
    for ax, p, j in zip(self.axes, coords, cells, strict=True):
      if ax.degree == 0:  # a constant never reads its coordinate, so NaN is passed on by hand
        guards.append((not_equal(p, p), math.nan))
      if ax.extrap in ("clamp", "fill"):
        if ax.extrap == "fill":
          guards.append(((p < ax.lo) | (p > ax.hi), ax.fill))
        p = where(p < ax.lo, ax.lo, where(p > ax.hi, ax.hi, p))  # NaN passes through
      if ax.outer or (ax.extrap == "extend" and self.strategy == "basis"):
        # The continuation is followed out to where the k-th power of the distance still fits a
        # float, and held beyond: an infinity or a point past 1e300 ** (1 / k) (float32: 1e30 **
        # (1 / k)) then never meets an outer cell's zero coefficient as 0 * inf, forward or
        # reverse, nor a basis function's overflowed value as inf - inf. Nothing in reach of a
        # grid is changed. An extend axis under "pp" needs no hold: its infinities are clean.
        far = 10.0 ** ((30.0 if self.dtype == dtypes.float32 else 300.0) / max(ax.degree, 1))
        p = where(p < -far, -far, where(p > far, far, p))
      i = cast(ax.table_cell(p, j), dtypes.int64)
      spacing = None if ax.outer else ax.spacing
      if spacing is not None:  # the centers of uniform cells are arithmetic: no table
        center = num(spacing[0], p) + cast(i, self.dtype) * num(spacing[1], p)
      else:
        center = _gather(ax.table_centers, i, self.dtype)
      local.append((i, p - center))
    value = self._pp_eval(local) if self.strategy == "pp" else self._basis_eval(local)
    for out, fill in reversed(guards):
      value = where(out, fill, value)
    return value

  def _pp_table(self) -> np.ndarray:
    """Cell-major, then the ``prod (k_d + 1)`` powers, then the output: the Taylor coefficients of
    every cell at its center."""
    if self._pp is None:
      t = self._numeric("strategy='pp'")
      for d, ax in enumerate(self.axes):
        t = np.moveaxis(ax.table_taylor(np.moveaxis(t, 2 * d, 0)), [0, 1], [2 * d, 2 * d + 1])
      nd = self.ndim
      self._pp = np.ascontiguousarray(t.transpose([*range(0, 2 * nd, 2), *range(1, 2 * nd, 2), *range(2 * nd, t.ndim)])).reshape(-1)
    return self._pp

  def _pp_eval(self, local: list[tuple[Expr, Expr]]) -> Expr:
    orders = [ax.degree + 1 for ax in self.axes]
    base = local[0][0]
    for (i, _), ax in zip(local[1:], self.axes[1:], strict=True):
      base = base * ax.table_cells + i
    coef = _block(self._pp_table(), base, math.prod(orders) * math.prod(self.out_shape), self.dtype).reshape((*orders, *self.out_shape))
    for d in reversed(range(self.ndim)):  # nested Horner, the last axis innermost
      coef = _horner([coef[(*(slice(None),) * d, m)] for m in range(orders[d])], local[d][1])
    return coef

  def _basis_eval(self, local: list[tuple[Expr, Expr]]) -> Expr:
    orders = [ax.degree + 1 for ax in self.axes]
    out_size = math.prod(self.out_shape)
    strides = [math.prod(ax.n for ax in self.axes[d + 1 :]) * out_size for d in range(self.ndim)]
    bases, first = [], []
    for (i, s), ax, stride in zip(local, self.axes, strides, strict=True):
      k1 = ax.degree + 1
      mats = _block(ax.table_local.reshape(-1), i, k1 * k1, self.dtype).reshape((k1, k1))
      bases.append(_horner([mats[:, m] for m in range(k1)], s))
      first.append(cast(_gather(ax.table_offsets.astype(np.float64), i, self.dtype), dtypes.int64) * stride)  # exact below 2^24 in float32
    offsets = sum(
      np.arange(k).reshape((*(1,) * d, k, *(1,) * (self.ndim - d - 1))) * stride for d, (k, stride) in enumerate(zip(orders, strides, strict=True))
    )
    offsets = (np.asarray(offsets).reshape(-1)[:, None] + np.arange(out_size)[None, :]).reshape(-1)
    base = first[0]
    for f in first[1:]:
      base = base + f
    index = stack([base]) + Expr.const(offsets, dtype=dtypes.int64)
    coef = take(self._flat_coeffs(), index, in_range=True)
    for b, k in zip(bases, orders, strict=True):  # contract one axis at a time
      coef = b @ coef.reshape((k, coef.size // k))
    return coef.reshape(self.out_shape)

  def _search_function(self) -> ConcreteFunction[Any, Any, Any, Any]:
    point = () if self.ndim == 1 else (self.ndim,)
    return _interned(
      f"{self.name}_{self.digest}_cell",
      self.digest,
      lambda: ConcreteFunction(
        f"{self.name}_{self.digest}_cell",
        lambda x: stack([ax.cell(p) for ax, p in zip(self.axes, self._coordinates(x), strict=True)]),
        param_list(L("x", point, dtype=self.dtype)),
        L("cell", (self.ndim,), dtype=self.dtype),
      ),
    )

  def _cell_function(self) -> ConcreteFunction[Any, Any, Any, Any]:
    point = () if self.ndim == 1 else (self.ndim,)
    name = f"{self.name}_{self.digest}_at"
    if self.constant:
      return _interned(
        name,
        self.digest,
        lambda: ConcreteFunction(
          name,
          lambda x, cell: self._at(self._coordinates(x), tuple(cell[d] for d in range(self.ndim))),
          param_list(L("x", point, dtype=self.dtype), L("cell", (self.ndim,), dtype=self.dtype)),
          L("y", self.out_shape, dtype=self.dtype),
        ),
      )
    return _interned(
      name,
      self.digest,
      lambda: ConcreteFunction(
        name,
        lambda x, cell, c: self._rebound(c)._at(self._coordinates(x), tuple(cell[d] for d in range(self.ndim))),
        param_list(L("x", point, dtype=self.dtype), L("cell", (self.ndim,), dtype=self.dtype), L("c", (self.coeffs.size,), dtype=self.dtype)),
        L("y", self.out_shape, dtype=self.dtype),
      ),
    )

  def _rebound(self, flat: Expr) -> BSpline:
    """This spline over other coefficients, given flat: a Function's own input."""
    return BSpline.from_axes(self.axes, flat.reshape(self.coeffs.shape), strategy="basis", dtype=self.dtype, name=self.name)

  @property
  def digest(self) -> str:
    """Eight hex digits naming the spline's content, for ``function()`` names that never collide:
    the axes and the coefficients, or for ``Expr`` coefficients (an input of the Function) their
    shape."""
    h = hashlib.sha1(f"{self.strategy} {self.dtype.name}".encode())
    for ax in self.axes:
      h.update(repr((ax.degree, ax.side, ax.extrap, ax.fill, ax.search)).encode())
      h.update(ax.knots.tobytes())
      h.update(ax.edges.tobytes())
    h.update(repr((self.coeffs.shape, self.constant)).encode())
    if isinstance(self.coeffs, np.ndarray):
      h.update(np.ascontiguousarray(self.coeffs).tobytes())
    return h.hexdigest()[:8]

  def function(self, name: str | None = None) -> ConcreteFunction[Any, Any, Any, Any]:
    """The spline as a Function of one point, ``x -> f(x)``: its calls share one procedure, where
    inlining would give each calling procedure its own copy of the evaluation (a derivative of it
    is a procedure of its own, with its own copy of the table). With ``Expr`` coefficients it is
    ``(x, c) -> f(x)``, ``c`` the coefficients flat. Named ``{name}_{digest}`` unless ``name`` is
    given; two splines with the same content share it, and two with different content never do,
    though a graph holding both under one given name is refused when it is lowered."""
    point = () if self.ndim == 1 else (self.ndim,)
    fname = name or f"{self.name}_{self.digest}"
    if self.constant:
      return _interned(
        fname,
        self.digest,
        lambda: ConcreteFunction(fname, lambda x: self(x), param_list(L("x", point, dtype=self.dtype)), L("y", self.out_shape, dtype=self.dtype)),
      )
    return _interned(
      fname,
      self.digest,
      lambda: ConcreteFunction(
        fname,
        lambda x, c: self._rebound(c)(x),
        param_list(L("x", point, dtype=self.dtype), L("c", (self.coeffs.size,), dtype=self.dtype)),
        L("y", self.out_shape, dtype=self.dtype),
      ),
    )

  def pack(self, coeffs: Any) -> np.ndarray:
    """The flat buffer ``function()`` takes as its coefficient input ``c``: coefficients of this
    spline's shape, in C order. For callers of generated code that fill the table at run time."""
    arr = np.asarray(coeffs, dtype=np.float64)
    if arr.shape != self.coeffs.shape:
      raise ValueError(f"pack() takes coefficients of shape {self.coeffs.shape}, got {arr.shape}")
    return np.ascontiguousarray(arr).reshape(-1)

  def _with_axis(self, axis: int, new: Axis, coeffs: Any, suffix: str) -> BSpline:
    axes = (*self.axes[:axis], new, *self.axes[axis + 1 :])
    return BSpline.from_axes(axes, coeffs, strategy=self.strategy, dtype=self.dtype, name=f"{self.name}_{suffix}")

  def derivative(self, nu: int = 1, axis: int = 0) -> BSpline:
    """The ``nu``-th partial derivative along ``axis``, as a spline of degree ``k - nu``: its
    coefficients by de Boor's differencing, on the same partition, so an ``index`` found for this
    spline serves it too. Outside, it is the derivative of this spline's extrapolation: ``linear``
    gives the end slope held (``clamp``), ``clamp`` and ``fill`` give zero (``fill``),
    ``extend`` and ``periodic`` carry over. For a Jacobian, differentiate the spline instead; this
    is for when the derivative must itself be a spline (a curvature table, a bound on ``f'``)."""
    if not 0 <= axis < self.ndim:
      raise ValueError(f"axis must be 0 to {self.ndim - 1} for a {self.ndim}-D spline, got {axis}")
    ax = self.axes[axis]
    if not 0 <= nu <= ax.degree:
      raise ValueError(f"nu must be 0 to the axis's degree {ax.degree}, got {nu}")
    t, k, coeffs, extrap, fill = ax.knots, ax.degree, self.coeffs, ax.extrap, ax.fill
    for _ in range(nu):
      coeffs = _along(derivative_matrix(t, k), coeffs, axis)
      t, k = t[1:-1], k - 1
      extrap, fill = {"linear": ("clamp", fill), "clamp": ("fill", 0.0), "fill": ("fill", 0.0)}.get(extrap, (extrap, fill))
    new = Axis(t, k, edges=ax.edges, side=ax.side, extrap=extrap, fill=fill, search=ax.search)
    return self if nu == 0 else self._with_axis(axis, new, coeffs, f"d{nu}{axis}")

  def antiderivative(self, axis: int = 0) -> BSpline:
    """The integral along ``axis`` from the start of the base interval, a spline of degree
    ``k + 1`` on the same partition. Outside, it integrates this spline's extrapolation where that is
    a spline extrapolation too: ``extend`` stays ``extend``, ``clamp`` becomes ``linear``, a zero
    ``fill`` becomes ``clamp`` and a NaN one stays. For ``linear`` it continues its end polynomials
    (``extend``), which integrate the end tangent only to first order; a periodic or other constant
    fill has no such form and is refused."""
    ax = self.axes[axis]
    if ax.extrap == "periodic" or (ax.extrap == "fill" and ax.fill != 0.0 and not math.isnan(ax.fill)):
      raise ValueError(f"the antiderivative of a spline with extrap={ax.extrap!r} (fill={ax.fill}) is not a spline extrapolation")
    t, k, n = ax.knots, ax.degree, ax.n
    knots = np.concatenate([t[:1], t, t[-1:]])
    step = (t[k + 1 : k + 1 + n] - t[:n]) / (k + 1)
    start = design_matrix([knots], [k + 1], np.array([[ax.lo]])).toarray()[0]
    if isinstance(self.coeffs, np.ndarray):
      moved = np.moveaxis(self.coeffs, axis, 0)
      cum = np.concatenate([np.zeros((1, *moved.shape[1:])), np.cumsum(moved * step.reshape(n, *(1,) * (moved.ndim - 1)), axis=0)])
      coeffs = np.moveaxis(cum - np.tensordot(start, cum, axes=(0, 0)), 0, axis)
    else:  # the cumulative sum, less its value at the start, as one constant map
      cumsum = np.tril(np.ones((n + 1, n)), -1) * step[None, :]
      coeffs = _along(cumsum - np.outer(np.ones(n + 1), start @ cumsum), self.coeffs, axis)
    after: dict[str, tuple[Extrap, float]] = {
      "extend": ("extend", ax.fill),
      "linear": ("extend", ax.fill),
      "clamp": ("linear", ax.fill),
      "fill": ("clamp" if ax.fill == 0.0 else "fill", ax.fill),
    }
    extrap, fill = after[ax.extrap]
    new = Axis(knots, k + 1, edges=ax.edges, side=ax.side, extrap=extrap, fill=fill, search=ax.search)
    return self._with_axis(axis, new, coeffs, f"i{axis}")

  def integrate(self, a: Any, b: Any) -> Any:
    """The integral over ``[a, b]`` (a box ``a <= x <= b`` in n-D). Numbers give a number, or an
    array for a vector-valued spline, exact for every extrapolation mode in 1-D and inside the base
    box in n-D. ``Expr`` bounds give an ``Expr``, exact in 1-D for every mode; in n-D it is the
    antiderivative at the corners of the box, exact inside the base box (see ``antiderivative`` for
    what it integrates outside)."""
    numeric = all(
      not isinstance(v, Expr)
      for v in (np.ravel(a).tolist() if not isinstance(a, Expr) else [a]) + (np.ravel(b).tolist() if not isinstance(b, Expr) else [b])
    )
    if numeric and self.ndim == 1 and isinstance(self.coeffs, np.ndarray):
      return _integrate_1d(self, float(np.asarray(a)), float(np.asarray(b)))
    if numeric and self.ndim == 1:  # linear in the coefficients: the basis functions' integrals, and a fill's
      u, v, ax = float(np.asarray(a)), float(np.asarray(b)), self.axes[0]
      offset = float(_integrate_1d(BSpline.from_axes(self.axes, np.zeros(ax.n)), u, v))
      weights = _integrate_1d(BSpline.from_axes(self.axes, np.eye(ax.n)), u, v) - offset
      return _along(weights[None, :], self.coeffs, 0).reshape(self.out_shape) + offset
    if numeric:
      lo, hi = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
      if lo.size != self.ndim or hi.size != self.ndim:
        raise ValueError(f"a {self.ndim}-D integral needs {self.ndim} lower and upper bounds")
      from scipy.interpolate import BSpline as SciBSpline

      out = self.coeffs
      for ax, u, v in zip(self.axes, lo, hi, strict=True):
        if min(u, v) < ax.lo or max(u, v) > ax.hi:
          raise ValueError("an n-D integral with numeric bounds must lie inside the base box")
        out = _along(SciBSpline(ax.knots, np.eye(ax.n), ax.degree).integrate(u, v)[None, :], out, 0)
        out = out.reshape(out.shape[1:])
      return out
    if self.ndim == 1:
      primitive = self._primitive()
      return primitive(as_expr(b)) - primitive(as_expr(a))
    total = self
    for d in range(self.ndim):
      total = total.antiderivative(d)
    corners = []
    for signs in np.ndindex(*(2,) * self.ndim):
      point = [as_expr(b if bit else a) if self.ndim == 1 else as_expr((b if bit else a)[d]) for d, bit in enumerate(signs)]
      value = total(point[0] if self.ndim == 1 else stack(point))
      corners.append(value if (self.ndim - sum(signs)) % 2 == 0 else -value)
    result = corners[0]
    for c in corners[1:]:
      result = result + c
    return result

  def _primitive(self) -> Callable[[Expr], Expr]:
    """The integral from the start of the base interval of a 1-D spline and its continuation, as
    an expression of the upper bound: the antiderivative inside; past each end the integral of the
    continuation there (the tangent's is quadratic, a fill's linear); and whole periods counted for
    a periodic axis. ``extend`` is the antiderivative's own continuation."""
    ax = self.axes[0]
    if ax.extrap == "extend":
      return self.antiderivative()
    inside = Axis(ax.knots, ax.degree, edges=ax.edges, side=ax.side, extrap="clamp", search=ax.search, dtype=ax.dtype)
    inner = self._with_axis(0, inside, self.coeffs, "in").antiderivative()  # read inside only
    if ax.extrap == "periodic":
      period, whole = ax.hi - ax.lo, inner.at(np.array([ax.hi]))[0]

      def periodic(x: Expr) -> Expr:
        x = x if x.type.dtype == self.dtype else cast(x, self.dtype)
        turns = ((x - ax.lo) * (1.0 / period)).floor()
        return turns * whole + inner(ax.wrap(x))

      return periodic
    ends = self.at(np.array([ax.lo, ax.hi]))  # the values at the ends, as the coefficients are
    slopes = self.derivative().at(np.array([ax.lo, ax.hi])) if ax.extrap == "linear" else None

    def tail(x: Expr, row: int, end: float) -> Expr:
      d = x - end
      if ax.extrap == "fill":
        return ax.fill * d
      value = ends[row] * d
      return value if slopes is None else value + slopes[row] * (0.5 * d * d)

    def primitive(x: Expr) -> Expr:
      x = x if x.type.dtype == self.dtype else cast(x, self.dtype)
      xc = where(x < ax.lo, ax.lo, where(x > ax.hi, ax.hi, x))
      below, above = tail(x, 0, ax.lo), tail(x, 1, ax.hi)
      return inner(xc) + where(x < ax.lo, below, where(x > ax.hi, above, 0.0 * below))

    return primitive

  def inverse(self, *, tol: float = 1e-15, max_iter: int = 60) -> BSpline | Inverse:
    """``x = f^{-1}(y)`` for a 1-D scalar spline that is strictly monotone (checked). Degree 1 gives
    the exact table with the axes swapped, a ``BSpline``; a higher degree an ``Inverse``, solved by
    safeguarded Newton. Outside the spline's range the inverse continues linearly."""
    if self.ndim != 1 or self.out_shape:
      raise ValueError("inverse() needs a 1-D scalar spline")
    self._numeric("inverse()")
    if self.dtype != dtypes.float64:
      raise ValueError("inverse() evaluates in float64")
    ax = self.axes[0]
    if ax.extrap == "periodic" or ax.degree == 0:
      raise ValueError(f"a spline of degree {ax.degree} with extrap={ax.extrap!r} is not invertible")
    sign = _monotone_sign(self)
    values = self.to_scipy()(ax.edges)
    if ax.degree == 1:
      order = slice(None) if sign > 0 else slice(None, None, -1)
      extrap = {"extend": "extend", "clamp": "clamp"}.get(ax.extrap, "fill")
      return BSpline.from_axes(
        [Axis(np.concatenate([values[order][:1], values[order], values[order][-1:]]), 1, edges=values[order], extrap=extrap)],  # ty: ignore[invalid-argument-type]
        ax.edges[order],
        strategy=self.strategy,
        name=f"{self.name}_inv",
      )
    return Inverse(self, sign, values, tol, max_iter)

  def to_scipy(self) -> Any:
    """The same spline as a SciPy ``BSpline`` (1-D) or ``NdBSpline``, extrapolating by its end
    polynomials; for tests and plotting."""
    from scipy.interpolate import BSpline as SciBSpline
    from scipy.interpolate import NdBSpline

    coeffs = self._numeric("to_scipy()")
    if self.ndim == 1:
      return SciBSpline(self.axes[0].knots, coeffs, self.axes[0].degree)
    return NdBSpline(self.knots, coeffs, self.degree)

  def basis(self, points: Any) -> sparse.csr_array:
    """The design matrix at points known now, ``(m,)`` in 1-D or ``(m, D)``: an ``(m, prod n_d)``
    sparse matrix whose row ``i`` is the tensor-product basis at point ``i`` with this spline's
    extrapolation (zero where a ``fill`` axis is left), its columns the coefficients' C order.
    ``basis(points) @ coeffs`` is the spline at the points, less the ``fill`` value where a point
    leaves a ``fill`` axis (``at`` adds it)."""
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[:, None] if self.ndim == 1 and pts.ndim == 1 else pts
    if pts.ndim != 2 or pts.shape[1] != self.ndim or not np.all(np.isfinite(pts)):
      raise ValueError(f"points must be finite, shaped (m,) in 1-D or (m, {self.ndim}), got {pts.shape}")
    m = pts.shape[0]
    cols, vals, filled = np.zeros((m, 1), dtype=np.int64), np.ones((m, 1)), np.zeros(m, dtype=bool)
    for d, ax in enumerate(self.axes):
      b, idx, out = ax.local_basis(pts[:, d])
      cols = (cols[:, :, None] * ax.n + idx[:, None, :]).reshape(m, -1)
      vals = (vals[:, :, None] * b[:, None, :]).reshape(m, -1)
      filled |= out
    vals[filled] = 0.0
    return sparse.csr_array(
      (vals.reshape(-1), (np.repeat(np.arange(m), cols.shape[1]), cols.reshape(-1))), shape=(m, math.prod(ax.n for ax in self.axes))
    )

  def at(self, points: Any) -> Expr:
    """The spline at points known now, ``(m, *out_shape)``: ``basis(points)`` times the coefficients
    as one sparse product, so its Jacobian with respect to ``Expr`` coefficients has exactly the
    basis's pattern, and a constant for constant coefficients. The way to fit a table to
    measurements, or to parametrize a trajectory by a spline."""
    B = self.basis(points)
    m, out_size = B.shape[0], math.prod(self.out_shape)
    fills = np.zeros(m)
    pts = np.asarray(points, dtype=np.float64).reshape(m, self.ndim)
    for d in reversed(range(self.ndim)):
      ax = self.axes[d]
      if ax.extrap == "fill":
        fills = np.where((pts[:, d] < ax.lo) | (pts[:, d] > ax.hi), ax.fill, fills)
    fills = fills[:, None]
    if isinstance(self.coeffs, np.ndarray):
      return Expr.const(((B @ self.coeffs.reshape(-1, out_size)) + fills).reshape((m, *self.out_shape)), dtype=self.dtype)
    from ..linalg.sparse import SparseMatrix

    coeffs = self.coeffs if self.dtype == dtypes.float64 else cast(self.coeffs, dtypes.float64)  # the product's values are float64
    values = SparseMatrix.from_scipy(B) @ coeffs.reshape((B.shape[1], out_size))
    values = (values + Expr.const(fills) if np.any(fills) else values).reshape((m, *self.out_shape))
    return values if self.dtype == dtypes.float64 else cast(values, self.dtype)


def _along(matrix: Any, values: Any, axis: int) -> Any:
  """``matrix`` (dense or SciPy sparse, constant) applied to ``values`` along ``axis``: NumPy for a
  NumPy array, a constant product (sparse through ``SparseMatrix``) for an ``Expr``."""
  if isinstance(values, np.ndarray):
    moved = np.moveaxis(values, axis, 0)
    out = np.asarray(matrix @ moved.reshape(moved.shape[0], -1)).reshape(matrix.shape[0], *moved.shape[1:])
    return np.moveaxis(out, 0, axis)
  from ..linalg.sparse import SparseMatrix

  order = (axis, *(d for d in range(len(values.shape)) if d != axis))
  moved = values.transpose(order) if axis else values
  rows = moved.shape[0]
  flat = moved.reshape((rows, moved.size // rows))
  if sparse.issparse(matrix) and values.type.dtype == dtypes.float64:
    out = SparseMatrix.from_scipy(sparse.csr_array(matrix)) @ flat
  else:
    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    out = Expr.const(dense, dtype=values.type.dtype) @ flat
  out = out.reshape((matrix.shape[0], *moved.shape[1:]))
  return out.transpose(tuple(int(i) for i in np.argsort(order))) if axis else out


def _integrate_1d(f: BSpline, a: float, b: float) -> Any:
  """The exact integral over ``[a, b]`` of a 1-D spline and its extrapolation."""
  if a > b:
    return -_integrate_1d(f, b, a)
  ax, s = f.axes[0], f.to_scipy()
  if ax.extrap in ("extend", "periodic"):
    return s.integrate(a, b, extrapolate=True if ax.extrap == "extend" else "periodic")
  total = s.integrate(max(a, ax.lo), min(b, ax.hi)) if a < ax.hi and b > ax.lo else np.zeros(f.out_shape)
  for end, u, v in ((ax.lo, a, min(b, ax.lo)), (ax.hi, max(a, ax.hi), b)):  # the part of [a, b] beyond each end
    if v > u:
      if ax.extrap == "fill":
        total = total + ax.fill * (v - u)
      else:  # clamp: the end value; linear: the tangent there, integrated over [u, v]
        total = total + s(end) * (v - u) + (s(end, 1) * ((v - end) ** 2 - (u - end) ** 2) / 2 if ax.extrap == "linear" else 0.0)
  return total


def _monotone_sign(f: BSpline) -> int:
  """+1 or -1 when the 1-D spline is strictly monotone: its derivative keeps one sign on every cell
  (checked at the cell ends and at the derivative's own extrema) and its values at the partition
  edges are strictly monotone."""
  ax = f.axes[0]
  values = f.to_scipy()(ax.edges)
  steps = np.diff(values)
  sign = 1 if np.all(steps > 0) else -1 if np.all(steps < 0) else 0
  if sign == 0:
    raise ValueError("inverse() needs a strictly monotone spline; its values at the knots are not")
  if ax.degree >= 2:
    from scipy.interpolate import PPoly

    deriv = f.derivative().to_scipy()
    probe = [ax.edges, np.nextafter(ax.edges[1:], -np.inf)]
    if ax.degree >= 3:  # the derivative's extrema inside a cell are roots of the second derivative
      roots = PPoly.from_spline(f.derivative(2).to_scipy()).roots(extrapolate=False)
      probe.append(roots[np.isfinite(roots)])
    slopes = sign * deriv(np.concatenate(probe))
    if np.min(slopes) < -1e-12 * float(np.max(np.abs(slopes))):
      raise ValueError("inverse() needs a strictly monotone spline; its derivative changes sign")
  return sign


class Inverse:
  """``x = f^{-1}(y)`` for a strictly monotone 1-D spline of degree 2 or more, from
  ``BSpline.inverse``. Newton's method on the cell the linear table through the spline's values at
  its partition edges names, bisecting whenever a step would leave the cell (``roots.NewtonBisection``);
  the derivative is ``dx/dy = 1/f'(x)`` by ``custom_derivative``, so it never differentiates the
  iterations. Outside the range of ``f`` the inverse continues along its end tangent. Calling it
  works as for a spline: a scalar, or a batch ``(N,)`` as one map."""

  def __init__(self, f: BSpline, sign: int, values: np.ndarray, tol: float, max_iter: int) -> None:
    self.f, self.sign, self.tol, self.max_iter = f, sign, tol, max_iter
    self._levels = sign * values  # increasing: the values of sign * f at the edges
    self._table = Axis(self._levels, 0, extrap="clamp")

  def __repr__(self) -> str:
    return f"Inverse({self.f!r})"

  def function(self) -> ConcreteFunction[Any, Any, Any, Any]:
    """The inverse as a Function of one value, ``y -> x``, with its derivative rule attached."""
    settings = hashlib.sha1(repr((float(self.tol), int(self.max_iter))).encode()).hexdigest()[:6]
    name = f"{self.f.name}_{self.f.digest}_inv{settings}"
    return _interned(name, self.f.digest, lambda: self._build(name))

  def __call__(self, y: Any) -> Expr:
    y = as_expr(y)
    if y.type.dtype != dtypes.float64:
      y = cast(y, dtypes.float64)
    if not y.shape:
      return self.function()(y)
    if len(y.shape) != 1:
      raise ValueError(f"an inverse takes a scalar or a batch (N,), got shape {y.shape}")
    return vmap(self.function(), y.shape[0], [(y, 0, 1)])

  def _build(self, name: str) -> ConcreteFunction[Any, Any, Any, Any]:
    f, sign, edges, levels = self.f, float(self.sign), self.f.axes[0].edges, self._levels
    deriv = f.derivative()

    def g(x: Expr, j: Expr) -> Expr:
      return sign * f._at([x], (j,))

    def dg(x: Expr, j: Expr) -> Expr:
      return sign * deriv._at([x], (j,))

    # Outside the range the inverse continues along the end tangents of sign * f, whose slopes are
    # known now; a flat end (a slope negligible against the mean) holds the end instead, as the
    # tangent there never reaches a value outside, and an ulp past the end (the spline's value at its
    # end and the data's can differ by one) would otherwise go a long way.
    mean = (levels[-1] - levels[0]) / (edges[-1] - edges[0])
    ends = [sign * float(v) for v in deriv.to_scipy()(edges[[0, -1]])]
    rates = [1.0 / e if e > 1e-12 * mean else 0.0 for e in ends]

    def clamp(y: Expr) -> tuple[Expr, Expr, Expr]:
      z = sign * y
      zc = where(z < float(levels[0]), float(levels[0]), where(z > float(levels[-1]), float(levels[-1]), z))
      return z, zc, equal(z, zc)

    def outside_rate(z: Expr, zc: Expr) -> Expr:
      return where(z < zc, rates[0], rates[1])

    def solve(y: Expr) -> Expr:
      z, zc, inside = clamp(y)
      j = self._table.cell(zc)
      i = cast(j, dtypes.int64)
      x_lo, x_hi = _gather(edges, i), _gather(edges, i + 1)
      z_lo, z_hi = _gather(levels, i), _gather(levels, i + 1)
      x0 = x_lo + (zc - z_lo) * (x_hi - x_lo) / (z_hi - z_lo)
      # Safeguarded Newton on the cell: it stops once a step moves x by no more than tol times the
      # larger of |x| and the cell's width, or the bracket closes to that, or the residual is zero.
      x, _ = NewtonBisection(tol=self.tol, max_iter=self.max_iter).iterate(
        lambda x, p: g(x, p[1]) - p[0], x0, x_lo, x_hi, [zc, j], name=name, names=("z", "j"), slope=lambda x, p: dg(x, p[1])
      )
      return where(inside, x, x + (z - zc) * outside_rate(z, zc))

    slope = deriv.function()

    def rate(y: Expr, x: Expr) -> Expr:
      """``dx/dy``: ``1/f'(x)`` inside, the end tangent's (zero at a flat end) outside, NaN at NaN."""
      z, zc, inside = clamp(y)
      return where(not_equal(y, y), math.nan, where(inside, 1.0 / slope(x), sign * outside_rate(z, zc)))

    primal = ConcreteFunction(f"{name}_solve", solve, param_list(L("y", ())), L("x", ()))
    jvp = ConcreteFunction(f"{name}_jvp", lambda y, dy: dy * rate(y, as_expr(primal(y))), param_list(L("y", ()), L("dy", ())), L("dx", ()))
    vjp = ConcreteFunction(f"{name}_vjp", lambda y, x, xbar: xbar * rate(y, x), param_list(L("y", ()), L("x", ()), L("xbar", ())), L("ybar", ()))
    return custom_derivative(primal, jvp=jvp, vjp=vjp)


def design_matrix(knots: Sequence[np.ndarray], degrees: Sequence[int], points: np.ndarray) -> sparse.csr_array:
  """The ``(m, prod n_d)`` matrix of the tensor-product basis at ``m`` points ``(m, D)`` inside the
  base box, columns in the coefficients' C order: the rows of the axes' design matrices multiplied
  out. (SciPy's ``NdBSpline.design_matrix`` sizes its columns by the last one used.)"""
  from scipy.interpolate import BSpline as SciBSpline

  m = points.shape[0]
  cols, vals, width = np.zeros((m, 1), dtype=np.int64), np.ones((m, 1)), 1
  for d, (t, k) in enumerate(zip(knots, degrees, strict=True)):
    b = SciBSpline.design_matrix(points[:, d], t, k).tocsr()
    n = t.size - k - 1
    idx, val = b.indices.reshape(m, k + 1), b.data.reshape(m, k + 1)
    cols = (cols[:, :, None] * n + idx[:, None, :]).reshape(m, -1)
    vals = (vals[:, :, None] * val[:, None, :]).reshape(m, -1)
    width *= n
  rows = np.repeat(np.arange(m), cols.shape[1])
  return sparse.csr_array((vals.reshape(-1), (rows, cols.reshape(-1))), shape=(m, width))


_FUNCTIONS: weakref.WeakValueDictionary[str, ConcreteFunction[Any, Any, Any, Any]] = weakref.WeakValueDictionary()


def _interned(name: str, digest: str, build: Callable[[], ConcreteFunction[Any, Any, Any, Any]]) -> ConcreteFunction[Any, Any, Any, Any]:
  """One Function per name and content: a name given by hand does not carry the digest, and two
  splines under one name must not share a Function (lowering refuses two different Functions of
  one name in one graph)."""
  key = f"{name}\0{digest}"
  fn = _FUNCTIONS.get(key)
  if fn is None:
    fn = _FUNCTIONS[key] = build()
  return fn


def _gather(table: np.ndarray, i: Expr, dtype: DType = dtypes.float64) -> Expr:
  """``table[i]`` for a constant vector and a scalar ``int64`` index in range."""
  return take(Expr.const(np.ascontiguousarray(table, dtype=np.float64), dtype=dtype), stack([i]), in_range=True)[0]


def _block(table: np.ndarray, base: Expr, width: int, dtype: DType = dtypes.float64) -> Expr:
  """``table[base * width + arange(width)]`` for a scalar ``int64`` ``base``."""
  return take(Expr.const(table, dtype=dtype), stack([base]) * width + Expr.const(np.arange(width), dtype=dtypes.int64), in_range=True)


def _horner(coef: list[Expr], s: Expr) -> Expr:
  """``sum_m coef[m] s^m``."""
  q = coef[-1]
  for c in reversed(coef[:-1]):
    q = q * s + c
  return q
