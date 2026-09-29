"""The interp methods, one class per kind of fit: the interpolating kinds, each along every axis or mixed per axis, the smoothing spline and the shape-constrained least squares, each building a ``BSpline`` from a ``Fit``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from ..function.method import Support
from .constrained import constrained
from .fit import BOUNDARIES, HERMITE, interpolant, smoothing
from .method import METHOD_API, Fit
from .spline import BSpline


class _Method:
  """What every interp method shares: it fits a ``Fit`` and builds a ``BSpline``."""

  name: ClassVar[str]
  problem: ClassVar[type] = Fit
  api: ClassVar[int] = METHOD_API

  def supports(self, problem: Any) -> Support:
    """Whether this method can fit ``problem``: the reasons it cannot, if any."""
    return Support() if isinstance(problem, Fit) else Support((f"{type(problem).__name__} is not a Fit",))

  @staticmethod
  def _spline(problem: Fit, name: str | None) -> dict[str, Any]:
    return {
      "extrap": problem.extrap,
      "fill": problem.fill,
      "search": problem.search,
      "strategy": problem.strategy,
      "dtype": problem.dtype,
      "name": name or problem.name,
    }


class Interpolating(_Method):
  """A spline through data on a grid, of one ``kind`` of ``interpolant`` along every axis; ``PerAxis``
  mixes them. The fit is done by SciPy when the spline is built for NumPy data, and is in the graph
  for ``Expr`` values."""

  kind: ClassVar[str]

  def axis(self) -> dict[str, Any]:
    """This kind's options along one axis, as ``interpolant`` takes them."""
    return {"kind": self.kind, "bc": "not-a-knot", "degree": 3, "period": None, "frac": None}

  def supports(self, problem: Any) -> Support:
    """Whether this method can fit ``problem``: the reasons it cannot, if any."""
    if not isinstance(problem, Fit):
      return super().supports(problem)
    if not problem.gridded:
      return Support(("the data are scattered points; an interpolant needs them on a grid (Smoothing or Constrained fit points)",))
    if self.kind in HERMITE and problem.ndim > 1:
      return Support((f"{self.kind} is 1-D only",))
    return Support()

  def build(self, problem: Fit, *, name: str | None = None) -> BSpline:
    """The spline this method fits to ``problem``, named ``name`` or the problem's name."""
    return _interpolant(problem, [self.axis()] * problem.ndim, name)


def _interpolant(problem: Fit, axes: list[dict[str, Any]], name: str | None) -> BSpline:
  fracs = {a["frac"] for a in axes if a["frac"] is not None}
  if len(fracs) > 1:
    raise ValueError(f"SmoothLinear axes must share one frac, got {sorted(fracs)}")
  per = {key: tuple(a[key] for a in axes) for key in ("kind", "bc", "degree", "period")}
  return interpolant(
    problem.x,
    problem.y,
    per["kind"],
    bc=per["bc"],
    degree=per["degree"],
    period=per["period"],
    frac=fracs.pop() if fracs else 0.1,
    **_Method._spline(problem, name),
  )


@dataclass(frozen=True)
class Nearest(Interpolating):
  """The value of the nearest site, a piecewise constant; a point midway takes the lower one."""

  name: ClassVar[str] = "interp.nearest"
  kind: ClassVar[str] = "nearest"


@dataclass(frozen=True)
class ZOH(Interpolating):
  """A zero-order hold: each site's value until the next site; the last one for ``period`` past it,
  by default the last interval's length."""

  name: ClassVar[str] = "interp.zoh"
  kind: ClassVar[str] = "zoh"

  period: float | None = None

  def axis(self) -> dict[str, Any]:
    """This kind's options along one axis, as ``interpolant`` takes them."""
    return {**super().axis(), "period": self.period}


@dataclass(frozen=True)
class Linear(Interpolating):
  """Piecewise-linear interpolation, multilinear on a grid."""

  name: ClassVar[str] = "interp.linear"
  kind: ClassVar[str] = "linear"


@dataclass(frozen=True)
class Cubic(Interpolating):
  """The C2 cubic spline with boundary condition ``bc``: ``"not-a-knot"``, ``"natural"``,
  ``"clamped"`` or ``"periodic"``."""

  name: ClassVar[str] = "interp.cubic"
  kind: ClassVar[str] = "cubic"

  bc: str = "not-a-knot"

  def __post_init__(self) -> None:
    if self.bc not in BOUNDARIES:
      raise ValueError(f"bc must be one of {BOUNDARIES}, got {self.bc!r}")

  def axis(self) -> dict[str, Any]:
    """This kind's options along one axis, as ``interpolant`` takes them."""
    return {**super().axis(), "bc": self.bc}


@dataclass(frozen=True)
class Spline(Interpolating):
  """The interpolating spline of ``degree`` 1 to 5, with ``bc`` ``"not-a-knot"`` or ``"periodic"``."""

  name: ClassVar[str] = "interp.spline"
  kind: ClassVar[str] = "spline"

  degree: int = 3
  bc: str = "not-a-knot"

  def __post_init__(self) -> None:
    if int(self.degree) != self.degree or not 1 <= self.degree <= 5:
      raise ValueError(f"degree must be 1 to 5, got {self.degree!r}")
    if self.bc not in ("not-a-knot", "periodic"):
      raise ValueError(f"bc must be 'not-a-knot' or 'periodic', got {self.bc!r}")

  def axis(self) -> dict[str, Any]:
    """This kind's options along one axis, as ``interpolant`` takes them."""
    return {**super().axis(), "degree": int(self.degree), "bc": self.bc}


@dataclass(frozen=True)
class PCHIP(Interpolating):
  """Fritsch-Carlson's monotone cubic Hermite interpolant (SciPy's ``PchipInterpolator``); 1-D."""

  name: ClassVar[str] = "interp.pchip"
  kind: ClassVar[str] = "pchip"


@dataclass(frozen=True)
class Akima(Interpolating):
  """Akima's cubic Hermite interpolant, little overshoot near outliers; 1-D."""

  name: ClassVar[str] = "interp.akima"
  kind: ClassVar[str] = "akima"


@dataclass(frozen=True)
class Makima(Interpolating):
  """The modified Akima interpolant, no overshoot on flat runs; 1-D."""

  name: ClassVar[str] = "interp.makima"
  kind: ClassVar[str] = "makima"


@dataclass(frozen=True)
class Steffen(Interpolating):
  """Steffen's monotone cubic Hermite interpolant, no overshoot between the data; 1-D."""

  name: ClassVar[str] = "interp.steffen"
  kind: ClassVar[str] = "steffen"


@dataclass(frozen=True)
class SmoothLinear(Interpolating):
  """CasADi's ``smooth_linear``: the linear interpolant with each corner rounded by a cubic over
  ``frac`` of the shortest interval either side, C2."""

  name: ClassVar[str] = "interp.smooth_linear"
  kind: ClassVar[str] = "smooth_linear"

  frac: float = 0.1

  def __post_init__(self) -> None:
    if not 0.0 < self.frac < 0.5:
      raise ValueError(f"frac must be in (0, 0.5), got {self.frac!r}")

  def axis(self) -> dict[str, Any]:
    """This kind's options along one axis, as ``interpolant`` takes them."""
    return {**super().axis(), "frac": self.frac}


class PerAxis(_Method):
  """A different interpolating method along each axis of a grid: ``PerAxis(Linear(), Cubic())``."""

  name: ClassVar[str] = "interp.per_axis"

  def __init__(self, *axes: Interpolating) -> None:
    if not axes or not all(isinstance(a, Interpolating) for a in axes):
      raise TypeError("PerAxis takes one interpolating method (Linear(), Cubic(), ...) per axis")
    self.axes = axes

  def __repr__(self) -> str:
    return f"PerAxis({', '.join(map(repr, self.axes))})"

  def supports(self, problem: Any) -> Support:
    """Whether this method can fit ``problem``: the reasons it cannot, if any."""
    if not isinstance(problem, Fit):
      return super().supports(problem)
    if not problem.gridded:
      return Support(("the data are scattered points; an interpolant needs them on a grid",))
    if problem.ndim != len(self.axes):
      return Support((f"{len(self.axes)} methods for {problem.ndim} axes",))
    return Support(tuple(f"{a.kind} is 1-D only" for a in self.axes if a.kind in HERMITE and problem.ndim > 1))

  def build(self, problem: Fit, *, name: str | None = None) -> BSpline:
    """The spline this method fits to ``problem``, named ``name`` or the problem's name."""
    return _interpolant(problem, [a.axis() for a in self.axes], name)


def _per_axis(value: Any) -> Any:
  return tuple(value) if isinstance(value, (list, tuple)) else value


@dataclass(frozen=True)
class Smoothing(_Method):
  """A smoothing spline through noisy data, on a grid or at scattered points: ``"pspline"`` (Eilers
  and Marx) is a B-spline of ``degree`` on ``segments`` equal intervals per axis, fitted by least
  squares with the ``penalty``-th differences of its coefficients penalized by ``lam`` (chosen by
  generalized cross-validation with ``"gcv"``); ``"cubic"`` is SciPy's ``make_smoothing_spline``, 1-D.
  As ``smoothing``."""

  name: ClassVar[str] = "interp.smoothing"

  degree: int | tuple[int, ...] = 3
  segments: int | tuple[int, ...] = 20
  penalty: int = 2
  lam: float | Literal["gcv"] = "gcv"
  method: Literal["pspline", "cubic"] = "pspline"

  def __post_init__(self) -> None:
    object.__setattr__(self, "degree", _per_axis(self.degree))
    object.__setattr__(self, "segments", _per_axis(self.segments))
    if self.method not in ("pspline", "cubic"):
      raise ValueError(f"method must be 'pspline' or 'cubic', got {self.method!r}")

  def build(self, problem: Fit, *, name: str | None = None) -> BSpline:
    """The spline this method fits to ``problem``, named ``name`` or the problem's name."""
    return smoothing(
      problem.x,
      problem.y,
      degree=self.degree,
      segments=self.segments,
      penalty=self.penalty,
      lam=self.lam,
      method=self.method,
      **_Method._spline(problem, name),
    )


@dataclass(frozen=True)
class Constrained(_Method):
  """The B-spline closest to scattered data in least squares whose coefficients make its shape
  what the physics says (``monotone``, ``convex``, ``bounds``, pinned values ``equal``,
  ``periodic``): a quadratic program solved when the spline is built by ``qp``, an ``sc.opt`` QP
  method that takes the constraints' bounds as data at run time (by default the PIQP library at
  1e-12; not the generated IPM, whose code keeps only the bounds finite when it was built), and
  refined to rounding. As ``constrained``."""

  name: ClassVar[str] = "interp.constrained"

  degree: int | tuple[int, ...] = 3
  knots: Any = 16
  weights: tuple[float, ...] | None = None
  monotone: Any = None
  convex: Any = None
  bounds: tuple[float, float] | None = None
  equal: tuple[Any, ...] = ()
  periodic: bool = False
  penalty: int = 2
  lam: float = 0.0
  qp: Any = None

  def __post_init__(self) -> None:
    object.__setattr__(self, "degree", _per_axis(self.degree))
    if not isinstance(self.knots, (int, tuple)):  # interior knots, held as floats in a tuple, compared by value
      object.__setattr__(self, "knots", tuple(float(k) for k in self.knots))

  def _knots(self) -> Any:
    """``knots`` as ``constrained`` reads it: a tuple of floats is interior knots, not one spec per axis."""
    if isinstance(self.knots, tuple) and self.knots and all(isinstance(k, float) for k in self.knots):
      return list(self.knots)
    return self.knots
    if self.weights is not None:
      object.__setattr__(self, "weights", tuple(float(w) for w in self.weights))
    object.__setattr__(self, "equal", tuple(tuple(e) for e in self.equal))

  def supports(self, problem: Any) -> Support:
    """Whether this method can fit ``problem``: the reasons it cannot, if any."""
    if not isinstance(problem, Fit):
      return super().supports(problem)
    return Support.unless(
      the_data_are_an_Expr_and_the_fit_is_solved_when_the_spline_is_built=problem.symbolic,
      the_data_are_on_a_grid_and_constrained_fits_scattered_points=isinstance(problem.x, (tuple, list)),
    )

  def build(self, problem: Fit, *, name: str | None = None) -> BSpline:
    """The spline this method fits to ``problem``, named ``name`` or the problem's name."""
    return constrained(
      problem.x,
      problem.y,
      degree=self.degree,
      knots=self._knots(),
      weights=None if self.weights is None else list(self.weights),
      monotone=self.monotone,
      convex=self.convex,
      bounds=self.bounds,
      equal=self.equal,
      periodic=self.periodic,
      penalty=self.penalty,
      lam=self.lam,
      qp=self.qp,
      **_Method._spline(problem, name),
    )


__all__ = [
  "PCHIP",
  "ZOH",
  "Akima",
  "Constrained",
  "Cubic",
  "Interpolating",
  "Linear",
  "Makima",
  "Nearest",
  "PerAxis",
  "Smoothing",
  "SmoothLinear",
  "Spline",
  "Steffen",
]
