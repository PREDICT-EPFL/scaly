"""The integrator methods, one class per named method: explicit and implicit Runge-Kutta, the adaptive pairs and the symplectic schemes, each building the discrete map of an ``ODE``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from ..function.method import Support
from ..roots import Newton
from .explicit import adaptive, explicit, symplectic
from .implicit import implicit_map
from .method import METHOD_API, ODE
from .tableau import TABLEAUS, Tableau, gauss_legendre, lobatto_iiia, lobatto_iiic, radau_iia


def _positive_int(value: Any, what: str) -> None:
  if int(value) != value or value < 1:
    raise ValueError(f"{what} must be a positive integer, got {value!r}")


class _Method:
  """What every integrator method shares: it solves an ``ODE``, and ``label`` names its maps."""

  name: ClassVar[str]
  problem: ClassVar[type] = ODE
  api: ClassVar[int] = METHOD_API

  @property
  def label(self) -> str:
    """How the method is spelled in the names of its maps (``{model}_{label}``)."""
    raise NotImplementedError

  def supports(self, problem: Any) -> Support:
    """Whether this method can integrate ``problem``: the reasons it cannot, if any."""
    return Support() if isinstance(problem, ODE) else Support((f"{type(problem).__name__} is not an ODE",))


@dataclass(frozen=True)
class ExplicitRK(_Method):
  """An explicit Runge-Kutta method: the class's ``table``, ``steps`` equal substeps per interval
  (unrolled up to ``UNROLL_STEPS``, looped beyond). Each stage calls the model once. A package adds
  a method by subclassing with its ``name`` and ``table`` and declaring it in ``scaly.methods``."""

  table: ClassVar[Tableau]

  steps: int = 1

  def __post_init__(self) -> None:
    _positive_int(self.steps, "steps")

  @property
  def tableau(self) -> Tableau:
    """The Butcher tableau of the step."""
    return self.table

  @property
  def label(self) -> str:
    """How the method is spelled in the names of its maps (``{model}_{label}``)."""
    return self.table.name

  def build(self, problem: ODE, *, name: str | None = None) -> Any:
    """The discrete map of ``problem``, named ``name`` or ``{model}_{label}``."""
    return explicit(problem.f, self.table, dt=problem.dt, steps=self.steps, name=name)


@dataclass(frozen=True)
class Euler(ExplicitRK):
  """Forward Euler, order 1."""

  name: ClassVar[str] = "integrators.euler"
  table: ClassVar[Tableau] = TABLEAUS["euler"]


@dataclass(frozen=True)
class Heun(ExplicitRK):
  """Heun's method, the explicit trapezoidal rule, order 2."""

  name: ClassVar[str] = "integrators.heun"
  table: ClassVar[Tableau] = TABLEAUS["heun"]


@dataclass(frozen=True)
class Midpoint(ExplicitRK):
  """The explicit midpoint method, order 2."""

  name: ClassVar[str] = "integrators.midpoint"
  table: ClassVar[Tableau] = TABLEAUS["midpoint"]


@dataclass(frozen=True)
class Ralston(ExplicitRK):
  """Ralston's second-order method, the least truncation error of the two-stage ones."""

  name: ClassVar[str] = "integrators.ralston"
  table: ClassVar[Tableau] = TABLEAUS["ralston"]


@dataclass(frozen=True)
class RK3(ExplicitRK):
  """Kutta's third-order method."""

  name: ClassVar[str] = "integrators.rk3"
  table: ClassVar[Tableau] = TABLEAUS["rk3"]


@dataclass(frozen=True)
class SSPRK3(ExplicitRK):
  """The strong-stability-preserving third-order method of Shu and Osher."""

  name: ClassVar[str] = "integrators.ssprk3"
  table: ClassVar[Tableau] = TABLEAUS["ssprk3"]


@dataclass(frozen=True)
class RK4(ExplicitRK):
  """The classical fourth-order Runge-Kutta method."""

  name: ClassVar[str] = "integrators.rk4"
  table: ClassVar[Tableau] = TABLEAUS["rk4"]


@dataclass(frozen=True)
class RK38(ExplicitRK):
  """Kutta's 3/8 rule, order 4."""

  name: ClassVar[str] = "integrators.rk38"
  table: ClassVar[Tableau] = TABLEAUS["rk38"]


@dataclass(frozen=True)
class BS32(ExplicitRK):
  """The Bogacki-Shampine 3(2) pair's third-order step; an embedded pair for ``Adaptive``."""

  name: ClassVar[str] = "integrators.bs32"
  table: ClassVar[Tableau] = TABLEAUS["bs32"]


@dataclass(frozen=True)
class DOPRI5(ExplicitRK):
  """The Dormand-Prince 5(4) pair's fifth-order step (``ode45``, SciPy's ``RK45``); an embedded pair for ``Adaptive``."""

  name: ClassVar[str] = "integrators.dopri5"
  table: ClassVar[Tableau] = TABLEAUS["dopri5"]


@dataclass(frozen=True)
class Tsit5(ExplicitRK):
  """Tsitouras's 5(4) pair's fifth-order step; an embedded pair for ``Adaptive``."""

  name: ClassVar[str] = "integrators.tsit5"
  table: ClassVar[Tableau] = TABLEAUS["tsit5"]


_NEWTON = Newton(tol=None, max_iter=3, simplified=True)


class _Implicit(_Method):
  """What the implicit methods share: the stage equations solved by ``newton``, a ``roots.Newton``
  whose ``tol=None`` takes a fixed number of iterations and whose ``simplified`` factors the stage
  matrix once per step (split by the eigenvalues of ``A`` for a coupled method, as RADAU5 does). The
  stage matrix's own solve replaces its ``linear``. The derivative is the implicit function
  theorem's at the stages found, never the iterations'."""

  steps: int
  newton: Newton

  @property
  def tableau(self) -> Tableau:
    """The Butcher tableau of the step."""
    raise NotImplementedError

  def _check(self) -> None:
    _positive_int(self.steps, "steps")
    if not isinstance(self.newton, Newton):
      raise TypeError(f"newton must be an sc.roots.Newton, got {type(self.newton).__name__}")
    if self.newton.linear != "lu":
      raise ValueError("the stage equations are solved with the stage matrix's own solve; leave the Newton method's linear at 'lu'")

  @property
  def label(self) -> str:
    """How the method is spelled in the names of its maps (``{model}_{label}``)."""
    return self.tableau.name

  def build(self, problem: ODE, *, name: str | None = None) -> Any:
    """The discrete map of ``problem``, named ``name`` or ``{model}_{label}``."""
    return implicit_map(problem.f, self.tableau, dt=problem.dt, steps=self.steps, newton=self.newton, name=name)


@dataclass(frozen=True)
class ImplicitRK(_Implicit):
  """An implicit Runge-Kutta method of the class's fixed ``table``. A package adds a method by
  subclassing with its ``name`` and ``table`` and declaring it in ``scaly.methods``."""

  table: ClassVar[Tableau]

  steps: int = 1
  newton: Newton = _NEWTON

  def __post_init__(self) -> None:
    self._check()

  @property
  def tableau(self) -> Tableau:
    """The Butcher tableau of the step."""
    return self.table


@dataclass(frozen=True)
class BackwardEuler(ImplicitRK):
  """Backward Euler, order 1, L-stable."""

  name: ClassVar[str] = "integrators.backward_euler"
  table: ClassVar[Tableau] = TABLEAUS["backward_euler"]


@dataclass(frozen=True)
class ImplicitMidpoint(ImplicitRK):
  """The implicit midpoint rule, order 2, symplectic; the one-stage Gauss-Legendre method."""

  name: ClassVar[str] = "integrators.implicit_midpoint"
  table: ClassVar[Tableau] = TABLEAUS["implicit_midpoint"]


@dataclass(frozen=True)
class Trapezoidal(ImplicitRK):
  """The trapezoidal rule, order 2, A-stable; the two-stage Lobatto IIIA method."""

  name: ClassVar[str] = "integrators.trapezoidal"
  table: ClassVar[Tableau] = TABLEAUS["trapezoidal"]


@dataclass(frozen=True)
class SDIRK2(ImplicitRK):
  """Alexander's two-stage SDIRK, order 2, L-stable; its stages solve one after another."""

  name: ClassVar[str] = "integrators.sdirk2"
  table: ClassVar[Tableau] = TABLEAUS["sdirk2"]


@dataclass(frozen=True)
class SDIRK3(ImplicitRK):
  """Alexander's three-stage SDIRK, order 3, L-stable; its stages solve one after another."""

  name: ClassVar[str] = "integrators.sdirk3"
  table: ClassVar[Tableau] = TABLEAUS["sdirk3"]


class _Family(_Implicit):
  """A family of collocation methods, the member of ``stages`` stages."""

  family: ClassVar[Any]
  stages: int

  def __post_init__(self) -> None:
    self._check()
    object.__setattr__(self, "_tableau", type(self).family(self.stages))

  @property
  def tableau(self) -> Tableau:
    """The Butcher tableau of the step."""
    return getattr(self, "_tableau")


@dataclass(frozen=True)
class GaussLegendre(_Family):
  """The ``stages``-stage Gauss-Legendre method: order ``2 stages``, A-stable and symplectic."""

  name: ClassVar[str] = "integrators.gauss_legendre"
  family: ClassVar[Any] = staticmethod(gauss_legendre)

  stages: int = 2
  steps: int = 1
  newton: Newton = _NEWTON


@dataclass(frozen=True)
class RadauIIA(_Family):
  """The ``stages``-stage Radau IIA method: order ``2 stages - 1``, L-stable and stiffly accurate."""

  name: ClassVar[str] = "integrators.radau_iia"
  family: ClassVar[Any] = staticmethod(radau_iia)

  stages: int = 3
  steps: int = 1
  newton: Newton = _NEWTON


@dataclass(frozen=True)
class LobattoIIIA(_Family):
  """The ``stages``-stage Lobatto IIIA method: order ``2 stages - 2``, its first stage explicit."""

  name: ClassVar[str] = "integrators.lobatto_iiia"
  family: ClassVar[Any] = staticmethod(lobatto_iiia)

  stages: int = 3
  steps: int = 1
  newton: Newton = _NEWTON


@dataclass(frozen=True)
class LobattoIIIC(_Family):
  """The ``stages``-stage Lobatto IIIC method: order ``2 stages - 2``, L-stable and stiffly accurate."""

  name: ClassVar[str] = "integrators.lobatto_iiic"
  family: ClassVar[Any] = staticmethod(lobatto_iiic)

  stages: int = 3
  steps: int = 1
  newton: Newton = _NEWTON


@dataclass(frozen=True)
class Adaptive(_Method):
  """An embedded ``pair`` (``DOPRI5()``, ``Tsit5()``, ``BS32()``) with error control: the steps are
  chosen as the integration goes, in a ``while_loop`` of at most ``max_steps`` steps, accepted when
  the scaled error's root mean square is at most 1 (``atol + rtol |x|`` entry by entry), as
  ``solve_ivp`` does. ``h0`` is the first step, by default the interval over 100. For a plant model
  in closed-loop simulation, whose accuracy should not depend on the sampling time."""

  name: ClassVar[str] = "integrators.adaptive"

  pair: ExplicitRK = DOPRI5()
  rtol: float = 1e-6
  atol: float = 1e-9
  max_steps: int = 10_000
  h0: float | None = None

  def __post_init__(self) -> None:
    if not isinstance(self.pair, ExplicitRK) or self.pair.tableau.b_err is None:
      raise ValueError(f"pair must be an embedded explicit pair (DOPRI5(), Tsit5(), BS32()), got {self.pair!r}")
    if not self.rtol > 0 or not self.atol > 0 or (self.h0 is not None and not self.h0 > 0):
      raise ValueError(f"rtol, atol and h0 must be positive, got {self.rtol}, {self.atol}, {self.h0}")
    _positive_int(self.max_steps, "max_steps")

  @property
  def label(self) -> str:
    """How the method is spelled in the names of its maps (``{model}_{label}``)."""
    return f"{self.pair.tableau.name}_adaptive"

  def build(self, problem: ODE, *, name: str | None = None) -> Any:
    """The discrete map of ``problem``, named ``name`` or ``{model}_{label}``."""
    return adaptive(problem.f, self.pair.tableau, dt=problem.dt, rtol=self.rtol, atol=self.atol, max_steps=self.max_steps, h0=self.h0, name=name)


class _Symplectic(_Method):
  """A symplectic scheme for a state of positions then velocities, ``x = [q, v]``, ``q = x[:split]``."""

  scheme: ClassVar[str]
  split: int | None
  steps: int

  def __post_init__(self) -> None:
    _positive_int(self.steps, "steps")
    if self.split is not None:
      _positive_int(self.split, "split")

  @property
  def label(self) -> str:
    """How the method is spelled in the names of its maps (``{model}_{label}``)."""
    return self.scheme

  def supports(self, problem: Any) -> Support:
    """Whether this method can integrate ``problem``: the reasons it cannot, if any."""
    if not isinstance(problem, ODE):
      return super().supports(problem)
    return Support(("needs split, the number of positions in the state",) if self.split is None else ())

  def build(self, problem: ODE, *, name: str | None = None) -> Any:
    """The discrete map of ``problem``, named ``name`` or ``{model}_{label}``."""
    if self.split is None:
      raise ValueError(f"{self.name} needs split, the number of positions in the state")
    return symplectic(problem.f, self.scheme, split=self.split, dt=problem.dt, steps=self.steps, name=name)  # ty: ignore[invalid-argument-type]


@dataclass(frozen=True)
class StormerVerlet(_Symplectic):
  """Stormer-Verlet, order 2: a half kick, a drift, a half kick. Symplectic for a separable model,
  where the energy error stays bounded over long horizons."""

  name: ClassVar[str] = "integrators.stormer_verlet"
  scheme: ClassVar[str] = "stormer_verlet"

  split: int | None = None
  steps: int = 1


@dataclass(frozen=True)
class SymplecticEuler(_Symplectic):
  """Symplectic Euler, order 1: a kick, then a drift."""

  name: ClassVar[str] = "integrators.symplectic_euler"
  scheme: ClassVar[str] = "symplectic_euler"

  split: int | None = None
  steps: int = 1


__all__ = [
  "BS32",
  "DOPRI5",
  "RK38",
  "RK3",
  "RK4",
  "SDIRK2",
  "SDIRK3",
  "SSPRK3",
  "Adaptive",
  "BackwardEuler",
  "Euler",
  "ExplicitRK",
  "GaussLegendre",
  "Heun",
  "ImplicitMidpoint",
  "ImplicitRK",
  "LobattoIIIA",
  "LobattoIIIC",
  "Midpoint",
  "RadauIIA",
  "Ralston",
  "StormerVerlet",
  "SymplecticEuler",
  "Trapezoidal",
  "Tsit5",
]
