"""Optimal control problems: ``ContinuousOCP`` over a model and a horizon length, ``DiscreteOCP`` over a map and a number of stages, and ``transcribe`` from the first to the second."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np

from ..function.model import ConcreteFunction, Function
from ..function.tree import L, param_list
from ..integrators.model import check_model
from ..ir.expr import Expr
from ..ir.types import TensorType
from .transcription import Interval, MultipleShooting, Transcription

METHOD_API = 1
"""The version of the method API of ``DiscreteOCP`` every OCP method implements: bump it whenever what
a method's ``build`` reads from the problem, or the Function it must return, changes."""


@dataclass(frozen=True)
class Quadratic:
  """A quadratic cost ``(x - x_ref)' Q (x - x_ref) + (u - u_ref)' R (u - u_ref)``, as a stage cost, or
  without ``R`` as a terminal cost. A reference is an array, the name of a parameter of that size
  (which the OCP then takes), or ``None`` for the origin."""

  Q: Any
  R: Any = None
  x_ref: Any = None
  u_ref: Any = None


@dataclass(frozen=True)
class Path:
  """A path constraint ``lo <= g(x_k, u_k, ...) <= hi`` at every stage ``k < N``, ``g`` a Function of
  the state, the control and any parameters (by name). With ``soft``, each row gets a slack, zero at
  the least, and the cost ``soft`` times the slacks' sum, an exact penalty: the constraint holds
  whenever it can, and the problem stays feasible when it cannot."""

  fn: Function[Any, Any, Any, Any]
  lo: Any = None
  hi: Any = None
  soft: float | None = None


@dataclass(frozen=True)
class TerminalEquality:
  """The terminal constraint ``x_N = x_ref``: an array, a parameter's name, or ``None`` for the origin."""

  x_ref: Any = None


@dataclass(frozen=True)
class Param:
  """A parameter of an OCP: a name and the type of one value (``N + 1`` of them when it varies)."""

  name: str
  type: TensorType


def _model(source: Function[Any, Any, Any, Any]) -> ConcreteFunction[Any, Any, Any, Any]:
  model = source.concrete
  check_model(model, model.name)
  if len(model.input_tree.parts) < 2 or len(model.inputs[1].shape) != 1:
    raise ValueError(f"{model.name}: the model takes the state, then the control, one vector: f(x, u, *params)")
  return model


@dataclass(frozen=True, eq=False)
class ContinuousOCP:
  """An optimal control problem over a continuous-time model and a horizon of length ``T``: what is
  to be optimized, before a transcription says how. It is not solvable itself; ``transcribe`` makes
  the ``DiscreteOCP`` a method solves.

  Args:
    ode: the model ``f(x, u, *params) -> xdot``.
    T: the horizon's length.
    stage_cost: ``l(x, u, *params)`` (one value) or a ``Quadratic``: the running cost, integrated over
      the horizon at the grid points (``cost="points"``, ``dt`` times the sum) or by the
      transcription's quadrature over each interval (``cost="integral"``).
    terminal_cost: ``Vf(x, *params)`` (one value) or a ``Quadratic`` without ``R``.
    x_bounds, u_bounds: ``(lo, hi)``, each a number, an array of the state's (control's) size or
      ``None``; on every state but the initial one, the transcription's internal states included,
      and on every control.
    constraints: ``Path`` constraints.
    terminal: a terminal set: ``TerminalEquality``, or a set from ``scaly.sets``.
    varying: names of parameters that take one value per grid point, ``N + 1`` of them.
    cost: ``"points"`` or ``"integral"``, as above; by default ``"integral"`` for a transcription that
      puts controls inside an interval (``Pseudospectral``), and ``"points"`` otherwise.
    name: the problem's name, by default ``{model}_ocp``.
  """

  ode: Function[Any, Any, Any, Any]
  T: float
  stage_cost: Any = None
  terminal_cost: Any = None
  x_bounds: tuple[Any, Any] | None = None
  u_bounds: tuple[Any, Any] | None = None
  constraints: Sequence[Path] = ()
  terminal: Any = None
  varying: Sequence[str] = ()
  cost: Literal["points", "integral"] | None = None
  name: str | None = None

  def __post_init__(self) -> None:
    _model(self.ode)
    if not float(self.T) > 0:
      raise ValueError(f"T, the horizon's length, must be positive, got {self.T}")
    if self.cost not in (None, "points", "integral"):
      raise ValueError(f"cost is 'points' or 'integral', got {self.cost!r}")
    object.__setattr__(self, "constraints", tuple(self.constraints))
    object.__setattr__(self, "varying", tuple(self.varying))


def transcribe(problem: ContinuousOCP, transcription: Transcription | None = None, *, N: int) -> DiscreteOCP:
  """The ``DiscreteOCP`` of ``problem`` over ``N`` intervals of length ``T / N``, each transcribed by
  ``transcription`` (``MultipleShooting(RK4())`` by default, ``Collocation``, ``Pseudospectral``):
  its stage is the transcription's interval, whose own variables are the stage's internal ones."""
  if not isinstance(problem, ContinuousOCP):
    raise TypeError(f"transcribe takes a ContinuousOCP, got {type(problem).__name__}")
  if int(N) != N or N < 1:
    raise ValueError(f"N must be a positive integer, got {N}")
  return DiscreteOCP._transcribed(problem, transcription, int(N), float(problem.T) / int(N))


class DiscreteOCP:
  """An optimal control problem over ``N`` stages: the general multistage form every OCP method solves.

  Each stage ``k`` has the state ``x_k``, the control ``u_k`` and the stage's own variables ``w_k`` (a
  collocation's internal states, say), dynamics that relate them to ``x_{k+1}``, a stage cost, path
  constraints and bounds, and the parameters, some of them per stage; the last state has a terminal
  cost and a terminal set. Built from a discrete-time map, ``DiscreteOCP(step=F, N=...)`` (the
  dynamics ``x_{k+1} = F(x_k, u_k, *params)``, no ``w_k``), or by ``transcribe`` from a
  ``ContinuousOCP``. It keeps its stage structure explicit (``stage``, ``params``, and the layout
  ``to_problem`` gives), which structured methods read.

  Args:
    step: the map ``F(x, u, *params) -> x_next``.
    N: the number of stages.
    stage_cost: ``l(x, u, *params)`` (one value) or a ``Quadratic``; the cost is its sum over the stages.
    terminal_cost, x_bounds, u_bounds, constraints, terminal, varying, name: as for ``ContinuousOCP``.
    dt: the stage's length, only for ``times``.

  A Function's parameters are its inputs after the state and the control (after the state for a
  terminal cost), one vector each, and are matched by name across Functions: a model taking ``mass``
  and a stage cost taking ``r`` give the OCP the parameters ``mass`` and ``r``.
  """

  method_api: ClassVar[int] = METHOD_API

  def __init__(
    self,
    *,
    step: Function[Any, Any, Any, Any],
    N: int,
    stage_cost: Function[Any, Any, Any, Any] | Quadratic | None = None,
    terminal_cost: Function[Any, Any, Any, Any] | Quadratic | None = None,
    x_bounds: tuple[Any, Any] | None = None,
    u_bounds: tuple[Any, Any] | None = None,
    constraints: Sequence[Path] = (),
    terminal: Any = None,
    varying: Sequence[str] = (),
    dt: float | None = None,
    name: str | None = None,
  ) -> None:
    self._setup(step, N, dt, None, "points", False, stage_cost, terminal_cost, x_bounds, u_bounds, constraints, terminal, varying, name)

  @classmethod
  def _transcribed(cls, problem: ContinuousOCP, transcription: Transcription | None, n: int, dt: float) -> DiscreteOCP:
    self = cls.__new__(cls)
    transcription = transcription if transcription is not None else MultipleShooting()
    node_controls = getattr(transcription, "node_controls", False)
    cost = problem.cost if problem.cost is not None else "integral" if node_controls else "points"
    self._setup(
      problem.ode,
      n,
      dt,
      transcription,
      cost,
      True,
      problem.stage_cost,
      problem.terminal_cost,
      problem.x_bounds,
      problem.u_bounds,
      problem.constraints,
      problem.terminal,
      problem.varying,
      problem.name,
    )
    return self

  def _setup(
    self,
    source: Function[Any, Any, Any, Any],
    horizon: int,
    dt: float | None,
    transcription: Transcription | None,
    cost: str,
    continuous: bool,
    stage_cost: Any,
    terminal_cost: Any,
    x_bounds: Any,
    u_bounds: Any,
    constraints: Sequence[Path],
    terminal: Any,
    varying: Sequence[str],
    name: str | None,
  ) -> None:
    model = _model(source)
    if int(horizon) != horizon or horizon < 1:
      raise ValueError(f"N must be a positive integer, got {horizon}")
    self.name = name or f"{model.name}_ocp"
    self.continuous, self.N, self.dt = continuous, int(horizon), None if dt is None else float(dt)
    self.nx, self.nu = model.inputs[0].size, model.inputs[1].size
    self.transcription, self.cost_rule = transcription, cost
    self.constraints, self.terminal, self.x_bounds, self.u_bounds = tuple(constraints), terminal, x_bounds, u_bounds

    # Every parameter any Function names, in order of appearance, one type per name.
    self.params: list[Param] = []
    self._collect(model, 2)
    # A method that works on the matrices themselves (TinyADMM's Riccati cache) reads the specs.
    self.stage_quadratic = stage_cost if isinstance(stage_cost, Quadratic) else None
    self.terminal_quadratic = terminal_cost if isinstance(terminal_cost, Quadratic) else None
    stage = self._quadratic(stage_cost, "stage_cost", True) if isinstance(stage_cost, Quadratic) else stage_cost
    terminal_fn = self._quadratic(terminal_cost, "terminal_cost", False) if isinstance(terminal_cost, Quadratic) else terminal_cost
    for fn, lead in ((stage, 2), *((c.fn, 2) for c in self.constraints), (terminal_fn, 1)):
      if fn is not None:
        self._collect(fn.concrete, lead)
    if isinstance(terminal, TerminalEquality) and isinstance(terminal.x_ref, str):
      self._declare(terminal.x_ref, TensorType((self.nx,)))
    unknown = set(varying) - {p.name for p in self.params}
    if unknown:
      raise ValueError(f"varying names {sorted(unknown)}, which no Function takes; parameters: {[p.name for p in self.params]}")
    self.varying = tuple(varying)

    self.model = self._over(model, 2, "model")
    self.stage_cost = None if stage is None else self._over(stage.concrete, 2, "stage_cost", scalar=True)
    self.terminal_cost = None if terminal_fn is None else self._over(terminal_fn.concrete, 1, "terminal_cost", scalar=True)
    self.paths = tuple(self._over(c.fn.concrete, 2, f"path{i}") for i, c in enumerate(self.constraints))
    self.interval = self._interval()
    self._formulations: dict[str, Any] = {}

  # -- parameters -------------------------------------------------------------------------------

  def _declare(self, name: str, type_: TensorType) -> None:
    for p in self.params:
      if p.name == name:
        if p.type.shape != type_.shape:
          raise ValueError(f"parameter {name!r} is {p.type.shape} in one Function and {type_.shape} in another")
        return
    if name in ("x0", "guess", "warm"):
      raise ValueError(f"a parameter may not be named {name!r}: a solver takes that input itself")
    self.params.append(Param(name, TensorType(type_.shape, type_.dtype)))

  def _collect(self, fn: ConcreteFunction[Any, Any, Any, Any], lead: int) -> None:
    parts = fn.input_tree.parts
    if any(len(part.names) != 1 for part in parts[lead:]):
      raise ValueError(f"{fn.name}: every parameter after the state{' and the control' if lead == 2 else ''} must be one vector")
    for part, expr in zip(parts[lead:], fn.inputs[lead:], strict=True):
      self._declare(part.names[0], expr.type)

  def param_size(self, p: Param) -> int:
    """The size of a parameter's value: one, or ``N + 1`` of them for a varying one, flat."""
    return p.type.size * (self.N + 1 if p.name in self.varying else 1)

  def _over(self, fn: ConcreteFunction[Any, Any, Any, Any], lead: int, role: str, *, scalar: bool = False) -> ConcreteFunction[Any, Any, Any, Any]:
    """``fn`` over its leading inputs and every OCP parameter, reading its own by name."""
    names = [part.names[0] for part in fn.input_tree.parts[lead:]]
    position = {p.name: i for i, p in enumerate(self.params)}
    if scalar and (len(fn.outputs) != 1 or fn.outputs[0].size != 1):
      raise ValueError(f"{fn.name}: a cost gives one value")

    def body(*args: Expr) -> Expr:
      lead_args, params = args[:lead], args[lead:]
      out = fn(*lead_args, *(params[position[n]] for n in names))
      return out.reshape(()) if scalar else out

    leading = [L(n, e.type) for n, e in zip(("x", "u"), fn.inputs[:lead], strict=False)]
    slots = param_list(*leading, *(L(p.name, p.type) for p in self.params))
    output = L(role, ()) if scalar else L(role, fn.outputs[0].type)
    return ConcreteFunction(f"{self.name}_{role}", body, slots, output)

  def _quadratic(self, spec: Quadratic, role: str, stage: bool) -> ConcreteFunction[Any, Any, Any, Any]:
    q = np.atleast_2d(np.asarray(spec.Q, dtype=np.float64))
    r = None if spec.R is None else np.atleast_2d(np.asarray(spec.R, dtype=np.float64))
    if (
      q.shape != (self.nx, self.nx) or (r is not None and r.shape != (self.nu, self.nu)) or (not stage and (r is not None or spec.u_ref is not None))
    ):
      raise ValueError(
        f"{role}: Q must be {self.nx}x{self.nx}{f' and R {self.nu}x{self.nu}' if stage else ', and a terminal cost has no R or u_ref'}"
      )
    refs = [(n, size) for n, size in ((spec.x_ref, self.nx), (spec.u_ref, self.nu)) if isinstance(n, str)]
    for n, size in refs:
      self._declare(n, TensorType((size,)))
    slots = [L("x", self.nx), *([L("u", self.nu)] if stage else []), *(L(n, size) for n, size in refs)]
    fixed = {
      k: np.zeros(size) if v is None else np.asarray(v, dtype=np.float64)
      for k, v, size in (("x", spec.x_ref, self.nx), ("u", spec.u_ref, self.nu))
      if not isinstance(v, str)
    }

    def body(*args: Expr) -> Expr:
      x, u = args[0], args[1] if stage else None
      named = dict(zip((n for n, _ in refs), args[2 if stage else 1 :], strict=True))
      ex = x - (named[spec.x_ref] if isinstance(spec.x_ref, str) else Expr.const(fixed["x"]))
      value = ex @ (Expr.const(q) @ ex)
      if r is not None and u is not None:
        eu = u - (named[spec.u_ref] if isinstance(spec.u_ref, str) else Expr.const(fixed["u"]))
        value = value + eu @ (Expr.const(r) @ eu)
      return value

    return ConcreteFunction(f"{self.name}_{role}_quadratic", body, param_list(*slots), L(role, ()))

  # -- the stage --------------------------------------------------------------------------------

  def _interval(self) -> Interval:
    """The stage every ``k`` maps: the transcription's interval, or ``F(x, u) - xnext`` for a map."""
    integral = self.cost_rule == "integral" and self.stage_cost is not None
    if self.transcription is not None:
      return self.transcription.interval(self.model, self.stage_cost if integral else None, dt=self.dt, name=f"{self.name}_interval")
    model, nx = self.model, self.nx

    def body(*args: Expr) -> Expr:
      x, u, xnext, params = args[0], args[1], args[2], args[3:]
      return model(x, u, *params) - xnext

    slots = param_list(L("x", self.nx), L("u", self.nu), L("xnext", nx), *(L(p.name, p.type) for p in self.params))
    fn = ConcreteFunction(f"{self.name}_interval", body, slots, L("r", nx))
    return Interval(fn, 0, nx, np.zeros(0), np.zeros(0), False)

  @property
  def stage(self) -> StageStructure:
    """The shape of one stage: the state's, the control's and the stage's own variables' sizes, and
    how many dynamics residuals relate them to the next state."""
    return StageStructure(self.nx, self.nu, self.interval.n_internal, self.interval.n_residual)

  @property
  def times(self) -> np.ndarray:
    """The grid times ``0, dt, ..., N dt`` (steps of 1 without ``dt``)."""
    return np.arange(self.N + 1) * (self.dt if self.dt is not None else 1.0)

  def at_end(self, params: dict[str, Expr], p: Param) -> Expr:
    """The value of ``p`` the terminal cost reads: the last one of a varying parameter."""
    value = params[p.name]
    return value[self.N * p.type.size :] if p.name in self.varying else value

  @property
  def shooting(self) -> bool:
    """Whether the stage is an explicit map from ``x_k`` to ``x_{k+1}``: a discrete-time model, or
    multiple shooting. The condensed form needs one."""
    return self.transcription is None or isinstance(self.transcription, MultipleShooting)


@dataclass(frozen=True)
class StageStructure:
  """One stage of a ``DiscreteOCP``: ``nx`` states, ``nu`` controls, ``nw`` of the stage's own
  variables (a transcription's internal states and controls), and ``n_dynamics`` equality residuals
  that relate them to the next state."""

  nx: int
  nu: int
  nw: int
  n_dynamics: int


__all__ = ["METHOD_API", "ContinuousOCP", "DiscreteOCP", "Param", "Path", "Quadratic", "StageStructure", "TerminalEquality", "transcribe"]
