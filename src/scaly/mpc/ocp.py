"""The OCP: dynamics, costs, constraints, horizon and transcription, and the ``sc.opt.problem`` it builds."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..function.model import ConcreteFunction, Function
from ..function.sugar import scan, vmap
from ..function.tree import G, L, param_list
from ..integrators.method import ODE
from ..integrators.method import solver as integrate
from ..integrators.model import check_model
from ..integrators.transcription import Interval, MultipleShooting, Transcription
from ..ir.expr import Expr, concat, gather
from ..ir.types import TensorType
from ..opt.problem import NLP, Bounded, ProblemSpec, bounded, problem

__all__ = ["OCP", "Path", "Quadratic", "TerminalEquality", "linear"]


def linear(a: Any, b: Any, *, name: str = "linear") -> ConcreteFunction[Any, Any, Any, Any]:
  """The discrete-time map ``x_next = A x + B u``, for an OCP's ``step=``: with ``Quadratic`` costs and
  polytopic constraints the OCP is a QP. A continuous-time pair goes through ``si.zoh`` first."""
  a, b = np.atleast_2d(np.asarray(a, dtype=np.float64)), np.asarray(b, dtype=np.float64)
  b = b.reshape(a.shape[0], -1)
  if a.shape[0] != a.shape[1]:
    raise ValueError(f"A must be square, got {a.shape}")

  def body(x: Expr, u: Expr) -> Expr:
    return Expr.const(a) @ x + Expr.const(b) @ u

  return ConcreteFunction(name, body, param_list(L("x", a.shape[0]), L("u", b.shape[1])), L("xnext", a.shape[0]))


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
class _Param:
  name: str
  type: TensorType


class OCP:
  """An optimal control problem over a horizon of ``horizon`` intervals, transcribed to an
  ``sc.opt.problem`` whose variables are the states at the grid points, the controls, and whatever the
  transcription adds, and whose parameters are the initial state and every parameter the Functions
  name.

  Args:
    ode: a continuous-time model ``f(x, u, *params) -> xdot``; or
    step: a discrete-time map ``F(x, u, *params) -> x_next``. Exactly one.
    horizon: the number of intervals ``N``.
    dt: the interval's length. Required with ``ode``; with ``step`` it only sets the times.
    transcription: how ``ode`` becomes constraints; ``si.MultipleShooting(si.RK4())`` by default.
    stage_cost: ``l(x, u, *params)`` (one value) or a ``Quadratic``. With ``ode``, the running cost
      is ``dt * sum_k l(x_k, u_k)`` (``cost="points"``) or the transcription's integral of ``l`` over
      each interval (``cost="integral"``); with ``step`` it is ``sum_k l(x_k, u_k)``.
    terminal_cost: ``Vf(x, *params)`` (one value) or a ``Quadratic`` without ``R``.
    x_bounds, u_bounds: ``(lo, hi)``, each a number, an array of the state's (control's) size or
      ``None``; on every state but the initial one, the transcription's internal states included,
      and on every control.
    constraints: ``Path`` constraints.
    terminal: a terminal set: ``TerminalEquality``, or a set from ``mpc.terminal``.
    varying: names of parameters that take one value per grid point, ``N + 1`` of them: the stage
      ``k`` reads value ``k``, the terminal cost value ``N``. The others hold over the horizon.
    cost: ``"points"`` or ``"integral"``, as above. By default ``"integral"`` for a transcription
      that puts controls inside an interval (``Pseudospectral``), whose controls the points alone
      would leave out of the cost, and ``"points"`` otherwise.
    condensed: eliminate the states: they become a ``scan`` of the map from ``x0``, the variables
      only the controls (and slacks), and state bounds inequalities. For a discrete map or multiple
      shooting. A linear OCP is then a dense QP in the controls, smaller and denser than the sparse
      form.
    name: the problem's name, by default ``{model}_ocp``; every generated Function is named from it.

  A Function's parameters are its inputs after the state and the control (after the state for a
  terminal cost), one vector each, and are matched by name across Functions: a model taking ``mass``
  and a stage cost taking ``r`` give the OCP the parameters ``mass`` and ``r``.
  """

  def __init__(
    self,
    *,
    ode: Function[Any, Any, Any, Any] | None = None,
    step: Function[Any, Any, Any, Any] | None = None,
    horizon: int,
    dt: float | None = None,
    transcription: Transcription | None = None,
    stage_cost: Function[Any, Any, Any, Any] | Quadratic | None = None,
    terminal_cost: Function[Any, Any, Any, Any] | Quadratic | None = None,
    x_bounds: tuple[Any, Any] | None = None,
    u_bounds: tuple[Any, Any] | None = None,
    constraints: Sequence[Path] = (),
    terminal: Any = None,
    varying: Sequence[str] = (),
    cost: Literal["points", "integral"] | None = None,
    condensed: bool = False,
    name: str | None = None,
  ) -> None:
    if (ode is None) == (step is None):
      raise ValueError("an OCP takes exactly one of ode= (a continuous-time model) and step= (a discrete-time map)")
    source = ode if ode is not None else step
    assert source is not None
    model = source.concrete
    check_model(model, model.name)
    if len(model.input_tree.parts) < 2 or len(model.inputs[1].shape) != 1:
      raise ValueError(f"{model.name}: the model takes the state, then the control, one vector: f(x, u, *params)")
    if int(horizon) != horizon or horizon < 1:
      raise ValueError(f"horizon must be a positive integer, got {horizon}")
    if ode is not None and not (dt is not None and float(dt) > 0):
      raise ValueError("a continuous-time OCP needs a positive dt, the interval's length")
    if cost not in (None, "points", "integral") or (cost == "integral" and ode is None):
      raise ValueError("cost is 'points', or 'integral' for a continuous-time model")
    self.name = name or f"{model.name}_ocp"
    self.continuous, self.horizon, self.dt = ode is not None, int(horizon), None if dt is None else float(dt)
    self.nx, self.nu = model.inputs[0].size, model.inputs[1].size
    self.transcription = transcription if transcription is not None else (MultipleShooting() if ode is not None else None)
    if self.transcription is not None and ode is None:
      raise ValueError("a transcription goes with ode=; a discrete-time map is its own transcription")
    node_controls = getattr(self.transcription, "node_controls", False)
    self.cost_rule = cost if cost is not None else "integral" if node_controls else "points"
    if condensed and not (ode is None or isinstance(self.transcription, MultipleShooting)) or (condensed and self.cost_rule == "integral"):
      raise ValueError("the condensed form takes a discrete map, or multiple shooting with costs at the points")
    self.condensed = bool(condensed)
    self.constraints, self.terminal, self.x_bounds, self.u_bounds = tuple(constraints), terminal, x_bounds, u_bounds

    # Every parameter any Function names, in order of appearance, one type per name.
    self.params: list[_Param] = []
    self._collect(model, 2)
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
    self._map = self._rollout_map() if self.condensed else None
    self.states = self._states() if self.condensed else None
    self.layout = _Layout(self)
    # The per-stage runs of the equality and inequality multipliers, which _problem records as it builds.
    self._eq_runs: list[tuple[int, int, int]] = []
    self._ineq_runs: list[tuple[int, int, int]] = []
    self.problem = self._problem()

  # -- parameters -------------------------------------------------------------------------------

  def _declare(self, name: str, type_: TensorType) -> None:
    for p in self.params:
      if p.name == name:
        if p.type.shape != type_.shape:
          raise ValueError(f"parameter {name!r} is {p.type.shape} in one Function and {type_.shape} in another")
        return
    if name in ("x0", "guess"):
      raise ValueError(f"a parameter may not be named {name!r}: the control law takes that input itself")
    self.params.append(_Param(name, TensorType(type_.shape, type_.dtype)))

  def _collect(self, fn: ConcreteFunction[Any, Any, Any, Any], lead: int) -> None:
    parts = fn.input_tree.parts
    if any(len(part.names) != 1 for part in parts[lead:]):
      raise ValueError(f"{fn.name}: every parameter after the state{' and the control' if lead == 2 else ''} must be one vector")
    for part, expr in zip(parts[lead:], fn.inputs[lead:], strict=True):
      self._declare(part.names[0], expr.type)

  def param_size(self, p: _Param) -> int:
    return p.type.size * (self.horizon + 1 if p.name in self.varying else 1)

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

  # -- the transcription ------------------------------------------------------------------------

  def _interval(self) -> Interval:
    """The interval every stage maps: the transcription's, or for a discrete map ``F(x, u) - xnext``."""
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

  def _stage_specs(
    self, xs: Expr, us: Expr, zs: Expr | None, params: dict[str, Expr], *, with_z: bool, with_next: bool
  ) -> list[tuple[Expr, int, int]]:
    """The ``vmap`` slices of one stage's arguments: state, control, [internal], [next state], params."""
    nx, nu, k = self.nx, self.nu, self.interval.n_internal
    specs = [(xs, 0, nx), (us, 0, nu)]
    if with_z and k:
      assert zs is not None
      specs.append((zs, 0, k))
    if with_next:
      specs.append((xs, nx, nx))
    for p in self.params:
      value = params[p.name]
      specs.append((value, 0, p.type.size if p.name in self.varying else 0))
    return specs

  def _problem(self) -> NLP[Any, Any, Any, Any]:
    layout, n, nx = self.layout, self.horizon, self.nx
    param_size = sum(self.param_size(p) for p in self.params)
    params_tree = G(L("x0", nx), L("p", param_size)) if self.params else L("x0", nx)

    def body(v: Any, prm: Any) -> ProblemSpec[Any]:
      leaves = dict(zip(layout.var_names, v if isinstance(v, tuple) else (v,), strict=True))
      us, zs, slack = leaves["us"], leaves.get("zs"), leaves.get("slack")
      params: dict[str, Expr] = {}
      if self.params:
        x0, flat = prm
        offset = 0
        for p in self.params:
          params[p.name] = flat[offset : offset + self.param_size(p)]
          offset += self.param_size(p)
      else:
        x0 = prm
      xs = self._rollout(x0, us, params) if self.condensed else leaves["xs"]
      eq, ineq = _Constraints(), _Constraints()
      terms: list[Expr] = []
      if not self.condensed:
        eq.add(xs[:nx] - x0)
        eq.add(vmap(self.interval.fn, n, self._stage_specs(xs, us, zs, params, with_z=True, with_next=True), output=0), self.interval.n_residual)
      elif self.x_bounds is not None:
        lo, hi = _pair(self.x_bounds, nx)
        rows = np.flatnonzero(np.isfinite(lo) | np.isfinite(hi))  # the constrained coordinates, every stage after the first
        if rows.size:
          picked = xs.reshape((n + 1, nx))[1:, :].reshape((n * nx,)) if rows.size == nx else gather_rows(xs, rows, n, nx)
          ineq.add(bounded(picked, lo=Expr.const(np.tile(lo[rows], n)), hi=Expr.const(np.tile(hi[rows], n)), name="states"), rows.size)
      if self.stage_cost is not None:
        if self.cost_rule == "integral":
          terms.append(vmap(self.interval.fn, n, self._stage_specs(xs, us, zs, params, with_z=True, with_next=True), output=1).sum())
        else:
          stage_costs = vmap(self.stage_cost, n, self._stage_specs(xs, us, zs, params, with_z=False, with_next=False)).sum()
          terms.append(stage_costs * (self.dt if self.continuous else 1.0))
      at_end = [self._at_end(params, p) for p in self.params]
      x_end = xs[n * nx :]
      if self.terminal_cost is not None:
        terms.append(self.terminal_cost(x_end, *at_end))
      for i, (spec, fn) in enumerate(zip(self.constraints, self.paths, strict=True)):
        g = vmap(fn, n, self._stage_specs(xs, us, zs, params, with_z=False, with_next=False))
        rows = g.size // n
        lo, hi = _tiled(spec.lo, rows, n), _tiled(spec.hi, rows, n)
        if spec.soft is None or slack is None:
          ineq.add(bounded(g, lo=lo, hi=hi, name=f"path{i}"), rows)
          continue
        s = slack[layout.slack_offsets[i] : layout.slack_offsets[i] + g.size]
        if hi is not None:
          ineq.add(bounded(g - s, hi=hi, name=f"path{i}_upper"), rows)
        if lo is not None:
          ineq.add(bounded(g + s, lo=lo, name=f"path{i}_lower"), rows)
        terms.append(float(spec.soft) * s.sum())
      if isinstance(self.terminal, TerminalEquality):
        ref = self.terminal.x_ref
        target = params[ref] if isinstance(ref, str) else Expr.const(np.zeros(nx) if ref is None else np.asarray(ref, dtype=np.float64))
        eq.add(x_end - (target if ref not in self.varying else target[n * nx :]))
      elif self.terminal is not None:
        eq_rows, ineq_rows = self.terminal.constraints(x_end)
        for row in eq_rows:
          eq.add(row)
        for group in ineq_rows:
          ineq.add(group)
      self._eq_runs, self._ineq_runs = eq.runs, ineq.runs
      objective = sum(terms[1:], terms[0]) if terms else Expr.const(0.0)
      lb, ub = layout.bounds()
      return ProblemSpec(minimize=objective, eq=tuple(eq.items), ineq=tuple(ineq.items), lb=lb, ub=ub)

    return problem(vars=layout.vars_tree, params=params_tree, name=self.name)(body)

  def _rollout_map(self) -> ConcreteFunction[Any, Any, Any, Any]:
    """One step of the condensed form's ``scan``: ``(x, u, *params) -> (x_next, x_next)``, the carry
    and the state it stacks."""
    if self.transcription is None:
      step = self.model
    else:
      assert isinstance(self.transcription, MultipleShooting)
      step = integrate(ODE(self.model, dt=self.dt), self.transcription.method, name=f"{self.name}_step")

    def body(x: Expr, u: Expr, *params: Expr) -> tuple[Expr, Expr]:
      nxt = step(x, u, *params)
      return nxt, nxt

    slots = param_list(L("x", self.nx), L("u", self.nu), *(L(p.name, p.type) for p in self.params))
    return ConcreteFunction(f"{self.name}_rollout", body, slots, G(L("xnext", self.nx), L("state", self.nx)))

  def _states(self) -> ConcreteFunction[Any, Any, Any, Any]:
    """``(x0, us, *params) -> xs``: the condensed form's states, for reading a solution."""
    slots = [L("x0", self.nx), L("us", self.horizon * self.nu), *(L(p.name, (self.param_size(p),)) for p in self.params)]

    def body(x0: Expr, us: Expr, *values: Expr) -> Expr:
      return self._rollout(x0, us, dict(zip((p.name for p in self.params), values, strict=True)))

    return ConcreteFunction(f"{self.name}_states", body, param_list(*slots), L("xs", ((self.horizon + 1) * self.nx,)))

  def _rollout(self, x0: Expr, us: Expr, params: dict[str, Expr]) -> Expr:
    """The states of the condensed form, ``x_0 .. x_N``, a ``scan`` of the map from ``x0``."""
    assert self._map is not None
    fn = self._map
    specs = [(us, 0, self.nu), *((params[p.name], 0, p.type.size if p.name in self.varying else 0) for p in self.params)]
    _, stacked = scan(fn, x0, specs, length=self.horizon)
    return concat([x0, stacked])

  def _at_end(self, params: dict[str, Expr], p: _Param) -> Expr:
    value = params[p.name]
    return value[self.horizon * p.type.size :] if p.name in self.varying else value

  @property
  def times(self) -> np.ndarray:
    """The grid times ``0, dt, ..., N dt`` (steps of 1 for a discrete map without ``dt``)."""
    return np.arange(self.horizon + 1) * (self.dt if self.dt is not None else 1.0)


class _Constraints:
  """Constraint groups as the problem lists them, and the ``(offset, size, block)`` runs of those
  that repeat per stage, for the warm start's shift."""

  def __init__(self) -> None:
    self.items: list[Any] = []
    self.runs: list[tuple[int, int, int]] = []
    self.size = 0

  def add(self, item: Any, block: int = 0) -> None:
    size = (item.expr if isinstance(item, Bounded) else item).size
    if block:
      self.runs.append((self.size, size, block))
    self.items.append(item)
    self.size += size


def gather_rows(xs: Expr, rows: np.ndarray, n: int, nx: int) -> Expr:
  """``xs[k nx + rows]`` for every stage ``k`` after the first, flat."""
  return gather(xs, (np.arange(1, n + 1)[:, None] * nx + rows[None, :]).reshape(-1))


def _tiled(bound: Any, rows: int, n: int) -> Expr | None:
  if bound is None:
    return None
  value = np.broadcast_to(np.asarray(bound, dtype=np.float64), (rows,))
  return Expr.const(np.tile(value, n))


class _Layout:
  """Where everything sits in the problem's variables and multipliers, and how a solution shifts by
  one interval to warm-start the next: every stage block moves up one, and the last is repeated."""

  def __init__(self, ocp: OCP) -> None:
    n, nx, nu, k = ocp.horizon, ocp.nx, ocp.nu, ocp.interval.n_internal
    self.ocp = ocp
    self.slack_offsets, rows = [], 0
    for i, spec in enumerate(ocp.constraints):
      rows_per_stage = ocp.paths[i].outputs[0].size
      self.slack_offsets.append(rows if spec.soft is not None else -1)
      rows += rows_per_stage * n if spec.soft is not None else 0
    states = [] if ocp.condensed else [("xs", (n + 1) * nx, nx)]
    leaves = [*states, ("us", n * nu, nu), *([("zs", n * k, k)] if k else []), *([("slack", rows, rows // n)] if rows else [])]
    self.var_names = tuple(name for name, _, _ in leaves)
    self.var_sizes = tuple(size for _, size, _ in leaves)
    self.var_blocks = tuple(block for _, _, block in leaves)  # the size of one stage's block
    self.vars_tree = G(*(L(name, size) for name, size, _ in leaves)) if len(leaves) > 1 else L(leaves[0][0], leaves[0][1])
    self.n_vars = sum(self.var_sizes)

  def bounds(self) -> tuple[Any, Any]:
    """``(lb, ub)`` with the variables' tree structure: states bounded after the first, internal
    states and controls as the transcription lays them out, slacks nonnegative."""
    ocp, n, nx, nu = self.ocp, self.ocp.horizon, self.ocp.nx, self.ocp.nu
    x_lo, x_hi = _pair(ocp.x_bounds, nx)
    u_lo, u_hi = _pair(ocp.u_bounds, nu)
    lower, upper = [], []
    for name in self.var_names:
      if name == "xs":
        lo, hi = np.concatenate([np.full(nx, -np.inf), np.tile(x_lo, n)]), np.concatenate([np.full(nx, np.inf), np.tile(x_hi, n)])
      elif name == "us":
        lo, hi = np.tile(u_lo, n), np.tile(u_hi, n)
      elif name == "zs":
        interval = ocp.interval
        states, controls = interval.state_times.size, interval.control_times.size
        lo = np.tile(np.concatenate([np.tile(x_lo, states), np.tile(u_lo, controls)]), n)
        hi = np.tile(np.concatenate([np.tile(x_hi, states), np.tile(u_hi, controls)]), n)
      else:
        lo, hi = np.zeros(self.var_sizes[self.var_names.index(name)]), np.full(self.var_sizes[self.var_names.index(name)], np.inf)
      lower.append(Expr.const(lo))
      upper.append(Expr.const(hi))
    return (tuple(lower), tuple(upper)) if len(lower) > 1 else (lower[0], upper[0])

  def multiplier_blocks(self) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """``(offset, size, stage block)`` runs of the equality and inequality multipliers that are per
    stage and so shift, as the problem recorded them when it was built; the rest (the initial
    state's, a terminal set's) stay."""
    return self.ocp._eq_runs, self.ocp._ineq_runs


def _pair(bounds: tuple[Any, Any] | None, size: int) -> tuple[np.ndarray, np.ndarray]:
  lo, hi = (None, None) if bounds is None else bounds
  as_array: Callable[[Any, float], np.ndarray] = lambda v, fill: (
    np.full(size, fill) if v is None else np.broadcast_to(np.asarray(v, dtype=np.float64), (size,)).copy()
  )  # noqa: E731
  return as_array(lo, -np.inf), as_array(hi, np.inf)
