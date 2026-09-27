"""The OCP: dynamics, costs, constraints, horizon and transcription, and the ``sc.problem`` it builds."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..function.model import ConcreteFunction, Function
from ..function.sugar import vmap
from ..function.tree import G, L, param_list
from ..integrators.model import check_model
from ..integrators.transcription import Interval, MultipleShooting, Transcription
from ..ir.expr import Expr
from ..ir.types import TensorType
from ..solvers.problem import Problem, ProblemSpec, bounded, problem

__all__ = ["OCP", "Path", "Quadratic", "TerminalEquality"]


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
  ``sc.problem`` whose variables are the states at the grid points, the controls, and whatever the
  transcription adds, and whose parameters are the initial state and every parameter the Functions
  name.

  Args:
    ode: a continuous-time model ``f(x, u, *params) -> xdot``; or
    step: a discrete-time map ``F(x, u, *params) -> x_next``. Exactly one.
    horizon: the number of intervals ``N``.
    dt: the interval's length. Required with ``ode``; with ``step`` it only sets the times.
    transcription: how ``ode`` becomes constraints; ``si.MultipleShooting(si.rk4)`` by default.
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
    self.layout = _Layout(self)
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

  def _problem(self) -> Problem[Any, Any, Any, Any]:
    layout, n, nx = self.layout, self.horizon, self.nx
    param_size = sum(self.param_size(p) for p in self.params)
    params_tree = G(L("x0", nx), L("p", param_size)) if self.params else L("x0", nx)

    def body(v: Any, prm: Any) -> ProblemSpec[Any]:
      leaves = dict(zip(layout.var_names, v if isinstance(v, tuple) else (v,), strict=True))
      xs, us, zs, slack = leaves["xs"], leaves["us"], leaves.get("zs"), leaves.get("slack")
      params: dict[str, Expr] = {}
      if self.params:
        x0, flat = prm
        offset = 0
        for p in self.params:
          params[p.name] = flat[offset : offset + self.param_size(p)]
          offset += self.param_size(p)
      else:
        x0 = prm
      mapped = vmap(self.interval.fn, n, self._stage_specs(xs, us, zs, params, with_z=True, with_next=True), output=0)
      terms: list[Expr] = []
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
      eq: list[Expr] = [xs[:nx] - x0, mapped]
      ineq = []
      for i, (spec, fn) in enumerate(zip(self.constraints, self.paths, strict=True)):
        g = vmap(fn, n, self._stage_specs(xs, us, zs, params, with_z=False, with_next=False))
        lo, hi = _tiled(spec.lo, g.size // n, n), _tiled(spec.hi, g.size // n, n)
        if spec.soft is None or slack is None:
          ineq.append(bounded(g, lo=lo, hi=hi, name=f"path{i}"))
          continue
        s = slack[layout.slack_offsets[i] : layout.slack_offsets[i] + g.size]
        if hi is not None:
          ineq.append(bounded(g - s, hi=hi, name=f"path{i}_upper"))
        if lo is not None:
          ineq.append(bounded(g + s, lo=lo, name=f"path{i}_lower"))
        terms.append(float(spec.soft) * s.sum())
      if isinstance(self.terminal, TerminalEquality):
        ref = self.terminal.x_ref
        target = params[ref] if isinstance(ref, str) else Expr.const(np.zeros(nx) if ref is None else np.asarray(ref, dtype=np.float64))
        eq.append(x_end - (target if ref not in self.varying else target[n * nx :]))
      elif self.terminal is not None:
        eq_rows, ineq_rows = self.terminal.constraints(x_end)
        eq += eq_rows
        ineq += ineq_rows
      objective = sum(terms[1:], terms[0]) if terms else Expr.const(0.0)
      lb, ub = layout.bounds()
      return ProblemSpec(minimize=objective, eq=tuple(eq), ineq=tuple(ineq), lb=lb, ub=ub)

    return problem(vars=layout.vars_tree, params=params_tree, name=self.name)(body)

  def _at_end(self, params: dict[str, Expr], p: _Param) -> Expr:
    value = params[p.name]
    return value[self.horizon * p.type.size :] if p.name in self.varying else value

  @property
  def times(self) -> np.ndarray:
    """The grid times ``0, dt, ..., N dt`` (steps of 1 for a discrete map without ``dt``)."""
    return np.arange(self.horizon + 1) * (self.dt if self.dt is not None else 1.0)


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
    leaves = [("xs", (n + 1) * nx, nx), ("us", n * nu, nu), *([("zs", n * k, k)] if k else []), *([("slack", rows, rows // n)] if rows else [])]
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
    """``(offset, size, stage block)`` runs of the equality and inequality multipliers that are
    per stage and so shift; the rest (the initial state's, the terminal set's) stay."""
    ocp, n = self.ocp, self.ocp.horizon
    eq = [(ocp.nx, n * ocp.interval.n_residual, ocp.interval.n_residual)]
    ineq, offset = [], 0
    for spec, fn in zip(ocp.constraints, ocp.paths, strict=True):
      rows = fn.outputs[0].size
      groups = 1 if spec.soft is None else int(spec.hi is not None) + int(spec.lo is not None)
      for _ in range(groups):
        ineq.append((offset, n * rows, rows))
        offset += n * rows
    return eq, ineq


def _pair(bounds: tuple[Any, Any] | None, size: int) -> tuple[np.ndarray, np.ndarray]:
  lo, hi = (None, None) if bounds is None else bounds
  as_array: Callable[[Any, float], np.ndarray] = lambda v, fill: (
    np.full(size, fill) if v is None else np.broadcast_to(np.asarray(v, dtype=np.float64), (size,)).copy()
  )  # noqa: E731
  return as_array(lo, -np.inf), as_array(hi, np.inf)
