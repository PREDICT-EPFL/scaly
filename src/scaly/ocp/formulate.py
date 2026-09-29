"""Formulation: a ``DiscreteOCP`` as an ``sc.opt`` problem, sparse (every state a variable) or condensed (the states a rollout of the controls), and the ``Layout`` of its variables and multipliers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from ..function.model import ConcreteFunction
from ..function.sugar import scan, vmap
from ..function.tree import G, L, param_list
from ..integrators.method import ODE
from ..integrators.method import solver as integrate
from ..ir.expr import Expr, concat, gather
from ..opt.problem import NLP, Bounded, ProblemSpec, bounded, problem
from .problem import DiscreteOCP, TerminalEquality
from .transcription import MultipleShooting

type Form = Literal["sparse", "condensed"]


def to_problem(ocp: DiscreteOCP, form: Form = "sparse") -> tuple[NLP[Any, Any, Any, Any], Layout]:
  """``ocp`` as an ``sc.opt`` problem, and the ``Layout`` of its variables and multipliers.

  ``form="sparse"``: the variables are every state ``x_1 .. x_N`` (``x_0`` too, pinned to the
  parameter), the controls, the stages' own variables and the soft constraints' slacks, and the
  dynamics are equality constraints, banded stage by stage. ``form="condensed"``: the states are a
  ``scan`` of the stage map from ``x0``, the variables only the controls (and slacks), and the state
  bounds inequalities; for a discrete-time map or multiple shooting with costs at the points. A
  linear OCP is then a dense QP in the controls, smaller and denser than the sparse form.

  The problem's parameters are the initial state ``x0``, then the OCP's parameters as one flat
  vector ``p`` in the order of ``ocp.params``. Built once per form and kept."""
  if form not in ("sparse", "condensed"):
    raise ValueError(f"form is 'sparse' or 'condensed', got {form!r}")
  if form == "condensed" and (not ocp.shooting or ocp.cost_rule == "integral"):
    raise ValueError("the condensed form takes a discrete map, or multiple shooting with costs at the points")
  if form not in ocp._formulations:
    layout = Layout(ocp, form == "condensed")
    ocp._formulations[form] = (_problem(ocp, layout), layout)
  return ocp._formulations[form]


class Layout:
  """Where everything sits in a formulated OCP's variables and multipliers: the variable leaves
  (``xs`` in the sparse form, ``us``, ``zs`` for the stages' own variables, ``slack``), their sizes
  and per-stage blocks, and the per-stage runs of the multipliers, which is how a solution shifts by
  one stage to warm-start the next. The condensed form's ``states`` Function rolls the states out."""

  def __init__(self, ocp: DiscreteOCP, condensed: bool) -> None:
    n, nx, nu, k = ocp.N, ocp.nx, ocp.nu, ocp.interval.n_internal
    self.ocp, self.condensed = ocp, condensed
    self.slack_offsets, rows = [], 0
    for i, spec in enumerate(ocp.constraints):
      rows_per_stage = ocp.paths[i].outputs[0].size
      self.slack_offsets.append(rows if spec.soft is not None else -1)
      rows += rows_per_stage * n if spec.soft is not None else 0
    states = [] if condensed else [("xs", (n + 1) * nx, nx)]
    leaves = [*states, ("us", n * nu, nu), *([("zs", n * k, k)] if k else []), *([("slack", rows, rows // n)] if rows else [])]
    self.var_names = tuple(name for name, _, _ in leaves)
    self.var_sizes = tuple(size for _, size, _ in leaves)
    self.var_blocks = tuple(block for _, _, block in leaves)  # the size of one stage's block
    self.vars_tree = G(*(L(name, size) for name, size, _ in leaves)) if len(leaves) > 1 else L(leaves[0][0], leaves[0][1])
    self.n_vars = sum(self.var_sizes)
    # The per-stage runs of the equality and inequality multipliers, which the problem records as it is built.
    self.eq_runs: list[tuple[int, int, int]] = []
    self.ineq_runs: list[tuple[int, int, int]] = []
    self.map = _rollout_map(ocp) if condensed else None
    self.states = _states(ocp, self) if condensed else None

  def bounds(self) -> tuple[Any, Any]:
    """``(lb, ub)`` with the variables' tree structure: states bounded after the first, internal
    states and controls as the transcription lays them out, slacks nonnegative."""
    ocp, n, nx, nu = self.ocp, self.ocp.N, self.ocp.nx, self.ocp.nu
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
    return self.eq_runs, self.ineq_runs


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


def _stage_specs(
  ocp: DiscreteOCP, xs: Expr, us: Expr, zs: Expr | None, params: dict[str, Expr], *, with_z: bool, with_next: bool
) -> list[tuple[Expr, int, int]]:
  """The ``vmap`` slices of one stage's arguments: state, control, [internal], [next state], params."""
  nx, nu, k = ocp.nx, ocp.nu, ocp.interval.n_internal
  specs = [(xs, 0, nx), (us, 0, nu)]
  if with_z and k:
    assert zs is not None
    specs.append((zs, 0, k))
  if with_next:
    specs.append((xs, nx, nx))
  for p in ocp.params:
    value = params[p.name]
    specs.append((value, 0, p.type.size if p.name in ocp.varying else 0))
  return specs


def _problem(ocp: DiscreteOCP, layout: Layout) -> NLP[Any, Any, Any, Any]:
  n, nx = ocp.N, ocp.nx
  param_size = sum(ocp.param_size(p) for p in ocp.params)
  params_tree = G(L("x0", nx), L("p", param_size)) if ocp.params else L("x0", nx)

  def body(v: Any, prm: Any) -> ProblemSpec[Any]:
    leaves = dict(zip(layout.var_names, v if isinstance(v, tuple) else (v,), strict=True))
    us, zs, slack = leaves["us"], leaves.get("zs"), leaves.get("slack")
    params: dict[str, Expr] = {}
    if ocp.params:
      x0, flat = prm
      offset = 0
      for p in ocp.params:
        params[p.name] = flat[offset : offset + ocp.param_size(p)]
        offset += ocp.param_size(p)
    else:
      x0 = prm
    xs = _rollout(ocp, layout, x0, us, params) if layout.condensed else leaves["xs"]
    eq, ineq = _Constraints(), _Constraints()
    terms: list[Expr] = []
    if not layout.condensed:
      eq.add(xs[:nx] - x0)
      eq.add(vmap(ocp.interval.fn, n, _stage_specs(ocp, xs, us, zs, params, with_z=True, with_next=True), output=0), ocp.interval.n_residual)
    elif ocp.x_bounds is not None:
      lo, hi = _pair(ocp.x_bounds, nx)
      rows = np.flatnonzero(np.isfinite(lo) | np.isfinite(hi))  # the constrained coordinates, every stage after the first
      if rows.size:
        picked = xs.reshape((n + 1, nx))[1:, :].reshape((n * nx,)) if rows.size == nx else _gather_rows(xs, rows, n, nx)
        ineq.add(bounded(picked, lo=Expr.const(np.tile(lo[rows], n)), hi=Expr.const(np.tile(hi[rows], n)), name="states"), rows.size)
    if ocp.stage_cost is not None:
      if ocp.cost_rule == "integral":
        terms.append(vmap(ocp.interval.fn, n, _stage_specs(ocp, xs, us, zs, params, with_z=True, with_next=True), output=1).sum())
      else:
        stage_costs = vmap(ocp.stage_cost, n, _stage_specs(ocp, xs, us, zs, params, with_z=False, with_next=False)).sum()
        terms.append(stage_costs * (ocp.dt if ocp.continuous else 1.0))
    at_end = [ocp.at_end(params, p) for p in ocp.params]
    x_end = xs[n * nx :]
    if ocp.terminal_cost is not None:
      terms.append(ocp.terminal_cost(x_end, *at_end))
    for i, (spec, fn) in enumerate(zip(ocp.constraints, ocp.paths, strict=True)):
      g = vmap(fn, n, _stage_specs(ocp, xs, us, zs, params, with_z=False, with_next=False))
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
    if isinstance(ocp.terminal, TerminalEquality):
      ref = ocp.terminal.x_ref
      target = params[ref] if isinstance(ref, str) else Expr.const(np.zeros(nx) if ref is None else np.asarray(ref, dtype=np.float64))
      eq.add(x_end - (target if ref not in ocp.varying else target[n * nx :]))
    elif ocp.terminal is not None:
      groups = ocp.terminal.constraints(x_end)
      for i, (g, lo, hi) in enumerate(groups):
        ineq.add(bounded(g, lo=lo, hi=hi, name="terminal_set" if len(groups) == 1 else f"terminal_set{i}"))
    layout.eq_runs, layout.ineq_runs = eq.runs, ineq.runs
    objective = sum(terms[1:], terms[0]) if terms else Expr.const(0.0)
    lb, ub = layout.bounds()
    return ProblemSpec(minimize=objective, eq=tuple(eq.items), ineq=tuple(ineq.items), lb=lb, ub=ub)

  return problem(vars=layout.vars_tree, params=params_tree, name=ocp.name)(body)


def _rollout_map(ocp: DiscreteOCP) -> ConcreteFunction[Any, Any, Any, Any]:
  """One step of the condensed form's ``scan``: ``(x, u, *params) -> (x_next, x_next)``, the carry
  and the state it stacks."""
  if ocp.transcription is None:
    step = ocp.model
  else:
    assert isinstance(ocp.transcription, MultipleShooting)
    step = integrate(ODE(ocp.model, dt=ocp.dt), ocp.transcription.method, name=f"{ocp.name}_step")

  def body(x: Expr, u: Expr, *params: Expr) -> tuple[Expr, Expr]:
    nxt = step(x, u, *params)
    return nxt, nxt

  slots = param_list(L("x", ocp.nx), L("u", ocp.nu), *(L(p.name, p.type) for p in ocp.params))
  return ConcreteFunction(f"{ocp.name}_rollout", body, slots, G(L("xnext", ocp.nx), L("state", ocp.nx)))


def _states(ocp: DiscreteOCP, layout: Layout) -> ConcreteFunction[Any, Any, Any, Any]:
  """``(x0, us, *params) -> xs``: the condensed form's states, for reading a solution."""
  slots = [L("x0", ocp.nx), L("us", ocp.N * ocp.nu), *(L(p.name, (ocp.param_size(p),)) for p in ocp.params)]

  def body(x0: Expr, us: Expr, *values: Expr) -> Expr:
    return _rollout(ocp, layout, x0, us, dict(zip((p.name for p in ocp.params), values, strict=True)))

  return ConcreteFunction(f"{ocp.name}_states", body, param_list(*slots), L("xs", ((ocp.N + 1) * ocp.nx,)))


def _rollout(ocp: DiscreteOCP, layout: Layout, x0: Expr, us: Expr, params: dict[str, Expr]) -> Expr:
  """The states of the condensed form, ``x_0 .. x_N``, a ``scan`` of the map from ``x0``."""
  assert layout.map is not None
  specs = [(us, 0, ocp.nu), *((params[p.name], 0, p.type.size if p.name in ocp.varying else 0) for p in ocp.params)]
  _, stacked = scan(layout.map, x0, specs, length=ocp.N)
  return concat([x0, stacked])


def _gather_rows(xs: Expr, rows: np.ndarray, n: int, nx: int) -> Expr:
  """``xs[k nx + rows]`` for every stage ``k`` after the first, flat."""
  return gather(xs, (np.arange(1, n + 1)[:, None] * nx + rows[None, :]).reshape(-1))


def _tiled(bound: Any, rows: int, n: int) -> Expr | None:
  if bound is None:
    return None
  value = np.broadcast_to(np.asarray(bound, dtype=np.float64), (rows,))
  return Expr.const(np.tile(value, n))


def _pair(bounds: tuple[Any, Any] | None, size: int) -> tuple[np.ndarray, np.ndarray]:
  lo, hi = (None, None) if bounds is None else bounds
  as_array: Callable[[Any, float], np.ndarray] = lambda v, fill: (
    np.full(size, fill) if v is None else np.broadcast_to(np.asarray(v, dtype=np.float64), (size,)).copy()
  )  # noqa: E731
  return as_array(lo, -np.inf), as_array(hi, np.inf)


__all__ = ["Form", "Layout", "to_problem"]
