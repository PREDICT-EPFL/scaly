"""The MPC controller: an OCP's solver, its control law with the warm start shifted in generated code, and closed-loop simulation."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..function.model import ConcreteFunction
from ..function.tree import G, L, param_list
from ..ir.expr import Expr, concat
from ..solvers.solver import solver as make_solver
from ..solvers.stats import SolverStatus
from .ocp import OCP

__all__ = ["MPC", "ClosedLoop", "Solution", "simulate"]


@dataclass(frozen=True)
class Solution:
  """One solve of an OCP.

  Attributes:
    xs: the states at the grid points, ``(N + 1, nx)``.
    us: the controls, ``(N, nu)``; ``us[0]`` is what a receding horizon applies.
    zs: the transcription's own variables per interval, ``(N, n_internal)``, or ``None``.
    slack: the soft constraints' slacks, flat, or ``None``.
    times: the grid times.
    cost: the optimal cost.
    status: the solver's status; ``status.ok`` for a converged or acceptable solve.
    guess: the whole primal-dual point, flat, as the next solve's warm start takes it (unshifted).
  """

  xs: np.ndarray
  us: np.ndarray
  zs: np.ndarray | None
  slack: np.ndarray | None
  times: np.ndarray
  cost: float
  status: SolverStatus
  guess: np.ndarray


@dataclass(frozen=True)
class ClosedLoop:
  """A closed-loop simulation: ``xs`` the ``steps + 1`` states, ``us`` the ``steps`` controls applied,
  and per step the solver's status name, its iterations and its own solve time in seconds."""

  xs: np.ndarray
  us: np.ndarray
  statuses: list[str]
  iterations: np.ndarray
  solve_times: np.ndarray


class MPC:
  """A model predictive controller: ``ocp`` solved by ``solver`` (``"ipopt"``, ``"sqp"``, or ``"piqp"``
  for a problem that is a QP) with ``options``, from a warm start that each call shifts by one
  interval. PIQP runs its sparse backend unless the OCP is condensed or ``options`` say otherwise.

  ``controller(x, **params)`` returns the first control. ``controller.law`` is the same control law as
  one Function, ``law(x0, *params, guess) -> (u, guess_next)``, the solver nested in it and the shift
  done in its code, so that generated C holds one buffer of ``guess_size`` doubles between calls;
  ``scaly.codegen.write_module(controller.law, ...)`` writes it. Parameters come in the OCP's order
  (``ocp.params``), a varying one with ``N + 1`` values.
  """

  def __init__(self, ocp: OCP, solver: str = "ipopt", *, options: dict[str, Any] | None = None, name: str | None = None) -> None:
    self.ocp, self.backend = ocp, solver
    self.solver_name = name or f"{ocp.name}_{solver}"
    if solver == "piqp" and not ocp.condensed:
      options = {"sparse": True, **(options or {})}  # a horizon's QP is banded: PIQP's sparse backend
    self.solver = make_solver(ocp.problem, solver, name=self.solver_name, options=options)
    layout = ocp.layout
    self.n_eq, self.n_ineq = ocp.problem.n_eq, ocp.problem.n_ineq
    self.guess_size = 2 * layout.n_vars + self.n_eq + self.n_ineq
    self.shift = self._shift_function()
    self.law = self._law()
    # The problem's own objective, for a solution's cost: a QP solver's objective leaves out the
    # constant terms, which in the condensed form are everything that depends on x0 alone.
    problem = ocp.problem
    symbols = [*problem._var_symbols, *problem._param_symbols]
    self.objective = ConcreteFunction._from_exprs(
      f"{ocp.name}_objective", symbols, [problem.spec.minimize], [f"a{i}" for i in range(len(symbols))], ["cost"]
    )
    self._guess: np.ndarray | None = None

  # -- the flat primal-dual point -----------------------------------------------------------------

  def _split(self, flat: Any) -> tuple[list[Any], list[Any], Any, Any]:
    layout, cut, n = self.ocp.layout, 0, self.ocp.layout.n_vars
    primal, box = [], []
    for size in layout.var_sizes:
      primal.append(flat[cut : cut + size])
      box.append(flat[n + cut : n + cut + size])
      cut += size
    return primal, box, flat[2 * n : 2 * n + self.n_eq], flat[2 * n + self.n_eq :]

  def _shifted(self, flat: Expr) -> Expr:
    """The point moved up one interval: each per-stage block of a variable, of a bound's multiplier,
    of the dynamics' and the path constraints' multipliers, by one stage, the last repeated."""
    primal, box, lam_eq, lam_ineq = self._split(flat)
    layout = self.ocp.layout

    def moved(seg: Expr, block: int) -> Expr:
      return seg if block == 0 or seg.size <= block else concat([seg[block:], seg[seg.size - block :]])

    def runs(seg: Expr, blocks: list[tuple[int, int, int]]) -> Expr:
      pieces, cut = [], 0
      for offset, size, block in blocks:
        pieces += [seg[cut:offset], moved(seg[offset : offset + size], block)]
        cut = offset + size
      pieces.append(seg[cut:])
      return concat([p for p in pieces if p.size]) if any(p.size for p in pieces) else seg

    per_leaf = [self._leaf_runs(name, size) for name, size in zip(layout.var_names, layout.var_sizes, strict=True)]
    eq_runs, ineq_runs = layout.multiplier_blocks()
    parts = [runs(p, r) for p, r in zip(primal, per_leaf, strict=True)] + [runs(b, r) for b, r in zip(box, per_leaf, strict=True)]
    return concat([*parts, *([runs(lam_eq, eq_runs)] if self.n_eq else []), *([runs(lam_ineq, ineq_runs)] if self.n_ineq else [])])

  def _leaf_runs(self, name: str, size: int) -> list[tuple[int, int, int]]:
    ocp, layout = self.ocp, self.ocp.layout
    if name != "slack":
      return [(0, size, layout.var_blocks[layout.var_names.index(name)])]
    return [
      (offset, fn.outputs[0].size * ocp.horizon, fn.outputs[0].size)
      for offset, fn in zip(layout.slack_offsets, ocp.paths, strict=True)
      if offset >= 0
    ]

  def _shift_function(self) -> ConcreteFunction[Any, Any, Any, Any]:
    return ConcreteFunction(f"{self.ocp.name}_shift", self._shifted, param_list(L("guess", self.guess_size)), L("guess_next", self.guess_size))

  def _solver_args(self, flat: Any, x0: Any, param_values: list[Any], concat_fn: Callable[[list[Any]], Any]) -> tuple[Any, ...]:
    primal, box, lam_eq, lam_ineq = self._split(flat)
    prm = (x0, concat_fn(param_values)) if self.ocp.params else x0
    return _tree(primal), _tree(box), lam_eq, lam_ineq, prm

  def _solve(self, *args: Any) -> tuple[list[Any], list[Any], Any, Any]:
    """The solver's four outputs, the variable trees as lists of leaves (one leaf comes bare)."""
    primal, box, lam_eq, lam_ineq = self.solver(*args)
    return _leaves(primal), _leaves(box), lam_eq, lam_ineq

  def _law(self) -> ConcreteFunction[Any, Any, Any, Any]:
    ocp = self.ocp

    def body(x0: Expr, *rest: Expr) -> tuple[Expr, Expr]:
      *param_values, guess = rest
      primal, box, lam_eq, lam_ineq = self._solve(*self._solver_args(guess, x0, [p.reshape((p.size,)) for p in param_values], concat))
      solution = concat([*primal, *box, *([lam_eq] if self.n_eq else []), *([lam_ineq] if self.n_ineq else [])])
      return primal[ocp.layout.var_names.index("us")][: ocp.nu], self._shifted(solution)

    slots = [L("x0", ocp.nx), *(L(p.name, (ocp.param_size(p),)) for p in ocp.params), L("guess", self.guess_size)]
    return ConcreteFunction(f"{ocp.name}_law", body, param_list(*slots), G(L("u", ocp.nu), L("guess_next", self.guess_size)))

  # -- Python-side use -----------------------------------------------------------------------------

  def initial_guess(self, x0: Any, u: Any = None) -> np.ndarray:
    """A starting point: every state ``x0``, every control ``u`` (zeros by default), the
    transcription's own variables from its ``guess``, slacks and multipliers zero."""
    ocp, layout = self.ocp, self.ocp.layout
    x0, u = np.ravel(np.asarray(x0, dtype=np.float64)), np.zeros(ocp.nu) if u is None else np.ravel(np.asarray(u, dtype=np.float64))
    values = {"xs": np.tile(x0, ocp.horizon + 1), "us": np.tile(u, ocp.horizon), "zs": np.tile(ocp.interval.guess(x0, u), ocp.horizon)}
    primal = [values.get(name, np.zeros(size)) for name, size in zip(layout.var_names, layout.var_sizes, strict=True)]
    return np.concatenate([*primal, np.zeros(self.guess_size - layout.n_vars)])

  def reset(self, guess: np.ndarray | None = None) -> None:
    """Forget the warm start, or set it: the next call starts from ``guess``, or from ``initial_guess``."""
    self._guess = None if guess is None else np.asarray(guess, dtype=np.float64)

  def _params(self, given: dict[str, Any]) -> list[np.ndarray]:
    names = [p.name for p in self.ocp.params]
    unknown = set(given) - set(names)
    missing = [n for n in names if n not in given]
    if unknown or missing:
      raise TypeError(f"{self.ocp.name} takes the parameters {names}; got {sorted(given)}")
    return [np.ravel(np.asarray(given[p.name], dtype=np.float64)) for p in self.ocp.params]

  def __call__(self, x0: Any, **params: Any) -> np.ndarray:
    """The control to apply at state ``x0``: the first of the solution, from the shifted last one."""
    x0 = np.ravel(np.asarray(x0, dtype=np.float64))
    guess = self._guess if self._guess is not None else self.initial_guess(x0)
    u, self._guess = self.law(x0, *self._params(params), guess)
    return np.asarray(u)

  def solve(self, x0: Any, *, guess: np.ndarray | None = None, **params: Any) -> Solution:
    """The whole solution at ``x0``, from ``guess`` or the warm start; the warm start moves on to it."""
    ocp, layout = self.ocp, self.ocp.layout
    x0 = np.ravel(np.asarray(x0, dtype=np.float64))
    start = guess if guess is not None else self._guess if self._guess is not None else self.initial_guess(x0)
    values = self._params(params)
    primal, box, lam_eq, lam_ineq = self._solve(*self._solver_args(start, x0, values, np.concatenate))
    flat = np.concatenate([np.ravel(a) for a in (*primal, *box, lam_eq, lam_ineq)])
    self._guess = np.asarray(self.shift(flat))
    leaves = dict(zip(layout.var_names, (np.asarray(a) for a in primal), strict=True))
    k = ocp.interval.n_internal
    stats = self.solver.solver_stats()
    prm = [x0, np.concatenate(values)] if ocp.params else [x0]
    cost = self.objective((*(np.ravel(a) for a in primal), *prm))  # its inputs are one group, as _from_exprs builds it
    return Solution(
      xs=(leaves["xs"] if ocp.states is None else np.asarray(ocp.states(x0, leaves["us"], *values))).reshape(ocp.horizon + 1, ocp.nx),
      us=leaves["us"].reshape(ocp.horizon, ocp.nu),
      zs=leaves["zs"].reshape(ocp.horizon, k) if k else None,
      slack=leaves.get("slack"),
      times=ocp.times,
      cost=float(cost),
      status=stats.to_solver_status(),
      guess=flat,
    )

  @property
  def status(self) -> SolverStatus:
    """The status of the last call of the control law."""
    return self.law.solver_stats(self.solver_name).to_solver_status()


def _tree(leaves: list[Any]) -> Any:
  return leaves[0] if len(leaves) == 1 else tuple(leaves)


def _leaves(tree: Any) -> list[Any]:
  return list(tree) if isinstance(tree, tuple) else [tree]


def simulate(controller: MPC, plant: Callable[[np.ndarray, np.ndarray], Any], x0: Any, steps: int, **params: Any) -> ClosedLoop:
  """Run ``controller`` in closed loop on ``plant(x, u) -> x_next`` (an ``si.adaptive`` map, say) for
  ``steps`` intervals from ``x0``. A parameter may be a callable of the step number."""
  x = np.ravel(np.asarray(x0, dtype=np.float64))
  xs, us, statuses, iterations, times = [x], [], [], [], []
  for k in range(int(steps)):
    values = {name: value(k) if callable(value) else value for name, value in params.items()}
    started = time.perf_counter()
    u = controller(x, **values)
    elapsed = time.perf_counter() - started
    stats = controller.law.solver_stats(controller.solver_name)
    statuses.append(stats.to_solver_status().name)
    iterations.append(stats.iter)
    times.append(stats.t_total if stats.t_total > 0 else elapsed)
    x = np.ravel(np.asarray(plant(x, u), dtype=np.float64))
    xs.append(x)
    us.append(u)
  return ClosedLoop(np.array(xs), np.array(us), statuses, np.array(iterations), np.array(times))
