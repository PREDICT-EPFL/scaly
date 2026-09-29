"""``ALTRO`` (experimental): augmented-Lagrangian iLQR with an optional projected Newton phase, the whole solve generated as loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ..function.api import function, gradient, hessian, jacobian, sparse_jacobian
from ..function.method import Status, Support
from ..function.model import ConcreteFunction
from ..function.sugar import scan, vmap, while_loop
from ..function.tree import G, L, param_list
from ..ir.expr import Expr, cast, concat, equal, greater, greater_equal, isfinite, less, less_equal, logical_and, logical_or, maximum, minimum, where
from ..linalg import SparseLDL, SparseMatrix, cho_solve, cholesky
from .formulate import step_map
from .method import Info
from .problem import METHOD_API, DiscreteOCP, TerminalEquality
from ..utils.experimental import warn_experimental

warn_experimental(__name__)


@dataclass(frozen=True)
class ALTRO:
  """ALTRO (Howell, Jackson and Manchester, IROS 2019), following Altro.jl 0.5 (experimental).

  For a ``DiscreteOCP`` whose stage is a map, with costs at the points: the inner solver is iLQR on
  the augmented Lagrangian ``J + sum (lambda'c + 1/2 c' I_mu c) + nu'g + mu/2 |g|^2`` of the stage
  inequalities ``c(x_k, u_k) <= 0`` (the control bounds at every stage, the state bounds after the
  first, the hard ``Path`` constraints, each finite side one row), the terminal inequalities (the last
  state's bounds and a terminal set) and a terminal equality ``g = x_N - x_ref``, with ``I_mu`` the
  penalty on the active rows (``c >= 0`` or ``lambda > 0``): a rollout ``scan``, a backward Riccati
  ``scan`` on local models from AD, and a line search that halves ``alpha`` until the ratio of actual
  to predicted decrease is in ``[line_search_lower_bound, line_search_upper_bound]``. The outer loop
  stops when the largest violation is below ``constraint_tolerance``, and otherwise updates
  ``lambda <- max(0, lambda + mu c)``, ``nu <- nu + mu g`` and ``mu <- penalty_scaling mu``.

  With ``projected_newton``, ALTRO's second phase follows an augmented Lagrangian stopped at
  ``projected_newton_tolerance``: the states and controls become free variables, and each projection
  factors the KKT matrix of the cost's Hessian and the active constraints with a sparse ``L D L'``
  and takes minimum-norm steps onto the linearized constraints. It needs the cost's Hessian as a
  diagonal: ``Quadratic`` costs with diagonal weights.

  The warm start, and the point a solve returns, is the controls, flat; ``Info.iter`` counts the
  iLQR iterations, and ``primal_residual`` is the largest violation."""

  name: ClassVar[str] = "ocp.altro"
  problem: ClassVar[type] = DiscreteOCP
  api: ClassVar[int] = METHOD_API
  label: ClassVar[str] = "altro"

  constraint_tolerance: float = 1e-6
  cost_tolerance_intermediate: float = 1e-4
  gradient_tolerance: float = 10.0
  gradient_tolerance_intermediate: float = 1.0
  penalty_initial: float = 1.0
  penalty_scaling: float = 10.0
  penalty_max: float = 1e8
  dual_max: float = 1e8
  iterations_outer: int = 30
  iterations_inner: int = 300
  iterations_linesearch: int = 20
  line_search_lower_bound: float = 1e-8
  line_search_upper_bound: float = 10.0
  expected_decrease_tolerance: float = 1e-10
  dj_counter_limit: int = 10
  bp_reg_increase_factor: float = 1.6
  bp_reg_min: float = 1e-8
  bp_reg_max: float = 1e8
  bp_reg_fp: float = 10.0
  projected_newton: bool = False
  projected_newton_tolerance: float = 1e-4
  n_steps: int = 2
  active_set_tolerance_pn: float = 1e-3
  rho_primal: float = 1e-8
  rho_dual: float = 1e-8
  r_threshold: float = 1.1
  max_refinements: int = 10

  def supports(self, problem: Any) -> Support:
    """Whether this method can solve ``problem``: the reasons it cannot, if any."""
    if not isinstance(problem, DiscreteOCP):
      return Support((f"{type(problem).__name__} is not a DiscreteOCP",))
    reasons = []
    if not problem.shooting:
      reasons.append("ALTRO takes a discrete map or multiple shooting, not a transcription with variables of its own")
    if problem.cost_rule == "integral" and problem.stage_cost is not None:
      reasons.append("ALTRO takes the running cost at the points, not integrated")
    if any(c.soft is not None for c in problem.constraints):
      reasons.append("ALTRO takes hard path constraints, not soft ones")
    if self.projected_newton and _diagonal_hessian(problem) is None:
      reasons.append("ALTRO's projected Newton phase takes Quadratic costs with diagonal weights")
    return Support(tuple(reasons))

  def warm_size(self, problem: DiscreteOCP) -> int:
    """The size of the warm start: the controls, flat."""
    return problem.N * problem.nu

  def solver(self, problem: DiscreteOCP, *, name: str) -> ConcreteFunction[Any, Any, Any, Any]:
    """The solve with ALTRO's own counters: ``(x0, *params, warm) -> (us, xs, cost, outer, inner,
    violation, projections, projection_steps)``, ``cost`` the objective without the augmented
    Lagrangian's terms."""
    return _Build(self, problem, name).function()

  def build(self, problem: DiscreteOCP, *, name: str) -> ConcreteFunction[Any, Any, Any, Any]:
    """The solver Function: ``(x0, *params, warm) -> (xs, us, point, info)``."""
    inner = self.solver(problem, name=f"{name}_solve")
    nx, nu, n = problem.nx, problem.nu, problem.N
    tolerance = self.projected_newton_tolerance if self.projected_newton else self.constraint_tolerance
    final = self.constraint_tolerance

    def body(x0: Expr, *rest: Expr) -> Any:
      us, xs, cost, _, iterations, violation, _, _ = inner(x0, *rest)
      met = less(violation, final if self.projected_newton else tolerance)
      status = where(isfinite(violation), where(met, float(Status.OK), float(Status.MAX_ITER)), float(Status.NUMERICS))
      info = Info(status=status, iter=cast(iterations, "float64"), objective=cost, primal_residual=violation)
      return xs, us, us.reshape((n * nu,)), info

    size = self.warm_size(problem)
    inputs = param_list(L("x0", nx), *(L(p.name, (problem.param_size(p),)) for p in problem.params), L("warm", size))
    outputs = G(L("xs", (n + 1, nx)), L("us", (n, nu)), L("point", size), Info.tree())
    return ConcreteFunction(name, body, inputs, outputs)

  def shift(self, problem: DiscreteOCP) -> ConcreteFunction[Any, Any, Any, Any]:
    """``point -> warm``: the controls moved up one stage, the last repeated (``sc.ocp.shift``)."""
    nu, size = problem.nu, self.warm_size(problem)

    def body(point: Expr) -> Expr:
      return concat([point[nu:], point[size - nu :]]) if problem.N > 1 else point

    return ConcreteFunction(f"{problem.name}_shift_altro", body, param_list(L("point", size)), L("warm", size))

  def initial_guess(self, problem: DiscreteOCP, x0: Any, u: Any = None) -> np.ndarray:
    """A first warm start (``sc.ocp.initial_guess``): every control ``u``, zeros by default."""
    u = np.zeros(problem.nu) if u is None else np.ravel(np.asarray(u, dtype=np.float64))
    return np.tile(u, problem.N)


def _pair(bounds: tuple[Any, Any] | None, size: int) -> tuple[np.ndarray, np.ndarray]:
  lo, hi = (None, None) if bounds is None else bounds
  side = lambda v, fill: np.full(size, fill) if v is None else np.broadcast_to(np.asarray(v, dtype=np.float64), (size,)).copy()  # noqa: E731
  return side(lo, -np.inf), side(hi, np.inf)


def _diagonal_hessian(problem: DiscreteOCP) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
  """The diagonals of the costs' Hessians (state, control per stage, terminal state), or ``None`` when
  a cost is not a ``Quadratic`` with diagonal weights."""
  stage, terminal = problem.stage_quadratic, problem.terminal_quadratic
  if stage is None or stage.R is None or (problem.terminal_cost is not None and terminal is None):
    return None
  mats = [np.atleast_2d(np.asarray(m, dtype=np.float64)) for m in (stage.Q, stage.R, *([terminal.Q] if terminal is not None else []))]
  if any(np.any(m - np.diag(np.diag(m))) for m in mats):
    return None
  scale = problem.dt if problem.continuous and problem.dt is not None else 1.0
  qn = 2.0 * np.diag(mats[2]) if terminal is not None else np.zeros(problem.nx)
  return 2.0 * scale * np.diag(mats[0]), 2.0 * scale * np.diag(mats[1]), qn


class _Build:
  """The generated ALTRO for one problem: its rows, masks and loops."""

  def __init__(self, options: ALTRO, problem: DiscreteOCP, name: str) -> None:
    self.o, self.p, self.tag = options, problem, name
    nx, nu = problem.nx, problem.nu
    self.nx, self.nu, self.K = nx, nu, problem.N
    self.step = step_map(problem)
    self.scale = problem.dt if problem.continuous and problem.dt is not None else 1.0
    self.sizes = [p.type.size for p in problem.params]
    self.flat_sizes = [problem.param_size(p) for p in problem.params]
    self.varying = [p.name in problem.varying for p in problem.params]
    u_lo, u_hi = _pair(problem.u_bounds, nu)
    x_lo, x_hi = _pair(problem.x_bounds, nx)
    self.u_rows = (np.flatnonzero(np.isfinite(u_hi)), u_hi, np.flatnonzero(np.isfinite(u_lo)), u_lo)
    self.x_rows = (np.flatnonzero(np.isfinite(x_hi)), x_hi, np.flatnonzero(np.isfinite(x_lo)), x_lo)
    n_u = self.u_rows[0].size + self.u_rows[2].size
    n_x = self.x_rows[0].size + self.x_rows[2].size
    self.path_rows = []
    for spec, fn in zip(problem.constraints, problem.paths, strict=True):
      rows = fn.outputs[0].size
      lo, hi = (np.broadcast_to(np.asarray(v, dtype=np.float64), (rows,)) if v is not None else None for v in (spec.lo, spec.hi))
      self.path_rows.append((fn, hi, lo))
    n_path = sum(fn.outputs[0].size * ((hi is not None) + (lo is not None)) for fn, hi, lo in self.path_rows)
    self.n_rows = n_u + n_x + n_path
    # Without a row a stage keeps one, masked to zero, so every slice of the multipliers has a size.
    self.nc = max(self.n_rows, 1)
    masks = np.ones((self.K, self.nc)) if self.n_rows else np.zeros((self.K, 1))
    masks[0, n_u : n_u + n_x] = 0.0  # the initial state is data: the first stage bounds the controls only
    self.masks = masks
    self.goal = isinstance(problem.terminal, TerminalEquality)
    # The last state's inequalities: its bounds (implied by a terminal equality, so left to it) and a set.
    self.terminal_bounds = not self.goal and n_x > 0
    self.terminal_set = problem.terminal is not None and not self.goal
    probe = self.terminal_ineq(Expr.sym("x", nx), [Expr.sym(f"p{i}", s) for i, s in enumerate(self.flat_sizes)])
    self.nt = 0 if probe is None else probe.size
    self.diagonal = _diagonal_hessian(problem)

  # -- the problem's pieces --------------------------------------------------------------------------

  def stage(self, x: Expr, u: Expr, values: tuple[Expr, ...]) -> Expr:
    fn = self.p.stage_cost
    return fn(x, u, *values) * self.scale if fn is not None else Expr.const(0.0) * x[0]

  def ends(self, flat: list[Expr]) -> list[Expr]:
    out = []
    for v, s, vary, p in zip(flat, self.sizes, self.varying, self.p.params, strict=True):
      out.append((v[self.K * s :] if vary else v).reshape(p.type.shape))
    return out

  def terminal(self, x: Expr, flat: list[Expr]) -> Expr:
    fn = self.p.terminal_cost
    return fn(x, *self.ends(flat)) if fn is not None else Expr.const(0.0) * x[0]

  def ineq(self, x: Expr, u: Expr, values: tuple[Expr, ...]) -> Expr:
    """The stage's rows ``c <= 0``: control bounds (upper, then lower), state bounds, paths."""
    rows = []
    for vec, (up, hi, down, lo) in ((u, self.u_rows), (x, self.x_rows)):
      rows += [vec[int(i)] - float(hi[i]) for i in up]
      rows += [float(lo[i]) - vec[int(i)] for i in down]
    parts = [concat([r.reshape((1,)) for r in rows])] if rows else []
    for fn, hi, lo in self.path_rows:
      g = fn(x, u, *values)
      if hi is not None:
        parts.append(g - Expr.const(np.asarray(hi)))
      if lo is not None:
        parts.append(Expr.const(np.asarray(lo)) - g)
    return concat(parts) if parts else Expr.const(np.zeros(1)) * x[0]

  def goal_ref(self, flat: list[Expr]) -> Expr:
    assert isinstance(self.p.terminal, TerminalEquality)
    ref = self.p.terminal.x_ref
    if isinstance(ref, str):
      i = [p.name for p in self.p.params].index(ref)
      return flat[i][self.K * self.nx :] if ref in self.p.varying else flat[i]
    return Expr.const(np.zeros(self.nx) if ref is None else np.asarray(ref, dtype=np.float64))

  def terminal_ineq(self, x: Expr, flat: list[Expr]) -> Expr | None:
    """The last state's rows ``c <= 0``: its bounds, then a terminal set's constraints."""
    rows = []
    if self.terminal_bounds:
      up, hi, down, lo = self.x_rows
      rows += [x[int(i)] - float(hi[i]) for i in up] + [float(lo[i]) - x[int(i)] for i in down]
    parts = [concat([r.reshape((1,)) for r in rows])] if rows else []
    if self.terminal_set:
      for g, lo, hi in self.p.terminal.constraints(x):
        g = g.reshape((g.size,))
        if hi is not None:
          parts.append(g - hi)
        if lo is not None:
          parts.append(lo - g)
    return concat(parts) if parts else None

  # -- the generated solve ---------------------------------------------------------------------------

  def function(self) -> ConcreteFunction[Any, Any, Any, Any]:
    o, p, tag = self.o, self.p, self.tag
    nx, nu, nc, K, nt = self.nx, self.nu, self.nc, self.K, self.nt
    al_tolerance = o.projected_newton_tolerance if o.projected_newton else o.constraint_tolerance
    step = self.step
    MASKS = Expr.const(self.masks.reshape(-1))
    Gn = nu + nu * nx  # feedforward and feedback gains per stage
    Lc = K * nc
    ng = nx if self.goal else 0
    slots = [L(q.name, q.type) for q in p.params]

    def split(pv: Expr) -> list[Expr]:
      out, cut = [], 0
      for size in self.flat_sizes:
        out.append(pv[cut : cut + size])
        cut += size
      return out

    def forward(values: list[Expr]) -> list[tuple[Expr, int, int]]:
      return [(v, 0, s if vary else 0) for v, s, vary in zip(values, self.sizes, self.varying, strict=True)]

    def backward(values: list[Expr]) -> list[tuple[Expr, int, int]]:
      return [(v, (K - 1) * s, -s) if vary else (v, 0, 0) for v, s, vary in zip(values, self.sizes, self.varying, strict=True)]

    def active_penalty(c: Expr, lam: Expr, mu: Expr) -> Expr:
      return where(logical_or(greater_equal(c, 0.0), greater(lam, 0.0)), mu, 0.0)

    def al_stage(x: Expr, u: Expr, lam: Expr, mask: Expr, mu: Expr, values: tuple[Expr, ...]) -> Expr:
      c = mask * self.ineq(x, u, values)
      return self.stage(x, u, values) + (lam * c).sum() + 0.5 * (active_penalty(c, lam, mu) * c * c).sum()

    def al_terminal(x: Expr, duals: Expr, mu: Expr, flat: list[Expr]) -> Expr:
      value = self.terminal(x, flat)
      if self.goal:
        g = x - self.goal_ref(flat)
        value = value + (duals[:ng] * g).sum() + 0.5 * mu * (g * g).sum()
      if nt:
        c = self.terminal_ineq(x, flat)
        assert c is not None
        lam_t = duals[ng : ng + nt]
        value = value + (lam_t * c).sum() + 0.5 * (active_penalty(c, lam_t, mu) * c * c).sum()
      return value

    def fn(name: str, body: Any, *leading: Any, output: Any) -> ConcreteFunction[Any, Any, Any, Any]:
      return ConcreteFunction(f"{tag}_{name}", body, param_list(*leading, *slots), output)

    # Rollout under the nominal inputs: states x_0 .. x_{K-1} stacked, augmented stage costs.
    rollout_step = fn(
      "rollout_step",
      lambda x, u, lam, mask, mu, *values: (step(x, u, *values), x, al_stage(x, u, lam, mask, mu[0], values).reshape((1,))),
      L("x", nx),
      L("u", nu),
      L("lam", nc),
      L("mask", nc),
      L("mu", 1),
      output=G(L("xnext", nx), L("state", nx), L("cost", 1)),
    )

    def backward_body(value: Expr, x: Expr, u: Expr, lam: Expr, mask: Expr, mr: Expr, *values: Expr) -> tuple[Expr, Expr, Expr]:
      mu, rho = mr[0], mr[1]
      vx, vxx = value[:nx], value[nx:].reshape((nx, nx))
      x_next, cost = step(x, u, *values), al_stage(x, u, lam, mask, mu, values)
      fx, fu = jacobian(x_next, x), jacobian(x_next, u)
      lx, lu = gradient(cost, x), gradient(cost, u)
      qx, qu = lx + fx.T @ vx, lu + fu.T @ vx
      qxx = hessian(cost, x) + fx.T @ vxx @ fx
      quu = hessian(cost, u) + fu.T @ vxx @ fu
      qux = jacobian(lu, x) + fu.T @ vxx @ fx
      chol = cholesky(quu + Expr.const(np.eye(nu)) * rho)
      k = -cho_solve(chol, qu)
      gain = -cho_solve(chol, qux)
      vx_prev = qx + gain.T @ (quu @ k) + gain.T @ qu + qux.T @ k
      vxx_prev = qxx + gain.T @ quu @ gain + gain.T @ qux + qux.T @ gain
      vxx_prev = 0.5 * (vxx_prev + vxx_prev.T)
      dv = concat([(k @ qu).reshape((1,)), (0.5 * (k @ (quu @ k))).reshape((1,))])
      return concat([vx_prev, vxx_prev.reshape((nx * nx,))]), concat([k, gain.reshape((nu * nx,))]), dv

    backward_step = fn(
      "backward_step",
      backward_body,
      L("value", nx + nx * nx),
      L("x", nx),
      L("u", nu),
      L("lam", nc),
      L("mask", nc),
      L("mr", 2),
      output=G(L("value_prev", nx + nx * nx), L("gains", Gn), L("dv", 2)),
    )

    def policy_body(x: Expr, x_nom: Expr, u_nom: Expr, gains: Expr, lam: Expr, mask: Expr, am: Expr, *values: Expr) -> tuple[Expr, Expr, Expr]:
      u = u_nom + am[0] * gains[:nu] + gains[nu:].reshape((nu, nx)) @ (x - x_nom)
      return step(x, u, *values), u, al_stage(x, u, lam, mask, am[1], values).reshape((1,))

    policy_step = fn(
      "policy_step",
      policy_body,
      L("x", nx),
      L("x_nom", nx),
      L("u_nom", nu),
      L("gains", Gn),
      L("lam", nc),
      L("mask", nc),
      L("am", 2),
      output=G(L("xnext", nx), L("u", nu), L("cost", 1)),
    )

    # Line search. Carry: (state, cost, alpha, us) with state 0 searching, 1 accepted, 2 no step.
    @function(name=f"{tag}_line_search")
    def line_search_step(
      carry: Expr, trial: Expr, x0: Expr, pv: Expr, xs: Expr, us: Expr, gains: Expr, j_dv: Expr, lam: Expr, duals: Expr, mu: Expr
    ) -> Expr:
      values = split(pv)
      alpha = Expr.const(0.5) ** trial.cast("float64")
      specs = [
        (xs, 0, nx),
        (us, 0, nu),
        (gains, (K - 1) * Gn, -Gn),
        (lam, 0, nc),
        (MASKS, 0, nc),
        (concat([alpha.reshape((1,)), mu]), 0, 0),
        *forward(values),
      ]
      x_final, u_new, costs = scan(policy_step, x0, specs, length=K)
      cost = costs.sum() + al_terminal(x_final, duals, mu[0], values)
      expected = -alpha * (j_dv[1] + alpha * j_dv[2])
      tiny = logical_and(greater(expected, 0.0), less(expected, o.expected_decrease_tolerance))
      z = where(greater(expected, 0.0), (j_dv[0] - cost) / expected, -1.0)
      accepted = logical_and(greater_equal(z, o.line_search_lower_bound), less_equal(z, o.line_search_upper_bound))
      state = where(tiny, 2.0, where(accepted, 1.0, 0.0))
      return concat([state.reshape((1,)), cost.reshape((1,)), alpha.reshape((1,)), u_new])

    @function(name=f"{tag}_searching")
    def searching(carry: Expr, x0: Expr, pv: Expr, xs: Expr, us: Expr, gains: Expr, j_dv: Expr, lam: Expr, duals: Expr, mu: Expr) -> Expr:
      return less(carry[0], 0.5)

    # Altro's gradient measure, summed here and averaged by the caller: max_i |d_i| / (|u_i| + 1) per stage.
    gradient_step = ConcreteFunction(
      f"{tag}_gradient_step",
      lambda acc, u, gains: acc + (gains[:nu].abs() / (u.abs() + 1.0)).max(),
      param_list(L("acc", 1), L("u", nu), L("gains", Gn)),
      L("next", 1),
    )

    # One iLQR iteration. Carry: (us, cost, rho, drho, done, iterations, iterations without a change).
    @function(name=f"{tag}_ilqr_iteration")
    def ilqr_iteration(carry: Expr, x0: Expr, pv: Expr, lam: Expr, duals: Expr, mu: Expr, grad_tol: Expr) -> Expr:
      values = split(pv)
      us, rho, drho, count, unchanged = carry[: K * nu], carry[K * nu + 1], carry[K * nu + 2], carry[K * nu + 4], carry[K * nu + 5]
      x_final, xs, costs = scan(rollout_step, x0, [(us, 0, nu), (lam, 0, nc), (MASKS, 0, nc), (mu, 0, 0), *forward(values)], length=K)
      terminal = al_terminal(x_final, duals, mu[0], values)
      j_prev = costs.sum() + terminal
      value = concat([gradient(terminal, x_final), hessian(terminal, x_final).reshape((nx * nx,))])
      mr = concat([mu, rho.reshape((1,))])
      specs = [(xs, (K - 1) * nx, -nx), (us, (K - 1) * nu, -nu), (lam, (K - 1) * nc, -nc), (MASKS, (K - 1) * nc, -nc), (mr, 0, 0), *backward(values)]
      _, gains, dv = scan(backward_step, value, specs, length=K)
      dv = dv.reshape((K, 2))
      j_dv = concat([j_prev.reshape((1,)), dv[:, 0].sum().reshape((1,)), dv[:, 1].sum().reshape((1,))])
      # A failed factorization (NaN gains) skips the line search, as a negligible expected decrease does.
      start = concat([where(isfinite(j_dv[1] + j_dv[2]), 0.0, 2.0).reshape((1,)), Expr.const(np.zeros(2)), us])
      ls, _ = while_loop(
        searching, line_search_step, start, max_iter=o.iterations_linesearch, index=True, params=(x0, pv, xs, us, gains, j_dv, lam, duals, mu)
      )
      accepted = logical_and(greater(ls[0], 0.5), less(ls[0], 1.5))
      us_next = where(accepted, ls[3:], us)
      j_next = where(accepted, ls[1], j_prev)
      # Regularization as Altro's: decreased after every backward pass, increased when no step is taken,
      # and bumped by bp_reg_fp when the line search runs out of trials.
      f = o.bp_reg_increase_factor
      drho_down = minimum(drho / f, 1.0 / f)
      rho_down = maximum(rho * drho_down, o.bp_reg_min)
      drho_up = maximum(drho_down * f, f)
      rho_up = maximum(rho_down * drho_up, o.bp_reg_min) + where(less(ls[0], 0.5), o.bp_reg_fp, 0.0)
      rho_next, drho_next = where(accepted, rho_down, rho_up), where(accepted, drho_down, drho_up)
      (grad_sum,) = scan(gradient_step, Expr.const(np.zeros(1)), [(us_next, 0, nu), (gains, (K - 1) * Gn, -Gn)], length=K)
      grad = grad_sum[0] / K
      dj = j_prev - j_next
      converged = logical_and(
        logical_and(accepted, greater_equal(dj, 0.0)), logical_and(less(dj, o.cost_tolerance_intermediate), less(grad, grad_tol[0]))
      )
      # Altro's dJ_zero_counter: a run of iterations that change nothing (no step, or a negligible
      # expected decrease) ends the inner solve.
      unchanged_next = where(equal(dj, 0.0), unchanged + 1.0, 0.0)
      done = logical_or(logical_or(converged, greater(rho_next, o.bp_reg_max)), greater(unchanged_next, float(o.dj_counter_limit)))
      return concat(
        [
          us_next,
          j_next.reshape((1,)),
          rho_next.reshape((1,)),
          drho_next.reshape((1,)),
          cast(done, "float64").reshape((1,)),
          (count + 1.0).reshape((1,)),
          unchanged_next.reshape((1,)),
        ]
      )

    @function(name=f"{tag}_ilqr_running")
    def ilqr_running(carry: Expr, x0: Expr, pv: Expr, lam: Expr, duals: Expr, mu: Expr, grad_tol: Expr) -> Expr:
      return less(carry[K * nu + 3], 0.5)

    constraint_step = fn(
      "constraint_step",
      lambda x, u, mask, *values: (step(x, u, *values), mask * self.ineq(x, u, values), self.stage(x, u, values).reshape((1,))),
      L("x", nx),
      L("u", nu),
      L("mask", nc),
      output=G(L("xnext", nx), L("c", nc), L("cost", 1)),
    )

    nd = max(ng + nt, 1)  # terminal duals: the goal's, then the terminal inequalities'; one unused slot without
    base = K * nu + Lc + nd

    def violation_of(c_all: Expr, x_final: Expr, values: list[Expr]) -> Expr:
      v = maximum(c_all.max(), 0.0)
      if self.goal:
        v = maximum(v, (x_final - self.goal_ref(values)).abs().max())
      if nt:
        c_t = self.terminal_ineq(x_final, values)
        assert c_t is not None
        v = maximum(v, maximum(c_t.max(), 0.0))
      return v

    # One outer iteration. Carry: (us, lambda, terminal duals, mu, cost, converged, iLQR iterations, violation).
    @function(name=f"{tag}_al_iteration")
    def al_iteration(carry: Expr, outer: Expr, x0: Expr, pv: Expr) -> Expr:
      values = split(pv)
      us, lam, duals, mu, inner = carry[: K * nu], carry[K * nu : K * nu + Lc], carry[K * nu + Lc : base], carry[base], carry[base + 3]
      last = greater_equal(outer.cast("float64"), o.iterations_outer - 1.0)
      grad_tol = where(last, o.gradient_tolerance, o.gradient_tolerance_intermediate).reshape((1,))
      # Every inner solve starts from zero regularization, as Altro resets its iLQR solver.
      start = concat([us, Expr.const(np.array([np.inf, 0.0, 0.0, 0.0, 0.0, 0.0]))])
      ic, _ = while_loop(ilqr_running, ilqr_iteration, start, max_iter=o.iterations_inner, params=(x0, pv, lam, duals, mu.reshape((1,)), grad_tol))
      us_new = ic[: K * nu]
      x_final, c_all, costs = scan(constraint_step, x0, [(us_new, 0, nu), (MASKS, 0, nc), *forward(values)], length=K)
      violation = violation_of(c_all, x_final, values)
      converged = less(violation, al_tolerance)
      lam_new = minimum(maximum(lam + mu * c_all, 0.0), o.dual_max)
      new_duals = []
      if self.goal:
        new_duals.append(minimum(maximum(duals[:ng] + mu * (x_final - self.goal_ref(values)), -o.dual_max), o.dual_max))
      if nt:
        c_t = self.terminal_ineq(x_final, values)
        assert c_t is not None
        new_duals.append(minimum(maximum(duals[ng : ng + nt] + mu * c_t, 0.0), o.dual_max))
      duals_new = concat(new_duals) if new_duals else duals
      mu_new = minimum(mu * o.penalty_scaling, o.penalty_max)
      return concat(
        [
          us_new,
          where(converged, lam, lam_new),
          where(converged, duals, duals_new),
          where(converged, mu, mu_new).reshape((1,)),
          (costs.sum() + self.terminal(x_final, values)).reshape((1,)),
          cast(converged, "float64").reshape((1,)),
          (inner + ic[K * nu + 4]).reshape((1,)),
          violation.reshape((1,)),
        ]
      )

    @function(name=f"{tag}_al_running")
    def al_running(carry: Expr, x0: Expr, pv: Expr) -> Expr:
      return less(carry[base + 2], 0.5)

    def body(x0: Expr, *rest: Expr) -> Any:
      *raw, warm = rest
      flat = [v.reshape((v.size,)) for v in raw]
      pv = concat(flat) if flat else Expr.const(np.zeros(1))
      start = concat([warm, Expr.const(np.zeros(Lc + nd)), Expr.const(np.array([o.penalty_initial, np.inf, 0.0, 0.0, np.inf]))])
      carry, outer = while_loop(al_running, al_iteration, start, max_iter=o.iterations_outer, index=True, params=(x0, pv))
      us = carry[: K * nu]
      specs = [(us, 0, nu), (Expr.const(np.zeros(Lc)), 0, nc), (MASKS, 0, nc), (Expr.const(np.zeros(1)), 0, 0), *forward(flat)]
      x_final, xs, _ = scan(rollout_step, x0, specs, length=K)
      if not o.projected_newton:
        zero = Expr.const(0.0)
        counts = (outer.cast("float64"), carry[base + 3])
        return us.reshape((K, nu)), concat([xs, x_final]).reshape((K + 1, nx)), carry[base + 1], *counts, carry[base + 4], zero, zero
      return self.projected(x0, flat, xs, x_final, us, outer.cast("float64"), carry[base + 3])

    inputs = param_list(L("x0", nx), *(L(q.name, (p.param_size(q),)) for q in p.params), L("warm", K * nu))
    outputs = G(
      L("us", (K, nu)),
      L("xs", (K + 1, nx)),
      L("cost", ()),
      L("outer", ()),
      L("inner", ()),
      L("violation", ()),
      L("projections", ()),
      L("projection_steps", ()),
    )
    return ConcreteFunction(f"{tag}", body, inputs, outputs)

  def projected(self, x0: Expr, flat: list[Expr], xs: Expr, x_final: Expr, us: Expr, outer: Expr, inner: Expr) -> tuple[Expr, ...]:
    """ALTRO's projected Newton phase. The primal vector ``z`` interleaves ``(x_k, u_k)`` for ``k < K``,
    then ``x_K``; the rows are the initial state, each stage's dynamics defect and masked inequalities,
    the goal, and the terminal inequalities."""
    o, p, tag = self.o, self.p, self.tag
    nx, nu, nc, K, nt = self.nx, self.nu, self.nc, self.K, self.nt
    assert self.diagonal is not None
    hq, hr, hn = self.diagonal
    step, masks_np = self.step, self.masks
    MASKS = Expr.const(masks_np.reshape(-1))
    NZ = nx + nu
    NP = K * NZ + nx
    ng = nx if self.goal else 0
    HDIAG = Expr.const(np.concatenate([*[np.concatenate([hq, hr])] * K, hn]) + o.rho_primal)
    eq_np = np.concatenate([np.ones(nx), *[np.concatenate([np.ones(nx), np.zeros(nc)]) for _ in range(K)], np.ones(ng), np.zeros(nt)])
    rows_np = np.concatenate([np.ones(nx), *[np.concatenate([np.ones(nx), masks_np[k]]) for k in range(K)], np.ones(ng), np.ones(nt)])
    EQ, ROWS = Expr.const(eq_np), Expr.const(rows_np)
    slots = [L(q.name, q.type) for q in p.params]
    sizes, varying = self.sizes, self.varying

    pn_stage = ConcreteFunction(
      f"{tag}_pn_stage",
      lambda xu, x_next, mask, *values: concat([step(xu[:nx], xu[nx:], *values) - x_next, mask * self.ineq(xu[:nx], xu[nx:], values)]),
      param_list(L("xu", NZ), L("x_next", nx), L("mask", nc), *slots),
      L("rows", nx + nc),
    )

    def pn_constraints(z: Expr, x0: Expr, values: list[Expr]) -> Expr:
      specs = [(z, 0, NZ), (z, NZ, NZ), (MASKS, 0, nc), *((v, 0, s if vary else 0) for v, s, vary in zip(values, sizes, varying, strict=True))]
      parts = [z[:nx] - x0, vmap(pn_stage, K, specs)]
      if self.goal:
        parts.append(z[K * NZ :] - self.goal_ref(values))
      if nt:
        c_t = self.terminal_ineq(z[K * NZ :], values)
        assert c_t is not None
        parts.append(c_t)
      return concat(parts)

    def pn_violation(c: Expr) -> Expr:
      return maximum((c.abs() * EQ).max(), (maximum(c, 0.0) * (1.0 - EQ)).max())

    def split(pv: Expr) -> list[Expr]:
      out, cut = [], 0
      for size in self.flat_sizes:
        out.append(pv[cut : cut + size])
        cut += size
      return out

    @function(name=f"{tag}_pn_line_search")
    def pn_line_search(carry: Expr, trial: Expr, z0: Expr, step_dir: Expr, v0: Expr, x0: Expr, pv: Expr) -> Expr:
      zt = z0 + (Expr.const(0.5) ** trial.cast("float64")) * step_dir
      v = pn_violation(pn_constraints(zt, x0, split(pv)))
      return concat([cast(less(v, v0), "float64").reshape((1,)), v.reshape((1,)), zt])

    @function(name=f"{tag}_pn_searching")
    def pn_searching(carry: Expr, z0: Expr, step_dir: Expr, v0: Expr, x0: Expr, pv: Expr) -> Expr:
      return less(carry[0], 0.5)

    def pn_kkt(z: Expr, x0: Expr, values: list[Expr]) -> tuple[SparseMatrix, Expr]:
      c = pn_constraints(z, x0, values)
      active = EQ + (1.0 - EQ) * ROWS * cast(greater(c, -o.active_set_tolerance_pn), "float64")
      jac = SparseMatrix.from_sparse_jacobian(sparse_jacobian(c, z))
      dual = -(o.rho_dual * active + (1.0 - active))  # an inactive row reads lambda_i = 0
      kkt = SparseMatrix.block([[SparseMatrix.diag(HDIAG), None], [jac.scale_rows(active), SparseMatrix.diag(dual)]])
      return kkt, active

    # The projection's factorization, whose analysis the refinement steps solve with (set while tracing it).
    pn_ldl: dict[str, SparseLDL] = {}

    # One refinement: a step from the current point with the projection's factor, backtracked on the violation.
    @function(name=f"{tag}_pn_refinement")
    def pn_refinement(carry: Expr, factor: Expr, active: Expr, x0: Expr, pv: Expr) -> Expr:
      z, v_prev, steps = carry[:NP], carry[NP], carry[NP + 2]
      c = pn_constraints(z, x0, split(pv))
      v0 = pn_violation(c)
      direction = pn_ldl["kkt"].solve_with(factor, concat([Expr.const(np.zeros(NP)), -active * c]))[:NP]
      ls, _ = while_loop(
        pn_searching, pn_line_search, concat([Expr.const(np.zeros(2)), z]), max_iter=10, index=True, params=(z, direction, v0, x0, pv)
      )
      ok = greater(ls[0], 0.5)
      v = where(ok, ls[1], np.nan)
      rate = v.log() / v_prev.log()
      stop = logical_or(less(v, o.constraint_tolerance), less(rate, o.r_threshold))
      return concat([where(ok, ls[2:], z), v.reshape((1,)), cast(stop, "float64").reshape((1,)), (steps + 1.0).reshape((1,))])

    @function(name=f"{tag}_pn_refining")
    def pn_refining(carry: Expr, factor: Expr, active: Expr, x0: Expr, pv: Expr) -> Expr:
      return less(carry[NP + 1], 0.5)

    # One projection: linearize, factor, refine. Carry: (z, violation, refinement steps).
    @function(name=f"{tag}_pn_projection")
    def pn_projection(carry: Expr, x0: Expr, pv: Expr) -> Expr:
      z, steps = carry[:NP], carry[NP + 1]
      values = split(pv)
      kkt, active = pn_kkt(z, x0, values)
      pn_ldl["kkt"] = SparseLDL(kkt, name=f"{tag}_pn_kkt")
      factor = pn_ldl["kkt"].values
      start = concat([z, carry[NP].reshape((1,)), Expr.const(np.zeros(2))])
      rc, _ = while_loop(pn_refining, pn_refinement, start, max_iter=o.max_refinements - 1, params=(factor, active, x0, pv))
      z_new = rc[:NP]
      return concat([z_new, pn_violation(pn_constraints(z_new, x0, values)).reshape((1,)), (steps + rc[NP + 2]).reshape((1,))])

    @function(name=f"{tag}_pn_running")
    def pn_running(carry: Expr, x0: Expr, pv: Expr) -> Expr:
      return greater(carry[NP], o.constraint_tolerance)

    pv = concat(flat) if flat else Expr.const(np.zeros(1))
    z = concat([concat([xs.reshape((K, nx)), us.reshape((K, nu))], axis=1).reshape((K * NZ,)), x_final])
    start = concat([z, pn_violation(pn_constraints(z, x0, flat)).reshape((1,)), Expr.const(np.zeros(1))])
    pc, projections = while_loop(pn_running, pn_projection, start, max_iter=o.n_steps + 1, params=(x0, pv))
    z = pc[:NP]
    xu = z[: K * NZ].reshape((K, NZ))
    xs_k, us = xu[:, :nx].reshape((K * nx,)), xu[:, nx:].reshape((K * nu,))
    specs = [(xs_k, 0, nx), (us, 0, nu), *((v, 0, s if vary else 0) for v, s, vary in zip(flat, sizes, varying, strict=True))]
    stage_fn = p.stage_cost
    cost = (vmap(stage_fn, K, specs).sum() * self.scale if stage_fn is not None else Expr.const(0.0)) + self.terminal(z[K * NZ :], flat)
    violation = pn_violation(pn_constraints(z, x0, flat))
    xs_all = concat([xs_k, z[K * NZ :]]).reshape((K + 1, nx))
    return us.reshape((K, nu)), xs_all, cost, outer, inner, violation, projections.cast("float64"), pc[NP + 1]


__all__ = ["ALTRO"]
