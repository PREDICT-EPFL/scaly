"""Augmented-Lagrangian iLQR written in Scaly: ALTRO's first phase as one generated C function.

For an OCP with discrete dynamics `x_{k+1} = F(x_k, u_k)`, the stage cost
`dt (1/2 e'Qe + 1/2 u'Ru)` with `e = x - x_f`, the terminal cost `1/2 e_N' Q_f e_N`, stage inequalities
`c_k(x_k, u_k) <= 0` and the goal `x_N = x_f`, the solver is the augmented Lagrangian iLQR of ALTRO
(Howell, Jackson and Manchester, IROS 2019), following the logic of Altro.jl 0.5
(`augmented_lagrangian/al_solve.jl`, `ilqr/ilqr_solve.jl`, `ilqr/forwardpass.jl`):

- the inner solver is iLQR on the augmented Lagrangian
  `J + sum (lambda'c + 1/2 c' I_mu c) + nu'g + mu/2 |g|^2`, `g = x_N - x_f`, with `I_mu` the penalty on
  the active rows (`c >= 0` or `lambda > 0`): a rollout `scan`, a backward Riccati `scan` with the local
  models from AD, and a line search `while_loop` that halves `alpha` until the ratio of actual to
  predicted decrease is in `[1e-8, 10]`. It stops when the decrease is below `cost_tolerance` and the
  mean relative feedforward step below `gradient_tolerance`;
- the outer loop stops as soon as the largest violation is below `constraint_tolerance` and otherwise
  updates `lambda <- max(0, lambda + mu c)`, `nu <- nu + mu g`, `mu <- 10 mu`;
- with `projected_newton`, ALTRO's second phase (`direct/pn_solve.jl`) follows an augmented Lagrangian
  stopped at the looser `projected_newton_tolerance`: states and controls become free variables, and
  each projection factors the KKT matrix `[H, D_a'; D_a, -reg]` of the cost Hessian and the active
  constraints (initial state, dynamics, inequalities within `active_set_tolerance_pn` of binding, goal)
  with a sparse `L D L'` and takes minimum-norm steps onto the linearized constraints, reusing the
  factor while the violation falls fast enough, each step backtracked on the largest violation.

Every loop is a Scaly loop, so `solver(ocp)` is one Function whose generated C holds the complete
solve, with no allocation and nothing to call back.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

import scaly as sc
from scaly import integrators as si
from scaly import linalg


@dataclass
class OCP:
  """What a problem supplies. `ineq(x, u)` returns the stage inequalities `c <= 0`, and `masks[k]` says
  which of its rows apply at stage `k` (0 or 1), for the `n_knots - 1` stages that carry a control."""

  name: str
  nx: int
  nu: int
  n_knots: int  # N: states x_0 .. x_{N-1}, controls u_0 .. u_{N-2}
  dt: float
  dynamics: Callable[[sc.Expr, sc.Expr], sc.Expr]
  q: np.ndarray  # diagonals; the stage cost multiplies them by dt
  r: np.ndarray
  qf: np.ndarray
  x0: np.ndarray
  xf: np.ndarray
  u0: np.ndarray  # (N - 1, nu)
  ineq: Callable[[sc.Expr, sc.Expr], sc.Expr]
  nc: int
  masks: np.ndarray  # (N - 1, nc)
  integrator: str = "rk3"


@dataclass
class Options:
  """Altro.jl 0.5's defaults."""

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
  bp_reg_increase_factor: float = 1.6
  bp_reg_min: float = 1e-8
  bp_reg_max: float = 1e8
  bp_reg_fp: float = 10.0
  projected_newton: bool = False
  projected_newton_tolerance: float = 1e-4  # the augmented Lagrangian's tolerance when the projection follows
  n_steps: int = 2
  active_set_tolerance_pn: float = 1e-3
  rho_primal: float = 1e-8
  rho_dual: float = 1e-8
  r_threshold: float = 1.1
  max_refinements: int = 10


def solver(ocp: OCP, opts: Options | None = None) -> sc.Function:
  """`solve(x0) -> (us, xs, cost, outer iterations, iLQR iterations, max violation, projections, projection steps)`,
  `cost` the objective without the augmented-Lagrangian terms."""
  o = opts or Options()
  al_tolerance = o.projected_newton_tolerance if o.projected_newton else o.constraint_tolerance
  nx, nu, nc, K, dt, tag = ocp.nx, ocp.nu, ocp.nc, ocp.n_knots - 1, ocp.dt, ocp.name
  Q, R, QF = (sc.const(np.diag(np.asarray(a, dtype=np.float64))) for a in (ocp.q, ocp.r, ocp.qf))
  XF = sc.const(np.asarray(ocp.xf, dtype=np.float64))
  MASKS = sc.const(np.asarray(ocp.masks, dtype=np.float64).reshape(-1))
  G = nu + nu * nx  # feedforward and feedback gains per stage
  L = K * nc

  @sc.function(sc.L("x", nx), sc.L("u", nu), output="xdot", name=f"{tag}_model")
  def model(x, u):
    return ocp.dynamics(x, u)

  step = si.explicit(model, ocp.integrator, dt=dt, name=f"{tag}_{ocp.integrator}")

  def stage_cost(x, u):
    e = x - XF
    return dt * (0.5 * (e @ Q @ e) + 0.5 * (u @ R @ u))

  def terminal_cost(x):
    e = x - XF
    return 0.5 * (e @ QF @ e)

  def al_stage(x, u, lam, mask, mu):
    c = mask * ocp.ineq(x, u)
    penalty = sc.where(sc.logical_or(sc.greater_equal(c, 0.0), sc.greater(lam, 0.0)), mu, 0.0)
    return stage_cost(x, u) + (lam * c).sum() + 0.5 * (penalty * c * c).sum()

  def al_terminal(x, nu_goal, mu):
    g = x - XF
    return terminal_cost(x) + (nu_goal * g).sum() + 0.5 * mu * (g * g).sum()

  # Rollout under the nominal inputs: states x_0 .. x_{K-1} stacked, AL stage costs. Broadcast: mu.
  @sc.function(sc.L("x", nx), sc.L("u", nu), sc.L("lam", nc), sc.L("mask", nc), sc.L("mu", 1), name=f"{tag}_rollout_step")
  def rollout_step(x, u, lam, mask, mu):
    return step(x, u), x, al_stage(x, u, lam, mask, mu[0]).reshape((1,))

  # Backward pass. Carry: V_x, V_xx. Sliced backwards: x_k, u_k, lambda_k, mask_k. Broadcast: [mu, rho].
  @sc.function(
    sc.L("value", nx + nx * nx), sc.L("x", nx), sc.L("u", nu), sc.L("lam", nc), sc.L("mask", nc), sc.L("mr", 2), name=f"{tag}_backward_step"
  )
  def backward_step(value, x, u, lam, mask, mr):
    mu, rho = mr[0], mr[1]
    vx, vxx = value[:nx], value[nx:].reshape((nx, nx))
    x_next, cost = step(x, u), al_stage(x, u, lam, mask, mu)
    fx, fu = sc.jacobian(x_next, x), sc.jacobian(x_next, u)
    lx, lu = sc.gradient(cost, x), sc.gradient(cost, u)
    qx, qu = lx + fx.T @ vx, lu + fu.T @ vx
    qxx = sc.hessian(cost, x) + fx.T @ vxx @ fx
    quu = sc.hessian(cost, u) + fu.T @ vxx @ fu
    qux = sc.jacobian(lu, x) + fu.T @ vxx @ fx
    chol = linalg.cholesky(quu + sc.const(np.eye(nu)) * rho)
    k = -linalg.cho_solve(chol, qu)
    gain = -linalg.cho_solve(chol, qux)
    vx_prev = qx + gain.T @ (quu @ k) + gain.T @ qu + qux.T @ k
    vxx_prev = qxx + gain.T @ quu @ gain + gain.T @ qux + qux.T @ gain
    vxx_prev = 0.5 * (vxx_prev + vxx_prev.T)
    dv = sc.stack([k @ qu, 0.5 * (k @ (quu @ k))])
    return sc.concat([vx_prev, vxx_prev.reshape((nx * nx,))]), sc.concat([k, gain.reshape((nu * nx,))]), dv

  # Forward pass under the policy. Carry: x. Sliced: x_k, u_k, gains (stored backwards), lambda_k, mask_k. Broadcast: [alpha, mu].
  @sc.function(
    sc.L("x", nx), sc.L("x_nom", nx), sc.L("u_nom", nu), sc.L("gains", G), sc.L("lam", nc), sc.L("mask", nc), sc.L("am", 2), name=f"{tag}_policy_step"
  )
  def policy_step(x, x_nom, u_nom, gains, lam, mask, am):
    u = u_nom + am[0] * gains[:nu] + gains[nu:].reshape((nu, nx)) @ (x - x_nom)
    return step(x, u), u, al_stage(x, u, lam, mask, am[1]).reshape((1,))

  # Line search. Carry: (state, cost, alpha, us) with state 0 searching, 1 accepted, 2 no step.
  @sc.function(name=f"{tag}_line_search")
  def line_search_step(carry, trial, x0, xs, us, gains, j_dv, lam, nu_goal, mu):
    alpha = sc.const(0.5) ** trial.cast("float64")
    x_final, u_new, costs = sc.scan(
      policy_step,
      x0,
      [(xs, 0, nx), (us, 0, nu), (gains, (K - 1) * G, -G), (lam, 0, nc), (MASKS, 0, nc), (sc.concat([alpha.reshape((1,)), mu]), 0, 0)],
      length=K,
    )
    cost = costs.sum() + al_terminal(x_final, nu_goal, mu[0])
    expected = -alpha * (j_dv[1] + alpha * j_dv[2])
    tiny = sc.logical_and(sc.greater(expected, 0.0), sc.less(expected, o.expected_decrease_tolerance))
    z = sc.where(sc.greater(expected, 0.0), (j_dv[0] - cost) / expected, -1.0)
    accepted = sc.logical_and(sc.greater_equal(z, o.line_search_lower_bound), sc.less_equal(z, o.line_search_upper_bound))
    state = sc.where(tiny, 2.0, sc.where(accepted, 1.0, 0.0))
    return sc.concat([state.reshape((1,)), cost.reshape((1,)), alpha.reshape((1,)), u_new])

  @sc.function(name=f"{tag}_searching")
  def searching(carry, x0, xs, us, gains, j_dv, lam, nu_goal, mu):
    return sc.less(carry[0], 0.5)

  # Altro's gradient measure, summed here and averaged by the caller: max_i |d_i| / (|u_i| + 1) per stage.
  @sc.function(sc.L("acc", 1), sc.L("u", nu), sc.L("gains", G), name=f"{tag}_gradient_step")
  def gradient_step(acc, u, gains):
    return acc + (gains[:nu].abs() / (u.abs() + 1.0)).max()

  # One iLQR iteration. Carry: (us, cost, rho, drho, done, iterations); params x0, lambda, nu, mu, gradient tolerance.
  @sc.function(name=f"{tag}_ilqr_iteration")
  def ilqr_iteration(carry, x0, lam, nu_goal, mu, grad_tol):
    us, rho, drho, count = carry[: K * nu], carry[K * nu + 1], carry[K * nu + 2], carry[K * nu + 4]
    x_final, xs, costs = sc.scan(rollout_step, x0, [(us, 0, nu), (lam, 0, nc), (MASKS, 0, nc), (mu, 0, 0)], length=K)
    terminal = al_terminal(x_final, nu_goal, mu[0])
    j_prev = costs.sum() + terminal
    value = sc.concat([sc.gradient(terminal, x_final), sc.hessian(terminal, x_final).reshape((nx * nx,))])
    mr = sc.concat([mu, rho.reshape((1,))])
    _, gains, dv = sc.scan(
      backward_step,
      value,
      [(xs, (K - 1) * nx, -nx), (us, (K - 1) * nu, -nu), (lam, (K - 1) * nc, -nc), (MASKS, (K - 1) * nc, -nc), (mr, 0, 0)],
      length=K,
    )
    dv = dv.reshape((K, 2))
    j_dv = sc.stack([j_prev, dv[:, 0].sum(), dv[:, 1].sum()])
    # A failed factorization (NaN gains) skips the line search, as a negligible expected decrease does.
    start = sc.concat([sc.where(sc.isfinite(j_dv[1] + j_dv[2]), 0.0, 2.0).reshape((1,)), sc.const(np.zeros(2)), us])
    ls, _ = sc.while_loop(
      searching, line_search_step, start, max_iter=o.iterations_linesearch, index=True, params=(x0, xs, us, gains, j_dv, lam, nu_goal, mu)
    )
    accepted = sc.logical_and(sc.greater(ls[0], 0.5), sc.less(ls[0], 1.5))
    us_next = sc.where(accepted, ls[3:], us)
    j_next = sc.where(accepted, ls[1], j_prev)
    # Regularization as Altro's: decreased after every backward pass, increased when no step is taken,
    # and bumped by bp_reg_fp when the line search runs out of trials.
    f = o.bp_reg_increase_factor
    drho_down = sc.minimum(drho / f, 1.0 / f)
    rho_down = sc.maximum(rho * drho_down, o.bp_reg_min)
    drho_up = sc.maximum(drho_down * f, f)
    rho_up = sc.maximum(rho_down * drho_up, o.bp_reg_min) + sc.where(sc.less(ls[0], 0.5), o.bp_reg_fp, 0.0)
    rho_next, drho_next = sc.where(accepted, rho_down, rho_up), sc.where(accepted, drho_down, drho_up)
    (grad_sum,) = sc.scan(gradient_step, sc.const(np.zeros(1)), [(us_next, 0, nu), (gains, (K - 1) * G, -G)], length=K)
    grad = grad_sum[0] / K
    dj = j_prev - j_next
    converged = sc.logical_and(
      sc.logical_and(accepted, sc.greater_equal(dj, 0.0)), sc.logical_and(sc.less(dj, o.cost_tolerance_intermediate), sc.less(grad, grad_tol[0]))
    )
    done = sc.logical_or(converged, sc.greater(rho_next, o.bp_reg_max))
    return sc.concat(
      [
        us_next,
        j_next.reshape((1,)),
        rho_next.reshape((1,)),
        drho_next.reshape((1,)),
        sc.cast(done, "float64").reshape((1,)),
        (count + 1.0).reshape((1,)),
      ]
    )

  @sc.function(name=f"{tag}_ilqr_running")
  def ilqr_running(carry, x0, lam, nu_goal, mu, grad_tol):
    return sc.less(carry[K * nu + 3], 0.5)

  @sc.function(sc.L("x", nx), sc.L("u", nu), sc.L("mask", nc), name=f"{tag}_constraint_step")
  def constraint_step(x, u, mask):
    return step(x, u), mask * ocp.ineq(x, u), stage_cost(x, u).reshape((1,))

  # One outer iteration. Carry: (us, lambda, nu, mu, cost, converged, iLQR iterations, violation).
  base = K * nu + L + nx

  @sc.function(name=f"{tag}_al_iteration")
  def al_iteration(carry, outer, x0):
    us, lam, nu_goal, mu, inner = carry[: K * nu], carry[K * nu : K * nu + L], carry[K * nu + L : base], carry[base], carry[base + 3]
    last = sc.greater_equal(outer.cast("float64"), o.iterations_outer - 1.0)
    grad_tol = sc.where(last, o.gradient_tolerance, o.gradient_tolerance_intermediate).reshape((1,))
    # Every inner solve starts from zero regularization, as Altro resets its iLQR solver.
    start = sc.concat([us, sc.const(np.array([np.inf, 0.0, 0.0, 0.0, 0.0]))])
    ic, _ = sc.while_loop(ilqr_running, ilqr_iteration, start, max_iter=o.iterations_inner, params=(x0, lam, nu_goal, mu.reshape((1,)), grad_tol))
    us_new = ic[: K * nu]
    x_final, c_all, costs = sc.scan(constraint_step, x0, [(us_new, 0, nu), (MASKS, 0, nc)], length=K)
    goal = x_final - XF
    violation = sc.maximum(sc.maximum(c_all.max(), 0.0), goal.abs().max())
    converged = sc.less(violation, al_tolerance)
    lam_new = sc.minimum(sc.maximum(lam + mu * c_all, 0.0), o.dual_max)
    nu_new = sc.minimum(sc.maximum(nu_goal + mu * goal, -o.dual_max), o.dual_max)
    mu_new = sc.minimum(mu * o.penalty_scaling, o.penalty_max)
    return sc.concat(
      [
        us_new,
        sc.where(converged, lam, lam_new),
        sc.where(converged, nu_goal, nu_new),
        sc.where(converged, mu, mu_new).reshape((1,)),
        (costs.sum() + terminal_cost(x_final)).reshape((1,)),
        sc.cast(converged, "float64").reshape((1,)),
        (inner + ic[K * nu + 4]).reshape((1,)),
        violation.reshape((1,)),
      ]
    )

  @sc.function(name=f"{tag}_al_running")
  def al_running(carry, x0):
    return sc.less(carry[base + 2], 0.5)

  # ALTRO's projected Newton phase. The primal vector z interleaves (x_k, u_k) for k < K, then x_K; the
  # rows are the initial state, each stage's dynamics defect and masked inequalities, and the goal.
  NZ = nx + nu
  NP = K * NZ + nx
  q, r, qf = (np.asarray(a, dtype=np.float64) for a in (ocp.q, ocp.r, ocp.qf))
  HDIAG = sc.const(np.concatenate([*[np.concatenate([dt * q, dt * r])] * K, qf]) + o.rho_primal)
  masks_np = np.asarray(ocp.masks, dtype=np.float64)
  eq_np = np.concatenate([np.ones(nx), *[np.concatenate([np.ones(nx), np.zeros(nc)]) for _ in range(K)], np.ones(nx)])
  rows_np = np.concatenate([np.ones(nx), *[np.concatenate([np.ones(nx), masks_np[k]]) for k in range(K)], np.ones(nx)])
  EQ, ROWS = sc.const(eq_np), sc.const(rows_np)

  @sc.function(sc.L("xu", NZ), sc.L("x_next", nx), sc.L("mask", nc), name=f"{tag}_pn_stage")
  def pn_stage(xu, x_next, mask):
    x, u = xu[:nx], xu[nx:]
    return sc.concat([step(x, u) - x_next, mask * ocp.ineq(x, u)])

  def pn_constraints(z, x0):
    stages = sc.vmap(pn_stage, K, [(z, 0, NZ), (z, NZ, NZ), (MASKS, 0, nc)])
    return sc.concat([z[:nx] - x0, stages, z[K * NZ :] - XF])

  def pn_violation(c):
    return sc.maximum((c.abs() * EQ).max(), (sc.maximum(c, 0.0) * (1.0 - EQ)).max())

  @sc.function(name=f"{tag}_pn_line_search")
  def pn_line_search(carry, trial, z0, p, v0, x0):
    zt = z0 + (sc.const(0.5) ** trial.cast("float64")) * p
    v = pn_violation(pn_constraints(zt, x0))
    return sc.concat([sc.cast(sc.less(v, v0), "float64").reshape((1,)), v.reshape((1,)), zt])

  @sc.function(name=f"{tag}_pn_searching")
  def pn_searching(carry, z0, p, v0, x0):
    return sc.less(carry[0], 0.5)

  def pn_kkt(z, x0):
    c = pn_constraints(z, x0)
    active = EQ + (1.0 - EQ) * ROWS * sc.cast(sc.greater(c, -o.active_set_tolerance_pn), "float64")
    jac = linalg.SparseMatrix.from_sparse_jacobian(sc.sparse_jacobian(c, z))
    dual = -(o.rho_dual * active + (1.0 - active))  # an inactive row reads lambda_i = 0
    kkt = linalg.SparseMatrix.block([[linalg.SparseMatrix.diag(HDIAG), None], [jac.scale_rows(active), linalg.SparseMatrix.diag(dual)]])
    return kkt, active

  # The projection's factorization, whose analysis the refinement steps solve with (set while tracing it).
  pn_ldl: dict[str, linalg.SparseLDL] = {}

  # One refinement: a step from the current point with the projection's factor, backtracked on the violation.
  # Carry: (z, violation, stop, steps).
  @sc.function(name=f"{tag}_pn_refinement")
  def pn_refinement(carry, factor, active, x0):
    z, v_prev, steps = carry[:NP], carry[NP], carry[NP + 2]
    c = pn_constraints(z, x0)
    v0 = pn_violation(c)
    p = pn_ldl["kkt"].solve_with(factor, sc.concat([sc.const(np.zeros(NP)), -active * c]))[:NP]
    ls, _ = sc.while_loop(pn_searching, pn_line_search, sc.concat([sc.const(np.zeros(2)), z]), max_iter=10, index=True, params=(z, p, v0, x0))
    ok = sc.greater(ls[0], 0.5)
    v = sc.where(ok, ls[1], np.nan)
    rate = v.log() / v_prev.log()
    stop = sc.logical_or(sc.less(v, o.constraint_tolerance), sc.less(rate, o.r_threshold))
    return sc.concat([sc.where(ok, ls[2:], z), v.reshape((1,)), sc.cast(stop, "float64").reshape((1,)), (steps + 1.0).reshape((1,))])

  @sc.function(name=f"{tag}_pn_refining")
  def pn_refining(carry, factor, active, x0):
    return sc.less(carry[NP + 1], 0.5)

  # One projection: linearize, factor, refine. Carry: (z, violation, refinement steps).
  @sc.function(name=f"{tag}_pn_projection")
  def pn_projection(carry, x0):
    z, v, steps = carry[:NP], carry[NP], carry[NP + 1]
    kkt, active = pn_kkt(z, x0)
    pn_ldl["kkt"] = linalg.SparseLDL(kkt, name=f"{tag}_pn_kkt")
    factor = pn_ldl["kkt"].values
    start = sc.concat([z, v.reshape((1,)), sc.const(np.zeros(2))])
    rc, _ = sc.while_loop(pn_refining, pn_refinement, start, max_iter=o.max_refinements - 1, params=(factor, active, x0))
    z_new = rc[:NP]
    return sc.concat([z_new, pn_violation(pn_constraints(z_new, x0)).reshape((1,)), (steps + rc[NP + 2]).reshape((1,))])

  @sc.function(name=f"{tag}_pn_running")
  def pn_running(carry, x0):
    return sc.greater(carry[NP], o.constraint_tolerance)

  Q_ROWS, R_ROWS = sc.const(np.tile(q, K)), sc.const(np.tile(r, K))

  @sc.function(
    sc.L("x0", nx), output=sc.G("us", "xs", "cost", "outer", "inner", "violation", "projections", "projection_steps"), name=f"{tag}_al_ilqr"
  )
  def solve(x0):
    u0 = sc.const(np.asarray(ocp.u0, dtype=np.float64).reshape(-1))
    start = sc.concat([u0, sc.const(np.zeros(L + nx)), sc.const(np.array([o.penalty_initial, np.inf, 0.0, 0.0, np.inf]))])
    carry, outer = sc.while_loop(al_running, al_iteration, start, max_iter=o.iterations_outer, index=True, params=(x0,))
    us = carry[: K * nu]
    x_final, xs, _ = sc.scan(rollout_step, x0, [(us, 0, nu), (sc.const(np.zeros(L)), 0, nc), (MASKS, 0, nc), (sc.const(np.zeros(1)), 0, 0)], length=K)
    if not o.projected_newton:
      zero = sc.const(0.0)
      return us.reshape((K, nu)), sc.concat([xs, x_final]).reshape((K + 1, nx)), carry[base + 1], outer, carry[base + 3], carry[base + 4], zero, zero
    z = sc.concat([sc.concat([xs.reshape((K, nx)), us.reshape((K, nu))], axis=1).reshape((K * NZ,)), x_final])
    pc, projections = sc.while_loop(
      pn_running,
      pn_projection,
      sc.concat([z, pn_violation(pn_constraints(z, x0)).reshape((1,)), sc.const(np.zeros(1))]),
      max_iter=o.n_steps + 1,
      params=(x0,),
    )
    z = pc[:NP]
    xu = z[: K * NZ].reshape((K, NZ))
    xs_k, us = xu[:, :nx].reshape((K * nx,)), xu[:, nx:].reshape((K * nu,))
    e = xs_k - sc.const(np.tile(ocp.xf, K))
    cost = dt * 0.5 * ((e * e * Q_ROWS).sum() + (us * us * R_ROWS).sum()) + terminal_cost(z[K * NZ :])
    violation = pn_violation(pn_constraints(z, x0))
    return us.reshape((K, nu)), sc.concat([xs_k, z[K * NZ :]]).reshape((K + 1, nx)), cost, outer, carry[base + 3], violation, projections, pc[NP + 1]

  return solve
