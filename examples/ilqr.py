"""iLQR written in Scaly: a whole trajectory optimizer generated as one C function.

A car-like robot, ``x = (px, py, heading, speed)``, steered by ``u = (acceleration, curvature)``,

    px+ = px + dt v cos(h),  py+ = py + dt v sin(h),  h+ = h + dt v kappa,  v+ = v + dt a,

must reach a parking pose past two circular obstacles. The cost is quadratic in the state error and
the inputs plus a smooth exponential penalty inside each obstacle's safety radius.

Iterative LQR (Li and Todorov 2004, with the regularization and line search of Tassa et al. 2012)
alternates three loops, and each is a Scaly loop, so the generated C contains the complete solver:

* **rollout**: a ``sc.scan`` of the dynamics from ``x0`` that also sums the cost;
* **backward pass**: a ``sc.scan`` backwards over the horizon (negative strides) that carries the
  value function's gradient and Hessian. Each step builds its local model from AD, the dynamics
  Jacobians with ``sc.jacobian`` and the cost's gradients and Hessians with ``sc.gradient`` and
  ``sc.hessian``, and solves the ``2 x 2`` system for the feedforward and feedback gains with the
  generated ``cholesky``;
* **forward pass**: a ``sc.while_loop`` halving the step ``alpha`` (``index=True`` gives the trial
  number) around a rollout of the feedback policy ``u = u_k + alpha k_k + K_k (x - x_k)``, accepted on
  sufficient decrease against the backward pass's predicted change;

all inside an outer ``sc.while_loop`` that adapts the Levenberg-Marquardt regularization ``mu`` of
``Q_uu``. A factorization that fails (a non-positive-definite ``Q_uu``) gives NaN gains, the trial
cost is NaN, no step is accepted, and ``mu`` grows: the NaN is the error signal.

The whole solve is one call of ``ilqr(x0, goal)``; the solution is checked for first-order
optimality with ``sc.gradient`` of the rolled-out cost in the inputs (the single-shooting gradient,
by one reverse sweep), and compared with SciPy's L-BFGS on that same gradient.

The generated C lands in ``examples/generated/ilqr/``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy import optimize

import scaly as sc
from scaly import linalg
from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parent / "generated" / "ilqr"
NX, NU, N, DT = 4, 2, 60, 0.1
Q = np.diag([0.0, 0.0, 0.0, 0.1])
R = np.diag([0.05, 0.5])
QF = np.diag([100.0, 100.0, 50.0, 10.0])
OBSTACLES = np.array([[2.0, 0.6, 0.6], [4.0, 1.7, 0.5]])  # (x, y, radius)
PENALTY, SOFTNESS = 20.0, 0.1
MAX_OUTER, MAX_LINE_SEARCH, TOL = 200, 12, 1e-10
MU_MIN, MU_MAX = 1e-8, 1e10


def step(x: sc.Expr, u: sc.Expr) -> sc.Expr:
  px, py, h, v = x[0], x[1], x[2], x[3]
  return sc.stack([px + DT * v * h.cos(), py + DT * v * h.sin(), h + DT * v * u[1], v + DT * u[0]])


def stage_cost(x: sc.Expr, u: sc.Expr, goal: sc.Expr) -> sc.Expr:
  e = x - goal
  cost = 0.5 * (e @ sc.const(Q) @ e) + 0.5 * (u @ sc.const(R) @ u)
  for ox, oy, radius in OBSTACLES:
    gap = (x[0] - ox) ** 2 + (x[1] - oy) ** 2 - radius**2
    cost = cost + PENALTY * (-gap / SOFTNESS).exp()
  return cost * DT


def terminal_cost(x: sc.Expr, goal: sc.Expr) -> sc.Expr:
  e = x - goal
  return 0.5 * (e @ sc.const(QF) @ e)


# Rollout under the nominal inputs: states x_0 ... x_{N-1} stacked, costs per step.
@sc.function(NX, NU, NX)
def rollout_step(x: sc.Expr, u: sc.Expr, goal: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
  return step(x, u), x, stage_cost(x, u, goal).reshape((1,))


def rollout(x0: sc.Expr, us: sc.Expr, goal: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
  """``(x_N, stacked x_0..x_{N-1}, total cost)``."""
  x_final, xs, costs = sc.scan(rollout_step, x0, [(us, 0, NU), (goal, 0, 0)], length=N)
  return x_final, xs, costs.sum() + terminal_cost(x_final, goal)


# Backward pass. Carry: V_x and V_xx. Sliced (backwards): x_k, u_k. Broadcast: goal and mu.
@sc.function(NX + NX * NX, NX, NU, NX + 1)
def backward_step(value: sc.Expr, x: sc.Expr, u: sc.Expr, goal_mu: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
  goal, mu = goal_mu[:NX], goal_mu[NX]
  vx, vxx = value[:NX], value[NX:].reshape((NX, NX))
  x_next, cost = step(x, u), stage_cost(x, u, goal)
  fx, fu = sc.jacobian(x_next, x), sc.jacobian(x_next, u)
  lx, lu = sc.gradient(cost, x), sc.gradient(cost, u)
  qx = lx + fx.T @ vx
  qu = lu + fu.T @ vx
  qxx = sc.hessian(cost, x) + fx.T @ vxx @ fx
  quu = sc.hessian(cost, u) + fu.T @ vxx @ fu
  qux = sc.jacobian(lu, x) + fu.T @ vxx @ fx
  chol = linalg.cholesky(quu + sc.const(np.eye(NU)) * mu)
  k = -linalg.cho_solve(chol, qu)
  gain = -linalg.cho_solve(chol, qux)
  vx_prev = qx + gain.T @ (quu @ k) + gain.T @ qu + qux.T @ k
  vxx_prev = qxx + gain.T @ quu @ gain + gain.T @ qux + qux.T @ gain
  vxx_prev = 0.5 * (vxx_prev + vxx_prev.T)
  dv = sc.stack([k @ qu, 0.5 * (k @ (quu @ k))])
  return sc.concat([vx_prev, vxx_prev.reshape((NX * NX,))]), sc.concat([k, gain.reshape((NU * NX,))]), dv


# Forward pass under the policy. Carry: x. Sliced: x_k, u_k, gains (read in reverse). Broadcast: alpha, goal.
@sc.function(NX, NX, NU, NU + NU * NX, 1 + NX)
def policy_step(x: sc.Expr, x_nom: sc.Expr, u_nom: sc.Expr, gains: sc.Expr, alpha_goal: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
  alpha, goal = alpha_goal[0], alpha_goal[1:]
  u = u_nom + alpha * gains[:NU] + gains[NU:].reshape((NU, NX)) @ (x - x_nom)
  return step(x, u), u, stage_cost(x, u, goal).reshape((1,))


# The line-search carry is (accepted, cost, alpha, inputs); the rest are loop parameters.
@sc.function
def line_search_step(carry: sc.Expr, trial: sc.Expr, x0: sc.Expr, goal: sc.Expr, xs: sc.Expr, us: sc.Expr, gains: sc.Expr, j_dv: sc.Expr) -> sc.Expr:
  alpha = sc.const(0.5) ** trial.cast("float64")
  last = (N - 1) * (NU + NU * NX)
  x_final, u_new, costs = sc.scan(
    policy_step, x0, [(xs, 0, NX), (us, 0, NU), (gains, last, -(NU + NU * NX)), (sc.concat([alpha.reshape((1,)), goal]), 0, 0)], length=N
  )
  cost = costs.sum() + terminal_cost(x_final, goal)
  expected = alpha * j_dv[1] + alpha * alpha * j_dv[2]  # negative: the predicted change
  accepted = sc.less(cost - j_dv[0], 1e-4 * expected)
  return sc.concat([sc.cast(accepted, "float64").reshape((1,)), cost.reshape((1,)), alpha.reshape((1,)), u_new])


@sc.function
def not_accepted(carry: sc.Expr, x0: sc.Expr, goal: sc.Expr, xs: sc.Expr, us: sc.Expr, gains: sc.Expr, j_dv: sc.Expr) -> sc.Expr:
  return sc.less(carry[0], 0.5)


# The outer carry is (inputs, cost, mu, done); the initial state and the goal are loop parameters.
@sc.function
def ilqr_iteration(carry: sc.Expr, x0: sc.Expr, goal: sc.Expr) -> sc.Expr:
  us, mu = carry[: N * NU], carry[N * NU + 1]
  x_final, xs, cost = rollout(x0, us, goal)
  terminal = sc.concat([sc.gradient(terminal_cost(x_final, goal), x_final), sc.hessian(terminal_cost(x_final, goal), x_final).reshape((NX * NX,))])
  goal_mu = sc.concat([goal, mu.reshape((1,))])
  _, gains, dv = sc.scan(backward_step, terminal, [(xs, (N - 1) * NX, -NX), (us, (N - 1) * NU, -NU), (goal_mu, 0, 0)], length=N)
  j_dv = sc.stack([cost, dv.reshape((N, 2))[:, 0].sum(), dv.reshape((N, 2))[:, 1].sum()])
  start = sc.concat([sc.const(np.zeros(3)), us])
  ls, _ = sc.while_loop(not_accepted, line_search_step, start, max_iter=MAX_LINE_SEARCH, index=True, params=(x0, goal, xs, us, gains, j_dv))
  accepted = sc.greater(ls[0], 0.5)
  us_next = sc.where(accepted, ls[3:], us)
  cost_next = sc.where(accepted, ls[1], cost)
  mu_next = sc.where(accepted, sc.maximum(mu * 0.1, MU_MIN), mu * 10.0)
  improvement = (cost - cost_next) / sc.maximum(cost.abs(), 1.0)
  done = sc.logical_or(sc.logical_and(accepted, sc.less(improvement, TOL)), sc.greater(mu_next, MU_MAX))
  return sc.concat([us_next, cost_next.reshape((1,)), mu_next.reshape((1,)), sc.cast(done, "float64").reshape((1,))])


@sc.function
def not_done(carry: sc.Expr, x0: sc.Expr, goal: sc.Expr) -> sc.Expr:
  return sc.less(carry[N * NU + 2], 0.5)


@sc.function(NX, NX, output=sc.G("us", "xs", "cost", "iterations", "mu"))
def ilqr(x0: sc.Expr, goal: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr]:
  start = sc.concat([sc.const(np.zeros(N * NU)), sc.const(np.array([np.inf, 1.0, 0.0]))])
  carry, n_iter = sc.while_loop(not_done, ilqr_iteration, start, max_iter=MAX_OUTER, params=(x0, goal))
  us = carry[: N * NU]
  x_final, xs, cost = rollout(x0, us, goal)
  return us.reshape((N, NU)), sc.concat([xs, x_final]).reshape((N + 1, NX)), cost, n_iter, carry[N * NU + 1]


@sc.function(N * NU, NX, NX, output=sc.G("cost", "grad"))
def shooting_cost(us: sc.Expr, x0: sc.Expr, goal: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
  cost = rollout(x0, us, goal)[2]
  return cost, sc.gradient(cost, us)


def main() -> dict:
  x0, goal = np.zeros(NX), np.array([6.0, 2.5, np.pi / 2, 0.0])
  ilqr(x0, goal)  # compile
  start = time.perf_counter()
  us, xs, cost, iterations, mu = ilqr(x0, goal)
  elapsed = time.perf_counter() - start
  _, grad = shooting_cost(us.reshape(-1), x0, goal)
  lbfgs = optimize.minimize(
    lambda u: shooting_cost(u, x0, goal), np.zeros(N * NU), jac=True, method="L-BFGS-B", options={"maxiter": 5000, "gtol": 1e-9}
  )
  clearance = min(float(np.min(np.hypot(xs[:, 0] - ox, xs[:, 1] - oy) - r)) for ox, oy, r in OBSTACLES)
  return {
    "us": us,
    "xs": xs,
    "cost": float(cost),
    "iterations": int(iterations),
    "seconds": elapsed,
    "gradient_norm": float(np.abs(grad).max()),
    "final_error": xs[-1] - goal,
    "clearance": clearance,
    "lbfgs_cost": float(lbfgs.fun),
    "lbfgs_evaluations": int(lbfgs.nfev),
  }


if __name__ == "__main__":
  out = main()
  print(f"iLQR: cost {out['cost']:.6f} after {out['iterations']} iterations, {1e3 * out['seconds']:.2f} ms for the whole solve in generated C")
  print(f"first-order optimality: |d cost / d u|_inf = {out['gradient_norm']:.1e} (reverse mode through the rollout)")
  print(f"final pose error {np.array2string(out['final_error'], precision=3)}; closest approach to an obstacle's edge {out['clearance']:.3f} m")
  print(f"L-BFGS on the same single-shooting gradient: cost {out['lbfgs_cost']:.6f} after {out['lbfgs_evaluations']} evaluations")
  for fn in (ilqr, shooting_cost):
    write_module(fn, GENERATED)
  print(f"generated C in {GENERATED}")
