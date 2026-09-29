# /// script
# requires-python = ">=3.12"
# dependencies = ["scaly"]
# ///
"""iLQR on a car-like robot: an optimal control problem solved by ``sc.ocp.ILQR``, the whole
trajectory optimizer generated as one C function.

A car-like robot, ``x = (px, py, heading, speed)``, steered by ``u = (acceleration, curvature)``,

    px' = v cos(h),  py' = v sin(h),  h' = v kappa,  v' = a,

discretized by forward Euler over steps of ``dt = 0.1`` s, must reach a parking pose past two
circular obstacles. The cost is quadratic in the state error and the inputs plus a smooth exponential
penalty inside each obstacle's safety radius; the goal is a parameter of the problem.

``sc.ocp.ILQR`` is iterative LQR (Li and Todorov 2004, with the regularization and line search of
Tassa et al. 2012) as three Scaly loops, so the generated C contains the complete solver:

* **rollout**: a ``sc.scan`` of the dynamics from ``x0`` that also sums the cost;
* **backward pass**: a ``sc.scan`` backwards over the horizon (negative strides) that carries the
  value function's gradient and Hessian, each step's local model from AD (the map's Jacobians, the
  cost's gradients and Hessians) and its gains from a generated ``cholesky``;
* **forward pass**: a ``sc.while_loop`` halving the step ``alpha`` around a rollout of the feedback
  policy ``u = u_k + alpha k_k + K_k (x - x_k)``, accepted on sufficient decrease;

all inside an outer ``sc.while_loop`` that adapts the Levenberg-Marquardt regularization ``mu`` of
``Q_uu``; a factorization that fails gives NaN gains, no step is accepted, and ``mu`` grows.
``examples/ocp/ilqr.ipynb`` builds the same loops one by one.

The solve is one call of the solver Function; the solution is checked for first-order optimality
with ``sc.gradient`` of the rolled-out cost in the inputs (the single-shooting gradient, by one
reverse sweep), and compared with SciPy's L-BFGS on that same gradient.

The generated C lands in ``examples/generated/ilqr/``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy import optimize

import scaly as sc
from scaly import integrators as si
from scaly import ocp
from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parents[1] / "generated" / "ilqr"
NX, NU, N, DT = 4, 2, 60, 0.1
Q = np.diag([0.0, 0.0, 0.0, 0.1])
R = np.diag([0.05, 0.5])
QF = np.diag([100.0, 100.0, 50.0, 10.0])
OBSTACLES = np.array([[2.0, 0.6, 0.6], [4.0, 1.7, 0.5]])  # (x, y, radius)
PENALTY, SOFTNESS = 20.0, 0.1


@sc.function(NX, NU, output="xdot", name="ilqr_car")
def car(x: sc.Expr, u: sc.Expr) -> sc.Expr:
  h, v = x[2], x[3]
  return sc.stack([v * h.cos(), v * h.sin(), v * u[1], u[0]])


@sc.function(NX, NU, sc.L("goal", NX), output="l", name="ilqr_stage_cost")
def stage_cost(x: sc.Expr, u: sc.Expr, goal: sc.Expr) -> sc.Expr:
  e = x - goal
  cost = 0.5 * (e @ sc.const(Q) @ e) + 0.5 * (u @ sc.const(R) @ u)
  for ox, oy, radius in OBSTACLES:
    gap = (x[0] - ox) ** 2 + (x[1] - oy) ** 2 - radius**2
    cost = cost + PENALTY * (-gap / SOFTNESS).exp()
  return cost


@sc.function(NX, sc.L("goal", NX), output="vf", name="ilqr_terminal_cost")
def terminal_cost(x: sc.Expr, goal: sc.Expr) -> sc.Expr:
  e = x - goal
  return 0.5 * (e @ sc.const(QF) @ e)


# The running cost at the grid points is dt times their sum; Euler shooting makes the map x + dt f.
PROBLEM = ocp.transcribe(
  ocp.ContinuousOCP(car, T=N * DT, stage_cost=stage_cost, terminal_cost=terminal_cost, name="ilqr_parking"), ocp.MultipleShooting(si.Euler()), N=N
)
METHOD = ocp.ILQR(max_iter=200, tol=1e-10, max_line_search=12)
solve = ocp.solver(PROBLEM, METHOD)
step = si.explicit(car, "euler", dt=DT, name="ilqr_car_step")


@sc.function(sc.L("us", N * NU), sc.L("x0", NX), sc.L("goal", NX), output=sc.G("cost", "grad"), name="ilqr_shooting_cost")
def shooting_cost(us: sc.Expr, x0: sc.Expr, goal: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
  """The cost of the controls ``us`` rolled out from ``x0``, and its gradient in them."""
  x, cost = x0, 0.0
  for k in range(N):
    u = us[k * NU : (k + 1) * NU]
    cost = cost + DT * stage_cost(x, u, goal)
    x = step(x, u)
  cost = cost + terminal_cost(x, goal)
  return cost, sc.gradient(cost, us)


def main() -> dict:
  x0, goal = np.zeros(NX), np.array([6.0, 2.5, np.pi / 2, 0.0])
  warm = ocp.initial_guess(PROBLEM, METHOD, x0)  # every control zero
  solve(x0, goal, warm)  # compile
  start = time.perf_counter()
  xs, us, _, info = solve(x0, goal, warm)
  elapsed = time.perf_counter() - start
  _, grad = shooting_cost(us.reshape(-1), x0, goal)
  lbfgs = optimize.minimize(
    lambda u: shooting_cost(u, x0, goal), np.zeros(N * NU), jac=True, method="L-BFGS-B", options={"maxiter": 5000, "gtol": 1e-9}
  )
  clearance = min(float(np.min(np.hypot(xs[:, 0] - ox, xs[:, 1] - oy) - r)) for ox, oy, r in OBSTACLES)
  return {
    "us": us,
    "xs": xs,
    "cost": float(info.objective),
    "status": sc.Status(int(info.status)).name,
    "iterations": int(info.iter),
    "seconds": elapsed,
    "gradient_norm": float(np.abs(grad).max()),
    "final_error": xs[-1] - goal,
    "clearance": clearance,
    "lbfgs_cost": float(lbfgs.fun),
    "lbfgs_evaluations": int(lbfgs.nfev),
  }


if __name__ == "__main__":
  out = main()
  print(
    f"iLQR: {out['status']}, cost {out['cost']:.6f} after {out['iterations']} iterations, {1e3 * out['seconds']:.2f} ms for the whole solve in generated C"
  )
  print(f"first-order optimality: |d cost / d u|_inf = {out['gradient_norm']:.1e} (reverse mode through the rollout)")
  print(f"final pose error {np.array2string(out['final_error'], precision=3)}; closest approach to an obstacle's edge {out['clearance']:.3f} m")
  print(f"L-BFGS on the same single-shooting gradient: cost {out['lbfgs_cost']:.6f} after {out['lbfgs_evaluations']} evaluations")
  for fn in (solve, shooting_cost):
    write_module(fn, GENERATED)
  print(f"generated C in {GENERATED}")
