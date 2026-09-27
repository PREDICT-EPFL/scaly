"""Moving-horizon estimation of a pendulum's state and friction with the generated SQP solver.

A damped pendulum, ``theta'' = -(g / l) sin(theta) - b theta' + w``, with an unknown friction
coefficient ``b`` and a random torque ``w``, is observed through noisy angle measurements only. At
each sample a moving-horizon estimator solves, over the last ``M`` samples,

    minimize   1/2 |x_0 - xbar|^2_P + 1/2 (b - bbar)^2 / s_b^2 + 1/2 sum_k (y_k - theta_k)^2 / s_y^2 + 1/2 sum_k w_k^2 / s_w^2
    subject to x_{k+1} = F(x_k, b) + (0, dt w_k),   b >= 0,

for the states in the window, the noise sequence and ``b``. ``F`` is an RK4 step; the defects are one
``sc.vmap`` with ``b`` broadcast to every stage (stride 0). The arrival cost is centred on the
previous window's estimate of its second state.

The solver is ``sc.solver(problem, "sqp")``: Scaly's own SQP, generated as C around PIQP
subproblems, which warm-starts primal and dual iterates unconditionally, the natural fit for a
receding horizon. The same ``Problem`` given to IPOPT gives the same estimate on a window, since the
problem description names no backend.

The generated C lands in ``examples/generated/mhe/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import scaly as sc
from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parent / "generated" / "mhe"
M, DT, T = 20, 0.05, 240
G_L, B_TRUE = 9.81 / 0.5, 0.6
S_Y, S_W, S_B = 0.03, 0.3, 0.05  # a small S_B makes b a slow random walk across windows
P_ARRIVAL = np.array([1 / 0.05**2, 1 / 0.3**2])


def rk4(x: sc.Expr, b: sc.Expr) -> sc.Expr:
  def f(x: sc.Expr) -> sc.Expr:
    return sc.stack([x[1], -G_L * x[0].sin() - b * x[1]])

  k1 = f(x)
  k2 = f(x + 0.5 * DT * k1)
  k3 = f(x + 0.5 * DT * k2)
  k4 = f(x + DT * k3)
  return x + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


@sc.function(2, 1, 1, 2)
def defect(x: sc.Expr, w: sc.Expr, b: sc.Expr, x_next: sc.Expr) -> sc.Expr:
  return rk4(x, b[0]) + sc.stack([sc.const(0.0), DT * w[0]]) - x_next


@sc.problem(
  vars=sc.G(sc.L("xs", 2 * (M + 1)), sc.L("ws", M), sc.L("b", 1)),
  params=sc.G(sc.L("ys", M + 1), sc.L("xbar", 2), sc.L("bbar", 1)),
)
def estimation(variables: tuple[sc.Expr, ...], params: tuple[sc.Expr, ...]) -> sc.ProblemSpec:
  xs, ws, b = variables
  ys, xbar, bbar = params
  thetas = xs.reshape((M + 1, 2))[:, 0]
  cost = (
    0.5 * (sc.const(P_ARRIVAL) * (xs[:2] - xbar) ** 2).sum()
    + 0.5 * sc.sumsqr(b - bbar) / S_B**2
    + 0.5 * sc.sumsqr(ys - thetas) / S_Y**2
    + 0.5 * sc.sumsqr(ws) / S_W**2
  )
  defects = sc.vmap(defect, M, [(xs, 0, 2), (ws, 0, 1), (b, 0, 0), (xs, 2, 2)])
  return sc.ProblemSpec(
    minimize=cost,
    eq=(defects,),
    lb=(sc.NO_LB, sc.NO_LB, sc.const(np.zeros(1))),
    ub=None,
  )


mhe_sqp = sc.solver(estimation, "sqp", name="mhe_sqp", options={"tol": 1e-9, "dual_tol": 1e-8, "qp_tol": 1e-10})


def simulate(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(seed)
  x, xs, ys = np.array([1.2, 0.0]), [], []

  def f(x: np.ndarray) -> np.ndarray:
    return np.array([x[1], -G_L * np.sin(x[0]) - B_TRUE * x[1]])

  for _ in range(T):
    xs.append(x)
    ys.append(x[0] + S_Y * rng.standard_normal())
    k1 = f(x)
    k2 = f(x + 0.5 * DT * k1)
    k3 = f(x + 0.5 * DT * k2)
    k4 = f(x + DT * k3)
    x = x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4) + np.array([0.0, DT * S_W * rng.standard_normal()])
  return np.array(xs), np.array(ys)


def window_guess(ys: np.ndarray) -> np.ndarray:
  """States from the measurements, rates from differences: a crude start for the first window."""
  rates = np.gradient(ys, DT)
  return np.stack([ys, rates], axis=1).reshape(-1)


def main() -> dict:
  truth, ys = simulate()
  n_eq = estimation.n_eq
  xbar, bbar = np.array([ys[0], 0.0]), np.array([0.2])
  xs, ws, b = window_guess(ys[: M + 1]), np.zeros(M), bbar.copy()
  lam_box, lam_eq = (np.zeros(2 * (M + 1)), np.zeros(M), np.zeros(1)), np.zeros(n_eq)
  estimates, b_hist, iterations = [], [], []
  for k in range(M, T):
    window = ys[k - M : k + 1]
    (xs, ws, b), lam_box, lam_eq, _ = mhe_sqp((xs, ws, b), lam_box, lam_eq, np.zeros(0), (window, xbar, bbar))
    iterations.append(mhe_sqp.solver_stats().iter)
    estimates.append(xs[-2:])
    b_hist.append(float(b[0]))
    # Arrival cost for the next window, then shift the solution.
    xbar, bbar = xs[2:4].copy(), b.copy()
    xs = np.r_[xs[2:], xs[-2:]]
    ws = np.r_[ws[1:], 0.0]
  estimates = np.array(estimates)
  rate_error = estimates[:, 1] - truth[M:, 1]
  naive = np.gradient(ys, DT)[M:] - truth[M:, 1]

  # The same window with IPOPT: the Problem is backend-independent.
  mhe_ipopt = sc.solver(estimation, "ipopt", name="mhe_ipopt", options={"tol": 1e-10})
  params = (ys[T - M - 1 :], xbar, bbar)
  zeros = ((np.zeros(2 * (M + 1)), np.zeros(M), np.zeros(1)), (np.zeros(2 * (M + 1)), np.zeros(M), np.zeros(1)), np.zeros(n_eq), np.zeros(0))
  (x_s, _, b_s), *_ = mhe_sqp((window_guess(params[0]), np.zeros(M), bbar), *zeros[1:], params)
  (x_i, _, b_i), *_ = mhe_ipopt((window_guess(params[0]), np.zeros(M), bbar), *zeros[1:], params)
  return {
    "b": np.array(b_hist),
    "rate_rmse": float(np.sqrt(np.mean(rate_error[20:] ** 2))),
    "naive_rmse": float(np.sqrt(np.mean(naive[20:] ** 2))),
    "angle_rmse": float(np.sqrt(np.mean((estimates[20:, 0] - truth[M + 20 :, 0]) ** 2))),
    "iterations": np.array(iterations),
    "sqp_vs_ipopt": max(np.abs(x_s - x_i).max(), float(abs(b_s[0] - b_i[0]))),
  }


if __name__ == "__main__":
  out = main()
  print(f"{T - M} MHE windows of {M} samples, SQP iterations: median {np.median(out['iterations']):.0f}, max {out['iterations'].max()}")
  print(f"friction estimate {out['b'][0]:.3f} -> {out['b'][-1]:.3f} (true {B_TRUE})")
  print(
    f"angle RMSE {out['angle_rmse']:.4f} rad (measurement noise {S_Y}); rate RMSE {out['rate_rmse']:.3f} rad/s vs {out['naive_rmse']:.3f} by differencing"
  )
  print(f"the same window with IPOPT agrees to {out['sqp_vs_ipopt']:.1e}")
  write_module(mhe_sqp, GENERATED)
  print(f"generated C for the MHE (SQP around PIQP) in {GENERATED}")
