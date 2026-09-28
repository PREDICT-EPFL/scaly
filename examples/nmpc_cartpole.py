"""Nonlinear MPC: swinging up a cart-pole with multiple shooting and IPOPT, in closed loop.

The cart-pole (cart position ``p``, pole angle ``theta`` from upright, and their rates) starts hanging
down and must be swung up and balanced with a bounded force on a bounded track. At every sampling
instant the controller solves

    minimize   sum_k l(x_k, u_k) + l_N(x_N)
    subject to x_0 = x_measured,   x_{k+1} = F(x_k, u_k)  (k = 0..N-1),   |u_k| <= u_max,   |p_k| <= p_max + s_k,   s_k >= 0,

with ``F`` two RK4 substeps of the nonlinear dynamics, a stage cost on ``1 - cos(theta)`` so that
both upright angles count, and the track limit softened by slacks with an exact l1 penalty, since
a hard state constraint can become infeasible when the plant does not follow the model exactly.
What it shows:

* the shooting defects are one ``sc.vmap`` of a single-stage ``Function``, so IPOPT's sparse
  constraint Jacobian and Lagrangian Hessian are built from one stage's derivatives, and the
  generated code has one loop, not ``N`` copies;
* the variables are a tree, ``(states, inputs, slacks)``, with bounds per leaf, the track limits are
  two ``sc.opt.bounded`` inequality groups, and the measured state is a parameter, so one generated
  solver runs the whole closed loop;
* the loop warm-starts each solve from the previous solution shifted by one stage, and reads
  IPOPT's iteration count and timings from ``solver_stats()``.

The generated C lands in ``examples/generated/nmpc_cartpole/``; it links against the vendored IPOPT.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

import scaly as sc
from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parent / "generated" / "nmpc_cartpole"
NX, NU, N, DT = 4, 1, 40, 0.05
M_CART, M_POLE, LENGTH, G = 1.0, 0.3, 0.5, 9.81
U_MAX, P_MAX = 15.0, 1.0
Q = np.array([2.0, 20.0, 0.1, 0.1])  # p, 1 - cos(theta), rates
R, QN = 0.02, 10.0
SIM_STEPS = 120


def dynamics(x: sc.Expr, u: sc.Expr) -> sc.Expr:
  theta, pd, td = x[1], x[2], x[3]
  s, c = theta.sin(), theta.cos()
  den = M_CART + M_POLE * s * s
  pdd = (u[0] + M_POLE * LENGTH * td * td * s - M_POLE * G * s * c) / den
  tdd = (G * s * (M_CART + M_POLE) - c * (u[0] + M_POLE * LENGTH * td * td * s)) / (LENGTH * den)
  return sc.stack([pd, td, pdd, tdd])


def rk4(x: sc.Expr, u: sc.Expr, h: float) -> sc.Expr:
  k1 = dynamics(x, u)
  k2 = dynamics(x + 0.5 * h * k1, u)
  k3 = dynamics(x + 0.5 * h * k2, u)
  k4 = dynamics(x + h * k3, u)
  return x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


@sc.function(NX, NU, NX)
def defect(x: sc.Expr, u: sc.Expr, x_next: sc.Expr) -> sc.Expr:
  return rk4(rk4(x, u, DT / 2), u, DT / 2) - x_next


@sc.function(NX, NU)
def stage_cost(x: sc.Expr, u: sc.Expr) -> sc.Expr:
  e = sc.stack([x[0], 1.0 - x[1].cos(), x[2], x[3]])
  return ((sc.const(Q) * e * e).sum() + R * u[0] * u[0]).reshape((1,)) * DT


def terminal_cost(x: sc.Expr) -> sc.Expr:
  e = sc.stack([x[0], 1.0 - x[1].cos(), x[2], x[3]])
  return QN * (sc.const(Q) * e * e).sum()


SLACK_PENALTY = 1e3  # an exact (l1) penalty: the track limit holds whenever it can


@sc.opt.problem(vars=sc.G(sc.L("xs", (N + 1) * NX), sc.L("us", N * NU), sc.L("slack", N)), params=sc.L("x0", NX))
def swing_up(variables: tuple[sc.Expr, sc.Expr, sc.Expr], x0: sc.Expr) -> sc.opt.ProblemSpec:
  xs, us, slack = variables
  defects = sc.vmap(defect, N, [(xs, 0, NX), (us, 0, NU), (xs, NX, NX)])
  costs = sc.vmap(stage_cost, N, [(xs, 0, NX), (us, 0, NU)])
  positions = xs.reshape((N + 1, NX))[1:, 0]
  return sc.opt.ProblemSpec(
    minimize=costs.sum() + terminal_cost(xs[N * NX :]) + SLACK_PENALTY * slack.sum(),
    eq=(xs[:NX] - x0, defects),
    ineq=(sc.opt.bounded(positions - slack, hi=P_MAX, name="track_right"), sc.opt.bounded(positions + slack, lo=-P_MAX, name="track_left")),
    lb=(sc.opt.NO_LB, sc.const(np.full(N * NU, -U_MAX)), sc.const(np.zeros(N))),
    ub=(sc.opt.NO_UB, sc.const(np.full(N * NU, U_MAX)), sc.opt.NO_UB),
  )


mpc = sc.opt.solver(swing_up, sc.opt.IPOPT(options={"print_level": 0, "tol": 1e-8, "max_iter": 500}), name="cartpole_mpc")


def plant_step(x: np.ndarray, u: float, substeps: int = 10) -> np.ndarray:
  """The true system, integrated more finely than the controller's model."""
  h = DT / substeps

  def f(x: np.ndarray) -> np.ndarray:
    theta, pd, td = x[1:]
    s, c = np.sin(theta), np.cos(theta)
    den = M_CART + M_POLE * s * s
    return np.array(
      [
        pd,
        td,
        (u + M_POLE * LENGTH * td**2 * s - M_POLE * G * s * c) / den,
        (G * s * (M_CART + M_POLE) - c * (u + M_POLE * LENGTH * td**2 * s)) / (LENGTH * den),
      ]
    )

  for _ in range(substeps):
    k1 = f(x)
    k2 = f(x + 0.5 * h * k1)
    k3 = f(x + 0.5 * h * k2)
    k4 = f(x + h * k3)
    x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
  return x


def main(sim_steps: int = SIM_STEPS) -> dict:
  x = np.array([0.0, np.pi, 0.0, 0.0])
  # Hanging still with u = 0 is itself a KKT point, by symmetry; a pumping guess breaks the tie.
  xs_guess, us_guess, slack = np.tile(x, N + 1), 0.5 * U_MAX * np.sin(2 * np.pi * np.arange(N) / N), np.zeros(N)
  lam_box = (np.zeros((N + 1) * NX), np.zeros(N * NU), np.zeros(N))
  lam_eq, lam_ineq = np.zeros(swing_up.n_eq), np.zeros(swing_up.n_ineq)
  history, inputs, iterations, times, statuses = [x], [], [], [], []
  for _ in range(sim_steps):
    (xs, us, slack), lam_box, lam_eq, lam_ineq, _ = mpc((xs_guess, us_guess, slack), lam_box, lam_eq, lam_ineq, x)
    stats = sc.opt.solver_stats(mpc)
    iterations.append(stats.iter)
    times.append(stats.t_total)
    statuses.append(stats.to_solver_status().name)
    u = float(us[0])
    x = plant_step(x, u)
    history.append(x)
    inputs.append(u)
    # Shift the solution one stage for the next warm start.
    xs_guess = np.r_[xs[NX:], xs[-NX:]]
    us_guess = np.r_[us[NU:], us[-NU:]]
    slack = np.r_[slack[1:], slack[-1:]]
  history, inputs = np.array(history), np.array(inputs)
  upright = np.abs(np.cos(history[:, 1]) - 1) < 1e-2
  return {
    "history": history,
    "inputs": inputs,
    "iterations": np.array(iterations),
    "times": np.array(times),
    "statuses": statuses,
    "upright_from": float(DT * (len(upright) - np.argmin(upright[::-1]))) if upright.any() else np.inf,  # stays upright from here
    "final": history[-1],
  }


if __name__ == "__main__":
  out = main()
  print(f"{SIM_STEPS} closed-loop steps ({SIM_STEPS * DT:.0f} s), IPOPT statuses: {dict(Counter(out['statuses']))}")
  print(
    f"IPOPT iterations: first solve {out['iterations'][0]}, then median {np.median(out['iterations'][1:]):.0f} warm-started; median solve time {1e3 * np.median(out['times']):.2f} ms"
  )
  print(f"pole within 1e-2 of upright for good from t = {out['upright_from']:.2f} s; final state {np.array2string(out['final'], precision=4)}")
  print(
    f"largest |force| {np.abs(out['inputs']).max():.2f} N (bound {U_MAX}), largest |p| {np.abs(out['history'][:, 0]).max():.3f} m (bound {P_MAX})"
  )
  module = write_module(mpc, GENERATED)
  print(f"generated C for the MPC solver ({len(module.source.splitlines())} lines for a {N}-stage horizon) in {GENERATED}")
