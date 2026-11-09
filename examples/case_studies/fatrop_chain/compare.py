"""Check the hanging chain against an independent CasADi transcription and IPOPT.

Vanroye et al., "FATROP", IROS 2023 (https://arxiv.org/abs/2303.16746).
"""

from __future__ import annotations

import casadi as ca
import numpy as np

import scaly as sc
from scaly_impl import build, initial_guess, initial_state, sizes, SPRING, REST_LENGTH, MASS, GRAVITY


def casadi_problem(no_masses: int = 6, horizon: int = 25, duration: float = 4.0):
  nx, nu = sizes(no_masses)
  width = nx + nu
  x, u = ca.SX.sym("x", nx), ca.SX.sym("u", nu)
  positions = [ca.DM.zeros(2), *(x[2 * i : 2 * i + 2] for i in range(no_masses + 1))]
  forces = []
  for left, right in zip(positions[:-1], positions[1:], strict=True):
    displacement = right - left
    forces.append(SPRING * (1.0 - REST_LENGTH / ca.norm_2(displacement)) * displacement)
  acceleration = ca.vertcat(*((forces[i + 1] - forces[i]) / MASS + ca.vertcat(0.0, -GRAVITY) for i in range(no_masses)))
  rhs = ca.Function("rhs", [x, u], [ca.vertcat(x[2 * (no_masses + 1) :], u, acceleration)])
  dt = duration / horizon
  k1 = rhs(x, u)
  k2 = rhs(x + dt * k1 / 2, u)
  k3 = rhs(x + dt * k2 / 2, u)
  k4 = rhs(x + dt * k3, u)
  step = ca.Function("step", [x, u], [x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)])
  z, p = ca.MX.sym("z", horizon * width + nx), ca.MX.sym("p", nx)
  gaps, controls, cost = [], [], 0
  for k in range(horizon):
    state, control = z[k * width : k * width + nx], z[k * width + nx : (k + 1) * width]
    gaps.append(z[(k + 1) * width : (k + 1) * width + nx] - step(state, control))
    controls.append(control)
    cost += 25.0 * ca.sumsqr(state[2 * no_masses : 2 * no_masses + 2] - ca.vertcat(1.0, 0.0))
    cost += ca.sumsqr(state[2 * (no_masses + 1) :]) + 0.01 * ca.sumsqr(control)
  rows, lower, upper, equality = [], [], [], []
  for k in range(horizon):
    local_eq = ca.vertcat(gaps[k], z[:nx] - p) if k == 0 else gaps[k]
    rows.extend([local_eq, controls[k]])
    lower.extend([0.0] * local_eq.numel() + [-1.0] * nu)
    upper.extend([0.0] * local_eq.numel() + [1.0] * nu)
    equality.extend([True] * local_eq.numel() + [False] * nu)
  return {"x": z, "p": p, "f": cost, "g": ca.vertcat(*rows)}, lower, upper, equality


def main() -> None:
  problem = build()
  x0, guess = initial_state(), initial_guess()
  fatrop = sc.solver(problem, "fatrop", options={"tol": 1e-10})
  ipopt = sc.solver(problem, "ipopt", options={"tol": 1e-10, "sb": "yes"})
  actual, reference = fatrop(x0, x0=guess)[0], ipopt(x0, x0=guess)[0]
  assert fatrop.stats().to_solver_status().ok and ipopt.stats().to_solver_status().ok
  np.testing.assert_allclose(actual, reference, atol=1e-8, rtol=0)
  ca_problem, lower, upper, equality = casadi_problem()
  ca_solver = ca.nlpsol(
    "paper_chain",
    "fatrop",
    ca_problem,
    {
      "expand": True,
      "print_time": False,
      "structure_detection": "auto",
      "equality": equality,
      "fatrop": {"print_level": 0, "tol": 1e-10},
    },
  )
  ca_result = ca_solver(x0=guess, p=x0, lbg=lower, ubg=upper)
  assert ca_solver.stats()["success"]
  ca_iterations = ca_solver.stats()["iter_count"]
  assert fatrop.stats().iter == ca_iterations
  np.testing.assert_allclose(actual, np.asarray(ca_result["x"]).ravel(), atol=1e-8, rtol=0)
  print(f"Scaly/Fatrop {fatrop.stats().iter}, CasADi/Fatrop {ca_iterations}, IPOPT {ipopt.stats().iter} iterations")
  print(f"Maximum solution difference from IPOPT: {np.max(np.abs(actual - reference)):.3e}")


if __name__ == "__main__":
  main()
