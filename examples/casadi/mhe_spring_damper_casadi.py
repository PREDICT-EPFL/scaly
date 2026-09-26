"""Moving-horizon estimation of a spring-damper from noisy position measurements (CasADi).

A mass on a spring and damper, driven by a known force and an unknown process noise w, is observed
through noisy positions. Every step, an NLP over the last N = 10 samples estimates the states and
the noise; the arrival cost is updated by an extended Kalman filter step, which needs the Jacobians
of the RK4 step and of the measurement.

After casadi/docs/examples/python/mhe_spring_damper.py.
"""

import casadi as ca
import numpy as np
from casadi.tools import entry, struct_SX, struct_symSX
from scipy import linalg

from _common import as_arrays, casadi_ipopt_options, casadi_jit, show

N = 10  # horizon
DT = 0.05
SIGMA_P, SIGMA_W = 0.005, 0.1  # measurement and process noise standard deviations
R, Q = 1 / SIGMA_P**2, 1 / SIGMA_W**2
N_SIM = 1000
M, K, C = 1.0, 1.0, 0.5  # mass, spring constant, damping


def build(verbose: bool = False):
  states = struct_symSX(["x", "dx"])
  x, dx = states[...]
  controls = struct_symSX(["F"])
  (F,) = controls[...]
  disturbances = struct_symSX(["w"])
  (w,) = disturbances[...]
  shooting = struct_symSX([(entry("X", repeat=N, struct=states), entry("W", repeat=N - 1, struct=disturbances))])
  parameters = struct_symSX(
    [(entry("U", repeat=N - 1, struct=controls), entry("Y", repeat=N, shape=1), entry("S", shape=(2, 2)), entry("x0", shape=(2, 1)))]
  )

  rhs = struct_SX(states)
  rhs["x"] = dx
  rhs["dx"] = (-K * x - C * dx + F) / M + w
  f = ca.Function("f", [states, controls, disturbances], [rhs])
  k1 = f(states, controls, disturbances)
  k2 = f(states + DT / 2.0 * k1, controls, disturbances)
  k3 = f(states + DT / 2.0 * k2, controls, disturbances)
  k4 = f(states + DT * k3, controls, disturbances)
  phi = ca.Function("phi", [states, controls, disturbances], [states + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)], ["x", "u", "d"], ["x1"])
  PHI = phi.factory("PHI", ["x", "u", "d"], ["jac:x1:x"])
  h = ca.Function("h", [states], [x], ["x"], ["y"])
  H = h.factory("H", ["x"], ["jac:y:x"])

  e0 = shooting["X", 0] - parameters["x0"]
  obj = ca.mtimes([e0.T, parameters["S"], e0])
  for i in range(N):
    vm = h(shooting["X", i]) - parameters["Y", i]
    obj += vm.T * R * vm
  for i in range(N - 1):
    obj += shooting["W", i].T * Q * shooting["W", i]
  g = [shooting["X", i + 1] - phi(shooting["X", i], parameters["U", i], shooting["W", i]) for i in range(N - 1)]
  nlp = {"x": shooting, "p": parameters, "f": obj, "g": ca.vertcat(*g)}
  solver = ca.nlpsol("nlpsol", "ipopt", nlp, {**casadi_ipopt_options(False, max_iter=100), **casadi_jit()})

  # The simulated plant and its measurements, as in the original.
  np.random.seed(0)
  sim_X = ca.DM.zeros(2, N_SIM)
  sim_X[:, 0] = ca.DM([1, 0])
  t = np.linspace(0, (N_SIM - 1) * DT, N_SIM)
  sim_U = ca.DM(np.cos(t[0:-1])).T
  sim_U[:, int(N_SIM / 2) :] = 0.0
  sim_W = ca.DM(SIGMA_W * np.random.randn(1, N_SIM - 1))
  for i in range(N_SIM - 1):
    sim_X[:, i + 1] = phi(sim_X[:, i], sim_U[:, i], sim_W[:, i])
  sim_Y = ca.DM.zeros(1, N_SIM)
  for i in range(N_SIM):
    sim_Y[:, i] = h(sim_X[:, i])
  sim_Y += SIGMA_P * np.random.randn(1, N_SIM)
  P0 = 0.01**2 * ca.DM.eye(2)
  x0_guess = sim_X[:, 0] + 0.01 * np.random.randn(2, 1)

  def run():
    P, x0 = P0, x0_guess
    est_X, est_W = ca.DM.zeros(2, N_SIM), ca.DM.zeros(1, N_SIM - 1)
    p = parameters(0)
    p["U", lambda v: ca.horzcat(*v)] = sim_U[:, 0 : N - 1]
    p["Y", lambda v: ca.horzcat(*v)] = sim_Y[:, 0:N]
    p["S"] = linalg.inv(P)
    p["x0"] = x0
    init = shooting(0)
    init["X", lambda v: ca.horzcat(*v)] = sim_X[:, 0:N]
    sol = shooting(solver(p=p, x0=init, lbg=0, ubg=0)["x"])
    est_X[:, 0:N] = sol["X", lambda v: ca.horzcat(*v)]
    est_W[:, 0 : N - 1] = sol["W", lambda v: ca.horzcat(*v)]
    iterations = [solver.stats()["iter_count"]]
    for i in range(1, N_SIM - N + 1):
      # EKF update of the arrival cost
      H0 = H(sol["X", 0])
      Kk = ca.mtimes([P, H0.T, linalg.inv(ca.mtimes([H0, P, H0.T]) + R)])
      P = (ca.DM.eye(2) - Kk @ H0) @ P
      x0 = x0 + Kk @ (p["Y", 0] - h(sol["X", 0]) - H0 @ (x0 - sol["X", 0]))
      x0 = phi(x0, p["U", 0], sol["W", 0])
      Fk = PHI(sol["X", 0], p["U", 0], sol["W", 0])
      P = ca.mtimes([Fk, P, Fk.T]) + 1 / Q
      # shift the window and warm start from the previous estimate
      p["U", lambda v: ca.horzcat(*v)] = sim_U[:, i : i + N - 1]
      p["Y", lambda v: ca.horzcat(*v)] = sim_Y[:, i : i + N]
      p["S"] = linalg.inv(P)
      p["x0"] = x0
      init["W", lambda v: ca.horzcat(*v), 0 : N - 2] = est_W[:, i : i + N - 2]
      init["W", N - 2] = ca.DM.zeros(1, 1)
      init["X", lambda v: ca.horzcat(*v), 0 : N - 1] = est_X[:, i : i + N - 1]
      init["X", N - 1] = phi(init["X", N - 1], p["U", -1], init["W", -1])
      sol = shooting(solver(p=p, x0=init, lbg=0, ubg=0)["x"])
      est_X[:, N - 1 + i] = sol["X", N - 1]
      est_W[:, N - 2 + i] = sol["W", N - 2]
      iterations.append(solver.stats()["iter_count"])
    error = est_X[0, :] - sim_X[0, :]
    return as_arrays({"x_est": est_X[0, :], "dx_est": est_X[1, :], "error_sq": error @ error.T, "iter_total": sum(iterations)})

  return run


if __name__ == "__main__":
  out = build(verbose=True)()
  show(out)
  assert out["error_sq"] < 0.01
