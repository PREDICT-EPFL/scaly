"""Moving-horizon estimation of a spring-damper from noisy position measurements (Scaly).

A mass on a spring and damper, driven by a known force and an unknown process noise w, is observed
through noisy positions. Every step, an NLP over the last N = 10 samples estimates the states and
the noise; the arrival cost is updated by an extended Kalman filter step, which needs the Jacobians
of the RK4 step and of the measurement. The window's dynamics are one ``vmap`` of the RK4 step.

After casadi/docs/examples/python/mhe_spring_damper.py.
"""

import numpy as np
from scipy import linalg

import scaly as sc
from scaly import integrators as si
from _common import scaly_ipopt_options, show

N = 10  # horizon
DT = 0.05
SIGMA_P, SIGMA_W = 0.005, 0.1  # measurement and process noise standard deviations
R, Q = 1 / SIGMA_P**2, 1 / SIGMA_W**2
N_SIM = 1000
M, K, C = 1.0, 1.0, 0.5  # mass, spring constant, damping


@sc.function(2, 1, 1, output="xdot")
def f(x, u, w):
  return sc.stack([x[1], (-K * x[0] - C * x[1] + u[0]) / M + w[0]])


phi = si.rk4(f, dt=DT, name="phi")  # phi(x, u, w) -> xnext


@sc.function(2, output="y")
def h(x):
  return x[0:1]


PHI = sc.jacobian(phi, "x")
H = sc.jacobian(h, "x")


@sc.function(2, 1, 1, 2)
def gap(x, u, w, xnext):
  return xnext - phi(x, u, w)


@sc.opt.problem(
  vars=sc.G(sc.L("X", 2 * N), sc.L("W", N - 1)),
  params=sc.G(sc.L("U", N - 1), sc.L("Y", N), sc.L("S", (2, 2)), sc.L("x0", 2)),
)
def mhe(variables, params):
  X, W = variables
  U, Y, S, x0 = params
  e0 = X[0:2] - x0
  obj = sc.dot(e0, S @ e0) + R * sc.sumsqr(X.reshape((N, 2))[:, 0] - Y) + Q * sc.sumsqr(W)
  gaps = sc.vmap(gap, N - 1, [(X, 0, 2), (U, 0, 1), (W, 0, 1), (X, 2, 2)])
  return sc.opt.ProblemSpec(minimize=obj, eq=(gaps,))


def build(verbose: bool = False):
  solve = sc.opt.solver(mhe, sc.opt.IPOPT(options=scaly_ipopt_options(False, max_iter=100)))

  # The simulated plant and its measurements, as in the original.
  np.random.seed(0)
  sim_X = np.zeros((N_SIM, 2))
  sim_X[0] = [1, 0]
  t = np.linspace(0, (N_SIM - 1) * DT, N_SIM)
  sim_U = np.cos(t[0:-1])
  sim_U[int(N_SIM / 2) :] = 0.0
  sim_W = SIGMA_W * np.random.randn(N_SIM - 1)
  for i in range(N_SIM - 1):
    sim_X[i + 1] = phi(sim_X[i], sim_U[i : i + 1], sim_W[i : i + 1])
  sim_Y = sim_X[:, 0] + SIGMA_P * np.random.randn(N_SIM)
  P0 = 0.01**2 * np.eye(2)
  x0_guess = sim_X[0] + 0.01 * np.random.randn(2)

  def run():
    P, x0 = P0, x0_guess
    est_X, est_W = np.zeros((N_SIM, 2)), np.zeros(N_SIM - 1)
    zeros = (np.zeros(2 * N), np.zeros(N - 1)), np.zeros(2 * (N - 1)), np.zeros(0)
    init_X, init_W = sim_X[0:N].copy(), np.zeros(N - 1)
    U, Y = sim_U[0 : N - 1], sim_Y[0:N]
    (X, W), *_ = solve((init_X.reshape(-1), init_W), *zeros, (U, Y, linalg.inv(P), x0))
    X = X.reshape(N, 2)
    est_X[0:N], est_W[0 : N - 1] = X, W
    iterations = sc.opt.solver_stats(solve).iter
    for i in range(1, N_SIM - N + 1):
      # EKF update of the arrival cost
      H0 = H(X[0])
      Kk = P @ H0.T @ linalg.inv(H0 @ P @ H0.T + R)
      P = (np.eye(2) - Kk @ H0) @ P
      x0 = x0 + Kk @ (Y[0:1] - h(X[0]) - H0 @ (x0 - X[0]))
      x0 = phi(x0, U[0:1], W[0:1])
      Fk = PHI(X[0], U[0:1], W[0:1])
      P = Fk @ P @ Fk.T + 1 / Q
      # shift the window and warm start from the previous estimate
      U, Y = sim_U[i : i + N - 1], sim_Y[i : i + N]
      init_W[0 : N - 2], init_W[N - 2] = est_W[i : i + N - 2], 0.0
      init_X[0 : N - 1] = est_X[i : i + N - 1]
      init_X[N - 1] = phi(init_X[N - 1], U[-1:], init_W[-1:])
      (X, W), *_ = solve((init_X.reshape(-1), init_W), *zeros, (U, Y, linalg.inv(P), x0))
      X = X.reshape(N, 2)
      est_X[N - 1 + i], est_W[N - 2 + i] = X[N - 1], W[N - 2]
      iterations += sc.opt.solver_stats(solve).iter
    error = est_X[:, 0] - sim_X[:, 0]
    return {"x_est": est_X[:, 0], "dx_est": est_X[:, 1], "error_sq": np.array([error @ error]), "iter_total": np.array([iterations])}

  return run


if __name__ == "__main__":
  out = build(verbose=True)()
  show(out)
  assert out["error_sq"] < 0.01
