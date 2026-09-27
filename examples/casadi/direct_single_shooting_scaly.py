"""Van der Pol optimal control by direct single shooting, RK4 integrator (Scaly).

    minimize  int_0^10 x1^2 + x2^2 + u^2 dt
    subject to  x1' = (1 - x2^2) x1 - x2 + u,  x2' = x1,  x(0) = (0, 1),
                -1 <= u <= 1,  x1 >= -0.25 at the end of every interval

The controls on N = 20 intervals are the only variables; each interval is 4 RK4 steps, and the
N intervals are a ``scan`` that stacks each interval's end state and cost.

After casadi/docs/examples/python/direct_single_shooting.py (Joel Andersson, 2016).
"""

import numpy as np

import scaly as sc
from _common import scaly_ipopt_options, show

T = 10.0  # horizon
N = 20  # control intervals
M = 4  # RK4 steps per interval
X_START = np.array([0.0, 1.0])


@sc.function(sc.G(sc.L("x", 2), sc.L("u", 1)), output=sc.G(sc.L("xdot", ...), sc.L("L", ...)))
def f(inputs):
  x, u = inputs
  x1, x2 = x[0], x[1]
  return sc.stack([(1 - x2**2) * x1 - x2 + u[0], x1]), sc.stack([x1**2 + x2**2 + u[0] ** 2])


@sc.function(sc.G(sc.L("x0", 2), sc.L("p", 1)), output=sc.G(sc.L("xf", ...), sc.L("qf", ...)))
def F(inputs):
  X, U = inputs
  dt = T / N / M
  Q = sc.const(np.zeros(1))
  for _ in range(M):
    k1, k1_q = f((X, U))
    k2, k2_q = f((X + dt / 2 * k1, U))
    k3, k3_q = f((X + dt / 2 * k2, U))
    k4, k4_q = f((X + dt * k3, U))
    X = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    Q = Q + dt / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
  return X, Q


@sc.function(sc.G(sc.L("x", 2), sc.L("u", 1)), output=sc.G(sc.L("xnext", ...), sc.L("x_and_q", ...)))
def interval(inputs):
  xf, qf = F(inputs)
  return xf, sc.concat([xf, qf])


@sc.function(sc.L("u", N), output=sc.L("x_and_q", ...))
def rollout(u):
  _, x_and_q = sc.scan(interval, sc.const(X_START), [(u, 0, 1)], length=N)
  return x_and_q.reshape((N, 3))


@sc.problem(vars=sc.L("u", N))
def single_shooting(u):
  x_and_q = rollout(u)
  return sc.ProblemSpec(minimize=x_and_q[:, 2].sum(), ineq=(sc.bounded(x_and_q[:, 0], lo=-0.25),), lb=sc.const(-1.0), ub=sc.const(1.0))


def build(verbose: bool = False):
  solve = sc.solver(single_shooting, "ipopt", options=scaly_ipopt_options(verbose))

  def run():
    xf_test, qf_test = F((np.array([0.2, 0.3]), np.array([0.4])))  # the original's check of one interval
    u, *_ = solve(np.zeros(N), np.zeros(N), np.zeros(0), np.zeros(N), ())
    stats = solve.solver_stats()
    x = np.vstack([X_START, rollout(u)[:, :2]])
    return {"xf_test": xf_test, "qf_test": qf_test, "f": np.array([stats.obj]), "u": u, "x1": x[:, 0], "x2": x[:, 1], "iter": np.array([stats.iter])}

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
