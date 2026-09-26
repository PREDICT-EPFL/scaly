"""Van der Pol optimal control by direct multiple shooting, RK4 integrator (Scaly).

The problem of ``direct_single_shooting``; now the states at the interval boundaries are variables
too, and continuity is imposed as equality constraints (the "gaps"). The variables are laid out as
the CasADi original lays them out, [x_0, u_0, x_1, u_1, ..., x_N], so one ``vmap`` with stride 3
reads every interval's (x_k, u_k, x_k+1).

After casadi/docs/examples/python/direct_multiple_shooting.py (Joel Andersson, 2016).
"""

import numpy as np

import scaly as sc
from _common import scaly_ipopt_options, show

T = 10.0  # horizon
N = 20  # control intervals
M = 4  # RK4 steps per interval
NW = 3 * N + 2  # [x_0, u_0, x_1, u_1, ..., x_N]
LBW = np.tile([-0.25, -np.inf, -1.0], N + 1)[:NW]
UBW = np.tile([np.inf, np.inf, 1.0], N + 1)[:NW]
LBW[:2] = UBW[:2] = [0.0, 1.0]  # x_0 fixed
W0 = np.zeros(NW)
W0[:2] = [0.0, 1.0]


@sc.function(sc.G(sc.L("x", 2), sc.L("u", 1)), sc.G(sc.L("xdot", ...), sc.L("L", ...)))
def f(inputs):
  x, u = inputs
  x1, x2 = x[0], x[1]
  return sc.stack([(1 - x2**2) * x1 - x2 + u[0], x1]), sc.stack([x1**2 + x2**2 + u[0] ** 2])


@sc.function(sc.G(sc.L("x0", 2), sc.L("p", 1)), sc.G(sc.L("xf", ...), sc.L("qf", ...)))
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


@sc.function(sc.G(sc.L("x", 2), sc.L("u", 1), sc.L("xnext", 2)), sc.L("gap_and_q", ...))
def interval(inputs):
  x, u, xnext = inputs
  xf, qf = F((x, u))
  return sc.concat([xf - xnext, qf])


@sc.problem(vars=sc.L("w", NW))
def multiple_shooting(w):
  out = sc.vmap(interval, N, [(w, 0, 3), (w, 2, 3), (w, 3, 3)]).reshape((N, 3))
  return sc.ProblemSpec(minimize=out[:, 2].sum(), eq=(out[:, 0:2].reshape((2 * N,)),), lb=sc.const(LBW), ub=sc.const(UBW))


def build(verbose: bool = False):
  solve = sc.solver(multiple_shooting, "ipopt", options=scaly_ipopt_options(verbose))

  def run():
    w, _, lam_g, _ = solve((W0, np.zeros(NW), np.zeros(2 * N), np.zeros(0), ()))
    stats = solve.solver_stats()
    return {"f": np.array([stats.obj]), "x1": w[0::3], "x2": w[1::3], "u": w[2::3], "lam_g": lam_g, "iter": np.array([stats.iter])}

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
