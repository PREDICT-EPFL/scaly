# /// script
# requires-python = ">=3.12"
# dependencies = ["casadi"]
#
# [tool.ty.environment]
# extra-paths = ["."]  # the modules beside this script, which it imports
# ///
"""Van der Pol optimal control by direct multiple shooting, RK4 integrator (CasADi).

The problem of ``direct_single_shooting``; now the states at the interval boundaries are variables
too, and continuity is imposed as equality constraints (the "gaps").

After casadi/docs/examples/python/direct_multiple_shooting.py (Joel Andersson, 2016).
"""

import casadi as ca
from numpy import inf

from _common import as_arrays, casadi_ipopt_options, casadi_jit, show

T = 10.0  # horizon
N = 20  # control intervals
M = 4  # RK4 steps per interval


def integrator():
  x1, x2, u = ca.MX.sym("x1"), ca.MX.sym("x2"), ca.MX.sym("u")
  x = ca.vertcat(x1, x2)
  f = ca.Function("f", [x, u], [ca.vertcat((1 - x2**2) * x1 - x2 + u, x1), x1**2 + x2**2 + u**2])
  dt = T / N / M
  X0, U = ca.MX.sym("X0", 2), ca.MX.sym("U")
  X, Q = X0, 0
  for _ in range(M):
    k1, k1_q = f(X, U)
    k2, k2_q = f(X + dt / 2 * k1, U)
    k3, k3_q = f(X + dt / 2 * k2, U)
    k4, k4_q = f(X + dt * k3, U)
    X = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    Q = Q + dt / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
  return ca.Function("F", [X0, U], [X, Q], ["x0", "p"], ["xf", "qf"])


def build(verbose: bool = False):
  F = integrator()
  w, w0, lbw, ubw, g, lbg, ubg = [], [], [], [], [], [], []
  J = 0
  Xk = ca.MX.sym("X0", 2)
  w += [Xk]
  lbw += [0, 1]
  ubw += [0, 1]
  w0 += [0, 1]
  for k in range(N):
    Uk = ca.MX.sym("U_" + str(k))
    w += [Uk]
    lbw += [-1]
    ubw += [1]
    w0 += [0]
    Fk = F(x0=Xk, p=Uk)
    J = J + Fk["qf"]
    Xk = ca.MX.sym("X_" + str(k + 1), 2)
    w += [Xk]
    lbw += [-0.25, -inf]
    ubw += [inf, inf]
    w0 += [0, 0]
    g += [Fk["xf"] - Xk]
    lbg += [0, 0]
    ubg += [0, 0]
  solver = ca.nlpsol("solver", "ipopt", {"f": J, "x": ca.vertcat(*w), "g": ca.vertcat(*g)}, {**casadi_ipopt_options(verbose), **casadi_jit()})

  def run():
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol["x"].full().flatten()
    return as_arrays(
      {"f": sol["f"], "x1": w_opt[0::3], "x2": w_opt[1::3], "u": w_opt[2::3], "lam_g": sol["lam_g"], "iter": solver.stats()["iter_count"]}
    )

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
