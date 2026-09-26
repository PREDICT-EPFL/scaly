"""Van der Pol optimal control by direct single shooting, RK4 integrator (CasADi).

    minimize  int_0^10 x1^2 + x2^2 + u^2 dt
    subject to  x1' = (1 - x2^2) x1 - x2 + u,  x2' = x1,  x(0) = (0, 1),
                -1 <= u <= 1,  x1 >= -0.25 at the end of every interval

The controls on N = 20 intervals are the only variables; each interval is 4 RK4 steps.

After casadi/docs/examples/python/direct_single_shooting.py (Joel Andersson, 2016).
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
  Xk = ca.MX([0, 1])
  for k in range(N):
    Uk = ca.MX.sym("U_" + str(k))
    w += [Uk]
    lbw += [-1]
    ubw += [1]
    w0 += [0]
    Fk = F(x0=Xk, p=Uk)
    Xk = Fk["xf"]
    J = J + Fk["qf"]
    g += [Xk[0]]
    lbg += [-0.25]
    ubg += [inf]
  solver = ca.nlpsol("solver", "ipopt", {"f": J, "x": ca.vertcat(*w), "g": ca.vertcat(*g)}, {**casadi_ipopt_options(verbose), **casadi_jit()})

  def run():
    test = F(x0=[0.2, 0.3], p=0.4)  # the original's check of one interval
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    x_opt = [ca.DM([0, 1])]
    for k in range(N):
      x_opt.append(F(x0=x_opt[-1], p=sol["x"][k])["xf"])
    x_opt = ca.horzcat(*x_opt)
    return as_arrays(
      {
        "xf_test": test["xf"],
        "qf_test": test["qf"],
        "f": sol["f"],
        "u": sol["x"],
        "x1": x_opt[0, :],
        "x2": x_opt[1, :],
        "iter": solver.stats()["iter_count"],
      }
    )

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
