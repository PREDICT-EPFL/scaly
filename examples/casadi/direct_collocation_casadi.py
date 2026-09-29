# /// script
# requires-python = ">=3.12"
# dependencies = ["casadi"]
#
# [tool.ty.environment]
# extra-paths = ["."]  # the modules beside this script, which it imports
# ///
"""Van der Pol optimal control by direct collocation, Legendre points of degree 3 (CasADi).

The problem of ``direct_single_shooting``; on each of the N intervals the state is a degree-3
polynomial through the interval's start and three collocation points, all of them variables.

After casadi/docs/examples/python/direct_collocation.py (Joel Andersson, 2016).
"""

import casadi as ca
import numpy as np

from _common import as_arrays, casadi_ipopt_options, casadi_jit, show

D_DEG = 3  # polynomial degree
T = 10.0
N = 20  # control intervals


def collocation_coefficients(d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """C: derivative at the collocation points, D: value at the interval end, B: quadrature weights."""
  tau_root = np.append(0, ca.collocation_points(d, "legendre"))
  C, D, B = np.zeros((d + 1, d + 1)), np.zeros(d + 1), np.zeros(d + 1)
  for j in range(d + 1):
    p = np.poly1d([1])
    for r in range(d + 1):
      if r != j:
        p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])
    D[j] = p(1.0)
    pder = np.polyder(p)
    for r in range(d + 1):
      C[j, r] = pder(tau_root[r])
    B[j] = np.polyint(p)(1.0)
  return C, D, B


def build(verbose: bool = False):
  C, D, B = collocation_coefficients(D_DEG)
  x1, x2, u = ca.SX.sym("x1"), ca.SX.sym("x2"), ca.SX.sym("u")
  x = ca.vertcat(x1, x2)
  f = ca.Function("f", [x, u], [ca.vertcat((1 - x2**2) * x1 - x2 + u, x1), x1**2 + x2**2 + u**2], ["x", "u"], ["xdot", "L"])
  h = T / N

  w, w0, lbw, ubw, g, lbg, ubg, x_plot, u_plot = [], [], [], [], [], [], [], [], []
  J = 0
  Xk = ca.MX.sym("X0", 2)
  w.append(Xk)
  lbw.append([0, 1])
  ubw.append([0, 1])
  w0.append([0, 1])
  x_plot.append(Xk)
  for k in range(N):
    Uk = ca.MX.sym("U_" + str(k))
    w.append(Uk)
    lbw.append([-1])
    ubw.append([1])
    w0.append([0])
    u_plot.append(Uk)
    Xc = []
    for j in range(D_DEG):
      Xkj = ca.MX.sym("X_" + str(k) + "_" + str(j), 2)
      Xc.append(Xkj)
      w.append(Xkj)
      lbw.append([-0.25, -np.inf])
      ubw.append([np.inf, np.inf])
      w0.append([0, 0])
    Xk_end = D[0] * Xk
    for j in range(1, D_DEG + 1):
      xp = C[0, j] * Xk
      for r in range(D_DEG):
        xp = xp + C[r + 1, j] * Xc[r]
      fj, qj = f(Xc[j - 1], Uk)
      g.append(h * fj - xp)
      lbg.append([0, 0])
      ubg.append([0, 0])
      Xk_end = Xk_end + D[j] * Xc[j - 1]
      J = J + B[j] * qj * h
    Xk = ca.MX.sym("X_" + str(k + 1), 2)
    w.append(Xk)
    lbw.append([-0.25, -np.inf])
    ubw.append([np.inf, np.inf])
    w0.append([0, 0])
    x_plot.append(Xk)
    g.append(Xk_end - Xk)
    lbg.append([0, 0])
    ubg.append([0, 0])

  w = ca.vertcat(*w)
  solver = ca.nlpsol("solver", "ipopt", {"f": J, "x": w, "g": ca.vertcat(*g)}, {**casadi_ipopt_options(verbose), **casadi_jit()})
  trajectories = ca.Function("trajectories", [w], [ca.horzcat(*x_plot), ca.horzcat(*u_plot)], ["w"], ["x", "u"])
  w0, lbw, ubw, lbg, ubg = (np.concatenate(v) for v in (w0, lbw, ubw, lbg, ubg))

  def run():
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    x_opt, u_opt = trajectories(sol["x"])
    return as_arrays({"f": sol["f"], "x1": x_opt[0, :], "x2": x_opt[1, :], "u": u_opt, "iter": solver.stats()["iter_count"]})

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
