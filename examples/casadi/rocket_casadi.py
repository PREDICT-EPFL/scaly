# /// script
# requires-python = ">=3.12"
# dependencies = ["casadi"]
#
# [tool.ty.environment]
# extra-paths = ["."]  # the modules beside this script, which it imports
# ///
"""Minimum-effort rocket flight: reach position 10 at rest after 50 thrust intervals (CasADi).

Each interval is 20 explicit Euler steps of s' = v, v' = (u - 0.05 v^2) / m, m' = -0.1 u^2; the
controls are the only variables (single shooting).

After casadi/docs/examples/python/rocket.py.
"""

import casadi as ca

from _common import as_arrays, casadi_ipopt_options, casadi_jit, show

NU = 50  # control intervals
DT = 0.01  # Euler step, 20 per interval


def build(verbose: bool = False):
  u = ca.MX.sym("u")
  x = ca.MX.sym("x", 3)
  v, m = x[1], x[2]  # x = (position, speed, mass)
  f = ca.Function("f", [x, u], [ca.vertcat(v, (u - 0.05 * v * v) / m, -0.1 * u * u)])
  xj = x
  for _ in range(20):
    xj += DT * f(xj, u)
  F = ca.Function("F", [x, u], [xj])

  U = ca.MX.sym("U", NU)
  X = ca.MX([0, 0, 1])
  for k in range(NU):
    X = F(X, U[k])
  nlp = {"x": U, "f": U.T @ U, "g": X[0:2]}
  solver = ca.nlpsol("solver", "ipopt", nlp, {**casadi_ipopt_options(verbose, tol=1e-10), "expand": True, **casadi_jit()})

  def run():
    res = solver(lbx=-0.5, ubx=0.5, x0=0.4, lbg=[10, 0], ubg=[10, 0])
    return as_arrays({"f": res["f"], "u": res["x"], "lam_u": res["lam_x"], "lam_g": res["lam_g"], "iter": solver.stats()["iter_count"]})

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
