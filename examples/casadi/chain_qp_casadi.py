# /// script
# requires-python = ">=3.12"
# dependencies = ["casadi"]
#
# [tool.ty.environment]
# extra-paths = ["."]  # the modules beside this script, which it imports
# ///
"""A hanging chain of N masses resting on a sloped floor: a sparse QP, solved by qpOASES (CasADi).

    minimize    sum_i D/2 ((y_i - y_{i+1})^2 + (z_i - z_{i+1})^2) + g0 sum_i m z_i
    subject to  z_i >= 0.5,  z_i - 0.1 y_i >= 0.5,  both ends fixed

After casadi/docs/examples/python/chain_qp.py.
"""

import casadi as ca
from numpy import inf

from _common import as_arrays, show

N = 40
M_I = 40.0 / N
D_I = 70.0 * N
G0 = 9.81
ZMIN = 0.5


def build(verbose: bool = False):
  x, lbx, ubx, g, lbg, ubg = [], [], [], [], [], []
  vchain = 0
  for i in range(1, N + 1):
    if i > 1:
      y_prev, z_prev = y_i, z_i  # noqa: F821 (bound on the previous pass)
    y_i = ca.SX.sym("y_" + str(i))
    z_i = ca.SX.sym("z_" + str(i))
    x += [y_i, z_i]
    if i == 1:
      lbx += [-2.0, 1.0]
      ubx += [-2.0, 1.0]
    elif i == N:
      lbx += [2.0, 1.0]
      ubx += [2.0, 1.0]
    else:
      lbx += [-inf, ZMIN]
      ubx += [inf, inf]
    if i > 1:
      vchain += D_I / 2 * ((y_prev - y_i) ** 2 + (z_prev - z_i) ** 2)
    vchain += G0 * M_I * z_i
    g.append(z_i - 0.1 * y_i)
    lbg.append(0.5)
    ubg.append(inf)
  qp = {"x": ca.vertcat(*x), "f": vchain, "g": ca.vertcat(*g)}
  solver = ca.qpsol("solver", "qpoases", qp, {"sparse": True, "printLevel": "tabular" if verbose else "none", "print_time": verbose})

  def run():
    sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    return as_arrays({"f": sol["f"], "y": sol["x"][0::2], "z": sol["x"][1::2]})

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
