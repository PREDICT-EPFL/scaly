# /// script
# requires-python = ">=3.12"
# dependencies = ["casadi"]
#
# [tool.ty.environment]
# extra-paths = ["."]  # the modules beside this script, which it imports
# ///
"""Identification of a Hammerstein model from 2 000 samples by single shooting (CasADi).

    v[k] = N(u[k]),   y[k+1] = a1 y[k] + a2 y[k-1] + v[k],   N a cubic B-spline with 20 coefficients c

The static nonlinearity is CasADi's ``bspline`` node with MX coefficients, which must be built with
``inline=True``: the node's derivative with respect to its coefficients is otherwise zero. The
2 000 steps are one ``mapaccum``. IPOPT fits ``(a1, a2, c)`` to the measured output.
"""

import casadi as ca
import numpy as np

from _common import as_arrays, casadi_ipopt_options, casadi_jit, show
from _data import HAMMERSTEIN_KNOTS, hammerstein_data


def build(verbose: bool = False):
  data = hammerstein_data()
  n = data["u"].size
  x, u, a, c = ca.MX.sym("x", 2), ca.MX.sym("u"), ca.MX.sym("a", 2), ca.MX.sym("c", 20)
  v = ca.bspline(u, c, [list(HAMMERSTEIN_KNOTS)], [3], 1, {"inline": True})
  step = ca.Function("step", [x, u, a, c], [ca.vertcat(a[0] * x[0] + a[1] * x[1] + v, x[0])])
  simulate = step.mapaccum("simulate", n)
  theta = ca.vertcat(a, c)
  states = simulate(ca.DM.zeros(2), data["u"].reshape(1, -1), ca.repmat(a, 1, n), ca.repmat(c, 1, n))
  residual = states[0, :].T - data["y"]
  solver = ca.nlpsol("sysid", "ipopt", {"x": theta, "f": ca.sumsqr(residual)}, {**casadi_ipopt_options(verbose), **casadi_jit(expand=False)})
  guess = np.concatenate([data["a_guess"], data["c_guess"]])

  def run():
    sol = solver(x0=guess)
    theta = sol["x"].full().reshape(-1)
    return as_arrays({"a": theta[:2], "c": theta[2:], "f": sol["f"], "iter": solver.stats()["iter_count"]})

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
