# /// script
# requires-python = ">=3.12"
# dependencies = ["casadi"]
#
# [tool.ty.environment]
# extra-paths = ["."]  # the modules beside this script, which it imports
# ///
"""A 12 x 12 lookup table calibrated to 3 000 scattered noisy measurements by least squares (CasADi).

    minimize over T   sum_i (T(p_i) - z_i)^2,   T bilinear on a 12 x 12 grid with values T

The table's values are the decision variables: a parametric ``interpolant`` (its data an input)
mapped over the measurement points, solved by IPOPT. The derivative of a parametric interpolant with
respect to its data is zero unless it is built with ``inline=True``, so ``build`` uses the inlined
one. ``default_derivative_run`` solves once with the default and reports what IPOPT does with it;
running this file prints it.
"""

import casadi as ca
import numpy as np

from _common import as_arrays, casadi_ipopt_options, casadi_jit, show
from _data import calibration_data

def least_squares(grid, points, measured, opts, verbose):
  table = ca.interpolant("table", "linear", [list(grid), list(grid)], 1, opts)
  T = ca.MX.sym("T", grid.size**2)  # CasADi's data order: the first axis varies fastest
  p = ca.MX.sym("p", 2)
  point = ca.Function("point", [p, T], [table(p, T)])
  residual = point.map(points.shape[0])(points.T, T).T - measured
  nlp = {"x": T, "f": ca.sumsqr(residual)}
  return ca.nlpsol("calibrate", "ipopt", nlp, {**casadi_ipopt_options(verbose), **casadi_jit(expand=False)})


def default_derivative_run(verbose: bool = False) -> dict[str, object]:
  """The same fit without ``inline=True``: IPOPT's iterations, status, and how far the table moved."""
  grid, points, measured, _ = calibration_data()
  default = least_squares(grid, points, measured, {}, verbose)
  sol = default(x0=np.zeros(grid.size**2))
  stats = default.stats()
  return {"iterations": stats["iter_count"], "status": stats["return_status"], "moved": float(np.abs(sol["x"].full()).max())}


def build(verbose: bool = False):
  grid, points, measured, _ = calibration_data()
  n = grid.size
  solver = least_squares(grid, points, measured, {"inline": True}, verbose)

  def run():
    sol = solver(x0=np.zeros(n * n))
    table = sol["x"].full().reshape((n, n), order="F")
    return as_arrays({"table": table, "f": sol["f"], "iter": solver.stats()["iter_count"]})

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
  print(f"without inline=True: {default_derivative_run()}")
