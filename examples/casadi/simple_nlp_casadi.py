# /// script
# requires-python = ">=3.12"
# dependencies = ["casadi"]
#
# [tool.ty.environment]
# extra-paths = ["."]  # the modules beside this script, which it imports
# ///
"""The smallest NLP: the point on the line x0 + x1 = 10 closest to the origin (CasADi).

    minimize  x0^2 + x1^2   subject to  x0 + x1 - 10 >= 0

After casadi/docs/examples/python/simple_nlp.py (Joel Andersson, 2015).
"""

import casadi as ca

from _common import as_arrays, casadi_ipopt_options, casadi_jit, show


def build(verbose: bool = False):
  x = ca.SX.sym("x", 2)
  nlp = {"x": x, "f": x[0] ** 2 + x[1] ** 2, "g": x[0] + x[1] - 10}
  solver = ca.nlpsol("solver", "ipopt", nlp, {**casadi_ipopt_options(verbose), **casadi_jit()})

  def run():
    sol = solver(lbg=0)
    return as_arrays({"f": sol["f"], "x": sol["x"], "lam_x": sol["lam_x"], "lam_g": sol["lam_g"], "iter": solver.stats()["iter_count"]})

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
