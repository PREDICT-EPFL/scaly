"""Rosenbrock's problem as an equality-constrained NLP, solved by IPOPT (CasADi).

    minimize  x^2 + 100 z^2   subject to  z + (1 - x)^2 - y = 0

After casadi/docs/examples/python/rosenbrock.py (Joel Andersson, 2015).
"""

import casadi as ca

from _common import as_arrays, casadi_ipopt_options, casadi_jit, show


def build(verbose: bool = False):
  x, y, z = ca.SX.sym("x"), ca.SX.sym("y"), ca.SX.sym("z")
  nlp = {"x": ca.vertcat(x, y, z), "f": x**2 + 100 * z**2, "g": z + (1 - x) ** 2 - y}
  solver = ca.nlpsol("solver", "ipopt", nlp, {**casadi_ipopt_options(verbose), **casadi_jit()})

  def run():
    res = solver(x0=[2.5, 3.0, 0.75], lbg=0, ubg=0)
    return as_arrays({"f": res["f"], "x": res["x"], "lam_x": res["lam_x"], "lam_g": res["lam_g"], "iter": solver.stats()["iter_count"]})

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
