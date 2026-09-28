"""The smallest NLP: the point on the line x0 + x1 = 10 closest to the origin (Scaly).

    minimize  x0^2 + x1^2   subject to  x0 + x1 - 10 >= 0

After casadi/docs/examples/python/simple_nlp.py (Joel Andersson, 2015).
"""

import numpy as np

import scaly as sc
from _common import scaly_ipopt_options, show


@sc.problem(vars=sc.L("x", 2))
def simple_nlp(x):
  return sc.ProblemSpec(minimize=x[0] ** 2 + x[1] ** 2, ineq=(sc.bounded(x[0] + x[1] - 10, lo=0.0),))


def build(verbose: bool = False):
  solve = sc.solver(simple_nlp, "ipopt", options=scaly_ipopt_options(verbose))

  def run():
    x, lam_x, _, lam_g = solve(np.zeros(2), np.zeros(2), np.zeros(0), np.zeros(1), ())
    stats = sc.solver_stats(solve)
    return {"f": np.array([stats.obj]), "x": x, "lam_x": lam_x, "lam_g": lam_g, "iter": np.array([stats.iter])}

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
