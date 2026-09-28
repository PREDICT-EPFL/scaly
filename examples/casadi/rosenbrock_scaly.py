"""Rosenbrock's problem as an equality-constrained NLP, solved by IPOPT (Scaly).

    minimize  x^2 + 100 z^2   subject to  z + (1 - x)^2 - y = 0

After casadi/docs/examples/python/rosenbrock.py (Joel Andersson, 2015).
"""

import numpy as np

import scaly as sc
from _common import scaly_ipopt_options, show


@sc.opt.problem(vars=sc.L("w", 3))
def rosenbrock(w):
  x, y, z = w[0], w[1], w[2]
  return sc.opt.ProblemSpec(minimize=x**2 + 100 * z**2, eq=(z + (1 - x) ** 2 - y,))


def build(verbose: bool = False):
  solve = sc.opt.solver(rosenbrock, sc.opt.IPOPT(options=scaly_ipopt_options(verbose)))

  def run():
    w, lam_w, lam_eq, _, _ = solve(np.array([2.5, 3.0, 0.75]), np.zeros(3), np.zeros(1), np.zeros(0), ())
    stats = sc.opt.solver_stats(solve)
    return {"f": np.array([stats.obj]), "x": w, "lam_x": lam_w, "lam_g": lam_eq, "iter": np.array([stats.iter])}

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
