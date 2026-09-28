"""Hock-Schittkowski problem 15: Rosenbrock's valley cut by two nonconvex constraints (Scaly).

    minimize  100 (x1 - x0^2)^2 + (1 - x0)^2
    subject to  x0 x1 >= 1,  x0 + x1^2 >= 0,  x0 <= 1/2

The CasADi original solves it with Uno in its IPOPT preset; Scaly has no Uno backend and uses IPOPT.

After casadi/docs/examples/python/hs015.py (David Kiessling, 2026).
"""

import numpy as np

import scaly as sc
from _common import scaly_ipopt_options, show


@sc.problem(vars=sc.L("x", 2))
def hs015(x):
  return sc.ProblemSpec(
    minimize=100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2,
    ineq=(sc.bounded(sc.stack([x[0] * x[1], x[0] + x[1] ** 2]), lo=sc.const(np.array([1.0, 0.0]))),),
    ub=sc.const(np.array([0.5, np.inf])),
  )


def build(verbose: bool = False):
  solve = sc.solver(hs015, "ipopt", options=scaly_ipopt_options(verbose))

  def run():
    x, lam_x, _, lam_g = solve(np.array([-2.0, 1.0]), np.zeros(2), np.zeros(0), np.zeros(2), ())
    return {"f": np.array([sc.solver_stats(solve).obj]), "x": x, "lam_x": lam_x, "lam_g": lam_g}

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
