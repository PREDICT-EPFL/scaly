"""A two-variable linear program, solved by PIQP through Scaly's matrix-data QP (Scaly).

    minimize  3 x0 + 4 x1   subject to  x0 + 2 x1 <= 14,  3 x0 - x1 >= 0,  x0 - x1 <= 2

``sc.qp_problem`` is the counterpart of CasADi's ``conic``: the matrices are parameters, so one
generated solver takes any data of these dimensions. The Hessian is zero.

After casadi/docs/examples/python/simple_lp.py (Joel Andersson, 2015).
"""

import numpy as np

import scaly as sc
from _common import show


def build(verbose: bool = False):
  solve = sc.solver(sc.qp_problem(2, 0, 3), "piqp", options={"verbose": verbose})
  g = np.array([3.0, 4.0])
  a = np.array([[1.0, 2.0], [3.0, -1.0], [1.0, -1.0]])
  lba = np.array([-np.inf, 0.0, -np.inf])
  uba = np.array([14.0, np.inf, 2.0])
  data = ((np.zeros((2, 2)), g), (np.zeros((0, 2)), np.zeros(0)), (a, lba, uba))

  def run():
    x, _, _, lam_a = solve((np.zeros(2), np.zeros(2), np.zeros(0), np.zeros(3), data))
    return {"f": np.array([g @ x]), "x": x, "lam_a": lam_a}

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
