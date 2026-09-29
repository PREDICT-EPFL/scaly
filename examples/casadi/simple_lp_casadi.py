# /// script
# requires-python = ">=3.12"
# dependencies = ["casadi"]
#
# [tool.ty.environment]
# extra-paths = ["."]  # the modules beside this script, which it imports
# ///
"""A two-variable linear program, solved by qpOASES through CasADi's low-level ``conic`` interface.

    minimize  3 x0 + 4 x1   subject to  x0 + 2 x1 <= 14,  3 x0 - x1 >= 0,  x0 - x1 <= 2

After casadi/docs/examples/python/simple_lp.py (Joel Andersson, 2015).
"""

import casadi as ca
import numpy as np

from _common import as_arrays, show


def build(verbose: bool = False):
  solver = ca.conic("solver", "qpoases", {"a": ca.Sparsity.dense(3, 2)}, {"printLevel": "tabular" if verbose else "none"})
  g = ca.DM([3, 4])
  a = ca.DM([[1, 2], [3, -1], [1, -1]])
  lba = ca.DM([-np.inf, 0, -np.inf])
  uba = ca.DM([14, np.inf, 2])

  def run():
    sol = solver(g=g, a=a, lba=lba, uba=uba)
    return as_arrays({"f": sol["cost"], "x": sol["x"], "lam_a": sol["lam_a"]})

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
