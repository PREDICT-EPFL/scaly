"""A 12 x 12 lookup table calibrated to 3 000 scattered noisy measurements by least squares (Scaly).

    minimize over T   sum_i (T(p_i) - z_i)^2,   T bilinear on a 12 x 12 grid with values T

The table's values are the decision variables: ``interp.interpolant`` over an ``Expr``, read at
the measurement points with ``at()``. The points are known when the graph is built, so ``at()`` is
one sparse product of the basis there (four entries per row) with the table, and the residuals'
Jacobian has exactly that pattern. IPOPT solves it.
"""

import numpy as np

import scaly as sc
from _common import scaly_ipopt_options, show
from _data import calibration_data
from scaly import interp

GRID, POINTS, MEASURED, _ = calibration_data()
N = GRID.size


@sc.problem(vars=sc.L("T", (N, N)))
def calibrate(T):
  residual = interp.interpolant((GRID, GRID), T, kind="linear").at(POINTS) - MEASURED
  return sc.ProblemSpec(minimize=(residual**2).sum())


def build(verbose: bool = False):
  solve = sc.solver(calibrate, "ipopt", options=scaly_ipopt_options(verbose))

  def run():
    table, *_ = solve(np.zeros((N, N)), np.zeros((N, N)), np.zeros(0), np.zeros(0), ())
    stats = solve.solver_stats()
    return {"table": table, "f": np.array([stats.obj]), "iter": np.array([stats.iter])}

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
