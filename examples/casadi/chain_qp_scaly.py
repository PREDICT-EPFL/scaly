"""A hanging chain of N masses resting on a sloped floor: a sparse QP, solved by PIQP (Scaly).

    minimize    sum_i D/2 ((y_i - y_{i+1})^2 + (z_i - z_{i+1})^2) + g0 sum_i m z_i
    subject to  z_i >= 0.5,  z_i - 0.1 y_i >= 0.5,  both ends fixed

Scaly proves the problem is a QP (constant Hessian, affine constraints) before it accepts PIQP, and
``sparse=True`` bakes the sparsity patterns of the extracted matrices into the generated solver.

After casadi/docs/examples/python/chain_qp.py.
"""

import numpy as np

import scaly as sc
from _common import show

N = 40
M_I = 40.0 / N
D_I = 70.0 * N
G0 = 9.81
ZMIN = 0.5


def variable_bounds() -> tuple[np.ndarray, np.ndarray]:
  lb, ub = np.tile([-np.inf, ZMIN], N), np.full(2 * N, np.inf)
  lb[:2] = ub[:2] = [-2.0, 1.0]
  lb[-2:] = ub[-2:] = [2.0, 1.0]
  return lb, ub


LB, UB = variable_bounds()


@sc.problem(vars=sc.L("x", 2 * N))
def chain(x):
  y, z = x.reshape((N, 2))[:, 0], x.reshape((N, 2))[:, 1]
  vchain = D_I / 2 * (sc.sumsqr(y[1:] - y[:-1]) + sc.sumsqr(z[1:] - z[:-1])) + G0 * M_I * z.sum()
  return sc.ProblemSpec(minimize=vchain, ineq=(sc.bounded(z - 0.1 * y, lo=0.5),), lb=sc.const(LB), ub=sc.const(UB))


def build(verbose: bool = False):
  solve = sc.solver(chain, "piqp", options={"sparse": True, "verbose": verbose})

  def run():
    x, *_ = solve(np.zeros(2 * N), np.zeros(2 * N), np.zeros(0), np.zeros(N), ())
    return {"f": np.array([solve.solver_stats().obj]), "y": x[0::2], "z": x[1::2]}

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
