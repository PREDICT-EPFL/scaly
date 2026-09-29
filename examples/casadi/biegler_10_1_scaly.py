# /// script
# requires-python = ">=3.12"
# dependencies = ["scaly", "scaly-ipopt"]
#
# [tool.ty.environment]
# extra-paths = ["."]  # the modules beside this script, which it imports
# ///
"""Biegler's example 10.1: collocation on N = 1..10 elements for z' = z^2 - 2z + 1, z(0) = -3 (Scaly).

Radau collocation of degree K = 2 turns the initial-value problem into a square system of
equations, solved here by IPOPT as an NLP (the objective z(0)^2 is constant on the feasible set).
The exact solution is z(t) = (4t - 3) / (3t + 1), so z(1) = 1/4. Each N is its own problem, so its
own generated solver.

After casadi/docs/examples/python/biegler_10_1.py (Joel Andersson, 2012), from L. T. Biegler,
"Nonlinear Programming", SIAM 2010.
"""

import numpy as np

import scaly as sc
from _common import scaly_ipopt_options, show

K = 2
TAU_ROOT = [0.0, 0.211325, 0.788675]
Z0 = -3


def coefficients() -> tuple[np.ndarray, np.ndarray]:
  """D: the Lagrange polynomials at tau = 1, C[j, k]: their derivatives at the roots."""
  D, C = np.zeros(K + 1), np.zeros((K + 1, K + 1))
  for j in range(K + 1):
    L = np.poly1d([1.0])
    for k in range(K + 1):
      if k != j:
        L *= np.poly1d([1.0, -TAU_ROOT[k]]) / (TAU_ROOT[j] - TAU_ROOT[k])
    D[j] = L(1.0)
    C[j] = np.polyder(L)(TAU_ROOT)
  return D, C


def dz_dt(z: sc.Expr) -> sc.Expr:
  return z * z - 2 * z + 1


def problem(N: int) -> sc.opt.NLP:
  D, C = coefficients()
  h = 1.0 / N

  @sc.opt.problem(vars=sc.L("x", N * (K + 1)), name=f"biegler_N{N}")
  def collocation(x):
    Z = x.reshape((N, K + 1))
    g = []
    for i in range(N):
      for k in range(1, K + 1):
        g.append(h * dz_dt(Z[i, k]) - sum(Z[i, j] * C[j, k] for j in range(K + 1)))
      if i < N - 1:
        g.append(Z[i + 1, 0] - sum(D[j] * Z[i, j] for j in range(K + 1)))
    lb, ub = np.full(N * (K + 1), -100.0), np.full(N * (K + 1), 100.0)
    lb[0] = ub[0] = Z0
    return sc.opt.ProblemSpec(minimize=x[0] ** 2, eq=(sc.stack(g),), lb=sc.const(lb), ub=sc.const(ub))

  return collocation


def build(verbose: bool = False):
  solvers = [(N, sc.opt.solver(problem(N), sc.opt.IPOPT(options=scaly_ipopt_options(verbose, tol=1e-10)))) for N in range(1, 11)]

  def run():
    out = {}
    for N, solve in solvers:
      n, m = N * (K + 1), N * (K + 1) - 1
      out[f"z_N{N}"] = solve(np.zeros(n), np.zeros(n), np.zeros(m), np.zeros(0), ())[0]
    return out

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
