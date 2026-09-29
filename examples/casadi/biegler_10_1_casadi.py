# /// script
# requires-python = ">=3.12"
# dependencies = ["casadi"]
#
# [tool.ty.environment]
# extra-paths = ["."]  # the modules beside this script, which it imports
# ///
"""Biegler's example 10.1: collocation on N = 1..10 elements for z' = z^2 - 2z + 1, z(0) = -3 (CasADi).

Radau collocation of degree K = 2 turns the initial-value problem into a square system of
equations, solved here by IPOPT as an NLP (the objective z(0)^2 is constant on the feasible set).
The exact solution is z(t) = (4t - 3) / (3t + 1), so z(1) = 1/4.

After casadi/docs/examples/python/biegler_10_1.py (Joel Andersson, 2012), from L. T. Biegler,
"Nonlinear Programming", SIAM 2010.
"""

import casadi as ca

from _common import as_arrays, casadi_ipopt_options, casadi_jit, show

K = 2
TAU_ROOT = [0.0, 0.211325, 0.788675]
Z0 = -3


def build(verbose: bool = False):
  z, tau = ca.SX.sym("z"), ca.SX.sym("tau")
  F = ca.Function("dz_dt", [z], [z * z - 2 * z + 1])
  D, C = ca.DM.zeros(K + 1), ca.DM.zeros(K + 1, K + 1)
  for j in range(K + 1):
    L = 1
    for k in range(K + 1):
      if k != j:
        L *= (tau - TAU_ROOT[k]) / (TAU_ROOT[j] - TAU_ROOT[k])
    D[j] = ca.Function("lfcn", [tau], [L])(1.0)
    tfcn = ca.Function("tfcn", [tau], [ca.tangent(L, tau)])
    for k in range(K + 1):
      C[j, k] = tfcn(TAU_ROOT[k])

  solvers = []
  for N in range(1, 11):
    h = 1.0 / N
    Z = ca.SX.sym("Z", N, K + 1)
    x = ca.vec(Z.T)
    g = []
    for i in range(N):
      for k in range(1, K + 1):
        rhs = 0
        for j in range(K + 1):
          rhs += Z[i, j] * C[j, k]
        g.append(h * F(Z[i, k]) - rhs)
      rhs = 0
      for j in range(K + 1):
        rhs += D[j] * Z[i, j]
      if i < N - 1:
        g.append(Z[i + 1, 0] - rhs)
    nlp = {"x": x, "f": x[0] ** 2, "g": ca.vertcat(*g)}
    solvers.append((N, ca.nlpsol("solver", "ipopt", nlp, {**casadi_ipopt_options(verbose, tol=1e-10), **casadi_jit()})))

  def run():
    out = {}
    for N, solver in solvers:
      n = N * (K + 1)
      lbx, ubx = n * [-100], n * [100]
      lbx[0] = ubx[0] = Z0
      res = solver(x0=n * [0], lbx=lbx, ubx=ubx, lbg=0, ubg=0)
      out[f"z_N{N}"] = res["x"]
    return as_arrays(out)

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
