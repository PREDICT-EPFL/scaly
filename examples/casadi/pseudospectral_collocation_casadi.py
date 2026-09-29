# /// script
# requires-python = ">=3.12"
# dependencies = ["casadi"]
#
# [tool.ty.environment]
# extra-paths = ["."]  # the modules beside this script, which it imports
# ///
"""Legendre-Gauss-Lobatto pseudospectral transcription of an optimal-control problem, with ``Opti`` (CasADi).

    minimize  4 x1(2) + x2(2) + int_0^2 4 u^2 dt   subject to  x1' = x2^3,  x2' = u,  x(0) = (0, 1)

The state is one global polynomial through N + 1 LGL points; the dynamics hold at every point
through the differentiation matrix D. The problem has an analytical solution, so the errors in
the states, the control and the costates (recovered from the multipliers of the defects) show
spectral convergence as N grows from 5 to 45.

After casadi/docs/examples/python/pseudospectral_collocation.py.
"""

import casadi as cs
import numpy as np

from _common import casadi_jit, show
from _lgl import lgl_nodes, lgl_setup

T0, TF = 0, 2
DEGREES = range(5, 50, 5)


def analytical(t: np.ndarray) -> np.ndarray:
  """Columns x1, x2, u, lambda1, lambda2."""
  return np.column_stack([-64 / (5 * (2 + t) ** 5) + 2 / 5, 4 / ((2 + t) ** 2), -8 / ((2 + t) ** 3), 4 * t**0, 64 / ((2 + t) ** 3)])


def make_ocp(N: int, verbose: bool):
  x, u = cs.MX.sym("x", 2, 1), cs.MX.sym("u", 1, 1)
  xd = cs.Function("xd", [x, u], [cs.vertcat(x[1] ** 3, u)])
  lag = cs.Function("lag", [x, u], [4 * u**2])
  nlp = cs.Opti()
  X = nlp.variable(2, N + 1)
  U = nlp.variable(1, N + 1)
  tau, wi, D = lgl_setup(N)
  defect = X @ D.T - (TF - T0) / 2 * xd.map(N + 1, "serial")(X, U) == 0
  lagrange = 0.5 * (TF - T0) * cs.dot(wi.reshape(1, -1), lag.map(N + 1, "serial")(X, U))
  nlp.minimize(lagrange + 4 * X[0, -1] + X[1, -1])
  nlp.subject_to(defect)
  nlp.subject_to(X[:, 0] - cs.DM([0, 1]) == 0)
  nlp.solver("ipopt", {"print_time": verbose, **casadi_jit()}, {"print_level": 5 if verbose else 0, "sb": "yes"})
  nlp.set_initial(X, cs.vcat([cs.linspace(0, 1, N + 1).T, cs.linspace(0, 1, N + 1).T]))
  nlp.set_initial(U, cs.DM.ones(U.shape))

  def solve():
    sol = nlp.solve()
    adjoint = -np.asarray(sol.value(nlp.dual(defect) / cs.repmat(wi.reshape(1, -1), 2, 1)))
    numerical = np.hstack([np.asarray(sol.value(X)).T, np.asarray(sol.value(U)).reshape(-1, 1), adjoint.T])
    ts = (TF - T0) / 2 * lgl_nodes(N) + 0.5 * (TF + T0)
    return numerical, np.max(np.abs(analytical(ts) - numerical), axis=0), sol.stats()["iter_count"]

  return solve


def build(verbose: bool = False):
  ocps = {N: make_ocp(N, verbose) for N in DEGREES}

  def run():
    out = {}
    for N, solve in ocps.items():
      numerical, error, iterations = solve()
      out[f"error_N{N}"], out[f"iter_N{N}"] = error, np.array([iterations])
      if N == 25:
        out["x1_N25"], out["x2_N25"], out["u_N25"], out["lam1_N25"], out["lam2_N25"] = numerical.T
    return out

  return run


if __name__ == "__main__":
  show(build(verbose=False)())
