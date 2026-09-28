"""Legendre-Gauss-Lobatto pseudospectral transcription of an optimal-control problem (Scaly).

    minimize  4 x1(2) + x2(2) + int_0^2 4 u^2 dt   subject to  x1' = x2^3,  x2' = u,  x(0) = (0, 1)

The state is one global polynomial through N + 1 LGL points; the dynamics hold at every point
through the differentiation matrix D. The problem has an analytical solution, so the errors in
the states, the control and the costates (recovered from the multipliers of the defects) show
spectral convergence as N grows from 5 to 45.

After casadi/docs/examples/python/pseudospectral_collocation.py.
"""

import numpy as np

import scaly as sc
from _common import scaly_ipopt_options, show
from _lgl import lgl_nodes, lgl_setup

T0, TF = 0, 2
DEGREES = range(5, 50, 5)


def analytical(t: np.ndarray) -> np.ndarray:
  """Columns x1, x2, u, lambda1, lambda2."""
  return np.column_stack([-64 / (5 * (2 + t) ** 5) + 2 / 5, 4 / ((2 + t) ** 2), -8 / ((2 + t) ** 3), 4 * t**0, 64 / ((2 + t) ** 3)])


@sc.function(2, 1)
def xd(x, u):
  return sc.stack([x[1] ** 3, u[0]])


def make_ocp(N: int, verbose: bool):
  tau, wi, D = lgl_setup(N)

  @sc.problem(vars=sc.G(sc.L("X", 2 * (N + 1)), sc.L("U", N + 1)), name=f"lgl_N{N}")
  def ocp(variables):
    X, U = variables  # X stacks x at the N + 1 points
    Xm = X.reshape((N + 1, 2))
    defect = sc.const(D) @ Xm - (TF - T0) / 2 * sc.vmap(xd, N + 1, [(X, 0, 2), (U, 0, 1)]).reshape((N + 1, 2))
    lagrange = 0.5 * (TF - T0) * sc.dot(sc.const(wi), 4 * U * U)
    return sc.ProblemSpec(minimize=lagrange + 4 * Xm[N, 0] + Xm[N, 1], eq=(defect.reshape((2 * (N + 1),)), X[0:2] - sc.const(np.array([0.0, 1.0]))))

  solver = sc.solver(ocp, "ipopt", options=scaly_ipopt_options(verbose))
  X0 = np.repeat(np.linspace(0, 1, N + 1), 2)
  n_eq = 2 * (N + 1) + 2

  def solve():
    (X, U), _, lam_eq, _ = solver((X0, np.ones(N + 1)), (np.zeros(2 * (N + 1)), np.zeros(N + 1)), np.zeros(n_eq), np.zeros(0), ())
    adjoint = -lam_eq[: 2 * (N + 1)].reshape(N + 1, 2) / wi[:, None]
    numerical = np.hstack([X.reshape(N + 1, 2), U[:, None], adjoint])
    ts = (TF - T0) / 2 * lgl_nodes(N) + 0.5 * (TF + T0)
    return numerical, np.max(np.abs(analytical(ts) - numerical), axis=0), sc.solver_stats(solver).iter

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
