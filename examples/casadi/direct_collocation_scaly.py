"""Van der Pol optimal control by direct collocation, Legendre points of degree 3 (Scaly).

The problem of ``direct_single_shooting``; on each of the N intervals the state is a degree-3
polynomial through the interval's start and three collocation points, all of them variables. The
variables are laid out as in the CasADi original, [x_0, (u_k, xc_k (3 x 2), x_k+1) for each k], so
each interval's constraints are one ``vmap`` with stride 9.

After casadi/docs/examples/python/direct_collocation.py (Joel Andersson, 2016).
"""

import numpy as np
from numpy.polynomial import legendre

import scaly as sc
from _common import scaly_ipopt_options, show

D_DEG = 3  # polynomial degree
T = 10.0
N = 20  # control intervals
H = T / N
NW = 2 + N * (1 + 2 * D_DEG + 2)


def collocation_coefficients(d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """C: derivative at the collocation points, D: value at the interval end, B: quadrature weights."""
  tau_root = np.append(0, (legendre.leggauss(d)[0] + 1) / 2)  # Gauss-Legendre points on [0, 1]
  C, D, B = np.zeros((d + 1, d + 1)), np.zeros(d + 1), np.zeros(d + 1)
  for j in range(d + 1):
    p = np.poly1d([1])
    for r in range(d + 1):
      if r != j:
        p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])
    D[j] = p(1.0)
    pder = np.polyder(p)
    for r in range(d + 1):
      C[j, r] = pder(tau_root[r])
    B[j] = np.polyint(p)(1.0)
  return C, D, B


C, D, B = collocation_coefficients(D_DEG)


def bounds() -> tuple[np.ndarray, np.ndarray]:
  stage_lb = np.r_[-1.0, np.tile([-0.25, -np.inf], D_DEG + 1)]  # u, the collocation states, the next state
  stage_ub = np.r_[1.0, np.full(2 * D_DEG + 2, np.inf)]
  return np.r_[0.0, 1.0, np.tile(stage_lb, N)], np.r_[0.0, 1.0, np.tile(stage_ub, N)]


LBW, UBW = bounds()
W0 = np.r_[0.0, 1.0, np.zeros(NW - 2)]


@sc.function(2, 1, output=sc.G("xdot", "L"))
def f(x, u):
  x1, x2 = x[0], x[1]
  return sc.stack([(1 - x2**2) * x1 - x2 + u[0], x1]), x1**2 + x2**2 + u[0] ** 2


@sc.function(2, 1, 2 * D_DEG, 2, output="g_and_q")
def interval(x, u, xc, xnext):
  X = [x] + [xc[2 * j : 2 * j + 2] for j in range(D_DEG)]
  g, q = [], sc.const(0.0)
  for j in range(1, D_DEG + 1):
    xp = sum((C[r, j] * X[r] for r in range(1, D_DEG + 1)), start=C[0, j] * X[0])
    fj, qj = f(X[j], u)
    g.append(H * fj - xp)
    q = q + B[j] * qj * H
  x_end = sum((D[j] * X[j] for j in range(1, D_DEG + 1)), start=D[0] * X[0])
  return sc.concat([*g, x_end - xnext, q.reshape((1,))])


@sc.problem(vars=sc.L("w", NW))
def collocation(w):
  out = sc.vmap(interval, N, [(w, 0, 9), (w, 2, 9), (w, 3, 9), (w, 9, 9)]).reshape((N, 2 * D_DEG + 3))
  return sc.ProblemSpec(minimize=out[:, -1].sum(), eq=(out[:, :-1].reshape((N * (2 * D_DEG + 2),)),), lb=sc.const(LBW), ub=sc.const(UBW))


def build(verbose: bool = False):
  solve = sc.solver(collocation, "ipopt", options=scaly_ipopt_options(verbose))

  def run():
    w, *_ = solve(W0, np.zeros(NW), np.zeros(N * (2 * D_DEG + 2)), np.zeros(0), ())
    stats = sc.solver_stats(solve)
    x = np.vstack([w[0:2], w[2:].reshape(N, 9)[:, 7:9]])
    return {"f": np.array([stats.obj]), "x1": x[:, 0], "x2": x[:, 1], "u": w[2::9], "iter": np.array([stats.iter])}

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
