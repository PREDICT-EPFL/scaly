"""A Kalman measurement update for a spatial field with a sparse information-form prior.

The state is a field on ``N`` grid points with a Gaussian Markov random field prior: mean ``xbar``
and a sparse (banded) information matrix ``Lam = P^{-1}``, so the covariance ``P`` is dense and is
never formed. ``M`` sensors each read a weighted average of three neighbouring points,
``z = H x + v`` with ``v ~ N(0, diag(r))``. The posterior mean ``x = xbar + dx`` solves the
quasi-definite system

    [[Lam, H^T], [H, -diag(r)]] [dx; -mu] = [0; z - H xbar],

the optimality conditions of ``min 1/2 dx^T Lam dx + 1/2 v^T diag(r)^{-1} v`` subject to
``z = H (xbar + dx) + v``. It gives the same ``dx = P H^T (H P H^T + R)^{-1} (z - H xbar)`` as the
covariance-form update without inverting ``Lam``.

The update is one generated ``Function`` of the prior's information values, the prior mean, the
measurements and the noise variances. Its Jacobian in ``z`` is the Kalman gain, and derivatives in
``r`` (for tuning the noise model, say) come from the solve's implicit rule without
differentiating the factorization.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

import scaly as sc
from scaly import linalg

N, M = 200, 25
SENSORS = np.linspace(3, N - 4, M).astype(int)


def prior_information(tau: float = 50.0, eps: float = 1e-2) -> sparse.csc_array:
  """``tau (D^T D + eps I)`` with ``D`` the second difference: a smoothness prior, pentadiagonal."""
  d = sparse.diags_array([np.ones(N - 2), -2.0 * np.ones(N - 2), np.ones(N - 2)], offsets=[0, 1, 2], shape=(N - 2, N))
  return sparse.csc_array(tau * (d.T @ d + eps * sparse.eye_array(N)))


def measurement_matrix() -> sparse.csc_array:
  rows = np.repeat(np.arange(M), 3)
  cols = (SENSORS[:, None] + np.arange(-1, 2)[None, :]).reshape(-1)
  return sparse.csc_array((np.tile([0.25, 0.5, 0.25], M), (rows, cols)), shape=(M, N))


LAM_PATTERN = sparse.csc_array(sparse.tril(prior_information()))
LAM_PATTERN.sort_indices()
H_CONST = measurement_matrix()


@sc.function(
  sc.G(sc.L("lam", LAM_PATTERN.nnz), sc.L("xbar", N), sc.L("z", M), sc.L("r", M)),
  sc.L("x", N),
)
def kalman_update(inputs: tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr]) -> sc.Expr:
  lam_values, xbar, z, r = inputs
  lam = linalg.SparseMatrix.from_pattern(LAM_PATTERN, lam_values)  # the lower triangle of Lam, in its CSC order
  h = linalg.SparseMatrix.from_scipy(H_CONST)
  kkt = linalg.SparseMatrix.block([[lam, None], [h, linalg.SparseMatrix.diag(-r)]])
  innovation = z - h @ xbar
  step = linalg.SparseLDL(kkt).solve(sc.concat([sc.const(np.zeros(N)), innovation]))
  return xbar + step[:N]


@sc.function(
  sc.G(sc.L("lam", LAM_PATTERN.nnz), sc.L("xbar", N), sc.L("z", M), sc.L("r", M)),
  sc.G(sc.L("gain", (N, M)), sc.L("dx_dr", (N, M))),
)
def sensitivities(inputs: tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr]) -> tuple[sc.Expr, sc.Expr]:
  """The Kalman gain ``dx/dz`` and the sensitivity of the update to the noise variances."""
  lam_values, xbar, z, r = inputs
  x = kalman_update((lam_values, xbar, z, r))
  return sc.jacobian(x, z), sc.jacobian(x, r)


def lam_values() -> np.ndarray:
  return np.asarray(LAM_PATTERN.data, dtype=np.float64)


def main(seed: int = 0) -> dict[str, np.ndarray]:
  rng = np.random.default_rng(seed)
  grid = np.linspace(0.0, 1.0, N)
  truth = np.sin(2 * np.pi * grid) + 0.5 * np.cos(5 * np.pi * grid)
  r = np.full(M, 0.05**2)
  z = H_CONST @ truth + rng.normal(0.0, 0.05, M)
  xbar = np.zeros(N)
  x = kalman_update((lam_values(), xbar, z, r))
  gain, dx_dr = sensitivities((lam_values(), xbar, z, r))
  return {"truth": truth, "z": z, "r": r, "xbar": xbar, "x": x, "gain": gain, "dx_dr": dx_dr}


if __name__ == "__main__":
  out = main()
  prior_error = np.sqrt(np.mean((out["xbar"] - out["truth"]) ** 2))
  post_error = np.sqrt(np.mean((out["x"] - out["truth"]) ** 2))
  print(f"rms error: prior {prior_error:.3f}, posterior {post_error:.3f}")
  print(f"gain {out['gain'].shape}, largest entry {np.abs(out['gain']).max():.3f}")
