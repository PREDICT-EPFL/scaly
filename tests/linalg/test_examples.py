"""The two non-PIQP examples of sparse linear algebra, run and checked against dense NumPy."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def test_sqp_newton_step_converges_quadratically() -> None:
  ns = runpy.run_path(str(EXAMPLES / "linalg" / "sqp_newton_sparse.py"), run_name="sqp_example")
  out = ns["main"]()
  res = out["residuals"]
  assert res[-1] < 1e-9 and len(res) <= 8
  # Quadratic convergence at the end: each residual about the square of the one before.
  tail = res[-4:-1]
  assert np.all(tail[1:] < 10.0 * tail[:-1] ** 2 + 1e-12)
  # Inertia correction was needed at the start and not near the solution.
  assert out["rhos"][0] > 0.0 and np.all(out["rhos"][-3:] == 0.0)
  # The step equals a dense solve of the same KKT system.
  n, nc, n_steps = ns["NW"], ns["NC"], ns["N"]
  w, lam = out["w"], out["lam"]
  z0 = np.zeros(2)
  w1, lam1, _, inertia, healthy = ns["newton_step"](w, lam, z0, np.array(1e-3))
  assert healthy and tuple(inertia) == (n, nc, 0)
  assert np.abs(w1 - w).max() < 1e-8 and np.abs(lam1 - lam).max() < 1e-6
  assert n == 3 * n_steps + 2


def test_kalman_update_matches_the_covariance_form() -> None:
  ns = runpy.run_path(str(EXAMPLES / "linalg" / "kalman_update.py"), run_name="kalman_example")
  out = ns["main"]()
  lam = ns["prior_information"]().toarray()
  h = ns["H_CONST"].toarray()
  p = np.linalg.inv(lam)
  s = h @ p @ h.T + np.diag(out["r"])
  gain = p @ h.T @ np.linalg.inv(s)
  np.testing.assert_allclose(out["x"], out["xbar"] + gain @ (out["z"] - h @ out["xbar"]), rtol=1e-9, atol=1e-10)
  np.testing.assert_allclose(out["gain"], gain, rtol=1e-8, atol=1e-11)
  # d x / d r_i = -K e_i e_i^T S^{-1} (z - H xbar): the derivative of the gain in the noise variance.
  nu = np.linalg.solve(s, out["z"] - h @ out["xbar"])
  np.testing.assert_allclose(out["dx_dr"], -gain * nu[None, :], rtol=1e-7, atol=1e-10)
  assert np.sqrt(np.mean((out["x"] - out["truth"]) ** 2)) < 0.2
