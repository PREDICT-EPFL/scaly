from __future__ import annotations

import numpy as np

from benchmarks.harness.hyperdual import lagrangian_hessian_np


def test_lagrangian_hessian_matches_closed_form() -> None:
  def fn(x):
    return [x[0] * x[1] + x[0] ** 2 * x[2], x[0] * x[2]]

  hess = lagrangian_hessian_np(fn, np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0]))
  np.testing.assert_allclose(hess, [[6.0, 1.0, 3.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], rtol=0.0, atol=1e-15)


def test_lagrangian_hessian_matches_finite_differences_on_transcendentals() -> None:
  def fn(x):
    rhs = np.array([np.sin(x[0]) * np.exp(x[1]) / np.sqrt(x[2]), np.tanh(x[0] * x[2]) - 2.0 / x[1], np.cos(x[1]) ** 3])
    return [rhs @ (np.arange(9.0).reshape(3, 3) @ rhs), 1.0 - x[2] * 0.5, x[0] * x[1] * x[2]]

  x, lam = np.array([0.3, 1.2, 2.5]), np.array([0.7, -0.4, 1.1])
  hess = lagrangian_hessian_np(fn, x, lam)
  assert np.allclose(hess, hess.T, rtol=0.0, atol=1e-12)

  def value(point):
    return float(np.dot(lam, np.asarray(fn(point), dtype=np.float64)))

  step = 1e-4
  for i in range(3):
    for j in range(3):
      ei, ej = np.eye(3)[i] * step, np.eye(3)[j] * step
      fd = (value(x + ei + ej) - value(x + ei - ej) - value(x - ei + ej) + value(x - ei - ej)) / (4.0 * step * step)
      np.testing.assert_allclose(hess[i, j], fd, rtol=1e-5, atol=1e-6)
