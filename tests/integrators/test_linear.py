"""Exact discretization of linear systems, and linearization of models and maps."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import scaly as sc
from scaly import integrators as si

RNG = np.random.default_rng(11)
A = np.array([[0.0, 1.0, 0.0], [-2.0, -0.4, 1.0], [0.3, 0.0, -1.5]])
B = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])


def _flow(u_of_t, x0: np.ndarray, dt: float) -> np.ndarray:
  return solve_ivp(lambda t, x: A @ x + B @ u_of_t(t), (0, dt), x0, method="DOP853", rtol=1e-13, atol=1e-14).y[:, -1]


def test_zoh_is_the_exact_flow_with_the_input_held() -> None:
  dt, x0, u = 0.3, RNG.standard_normal(3), RNG.standard_normal(2)
  ad, bd = si.zoh(A, B, dt)
  np.testing.assert_allclose(ad @ x0 + bd @ u, _flow(lambda t: u, x0, dt), rtol=1e-11, atol=1e-12)
  np.testing.assert_allclose(bd, np.linalg.solve(A, (ad - np.eye(3)) @ B), rtol=1e-11, atol=1e-12)  # A is invertible here


def test_foh_is_the_exact_flow_with_the_input_ramped() -> None:
  dt, x0, u0, u1 = 0.3, RNG.standard_normal(3), RNG.standard_normal(2), RNG.standard_normal(2)
  ad, b0, b1 = si.foh(A, B, dt)
  np.testing.assert_allclose(ad, si.zoh(A, B, dt)[0], rtol=1e-14)
  np.testing.assert_allclose(ad @ x0 + b0 @ u0 + b1 @ u1, _flow(lambda t: u0 + (u1 - u0) * t / dt, x0, dt), rtol=1e-11, atol=1e-12)
  np.testing.assert_allclose(b0 + b1, si.zoh(A, B, dt)[1], rtol=1e-12, atol=1e-13)  # a constant input is held either way


def test_a_scalar_system_and_the_refusals() -> None:
  ad, bd = si.zoh(-2.0, 1.0, 0.5)
  np.testing.assert_allclose([ad[0, 0], bd[0, 0]], [np.exp(-1.0), (1 - np.exp(-1.0)) / 2], rtol=1e-14)
  with pytest.raises(ValueError, match="A must be square"):
    si.zoh(np.ones((2, 3)), np.ones((2, 1)), 0.1)
  with pytest.raises(ValueError, match="dt must be positive"):
    si.foh(A, B, 0.0)


@sc.function(3, sc.G(sc.L("u", 2), sc.L("gain", ())), output="xdot")
def grouped(x, inputs):
  u, gain = inputs
  return sc.const(A) @ x + gain * (sc.const(B) @ u) + 0.1 * sc.stack([x[0] * x[1], x[2].sin(), 0.0 * x[0]])


def test_linearize_gives_one_jacobian_per_leaf() -> None:
  x, u, gain = RNG.standard_normal(3), RNG.standard_normal(2), np.array(1.7)
  a, b, g = si.linearize(grouped, x, (u, gain))
  expected_a = A + 0.1 * np.array([[x[1], x[0], 0.0], [0.0, 0.0, np.cos(x[2])], [0.0, 0.0, 0.0]])
  np.testing.assert_allclose(a, expected_a, rtol=1e-13, atol=1e-14)
  np.testing.assert_allclose(b, gain * B, rtol=1e-14)
  np.testing.assert_allclose(g.ravel(), B @ u, rtol=1e-14)
  with pytest.raises(ValueError, match="takes 2 arguments, got 1"):
    si.linearize(grouped, x)
  with pytest.raises(TypeError, match="needs an sc.Function"):
    si.linearize(lambda x: x, x)  # ty: ignore[invalid-argument-type]


def test_linearizing_a_map_and_a_template() -> None:
  step = si.rk4(grouped, dt=0.1, steps=2)
  x, u, gain = RNG.standard_normal(3), RNG.standard_normal(2), np.array(0.5)
  ad, bd, _ = si.linearize(step, x, (u, gain))
  eps = 1e-6
  fd = np.stack([(step(x + e, (u, gain)) - step(x - e, (u, gain))) / (2 * eps) for e in np.eye(3) * eps], axis=1)
  np.testing.assert_allclose(ad, fd, rtol=1e-7, atol=1e-9)
  assert bd.shape == (3, 2)

  @sc.function(output="xdot")
  def decay(x, k):
    return -k * x * x

  a, k_col = si.linearize(decay, np.array([1.0, 2.0]), np.array(3.0))
  np.testing.assert_allclose(a, np.diag([-6.0, -12.0]))
  np.testing.assert_allclose(k_col.ravel(), [-1.0, -4.0])
