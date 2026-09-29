"""The variational step: its endpoint is the explicit map's, and the Jacobian it carries is AD's
derivative of that endpoint, under a zero-order and a per-component first-order hold, with a held
parameter; the stages it skips; what it refuses."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly import integrators as si
from scaly.integrators.variational import _needed


@sc.function(sc.L("x", 3), sc.L("u", 2), sc.L("gain", ()), output="xdot", name="var_model")
def model(x, u, gain):
  return sc.stack([x[1], -gain * x[0].sin() - 0.1 * x[1] * x[2] + u[0], u[1] * x[0] - 0.5 * x[2]])


X0, U0, U1, GAIN = np.array([0.3, -0.2, 0.5]), np.array([0.7, -0.4]), np.array([-0.1, 0.9]), np.array(2.0)


@pytest.mark.parametrize("method", ["rk4", "tsit5", "dopri5"])
def test_a_zero_order_hold_is_the_explicit_map_and_its_jacobian(method: str) -> None:
  step = si.variational(model, method, dt=0.2, steps=3, name=f"var_zoh_{method}")
  flow = si.explicit(model, method, dt=0.2, steps=3, name=f"var_map_{method}")

  @sc.function(sc.L("x", 3), sc.L("u", 2), sc.L("gain", ()), output=sc.G("x_next", "J"), name=f"var_ad_{method}")
  def reference(x, u, gain):
    x_next = flow(x, u, gain)
    return x_next, sc.concat([sc.jacobian(x_next, x), sc.jacobian(x_next, u)], axis=1)

  x, phi = step(X0, U0, GAIN)
  x_ref, jac = reference(X0, U0, GAIN)
  assert phi.shape == (3, 5)
  np.testing.assert_allclose(x, x_ref, rtol=1e-14, atol=1e-15)
  np.testing.assert_allclose(phi, jac, rtol=1e-12, atol=1e-13)


def test_a_first_order_hold_per_component_carries_the_endpoints_derivative() -> None:
  """The first control linear from ``u0`` to ``u1``, the second held: AD of the endpoint in
  ``(x0, u0, u1)`` is the carried ``Phi``, and the held component's ``u1`` columns are zero."""
  step = si.variational(model, "tsit5", dt=0.25, steps=2, hold=[1.0, 0.0], name="var_foh")

  @sc.function(sc.L("x0", 3), sc.L("u0", 2), sc.L("u1", 2), sc.L("gain", ()), output=sc.G("x", "J"), name="var_foh_ad")
  def reference(x0, u0, u1, gain):
    x, _ = step(x0, u0, u1, gain)
    return x, sc.concat([sc.jacobian(x, x0), sc.jacobian(x, u0), sc.jacobian(x, u1)], axis=1)

  x, phi = step(X0, U0, U1, GAIN)
  x_ad, jac = reference(X0, U0, U1, GAIN)
  assert phi.shape == (3, 7)
  np.testing.assert_array_equal(x, x_ad)
  np.testing.assert_allclose(phi, jac, rtol=1e-12, atol=1e-13)
  np.testing.assert_array_equal(phi[:, 6], 0.0)  # d x / d u1 of the held component
  # The endpoint against NumPy: the same tableau, the control linear in time over the whole interval.
  np.testing.assert_allclose(x, _numpy_foh(X0, U0, U1, 0.25, 2), rtol=1e-14, atol=1e-15)
  # A first-order hold with u1 = u0 is the zero-order hold.
  held, _ = si.variational(model, "tsit5", dt=0.25, steps=2, name="var_foh_zoh")(X0, U0, GAIN)
  np.testing.assert_allclose(step(X0, U0, U0, GAIN)[0], held, rtol=1e-15)


def _numpy_foh(x0: np.ndarray, u0: np.ndarray, u1: np.ndarray, dt: float, steps: int) -> np.ndarray:
  tab, h = si.tableau("tsit5"), dt / steps

  def rates(x: np.ndarray, t: float) -> np.ndarray:
    u = np.array([u0[0] + t / dt * (u1[0] - u0[0]), u0[1]])
    return np.array([x[1], -2.0 * np.sin(x[0]) - 0.1 * x[1] * x[2] + u[0], u[1] * x[0] - 0.5 * x[2]])

  x = x0.copy()
  for step in range(steps):
    ks: list[np.ndarray] = []
    for i in range(len(tab.b)):
      ks.append(rates(x + h * sum(tab.a[i, j] * ks[j] for j in range(i)), step * h + tab.c[i] * h))
    x = x + h * sum(tab.b[i] * ks[i] for i in range(len(tab.b)))
  return x


def test_the_stages_no_weight_reads_are_skipped() -> None:
  tsit5 = si.tableau("tsit5")
  assert _needed(tsit5.a, tsit5.b).tolist() == [True] * 6 + [False]  # first same as last
  assert _needed(si.tableau("rk4").a, si.tableau("rk4").b).all()


def test_what_variational_refuses() -> None:
  with pytest.raises(ValueError, match="explicit method"):
    si.variational(model, si.radau_iia(2), dt=0.1)
  with pytest.raises(ValueError, match="hold is 0"):
    si.variational(model, "rk4", dt=0.1, hold=0.5)
  with pytest.raises(ValueError, match="steps must be at least 1"):
    si.variational(model, "rk4", dt=0.1, steps=0)
