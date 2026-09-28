from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import BSpline as SciBSpline

from scaly.interp import constrained

from .helpers import numbers

pytestmark = pytest.mark.solver("piqp")


def ocv_data(rng: np.random.Generator, m: int = 120) -> tuple[np.ndarray, np.ndarray]:
  """A battery's open-circuit voltage against state of charge: steep at both ends, flat between,
  measured with noise that breaks its monotonicity."""
  soc = np.sort(rng.uniform(0.0, 1.0, m))
  ocv = 3.0 + 0.9 * soc + 0.25 * np.tanh(12 * (soc - 0.08)) - 0.2 * np.exp(-30 * (1 - soc)) + 0.03 * rng.normal(size=m)
  return soc, ocv


def test_a_monotone_bounded_fit_holds_everywhere() -> None:
  rng = np.random.default_rng(1)
  soc, ocv = ocv_data(rng)
  assert np.any(np.diff(ocv) < 0)  # the data are not monotone
  f = constrained(soc, ocv, knots=16, monotone="increasing", bounds=(2.6, 4.0))
  s = f.to_scipy()
  dense = np.linspace(soc[0], soc[-1], 10_000)
  assert s.derivative()(dense).min() >= -1e-9
  assert s(dense).min() >= 2.6 - 1e-9 and s(dense).max() <= 4.0 + 1e-9
  assert np.all(np.diff(numbers(f.coeffs)) >= -1e-10)


def test_a_convex_fit_holds_everywhere_and_concave_is_its_mirror() -> None:
  rng = np.random.default_rng(2)
  load = np.sort(rng.uniform(0.2, 1.0, 80))
  bump = 0.3 * np.exp(-(((load - 0.25) / 0.06) ** 2))  # concave at the left end, so convexity binds there
  cost = 1.0 + 0.5 * load + 2.0 * (load - 0.3) ** 2 + bump + 0.05 * rng.normal(size=80)
  dense = np.linspace(load[0], load[-1], 10_000)
  convex = constrained(load, cost, knots=10, convex="convex").to_scipy()
  assert convex.derivative(2)(dense).min() >= -1e-8
  concave = constrained(load, -cost, knots=10, convex="concave").to_scipy()
  np.testing.assert_allclose(concave(dense), -convex(dense), rtol=0, atol=1e-8)


def test_without_an_active_constraint_the_fit_is_least_squares() -> None:
  rng = np.random.default_rng(3)
  x = np.sort(rng.uniform(0.0, 2.0, 200))
  y = np.exp(0.5 * x) + 1e-3 * rng.normal(size=200)  # increasing and convex by a margin
  plain = constrained(x, y, knots=8)
  assert numbers(plain.coeffs).size == 8 + 3  # eight intervals of a cubic
  np.testing.assert_allclose(numbers(constrained(x, y, knots=plain.knots[0][4:-4]).coeffs), numbers(plain.coeffs), rtol=0, atol=1e-12)
  shaped = constrained(x, y, knots=8, monotone="increasing", convex="convex", bounds=(0.0, 10.0))
  B = SciBSpline.design_matrix(x, plain.knots[0], 3).toarray()
  lstsq = np.linalg.lstsq(B, y, rcond=None)[0]
  np.testing.assert_allclose(numbers(plain.coeffs), lstsq, rtol=0, atol=1e-10)
  np.testing.assert_allclose(numbers(shaped.coeffs), lstsq, rtol=0, atol=1e-10)


def test_integer_weights_are_repeated_data() -> None:
  rng = np.random.default_rng(7)
  soc, ocv = ocv_data(rng, 60)
  w = rng.integers(1, 4, size=60)
  weighted = constrained(soc, ocv, knots=10, weights=w, monotone="increasing")
  repeated = constrained(np.repeat(soc, w), np.repeat(ocv, w), knots=10, monotone="increasing")
  np.testing.assert_allclose(numbers(weighted.coeffs), numbers(repeated.coeffs), rtol=0, atol=1e-10)
  assert np.max(np.abs(numbers(weighted.coeffs) - numbers(constrained(soc, ocv, knots=10, monotone="increasing").coeffs))) > 1e-3


def test_it_agrees_with_trust_constr() -> None:
  from scipy.optimize import LinearConstraint, minimize

  rng = np.random.default_rng(4)
  x = np.sort(rng.uniform(0.0, 1.0, 40))
  y = np.where(x < 0.5, 0.0, 1.0) + 0.1 * rng.normal(size=40)  # a step: monotone and bounded bind
  f = constrained(x, y, knots=6, monotone="increasing", bounds=(0.0, 1.0), lam=1e-3)
  t = f.knots[0]
  n = t.size - 4
  B = SciBSpline.design_matrix(x, t, 3).toarray()
  D = np.diff(np.eye(n), n=2, axis=0)
  objective = lambda c: np.sum((B @ c - y) ** 2) + 1e-3 * np.sum((D @ c) ** 2)
  gradient = lambda c: 2 * B.T @ (B @ c - y) + 2e-3 * D.T @ (D @ c)
  constraints = [LinearConstraint(np.diff(np.eye(n), axis=0), 0.0, np.inf), LinearConstraint(np.eye(n), 0.0, 1.0)]
  ref = minimize(
    objective, np.full(n, 0.5), jac=gradient, method="trust-constr", constraints=constraints, options={"gtol": 1e-12, "xtol": 1e-14, "maxiter": 5000}
  )
  np.testing.assert_allclose(numbers(f.coeffs), ref.x, rtol=0, atol=1e-6)
  c = numbers(f.coeffs)
  assert np.all(c[:4] == 0.0) and np.all(np.abs(c[5:] - 1.0) < 1e-14)  # polished onto the bounds, not a barrier's width inside


def test_a_wrong_active_set_leaves_the_interior_point_answer() -> None:
  import sys

  polish = sys.modules["scaly.interp.constrained"]._polish
  M, target = np.eye(2), np.array([1.0, -1.0])  # min |c - target|^2 over c >= 0: c = (1, 0)
  G, lb, ub = np.eye(2), np.zeros(2), np.full(2, np.inf)
  x = np.array([1.0 - 1e-11, 1e-11])
  np.testing.assert_array_equal(polish(M, target, np.zeros((0, 2)), np.zeros(0), G, lb, ub, x, np.array([0.0, -1.0])), [1.0, 0.0])
  wrong = np.array([-1.0, 0.0])  # claims the first bound holds and the second does not
  assert polish(M, target, np.zeros((0, 2)), np.zeros(0), G, lb, ub, x, wrong) is x  # infeasible
  inside = np.array([1.0, 1.0])  # the target itself is feasible: no bound holds
  assert polish(M, inside, np.zeros((0, 2)), np.zeros(0), G, lb, ub, inside, 2 * wrong) is inside  # feasible, but the bound pulls


def test_pinned_values_and_derivatives_and_a_periodic_curve() -> None:
  rng = np.random.default_rng(5)
  x = np.sort(rng.uniform(0.0, 2 * np.pi, 150))
  x[0], x[-1] = 0.0, 2 * np.pi
  y = np.sin(x) + 0.1 * rng.normal(size=150)
  f = constrained(x, y, knots=12, equal=[(1.0, 0.5, 0), (2.0, -0.3, 1)], periodic=True).to_scipy()
  assert abs(f(1.0) - 0.5) < 1e-9 and abs(f(2.0, 1) + 0.3) < 1e-9
  for nu in range(3):
    assert abs(f(0.0, nu) - f(2 * np.pi, nu)) < 1e-8


def test_a_bounded_monotone_map_in_2d() -> None:
  rng = np.random.default_rng(6)
  speed, torque = rng.uniform(0.0, 1.0, 400), rng.uniform(0.0, 1.0, 400)
  eta = 0.95 - 0.3 * (speed - 0.6) ** 2 - 0.2 * (torque - 0.5) ** 2 + 0.04 * rng.normal(size=400)
  # increasing in speed and decreasing in torque, which the data contradict below 0.6 and 0.5
  f = constrained(np.column_stack([speed, torque]), eta, knots=(5, 5), bounds=(0.0, 0.95), monotone=("increasing", "decreasing"), lam=1e-4)
  s = f.to_scipy()
  grid = np.stack(
    np.meshgrid(np.linspace(speed.min(), speed.max(), 100), np.linspace(torque.min(), torque.max(), 100), indexing="ij"), axis=-1
  ).reshape(-1, 2)
  values = s(grid)
  assert values.max() <= 0.95 + 1e-9 and values.min() >= -1e-9
  assert s(grid, nu=(1, 0)).min() >= -1e-9 and s(grid, nu=(0, 1)).max() <= 1e-9


def test_infeasible_and_invalid_requests_are_refused() -> None:
  x = np.linspace(0.0, 1.0, 30)
  with pytest.raises(ValueError, match="failed"):
    constrained(x, x, knots=4, bounds=(0.0, 1.0), equal=[(0.5, 2.0, 0)])
  with pytest.raises(ValueError, match="takes 'increasing'"):
    constrained(x, x, monotone="up")  # ty: ignore[invalid-argument-type]
  with pytest.raises(ValueError, match="periodic is 1-D"):
    constrained(np.column_stack([x, x]), x, knots=(3, 3), periodic=True)
  with pytest.raises(ValueError, match="points"):
    constrained(x, x[:5])
  with pytest.raises(ValueError, match="order 2 needs degree"):
    constrained(x, x, degree=1, convex="convex")
