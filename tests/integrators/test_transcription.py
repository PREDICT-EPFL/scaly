"""Transcriptions: collocation equals the implicit method of its nodes, pseudospectral converges
spectrally, shooting integrates its cost with its step, and the interval layouts are as documented."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import root

import scaly as sc
from scaly import integrators as si

MU = 3.0


@sc.function(2, 1, output="xdot")
def van_der_pol(x, u):
  return sc.stack([x[1], MU * (1 - x[0] * x[0]) * x[1] - x[0] + u[0]])


@sc.function(2, 1, output="l")
def effort(x, u):
  return (x * x).sum() + u[0] * u[0]


def _consistent(
  interval: si.Interval, x: np.ndarray, u: np.ndarray, *, dt: np.ndarray | None = None, fixed_controls: bool = False
) -> tuple[np.ndarray, np.ndarray]:
  """``(z, xnext)`` zeroing the interval's residuals from ``x`` and ``u``, by SciPy; with
  ``fixed_controls`` the controls in ``z`` stay at ``u``, so only the states are unknowns."""
  n = x.size
  n_states = interval.state_times.size * n
  controls = np.tile(u, interval.control_times.size)

  def unpack(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.concatenate([v[:n_states], controls]) if fixed_controls else v[: interval.n_internal]
    return z, v[(n_states if fixed_controls else interval.n_internal) :]

  def residual(v: np.ndarray) -> np.ndarray:
    z, xnext = unpack(v)
    args = [x, u, *([z] if interval.n_internal else []), xnext, *([] if dt is None else [dt])]
    out = interval.fn(*args)
    return np.asarray(out[0] if interval.has_cost else out)

  start = np.concatenate([interval.guess(x, u)[: n_states if fixed_controls else interval.n_internal], x])
  solution = root(residual, start, method="hybr", options={"xtol": 1e-13})
  assert np.abs(residual(solution.x)).max() < 1e-12
  return unpack(solution.x)


@pytest.mark.parametrize(("points", "method"), [("radau", "radau_iia"), ("legendre", "gauss_legendre")])
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_collocation_is_the_implicit_method_of_its_nodes(points: str, method: str, degree: int) -> None:
  x, u = np.array([1.5, -0.3]), np.array([0.4])
  interval = si.Collocation(degree, points).interval(van_der_pol, dt=0.1, name=f"col_{points}{degree}")  # ty: ignore[invalid-argument-type]
  _, xnext = _consistent(interval, x, u)
  step = si.implicit(van_der_pol, method, stages=degree, dt=0.1, tol=1e-15, max_iter=60, name=f"ref_{method}{degree}")
  np.testing.assert_allclose(xnext, step(x, u), rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize(("points", "degree", "internal", "residual"), [("radau", 3, 4, 6), ("radau", 1, 0, 2), ("legendre", 3, 6, 8)])
def test_collocation_layouts(points: str, degree: int, internal: int, residual: int) -> None:
  interval = si.Collocation(degree, points).interval(van_der_pol, effort, dt=0.1)  # ty: ignore[invalid-argument-type]
  assert (interval.n_internal, interval.n_residual, interval.has_cost) == (internal, residual, True)
  assert interval.control_times.size == 0 and interval.state_times.size * 2 == internal
  assert interval.fn.input_names == ("x", "u", *(("z",) if internal else ()), "xnext")
  assert interval.fn.outputs[0].shape == (residual,) and interval.fn.outputs[1].shape == (1,) and interval.fn.output_names == ("r", "cost")


def test_the_cost_quadrature_is_exact_where_the_nodes_allow() -> None:
  """With ``x' = u`` the state is linear in time and a quadratic running cost is a polynomial of
  degree 2, which every rule here integrates exactly."""

  @sc.function(1, 1, output="xdot")
  def drift(x, u):
    return u + 0.0 * x

  @sc.function(1, 1, output="l")
  def square(x, u):
    return (x * x).sum()

  x0, u, h = np.array([0.5]), np.array([2.0]), 0.3
  exact = ((x0[0] + u[0] * h) ** 3 - x0[0] ** 3) / (3 * u[0])
  for transcription in (si.Collocation(2, "radau"), si.Collocation(2, "legendre"), si.Pseudospectral(3)):
    interval = transcription.interval(drift, square, dt=h, name=f"quad_{transcription.label}")
    z, xnext = _consistent(interval, x0, u, fixed_controls=True)
    args = [x0, u, *([z] if interval.n_internal else []), xnext]
    np.testing.assert_allclose(interval.fn(*args)[1], [exact], rtol=1e-12, err_msg=transcription.label)
  # One Radau point is backward Euler, whose rule is the end point's value times the length.
  interval = si.Collocation(1, "radau").interval(drift, square, dt=h, name="quad_backward_euler")
  _, xnext = _consistent(interval, x0, u)
  np.testing.assert_allclose(interval.fn(x0, u, xnext)[1], [h * xnext[0] ** 2], rtol=1e-14)


def test_shooting_steps_and_integrates_its_cost_with_its_method() -> None:
  x, u, xn = np.array([1.5, -0.3]), np.array([0.4]), np.array([1.4, -0.2])
  interval = si.MultipleShooting(si.rk4, steps=2).interval(van_der_pol, effort, dt=0.1)
  assert interval.n_internal == 0 and interval.fn.input_names == ("x", "u", "xnext")
  r, cost = interval.fn(x, u, xn)
  np.testing.assert_allclose(r, si.rk4(van_der_pol, dt=0.1, steps=2)(x, u) - xn, rtol=1e-14, atol=1e-15)
  exact = solve_ivp(
    lambda t, y: [*[y[1], MU * (1 - y[0] ** 2) * y[1] - y[0] + u[0]], y[0] ** 2 + y[1] ** 2 + u[0] ** 2], (0, 0.1), [*x, 0.0], rtol=1e-13, atol=1e-14
  ).y[:, -1]
  np.testing.assert_allclose(cost, exact[2:], rtol=1e-6)  # RK4 on the augmented model, to RK4's accuracy
  implicit = si.MultipleShooting(si.implicit, method="radau_iia", stages=2, tol=1e-14).interval(van_der_pol, dt=0.1)
  np.testing.assert_allclose(
    np.asarray(implicit.fn(x, u, xn)), si.implicit(van_der_pol, "radau_iia", stages=2, dt=0.1, tol=1e-14)(x, u) - xn, rtol=1e-13
  )
  assert implicit.fn.name == "van_der_pol_implicit_shooting"


def test_global_pseudospectral_converges_spectrally() -> None:
  x0, u, T = np.array([1.5, 0.0]), np.array([0.4]), 1.0
  exact = solve_ivp(lambda t, y: [y[1], MU * (1 - y[0] ** 2) * y[1] - y[0] + u[0]], (0, T), x0, method="DOP853", rtol=1e-13, atol=1e-14).y[:, -1]
  errors = []
  for nodes in (4, 8, 12):
    interval = si.Pseudospectral(nodes).interval(van_der_pol, dt=T, name=f"ps{nodes}")
    _, xnext = _consistent(interval, x0, u, fixed_controls=True)
    errors.append(float(np.abs(xnext - exact).max()))
  assert errors[2] < 1e-7 and errors[1] < 1e-2 * errors[0], errors  # faster than any fixed order


def test_pseudospectral_layout() -> None:
  interval = si.Pseudospectral(5).interval(van_der_pol, dt=0.2)
  assert interval.state_times.size == 4 and interval.control_times.size == 4 and interval.n_internal == 4 * 2 + 4 * 1 and interval.n_residual == 5 * 2
  np.testing.assert_allclose(interval.state_times, interval.control_times)
  assert 0 < interval.state_times[0] and interval.state_times[-1] < 1
  guess = interval.guess(np.array([1.0, 2.0]), np.array([3.0]))
  np.testing.assert_array_equal(guess, [1, 2] * 4 + [3] * 4)


@sc.function(2, 1, (), output="xdot")
def damped(x, u, mu):
  return sc.stack([x[1], mu * (1 - x[0] * x[0]) * x[1] - x[0] + u[0]])


@sc.function(2, 1, (), output="l")
def weighted(x, u, mu):
  return mu * (x * x).sum() + u[0] * u[0]


def test_an_interval_length_input_equals_the_folded_one() -> None:
  x, u, mu = np.array([1.5, -0.3]), np.array([0.4]), np.array(2.5)
  for transcription in (si.Collocation(3), si.Collocation(2, "legendre"), si.Pseudospectral(4), si.MultipleShooting(si.rk4)):
    folded = transcription.interval(damped, weighted, dt=0.1, name=f"fold_{transcription.label}")
    free = transcription.interval(damped, weighted, dt=None, name=f"free_{transcription.label}")
    assert free.fn.input_names[-2:] == ("p0", "dt")  # the parameters, then the length
    z = folded.guess(x, u) + 0.01
    args = [x, u, *([z] if folded.n_internal else []), x + 0.02, mu]
    for a, b in zip(free.fn(*args, np.array(0.1)), folded.fn(*args), strict=True):
      np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-15)


def test_a_horizon_of_intervals_differentiates() -> None:
  interval = si.Collocation(2).interval(van_der_pol, dt=0.1)
  n, k, horizon = 2, interval.n_internal, 5
  size = (horizon + 1) * n + horizon * (1 + k)

  @sc.function(size, output="r")
  def residuals(w):
    xs, us, zs = w[: (horizon + 1) * n], w[(horizon + 1) * n : (horizon + 1) * n + horizon], w[(horizon + 1) * n + horizon :]
    return sc.vmap(interval.fn, horizon, [(xs, 0, n), (us, 0, 1), (zs, 0, k), (xs, n, n)])

  w = np.random.default_rng(4).normal(size=size) * 0.3
  sparse, dense = sc.sparse_jacobian(residuals), sc.jacobian(residuals)(w)
  pattern = sparse.output_sparsities[0]
  assert pattern is not None and pattern.nnz < dense.size / 3  # each interval touches its own stage only
  np.testing.assert_allclose(sparse(w), dense[pattern.rows, pattern.cols], rtol=1e-13, atol=1e-14)
  assert np.count_nonzero(dense) <= pattern.nnz
  eps = 1e-6
  fd = np.stack([(residuals(w + e) - residuals(w - e)) / (2 * eps) for e in np.eye(size) * eps], axis=1)
  np.testing.assert_allclose(dense, fd, rtol=1e-6, atol=1e-8)


def test_refusals() -> None:
  with pytest.raises(ValueError, match="points must be"):
    si.Collocation(3, "lobatto")  # ty: ignore[invalid-argument-type]
  with pytest.raises(ValueError, match="degree of at least 1"):
    si.Collocation(0)
  with pytest.raises(ValueError, match="at least 2 nodes"):
    si.Pseudospectral(1)

  @sc.function(2, output="xdot")
  def free(x):
    return -x

  with pytest.raises(ValueError, match="takes the state, then the control"):
    si.Collocation().interval(free, dt=0.1)

  @sc.function(2, output="l")
  def state_only(x):
    return x.sum()

  with pytest.raises(ValueError, match="a running cost takes the model's inputs"):
    si.Collocation().interval(van_der_pol, state_only, dt=0.1)


def test_each_pseudospectral_control_acts_at_its_node() -> None:
  """With ``x' = u`` and the node controls sampled from ``u(t) = a + b t``, the state is quadratic in
  time and the end is exactly ``x0 + a + b / 2``; a control applied at the wrong node, or ``u``
  everywhere, misses it."""

  @sc.function(1, 1, output="xdot")
  def drift(x, u):
    return u + 0.0 * x

  a, b, x0 = 0.7, -1.3, np.array([0.2])
  interval = si.Pseudospectral(4).interval(drift, dt=1.0, name="ramp")
  n_states = interval.state_times.size
  controls = a + b * interval.control_times

  def residual(v: np.ndarray) -> np.ndarray:
    z = np.concatenate([v[:n_states], controls])
    return np.asarray(interval.fn(x0, np.array([a]), z, v[n_states:]))

  solution = root(residual, np.zeros(n_states + 1), method="hybr", options={"xtol": 1e-14})
  np.testing.assert_allclose(solution.x[n_states:], x0 + a + b / 2, rtol=1e-12)
  np.testing.assert_allclose(solution.x[:n_states], x0 + a * interval.state_times + b * interval.state_times**2 / 2, rtol=1e-11)
