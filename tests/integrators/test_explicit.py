"""Explicit Runge-Kutta maps: convergence order, agreement with a NumPy step, derivatives."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

import scaly as sc
from scaly import integrators as si
from scaly.integrators.explicit import _controller, _power_of_two
from scaly.ir.expr import ExprOp, topo

A = np.array([[0.0, 1.0], [-4.0, -0.3]])
B = np.array([[0.0], [1.0]])


@sc.function(2, 1, output="xdot")
def damped(x, u):
  return sc.const(A) @ x + sc.const(B) @ u


@sc.function(2, output="xdot")
def van_der_pol(x):
  return sc.stack([x[1], (1 - x[0] * x[0]) * x[1] - x[0]])


@sc.function(3, sc.G(sc.L("u", 2), sc.L("gain", (2, 2))), (), output="xdot")
def mixed(x, inputs, c):
  """A model whose second slot is a group, with a matrix leaf, and whose third is a scalar."""
  u, gain = inputs
  v = gain @ u
  return sc.stack([x[1] * x[2].sin() + v[0], -c * x[0] + v[1] * x[2], (x[0] * x[1]).tanh()])


def mixed_np(x, u, gain, c):
  v = gain @ u
  return np.array([x[1] * np.sin(x[2]) + v[0], -c * x[0] + v[1] * x[2], np.tanh(x[0] * x[1])])


def rk_np(tab, f, x, h):
  ks = []
  for i in range(tab.stages):
    ks.append(f(x + h * sum((tab.a[i, j] * ks[j] for j in range(i)), np.zeros_like(x))))
  return x + h * sum(b * k for b, k in zip(tab.b, ks))


def _order(errors: list[float]) -> float:
  return float(np.log2(errors[0] / errors[1]))


EXPLICIT = sorted(name for name, tab in si.TABLEAUS.items() if tab.explicit)


@pytest.mark.parametrize("method", EXPLICIT)
def test_linear_system_converges_at_the_method_order(method: str) -> None:
  tab = si.TABLEAUS[method]
  step = si.explicit(damped, method, dt=None)
  x0, u, T = np.array([1.0, 0.5]), np.array([0.7]), 1.0
  big = expm(np.block([[A, B], [np.zeros((1, 3))]]) * T)
  exact = big[:2, :2] @ x0 + big[:2, 2:] @ u
  errors = []
  for n in (8, 16) if tab.order >= 4 else (32, 64):
    x = x0
    for _ in range(n):
      x = step(x, u, np.array(T / n))
    errors.append(float(np.abs(x - exact).max()))
  if method == "tsit5":  # its leading error constant is small by design: the rate reads 5.1 to 5.3 here
    assert tab.order - 0.15 < _order(errors) < tab.order + 0.5, errors
  else:
    assert abs(_order(errors) - tab.order) < 0.15, errors


@pytest.mark.parametrize("method", ["heun", "ssprk3", "rk4", "rk38", "dopri5", "tsit5"])
def test_van_der_pol_converges_at_the_method_order(method: str) -> None:
  tab = si.TABLEAUS[method]
  T, x0 = 1.0, np.array([2.0, 0.0])
  exact = solve_ivp(lambda t, x: [x[1], (1 - x[0] ** 2) * x[1] - x[0]], (0, T), x0, method="DOP853", rtol=1e-13, atol=1e-14).y[:, -1]
  errors = []
  for n in (16, 32) if method in ("dopri5", "tsit5") else (64, 128):
    x = si.explicit(van_der_pol, method, dt=T, steps=n)(x0)  # a scan: more substeps than UNROLL_STEPS
    errors.append(float(np.abs(x - exact).max()))
  if method in ("dopri5", "tsit5"):  # their leading error constants are tiny: the rate reads about 6 until rounding takes over
    assert _order(errors) > tab.order - 0.15, errors
  else:
    assert abs(_order(errors) - tab.order) < 0.15, errors


@pytest.mark.parametrize("method", ["euler", "rk3", "rk4", "bs32"])
def test_matches_a_numpy_step_on_a_model_with_grouped_matrix_and_scalar_inputs(method: str) -> None:
  rng = np.random.default_rng(0)
  x, u, gain, c = rng.normal(size=3), rng.normal(size=2), rng.normal(size=(2, 2)), np.array(1.3)
  step = si.explicit(mixed, method, dt=0.1)
  assert step.input_names == ("x", "u", "gain", "c") and step.output_names == ("xnext",)
  expected = rk_np(si.TABLEAUS[method], lambda xs: mixed_np(xs, u, gain, c), x, 0.1)
  np.testing.assert_allclose(step(x, (u, gain), c), expected, rtol=1e-14, atol=1e-14)


def test_looped_substeps_equal_repeated_steps() -> None:
  rng = np.random.default_rng(1)
  x, u, gain, c = rng.normal(size=3), rng.normal(size=2), rng.normal(size=(2, 2)), np.array(0.4)
  n = si.UNROLL_STEPS + 3
  looped = si.explicit(mixed, "rk4", dt=0.05 * n, steps=n)(x, (u, gain), c)
  one = si.explicit(mixed, "rk4", dt=0.05)
  for _ in range(n):
    x = one(x, (u, gain), c)
  np.testing.assert_allclose(looped, x, rtol=1e-13, atol=1e-14)


def test_a_dt_input_equals_the_folded_interval() -> None:
  x, u = np.array([0.3, -0.2]), np.array([0.1])
  for steps in (1, 3, si.UNROLL_STEPS + 1):
    fixed = si.rk4(damped, dt=0.2, steps=steps)
    free = si.rk4(damped, dt=None, steps=steps)
    assert free.input_names == ("x", "u", "dt")
    np.testing.assert_allclose(free(x, u, np.array(0.2)), fixed(x, u), rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(free(x=x, u=u, dt=np.array(0.2)), fixed(x, u), rtol=1e-15, atol=1e-15)


def _fd_jacobian(fn, args: list[np.ndarray], k: int, eps: float = 1e-6) -> np.ndarray:
  cols = []
  for i in range(args[k].size):
    plus, minus = [a.copy() for a in args], [a.copy() for a in args]
    plus[k].flat[i] += eps
    minus[k].flat[i] -= eps
    cols.append((fn(*plus) - fn(*minus)) / (2 * eps))
  return np.stack(cols, axis=-1).reshape(-1, args[k].size)


@pytest.mark.parametrize("steps", [2, si.UNROLL_STEPS + 2])
def test_jacobians_match_differences(steps: int) -> None:
  step = si.explicit(van_der_pol, "dopri5", dt=None, steps=steps)
  args = [np.array([1.2, -0.4]), np.array(0.3)]
  for wrt, k in (("x", 0), ("dt", 1)):
    jac = sc.jacobian(step, wrt)
    np.testing.assert_allclose(jac(*args), _fd_jacobian(step, args, k), rtol=1e-6, atol=1e-8)


def test_reverse_mode_matches_forward_mode() -> None:
  step = si.rk4(van_der_pol, dt=0.4, steps=si.UNROLL_STEPS + 1)

  @sc.function(2, output="cost")
  def cost(x):
    y = step(x)
    return (y * y).sum()

  x = np.array([0.7, 0.2])
  np.testing.assert_allclose(sc.gradient(cost)(x), 2 * sc.jacobian(step)(x).T @ step(x), rtol=1e-12)


def test_each_stage_is_one_call_of_the_model() -> None:
  step = si.rk4(van_der_pol, dt=0.1)
  calls = [e for e in topo(step.outputs) if e.op == ExprOp.CALL]
  assert len(calls) == 4 and all(e.attrs["callee"] is van_der_pol for e in calls)
  dopri = si.explicit(van_der_pol, "dopri5", dt=0.1)  # the seventh stage feeds only the error estimate
  assert sum(e.op == ExprOp.CALL for e in topo(dopri.outputs)) == 6


def test_names_and_templates() -> None:
  assert si.rk4(damped, dt=0.1).name == "damped_rk4"
  assert si.explicit(damped, "heun", dt=0.1, name="plant").name == "plant"

  @sc.function(output="xdot")
  def decay(x, k):
    return -k * x

  step = si.rk4(decay, dt=0.5)
  assert not step.is_concrete and step.name == "decay_rk4"
  x = np.array([1.0, 2.0, 3.0])
  np.testing.assert_allclose(step(x, np.array(2.0)), x * (1 - 1 + 1 / 2 - 1 / 6 + 1 / 24), rtol=1e-15)
  assert list(step.instances) == ["decay__3_s_rk4"]
  named = si.rk4(decay, dt=None, name="dec")
  np.testing.assert_allclose(named(x, np.array(2.0), np.array(0.5)), step(x, np.array(2.0)), rtol=1e-15)
  assert list(named.instances) == ["dec__3_s"]


def test_refusals() -> None:
  with pytest.raises(ValueError, match="implicit method"):
    si.explicit(damped, si.Tableau(np.array([[1.0]]), np.array([1.0]), np.array([1.0]), 1), dt=0.1)
  with pytest.raises(ValueError, match="dt must be positive"):
    si.rk4(damped, dt=0.0)
  with pytest.raises(ValueError, match="steps must be a positive integer"):
    si.rk4(damped, dt=0.1, steps=0)
  with pytest.raises(TypeError, match="must be an sc.Function"):
    si.rk4(lambda x: x, dt=0.1)  # ty: ignore[invalid-argument-type]


def test_rk4_generates_the_code_of_a_hand_written_rk4() -> None:
  """The coefficients fold as a person would write them (``h/6 (k1 + 2 k2 + 2 k3 + k4)``), so a
  vmapped shooting defect is the same C as `examples/opt/nmpc_cartpole.py`'s hand-written RK4."""
  from scaly.codegen import render_c_module

  def ode(x, u):
    return sc.stack([x[1], -x[0].sin() + u[0] * x[0].cos()])

  def rk4(x, u, h):
    k1 = ode(x, u)
    k2 = ode(x + 0.5 * h * k1, u)
    k3 = ode(x + 0.5 * h * k2, u)
    k4 = ode(x + h * k3, u)
    return x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

  @sc.function(2, 1, output="xdot")
  def pendulum(x, u):
    return ode(x, u)

  library = si.rk4(pendulum, dt=0.1, steps=2)

  def horizon(step):
    @sc.function(2, 1, 2, output="eq", name="defect")
    def defect(x, u, xnext):
      return step(x, u) - xnext

    @sc.function(22, 10, output="eq", name="horizon")
    def fn(xs, us):
      return sc.vmap(defect, 10, [(xs, 0, 2), (us, 0, 1), (xs, 2, 2)])

    return render_c_module(fn).body

  assert horizon(library) == horizon(lambda x, u: rk4(rk4(x, u, 0.05), u, 0.05))


def vdp_np(t, x, u=0.3, mu=3.0):
  return [x[1], mu * (1 - x[0] ** 2) * x[1] - x[0] + u]


@sc.function(2, 1, output="xdot")
def stiffish(x, u):
  return sc.stack([x[1], 3.0 * (1 - x[0] * x[0]) * x[1] - x[0] + u[0]])


@pytest.mark.parametrize("pair", ["dopri5", "tsit5", "bs32"])
def test_adaptive_error_follows_the_tolerance(pair: str) -> None:
  x0, u, T = np.array([2.0, 0.0]), np.array([0.3]), 2.0
  exact = solve_ivp(vdp_np, (0, T), x0, method="DOP853", rtol=1e-13, atol=1e-14).y[:, -1]
  errors = []
  for tol in (1e-4, 1e-6, 1e-8):
    step = si.adaptive(stiffish, pair, rtol=tol, atol=tol * 1e-2, name=f"{pair}_{int(-np.log10(tol))}")
    errors.append(float(np.abs(step(x0, u, np.array(T)) - exact).max()))
  bound = 0.5 if pair == "dopri5" else 5.0  # DOPRI5 keeps its fifth-order solution, well inside the fourth-order estimate
  assert all(e < bound * tol for e, tol in zip(errors, (1e-4, 1e-6, 1e-8), strict=True)), errors
  assert errors[0] > errors[1] > errors[2]
  # A first step the length of the whole interval must be rejected, and the result is as accurate.
  whole = si.adaptive(stiffish, pair, rtol=1e-8, atol=1e-10, h0=T, name=f"{pair}_whole")
  assert np.abs(whole(x0, u, np.array(T)) - exact).max() < bound * 1e-8


def test_adaptive_runs_out_of_steps_into_nan_and_a_folded_interval() -> None:
  x0, u = np.array([2.0, 0.0]), np.array([0.3])
  assert np.isnan(si.adaptive(stiffish, max_steps=3, name="short")(x0, u, np.array(2.0))).all()
  folded = si.adaptive(stiffish, dt=0.5, rtol=1e-10, atol=1e-12, name="folded")
  free = si.adaptive(stiffish, rtol=1e-10, atol=1e-12, name="free")
  assert folded.input_names == ("x", "u") and free.input_names == ("x", "u", "dt")
  np.testing.assert_allclose(folded(x0, u), free(x0, u, np.array(0.5)), rtol=1e-14)
  np.testing.assert_allclose(free(x0, u, np.array(0.0)), x0)  # no time to go: no step taken
  given = si.adaptive(stiffish, rtol=1e-10, atol=1e-12, h0=1e-3, name="given_h0")
  np.testing.assert_allclose(given(x0, u, np.array(0.5)), free(x0, u, np.array(0.5)), rtol=1e-8)
  assert ExprOp.WHILE in {e.op for e in topo(free.outputs)}


def test_adaptive_derivatives_hold_the_step_sequence() -> None:
  step = si.adaptive(stiffish, rtol=1e-9, atol=1e-11, name="adaptive_d")
  x0, u, T = np.array([1.2, -0.4]), np.array([0.3]), np.array(1.3)
  np.testing.assert_allclose(sc.jacobian(step, "x")(x0, u, T), _fd_jacobian(step, [x0, u, T], 0), rtol=1e-5, atol=1e-7)
  np.testing.assert_allclose(sc.jacobian(step, "u")(x0, u, T), _fd_jacobian(step, [x0, u, T], 1), rtol=1e-5, atol=1e-7)
  end = step(x0, u, T)
  # Only the last step moves with dt, and its derivative in its length is f(x(T)) to the step's accuracy.
  np.testing.assert_allclose(sc.jacobian(step, "dt")(x0, u, T).ravel(), stiffish(end, u), rtol=1e-7)

  @sc.function(2, 1, (), output="c", name="adaptive_cost")
  def cost(x, u, t):
    y = step(x, u, t)
    return (y * y).sum()

  np.testing.assert_allclose(sc.gradient(cost, "x")(x0, u, T), 2 * sc.jacobian(step, "x")(x0, u, T).T @ end, rtol=1e-11)


def test_a_step_grows_at_most_fivefold() -> None:
  """With a constant derivative both solutions of the pair are exact, the error is zero, and every
  step is five times the last: from 2^-20, the tenth step is the first to reach t = 1."""

  @sc.function(1, output="xdot")
  def drift(x):
    return 0.0 * x + 1.0

  steps_needed = next(k for k in range(1, 40) if 2.0**-20 * (5**k - 1) / 4 >= 1.0)
  assert steps_needed == 10
  for budget, reaches in ((steps_needed - 1, False), (steps_needed, True)):
    step = si.adaptive(drift, h0=2.0**-20, max_steps=budget, name=f"drift_{budget}")
    np.testing.assert_allclose(step(np.zeros(1), np.array(1.0)), [1.0] if reaches else [np.nan])


def test_adaptive_refusals() -> None:
  with pytest.raises(ValueError, match="explicit embedded pair"):
    si.adaptive(stiffish, "rk4")
  with pytest.raises(ValueError, match="rtol and atol must be positive"):
    si.adaptive(stiffish, rtol=0.0)
  with pytest.raises(ValueError, match="max_steps a positive integer"):
    si.adaptive(stiffish, max_steps=0)


@sc.function(2, output="xdot")
def pendulum(x):
  return sc.stack([x[1], -x[0].sin()])


def _energy(x: np.ndarray) -> float:
  return 0.5 * x[1] ** 2 - np.cos(x[0])


def test_stormer_verlet_keeps_the_energy_where_rk4_loses_it() -> None:
  h, chunk, chunks = 0.3, 1000, 100
  verlet = si.symplectic(pendulum, split=1, dt=h * chunk, steps=chunk)
  euler = si.symplectic(pendulum, "symplectic_euler", split=1, dt=h * chunk, steps=chunk)
  rk4 = si.rk4(pendulum, dt=h * chunk, steps=chunk)
  x0 = np.array([1.5, 0.0])
  e0 = _energy(x0)
  xv, xe, xr, worst, worst_euler = x0, x0, x0, 0.0, 0.0
  for _ in range(chunks):
    xv, xe, xr = verlet(xv), euler(xe), rk4(xr)
    worst, worst_euler = max(worst, abs(_energy(xv) - e0)), max(worst_euler, abs(_energy(xe) - e0))
  assert worst < 0.03  # bounded over 1e5 steps (the potential spans 2)
  assert worst_euler < 0.3  # first order, so larger, but bounded too: explicit Euler would diverge
  assert abs(_energy(xr) - e0) > 0.3  # RK4's dissipation accumulates


@pytest.mark.parametrize(("method", "order"), [("stormer_verlet", 2), ("symplectic_euler", 1)])
def test_symplectic_methods_converge_at_their_order(method: Literal["stormer_verlet", "symplectic_euler"], order: int) -> None:
  x0, T = np.array([1.2, 0.3]), 2.0
  exact = solve_ivp(lambda t, x: [x[1], -np.sin(x[0])], (0, T), x0, method="DOP853", rtol=1e-13, atol=1e-14).y[:, -1]
  errors = [float(np.abs(si.symplectic(pendulum, method, split=1, dt=T, steps=n)(x0) - exact).max()) for n in (200, 400)]
  assert abs(np.log2(errors[0] / errors[1]) - order) < 0.1, errors
  step = si.symplectic(pendulum, method, split=1, dt=0.1)
  np.testing.assert_allclose(sc.jacobian(step, "x")(x0), _fd_jacobian(step, [x0], 0), rtol=1e-7, atol=1e-9)


def test_symplectic_refusals() -> None:
  with pytest.raises(ValueError, match="split must count the positions"):
    si.symplectic(pendulum, split=2, dt=0.1)
  with pytest.raises(ValueError, match="stormer_verlet"):
    si.symplectic(pendulum, "leapfrog", split=1, dt=0.1)  # ty: ignore[invalid-argument-type]


def test_the_controller_law() -> None:
  err = sc.sym("err", 5)
  fn = sc.Function.from_exprs("law", [err], [_controller(err, -0.2)], ["err"], ["factor"])
  (factor,) = fn._flat_numerical_call(np.array([0.0, 1e-12, 1.0, 32.0, 1e30]))
  expected = np.array([5.0, 5.0, 0.9, 0.9 * 32.0**-0.2, 0.2])
  assert np.all(np.abs(np.log2(factor / expected)) < 1 / 1024)  # capped growth, the law, floored shrinkage


def test_the_controller_factor_is_rounded_and_carries_no_derivative() -> None:
  v = sc.sym("v", 3)
  rounded = _power_of_two(v, 1024)
  fn = sc.Function.from_exprs("rounded", [v], [rounded, sc.jacobian(rounded * v, v)], ["v"], ["r", "j"])
  values = np.array([0.37, 1.0, 4.2])
  r, j = fn._flat_numerical_call(values)
  assert np.all(np.abs(np.log2(r / values)) < 1 / 1024)  # within one step of 2^(1/1024)
  np.testing.assert_allclose(j, np.diag(r))  # d(r v)/dv = r: r itself contributes nothing
  np.testing.assert_allclose(np.log2(r) * 1024, np.round(np.log2(r) * 1024), atol=1e-9)
