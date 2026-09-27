"""Explicit Runge-Kutta maps: convergence order, agreement with a NumPy step, derivatives."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

import scaly as sc
from scaly import integrators as si
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
  assert abs(_order(errors) - tab.order) < 0.15, errors


@pytest.mark.parametrize("method", ["heun", "ssprk3", "rk4", "rk38", "dopri5"])
def test_van_der_pol_converges_at_the_method_order(method: str) -> None:
  tab = si.TABLEAUS[method]
  T, x0 = 1.0, np.array([2.0, 0.0])
  exact = solve_ivp(lambda t, x: [x[1], (1 - x[0] ** 2) * x[1] - x[0]], (0, T), x0, method="DOP853", rtol=1e-13, atol=1e-14).y[:, -1]
  errors = []
  for n in (16, 32) if method == "dopri5" else (64, 128):
    x = si.explicit(van_der_pol, method, dt=T, steps=n)(x0)  # a scan: more substeps than UNROLL_STEPS
    errors.append(float(np.abs(x - exact).max()))
  if method == "dopri5":  # its leading error constant is tiny: the rate reads about 6 until rounding takes over
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
  vmapped shooting defect is the same C as `examples/nmpc_cartpole.py`'s hand-written RK4."""
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
