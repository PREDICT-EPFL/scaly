"""Derivatives of templates: a derived template whose instance for a call is the derivative of the source's instance."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import scaly as sc


def _cost() -> sc.Function[Any, Any, Any, Any]:
  @sc.function(sc.L(), (), output="f")
  def cost(x, p):
    return p * (x * x * x).sum()

  return cost


def _finite_difference(f: Any, x: np.ndarray, *rest: Any, h: float = 1e-6) -> np.ndarray:
  return np.array([(f(x + h * e, *rest) - f(x - h * e, *rest)) / (2 * h) for e in np.eye(x.size)])


def test_the_gradient_of_a_template_is_a_template_built_per_instance() -> None:
  cost = _cost()
  grad = sc.gradient(cost, "f", "x")
  assert type(grad) is sc.Function and not grad.is_concrete and grad.name == "cost_grad_f_x"
  assert cost.instances == {} and grad.instances == {}  # nothing is built until a call
  for n in (3, 5):
    x = np.linspace(-1.0, 1.0, n)
    np.testing.assert_allclose(grad(x, np.array(2.0)), 6.0 * x * x)
    np.testing.assert_allclose(grad(x, np.array(2.0)), _finite_difference(cost, x, np.array(2.0)), rtol=1e-6, atol=1e-8)
  assert list(grad.instances) == ["cost__3_grad_f_x", "cost__5_grad_f_x"]
  assert list(cost.instances) == ["cost__3", "cost__5"]
  # The same derivative of the concrete instance, built directly.
  direct = sc.gradient(cost.instances["cost__3"], "f", "x")
  assert type(direct) is sc.ConcreteFunction and direct.name == "cost__3_grad_f_x"
  np.testing.assert_allclose(direct(np.ones(3), np.array(2.0)), grad(np.ones(3), np.array(2.0)))
  assert repr(grad) == "Function('cost_grad_f_x', derived from 'cost', instances=['cost__3_grad_f_x', 'cost__5_grad_f_x'])"


def test_one_name_is_wrt_and_a_name_left_out_is_the_only_one() -> None:
  cost = _cost()
  x, p = np.array([1.0, 2.0]), np.array(3.0)
  expected = 9.0 * x * x
  for derived in (sc.gradient(cost, "x"), sc.gradient(cost, wrt="x"), sc.gradient(cost, "f", wrt="x"), sc.gradient(cost, of="f", wrt="x")):
    np.testing.assert_allclose(derived(x, p), expected)

  @sc.function
  def energy(q):
    return (q * q).sum()

  grad = sc.gradient(energy)
  np.testing.assert_allclose(grad(np.array([1.0, -2.0])), [2.0, -4.0])
  assert list(grad.instances) == ["energy__2_grad_energy_q"]
  # The template declares its names, so these fail when the derivative is built.
  with pytest.raises(TypeError, match=r"gradient: cost has inputs \('x', 'p'\); say which with wrt="):
    sc.gradient(cost)
  with pytest.raises(ValueError, match=r"gradient: 'f' is an output of cost, not an input; one name is wrt"):
    sc.gradient(cost, "f")

  @sc.function
  def two(a, b):
    return (a * b).sum()

  with pytest.raises(TypeError, match=r"gradient: two__2_2 has inputs \('a', 'b'\); say which with wrt="):
    sc.gradient(two)(np.ones(2), np.ones(2))  # a bare template's names are known only per instance


def test_declared_names_are_checked_now_and_shapes_at_the_call() -> None:
  cost = _cost()
  with pytest.raises(ValueError, match=r"unknown name 'z'; declared \('x', 'p'\)"):
    sc.gradient(cost, "f", "z")
  with pytest.raises(ValueError, match=r"unknown name 'g'; declared \('f',\)"):
    sc.gradient(cost, "g", "x")

  @sc.function(sc.L(), output="y")
  def square(x):
    return x * x

  grad = sc.gradient(square, "y", "x")  # a vector output: refused only once shapes are known
  with pytest.raises(ValueError):
    grad(np.ones(3))
  assert sc.gradient(square, "y", "x")(np.array(3.0)) == 6.0


def test_seeded_derivatives_of_a_template_append_their_argument() -> None:
  cost = _cost()
  x, p, v = np.array([1.0, 2.0, 3.0]), np.array(2.0), np.array([1.0, 0.0, -1.0])
  fwd = sc.forward(cost, "x")
  adj = sc.adjoint(cost, "f", "x")
  np.testing.assert_allclose(fwd(x, p, v), 6.0 * (x * x) @ v)
  np.testing.assert_allclose(adj(x, p, np.array(0.5)), 3.0 * x * x)
  assert fwd.instances["cost__3_fwd_f_x"].input_names == ("x", "p", "fwd:x")
  with pytest.raises(TypeError, match=r"cost_fwd_f_x\(\) takes 3 arguments \(x, p, seed\), got 2"):
    fwd(x, p)

  @sc.function(sc.L(), output=sc.G("u", "v"))
  def pair(x):
    return (x * x).sum(), x * x * x

  lag = sc.lagrangian_hessian(pair, "x")
  h = lag(np.array([1.0, 2.0]), (np.array(2.0), np.array([1.0, 1.0])))
  np.testing.assert_allclose(h, np.diag([4.0 + 6.0, 4.0 + 12.0]))
  sparse = sc.sparse_lagrangian_hessian(pair)
  np.testing.assert_allclose(sparse(np.array([1.0, 2.0]), (np.array(2.0), np.array([1.0, 1.0]))), [10.0, 16.0])
  assert list(lag.instances) == ["pair__2_hess_gamma_x_x"]


def test_a_given_name_is_the_template_name_and_the_instances_spell_their_shapes() -> None:
  cost = _cost()
  hess = sc.hessian(cost, "f", "x", name="curvature")
  np.testing.assert_allclose(hess(np.array([1.0, 2.0]), np.array(1.0)), np.diag([6.0, 12.0]))
  assert hess.name == "curvature" and list(hess.instances) == ["curvature__2"]
  concrete = sc.hessian(cost.instantiate(2, ()), "f", "x", name="curvature_2")
  assert concrete.name == "curvature_2"


def test_derivatives_compose_on_templates() -> None:
  cost = _cost()
  grad = sc.gradient(cost, "f", "x")
  jac = sc.jacobian(grad, wrt="x")
  np.testing.assert_allclose(jac(np.array([1.0, 2.0]), np.array(1.0)), np.diag([6.0, 12.0]))
  assert list(jac.instances) == ["cost__2_grad_f_x_jac_grad_f_x_x"]
  with pytest.raises(sc.NotConcrete, match="cost_grad_f_x leaves the shapes of cost's parameters to its calls"):
    sc.vmap(grad, 2, [(sc.sym("xs", 4), 0, 2), (sc.sym("p", ()), 0, 0)])
  assert type(grad.with_device("host")) is sc.Function


def test_custom_derivative_over_templates_and_template_rules() -> None:
  @sc.function
  def softplus(x):
    return (1.0 + x.exp()).log()

  @sc.function
  def softplus_jvp(x, dx):
    return dx / (1.0 + (-x).exp())

  calls = []

  @sc.function
  def softplus_vjp(x, y, lam):
    calls.append(x.shape)
    return lam / (1.0 + (-x).exp())

  smooth = sc.custom_derivative(softplus, jvp=softplus_jvp, vjp=softplus_vjp)
  assert type(smooth) is sc.Function

  @sc.function(3, output="total")
  def total(x):
    return smooth(x).sum()

  x = np.array([-1.0, 0.0, 2.0])
  np.testing.assert_allclose(sc.gradient(total, "total", "x")(x), 1.0 / (1.0 + np.exp(-x)))
  assert calls == [(3,)] and [name.startswith("softplus__3_cd") for name in smooth.instances] == [True]

  @sc.function(3, output="y")
  def square(x):
    return x * x

  checked = sc.custom_derivative(square, jvp=softplus_jvp)  # a concrete source instantiates template rules now
  assert type(checked) is sc.ConcreteFunction and "softplus_jvp__3_3" in softplus_jvp.instances


def test_a_derived_instance_is_built_once_and_names_carry_through_chains() -> None:
  cost = _cost()
  grad = sc.gradient(cost, "x")
  grad(np.ones(2), np.array(1.0))
  call = grad.symbolic_call(sc.sym("x", 2), sc.sym("p"))  # another argument key, the same binding
  assert list(grad.instances) == ["cost__2_grad_f_x"] and call.attrs["callee"] is grad.instances["cost__2_grad_f_x"]
  named = sc.jacobian(grad, wrt="x", name="curvature")
  np.testing.assert_allclose(named(np.array([1.0, 2.0]), np.array(1.0)), np.diag([6.0, 12.0]))
  assert list(named.instances) == ["curvature__2"]
