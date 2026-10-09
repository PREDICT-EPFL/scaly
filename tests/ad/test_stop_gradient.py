"""Stopped paths keep NumPy primal values while derivatives hold those paths fixed."""

from __future__ import annotations

import numpy as np
import pytest
from typing import Literal

import scaly as sc
from scaly.codegen import render_c_source


@pytest.mark.parametrize("shape", [(), (3,), (2, 0)])
@pytest.mark.parametrize("nseed", [0, 1, 3])
def test_stop_gradient_modes_and_sparsity(shape: tuple[int, ...], nseed: int) -> None:
  point = np.linspace(0.3, 1.2, int(np.prod(shape))).reshape(shape)
  seeds = np.arange(nseed * point.size, dtype=float).reshape((nseed, *shape))

  @sc.function(sc.arg("x", shape), outputs=sc.group(sc.arg("value"), sc.arg("many"), sc.arg("adj"), sc.arg("hess")))
  def fn(x: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr]:
    frozen = sc.stop_gradient(x)
    cost = sc.sumsqr(x * frozen)
    out = x * x + sc.stop_gradient(x**3)
    np.testing.assert_array_equal(sc.jacobian_sparsity(out, x).to_mask(), np.eye(x.size, dtype=bool))
    assert sc.jacobian_sparsity(frozen, x).nnz == 0
    return out, sc.jvp_many(out, x, sc.const(seeds)), sc.gradient(cost, x), sc.hessian(cost, x)

  value, many, adj, hess = fn(point)
  np.testing.assert_allclose(value, point**2 + point**3)
  np.testing.assert_allclose(many, seeds * (2 * point))
  np.testing.assert_allclose(adj, 2 * point**3)
  np.testing.assert_allclose(hess, np.diag(2 * point.reshape(-1) ** 2))


@pytest.mark.parametrize("mapped", [False, True])
def test_stop_gradient_inside_calls_and_maps(mapped: bool) -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.arg("out"))
  def piece(x: sc.Expr) -> sc.Expr:
    return x * sc.stop_gradient(x) + sc.stop_gradient(x.floor())

  @sc.function(sc.arg("x", (3, 2)), outputs=sc.group(sc.arg("value"), sc.arg("jac"), sc.arg("sparse"), sc.arg("adj")))
  def fn(x: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr]:
    out = sc.vmap(piece, 3)(x) if mapped else sc.stack([piece(x[i]) for i in range(3)])
    return out, sc.jacobian(out, x), sc.sparse_jacobian(out, x).to_dense(), sc.vjp((out,), (x,), (sc.const(np.ones(out.shape)),))[0]

  point = np.arange(6.0).reshape(3, 2) + 0.3
  value, jac, sparse, adj = fn(point)
  np.testing.assert_allclose(value, point**2 + np.floor(point))
  np.testing.assert_allclose(jac, np.diag(point.reshape(-1)))
  np.testing.assert_allclose(sparse, jac)
  np.testing.assert_allclose(adj, point)


@pytest.mark.parametrize("lowering", ["scalar", "block"])
def test_stop_gradient_primal_c_is_identical(lowering: Literal["scalar", "block"]) -> None:
  def build(stopped: bool) -> sc.Function:
    @sc.function(sc.arg("x", 3), outputs=sc.arg("y"), name="stopped_primal")
    def fn(x: sc.Expr) -> sc.Expr:
      intermediate = (x.sin() + x * x).reshape((3,))
      if stopped:
        intermediate = sc.stop_gradient(sc.stop_gradient(intermediate))
      return (intermediate * intermediate + intermediate).with_lowering(lowering)

    return fn

  ordinary, stopped = build(False), build(True)
  point = np.array([0.3, 0.7, 1.2])
  intermediate = np.sin(point) + point**2
  np.testing.assert_allclose(stopped(point), intermediate**2 + intermediate)
  np.testing.assert_array_equal(stopped(point), ordinary(point))
  assert render_c_source(stopped) == render_c_source(ordinary)


@pytest.mark.parametrize("stop_inside", [False, True])
def test_stop_gradient_mapped_primal_c_is_identical(stop_inside: bool) -> None:
  def build(stopped: bool) -> sc.Function:
    @sc.function(sc.arg("a", 2), outputs=sc.arg("y"), name="frozen_piece")
    def piece(a: sc.Expr) -> sc.Expr:
      value = a.sin() + a * a
      return sc.stop_gradient(value) if stopped and stop_inside else value

    @sc.function(sc.arg("x", 6), outputs=sc.group(sc.arg("y"), sc.arg("jac")), name="frozen_mapped")
    def fn(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
      p = x[1:5].reshape((2, 2))
      actual = sc.stop_gradient(p) if stopped and not stop_inside else p
      out = sc.vmap(piece, 2)(actual)
      return out, sc.jacobian(out, x)

    return fn

  ordinary, stopped = build(False), build(True)
  point = np.linspace(0.3, 1.2, 6)
  value, jac = stopped(point)
  np.testing.assert_allclose(value, (np.sin(point[1:5]) + point[1:5] ** 2).reshape(2, 2))
  np.testing.assert_array_equal(jac, np.zeros((4, 6)))
  np.testing.assert_array_equal(value, ordinary(point)[0])
  ordinary_value = ordinary.factory("frozen_primal", ["x"], ["y"])
  stopped_value = stopped.factory("frozen_primal", ["x"], ["y"])
  assert render_c_source(stopped_value) == render_c_source(ordinary_value)


def test_reverse_ignores_nonsmooth_operations_of_stopped_inputs() -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.group(sc.arg("value"), sc.arg("grad"), sc.arg("jac"), sc.arg("sparse")))
  def fn(x: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr]:
    scale = sc.maximum(sc.stop_gradient(sc.sumsqr(x)), 1e-6)
    cost = (x / scale).sum()
    return cost, sc.gradient(cost, x), sc.jacobian(cost, x), sc.sparse_jacobian(cost, x).to_dense()

  point = np.array([0.3, 2.0])
  value, grad, jac, sparse = fn(point)
  scale = max(np.sum(point**2), 1e-6)
  np.testing.assert_allclose(value, np.sum(point) / scale)
  expected = np.ones(2) / scale
  np.testing.assert_allclose(grad, expected)
  np.testing.assert_allclose(jac, expected.reshape(1, 2))
  np.testing.assert_allclose(sparse, jac)


@pytest.mark.parametrize("mapped", [False, True])
def test_reverse_ignores_nonsmooth_calls_of_stopped_inputs(mapped: bool) -> None:
  @sc.function(sc.arg("a", 2), outputs=sc.arg("out"))
  def rounded(a: sc.Expr) -> sc.Expr:
    return a.floor() * 2.0

  @sc.function(sc.arg("x", (2, 2)), outputs=sc.group(sc.arg("value"), sc.arg("grad"), sc.arg("jac"), sc.arg("sparse")))
  def fn(x: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr]:
    frozen = sc.stop_gradient(x)
    value = sc.vmap(rounded, 2)(frozen) if mapped else sc.stack([rounded(frozen[i]) for i in range(2)])
    cost = (x * value).sum()
    return cost, sc.gradient(cost, x), sc.jacobian(cost, x), sc.sparse_jacobian(cost, x).to_dense()

  point = np.array([[0.3, 2.5], [1.7, -0.4]])
  value, grad, jac, sparse = fn(point)
  expected = 2 * np.floor(point)
  np.testing.assert_allclose(value, np.sum(point * expected))
  np.testing.assert_allclose(grad, expected)
  np.testing.assert_allclose(jac, expected.reshape(1, 4))
  np.testing.assert_allclose(sparse, jac)


@pytest.mark.solver("sqp")
def test_reverse_ignores_a_solver_with_stopped_parameters() -> None:
  @sc.problem(vars=sc.arg("y", ()), params=sc.arg("p", ()))
  def inner_problem(y: sc.Expr, p: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    return sc.ProblemSpec(minimize=(y - p) ** 2)

  inner = sc.solver(inner_problem, "sqp", options={"tol": 1e-9, "dual_tol": 1e-9, "qp_tol": 1e-10})

  @sc.function(sc.arg("x", ()), outputs=sc.group(sc.arg("value"), sc.arg("grad"), sc.arg("jac")))
  def fn(x: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    cost = x * inner(sc.stop_gradient(x))[0]
    return cost, sc.gradient(cost, x), sc.jacobian(cost, x)

  point = np.array(1.5)
  value, grad, jac = fn(point)
  np.testing.assert_allclose(value, point**2, atol=1e-7)
  np.testing.assert_allclose(grad, point, atol=1e-7)
  np.testing.assert_allclose(jac, point.reshape(1, 1), atol=1e-7)
