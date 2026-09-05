"""Tests for typed function tree declarations and calls."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import alloy as al
from alloy.codegen import render_c_api_header
from alloy.function import Tree


@al.function(al.L("x", 3), al.G(al.L("first", ...), al.L("second", 3)))
def duplicate(x: al.Expr) -> tuple[al.Expr, al.Expr]:
  return x, x


@al.function(al.G(al.L("x", 3), al.L("y", 3)), al.L("prod", ...))
def multiply(inputs: tuple[al.Expr, al.Expr]) -> al.Expr:
  x, y = inputs
  return x * y


@al.function(al.L("x", 3), al.L("square", ...))
def square(x: al.Expr) -> al.Expr:
  return multiply.symbolic_call(duplicate.symbolic_call(x))


@al.function(al.G(al.L("x", 3), al.L("p", ())), al.L("f", ...))
def cost(inputs: tuple[al.Expr, al.Expr]) -> al.Expr:
  x, p = inputs
  return (x * x).sum() * p


@al.function(
  al.G(al.G(al.L("state", 4), al.L("u", 2)), al.G(al.L("pw", 10), al.L("physics", 3), al.L("dt", ()))),
  al.L("next", ...),
)
def step(inputs: tuple[tuple[al.Expr, al.Expr], tuple[al.Expr, al.Expr, al.Expr]]) -> al.Expr:
  (state, _u), (_pw, _physics, _dt) = inputs
  return state


@al.function(al.G(al.L("state", 4), al.L("u", 2), al.L("pw", 10), al.L("physics", 3), al.L("dt", ())), al.L("next", ...))
def step_flat(inputs: tuple[al.Expr, al.Expr, al.Expr, al.Expr, al.Expr]) -> al.Expr:
  state, _u, _pw, _physics, _dt = inputs
  return state


def test_leaf_shapes_and_group_widths() -> None:
  assert al.L("x", 3).shapes == ((3,),)
  assert al.L("P", (2, 2)).shapes == ((2, 2),)
  assert al.L("s", ()).shapes == ((),)
  with pytest.raises(TypeError, match="inferred"):
    _ = al.L("f", ...).shapes

  assert al.G(al.L("a", 1), al.L("b", 1)).names == ("a", "b")
  eight = cast(Any, al.G)(*(al.L(f"x{i}", 1) for i in range(8)))
  assert eight.size == 8
  with pytest.raises(TypeError, match="2 to 8"):
    cast(Any, al.G)(al.L("a", 1))
  with pytest.raises(TypeError, match="nest for more"):
    cast(Any, al.G)(*(al.L(f"x{i}", 1) for i in range(9)))

  nested = al.G(al.G(al.L("a", 1), al.L("b", 1)), al.L("c", 1))
  assert nested.names == ("a", "b", "c")
  assert nested.size == 3


def test_tree_names_are_unique_and_symbols_follow_structure() -> None:
  with pytest.raises(ValueError, match="duplicate"):
    al.G(al.L("x", 3), al.L("x", 3))
  with pytest.raises(ValueError, match="duplicate"):
    al.G(al.G(al.L("x", 3), al.L("y", 3)), al.L("x", 1))

  symbols = al.G(al.G(al.L("a", 1), al.L("b", 2)), al.L("c", ())).symbols()
  assert isinstance(symbols, tuple)
  assert isinstance(symbols[0], tuple)
  assert symbols[0][1].name == "b"
  assert symbols[0][1].shape == (2,)
  assert symbols[1].shape == ()

  tree = al.G(al.L("u", 2), al.L("s", ...))
  assert tree.relabel("lam:").names == ("lam:u", "lam:s")
  resolved = tree.with_types((al.TensorType((2,)), al.TensorType((1,))))
  assert resolved.shapes == ((2,), (1,))
  assert isinstance(resolved, Tree)


def test_function_declarations_own_names_shapes_and_structure() -> None:
  assert duplicate.input_names == ("x",)
  assert duplicate.output_names == ("first", "second")
  assert duplicate.output_shapes == ((3,), (3,))
  assert multiply.output_shapes == ((3,),)
  assert step.input_names == step_flat.input_names == ("state", "u", "pw", "physics", "dt")
  assert step.input_shapes == step_flat.input_shapes

  duplicated = duplicate.numerical_call(np.arange(3.0))
  assert isinstance(duplicated, tuple)
  assert len(duplicated) == 2
  np.testing.assert_array_equal(duplicated[0], np.arange(3.0))

  stepped = step.numerical_call(
    (
      (np.arange(4.0), np.arange(2.0)),
      (np.arange(10.0), np.arange(3.0), np.array(0.1)),
    )
  )
  np.testing.assert_array_equal(stepped, np.arange(4.0))
  assert square.output_shapes == ((3,),)

  assert render_c_api_header(step) == render_c_api_header(step_flat).replace("step_flat", "step")


def test_function_call_validation_uses_declared_tree() -> None:
  with pytest.raises(ValueError, match="expected shape"):
    duplicate.numerical_call(np.zeros(4))
  with pytest.raises(ValueError, match="expected shape"):
    multiply.symbolic_call((al.sym("x", 3), al.sym("y", 2)))
  with pytest.raises(ValueError, match="declared structure"):
    step.numerical_call(cast(Any, (np.zeros(4), np.zeros(2), np.zeros(10), np.zeros(3), np.array(0.1))))
  with pytest.raises(ValueError, match="declared structure"):
    step_flat.numerical_call(cast(Any, ((np.zeros(4), np.zeros(2)), (np.zeros(10), np.zeros(3), np.array(0.1)))))

  with pytest.raises(TypeError, match="expected shape"):
    al.function(al.L("x", 3), al.L("y", 2))(lambda x: x)
  with pytest.raises((TypeError, ValueError), match="declared 2 outputs|declared structure"):
    al.function(al.L("x", 3), al.G(al.L("a", ...), al.L("b", ...)))(cast(Any, lambda x: x))


def test_derivatives_preserve_source_trees() -> None:
  grad = al.gradient(cost, "f", "x")
  hess = al.hessian(cost, "f", "x")
  jac = al.jacobian(square, "square", "x")
  fwd = al.forward(cost, "f", "x")
  adj = al.adjoint(square, "square", "x")
  hess_l = al.lagrangian_hessian(duplicate, "x")

  assert grad.input_names == ("x", "p")
  assert grad.output_names == ("grad_f_x",)
  assert grad.output_shapes == ((3,),)
  assert hess.output_names == ("hess_f_x_x",)
  assert hess.output_shapes == ((3, 3),)
  assert jac.output_shapes == ((3, 3),)
  assert fwd.input_names == ("x", "p", "fwd:x")
  assert fwd.output_shapes == ((),)
  assert adj.input_names == ("x", "lam:square")
  assert adj.output_shapes == ((3,),)
  assert hess_l.input_names == ("x", "lam:first", "lam:second")
  assert hess_l.output_shapes == ((3, 3),)

  np.testing.assert_allclose(
    fwd.numerical_call(((np.arange(3.0), np.array(2.0)), np.ones(3))),
    12.0,
  )

  with pytest.raises(ValueError, match=r"declared \('x', 'p'\)"):
    al.gradient(cost, "f", "z")
  with pytest.raises(ValueError, match=r"declared \('f',\)"):
    al.gradient(cost, "h", "x")
  with pytest.raises(ValueError, match="scalar"):
    al.gradient(duplicate, "first", "x")
  with pytest.raises(ValueError, match="scalar"):
    al.hessian(duplicate, "first", "x")
  with pytest.raises(ValueError, match="unknown name 'z'"):
    al.forward(cost, "f", "z")
  with pytest.raises(ValueError, match="unknown name 'y'"):
    al.lagrangian_hessian(duplicate, "y")


def test_seeded_derivatives_accept_nondifferentiable_leaves() -> None:
  @al.function(al.L("x", al.TensorType((2,), diff=False)), al.L("y", ...))
  def frozen_square(x: al.Expr) -> al.Expr:
    return x * x

  fwd = al.forward(frozen_square, "y", "x")
  adj = al.adjoint(frozen_square, "y", "x")
  x = np.array([2.0, 3.0])

  assert fwd.input_tree.types[-1].diff
  assert adj.input_tree.types[-1].diff
  np.testing.assert_array_equal(fwd((x, np.ones(2))), 2.0 * x)
  np.testing.assert_array_equal(adj((x, np.ones(2))), 2.0 * x)


def test_lagrangian_hessians_accept_constant_output_leaves() -> None:
  @al.function(al.L("x", 2), al.G(al.L("cost", ...), al.L("constant", ...)))
  def objective(x: al.Expr) -> tuple[al.Expr, al.Expr]:
    return (x * x).sum(), al.const(1.0)

  dense = al.lagrangian_hessian(objective, "x")
  sparse = al.sparse_lagrangian_hessian(objective, "x")
  inputs = (np.array([2.0, 3.0]), (np.array(1.5), np.array(7.0)))

  assert dense.input_tree.types[-1].diff
  assert sparse.input_tree.types[-1].diff
  np.testing.assert_array_equal(dense(inputs), 3.0 * np.eye(2))
  np.testing.assert_array_equal(sparse(inputs), np.array([3.0, 3.0]))


def test_call_dispatches_on_leaf_kind() -> None:
  """``__call__`` routes to ``symbolic_call`` or ``numerical_call`` by the leaves it is given."""
  xv, yv = np.arange(3.0), np.ones(3)

  np.testing.assert_allclose(multiply((xv, yv)), multiply.numerical_call((xv, yv)))
  assert multiply((al.sym("a", 3), al.sym("b", 3))).op == al.ExprOp.CALL

  # Only G introduces a tuple: a one-leaf tree is the bare value on both sides.
  assert not isinstance(multiply((xv, yv)), tuple)
  first, second = duplicate(xv)
  np.testing.assert_allclose(first, xv)
  np.testing.assert_allclose(second, xv)

  with pytest.raises(TypeError, match="mix Expr and numerical leaves"):
    multiply(cast(Any, (al.sym("a", 3), yv)))

  # A constant leaf has to be lifted for the symbolic reading; the dispatch never guesses.
  assert multiply((al.const(xv), al.sym("b", 3))).op == al.ExprOp.CALL


def test_call_dispatch_predicates_ignore_structure() -> None:
  """A wrongly-shaped tree is reported against the declared names, not as a kind mismatch."""
  tree = al.G(al.L("x", 3), al.L("y", 3))
  assert tree.is_symbolic((al.sym("a", 3), al.sym("b", 3)))
  assert not tree.is_numerical((al.sym("a", 3), al.sym("b", 3)))
  assert tree.is_numerical((np.zeros(3), np.zeros(3)))
  assert not tree.is_symbolic((np.zeros(3), np.zeros(3)))
  # Neither predicate consults the structure, so the flattener owns that message.
  assert tree.is_symbolic(cast(Any, (al.sym("a", 3),)))
  with pytest.raises(ValueError, match="does not have the declared structure"):
    multiply(cast(Any, (np.zeros(3),)))
