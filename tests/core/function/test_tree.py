"""Tests for typed function tree declarations and calls."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import render_c_api_header
from scaly.function import Tree
from scaly.function.tree import Hole, is_symbolic_call, param_list


@sc.function(sc.L("x", 3), output=sc.G(sc.L("first", ...), sc.L("second", 3)))
def duplicate(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
  return x, x


@sc.function(sc.G(sc.L("x", 3), sc.L("y", 3)), output=sc.L("prod", ...))
def multiply(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
  x, y = inputs
  return x * y


@sc.function(sc.L("x", 3), output=sc.L("square", ...))
def square(x: sc.Expr) -> sc.Expr:
  return multiply.symbolic_call(duplicate.symbolic_call(x))


@sc.function(sc.G(sc.L("x", 3), sc.L("p", ())), output=sc.L("f", ...))
def cost(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
  x, p = inputs
  return (x * x).sum() * p


@sc.function(
  sc.G(sc.G(sc.L("state", 4), sc.L("u", 2)), sc.G(sc.L("pw", 10), sc.L("physics", 3), sc.L("dt", ()))),
  output=sc.L("next", ...),
)
def step(inputs: tuple[tuple[sc.Expr, sc.Expr], tuple[sc.Expr, sc.Expr, sc.Expr]]) -> sc.Expr:
  (state, _u), (_pw, _physics, _dt) = inputs
  return state


@sc.function(sc.G(sc.L("state", 4), sc.L("u", 2), sc.L("pw", 10), sc.L("physics", 3), sc.L("dt", ())), output=sc.L("next", ...))
def step_flat(inputs: tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr]) -> sc.Expr:
  state, _u, _pw, _physics, _dt = inputs
  return state


def test_leaf_shapes_and_group_widths() -> None:
  assert sc.L("x", 3).shapes == ((3,),)
  assert sc.L("P", (2, 2)).shapes == ((2, 2),)
  assert sc.L("s", ()).shapes == ((),)
  with pytest.raises(TypeError, match="inferred"):
    _ = sc.L("f", ...).shapes

  assert sc.G(sc.L("a", 1), sc.L("b", 1)).names == ("a", "b")
  eight = cast(Any, sc.G)(*(sc.L(f"x{i}", 1) for i in range(8)))
  assert eight.size == 8
  with pytest.raises(TypeError, match="2 to 8"):
    cast(Any, sc.G)(sc.L("a", 1))
  with pytest.raises(TypeError, match="nest for more"):
    cast(Any, sc.G)(*(sc.L(f"x{i}", 1) for i in range(9)))

  nested = sc.G(sc.G(sc.L("a", 1), sc.L("b", 1)), sc.L("c", 1))
  assert nested.names == ("a", "b", "c")
  assert nested.size == 3


def test_tree_names_are_unique_and_symbols_follow_structure() -> None:
  with pytest.raises(ValueError, match="duplicate"):
    sc.G(sc.L("x", 3), sc.L("x", 3))
  with pytest.raises(ValueError, match="duplicate"):
    sc.G(sc.G(sc.L("x", 3), sc.L("y", 3)), sc.L("x", 1))

  symbols = sc.G(sc.G(sc.L("a", 1), sc.L("b", 2)), sc.L("c", ())).symbols()
  assert isinstance(symbols, tuple)
  assert isinstance(symbols[0], tuple)
  assert symbols[0][1].name == "b"
  assert symbols[0][1].shape == (2,)
  assert symbols[1].shape == ()

  tree = sc.G(sc.L("u", 2), sc.L("s", ...))
  assert tree.relabel("lam:").names == ("lam:u", "lam:s")
  resolved = tree.with_types((sc.TensorType((2,)), sc.TensorType((1,))))
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
    multiply.symbolic_call((sc.sym("x", 3), sc.sym("y", 2)))
  with pytest.raises(ValueError, match="declared structure"):
    step.numerical_call(cast(Any, (np.zeros(4), np.zeros(2), np.zeros(10), np.zeros(3), np.array(0.1))))
  with pytest.raises(ValueError, match="declared structure"):
    step_flat.numerical_call(cast(Any, ((np.zeros(4), np.zeros(2)), (np.zeros(10), np.zeros(3), np.array(0.1)))))

  with pytest.raises(TypeError, match="expected shape"):
    sc.function(sc.L("x", 3), output=sc.L("y", 2))(lambda x: x)
  with pytest.raises((TypeError, ValueError), match="declared 2 outputs|declared structure"):
    sc.function(sc.L("x", 3), output=sc.G(sc.L("a", ...), sc.L("b", ...)))(cast(Any, lambda x: x))


def test_derivatives_preserve_source_trees() -> None:
  grad = sc.gradient(cost, "f", "x")
  hess = sc.hessian(cost, "f", "x")
  jac = sc.jacobian(square, "square", "x")
  fwd = sc.forward(cost, "f", "x")
  adj = sc.adjoint(square, "square", "x")
  hess_l = sc.lagrangian_hessian(duplicate, "x")

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
    fwd.numerical_call((np.arange(3.0), np.array(2.0)), np.ones(3)),
    12.0,
  )

  with pytest.raises(ValueError, match=r"declared \('x', 'p'\)"):
    sc.gradient(cost, "f", "z")
  with pytest.raises(ValueError, match=r"declared \('f',\)"):
    sc.gradient(cost, "h", "x")
  with pytest.raises(ValueError, match="scalar"):
    sc.gradient(duplicate, "first", "x")
  with pytest.raises(ValueError, match="scalar"):
    sc.hessian(duplicate, "first", "x")
  with pytest.raises(ValueError, match="unknown name 'z'"):
    sc.forward(cost, "f", "z")
  with pytest.raises(ValueError, match="unknown name 'y'"):
    sc.lagrangian_hessian(duplicate, "y")


def test_seeded_derivatives_accept_nondifferentiable_leaves() -> None:
  @sc.function(sc.L("x", sc.TensorType((2,), diff=False)), output=sc.L("y", ...))
  def frozen_square(x: sc.Expr) -> sc.Expr:
    return x * x

  fwd = sc.forward(frozen_square, "y", "x")
  adj = sc.adjoint(frozen_square, "y", "x")
  x = np.array([2.0, 3.0])

  assert fwd.input_tree.types[-1].diff
  assert adj.input_tree.types[-1].diff
  np.testing.assert_array_equal(fwd(x, np.ones(2)), 2.0 * x)
  np.testing.assert_array_equal(adj(x, np.ones(2)), 2.0 * x)


def test_lagrangian_hessians_accept_constant_output_leaves() -> None:
  @sc.function(sc.L("x", 2), output=sc.G(sc.L("cost", ...), sc.L("constant", ...)))
  def objective(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    return (x * x).sum(), sc.const(1.0)

  dense = sc.lagrangian_hessian(objective, "x")
  sparse = sc.sparse_lagrangian_hessian(objective, "x")
  inputs = (np.array([2.0, 3.0]), (np.array(1.5), np.array(7.0)))

  assert dense.input_tree.types[-1].diff
  assert sparse.input_tree.types[-1].diff
  np.testing.assert_array_equal(dense(*inputs), 3.0 * np.eye(2))
  np.testing.assert_array_equal(sparse(*inputs), np.array([3.0, 3.0]))


def test_call_dispatches_on_leaf_kind() -> None:
  """``__call__`` routes to ``symbolic_call`` or ``numerical_call`` by the leaves it is given."""
  xv, yv = np.arange(3.0), np.ones(3)

  np.testing.assert_allclose(multiply((xv, yv)), multiply.numerical_call((xv, yv)))
  assert multiply((sc.sym("a", 3), sc.sym("b", 3))).op == sc.ExprOp.CALL

  # Only G introduces a tuple: a one-leaf tree is the bare value on both sides.
  assert not isinstance(multiply((xv, yv)), tuple)
  first, second = duplicate(xv)
  np.testing.assert_allclose(first, xv)
  np.testing.assert_allclose(second, xv)

  with pytest.raises(TypeError, match="mix Expr and numerical leaves"):
    multiply(cast(Any, (sc.sym("a", 3), yv)))

  # A constant leaf has to be lifted for the symbolic reading; the dispatch never guesses.
  assert multiply((sc.const(xv), sc.sym("b", 3))).op == sc.ExprOp.CALL


def test_call_dispatch_ignores_structure() -> None:
  """A wrongly-shaped argument is reported against the declared names, not as a kind mismatch."""
  assert is_symbolic_call(((sc.sym("a", 3), sc.sym("b", 3)),), "f")
  assert not is_symbolic_call(((np.zeros(3), np.zeros(3)),), "f")
  assert not is_symbolic_call((), "f")
  with pytest.raises(TypeError, match="mix Expr and numerical leaves"):
    is_symbolic_call((sc.sym("a", 3), np.zeros(3)), "f")
  # The predicate does not consult the structure, so the flattener owns that message.
  assert is_symbolic_call(((sc.sym("a", 3),),), "f")
  with pytest.raises(ValueError, match="does not have the declared structure"):
    multiply(cast(Any, (np.zeros(3),)))


def test_leaf_declarations_take_a_name_a_shape_or_both() -> None:
  f64, i64 = sc.TensorType((3,)), sc.TensorType((2, 2), sc.dtypes.int64, diff=False)
  assert (sc.L().names, sc.L().decls) == (("",), (Hole(),))
  assert (sc.L("y").names, sc.L("y").decls) == (("y",), (Hole(),))
  assert sc.L((3, None)).decls == (Hole((3, None)),) and sc.L(..., dtype="int64").decls == (Hole(None, sc.dtypes.int64),)
  assert str(Hole((3, None))) == "(3, None)" and str(Hole()) == "any shape" and str(Hole((None,), sc.dtypes.int64)) == "(None,) int64"
  assert (sc.L(3).names, sc.L(3).decls) == (("",), (f64,))
  assert sc.L(np.int64(3)).decls == (f64,)
  assert sc.L((2, 2), dtype="int64", diff=False).decls == (i64,)
  assert sc.L("k", (), dtype=sc.dtypes.int64).decls == (sc.TensorType((), sc.dtypes.int64),)
  assert sc.L(3).named("x").names == ("x",) and sc.L("y", 3).named("x").names == ("y",)
  with pytest.raises(TypeError, match="use \\(\\) for a scalar"):
    sc.L(cast(Any, None))
  with pytest.raises(TypeError, match="a leaf shape is"):
    sc.L(cast(Any, 1.5))
  with pytest.raises(TypeError, match="a name and a shape, or a shape alone"):
    sc.L(3, cast(Any, 4))  # ty: ignore[invalid-argument-type]
  with pytest.raises(TypeError, match="carries its own dtype"):
    sc.L(f64, dtype="int64")
  with pytest.raises(ValueError, match="non-empty name"):
    sc.L("", 3)


def test_groups_and_parameter_lists_take_tree_specs() -> None:
  group = sc.G(3, "y", sc.L("z", ()))
  assert group.names == ("", "y", "z")
  assert group.named("p").names == ("p_0", "y", "z")
  assert group.decls[1] == Hole() and group.has_holes and not sc.G(3, 4).has_holes
  params = param_list(sc.L(2).named("x"), sc.G(1, 1).named("p"))
  assert params.names == ("x", "p_0", "p_1")
  assert params.symbols()[1][0].name == "p_0"
  with pytest.raises(ValueError, match="unnamed leaf"):
    sc.G(3, 4).symbols()
