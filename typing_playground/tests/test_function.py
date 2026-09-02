from __future__ import annotations

from typing import Any, cast

import pytest

from typing_playground.expr import Buffer, Expr
from typing_playground.function import forward, function, gradient, hessian, lagrangian_hessian
from typing_playground.tests.definitions import (
  adj_square_x,
  cost,
  cost_batch,
  duplicate,
  filter_problem,
  fwd_f_x,
  grad_f_x,
  hess_f_x,
  hess_l,
  jac_square_x,
  multiply,
  square,
  step,
  step_flat,
)
from typing_playground.trees import G, L


def test_names_and_shapes_are_declared_not_inferred() -> None:
  assert duplicate.input_names == ("x",) and duplicate.output_names == ("first", "second")
  assert duplicate.output_shapes == ((3,), (3,))  # `first` was `...`, traced to (3,)
  assert multiply.output_shapes == ((3,),)


def test_grouping_is_not_a_signature() -> None:
  assert step.input_names == step_flat.input_names == ("state", "u", "pw", "physics", "dt")
  assert step.c_signature() == step_flat.c_signature().replace("step_flat", "step")


def test_calls_return_the_declared_structure() -> None:
  out = duplicate.numerical_call(Buffer((3,)))
  assert isinstance(out, tuple) and len(out) == 2 and out[0].shape == (3,)
  assert step.numerical_call(((Buffer((4,)), Buffer((2,))), (Buffer((10,)), Buffer((3,)), Buffer(())))).shape == (4,)


def test_composition_through_symbolic_call() -> None:
  assert square.output_shapes == ((3,),)
  assert jac_square_x.output_shapes == ((3, 3),)


def test_wrong_shape_at_call_is_a_runtime_error() -> None:
  with pytest.raises(ValueError, match="expected shapes"):
    duplicate.numerical_call(Buffer((4,)))
  with pytest.raises(ValueError, match="expected shapes"):
    multiply.symbolic_call((Expr((3,)), Expr((2,))))


def test_wrong_structure_at_call_is_a_runtime_error() -> None:
  with pytest.raises(ValueError, match="structure"):
    step.numerical_call(cast(Any, (Buffer((4,)), Buffer((2,)), Buffer((10,)), Buffer((3,)), Buffer(()))))
  with pytest.raises(ValueError, match="structure"):
    step_flat.numerical_call(cast(Any, ((Buffer((4,)), Buffer((2,))), (Buffer((10,)), Buffer((3,)), Buffer(())))))


def test_output_count_and_shape_are_checked_at_the_decorator() -> None:
  with pytest.raises(TypeError, match=r"declared with shape \(2,\)"):
    function(L("x", 3), L("y", 2))(lambda x: x)
  with pytest.raises(TypeError, match="declared 2 leaves"):
    function(L("x", 3), G(L("a", ...), L("b", ...)))(cast(Any, lambda x: x))


def test_body_names_are_independent_of_declared_names() -> None:
  assert filter_problem.vars.names == ("u", "s") and filter_problem.params.names == ("x", "u_ref")


def test_derivatives_keep_the_source_inputs() -> None:
  assert grad_f_x.input_names == ("x", "p") and grad_f_x.output_names == ("grad_f_x",) and grad_f_x.output_shapes == ((3,),)
  assert hess_f_x.output_names == ("hess_f_x_x",) and hess_f_x.output_shapes == ((3, 3),)
  assert grad_f_x.c_signature() == "void cost_grad_f_x(const double* x, const double* p, double* grad_f_x)"


def test_seeded_modes_pair_inputs_with_the_new_group() -> None:
  assert fwd_f_x.input_names == ("x", "p", "fwd:x") and fwd_f_x.output_shapes == ((),)
  assert adj_square_x.input_names == ("x", "lam:square") and adj_square_x.output_shapes == ((3,),)
  assert hess_l.input_names == ("x", "lam:first", "lam:second") and hess_l.output_shapes == ((3, 3),)
  assert fwd_f_x.numerical_call(((Buffer((3,)), Buffer(())), Buffer((3,)))).shape == ()


def test_unknown_of_or_wrt_fails_at_build_time_naming_the_choices() -> None:
  with pytest.raises(ValueError, match=r"declared \('x', 'p'\)"):
    gradient(cost, "f", "z")
  with pytest.raises(ValueError, match=r"declared \('f',\)"):
    gradient(cost, "h", "x")
  with pytest.raises(ValueError, match="unknown name 'z'"):
    forward(cost, "f", "z")
  with pytest.raises(ValueError, match="unknown name 'y'"):
    lagrangian_hessian(duplicate, "y")


def test_gradient_and_hessian_need_a_scalar() -> None:
  with pytest.raises(ValueError, match="scalar"):
    gradient(duplicate, "first", "x")
  with pytest.raises(ValueError, match="scalar"):
    hessian(duplicate, "first", "x")


def test_vmap_batches_every_leaf_and_keeps_the_trees() -> None:
  assert cost_batch.input_names == ("x", "p") and cost_batch.input_shapes == ((7, 3), (7,))
  assert cost_batch.output_names == ("f",) and cost_batch.output_shapes == ((7,),)
  assert cost_batch.numerical_call((Buffer((7, 3)), Buffer((7,)))).shape == (7,)
