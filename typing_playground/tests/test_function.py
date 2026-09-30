from __future__ import annotations

from typing import Any, cast

import pytest

from typing_playground.expr import Buffer, Expr
from typing_playground.function import forward, function, gradient, hessian, lagrangian_hessian
from typing_playground.tests.definitions import (
  adj_square_x,
  constant,
  energy,
  cost,
  cost_batch,
  cost_packed,
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
from typing_playground.trees import arg, group


def test_names_and_shapes_are_declared_not_inferred() -> None:
  d = duplicate.instantiate()
  assert d.input_names == ("x",) and d.output_names == ("first", "second")
  assert d.output_shapes == ((3,), (3,))  # `first` was `...`, traced to (3,)
  assert multiply.instantiate().output_shapes == ((3,),)


def test_a_fully_shaped_declaration_is_one_instance_built_now_under_its_own_name() -> None:
  assert list(cost.instances.values()) == [cost.instantiate()]
  assert cost.instantiate().name == "cost"
  assert grad_f_x.instances and grad_f_x.instantiate().name == "cost_grad_f_x"  # and so are its derivatives


def test_grouping_is_not_a_signature() -> None:
  s, f = step.instantiate(), step_flat.instantiate()
  assert s.input_names == f.input_names == ("state", "u", "pw", "physics", "dt")
  assert s.c_signature() == f.c_signature().replace("step_flat", "step")


def test_calls_return_the_declared_structure() -> None:
  out = duplicate.numerical_call(Buffer((3,)))
  assert isinstance(out, tuple) and len(out) == 2 and out[0].shape == (3,)
  assert step.numerical_call((Buffer((4,)), Buffer((2,))), (Buffer((10,)), Buffer((3,)), Buffer(()))).shape == (4,)


def test_composition_through_symbolic_call() -> None:
  assert square.instantiate().output_shapes == ((3,),)
  assert jac_square_x.instantiate().output_shapes == ((3, 3),)


def test_wrong_shape_at_call_is_a_runtime_error() -> None:
  with pytest.raises(ValueError, match="expected shapes"):
    duplicate.numerical_call(Buffer((4,)))
  with pytest.raises(ValueError, match="expected shapes"):
    multiply.symbolic_call(Expr((3,)), Expr((2,)))


def test_wrong_structure_at_call_is_a_runtime_error() -> None:
  with pytest.raises(ValueError, match="structure"):
    cast(Any, step).numerical_call(Buffer((4,)), Buffer((2,)), Buffer((10,)), Buffer((3,)), Buffer(()))
  with pytest.raises(ValueError, match="structure"):
    cast(Any, step_flat).numerical_call((Buffer((4,)), Buffer((2,))), (Buffer((10,)), Buffer((3,)), Buffer(())))


def test_output_count_and_shape_are_checked_at_the_decorator() -> None:
  with pytest.raises(TypeError, match=r"declared with shape \(2,\)"):
    function(arg("x", 3), outputs=arg("y", 2))(lambda x: x)
  with pytest.raises(TypeError, match="declared 2 leaves"):
    function(arg("x", 3), outputs=group(arg("a", ...), arg("b", ...)))(cast(Any, lambda x: x))


def test_group_parts_are_the_body_parameters() -> None:
  f = function(arg("x", 3), arg("p", ()), outputs=arg("f", ...))(lambda x, p: (x * x).sum() * p)
  assert f.instantiate().input_names == ("x", "p") and f.instantiate().output_shapes == ((),)
  assert f.numerical_call(Buffer((3,)), Buffer(())).shape == ()
  with pytest.raises(TypeError):
    function(arg("x", 3), arg("p", ()), outputs=arg("f", ...))(cast(Any, lambda inputs: inputs[0]))  # the whole tree is not one parameter


def test_a_group_is_one_parameter() -> None:
  assert cost_packed.instantiate().c_signature() == cost.instantiate().c_signature().replace("cost", "cost_packed", 1)
  assert cost_packed.numerical_call((Buffer((3,)), Buffer(()))).shape == ()
  with pytest.raises(ValueError, match="structure"):
    cast(Any, cost_packed).numerical_call(Buffer((3,)), Buffer(()))
  one = function(group(arg("x", 3)), group(arg("y", 3)), outputs=arg("z", ...))(lambda x, y: x[0] + y[0])  # never normalized
  assert one.numerical_call((Buffer((3,)),), (Buffer((3,)),)).shape == (3,)


def test_zero_parameters() -> None:
  assert constant.instantiate().name == "constant" and constant.instantiate().input_names == ()
  assert constant().shape == (2,) and isinstance(constant(), Buffer)  # no arguments is an evaluation
  assert isinstance(constant.symbolic_call(), Expr)


def test_a_missing_output_declaration_is_read_off_the_trace() -> None:
  assert energy.outputs is None and energy.instantiate().output_names == ("energy",) and energy.instantiate().output_shapes == ((),)
  assert gradient(energy, "energy", "x").name == "energy_grad_energy_x"
  assert function(arg("x", 3), name="twice")(lambda x: 2.0 * x).instantiate().output_names == ("twice",)
  nested = function(arg("x", 3))(lambda x: (x, (x, x.sum())))
  assert nested.instantiate().output_names == ("out0", "out1", "out2")
  out = nested(Buffer((3,)))
  assert out[0].shape == (3,) and out[1][0].shape == (3,) and out[1][1].shape == ()


def test_body_names_are_independent_of_declared_names() -> None:
  assert filter_problem.vars.names == ("u", "s") and filter_problem.params.names == ("x", "u_ref")


def test_derivatives_keep_the_source_inputs() -> None:
  g, h = grad_f_x.instantiate(), hess_f_x.instantiate()
  assert g.input_names == ("x", "p") and g.output_names == ("grad_f_x",) and g.output_shapes == ((3,),)
  assert h.output_names == ("hess_f_x_x",) and h.output_shapes == ((3, 3),)
  assert g.c_signature() == "void cost_grad_f_x(const double* x, const double* p, double* grad_f_x)"


def test_seeded_modes_append_one_parameter() -> None:
  fwd, adj, lh = fwd_f_x.instantiate(), adj_square_x.instantiate(), hess_l.instantiate()
  assert fwd.input_names == ("x", "p", "fwd:x") and fwd.output_shapes == ((),)
  assert adj.input_names == ("x", "lam:square") and adj.output_shapes == ((3,),)
  assert lh.input_names == ("x", "lam:first", "lam:second") and lh.output_shapes == ((3, 3),)
  assert fwd_f_x.numerical_call(Buffer((3,)), Buffer(()), Buffer((3,))).shape == ()
  assert adj_square_x.numerical_call(Buffer((3,)), Buffer((3,))).shape == (3,)
  assert hess_l.numerical_call(Buffer((3,)), (Buffer((3,)), Buffer((3,)))).shape == (3, 3)


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
  b = cost_batch.instantiate()
  assert b.input_names == ("x", "p") and b.input_shapes == ((7, 3), (7,)) and b.name == "cost_vmap7"
  assert b.output_names == ("f",) and b.output_shapes == ((7,),)
  assert cost_batch.numerical_call(Buffer((7, 3)), Buffer((7,))).shape == (7,)


def test_call_dispatches_on_the_leaf_kind() -> None:
  assert isinstance(cost(Buffer((3,)), Buffer(())), Buffer)
  assert isinstance(cost(Expr((3,)), Expr(())), Expr)
  assert isinstance(cost.instantiate()(Expr((3,)), Expr(())), Expr)
  assert isinstance(step((Buffer((4,)), Buffer((2,))), (Buffer((10,)), Buffer((3,)), Buffer(()))), Buffer)
  with pytest.raises(TypeError, match="mix Expr and numerical"):
    cast(Any, cost)(Expr((3,)), Buffer(()))
  with pytest.raises(ValueError, match="expected shapes"):
    cost(Buffer((4,)), Buffer(()))
  inner = function(arg("x", 3), outputs=arg("y", ...))(lambda x: cost(x, x.sum()))  # symbolic inside a trace
  assert inner.instantiate().output_shapes == ((),)
