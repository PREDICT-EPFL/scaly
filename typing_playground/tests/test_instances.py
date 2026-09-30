from __future__ import annotations

from typing import Any, cast

import pytest

from typing_playground.expr import Buffer, Expr
from typing_playground.function import adjoint, forward, function, gradient, hessian, jacobian, lagrangian_hessian, vmap
from typing_playground.tests.definitions import caller, cost_t, fresh_cost, scale
from typing_playground.trees import arg, group, skeleton


def test_helpers_cost_nothing_until_called() -> None:
  assert fresh_cost().instances == {}


def test_calls_instantiate_and_cache() -> None:
  t = fresh_cost()
  out = t.numerical_call(Buffer((30, 40)), Buffer((5,)))
  assert out.shape == (5,) and len(t.instances) == 1
  first = t.instances[((30, 40), (5,))]
  t.numerical_call(Buffer((30, 40)), Buffer((5,)))
  assert t.instances[((30, 40), (5,))] is first
  t.numerical_call(Buffer((3,)), Buffer(()))
  assert len(t.instances) == 2


def test_instance_names_carry_the_signature() -> None:
  t = fresh_cost()
  assert t.instantiate(((30, 40), (5,))).name == "cost__30x40_5"
  assert t.instantiate((3, ())).name == "cost__3_s"
  assert t.instantiate((3, ())).c_signature() == "void cost__3_s(const double* x, const double* p, double* f)"


def test_trace_time_resolution_inside_another_body() -> None:
  assert ((2,), (2,)) in cost_t.instances and cost_t.instances[((2,), (2,))].name == "cost__2_2"
  assert caller.instantiate().output_shapes == ((2,),)


def test_declared_shapes_still_bind() -> None:
  t = function(arg("x", 3), arg("p"), outputs=arg("f"), name="mixed")(lambda x, p: (x * x).sum() * p)
  assert t.numerical_call(Buffer((3,)), Buffer((7,))).shape == (7,)
  with pytest.raises(ValueError, match="expected shapes"):
    t.numerical_call(Buffer((4,)), Buffer((7,)))
  with pytest.raises(TypeError, match=r"'x' declared with shape \(3,\)"):
    t.instantiate((4, 7))


def test_structure_is_checked_before_shapes() -> None:
  with pytest.raises(ValueError, match="structure"):
    cast(Any, fresh_cost()).numerical_call(Buffer((3,)))
  with pytest.raises(TypeError, match="declared 2 leaves"):
    fresh_cost().instantiate(((3,),))


def test_fully_shaped_is_eager_and_holes_need_shapes() -> None:
  t = function(arg("x", 3), arg("p", ()), outputs=arg("f"), name="full")(lambda x, p: (x * x).sum() * p)
  assert len(t.instances) == 1 and t.instantiate().name == "full"  # traced at the decorator, not mangled
  with pytest.raises(TypeError, match="pass the shapes"):
    fresh_cost().instantiate()
  with pytest.raises(TypeError, match=r"'f' declared with shape \(5,\)"):
    function(arg("x", 3), outputs=arg("f", 5))(lambda x: (x * x).sum())


def test_gradient_of_holes_binds_per_call() -> None:
  t = fresh_cost()
  g = gradient(t, "f", "x")
  assert g.instances == {}  # deriving alone builds nothing
  assert g.numerical_call(Buffer((3,)), Buffer(())).shape == (3,)
  assert ((3,), ()) in t.instances  # the source was instantiated for those shapes
  assert g.instances[((3,), ())].name == "cost__3_s_grad_f_x"
  with pytest.raises(ValueError, match=r"declared \('x', 'p'\)"):
    gradient(t, "f", "z")
  with pytest.raises(ValueError, match=r"declared \('f',\)"):
    gradient(t, "g", "x")


def test_seeded_transforms_append_a_parameter_with_holes() -> None:
  t = fresh_cost()
  fwd = forward(t, "f", "x")
  assert fwd.inputs is not None and fwd.inputs.names == ("x", "p", "fwd:x") and fwd.inputs.has_holes
  assert fwd.numerical_call(Buffer((3,)), Buffer(()), Buffer((3,))).shape == ()
  with pytest.raises(ValueError, match="transform's inputs"):
    fwd.numerical_call(Buffer((3,)), Buffer(()), Buffer((4,)))  # seed must match wrt
  lh = lagrangian_hessian(t, "x")
  assert lh.inputs is not None and lh.inputs.names == ("x", "p", "lam:f")
  assert lh.numerical_call(Buffer((3,)), Buffer(()), Buffer(())).shape == (3, 3)
  assert hessian(t, "f", "x").numerical_call(Buffer((3,)), Buffer(())).shape == (3, 3)


def test_scalar_checks_move_to_instantiation() -> None:
  t = function(arg("x"), outputs=group(arg("a"), arg("b")), name="dup")(lambda x: (x, x))
  g = gradient(t, "a", "x")  # names are fine; `a` is only known to be non-scalar once shaped
  with pytest.raises(ValueError, match="scalar"):
    g.numerical_call(Buffer((3,)))


def test_vmap_of_holes_strips_the_axis_to_bind_the_source() -> None:
  t = fresh_cost()
  batched = vmap(t, 4)
  assert batched.instances == {}
  assert batched.numerical_call(Buffer((4, 3)), Buffer((4,))).shape == (4,)
  assert ((3,), ()) in t.instances and batched.instances[((4, 3), (4,))].name == "cost__3_s_vmap4"
  with pytest.raises(ValueError, match="leading axis of 4"):
    batched.numerical_call(Buffer((5, 3)), Buffer((4,)))


def test_bare_function_infers_everything_from_the_call() -> None:
  out = scale.numerical_call(Buffer((3,)), Buffer(()))
  assert out.shape == (3,)
  inst = scale.instances[skeleton((Buffer((3,)), Buffer(())))]
  assert inst.input_names == ("in0", "in1") and inst.output_names == ("scale",)
  assert inst.name == "scale__3_s"
  scale.numerical_call(Buffer((3,)), Buffer(()))
  assert len(scale.instances) == 1


def test_bare_function_refuses_array_likes() -> None:
  with pytest.raises(TypeError, match="every leaf must be an Expr or an ndarray"):
    scale.numerical_call([1.0, 2.0], Buffer(()))


def test_bare_function_traces_once_per_instance() -> None:
  traces: list[Any] = []

  @function()
  def counted(x: Expr) -> Expr:
    traces.append(x)
    return x

  counted(Buffer((3,)))
  counted(Buffer((3,)))
  assert len(traces) == 1


def test_bare_function_is_transformed_by_its_inferred_names() -> None:
  assert gradient(scale, "scale", "in0").numerical_call(Buffer(()), Buffer(())).shape == ()
  assert jacobian(scale, "scale", "in0").numerical_call(Buffer((3,)), Buffer(())).shape == (3, 3)
  assert adjoint(scale, "scale", "in0").numerical_call(Buffer((3,)), Buffer(()), Buffer((3,))).shape == (3,)
  assert vmap(scale, 5).numerical_call(Buffer((5, 3)), Buffer((5,))).shape == (5, 3)
  unknown = gradient(scale, "out0", "in0")  # names exist only per instance, so they are checked at binding
  with pytest.raises(ValueError, match="unknown name 'out0'"):
    unknown.numerical_call(Buffer(()), Buffer(()))
  with pytest.raises(TypeError, match="bare"):
    scale.instantiate(((3,), ()))


def test_inferred_outputs_with_holes_are_transformed_at_binding() -> None:
  split = function(arg("x"))(lambda x: (x, x.sum()))
  assert split.outputs is None and split.instances == {}
  out = split(Buffer((4,)))
  assert out[0].shape == (4,) and out[1].shape == ()
  assert gradient(split, "out1", "x").numerical_call(Buffer((4,))).shape == (4,)
  lh = lagrangian_hessian(split, "x")  # the multipliers take the output tree, known only at binding
  assert lh.inputs is None and lh.numerical_call(Buffer((4,)), (Buffer((4,)), Buffer(()))).shape == (4, 4)
  with pytest.raises(ValueError, match="transform's inputs"):
    lh.numerical_call(Buffer((4,)), (Buffer((4,)),))


def test_jacobian_and_adjoint_of_holes() -> None:
  t = function(arg("x"), outputs=arg("y"), name="double")(lambda x: 2.0 * x)
  assert jacobian(t, "y", "x").numerical_call(Buffer((3,))).shape == (3, 3)
  adj = adjoint(t, "y", "x")
  assert adj.inputs is not None and adj.inputs.names == ("x", "lam:y") and adj.inputs.has_holes
  assert adj.numerical_call(Buffer((3,)), Buffer((3,))).shape == (3,)
  with pytest.raises(ValueError, match="transform's inputs"):
    adj.numerical_call(Buffer((3,)), Buffer((4,)))  # the seed must match the output


def test_call_binds_then_dispatches() -> None:
  t = fresh_cost()
  assert isinstance(t(Buffer((3,)), Buffer(())), Buffer) and isinstance(t(Expr((3,)), Expr(())), Expr)
  assert list(t.instances) == [((3,), ())]  # one instance for both kinds of call
  assert isinstance(scale(Expr((3,)), Expr(())), Expr)
