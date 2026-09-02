from __future__ import annotations

from typing import Any, cast

import pytest

from typing_playground.expr import Buffer
from typing_playground.function import vmap
from typing_playground.templates import forward, gradient, hessian, lagrangian_hessian, template
from typing_playground.tests.definitions import caller, cost_t, fresh_cost, scale
from typing_playground.trees import G, L


def test_helpers_cost_nothing_until_called() -> None:
  assert fresh_cost().instances == {}


def test_calls_instantiate_and_cache() -> None:
  t = fresh_cost()
  out = t.numerical_call((Buffer((30, 40)), Buffer((5,))))
  assert out.shape == (5,) and len(t.instances) == 1
  first = t.instances[((30, 40), (5,))]
  t.numerical_call((Buffer((30, 40)), Buffer((5,))))
  assert t.instances[((30, 40), (5,))] is first
  t.numerical_call((Buffer((3,)), Buffer(())))
  assert len(t.instances) == 2


def test_instance_names_carry_the_signature() -> None:
  t = fresh_cost()
  assert t.instantiate(((30, 40), (5,))).name == "cost__30x40_5"
  assert t.instantiate((3, ())).name == "cost__3_s"
  assert t.instantiate((3, ())).c_signature() == "void cost__3_s(const double* x, const double* p, double* f)"


def test_trace_time_resolution_inside_another_body() -> None:
  assert ((2,), (2,)) in cost_t.instances and cost_t.instances[((2,), (2,))].name == "cost__2_2"
  assert caller.output_shapes == ((2,),)


def test_declared_shapes_still_bind() -> None:
  t = template(G(L("x", 3), L("p")), L("f"), name="mixed")(lambda i: (i[0] * i[0]).sum() * i[1])
  assert t.numerical_call((Buffer((3,)), Buffer((7,)))).shape == (7,)
  with pytest.raises(TypeError, match=r"'x' declared with shape \(3,\)"):
    t.numerical_call((Buffer((4,)), Buffer((7,))))


def test_structure_is_checked_before_shapes() -> None:
  with pytest.raises(ValueError, match="structure"):
    fresh_cost().numerical_call(cast(Any, Buffer((3,))))
  with pytest.raises(TypeError, match="declared 2 leaves"):
    fresh_cost().instantiate(((3,),))


def test_fully_shaped_template_is_eager() -> None:
  t = template(G(L("x", 3), L("p", ())), L("f"), name="full")(lambda i: (i[0] * i[0]).sum() * i[1])
  assert len(t.instances) == 1  # traced at the decorator, like @function
  with pytest.raises(TypeError, match=r"'f' declared with shape \(5,\)"):
    template(L("x", 3), L("f", 5))(lambda x: (x * x).sum())


def test_gradient_of_a_template_is_a_template() -> None:
  t = fresh_cost()
  g = gradient(t, "f", "x")
  assert g.instances == {}  # deriving alone builds nothing
  assert g.numerical_call((Buffer((3,)), Buffer(()))).shape == (3,)
  assert ((3,), ()) in t.instances  # the source was instantiated for those shapes
  assert g.instances[((3,), ())].name == "cost__3_s_grad_f_x"
  with pytest.raises(ValueError, match=r"declared \('x', 'p'\)"):
    gradient(t, "f", "z")
  with pytest.raises(ValueError, match=r"declared \('f',\)"):
    gradient(t, "g", "x")


def test_seeded_transforms_extend_the_tree_with_holes() -> None:
  t = fresh_cost()
  fwd = forward(t, "f", "x")
  assert fwd.inputs is not None and fwd.inputs.names == ("x", "p", "fwd:x") and fwd.inputs.has_holes
  assert fwd.numerical_call(((Buffer((3,)), Buffer(())), Buffer((3,)))).shape == ()
  with pytest.raises(ValueError, match="transform's inputs"):
    fwd.numerical_call(((Buffer((3,)), Buffer(())), Buffer((4,))))  # seed must match wrt
  lh = lagrangian_hessian(t, "x")
  assert lh.inputs is not None and lh.inputs.names == ("x", "p", "lam:f")
  assert lh.numerical_call(((Buffer((3,)), Buffer(())), Buffer(()))).shape == (3, 3)
  assert hessian(t, "f", "x").numerical_call((Buffer((3,)), Buffer(()))).shape == (3, 3)


def test_scalar_checks_move_to_instantiation() -> None:
  t = template(L("x"), G(L("a"), L("b")), name="dup")(lambda x: (x, x))
  g = gradient(t, "a", "x")  # names are fine; `a` is only known to be non-scalar once shaped
  with pytest.raises(ValueError, match="scalar"):
    g.numerical_call(Buffer((3,)))


def test_vmap_over_a_template_goes_through_an_instance() -> None:
  batched = vmap(fresh_cost().instantiate((3, ())), 4)
  assert batched.input_shapes == ((4, 3), (4,)) and batched.name == "cost__3_s_vmap4"


def test_bare_template_infers_everything_from_the_call() -> None:
  out = scale.numerical_call((Buffer((3,)), Buffer(())))
  assert out.shape == (3,)
  inst = scale.instances[((3,), ())]
  assert inst.input_names == ("in0", "in1") and inst.output_names == ("out0",)
  assert inst.name == "scale__3_s"
  scale.numerical_call((Buffer((3,)), Buffer(())))
  assert len(scale.instances) == 1


def test_bare_template_refuses_array_likes() -> None:
  with pytest.raises(TypeError, match="every leaf must be an Expr or an ndarray"):
    scale.numerical_call(([1.0, 2.0], Buffer(())))


def test_bare_template_has_no_names_to_differentiate_or_bind() -> None:
  with pytest.raises(TypeError, match="bare"):
    gradient(scale, "out0", "in0")
  with pytest.raises(TypeError, match="bare"):
    scale.instantiate(((3,), ()))
