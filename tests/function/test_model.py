from __future__ import annotations

import numpy as np
import pytest

from scaly.function.model import as_concrete
from scaly.function.concrete import ConcreteFunction
import scaly as sc
from scaly.ir.expr import topo


def test_function_call_node_eval() -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y"))
  def inner(x):
    return x.sin()

  @sc.function(sc.arg("z", 2), outputs=sc.arg("out"))
  def outer(z):
    return inner(z) + 1.0

  np.testing.assert_allclose(outer(np.array([0.1, 0.2])), np.sin([0.1, 0.2]) + 1.0)
  assert any(e.op == sc.ExprOp.CALL for e in topo(as_concrete(outer).outputs))


def test_function_call_normalizes_raw_constant_args() -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y"))
  def inner(x):
    return x + 1.0

  inner_const = inner(sc.const([1.0, 2.0]))

  @sc.function(outputs=sc.arg("out", 2))
  def outer() -> sc.Expr:
    return inner_const

  np.testing.assert_allclose(outer(), np.array([2.0, 3.0]))
  assert not inner_const.type.diff


def test_function_signature_and_call_shape_errors() -> None:
  x = sc.sym("x", 2)

  try:
    _ = ConcreteFunction._from_exprs("bad", [x], [x], [], ["y"])
  except ValueError as e:
    assert "expected 1 names, got 0" in str(e)
  else:  # pragma: no cover
    raise AssertionError("input name arity mismatch should fail")

  y = sc.sym("y", 2)
  try:
    _ = ConcreteFunction._from_exprs("missing", [x], [x + y], ["x"], ["z"])
  except ValueError as e:
    assert "undeclared symbolic inputs: ['y']" in str(e)
  else:  # pragma: no cover
    raise AssertionError("undeclared graph input should fail")

  @sc.function(sc.arg("x", 2), outputs=sc.arg("y"))
  def f(x):
    return x

  try:
    _ = f(sc.sym("z", 3))
  except ValueError as e:
    assert "expected shape (2,) for 'x', got (3,)" in str(e)
  else:  # pragma: no cover
    raise AssertionError("call shape mismatch should fail")


def test_parameter_lists_and_inferred_output_nesting() -> None:
  @sc.function(sc.arg("x", 2), sc.group(sc.arg("p", ())))
  def nested(x: sc.Expr, p: tuple[sc.Expr]) -> tuple[sc.Expr, tuple[sc.Expr]]:
    return x * p[0], (x.sum(),)

  result = nested(np.array([2.0, 3.0]), (np.array(4.0),))
  np.testing.assert_array_equal(result[0], [8.0, 12.0])
  np.testing.assert_array_equal(result[1][0], 5.0)
  assert as_concrete(nested).output_names == ("out0", "out1")
  assert nested.symbolic_call(sc.sym("x", 2), (sc.sym("p", ()),))[0].op == sc.ExprOp.CALL


def test_zero_parameter_calls_are_numerical_and_explicitly_symbolic() -> None:
  @sc.function(outputs=sc.arg("value", 2))
  def constant() -> sc.Expr:
    return sc.const([2.0, 3.0])

  np.testing.assert_array_equal(constant(), [2.0, 3.0])
  assert constant.symbolic_call().op == sc.ExprOp.CALL


def test_declaration_rejects_wrong_body_arity() -> None:
  import pytest

  with pytest.raises(TypeError, match="one tree per positional parameter"):
    sc.function(sc.arg("x", 2), sc.arg("y", 2))(lambda x: x)  # ty: ignore[no-matching-overload]
  with pytest.raises(TypeError, match="one tree per positional parameter"):
    sc.function(sc.arg("x", 2))(lambda *, x: x)  # ty: ignore[no-matching-overload]


def test_concrete_functions_are_immutable_and_copies_leave_the_source_unchanged() -> None:
  x = sc.sym("x", 2)
  fn = ConcreteFunction._from_exprs("immutable", [x], [x.sin()], ["x"], ["y"])
  with pytest.raises(AttributeError):
    fn.name = "renamed"  # ty: ignore[invalid-assignment]

  input_tree = sc.group(sc.arg("x", 2))
  output_tree = sc.group(sc.arg("y", 2))
  typed = fn._with_trees(input_tree, output_tree)
  assert typed is not fn and typed.input_tree is input_tree and typed.output_tree is output_tree
  assert fn.input_tree is not input_tree and fn.output_tree is not output_tree
  with pytest.raises(ValueError, match="input tree"):
    fn._with_trees(sc.group(sc.arg("x", 3)), output_tree)

  placed = fn.with_device("host")
  assert placed is not fn and placed.outputs == fn.outputs


def test_build_takes_trees_and_records_descriptor_and_role() -> None:
  x = sc.sym("x", 2)
  input_tree, output_tree = sc.group(sc.arg("x", 2)), sc.arg("y", 2)
  descriptor = object()
  fn = ConcreteFunction.build("built", input_tree, [x], output_tree, [x.cos()], descriptor=descriptor, role="forward")
  assert (fn.descriptor, fn.role, fn.input_names, fn.output_names) == (descriptor, "forward", ("x",), ("y",))
  np.testing.assert_allclose(fn(np.array([0.1, 0.2])), np.cos([0.1, 0.2]))
  assert fn._replace(name="renamed").descriptor is descriptor
  with pytest.raises(ValueError, match="output tree"):
    ConcreteFunction.build("bad", input_tree, [x], sc.arg("y", 3), [x])
  with pytest.raises(ValueError, match="non-input"):
    ConcreteFunction.build("bad", sc.arg("c", ()), [sc.const(2.0)], sc.arg("y", ()), [sc.const(2.0)])
  with pytest.raises(TypeError, match="build"):
    ConcreteFunction("bad", input_tree, [x], output_tree, [x], (None,), (None,), fn.device)  # ty: ignore[invalid-argument-type]
  with pytest.raises(AttributeError):
    output_tree.names = ("renamed",)
  with pytest.raises(AttributeError):
    del output_tree.names


def test_call_helpers_record_their_role() -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y"))
  def inner(x):
    return (x * x).sum()

  @sc.function(sc.arg("z", (3, 2)), outputs=sc.arg("out"))
  def outer(z):
    return sc.vmap(inner, 3)(z).sum()

  def roles(derived) -> set[str | None]:
    return {expr.attrs["callee"].role for expr in topo(as_concrete(derived).outputs) if expr.op in (sc.ExprOp.CALL, sc.ExprOp.VMAP)}

  assert roles(outer) == {None}
  assert roles(sc.forward(outer, "out", "z")) == {"forward"}
  assert roles(sc.adjoint(outer, "out", "z")) == {"adjoint"}
