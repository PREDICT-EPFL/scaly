from __future__ import annotations

import numpy as np

import alloy as al
from alloy.ir.expr import topo


def test_function_call_node_eval() -> None:
  x = al.sym("x", 2)
  inner = al.Function("inner", [x], [x.sin()], ["x"], ["y"])
  z = al.sym("z", 2)
  (inner_z,) = inner.call([z])
  outer = al.Function("outer", [z], [inner_z + 1.0], ["z"], ["out"])

  np.testing.assert_allclose(outer(np.array([0.1, 0.2])), np.sin([0.1, 0.2]) + 1.0)
  assert any(e.op == al.ExprOp.CALL for e in topo(outer.outputs))


def test_function_call_normalizes_raw_constant_args() -> None:
  x = al.sym("x", 2)
  inner = al.Function("inner", [x], [x + 1.0], ["x"], ["y"])
  (inner_const,) = inner.call([[1.0, 2.0]])
  outer = al.Function("outer", [], [inner_const], [], ["out"])

  np.testing.assert_allclose(outer(), np.array([2.0, 3.0]))
  assert not inner_const.type.diff


def test_function_signature_and_call_shape_errors() -> None:
  x = al.sym("x", 2)

  try:
    _ = al.Function("bad", [x], [x], [], ["y"])
  except ValueError as e:
    assert "expected 1 input names, got 0" in str(e)
  else:  # pragma: no cover
    raise AssertionError("input name arity mismatch should fail")

  y = al.sym("y", 2)
  try:
    _ = al.Function("missing", [x], [x + y], ["x"], ["z"])
  except ValueError as e:
    assert "undeclared symbolic inputs: ['y']" in str(e)
  else:  # pragma: no cover
    raise AssertionError("undeclared graph input should fail")

  f = al.Function("f", [x], [x], ["x"], ["y"])
  try:
    _ = f.call([al.sym("z", 3)])
  except ValueError as e:
    assert "call argument 'x' has shape (3,), expected (2,)" in str(e)
  else:  # pragma: no cover
    raise AssertionError("call shape mismatch should fail")
