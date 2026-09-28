from __future__ import annotations

import numpy as np

import scaly as sc
from scaly.ir.expr import topo


def test_function_call_node_eval() -> None:
  x = sc.sym("x", 2)
  inner = sc.Function.from_exprs("inner", [x], [x.sin()], ["x"], ["y"])
  z = sc.sym("z", 2)
  inner_z = inner(z)
  outer = sc.Function.from_exprs("outer", [z], [inner_z + 1.0], ["z"], ["out"])

  np.testing.assert_allclose(outer(np.array([0.1, 0.2])), np.sin([0.1, 0.2]) + 1.0)
  assert any(e.op == sc.ExprOp.CALL for e in topo(outer.outputs))


def test_function_call_normalizes_raw_constant_args() -> None:
  x = sc.sym("x", 2)
  inner = sc.Function.from_exprs("inner", [x], [x + 1.0], ["x"], ["y"])
  inner_const = inner(sc.const([1.0, 2.0]))
  outer = sc.Function.from_exprs("outer", [], [inner_const], [], ["out"])

  np.testing.assert_allclose(outer(), np.array([2.0, 3.0]))
  assert not inner_const.type.diff


def test_function_signature_and_call_shape_errors() -> None:
  x = sc.sym("x", 2)

  try:
    _ = sc.Function.from_exprs("bad", [x], [x], [], ["y"])
  except ValueError as e:
    assert "expected 1 input names, got 0" in str(e)
  else:  # pragma: no cover
    raise AssertionError("input name arity mismatch should fail")

  y = sc.sym("y", 2)
  try:
    _ = sc.Function.from_exprs("missing", [x], [x + y], ["x"], ["z"])
  except ValueError as e:
    assert "undeclared symbolic inputs: ['y']" in str(e)
  else:  # pragma: no cover
    raise AssertionError("undeclared graph input should fail")

  f = sc.Function.from_exprs("f", [x], [x], ["x"], ["y"])
  try:
    _ = f(sc.sym("z", 3))
  except ValueError as e:
    assert "expected shape (2,) for 'x', got (3,)" in str(e)
  else:  # pragma: no cover
    raise AssertionError("call shape mismatch should fail")
