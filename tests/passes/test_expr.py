from __future__ import annotations

import math

import numpy as np

import alloy as al


def test_cse_merges_equivalent_subgraphs() -> None:
  x = al.sym("x", 2)
  y = al.cse((x + 1.0) * (x + 1.0))

  assert y.op == al.ExprOp.MUL
  assert y.args[0] is y.args[1]
  np.testing.assert_allclose(al.Function._from_exprs("cse_eval", [x], [y], ["x"], ["y"])(np.array([2.0, 3.0])), np.array([9.0, 16.0]))

  a, b = x[0], x[1]
  z = al.cse(al.stack([a * b, b * a]))
  assert z.args[0] is z.args[1]


def test_simplify_rewrites_algebraic_identities_and_folds_constants() -> None:
  x = al.sym("x", 3)
  y = al.simplify(((x + 0.0) * 1.0).reshape((3,)))
  assert y is x

  c = al.simplify((al.const([1.0, 2.0]) + al.const([3.0, 4.0])).sum())
  assert c.op == al.ExprOp.CONST
  assert c.value is not None
  np.testing.assert_allclose(c.value, 10.0)

  z = al.simplify(x * 0.0)
  assert z.op == al.ExprOp.CONST
  assert z.value is not None
  np.testing.assert_allclose(z.value, np.zeros(3))

  m = al.simplify(al.const(np.zeros((2, 3))) @ al.sym("v", 3))
  assert m.op == al.ExprOp.CONST
  assert m.value is not None
  np.testing.assert_allclose(m.value, np.zeros(2))

  q = al.sym("q", 2)
  np.testing.assert_allclose(al.Function._from_exprs("simp_sub", [q], [al.simplify(q - q)], ["q"], ["y"])(np.array([2.0, 3.0])), np.zeros(2))
  np.testing.assert_allclose(al.Function._from_exprs("simp_div", [q], [al.simplify(q / q)], ["q"], ["y"])(np.array([2.0, 3.0])), np.ones(2))
  np.testing.assert_allclose(
    al.Function._from_exprs("simp_cse", [q], [al.simplify(al.cse(q + q))], ["q"], ["y"])(np.array([2.0, 3.0])), np.array([4.0, 6.0])
  )

  assert al.simplify(q**1.0) is q
  power_zero = al.simplify(q**0.0)
  assert power_zero.op == al.ExprOp.CONST
  assert power_zero.value is not None
  np.testing.assert_allclose(power_zero.value, np.ones(2))


def test_simplify_constant_folds_erf() -> None:
  values = np.array([-2.0, 0.0, 0.5, 4.5])
  folded = al.simplify(al.const(values).erf())

  assert folded.op == al.ExprOp.CONST
  assert folded.value is not None
  assert folded.value.dtype == np.float64
  np.testing.assert_allclose(folded.value, [math.erf(float(x)) for x in values], rtol=1e-15, atol=1e-15)


def test_simplify_folds_matrix_transpose_into_matmul() -> None:
  A, B, v, w = al.sym("A", (3, 4)), al.sym("B", (3, 5)), al.sym("v", 3), al.sym("w", 4)

  folded = al.simplify(A.T @ v)
  assert folded.op == al.ExprOp.MATMUL and folded.args[0] is v and folded.args[1] is A
  folded = al.simplify(w @ A.T)
  assert folded.op == al.ExprOp.MATMUL and folded.args[0] is A and folded.args[1] is w

  # Only a matrix-vector product loses its transpose; mat @ mat keeps it.
  kept = al.simplify(A.T @ B)
  assert kept.args[0].op == al.ExprOp.TRANSPOSE and kept.args[0].args[0] is A
