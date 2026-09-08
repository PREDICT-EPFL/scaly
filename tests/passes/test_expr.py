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


def _eval(name: str, inputs: list[al.Expr], y: al.Expr, *values: np.ndarray) -> np.ndarray:
  fn = al.Function._from_exprs(name, inputs, [y], [str(x.name) for x in inputs], ["y"])
  return fn(values[0] if len(values) == 1 else values)


def test_matmul_with_ones_vector_becomes_sums() -> None:
  v, A = al.sym("v", 3), al.sym("A", (2, 3))
  ones = al.const(np.ones(3))
  rng = np.random.default_rng(3)
  vv = rng.normal(size=3)

  dot = al.simplify(v @ ones)
  assert dot.op == al.ExprOp.SUM and dot.args[0] is v
  assert al.simplify(ones @ v) is dot
  np.testing.assert_allclose(_eval("ones_dot", [v], dot, vv), vv.sum(), rtol=1e-14)

  assert al.simplify(A @ ones).op == al.ExprOp.MATMUL  # matrix forms wait for an axis reduction
  assert al.simplify(al.const(np.ones(2)) @ A).op == al.ExprOp.MATMUL
  not_ones = al.simplify(A @ al.const([1.0, 2.0, 1.0]))
  assert not_ones.op == al.ExprOp.MATMUL


def test_gather_with_identity_indices_is_a_reshape() -> None:
  x, M = al.sym("x", 3), al.sym("M", (2, 3))
  assert al.simplify(al.gather(x, np.arange(3))) is x
  reshaped = al.simplify(al.gather(M, np.arange(6).reshape(3, 2)))
  assert reshaped.op == al.ExprOp.RESHAPE and reshaped.args[0] is M and reshaped.shape == (3, 2)
  Mv = np.arange(6, dtype=float).reshape(2, 3) / 7
  np.testing.assert_array_equal(_eval("identity_gather", [M], reshaped, Mv), Mv.reshape(3, 2))
  permuted = al.simplify(al.gather(x, np.array([2, 0, 1])))
  assert permuted.op == al.ExprOp.GATHER
  partial = al.simplify(al.gather(x, np.array([0, 1])))
  assert partial.op == al.ExprOp.GATHER


def test_constant_mask_of_the_result_shape_folds_when_uniform() -> None:
  x = al.sym("x", 3)
  assert al.simplify(x * al.const([1.0, 1.0, 1.0])) is x
  zeros = al.simplify(x * al.const([0.0, 0.0, 0.0]))
  assert zeros.op == al.ExprOp.CONST and zeros.value is not None and not zeros.value.any()
  mixed = al.simplify(x * al.const([1.0, 0.0, 1.0]))
  assert mixed.op == al.ExprOp.MUL
  # a mask of another shape never stands in for the result
  row = al.sym("row", (1, 3))
  kept = al.simplify(row * al.const(np.ones((2, 3))))
  assert kept.op == al.ExprOp.MUL and kept.shape == (2, 3)
  data = np.array([0.5, -1.5, 2.5])
  np.testing.assert_array_equal(_eval("mask_mixed", [x], mixed, data), data * [1.0, 0.0, 1.0])
  np.testing.assert_array_equal(_eval("mask_broadcast", [row], kept, data[None, :]), np.tile(data, (2, 1)))
