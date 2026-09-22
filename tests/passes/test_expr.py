from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

import scaly as sc
from scaly.ir.types import Lowering


def test_cse_merges_equivalent_subgraphs() -> None:
  x = sc.sym("x", 2)
  y = sc.cse((x + 1.0) * (x + 1.0))

  assert y.op == sc.ExprOp.MUL
  assert y.args[0] is y.args[1]
  np.testing.assert_allclose(_eval("cse_eval", sc.L("x", 2), lambda x: sc.cse((x + 1.0) * (x + 1.0)), np.array([2.0, 3.0])), np.array([9.0, 16.0]))

  a, b = x[0], x[1]
  z = sc.cse(sc.stack([a * b, b * a]))
  assert z.args[0] is z.args[1]


def test_simplify_rewrites_algebraic_identities_and_folds_constants() -> None:
  x = sc.sym("x", 3)
  y = sc.simplify(((x + 0.0) * 1.0).reshape((3,)))
  assert y is x

  c = sc.simplify((sc.const([1.0, 2.0]) + sc.const([3.0, 4.0])).sum())
  assert c.op == sc.ExprOp.CONST
  assert c.value is not None
  np.testing.assert_allclose(c.value, 10.0)

  z = sc.simplify(x * 0.0)
  assert z.op == sc.ExprOp.CONST
  assert z.value is not None
  np.testing.assert_allclose(z.value, np.zeros(3))

  m = sc.simplify(sc.const(np.zeros((2, 3))) @ sc.sym("v", 3))
  assert m.op == sc.ExprOp.CONST
  assert m.value is not None
  np.testing.assert_allclose(m.value, np.zeros(2))

  q = sc.sym("q", 2)
  qv = np.array([2.0, 3.0])
  np.testing.assert_allclose(_eval("simp_sub", sc.L("q", 2), lambda q: sc.simplify(q - q), qv), np.zeros(2))
  np.testing.assert_allclose(_eval("simp_div", sc.L("q", 2), lambda q: sc.simplify(q / q), qv), np.ones(2))
  np.testing.assert_allclose(_eval("simp_cse", sc.L("q", 2), lambda q: sc.simplify(sc.cse(q + q)), qv), np.array([4.0, 6.0]))

  assert sc.simplify(q**1.0) is q
  power_zero = sc.simplify(q**0.0)
  assert power_zero.op == sc.ExprOp.CONST
  assert power_zero.value is not None
  np.testing.assert_allclose(power_zero.value, np.ones(2))


def test_simplify_constant_folds_erf() -> None:
  values = np.array([-2.0, 0.0, 0.5, 4.5])
  folded = sc.simplify(sc.const(values).erf())

  assert folded.op == sc.ExprOp.CONST
  assert folded.value is not None
  assert folded.value.dtype == np.float64
  np.testing.assert_allclose(folded.value, [math.erf(float(x)) for x in values], rtol=1e-15, atol=1e-15)


def test_simplify_folds_matrix_transpose_into_matmul() -> None:
  A, B, v, w = sc.sym("A", (3, 4)), sc.sym("B", (3, 5)), sc.sym("v", 3), sc.sym("w", 4)

  folded = sc.simplify(A.T @ v)
  assert folded.op == sc.ExprOp.MATMUL and folded.args[0] is v and folded.args[1] is A
  folded = sc.simplify(w @ A.T)
  assert folded.op == sc.ExprOp.MATMUL and folded.args[0] is A and folded.args[1] is w

  # Only a matrix-vector product loses its transpose; mat @ mat keeps it.
  kept = sc.simplify(A.T @ B)
  assert kept.args[0].op == sc.ExprOp.TRANSPOSE and kept.args[0].args[0] is A


def _eval(name: str, inputs: sc.L, body: Callable[[sc.Expr], sc.Expr], value: np.ndarray) -> np.ndarray:
  return sc.function(inputs, sc.L("y", ...), name=name)(body)(value)


def test_matmul_with_ones_vector_becomes_sums() -> None:
  v, A = sc.sym("v", 3), sc.sym("A", (2, 3))
  ones = sc.const(np.ones(3))
  rng = np.random.default_rng(3)
  vv = rng.normal(size=3)

  dot = sc.simplify(v @ ones)
  assert dot.op == sc.ExprOp.SUM and dot.args[0] is v
  assert sc.simplify(ones @ v) is dot
  np.testing.assert_allclose(_eval("ones_dot", sc.L("v", 3), lambda v: sc.simplify(v @ ones), vv), vv.sum(), rtol=1e-14)

  assert sc.simplify(A @ ones).op == sc.ExprOp.MATMUL  # matrix forms wait for an axis reduction
  assert sc.simplify(sc.const(np.ones(2)) @ A).op == sc.ExprOp.MATMUL
  not_ones = sc.simplify(A @ sc.const([1.0, 2.0, 1.0]))
  assert not_ones.op == sc.ExprOp.MATMUL


def test_gather_with_identity_indices_is_a_reshape() -> None:
  x, M = sc.sym("x", 3), sc.sym("M", (2, 3))
  assert sc.simplify(sc.gather(x, np.arange(3))) is x
  reshaped = sc.simplify(sc.gather(M, np.arange(6).reshape(3, 2)))
  assert reshaped.op == sc.ExprOp.RESHAPE and reshaped.args[0] is M and reshaped.shape == (3, 2)
  Mv = np.arange(6, dtype=float).reshape(2, 3) / 7
  np.testing.assert_array_equal(
    _eval("identity_gather", sc.L("M", (2, 3)), lambda M: sc.simplify(sc.gather(M, np.arange(6).reshape(3, 2))), Mv), Mv.reshape(3, 2)
  )
  permuted = sc.simplify(sc.gather(x, np.array([2, 0, 1])))
  assert permuted.op == sc.ExprOp.GATHER
  partial = sc.simplify(sc.gather(x, np.array([0, 1])))
  assert partial.op == sc.ExprOp.GATHER


def test_constant_mask_of_the_result_shape_folds_when_uniform() -> None:
  x = sc.sym("x", 3)
  assert sc.simplify(x * sc.const([1.0, 1.0, 1.0])) is x
  zeros = sc.simplify(x * sc.const([0.0, 0.0, 0.0]))
  assert zeros.op == sc.ExprOp.CONST and zeros.value is not None and not zeros.value.any()
  mixed = sc.simplify(x * sc.const([1.0, 0.0, 1.0]))
  assert mixed.op == sc.ExprOp.MUL
  # a mask of another shape never stands in for the result
  row = sc.sym("row", (1, 3))
  kept = sc.simplify(row * sc.const(np.ones((2, 3))))
  assert kept.op == sc.ExprOp.MUL and kept.shape == (2, 3)
  data = np.array([0.5, -1.5, 2.5])
  np.testing.assert_array_equal(_eval("mask_mixed", sc.L("x", 3), lambda x: sc.simplify(x * sc.const([1.0, 0.0, 1.0])), data), data * [1.0, 0.0, 1.0])
  np.testing.assert_array_equal(
    _eval("mask_broadcast", sc.L("row", (1, 3)), lambda row: sc.simplify(row * sc.const(np.ones((2, 3)))), data[None, :]), np.tile(data, (2, 1))
  )


@pytest.mark.parametrize("hint", ["scalar", "block", "opaque"])
def test_simplify_keeps_hint_when_replacement_is_declared_input(hint: Lowering) -> None:
  @sc.function(sc.L("x", 3), sc.L("y", ...), name="hinted_identity")
  def fn(x: sc.Expr) -> sc.Expr:
    return sc.simplify((x * 1.0).with_lowering(hint))

  (x,), (simplified,) = fn.inputs, fn.outputs
  assert simplified.lowering == hint
  assert simplified.op == sc.ExprOp.RESHAPE
  assert simplified.args[0] is x


def test_simplify_revisits_new_children_in_one_bounded_walk() -> None:
  x = sc.sym("x", 2)
  expr = (x + 0.0).reshape((2,)).reshape((2,)) * 1.0

  assert sc.simplify(expr) is x


def test_simplify_preserves_conflicting_hint_precedence() -> None:
  x = sc.sym("x", 2)

  assert sc.simplify((x.sin().block() + 0.0).scalar()).lowering == "block"
  annihilated = sc.simplify(x.block() * 0.0)
  assert annihilated.op == sc.ExprOp.CONST
  assert annihilated.lowering == "block"


def test_simplify_preserves_isolated_opaque_spelling() -> None:
  x = sc.sym("x", 2)

  assert sc.simplify((x + 0.0).opaque()).lowering == "opaque"
