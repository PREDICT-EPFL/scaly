from __future__ import annotations

import math

import numpy as np
import pytest

import scaly as sc
from scaly.ir.types import Lowering


def test_cse_merges_equivalent_subgraphs() -> None:
  x = sc.sym("x", 2)
  y = sc.cse((x + 1.0) * (x + 1.0))

  assert y.op == sc.ExprOp.MUL
  assert y.args[0] is y.args[1]
  np.testing.assert_allclose(sc.Function.from_exprs("cse_eval", [x], [y], ["x"], ["y"])(np.array([2.0, 3.0])), np.array([9.0, 16.0]))

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
  np.testing.assert_allclose(sc.Function.from_exprs("simp_sub", [q], [sc.simplify(q - q)], ["q"], ["y"])(np.array([2.0, 3.0])), np.zeros(2))
  np.testing.assert_allclose(sc.Function.from_exprs("simp_div", [q], [sc.simplify(q / q)], ["q"], ["y"])(np.array([2.0, 3.0])), np.ones(2))
  np.testing.assert_allclose(
    sc.Function.from_exprs("simp_cse", [q], [sc.simplify(sc.cse(q + q))], ["q"], ["y"])(np.array([2.0, 3.0])), np.array([4.0, 6.0])
  )

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


def _eval(name: str, inputs: list[sc.Expr], y: sc.Expr, *values: np.ndarray) -> np.ndarray:
  fn = sc.Function.from_exprs(name, inputs, [y], [str(x.name) for x in inputs], ["y"])
  return fn(values[0] if len(values) == 1 else values)


def test_matmul_with_ones_vector_becomes_sums() -> None:
  v, A = sc.sym("v", 3), sc.sym("A", (2, 3))
  ones = sc.const(np.ones(3))
  rng = np.random.default_rng(3)
  vv = rng.normal(size=3)

  dot = sc.simplify(v @ ones)
  assert dot.op == sc.ExprOp.SUM and dot.args[0] is v
  assert sc.simplify(ones @ v) is dot
  np.testing.assert_allclose(_eval("ones_dot", [v], dot, vv), vv.sum(), rtol=1e-14)

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
  np.testing.assert_array_equal(_eval("identity_gather", [M], reshaped, Mv), Mv.reshape(3, 2))
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
  np.testing.assert_array_equal(_eval("mask_mixed", [x], mixed, data), data * [1.0, 0.0, 1.0])
  np.testing.assert_array_equal(_eval("mask_broadcast", [row], kept, data[None, :]), np.tile(data, (2, 1)))


@pytest.mark.parametrize("hint", ["scalar", "block", "opaque"])
def test_simplify_keeps_hint_when_replacement_is_declared_input(hint: Lowering) -> None:
  x = sc.sym("x", 3)
  simplified = sc.simplify((x * 1.0).with_lowering(hint))
  fn = sc.Function.from_exprs("hinted_identity", [x], [simplified], ["x"], ["y"])

  assert simplified.lowering == hint
  assert simplified.op == sc.ExprOp.RESHAPE
  assert fn.inputs == (x,)


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


def test_gathers_scatters_and_transposes_compose_into_one_gather() -> None:
  """``gather(gather(x))``, a scatter that writes every entry once, and a transpose of a gather each
  simplify to a single gather of the source with the composed index table."""
  from scaly.passes.expr import simplify

  x = sc.sym("cg_x", 12)
  twice = simplify(sc.gather(sc.gather(x, np.arange(12)[::-1].copy()).reshape((3, 4)), np.array([0, 5, 11])))
  assert twice.op == sc.ExprOp.GATHER and twice.args[0] is x
  assert twice.attrs["indices"].tolist() == [11, 6, 0]
  perm = np.array([2, 0, 1, 5, 3, 4, 8, 6, 7, 11, 9, 10])
  scattered = simplify(sc.scatter(x.sin(), perm, (12,)))
  assert scattered.op == sc.ExprOp.GATHER
  transposed = simplify(sc.gather(x, np.arange(12)[::-1].copy()).reshape((3, 4)).T)
  assert transposed.op == sc.ExprOp.GATHER and transposed.args[0] is x
  fn = sc.Function.from_exprs("cg", [x], [twice, scattered, transposed], ["x"], ["a", "b", "c"])
  xv = np.arange(12.0) * 0.3
  a, b, c = fn(xv)
  np.testing.assert_array_equal(a, xv[::-1][[0, 5, 11]])
  expected = np.zeros(12)
  expected[perm] = np.sin(xv)
  np.testing.assert_array_equal(b, expected)
  np.testing.assert_array_equal(c, xv[::-1].reshape(3, 4).T)


def _ops(e: sc.Expr) -> list[sc.Expr]:
  from scaly.ir.expr import topo

  return topo([e])


def test_a_gather_of_a_transpose_reads_what_was_transposed() -> None:
  """A gather's table and the transpose under it compose when the graph is built: the transpose is
  never materialized, whatever it transposes."""
  from scaly.passes.expr import simplify

  x = sc.sym("gt_x", (4, 6))
  picks = np.array([0, 5, 7, 23, 11])
  moved = simplify(sc.gather(x.T, picks))
  assert moved.op == sc.ExprOp.GATHER and moved.args[0].op != sc.ExprOp.TRANSPOSE
  assert moved.attrs["indices"].tolist() == [int(q // 4 + (q % 4) * 6) for q in picks]
  cube = sc.sym("gt_c", (2, 3, 4))
  axes = (2, 0, 1)
  some = np.array([[0, 23], [7, 12]])
  turned = simplify(sc.gather(cube.transpose(axes), some))
  assert turned.shape == (2, 2) and not any(n.op == sc.ExprOp.TRANSPOSE for n in _ops(turned))
  computed = simplify(sc.gather(x.T.sin(), picks))  # an operation between them keeps both
  assert any(n.op == sc.ExprOp.TRANSPOSE for n in _ops(computed))
  fn = sc.Function.from_exprs("gt", [x, cube], [moved, turned], ["x", "c"], ["a", "b"])
  xv, cv = np.arange(24.0).reshape(4, 6) * 0.5, np.arange(24.0).reshape(2, 3, 4) - 7.0
  a, b = fn((xv, cv))
  np.testing.assert_array_equal(a, xv.T.ravel()[picks])
  np.testing.assert_array_equal(b, cv.transpose(axes).ravel()[some])


SCATTERED = {
  "each place once": [9, 7, 11],
  "a place several values went to": [2, 9],
  "nothing was placed there": [0, 9, 5],
  "only places nothing went to": [0, 1],
  "a place read twice": [7, 7, 11, 2],
  "a table of two axes": [[9, 0], [2, 2]],
}


@pytest.mark.parametrize("why", SCATTERED)
def test_a_gather_of_scattered_values_reads_the_values(why: str) -> None:
  """The zero array the values were placed in is never filled: a place gathered takes the values
  scattered to it, added in the order they were scattered, and zero if there were none."""
  from scaly.passes.expr import simplify

  v = sc.sym("gs_v", 6)
  at = np.array([7, 2, 9, 2, 2, 11])  # three values go to place 2, where their order decides the sum
  picks = np.array(SCATTERED[why])
  got = simplify(sc.gather(sc.scatter(v * 2.0, at, (12,)), picks))
  assert got.shape == picks.shape and not any(n.size == 12 for n in _ops(got))
  if why == "each place once":  # one value each: a plain gather of the values
    assert got.op == sc.ExprOp.GATHER and got.attrs["indices"].tolist() == [2, 0, 5]
    assert not any(n.op == sc.ExprOp.SEGMENT_REDUCE for n in _ops(got))
  if why == "only places nothing went to":
    assert got.op == sc.ExprOp.CONST and not np.asarray(got.value).any()
  vv = np.array([1.0, 1.0, 3.0, 1e16, -1e16, 5.0])
  dense = np.zeros(12)
  np.add.at(dense, at, vv * 2.0)
  assert dense[2] == 0.0  # (2 + 2e16) - 2e16: the other order gives 2
  np.testing.assert_array_equal(sc.Function.from_exprs(f"gs_{len(why)}", [v], [got.block()], ["v"], ["g"])(vv), dense[picks])


def test_a_gather_of_another_segment_reduction_keeps_it() -> None:
  """A maximum per place is no sum of placed values: the places are reduced, then gathered."""
  from scaly.ir.expr import segment_max
  from scaly.passes.expr import simplify

  v = sc.sym("gm_v", 5)
  at, picks = np.array([3, 1, 3, 0, 1]), np.array([3, 1])
  got = simplify(sc.gather(segment_max(v, at, 4, fill=-1.0), picks))
  assert got.op == sc.ExprOp.GATHER and got.args[0].op == sc.ExprOp.SEGMENT_REDUCE and got.args[0].size == 4
  vv = np.array([2.0, -3.0, 7.0, 1.0, 4.0])
  np.testing.assert_array_equal(sc.Function.from_exprs("gm", [v], [got.block()], ["v"], ["g"])(vv), [7.0, 4.0])


def test_a_gather_of_a_sum_of_placed_blocks_never_forms_the_array() -> None:
  """A sparse derivative's assembly: blocks scattered into a compressed matrix, summed, the matrix
  transposed, its nonzeros gathered. Simplified, each block's values go straight to the nonzeros
  they reach; nothing of the matrix's size is left. A sum of computed arrays stays one gather."""
  from scaly.passes.expr import simplify

  a, b, c = sc.sym("gp_a", 12), sc.sym("gp_b", (3, 4)), sc.sym("gp_c", 60)
  first = sc.scatter(a, np.arange(12) * 5, (60,)).reshape((6, 10))
  second = sc.scatter(b.T.reshape((12,)) * 3.0, 59 - np.arange(12) * 5, (60,)).reshape((6, 10))
  picks = np.array([0, 6, 12, 30, 59, 54, 1, 35])
  matrix = (first + second + c.reshape((6, 10)).cos()).T
  got = simplify(sc.gather(matrix, picks))
  assert got.shape == (8,) and not any(n.op in (sc.ExprOp.SEGMENT_REDUCE, sc.ExprOp.TRANSPOSE) and n.size == 60 for n in _ops(got))
  plain = simplify(sc.gather((c.sin() + c.cos()).reshape((6, 10)), picks))
  assert plain.op == sc.ExprOp.GATHER and plain.args[0].op in (sc.ExprOp.ADD, sc.ExprOp.RESHAPE)
  av, bv, cv = np.arange(1.0, 13.0), np.arange(12.0).reshape(3, 4) - 5.5, np.linspace(-1.0, 1.0, 60)
  dense = np.zeros(60)
  dense[np.arange(12) * 5] += av
  dense[59 - np.arange(12) * 5] += bv.T.ravel() * 3.0
  want = (dense + np.cos(cv)).reshape(6, 10).T.ravel()[picks]
  np.testing.assert_array_equal(sc.Function.from_exprs("gp", [a, b, c], [got.block()], ["a", "b", "c"], ["g"])((av, bv, cv)), want)
