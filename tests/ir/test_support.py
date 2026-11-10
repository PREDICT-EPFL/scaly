"""Value support rules, cache behavior and randomized soundness against NumPy."""

import math

import numpy as np
import pytest

import scaly as sc
from scaly.function.concrete import ConcreteFunction
from scaly.function.sugar import _mapped_call
from scaly.ir.expr import OP_INFO, Expr, ExprOp, binary, unary
from scaly.ir.types import SparsityPattern, TensorType


def _entries(mask) -> frozenset[int]:
  return frozenset(map(int, np.flatnonzero(mask)))


def _masked(name, mask) -> Expr:
  return sc.sym(name, np.shape(mask)) * sc.const(mask, dtype=sc.dtypes.float64)


def test_growing_support_caches_use_compact_immutable_coordinates() -> None:
  from dataclasses import FrozenInstanceError
  from scaly.ir.support import support_indices

  x = sc.sym("support_partial_sums", 2048)
  parts = [sc.scatter(x, np.arange(2048) + i * 4096, (100_000_000,)) for i in range(40)]
  sums = [parts[0]]
  for part in parts[1:]:
    sums.append(sums[-1] + part)
  assert sums[-1].support is not None and len(sums[-1].support) == 40 * 2048
  for node in sums:
    support = node.support
    assert support is not None
    coordinates = support_indices(support)
    assert coordinates.nbytes == len(support) * 8
    with pytest.raises(ValueError):
      coordinates.flags.writeable = True
    with pytest.raises((FrozenInstanceError, AttributeError)):
      support._entries = np.array([])
  assert (sums[-1] + sums[-1]).support is sums[-1].support


def test_support_is_lazy_cached_immutable_and_outside_identity() -> None:
  x = sc.sym("support_cache", 3)
  expr = x * sc.const([0, 1, 0])
  key = expr.structural_key()
  assert expr._support_cache is None and x._support_cache is None
  support = expr.support
  assert support == frozenset({1}) and expr.support is support
  assert (x * sc.const([0, 1, 0])) is expr
  assert expr.structural_key() is key
  assert isinstance(support, frozenset)
  assert x.support is None
  assert sc.const([0, np.nan, 2, -0.0]).support == frozenset({1, 2})


def test_support_walk_is_iterative_through_shared_graphs_and_callees() -> None:
  x = sc.sym("support_deep", 2)
  expr = x * sc.const([0, 1])
  for _ in range(1500):
    expr = expr + expr
  assert expr.support == frozenset({1})
  for i in range(1100):
    callee = ConcreteFunction._from_exprs(f"support_nested_{i}", [x], [expr], ["x"], ["y"])
    expr = callee(x)
  assert expr.support == frozenset({1})


@pytest.mark.parametrize("op", sorted(op for op, info in OP_INFO.items() if info.pattern_rule == "preserve"))
def test_support_uses_shared_zero_preserving_rules(op: ExprOp) -> None:
  x = _masked("support_unary", [0, 1, 0, 1])
  assert unary(op, x).support == x.support


@pytest.mark.parametrize("op", [ExprOp.COS, ExprOp.ACOS, ExprOp.COSH, ExprOp.EXP, ExprOp.LOG, ExprOp.POW, ExprOp.ATAN2, ExprOp.SOLVER_CALL])
def test_default_support_is_sound_unknown(op: ExprOp) -> None:
  x = sc.const([0, 0])
  args = (x, x) if op in {ExprOp.POW, ExprOp.ATAN2} else (x,)
  assert Expr(op, args, x.type).support is None


@pytest.mark.parametrize("op", [ExprOp.ADD, ExprOp.SUB, ExprOp.MUL, ExprOp.DIV, ExprOp.MAXIMUM, ExprOp.MINIMUM])
def test_elementwise_support_with_unequal_broadcast_ranks(op: ExprOp) -> None:
  a, b = np.array([[1, 0, 1]]), np.array([[[0], [1]], [[1], [0]]])
  x, y = _masked("support_broadcast_a", a), _masked("support_broadcast_b", b)
  expected = (a != 0) & (b != 0) if op == ExprOp.MUL else np.broadcast_to(a != 0, (2, 2, 3)) if op == ExprOp.DIV else (a != 0) | (b != 0)
  assert binary(op, x, y).support == _entries(expected)
  dense = sc.sym("support_broadcast_dense", (2, 2, 3))
  assert (x * dense).support == _entries(np.broadcast_to(a, dense.shape))
  assert (x + dense).support is None


def test_static_coordinate_images_and_reduction() -> None:
  mask = np.array([[[0, 1, 0], [1, 0, 1]], [[1, 0, 1], [0, 0, 0]]], dtype=bool)
  x = _masked("support_movement", mask)
  assert x.reshape((4, 3)).support == _entries(mask)
  assert x.transpose((2, 0, 1)).support == _entries(mask.transpose((2, 0, 1)))
  assert x[::-1, 1, ::-2].support == _entries(mask[::-1, 1, ::-2])
  indices = np.array([[11, 1, 1], [0, 3, 8]])
  assert x.gather(indices).support == _entries(mask.reshape(-1)[indices])
  values = _masked("support_scatter_values", [1, 0, 1, 0])
  assert sc.scatter(values, [2, 4, 2, 0], (6,)).support == frozenset({2})
  assert sc.scatter(sc.sym("support_scatter_unknown", 4), [2, 4, 2, 0], (6,)).support == frozenset({0, 2, 4})
  for axis in range(4):
    assert sc.stack([x, x * 0], axis=axis).support == _entries(np.stack([mask, np.zeros_like(mask)], axis=axis))
  for axis in range(3):
    assert sc.concat([x, x * 0], axis=axis).support == _entries(np.concatenate([mask, np.zeros_like(mask)], axis=axis))
  assert x.sum().support == frozenset({0})
  assert (x * 0).sum().support == frozenset()
  assert sc.print("x={}", x).support == x.support
  assert sc.sym("support_empty", (0, 3)).support == frozenset()


@pytest.mark.parametrize("shapes", [((4,), (4,)), ((3, 4), (4,)), ((4,), (4, 2)), ((3, 4), (4, 2)), ((2, 0), (0, 3))])
def test_matmul_support_is_boolean_product(shapes) -> None:
  rng = np.random.default_rng(40)
  for _ in range(20):
    a, b = [rng.random(shape) < 0.35 for shape in shapes]
    x, y = _masked("support_matmul_a", a), _masked("support_matmul_b", b)
    assert (x @ y).support == _entries(a @ b)
    dense = sc.sym("support_matmul_dense", b.shape)
    assert (x @ dense).support == _entries(a @ np.ones(b.shape, dtype=bool))
  assert (sc.const(np.ones((1, 256))) @ sc.const(np.ones((256, 1)))).support == frozenset({0})


def test_sparse_type_bounds_support_without_changing_stored_pattern() -> None:
  mask = np.eye(3, dtype=bool)
  pattern = SparsityPattern.from_mask(mask).canonicalize()[0]
  x = Expr(ExprOp.INPUT, type=TensorType((2, 3, 3), pattern=pattern), name="support_sparse")
  assert x.support == _entries(np.tile(mask, (2, 1, 1)))
  result = x * sc.const(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), dtype=x.type.dtype)
  assert result.support == frozenset({4, 13}) and result.type.pattern is pattern
  zero = x * 0
  assert zero.support == frozenset() and sc.simplify(zero).type.pattern is pattern
  assert sc.simplify(result * sc.const(1)).type.pattern is pattern


def test_calls_use_callee_support_and_maps_tile_it() -> None:
  @sc.function(sc.arg("x", 3), outputs=sc.arg("y"), name="support_callee")
  def callee(x):
    return x * sc.const([0, 1, 0])

  assert callee(sc.const([0, 0, 0])).support == frozenset({1})
  for length in (0, 1, 4):
    actual = sc.sym("support_map", 3 * length)
    mapped = _mapped_call(callee, length, [(actual, 0, 3)])
    assert mapped.support == frozenset(3 * i + 1 for i in range(length))


def test_randomized_support_soundness_against_numpy() -> None:
  rng = np.random.default_rng(400)
  for case in range(40):
    masks = [rng.random((3, 4)) < 0.35 for _ in range(2)]
    x, y = [_masked(f"sound_{case}_{i}", mask) for i, mask in enumerate(masks)]
    graphs = [x + y, x - y, x * y, x / (y * y + 1), x.sin(), x.maximum(y), x.minimum(y), x.T @ y, x @ y.T]
    for _ in range(5):
      a, b = [rng.normal(size=(3, 4)) * mask for mask in masks]
      expected = [a + b, a - b, a * b, a / (b * b + 1), np.sin(a), np.fmax(a, b), np.fmin(a, b), a.T @ b, a @ b.T]
      for expr, value in zip(graphs, expected, strict=True):
        support = expr.support
        if support is not None:
          assert _entries(value != 0) <= support
          assert all(value.reshape(-1)[i] == 0 for i in set(range(value.size)) - support)


def test_disjoint_products_fold_and_structural_zero_removes_nan() -> None:
  x = _masked("disjoint_a", [1, 0, 1, 0])
  y = _masked("disjoint_b", [0, 1, 0, 1])
  for expr in (x * y, x @ y, (x * y).gather([0, 2]), (x * y).sum()):
    folded = sc.simplify(expr)
    assert folded.op == ExprOp.CONST
    np.testing.assert_array_equal(folded.value, np.zeros(expr.shape))

  @sc.function(sc.arg("x", 2), outputs=sc.arg("y"), name="support_nan_product")
  def fn(x):
    return sc.simplify((x * sc.const([1, 0])) * sc.const([0, math.nan]))

  np.testing.assert_array_equal(fn(np.ones(2)), np.zeros(2))


def test_zero_folding_preserves_dtype_and_lowering_hint() -> None:
  x = sc.sym("support_float32", 2, dtype=sc.dtypes.float32)
  y = x * Expr.const([0, 1], dtype=x.type.dtype)
  expr = (y * Expr.const([1, 0], dtype=x.type.dtype)).scalar()
  result = sc.simplify(expr)
  assert result.type.dtype == x.type.dtype and result.lowering == "scalar"
  assert result.support == frozenset()
