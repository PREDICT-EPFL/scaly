"""Value support rules, cache behavior and randomized soundness against NumPy."""

import math

from scaly.ir.expr import _scatter_zero

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


@pytest.mark.parametrize("operation", ["stack", "concat"])
def test_support_combines_many_parts_in_one_coordinate_batch(operation, monkeypatch) -> None:
  part = _scatter_zero(sc.sym("many_support_entries", 2048), np.arange(2048) * 2, (4096,))
  support = part.support
  assert support is not None
  cls = type(support)
  original = cls.__or__
  work = 0

  def union(left, right):
    nonlocal work
    work += len(left) + len(right)
    return original(left, right)

  monkeypatch.setattr(cls, "__or__", union)
  result = getattr(sc, operation)([part] * 64).support
  assert result is not None and len(result) == 64 * 2048
  assert work <= 3 * len(result)


def test_compact_support_set_operations_and_sorted_coordinates() -> None:
  from scaly.ir.support import support_indices

  support = _scatter_zero(sc.sym("set_operations", 2048), np.arange(2048) * 2, (4096,)).support
  assert support is not None
  reference, small = frozenset(range(0, 4096, 2)), frozenset({8, 1000, 1001})
  assert support - small == reference - small
  assert small - support == small - reference
  assert support ^ small == reference ^ small
  assert hash(support) == hash(reference)
  np.testing.assert_array_equal(support_indices(frozenset([1000, 8])), [8, 1000])


@pytest.mark.parametrize("width", [400, 2048])
def test_growing_support_caches_use_compact_immutable_coordinates(width) -> None:
  from dataclasses import FrozenInstanceError
  from scaly.ir.support import support_indices

  x = sc.sym("support_partial_sums", width)
  parts = [_scatter_zero(x, np.arange(width) + i * 4096, (100_000_000,)) for i in range(40)]
  sums = [parts[0]]
  for part in parts[1:]:
    sums.append(sums[-1] + part)
  assert sums[-1].support is not None and len(sums[-1].support) == 40 * width
  for node in sums:
    support = node.support
    assert support is not None
    if len(support) <= 1024:
      assert isinstance(support, frozenset)
      continue
    coordinates = support_indices(support)
    assert coordinates.nbytes == len(support) * 8
    with pytest.raises(ValueError):
      coordinates.flags.writeable = True
    with pytest.raises((FrozenInstanceError, AttributeError)):
      support._entries = np.array([])
  assert (sums[-1] + sums[-1]).support is sums[-1].support


def test_support_is_lazy_cached_immutable_and_outside_identity(monkeypatch) -> None:
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
  import scaly.ir.support as support_module

  def no_walk(*args):
    pytest.fail("a cached root must not start a support walk")

  monkeypatch.setattr(support_module, "_run_support", no_walk)
  assert expr.support is support


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
  assert _scatter_zero(values, [2, 4, 2, 0], (6,)).support == frozenset({2})
  assert _scatter_zero(sc.sym("support_scatter_unknown", 4), [2, 4, 2, 0], (6,)).support == frozenset({0, 2, 4})
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


def test_calls_specialize_callee_support_and_maps_tile_it() -> None:
  @sc.function(sc.arg("x", 3), outputs=sc.arg("y"), name="support_callee")
  def callee(x):
    return x * sc.const([0, 1, 0])

  assert callee(sc.const([0, 0, 0])).support == frozenset()
  for length in (0, 1, 4):
    actual = sc.sym("support_map", 3 * length)
    mapped = _mapped_call(callee, length, [(actual, 0, 3)])
    assert mapped.support == frozenset(3 * i + 1 for i in range(length))


def test_call_support_cache_is_shared_by_outputs_and_argument_support() -> None:
  from scaly.function.model import as_concrete
  from scaly.ir.support import specialize_support

  @sc.function(sc.arg("x", 4), outputs=sc.group(sc.arg("a"), sc.arg("b")), name="support_siblings")
  def callee(x):
    return x.sin(), x[::-1]

  concrete = as_concrete(callee)
  x = _masked("support_actual", [0, 1, 0, 1])
  a, b = callee(x)
  assert a.support == frozenset({1, 3}) and b.support == frozenset({0, 2})
  same = _masked("support_other_actual", [0, 1, 0, 1])
  assert callee(same)[0].support is a.support
  context = specialize_support(concrete, (x.support,), OP_INFO)
  assert context(concrete.outputs[0]) is a.support
  assert len(concrete._memo.supports) == 1
  assert concrete.inputs[0].support is None and concrete.outputs[0].support is None


def test_call_support_specializations_are_bounded_and_keep_cached_patterns() -> None:
  from scaly.ir.support import _SPECIALIZATION_LIMIT

  width = _SPECIALIZATION_LIMIT + 9
  x = sc.sym("support_bound_formal", width)
  callee = ConcreteFunction._from_exprs("support_bound", [x], [x.sin()], ["x"], ["y"])
  actuals = [_masked(f"support_bound_{i}", np.arange(width) == i) for i in range(width)]
  for i, actual in enumerate(actuals):
    support = callee(actual).support
    assert support == (frozenset({i}) if i < _SPECIALIZATION_LIMIT else None)
  assert len(callee._memo.supports) == _SPECIALIZATION_LIMIT
  assert callee(actuals[0]).support == frozenset({0})
  assert callee(_masked("support_bound_reused", np.arange(width) == 0)).support == frozenset({0})
  assert len(callee._memo.supports) == _SPECIALIZATION_LIMIT


def test_interrupted_call_support_does_not_publish_an_incomplete_cache(monkeypatch) -> None:
  import scaly.ir.support as support

  x = sc.sym("support_interrupt_formal", 3)
  callee = ConcreteFunction._from_exprs("support_interrupt", [x], [(x.sin() + 1).cos() * x], ["x"], ["y"])
  actual = _masked("support_interrupt_actual", [0, 1, 0])
  original = support._rule

  def interrupt(node, op, support_of):
    if node.op == ExprOp.COS:
      raise KeyboardInterrupt
    return original(node, op, support_of)

  with monkeypatch.context() as patch:
    patch.setattr(support, "_rule", interrupt)
    with pytest.raises(KeyboardInterrupt):
      callee(actual).support
  assert not callee._memo.supports
  assert callee(actual).support == frozenset({1})


def test_specializations_share_argument_independent_leaf_supports() -> None:
  x = sc.sym("support_constant_formal", (40, 40))
  constant = sc.const(np.arange(1600, dtype=np.float64).reshape(40, 40))
  callee = ConcreteFunction._from_exprs("support_shared_constant", [x], [x * constant], ["x"], ["y"])
  for i in range(4):
    actual = _masked(f"support_shared_constant_{i}", (np.arange(1600) == i).reshape(40, 40))
    assert callee(actual).support == (frozenset() if i == 0 else frozenset({i}))
  assert len(callee._memo.supports) == 4
  assert all(context[constant] is constant.support for context in callee._memo.supports.values())


@pytest.mark.parametrize("start,stride", [(0, 3), (2, 2), (1, 0)])
def test_mapped_argument_support_uses_windows_and_repeated_stage_patterns(start, stride) -> None:
  x = sc.sym("support_window_formal", 3)
  callee = ConcreteFunction._from_exprs("support_window", [x], [x.sin()], ["x"], ["y"])
  length = 50
  mask = np.arange(start + (length - 1) * stride + 3) % 3 == 0
  actual = _masked("support_windows", mask)
  mapped = _mapped_call(callee, length, [(actual, start, stride)])
  expected = np.stack([mask[start + i * stride : start + i * stride + 3] for i in range(length)])
  assert mapped.support == _entries(expected)
  assert len(callee._memo.supports) == len({tuple(row) for row in expected})


def test_sparse_mapped_arguments_and_outputs_use_logical_support_coordinates() -> None:
  pattern = SparsityPattern.from_mask(np.eye(3, dtype=bool)).canonicalize()[0]
  x = Expr(ExprOp.INPUT, type=TensorType((3, 3), pattern=pattern), name="support_sparse_formal")
  callee = ConcreteFunction._from_exprs("support_sparse_call", [x], [x.sin()], ["x"], ["y"])
  mask = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 0, 0]])
  actual = sc.pack(_masked("support_sparse_stages", mask), pattern, shape=(4, 3, 3))
  mapped = _mapped_call(callee, 4, [(sc.values(actual).vec(), 0, 3)])
  assert mapped.support == _entries(mask)
  assert callee(sc.pack(_masked("support_sparse_single", mask[0]), pattern)).support == frozenset({0, 8})


def test_nested_call_support_is_iterative_under_argument_specialization() -> None:
  x = sc.sym("support_nested_formal", 3)
  output = x.sin()
  for i in range(1100):
    callee = ConcreteFunction._from_exprs(f"support_specialized_nested_{i}", [x], [output], ["x"], ["y"])
    output = callee(x)
  outer = ConcreteFunction._from_exprs("support_specialized_outer", [x], [output], ["x"], ["y"])
  assert outer(_masked("support_nested_actual", [0, 1, 0])).support == frozenset({1})
  assert output.support is None


def test_randomized_support_soundness_against_numpy() -> None:
  rng = np.random.default_rng(400)
  for case in range(40):
    masks = [rng.random((3, 4)) < 0.35 for _ in range(2)]
    x, y = [_masked(f"sound_{case}_{i}", mask) for i, mask in enumerate(masks)]
    graphs = [x + y, x - y, x * y, x / (y * y + 1), x.sin(), x.maximum(y), x.minimum(y), x.T @ y, x @ y.T]
    a, b = sc.sym(f"sound_formal_a_{case}", (3, 4)), sc.sym(f"sound_formal_b_{case}", (3, 4))
    body = [a + b, a - b, a * b, a / (b * b + 1), a.sin(), a.maximum(b), a.minimum(b), a.T @ b, a @ b.T]
    callee = ConcreteFunction._from_exprs(f"sound_calls_{case}", [a, b], body, ["a", "b"], [f"out_{i}" for i in range(len(body))])
    calls = callee(x, y)
    maps = [_mapped_call(callee, 3, [(sc.concat([x.vec(), x.vec() * 0, x.vec()]), 0, 12), (y.vec(), 0, 0)], i) for i in range(len(body))]
    zero_x = [x * 0 + y, x * 0 - y, x * 0 * y, x * 0 / (y * y + 1), (x * 0).sin(), (x * 0).maximum(y), (x * 0).minimum(y), (x * 0).T @ y, x * 0 @ y.T]
    for inline, call, mapped, middle in zip(graphs, calls, maps, zero_x, strict=True):
      assert call.support == inline.support
      expected_support = sc.concat([inline.vec(), middle.vec(), inline.vec()]).support
      assert mapped.support == expected_support
    for _ in range(5):
      a, b = [rng.normal(size=(3, 4)) * mask for mask in masks]
      expected = [a + b, a - b, a * b, a / (b * b + 1), np.sin(a), np.fmax(a, b), np.fmin(a, b), a.T @ b, a @ b.T]
      for expr, value in zip(graphs, expected, strict=True):
        support = expr.support
        if support is not None:
          assert _entries(value != 0) <= support
          assert all(value.reshape(-1)[i] == 0 for i in set(range(value.size)) - support)
      for expr, value in zip(calls, expected, strict=True):
        assert expr.support is None or _entries(value != 0) <= expr.support
      middle_values = [b, -b, np.zeros_like(a), np.zeros_like(a), np.zeros_like(a), np.fmax(0, b), np.fmin(0, b), np.zeros((4, 4)), np.zeros((3, 3))]
      for expr, value, middle in zip(maps, expected, middle_values, strict=True):
        actual = np.concatenate([value.reshape(-1), middle.reshape(-1), value.reshape(-1)])
        assert expr.support is None or _entries(actual != 0) <= expr.support


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


def test_predicates_select_and_casts_keep_zeros_only_where_numpy_does() -> None:
  """Support is a superset of NumPy's nonzeros: at an all-zero entry ``<`` and ``!=`` are false, ``<=``,
  ``==``, ``~`` and ``isfinite`` are true, and ``where`` is zero only where both branches are."""
  a, b = (Expr.const(np.array(v, dtype=float)) * sc.sym(f"support_pred_{n}", 4) for n, v in (("a", [0, 1, 0, 1]), ("b", [0, 0, 1, 1])))
  k = sc.cast(sc.cast(a, "int64"), "float64")
  cases = {
    "lt": (a < b, {1, 2, 3}),
    "ne": (sc.not_equal(a, b), {1, 2, 3}),
    "and": ((a > 0.0) & (b > 0.0), {1, 2, 3}),
    "le": (a <= b, None),
    "eq": (sc.equal(a, b), None),
    "not": (~(a > 0.0), None),
    "isfinite": (sc.isfinite(a), None),
    "select": (sc.where(sc.sym("support_pred_c", 4) > 0.0, a, b), {1, 2, 3}),
    "cast": (k, {1, 3}),
    "floordiv": (sc.cast(sc.cast(a, "int64") // sc.cast(b, "int64"), "float64"), {1, 3}),
  }
  for name, (expr, expected) in cases.items():
    assert expr.support == (None if expected is None else frozenset(expected)), name
