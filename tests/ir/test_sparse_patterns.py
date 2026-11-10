"""Stored-pattern identity, storage and fixed expression rules against NumPy references."""

from dataclasses import FrozenInstanceError, fields
from concurrent.futures import ThreadPoolExecutor
import gc
from threading import Barrier, Event
import weakref

import numpy as np
import pytest

from scaly.ir.expr import Expr, ExprOp, _attrs_key, binary, substitute, unary
from scaly.ir.expr_spec import verify_expr
from scaly.ir.spec import VerifyError
from scaly.ir.types import CanonicalPattern, SparsityPattern, TensorType
from scaly.passes.expr import cse_many


def _pattern(mask: np.ndarray) -> CanonicalPattern:
  return SparsityPattern.from_mask(mask).canonicalize()[0]


def _input(name: str, mask: np.ndarray, batch: tuple[int, ...] = ()) -> Expr:
  return Expr(ExprOp.INPUT, type=TensorType((*batch, *mask.shape), pattern=_pattern(mask)), name=name)


def _forge(node: Expr, **changes) -> Expr:
  forged = object.__new__(Expr)
  for field in fields(Expr):
    object.__setattr__(forged, field.name, changes.get(field.name, getattr(node, field.name)))
  return forged


def test_canonicalization_preserves_value_permutation_and_explicit_zeros() -> None:
  arbitrary = SparsityPattern((3, 3), (2, 0, 1), (0, 2, 1))
  values = np.array([0.0, 7.0, 4.0])
  pattern, perm = arbitrary.canonicalize()
  assert pattern.rows == (0, 1, 2) and pattern.cols == (2, 1, 0)
  assert perm == (1, 2, 0)
  assert arbitrary.rows == (2, 0, 1)
  expected = np.zeros((3, 3))
  expected[arbitrary.rows, arbitrary.cols] = values
  actual = np.zeros((3, 3))
  actual[pattern.rows, pattern.cols] = values[list(perm)]
  np.testing.assert_array_equal(actual, expected)
  assert pattern.nnz == 3
  assert CanonicalPattern((3, 3), (1, 2, 0), (1, 0, 2)) is pattern
  assert pattern.digest == arbitrary.canonicalize()[0].digest
  assert _attrs_key({"pattern": pattern}) == (("pattern", pattern),)
  assert CanonicalPattern.__eq__ is object.__eq__
  assert CanonicalPattern.__hash__ is object.__hash__
  with pytest.raises(FrozenInstanceError):
    pattern.rows = ()  # ty: ignore[invalid-assignment]
  reference = weakref.ref(CanonicalPattern((91, 17), (90,), (16,)))
  gc.collect()
  assert reference() is None


@pytest.mark.parametrize("shape", [(), (3,), (2, 3), (0, 2, 3), (4, 5, 2, 3)])
def test_tensor_storage_shape_and_size(shape: tuple[int, ...]) -> None:
  dense = TensorType(shape)
  assert dense.storage_shape == shape and dense.storage_size == dense.size
  pattern = CanonicalPattern((2, 3), (0, 1), (2, 0))
  if len(shape) < 2:
    with pytest.raises(ValueError, match="last two axes"):
      TensorType(shape, pattern=pattern)
  else:
    typed = TensorType(shape, pattern=pattern)
    assert typed.size == int(np.prod(shape))
    assert typed.storage_shape == (*shape[:-2], 2)
    assert typed.storage_size == int(np.prod((*shape[:-2], 2)))
    assert typed.pattern is pattern
  assert TensorType((7, 2, 3), pattern=CanonicalPattern((2, 3), (), ())).storage_size == 0
  with pytest.raises(ValueError, match="last two axes"):
    TensorType((2, 4), pattern=pattern)
  with pytest.raises(TypeError, match="canonical"):
    TensorType((2, 3), pattern=SparsityPattern((2, 3), (), ()))  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("op,reference", [(ExprOp.ADD, np.logical_or), (ExprOp.SUB, np.logical_or), (ExprOp.MUL, np.logical_and)])
def test_binary_patterns_match_numpy(op: ExprOp, reference) -> None:
  rng = np.random.default_rng(39)
  for _ in range(30):
    a, b = rng.random((4, 5)) < 0.35, rng.random((4, 5)) < 0.35
    x, y = _input("pattern_a", a), _input("pattern_b", b)
    result = binary(op, x, y)
    assert result.type.pattern is _pattern(reference(a, b))
    assert result.type.storage_size == int(reference(a, b).sum())
    verify_expr(result)


def test_broadcasting_shares_batch_pattern_and_expands_matrix_axes() -> None:
  a = np.array([[True, False, True]])
  b = np.array([[False], [True]])
  x, y = _input("broadcast_a", a, (4, 1)), _input("broadcast_b", b, (3,))
  result = x * y
  assert result.shape == (4, 3, 2, 3)
  assert result.type.pattern is _pattern(a & b)
  assert result.type.storage_shape == (4, 3, 2)
  dense = Expr.sym("broadcast_dense", (2, 3))
  assert (result * dense).type.pattern is result.type.pattern
  assert (x + dense).type.pattern is _pattern(np.ones((2, 3), dtype=bool))
  assert (x * 0.0).type.pattern is x.type.pattern
  verify_expr(result)


def test_matmul_pattern_is_boolean_product_including_empty_and_dense_operands() -> None:
  rng = np.random.default_rng(404)
  for m, k, n in [(3, 4, 5), (0, 2, 4), (3, 0, 2), (3, 2, 0)]:
    for _ in range(10):
      a, b = rng.random((m, k)) < 0.4, rng.random((k, n)) < 0.4
      x, y = _input("product_a", a), _input("product_b", b)
      result = x @ y
      assert result.type.pattern is _pattern(a @ b)
      verify_expr(result)
      dense = Expr.sym("product_dense", b.shape)
      assert (x @ dense).type.pattern is _pattern(a @ np.ones(b.shape, dtype=bool))
  a = _input("product_large_count", np.ones((1, 256), dtype=bool))
  b = _input("product_large_count_b", np.ones((256, 1), dtype=bool))
  result = (a @ b).type.pattern
  assert result is not None and result.nnz == 1


@pytest.mark.parametrize(
  "op",
  [
    ExprOp.NEG,
    ExprOp.SIN,
    ExprOp.TAN,
    ExprOp.ASIN,
    ExprOp.ATAN,
    ExprOp.SINH,
    ExprOp.TANH,
    ExprOp.ERF,
    ExprOp.SQRT,
    ExprOp.ABS,
    ExprOp.FLOOR,
    ExprOp.CEIL,
    ExprOp.STOP_GRADIENT,
  ],
)
def test_zero_preserving_unary_retains_pattern(op: ExprOp) -> None:
  x = _input("unary_pattern", np.eye(3, dtype=bool))
  result = unary(op, x)
  assert result.type.pattern is x.type.pattern
  verify_expr(result)


@pytest.mark.parametrize(
  "op",
  [
    ExprOp.COS,
    ExprOp.ACOS,
    ExprOp.COSH,
    ExprOp.EXP,
    ExprOp.LOG,
    ExprOp.SUM,
    ExprOp.SOLVER_CALL,
    ExprOp.PRINT,
    ExprOp.DIV,
    ExprOp.POW,
    ExprOp.ATAN2,
    ExprOp.MINIMUM,
    ExprOp.MAXIMUM,
  ],
)
def test_missing_pattern_rules_refuse_construction_and_forged_nodes(op: ExprOp) -> None:
  x = _input("unsupported_pattern", np.eye(3, dtype=bool))
  args = (x, x) if op in {ExprOp.DIV, ExprOp.POW, ExprOp.ATAN2, ExprOp.MINIMUM, ExprOp.MAXIMUM} else (x,)
  with pytest.raises(ValueError, match="to_dense"):
    Expr(op, args, x.type)
  bad = _forge(x, op=op, args=args)
  with pytest.raises(VerifyError, match="stored-pattern"):
    verify_expr(bad)


def test_patterns_participate_in_interning_structural_keys_substitution_and_verification() -> None:
  mask = np.eye(3, dtype=bool)
  x, same, different = _input("pattern_identity", mask), _input("pattern_identity", mask), _input("pattern_identity", ~mask)
  dense = Expr.sym("pattern_identity", (3, 3))
  assert x is same and x is not different and x is not dense
  assert x.structural_key() != different.structural_key() != dense.structural_key()
  assert cse_many([x, different, dense]) == (x, different, dense)
  z = _input("pattern_replacement", mask)
  assert substitute(x.sin(), {x: z}) is z.sin()
  for bad in (different, dense):
    with pytest.raises(ValueError, match="cannot substitute"):
      substitute(x.sin(), {x: bad})
  valid = x.sin()
  for bad_type in (dense.type, different.type):
    with pytest.raises(ValueError, match="pattern"):
      Expr(valid.op, valid.args, bad_type)
    with pytest.raises(VerifyError, match="stored-pattern"):
      verify_expr(_forge(valid, type=bad_type))
  with pytest.raises(ValueError, match="pattern"):
    Expr(ExprOp.CONST, type=x.type, value=np.eye(3))


def test_digest_is_computed_once_and_numpy_coordinates_normalize(monkeypatch: pytest.MonkeyPatch) -> None:
  from scaly.ir import types

  pattern = CanonicalPattern((11, 13), (10, 0), (12, 3))
  digest = pattern.digest

  def unexpected_digest(*args, **kwargs):
    raise AssertionError("an intern hit recomputed the digest")

  monkeypatch.setattr(types, "sha256", unexpected_digest)
  same = CanonicalPattern((np.int64(11), np.int64(13)), (np.int32(0), np.int32(10)), (np.int32(3), np.int32(12)))  # ty: ignore[invalid-argument-type]
  assert same is pattern and same.digest == digest
  assert TensorType((11, 13), pattern=same) == TensorType((11, 13), pattern=pattern)
  assert hash(TensorType((11, 13), pattern=pattern)) == hash(TensorType((11, 13), pattern=same))


@pytest.mark.parametrize("rows,cols", [((0, 0), (1, 1)), ((-1,), (0,)), ((3,), (0,)), ((0,), (3,)), ((0,), ())])
def test_canonical_patterns_reject_invalid_coordinates(rows, cols) -> None:
  with pytest.raises(ValueError):
    CanonicalPattern((3, 3), rows, cols)


def test_pattern_copies_mutable_input_and_verifier_checks_shape() -> None:
  rows, cols = [0, 1], [1, 0]
  arbitrary = SparsityPattern((2, 2), rows, cols)  # ty: ignore[invalid-argument-type]
  pattern, _ = arbitrary.canonicalize()
  rows[0], cols[0] = 1, 1
  assert arbitrary.rows == (0, 1) and pattern.rows == (0, 1)
  x = _input("forged_pattern_shape", np.eye(2, dtype=bool))
  invalid = object.__new__(TensorType)
  for field in fields(TensorType):
    object.__setattr__(invalid, field.name, getattr(x.type, field.name))
  object.__setattr__(invalid, "pattern", CanonicalPattern((3, 3), (), ()))
  with pytest.raises(VerifyError, match="last two axes"):
    verify_expr(_forge(x, type=invalid))


def test_sparse_graph_crosses_lowering_boundary() -> None:
  from scaly.function.concrete import ConcreteFunction

  x = _input("boundary_pattern", np.eye(3, dtype=bool))
  function = ConcreteFunction._from_exprs("sparse_boundary", [x], [x.sin()], ["x"], ["y"])
  np.testing.assert_allclose(function(np.arange(3.0)).toarray(), np.diag(np.sin(np.arange(3.0))))


def test_concurrent_canonicalization_has_one_identity_and_digest(monkeypatch: pytest.MonkeyPatch) -> None:
  from scaly.ir import types

  original = types.sha256
  calls = []

  def digest(data):
    calls.append(data)
    Event().wait(0.05)
    return original(data)

  monkeypatch.setattr(types, "sha256", digest)
  start = Barrier(8)

  def build(_):
    start.wait()
    return CanonicalPattern((101, 103), (0, 100), (102, 0))

  with ThreadPoolExecutor(max_workers=8) as pool:
    patterns = list(pool.map(build, range(8)))
  assert all(pattern is patterns[0] for pattern in patterns)
  assert len(calls) == 1
  x = Expr(ExprOp.INPUT, type=TensorType((101, 103), pattern=patterns[0]), name="thread_pattern_x")
  y = Expr(ExprOp.INPUT, type=TensorType((101, 103), pattern=patterns[-1]), name="thread_pattern_y")
  assert substitute(x.sin(), {x: y}) is y.sin()


def test_simplification_preserves_stored_patterns() -> None:
  from scaly.passes.expr import simplify

  x = _input("simplify_pattern", np.eye(3, dtype=bool))
  for expr in (x - x, x * 0.0, x * 1.0, x + 0.0, -(x + 0.0), (x + 0.0).sin(), x @ Expr.const(np.zeros((3, 3))), x + x, -(-x)):
    result = simplify(expr)
    assert result.type.pattern is expr.type.pattern
    assert result.type.storage_size == expr.type.storage_size
    verify_expr(result)
  for lowering in ("scalar", "block", "opaque"):
    expr = (x * 1.0).with_lowering(lowering)
    result = simplify(expr)
    assert result.type.pattern is expr.type.pattern and result.lowering == lowering
    verify_expr(result)
  mixed = x.scalar() * Expr.const(1.0).block()
  for expr in (mixed, -(-mixed)):
    assert simplify(expr).type.pattern is expr.type.pattern
