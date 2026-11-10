"""Stored-value conversions and sparse constants."""

import numpy as np
from scipy import sparse

import scaly as sc
from scaly.ir.types import CanonicalPattern
from scaly.function.concrete import ConcreteFunction


def _function(name, inputs, outputs):
  return ConcreteFunction._from_exprs(name, inputs, outputs, [x.name for x in inputs], [f"out{i}" for i in range(len(outputs))])


def test_pack_values_and_dense_projection_match_numpy():
  pattern = CanonicalPattern((3, 4), (0, 1, 2), (3, 0, 2))
  x = sc.sym("x", (2, 3))
  packed = sc.pack(x, pattern)
  assert packed.shape == (2, 3, 4)
  assert packed.type.storage_shape == x.shape
  assert sc.values(packed).shape == x.shape
  assert packed.to_dense().op == sc.ExprOp.RESHAPE
  data = np.arange(6.0).reshape(2, 3)
  expected = np.zeros(packed.shape)
  expected[:, pattern.rows, pattern.cols] = data
  fn = _function("packed_projection", [x], [packed.to_dense(), sc.values(packed)])
  dense, stored = fn(data)
  np.testing.assert_array_equal(dense, expected)
  np.testing.assert_array_equal(stored, data)


def test_scipy_constant_copies_sums_duplicates_and_keeps_zeros():
  source = sparse.coo_array(([2.0, -2.0, 0.0, 4.0], ([1, 1, 0, 2], [0, 0, 3, 1])), shape=(3, 4))
  expr = sc.const(source)
  source.data[:] = 99
  assert expr.type.pattern is not None
  assert expr.type.pattern.nnz == 3
  assert expr.support == frozenset({9})
  fn = _function("sparse_constant", [], [expr.to_dense(), sc.values(expr)])
  dense, stored = fn()
  expected = np.zeros((3, 4))
  expected[2, 1] = 4
  np.testing.assert_array_equal(dense, expected)
  np.testing.assert_array_equal(stored, [0, 0, 4])


def test_from_dense_and_sparsify_keep_declared_coordinates():
  pattern = CanonicalPattern((3, 4), (0, 1, 2), (3, 0, 2))
  x = sc.sym("x", (3, 4))
  expr = sc.from_dense(x, pattern)
  fn = _function("project_sparse", [x], [sc.values(expr)])
  data = np.arange(12.0).reshape(3, 4)
  np.testing.assert_array_equal(fn(data), data[pattern.rows, pattern.cols])
  compact = sc.sparsify(sc.const(data * (data % 3 == 0)))
  assert compact.type.pattern is not None
  assert compact.type.pattern.nnz == 3
  assert compact.type.storage_size == 3
  full = sc.sparsify(x)
  assert full.type.pattern is not None and full.type.pattern.nnz == x.size


def test_sparse_arithmetic_stored_slot_maps_match_scipy():
  a = sparse.csr_array(([3.0, 0.0, -4.0], ([0, 1, 2], [3, 0, 2])), shape=(3, 4))
  b = sparse.csc_array(([8.0, 5.0, 2.0], ([0, 1, 2], [0, 0, 1])), shape=(3, 4))
  x, y = sc.const(a), sc.const(b)
  fn = _function("sparse_ops", [], [(x + y).to_dense(), (x * y).to_dense(), x.sin().to_dense()])
  results = fn()
  for actual, expected in zip(results, [(a + b).toarray(), a.multiply(b).toarray(), np.sin(a.toarray())], strict=True):
    np.testing.assert_allclose(actual, expected)


def test_sparse_constants_return_scipy_and_keep_batched_shared_patterns():
  pattern = CanonicalPattern((3, 4), (0, 1, 2), (3, 0, 2))
  data = np.arange(6.0).reshape(2, 3)
  coords = (np.repeat([0, 1], 3), np.tile(pattern.rows, 2), np.tile(pattern.cols, 2))
  source = sparse.coo_array((data.reshape(-1), coords), shape=(2, *pattern.shape))
  constant = sc.const(source)
  assert constant.type.pattern is pattern
  result = _function("batched_sparse_constant", [], [constant])()
  assert isinstance(result, sparse.coo_array)
  np.testing.assert_array_equal(result.toarray(), source.toarray())
  dense = np.zeros((2, *pattern.shape))
  dense[0, 0, 3], dense[1, 2, 2] = 2.0, 4.0
  compact = sc.sparsify(sc.const(dense))
  assert compact.type.storage_size == 4
  np.testing.assert_array_equal(_function("batched_sparsify", [], [compact.to_dense()])(), dense)


def test_values_pattern_constants_preserve_the_callers_coordinate_order():
  pattern = sc.SparsityPattern((3, 4), (2, 0, 1), (2, 3, 0))
  original = np.array([3.0, 5.0, 0.0])
  constant = sc.const(original, pattern=pattern)
  original[:] = 99.0
  result = _function("coordinate_constant", [], [constant])()
  assert isinstance(result, sparse.csc_array)
  expected = np.zeros(pattern.shape)
  expected[pattern.rows, pattern.cols] = [3.0, 5.0, 0.0]
  np.testing.assert_array_equal(result.toarray(), expected)
  assert result.nnz == 3
