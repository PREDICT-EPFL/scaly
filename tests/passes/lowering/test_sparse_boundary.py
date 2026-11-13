"""Stored allocation and access maps at sparse function boundaries."""

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.ir.program import ProgramOp, walk_program
from scaly.ir.types import CanonicalPattern
from scaly.passes.lowering.ctx import lower_function


@pytest.mark.parametrize("lowering", ["scalar", "block"])
def test_sparse_arithmetic_with_different_patterns_batches_and_broadcasts(lowering):
  p = CanonicalPattern((3, 4), (0, 1, 2), (3, 0, 2))
  q = CanonicalPattern((3, 4), (0, 1, 2), (0, 0, 1))
  xleaf = sc.arg("x", sc.TensorType((2, 3, 4), pattern=p), layout="csc")
  yleaf = sc.arg("y", sc.TensorType(q.shape, pattern=q), layout="csr")

  @sc.function(xleaf, yleaf, sc.arg("scale", (2, 1, 1)))
  def f(x, y, scale):
    result = (x.sin() + y) * x * scale
    return result.with_lowering(lowering), result.sum()

  rng = np.random.default_rng(42)
  for _ in range(5):
    xv, yv = rng.normal(size=(2, 3)), rng.normal(size=3)
    scale = rng.normal(size=(2, 1, 1))
    x = sparse.coo_array((xv.reshape(-1), (np.repeat([0, 1], 3), np.tile(p.rows, 2), np.tile(p.cols, 2))), shape=(2, 3, 4))
    y = sparse.coo_array((yv, (q.rows, q.cols)), shape=q.shape)
    result, total = f(x, y, scale)
    expected = (np.sin(x.toarray()) + y.toarray()) * x.toarray() * scale
    np.testing.assert_allclose(result.toarray(), expected)
    np.testing.assert_allclose(total, expected.sum())


def test_large_sparse_calls_and_maps_allocate_only_stored_values():
  pattern = CanonicalPattern((100_000, 100_000), (0, 20, 99_999), (99_999, 30, 0))
  leaf = sc.arg("x", sc.TensorType(pattern.shape, pattern=pattern), layout="csc")
  stage = sc.function(leaf, name="large_sparse_stage")(lambda x: x.sin().block())
  nested = sc.function(leaf, name="large_sparse_nested")(lambda x: stage(x) * 2.0)
  mapped = sc.vmap(nested, 2)
  program = lower_function(mapped)
  buffers = [node for node in walk_program(program) if node.op == ProgramOp.BUFFER]
  assert buffers and all(np.prod(buf.attrs["shape"]) <= 6 for buf in buffers)
  for node in walk_program(program):
    if node.op == ProgramOp.RANGE:
      assert node.args[1].attrs["value"] <= 6
  data = np.arange(6.0, dtype=float).reshape(2, 3)
  coords = (np.repeat([0, 1], 3), np.tile(pattern.rows, 2), np.tile(pattern.cols, 2))
  result = mapped(sparse.coo_array((data.reshape(-1), coords), shape=(2, *pattern.shape)))
  np.testing.assert_allclose(result.data, (2 * np.sin(data[:, [2, 1, 0]])).reshape(-1))
