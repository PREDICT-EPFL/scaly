"""Axis reductions and diagonal builders against NumPy, including empty domains."""

from itertools import combinations
import warnings

import numpy as np
import pytest

import scaly as sc
from scaly.ir.expr import _reduce
from scaly.ir.program import ProgramOp, walk_program
from scaly.ir.types import CanonicalPattern
from scaly.passes.lowering import lower_function


def _subsets(rank):
  return [axes for count in range(rank + 1) for axes in combinations(range(rank), count)]


@pytest.mark.parametrize("shape", [(), (0,), (4,), (2, 3, 4), (2, 0, 3), (1, 3, 1)])
@pytest.mark.parametrize("lowering", ["scalar", "block"])
def test_sum_mean_every_axis_subset(shape, lowering):
  axes_cases = _subsets(len(shape))
  cases = [(keepdims, mean) for keepdims in (False, True) for mean in (False, True)]

  @sc.function(sc.arg("x", shape), outputs=sc.group(*(sc.group(*(sc.arg(f"out{i}_{j}") for j in range(4))) for i in range(len(axes_cases)))))
  def f(x):
    return tuple(
      tuple((x.mean if mean else x.sum)(tuple(axis - len(shape) for axis in axes), keepdims).with_lowering(lowering) for keepdims, mean in cases)
      for axes in axes_cases
    )

  data = np.arange(np.prod(shape), dtype=float).reshape(shape) / 7 - 2
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    for results, axes in zip(f(data), axes_cases, strict=True):
      for actual, (keepdims, mean) in zip(results, cases, strict=True):
        expected = (np.mean if mean else np.sum)(data, axis=axes, keepdims=keepdims)
        assert np.asarray(actual).shape == expected.shape
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("shape", [(4,), (0,), (2, 4), (4, 2), (0, 3), (3, 0)])
@pytest.mark.parametrize("offset", [-5, -1, 0, 1, 5])
def test_diag_numpy(shape, offset):
  @sc.function(sc.arg("x", shape), outputs=sc.arg("y"))
  def f(x):
    return sc.diag(x, k=offset)

  data = np.arange(np.prod(shape), dtype=float).reshape(shape)
  np.testing.assert_array_equal(f(data), np.diag(data, k=offset))


@pytest.mark.parametrize("shape", [(), (2, 3, 4), (2, 0, 3), (1, 3, 1)])
@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("lowering", ["scalar", "block"])
def test_internal_extrema_every_axis_subset(shape, kind, lowering):
  axes = _subsets(len(shape))

  @sc.function(sc.arg("x", shape), outputs=sc.group(*(sc.arg(f"out{i}") for i in range(len(axes)))))
  def f(x):
    return tuple(_reduce(x, kind, axis).with_lowering(lowering) for axis in axes)

  data = np.random.default_rng(66).normal(size=shape)
  for actual, axis in zip(f(data), axes, strict=True):
    expected = (np.max if kind == "max" else np.min)(data, axis=axis, initial=-np.inf if kind == "max" else np.inf)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape", [(4, 5), (2, 4, 5), (0, 4, 5), (4, 0)])
@pytest.mark.parametrize("kind", ["add", "max", "min", "mean"])
@pytest.mark.parametrize("lowering", ["scalar", "block"])
def test_sparse_reductions_every_axis_subset(shape, kind, lowering):
  from scipy import sparse

  rows, cols = np.nonzero(np.random.default_rng(66).random(shape[-2:]) < 0.4)
  pattern = CanonicalPattern(shape[-2:], tuple(map(int, rows)), tuple(map(int, cols)))
  axes = _subsets(len(shape))

  @sc.function(sc.arg("x", sc.TensorType(shape, pattern=pattern), layout="csc"), outputs=sc.group(*(sc.arg(f"out{i}") for i in range(len(axes)))))
  def f(x):
    return tuple((x.mean(axis) if kind == "mean" else _reduce(x, kind, axis)).with_lowering(lowering) for axis in axes)

  data = np.random.default_rng(165).normal(size=(*shape[:-2], pattern.nnz))
  dense = np.zeros(shape)
  dense[..., pattern.rows, pattern.cols] = data
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    for actual, axis in zip(f(data[..., np.lexsort((pattern.rows, pattern.cols))]), axes, strict=True):
      if kind in {"add", "mean"}:
        expected = (np.sum if kind == "add" else np.mean)(dense, axis=axis)
      else:
        expected = (np.max if kind == "max" else np.min)(dense, axis=axis, initial=-np.inf if kind == "max" else np.inf)
      np.testing.assert_allclose(sparse.coo_array(actual).toarray() if sparse.issparse(actual) else np.asarray(actual), expected, atol=1e-12)


@pytest.mark.parametrize("kind", ["add", "max", "min"])
def test_sparse_reductions_keep_loaded_bounds_and_linear_tables(kind):
  from scipy import sparse

  n = 41
  rows = (0, 0, 1, 3, 3, 7, 20, 40)
  cols = (0, 8, 3, 1, 4, 9, 2, 40)
  pattern = CanonicalPattern((n, n), rows, cols)

  @sc.function(sc.arg("x", sc.TensorType(pattern.shape, pattern=pattern), layout="csc"), outputs=sc.arg("y"))
  def f(x):
    return _reduce(x, kind, 1).block()

  observed = {}
  lower_function(f, observe=observed.__setitem__)
  nodes = list(walk_program(observed["lowered"]))
  assert any(node.op == ProgramOp.RANGE and any(bound.op == ProgramOp.LOAD for bound in node.args[:2]) for node in nodes)
  tables = [node for node in nodes if node.op == ProgramOp.BUFFER and "values" in node.attrs]
  assert sum(np.prod(node.attrs["shape"]) for node in tables) <= 5 * (pattern.nnz + n)
  for sign in (-1, 1):
    values = sign * np.arange(1.0, pattern.nnz + 1)
    source = sparse.coo_array((values, (pattern.rows, pattern.cols)), shape=pattern.shape)
    dense = source.toarray()
    expected = np.sum(dense, axis=1) if kind == "add" else (np.max if kind == "max" else np.min)(dense, axis=1)
    np.testing.assert_array_equal(f(source), expected)


@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("shape", [(3, 4), (3, 0)])
def test_sparse_extrema_without_stored_entries(kind, shape):
  pattern = CanonicalPattern(shape, (), ())

  @sc.function(sc.arg("x", sc.TensorType(shape, pattern=pattern)), outputs=sc.group(sc.arg("full"), sc.arg("rows")))
  def f(x):
    return _reduce(x, kind), _reduce(x, kind, 1)

  full, rows = f(np.empty(0))
  expected = 0 if shape[1] else -np.inf if kind == "max" else np.inf
  np.testing.assert_array_equal(full, expected)
  np.testing.assert_array_equal(rows, np.full(3, expected))


@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("lowering", ["scalar", "block"])
def test_sparse_extrema_full_partial_and_absent_rows(kind, lowering):
  pattern = CanonicalPattern((4, 3), (0, 0, 0, 1, 1, 3), (0, 1, 2, 0, 1, 2))

  @sc.function(sc.arg("x", sc.TensorType(pattern.shape, pattern=pattern)), outputs=sc.arg("y"))
  def f(x):
    return _reduce(x, kind, 1).with_lowering(lowering)

  sign = -1 if kind == "max" else 1
  data = sign * np.array([1.0, 2.0, 3.0, 4.0, 5.0, 0.0])
  np.testing.assert_array_equal(f(data), [sign, 0, 0, 0])
