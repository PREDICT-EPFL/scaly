"""Exact nonsmooth conventions and public extremum builders against NumPy."""

from itertools import combinations

import numpy as np
import pytest

import scaly as sc
from scaly.ir.types import CanonicalPattern


@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("shape", [(), (4,), (2, 3, 4), (1, 3, 1), (0, 3)])
@pytest.mark.parametrize("lowering", ["scalar", "block"])
def test_builders_numpy(kind, shape, lowering):
  axes = [a for count in range(len(shape) + 1) for a in combinations(range(len(shape)), count) if all(shape[i] for i in a)]
  build = sc.reduce_max if kind == "max" else sc.reduce_min
  reference = np.max if kind == "max" else np.min
  cases = [(a, k) for a in axes for k in (False, True)]

  @sc.function(sc.arg("x", shape), outputs=sc.group(*(sc.group(sc.arg(f"y{i}a"), sc.arg(f"y{i}b")) for i in range(len(axes)))))
  def f(x):
    return tuple(tuple(build(x, tuple(i - len(shape) for i in a), keepdims=k).with_lowering(lowering) for k in (False, True)) for a in axes)

  data = np.random.default_rng(65).normal(size=shape)
  for nan in (False, True):
    if nan and data.size:
      data.flat[data.size // 2] = np.nan
    for actual, (axis, keepdims) in zip((v for pair in f(data) for v in pair), cases, strict=True):
      np.testing.assert_equal(actual, reference(data, axis=axis, keepdims=keepdims))


@pytest.mark.parametrize(
  "build", [lambda x: x.abs(), lambda x: x.floor(), lambda x: x.ceil(), lambda x: sc.maximum(x, 0), lambda x: sc.minimum(x, 0)]
)
def test_exact_partials(build):
  point = np.array([-1.0, 0.0, 1.0])
  expected = {"abs": [-1, 0, 1], "floor": [0, 0, 0], "ceil": [0, 0, 0], "maximum": [0, 0.5, 1], "minimum": [1, 0.5, 0]}

  @sc.function(sc.arg("x", 3), outputs=sc.group(sc.arg("t"), sc.arg("g")))
  def f(x):
    y = build(x)
    return sc.jvp(y, x, sc.const(np.ones(3))), sc.gradient(y.sum(), x)

  op = build(sc.sym("x", 3)).op.value
  for actual in f(point):
    np.testing.assert_equal(actual, expected[op])


@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("sparse", [False, True])
def test_ties_and_sparse_zeros(kind, sparse):
  pattern = CanonicalPattern((2, 3), (0, 0, 1), (0, 1, 2))
  shape = (pattern.nnz,) if sparse else pattern.shape
  build = sc.reduce_max if kind == "max" else sc.reduce_min

  @sc.function(sc.arg("x", shape), outputs=sc.group(sc.arg("y"), sc.arg("t"), sc.arg("g"), sc.arg("j")))
  def f(x):
    source = sc.pack(x, pattern) if sparse else x
    y = build(source, 1)
    return y, sc.jvp(y, x, sc.const(np.ones(shape))), sc.gradient(y.sum(), x), sc.sparse_jacobian(y, x).to_dense()

  point = np.zeros(shape)
  y, t, g, j = map(np.asarray, f(point))
  np.testing.assert_equal(y, [0, 0])
  np.testing.assert_allclose(g, np.full(shape, 1 / 3))
  np.testing.assert_allclose(t, [2 / 3, 1 / 3] if sparse else [1, 1])
  np.testing.assert_allclose(np.asarray(j).sum(axis=0).reshape(shape), g)


@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("shape", [(), (1,), (2, 3, 4), (0, 3)])
def test_arg_numpy(kind, shape):
  build = sc.argmax if kind == "max" else sc.argmin
  reference = np.argmax if kind == "max" else np.argmin
  axes = ([None] if np.prod(shape) else []) + [i for i, n in enumerate(shape) if n]
  cases = [(a, k) for a in axes for k in (False, True)]

  @sc.function(sc.arg("x", shape), outputs=sc.group(*(sc.arg(f"y{i}") for i in range(len(cases)))))
  def f(x):
    return tuple(sc.cast(build(x, axis=a, keepdims=k), "float64") for a, k in cases)

  data = np.zeros(shape)
  for nan in (False, True):
    if nan and data.size:
      data.flat[-1] = np.nan
    for actual, (axis, keepdims) in zip(f(data), cases, strict=True):
      np.testing.assert_equal(actual, reference(data, axis=axis, keepdims=keepdims))
  x = sc.sym("x", shape)
  for axis, _ in cases:
    assert not build(x, axis=axis).type.diff
    assert build(x, axis=axis).type.dtype == sc.dtypes.int64


@pytest.mark.parametrize(
  "build", [lambda x: sc.reduce_max(x, 1), lambda x: sc.reduce_min(x, 1), lambda x: sc.argmax(x, 1), lambda x: sc.argmin(x, 1)]
)
def test_empty_domain_rejected(build):
  with pytest.raises(ValueError, match="empty"):
    build(sc.sym("x", (3, 0)))


def test_norms_nan_and_zero():
  @sc.function(sc.arg("x", 3), outputs=sc.group(sc.arg("inf"), sc.arg("one")))
  def f(x):
    return sc.norm_inf(x), sc.norm_1(x)

  for data in (np.array([-3.0, 2.0, 0.0]), np.zeros(3), np.array([1.0, np.nan, 2.0])):
    a, b = f(data)
    np.testing.assert_equal(a, np.max(np.abs(data)))
    np.testing.assert_equal(b, np.sum(np.abs(data)))


@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("sparse", [False, True])
def test_unchosen_nan_partial(kind, sparse):
  build = sc.reduce_max if kind == "max" else sc.reduce_min
  sign = 1 if kind == "max" else -1

  @sc.function(sc.arg("x", 2), outputs=sc.group(sc.arg("t"), sc.arg("g")))
  def f(x):
    source = sc.pack(x, CanonicalPattern((2, 2), (0, 1), (0, 1))) if sparse else x
    y = build(sign * source.sqrt())
    return sc.jvp(y, x, sc.const(np.ones(2))), sc.gradient(y, x)

  t, g = f(np.array([0.0, 4.0]))
  np.testing.assert_equal(t, sign * 0.25)
  np.testing.assert_equal(g, [0, sign * 0.25])


@pytest.mark.parametrize("build,reference", [(sc.maximum, np.maximum), (sc.minimum, np.minimum)])
@pytest.mark.parametrize("lowering", ["scalar", "block"])
def test_elementwise_nan_in_either_operand(build, reference, lowering):
  @sc.function(sc.arg("x", (2, 64)), outputs=sc.arg("y"))
  def f(x):
    return build(x[0], x[1]).with_lowering(lowering)

  data = np.tile([[-1.0], [2.0]], (1, 64))
  data[0, ::3] = np.nan
  data[1, 1::3] = np.nan
  np.testing.assert_equal(f(data), reference(*data))


@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("lowering", ["scalar", "block"])
def test_sparse_public_builders_nan_axes_and_indices(kind, lowering):
  from scipy import sparse

  pattern = CanonicalPattern((3, 4), (0, 0, 1, 2), (0, 2, 1, 3))
  build, arg, ref, aref = (sc.reduce_max, sc.argmax, np.max, np.argmax) if kind == "max" else (sc.reduce_min, sc.argmin, np.min, np.argmin)
  axes = [None, 0, -1, ()]

  @sc.function(sc.arg("x", pattern.nnz), outputs=sc.group(*(sc.group(sc.arg(f"v{i}"), sc.arg(f"a{i}")) for i in range(4))))
  def f(x):
    source = sc.pack(x, pattern).with_lowering(lowering)
    return tuple(
      (build(source, axis=axis, keepdims=True), sc.cast(arg(source, axis=axis if axis != () else 1, keepdims=True), "float64")) for axis in axes
    )

  for data in (np.array([-1.0, 2.0, 0.0, -3.0]), np.array([np.nan, 2.0, 0.0, np.nan])):
    dense = sparse.coo_array((data, (pattern.rows, pattern.cols)), shape=pattern.shape).toarray()
    for (value, index), axis in zip(f(data), axes, strict=True):
      np.testing.assert_equal(
        sparse.coo_array(value).toarray() if sparse.issparse(value) else np.asarray(value), ref(dense, axis=axis, keepdims=True)
      )
      np.testing.assert_equal(index, aref(dense, axis=axis if axis != () else 1, keepdims=True))


def test_sparse_leading_axis_tangent_keeps_pattern():
  pattern = CanonicalPattern((2, 3), (0, 1), (0, 2))

  @sc.function(sc.arg("x", (2, 2)), outputs=sc.group(sc.arg("t"), sc.arg("g")))
  def f(x):
    y = sc.reduce_max(sc.pack(x, pattern, shape=(2, 2, 3)), 0)
    return sc.to_dense(sc.jvp(y, x, sc.const(np.ones(x.shape)))), sc.gradient(y.sum(), x)

  t, g = map(np.asarray, f(np.array([[1.0, 2.0], [3.0, 2.0]])))
  np.testing.assert_equal(t, [[1, 0, 0], [0, 0, 1]])
  np.testing.assert_equal(g, [[0, 0.5], [1, 0.5]])


@pytest.mark.parametrize("name", ["abs", "floor", "ceil"])
def test_zero_partial_masks_an_infinite_inner_partial(name):
  @sc.function(sc.arg("x", 2), outputs=sc.group(sc.arg("t"), sc.arg("g")))
  def f(x):
    y = getattr(x.sqrt(), name)().sum()
    return sc.jvp(y, x, sc.const(np.ones(2))), sc.gradient(y, x)

  t, g = map(np.asarray, f(np.array([0.0, 4.0])))
  np.testing.assert_equal(t, 0.25 if name == "abs" else 0)
  np.testing.assert_equal(g, [0, 0.25] if name == "abs" else [0, 0])


def test_nonzero_ties_in_composed_extrema():
  @sc.function(sc.arg("x", 5), outputs=sc.group(sc.arg("t"), sc.arg("g"), sc.arg("h")))
  def f(x):
    y = sc.maximum(x, 1).sum() + sc.reduce_max(x) + sc.reduce_min(x)
    return sc.jvp_many(y, x, sc.const(np.eye(5))), sc.gradient(y, x), sc.hessian(y, x)

  t, g, h = map(np.asarray, f(np.array([1.0, 3.0, 3.0, -2.0, -2.0])))
  np.testing.assert_equal(t, [0.5, 1.5, 1.5, 0.5, 0.5])
  np.testing.assert_equal(g, t)
  np.testing.assert_equal(h, np.zeros((5, 5)))


def test_tied_norm_inf_through_a_transposed_view():
  @sc.function(sc.arg("x", (2, 3)), outputs=sc.group(sc.arg("g"), sc.arg("h")))
  def f(x):
    y = sc.norm_inf(x.T[:, ::-1])
    return sc.gradient(y, x), sc.hessian(y, x)

  g, h = map(np.asarray, f(np.array([[-3.0, 1.0, 3.0], [2.0, -3.0, 0.0]])))
  np.testing.assert_allclose(g, [[-1 / 3, 0, 1 / 3], [0, -1 / 3, 0]])
  np.testing.assert_equal(h, np.zeros((6, 6)))
