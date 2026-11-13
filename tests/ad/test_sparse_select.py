"""Sparse SELECT values, stored-coordinate derivatives and masked singular partials."""

from scaly.ir.expr import _scatter_zero

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.ad.derivatives import finite_difference
from scaly.ir.program import ProgramOp, walk_program
from scaly.ir.types import CanonicalPattern
from scaly.passes.lowering import lower_function


@pytest.mark.parametrize("branches", ["equal", "different", "dense"])
@pytest.mark.parametrize("constant", [False, True])
def test_sparse_select_patterns_support_and_derivative_sparsity(branches, constant):
  a = CanonicalPattern((3, 5), (0, 1, 2), (4, 0, 2))
  b = a if branches == "equal" else CanonicalPattern(a.shape, (0, 1, 2), (1, 0, 4))
  width = a.nnz + (15 if branches == "dense" else b.nnz)
  condition = np.array([[True, False, True, False, False]])

  def expression(v, c):
    x = sc.pack(v[: a.nnz], a).sin()
    y = v[a.nnz :].reshape(a.shape).tanh() if branches == "dense" else sc.pack(v[a.nnz :], b).tanh()
    result = sc.where(sc.const(condition) if constant else c > 0, x, y)
    expected = (
      a.union(CanonicalPattern(a.shape, tuple(np.repeat(np.arange(3), 5)), tuple(np.tile(np.arange(5), 3)))) if branches == "dense" else a.union(b)
    )
    assert result.type.pattern is expected
    if constant:
      possible = np.ones(a.shape, dtype=bool) if branches == "dense" else np.zeros(a.shape, dtype=bool)
      possible[b.rows, b.cols] = True
      left = np.zeros(a.shape, dtype=bool)
      left[a.rows, a.cols] = True
      assert result.support == frozenset(np.flatnonzero(np.where(condition, left, possible)))
    return result

  @sc.function(sc.arg("v", width), sc.arg("c", condition.shape), name=f"sparse_select_{branches}_{constant}")
  def fn(v, c):
    return sc.values(expression(v, c)).block()

  @sc.function(sc.arg("v", width), sc.arg("c", condition.shape), name=f"sparse_select_loss_{branches}_{constant}")
  def loss(v, c):
    return expression(v, c).sum().block()

  pattern = expression(sc.sym("select_v", width), sc.sym("select_c", condition.shape)).type.pattern
  assert pattern is not None
  point = np.linspace(0.2, 1.1, width)

  def numpy(v):
    x = np.zeros(a.shape)
    x[a.rows, a.cols] = v[: a.nnz]
    if branches == "dense":
      y = v[a.nnz :].reshape(a.shape)
    else:
      y = np.zeros(b.shape)
      y[b.rows, b.cols] = v[a.nnz :]
    return np.where(condition, np.sin(x), np.tanh(y))[pattern.rows, pattern.cols]

  np.testing.assert_allclose(fn(point, condition), numpy(point), atol=1e-12)
  reference = finite_difference(numpy, point)
  np.testing.assert_allclose(sc.jacobian(fn, wrt="v")(point, condition), reference, atol=1e-7, rtol=1e-7)
  np.testing.assert_allclose(sc.gradient(loss, wrt="v")(point, condition), reference.sum(axis=0), atol=1e-7, rtol=1e-7)
  x = sc.sym("select_pattern_v", width)
  c = sc.sym("select_pattern_c", condition.shape)
  relation = sc.jacobian_sparsity(sc.values(expression(x, c)), x)
  actual = set(zip(relation.rows, relation.cols, strict=True))
  expected = set(zip(*np.nonzero(reference)))
  assert actual == expected if constant else expected <= actual


@pytest.mark.parametrize("batch,condition_shape", [((), ()), ((), (1, 300)), ((2,), (2, 1, 300)), ((0,), (0, 1, 300))])
def test_sparse_select_union_broadcast_and_derivatives(batch, condition_shape):
  a = CanonicalPattern((200, 300), (0, 1, 4, 9, 10, 32, 100, 199), (2, 0, 299, 17, 42, 201, 2, 299))
  b = CanonicalPattern(a.shape, (0, 2, 4, 10, 51, 100, 199), (2, 3, 12, 42, 101, 2, 0))
  output = a.union(b)
  width = a.nnz + b.nnz
  v_leaf = sc.arg("v", (*batch, width))
  c_leaf = sc.arg("c", condition_shape)

  def expression(v, c):
    result = sc.where(c > 0, sc.pack(v[..., : a.nnz], a).sin(), sc.pack(v[..., a.nnz :], b).tanh())
    assert result.type.pattern is output
    assert result.type.storage_shape == (*batch, output.nnz)
    return result.block()

  fn = sc.function(v_leaf, c_leaf, name="sparse_select_union")(lambda v, c: sc.values(expression(v, c)))
  loss = sc.function(v_leaf, c_leaf, name="sparse_select_union_loss")(lambda v, c: expression(v, c).sum())
  random = np.random.default_rng(46)
  v = random.normal(size=(*batch, width))
  condition = np.asarray(random.random(size=condition_shape) > 0.4)

  def numpy(v):
    x, y = np.zeros((*batch, *a.shape)), np.zeros((*batch, *b.shape))
    x[..., a.rows, a.cols], y[..., b.rows, b.cols] = v[..., : a.nnz], v[..., a.nnz :]
    return np.where(condition, np.sin(x), np.tanh(y))[..., output.rows, output.cols]

  np.testing.assert_allclose(np.asarray(fn(v, condition)), numpy(v), atol=1e-12)
  for node in walk_program(lower_function(fn)):
    if node.op == ProgramOp.BUFFER and node.attrs["address_space"] == "private":
      assert np.prod(node.attrs["shape"]) <= 4 * max(1, np.prod(batch)) * (width + output.nnz) + np.prod(condition_shape)
  if v.size:
    reference = finite_difference(numpy, v)
    np.testing.assert_allclose(sc.jacobian(fn, wrt="v")(v, condition), reference, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(sc.gradient(loss, wrt="v")(v, condition), reference.sum(axis=0).reshape(v.shape), atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize(
  "movement",
  [
    "identity",
    "transpose",
    "reshape",
    "slice",
    "vector",
    "gather",
    "scatter",
    "stack",
    "concat",
    "sum",
    "sum_rows",
    "sum_rows_keepdims",
    "matvec",
    "rmatvec",
    "matmat",
    "rmatmat",
  ],
)
def test_sparse_select_mask_reaches_singular_partial_through_movement(movement):
  pattern = CanonicalPattern((4, 5), (0, 0, 1, 2, 3, 3), (0, 4, 1, 3, 0, 2))
  v_leaf = sc.arg("v", pattern.nnz)
  c_leaf = sc.arg("c", ())

  def body(v, c):
    x = sc.pack(v, pattern).sqrt()
    branch = {
      "identity": lambda: x,
      "transpose": lambda: x.T,
      "reshape": lambda: x.reshape((2, 10)),
      "slice": lambda: x[::-1, ::2],
      "vector": lambda: x[0],
      "gather": lambda: sc.gather(x, [[19, 0, 4], [10, 16, 9]]),
      "scatter": lambda: _scatter_zero(x, np.arange(20) % 9, (3, 3)),
      "stack": lambda: sc.stack((x, 2 * x)),
      "concat": lambda: sc.concat((x, 2 * x), axis=1),
      "sum": lambda: x.sum(),
      "sum_rows": lambda: x.sum(axis=0),
      "sum_rows_keepdims": lambda: x.sum(axis=0, keepdims=True),
      "matvec": lambda: x @ sc.const(np.ones(5)),
      "rmatvec": lambda: sc.const(np.ones(4)) @ x,
      "matmat": lambda: x @ sc.const(np.ones((5, 2))),
      "rmatmat": lambda: sc.const(np.ones((2, 4))) @ x,
    }[movement]()
    ones = sc.const(np.ones(branch.type.storage_shape))
    other = sc.pack(ones, branch.type.pattern, shape=branch.shape) if branch.type.pattern is not None else ones
    return sc.where(c > 0, branch, other).sum().block()

  loss = sc.function(v_leaf, c_leaf, name=f"sparse_select_mask_{movement}")(body)
  grad = sc.gradient(loss, wrt="v")
  zero = np.zeros(pattern.nnz)
  np.testing.assert_array_equal(grad(zero, np.asarray(False)), zero)
  positive = np.linspace(0.2, 1.4, pattern.nnz)
  for condition in (False, True):
    c = np.asarray(condition)
    reference = finite_difference(lambda v: loss(v, c), positive).reshape(positive.shape)
    np.testing.assert_allclose(np.asarray(grad(positive, c)), reference, atol=1e-7, rtol=1e-7)


def test_sparse_select_stored_bool_condition_and_hessian_without_logical_matrix_buffer():
  pattern = CanonicalPattern((1000, 1000), (0, 1, 7, 30, 103, 999), (999, 0, 40, 3, 2, 998))
  condition_pattern = CanonicalPattern(pattern.shape, (0, 30, 999), (999, 3, 998))
  c_leaf = sc.arg("c", condition_pattern.nnz)
  v_leaf = sc.arg("v", pattern.nnz)

  def body(v, c):
    x = sc.pack(v, pattern)
    c = sc.pack(sc.cast(c, "bool"), condition_pattern)
    return sc.where(c, x.sqrt(), sc.pack(sc.const(np.ones(pattern.nnz)), pattern)).sum().block()

  loss = sc.function(v_leaf, c_leaf, name="sparse_select_stored_bool")(body)
  grad, hess = sc.gradient(loss, wrt="v"), sc.sparse_hessian(loss, wrt="v")
  v = np.array([0.25, 0, 0, 0, 0, 1.0])
  condition = np.array([True, False, True])
  expected_grad = np.array([1.0, 0, 0, 0, 0, 0.5])
  np.testing.assert_array_equal(grad(v, condition), expected_grad)
  hp = hess.sparsity()
  assert hp is not None
  actual_hess = sparse.coo_array(hess(v, condition)).toarray()
  np.testing.assert_array_equal(actual_hess, np.diag([-2.0, 0, 0, 0, 0, -0.25]))
  difference = finite_difference(lambda z: grad(z, condition), v)
  np.testing.assert_allclose(actual_hess, difference, atol=1e-7, rtol=1e-7)
  for function in (loss, grad, hess):
    observed = {}
    lower_function(function, observe=observed.__setitem__)
    for program in (observed["lowered"], observed["pass:pack_workspace"]):
      for node in walk_program(program):
        if node.op == ProgramOp.BUFFER and node.attrs["address_space"] == "private":
          assert np.prod(node.attrs["shape"]) <= 20 * pattern.nnz**2


def test_sparse_select_leaf_derivatives_keep_stored_layout():
  pattern = CanonicalPattern((1000, 1000), (0, 0, 4, 99, 999), (9, 997, 3, 71, 998))
  leaf = sc.arg("a", sc.TensorType(pattern.shape, pattern=pattern), layout="csr")

  @sc.function(leaf, sc.arg("c", ()), name="sparse_select_leaf_csr")
  def loss(a, c):
    return sc.where(c > 0, a.sin(), a * a).sum().block()

  values = np.linspace(0.1, 0.9, pattern.nnz)
  a = sparse.coo_array((values, (pattern.rows, pattern.cols)), shape=pattern.shape)
  grad = sc.gradient(loss, wrt="a")
  for condition in (False, True):
    actual = sparse.coo_array(grad(a, np.asarray(condition))).toarray()
    expected = np.cos(values) if condition else 2 * values
    np.testing.assert_allclose(actual[pattern.rows, pattern.cols], expected, atol=1e-12)
    direction = sparse.coo_array((np.ones(pattern.nnz), (pattern.rows, pattern.cols)), shape=pattern.shape)
    eps = 1e-6
    difference = (loss(a + eps * direction, np.asarray(condition)) - loss(a - eps * direction, np.asarray(condition))) / (2 * eps)
    np.testing.assert_allclose(actual.sum(), difference, atol=1e-8)
  for node in walk_program(lower_function(grad)):
    if node.op == ProgramOp.BUFFER and node.attrs["address_space"] == "private":
      assert np.prod(node.attrs["shape"]) <= 20 * pattern.nnz**2


def test_constant_sparse_condition_narrows_support_and_derivative_rows():
  pattern = CanonicalPattern((3, 5), (0, 1, 2), (4, 0, 2))
  condition = sc.const(sparse.coo_array(([True, False], ([0, 2], [4, 2])), shape=pattern.shape))

  @sc.function(sc.arg("v", pattern.nnz), name="sparse_select_constant_bool")
  def fn(v):
    result = sc.where(condition, sc.pack(v, pattern), sc.pack(sc.const(np.zeros(pattern.nnz)), pattern))
    assert result.support == frozenset({4})
    return sc.values(result).block()

  point = np.array([0.3, 0.7, 1.2])
  np.testing.assert_array_equal(fn(point), [0.3, 0, 0])
  np.testing.assert_array_equal(sc.jacobian(fn)(point), np.diag([1.0, 0, 0]))


@pytest.mark.parametrize("path", ["nested", "stack", "concat"])
def test_sparse_select_masks_survive_nesting_and_pattern_normalization(path):
  p = CanonicalPattern((2, 3), (0, 1), (0, 2))
  q = CanonicalPattern(p.shape, (0, 1), (1, 0))

  @sc.function(sc.arg("v", p.nnz), sc.arg("c", 2), name=f"sparse_select_normalized_{path}")
  def loss(v, c):
    x = sc.pack(v, p).sqrt()
    other = sc.pack(sc.const(np.ones(p.nnz)), p if path == "nested" else q)
    if path == "nested":
      branch = sc.where(c[1] > 0, x, other)
    elif path == "stack":
      branch = sc.stack((x, other))
    else:
      branch = sc.concat((x.reshape((1, *p.shape)), other.reshape((1, *q.shape))))
    assert branch.type.pattern is not None
    ones = sc.pack(sc.const(np.ones(branch.type.storage_shape)), branch.type.pattern, shape=branch.shape)
    return sc.where(c[0] > 0, branch, ones).sum().block()

  grad = sc.gradient(loss, wrt="v")
  hess = sc.hessian(loss, wrt="v")
  conditions = ([0.0, 0.0], [0.0, 1.0], [1.0, 0.0]) if path == "nested" else ([0.0, 0.0],)
  for condition in conditions:
    c = np.asarray(condition)
    zero = np.zeros(p.nnz)
    np.testing.assert_array_equal(grad(zero, c), zero)
    np.testing.assert_array_equal(hess(zero, c), np.zeros((p.nnz, p.nnz)))
    np.testing.assert_array_equal(finite_difference(lambda v: loss(v, c), zero), zero.reshape((1, -1)))
  point = np.array([0.25, 1.0])
  for condition in ([1.0, 0.0], [1.0, 1.0]):
    c = np.asarray(condition)
    reference = finite_difference(lambda v: loss(v, c), point).reshape(point.shape)
    np.testing.assert_allclose(np.asarray(grad(point, c)), reference, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("shape", [(3, 4), (1, 4)])
def test_sparse_condition_with_dense_branches_matches_numpy(shape):
  pattern = CanonicalPattern(shape, (0, 1, 2) if shape[0] == 3 else (0, 0, 0), (3, 1, 2) if shape[0] == 3 else (1, 2, 3))

  def expression(v, c):
    condition = sc.pack(sc.cast(c, "bool"), pattern)
    result = sc.where(condition, v + 100.0, -v)
    assert result.type.pattern is None
    return result.block()

  leaves = (sc.arg("v", (3, 4)), sc.arg("c", pattern.nnz))
  fn = sc.function(*leaves, name=f"sparse_condition_dense_branches_{shape[0]}")(expression)
  loss = sc.function(*leaves, name=f"sparse_condition_dense_loss_{shape[0]}")(lambda v, c: expression(v, c).sum())
  c = np.array([True, False, True])
  condition = sparse.coo_array((c, (pattern.rows, pattern.cols)), shape=shape).toarray()
  point = np.arange(12, dtype=float).reshape((3, 4))
  numpy = lambda v: np.where(condition, v + 100.0, -v)
  np.testing.assert_array_equal(fn(point, c), numpy(point))
  reference = finite_difference(numpy, point)
  np.testing.assert_allclose(sc.jacobian(fn, wrt="v")(point, c), reference, atol=1e-7, rtol=1e-7)
  np.testing.assert_allclose(sc.gradient(loss, wrt="v")(point, c), reference.sum(axis=0).reshape(point.shape), atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("path", ["pack", "matvec", "rmatvec", "matmat", "rmatmat"])
def test_sparse_select_masks_singular_dense_operands_and_pack_input(path):
  pattern = CanonicalPattern((4, 5), (0, 0, 1, 2, 3, 3), (0, 4, 1, 3, 0, 2))
  size = {"pack": pattern.nnz, "matvec": 5, "rmatvec": 4, "matmat": 10, "rmatmat": 8}[path]

  @sc.function(sc.arg("v", size), sc.arg("c", ()), name=f"sparse_select_dense_partial_{path}")
  def loss(v, c):
    root = v.sqrt()
    matrix = sc.pack(sc.const(np.linspace(0.5, 1.5, pattern.nnz)), pattern)
    branch = {
      "pack": lambda: sc.pack(root, pattern),
      "matvec": lambda: matrix @ root,
      "rmatvec": lambda: root @ matrix,
      "matmat": lambda: matrix @ root.reshape((5, 2)),
      "rmatmat": lambda: root.reshape((2, 4)) @ matrix,
    }[path]()
    return sc.where(c > 0, branch, sc.const(np.ones(branch.shape))).sum().block()

  grad = sc.gradient(loss, wrt="v")
  zero = np.zeros(size)
  np.testing.assert_array_equal(grad(zero, np.asarray(False)), zero)
  point = np.linspace(0.2, 1.3, size)
  for condition in (False, True):
    c = np.asarray(condition)
    reference = finite_difference(lambda v: loss(v, c), point).reshape(point.shape)
    np.testing.assert_allclose(np.asarray(grad(point, c)), reference, atol=1e-7, rtol=1e-7)
