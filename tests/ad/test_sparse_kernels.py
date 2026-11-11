"""Stored-coordinate derivatives of sparse kernels against SciPy and finite differences."""

from scaly.ir.expr import _scatter_zero

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.ad.derivatives import finite_difference
from scaly.ir.types import CanonicalPattern
from scaly.ir.program import ProgramOp, walk_program
from scaly.passes.lowering import lower_function
from scaly.ad.forward import jvp_many
from scaly.ad.reverse import vjp
from scaly.ad.rules import ELEMENTWISE
from scaly.ir.expr import ExprOp
from scaly.codegen import render_c_module
from scaly.function.model import as_concrete


def _pattern(array):
  coo = sparse.coo_array(array)
  return CanonicalPattern(coo.shape, tuple(map(int, coo.row)), tuple(map(int, coo.col)))


@pytest.mark.parametrize("n,symmetric", [(100, True), (400, True), (100, False)])
def test_banded_sparse_leaf_gradient_and_hessian_have_no_square_buffer(n, symmetric):
  q = (
    sparse.diags_array([-np.ones(n - 1), 4 * np.ones(n), -np.ones(n - 1)], offsets=[-1, 0, 1], format="csc")
    if symmetric
    else sparse.diags_array([-np.ones(n - 1)], offsets=[1], shape=(n, n), format="csc")
  )
  pattern = _pattern(q)

  @sc.function(sc.arg("q", sc.TensorType(q.shape, pattern=pattern), layout="csc"), sc.arg("x", n), name=f"banded_sparse_{n}")
  def loss(q, x):
    return x @ (q @ x)

  grad = sc.gradient(loss, wrt="x")
  hess = sc.sparse_hessian(loss, wrt="x")
  for fn in (grad, hess):
    seen = {}
    lower_function(fn, observe=seen.__setitem__)
    for program in (seen["lowered"], seen["pass:pack_workspace"]):
      sizes = [np.prod(b.attrs["shape"]) for b in walk_program(program) if b.op == ProgramOp.BUFFER and b.attrs["address_space"] == "private"]
      assert max(sizes, default=0) <= 20 * n
  x = np.random.default_rng(46).normal(size=n)
  np.testing.assert_allclose(grad(q, x), (q + q.T) @ x, atol=1e-12)
  hp = hess.sparsity()
  assert hp is not None
  actual = hess(q, x)
  np.testing.assert_allclose(actual.toarray(), (q + q.T).toarray(), atol=1e-12)
  for direction in np.random.default_rng(47).normal(size=(3, n)):
    eps = 1e-5
    difference = (grad(q, x + eps * direction) - grad(q, x - eps * direction)) / (2 * eps)
    np.testing.assert_allclose(actual @ direction, difference, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("vector", [True, False])
def test_mixed_product_forward_and_sampled_reverse_match_finite_differences(left, vector):
  pattern = CanonicalPattern((3, 4), (0, 1, 2, 2), (3, 0, 1, 3))
  shape = ((4,) if left else (3,)) if vector else ((4, 2) if left else (2, 3))

  @sc.function(sc.arg("v", 4), sc.arg("b", shape), name=f"sparse_derivatives_{left}_{vector}")
  def fn(v, b):
    a = sc.pack(v, pattern)
    product = a @ b if left else b @ a
    assert (product.type.pattern is None) == vector
    return sc.values(product)

  v = np.array([0.2, -0.7, 0.9, 1.3])
  b = np.random.default_rng(46).normal(size=shape)
  for name, value in (("v", v), ("b", b)):
    derivative = sc.jacobian(fn, wrt=name)
    reference = finite_difference(lambda w: fn(w, b) if name == "v" else fn(v, w), value)
    np.testing.assert_allclose(derivative(v, b), reference, rtol=1e-7, atol=1e-7)

  @sc.function(sc.arg("v", 4), sc.arg("b", shape), name=f"sparse_loss_{left}_{vector}")
  def loss(v, b):
    return fn(v, b).sum()

  for name, value in (("v", v), ("b", b)):
    derivative = sc.gradient(loss, wrt=name)
    reference = finite_difference(lambda w: loss(w, b) if name == "v" else loss(v, w), value).reshape(value.shape)
    np.testing.assert_allclose(derivative(v, b), reference, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("operation", ["transpose", "slice", "reshape", "concat", "stack", "gather", "scatter", "sum", "elementwise"])
def test_sparse_movement_derivatives_match_finite_differences(operation):
  pattern = CanonicalPattern((3, 4), (0, 1, 2, 2), (3, 0, 1, 3))

  @sc.function(sc.arg("v", 4), name=f"sparse_movement_ad_{operation}")
  def fn(v):
    x = sc.pack(v, pattern)
    result = {
      "transpose": lambda: x.T,
      "slice": lambda: x[::-1, ::2],
      "reshape": lambda: x.reshape((2, 6)),
      "concat": lambda: sc.concat((x, -x), axis=1),
      "stack": lambda: sc.stack((x, -x)),
      "gather": lambda: sc.gather(x, [[11, 0, 3], [8, 9, 4]]),
      "scatter": lambda: _scatter_zero(x, [0, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 8], (3, 3)),
      "sum": lambda: x.sum(),
      "elementwise": lambda: x.sin() * x,
    }[operation]()
    return sc.values(result)

  v = np.array([0.2, -0.7, 0.9, 1.3])
  reference = finite_difference(fn, v)
  np.testing.assert_allclose(sc.jacobian(fn)(v), reference, rtol=1e-7, atol=1e-7)

  @sc.function(sc.arg("v", 4), name=f"sparse_movement_loss_{operation}")
  def loss(v):
    return fn(v).sum()

  np.testing.assert_allclose(sc.gradient(loss)(v), reference.sum(axis=0), rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("operation", ["slice", "gather_matrix", "gather_vector", "scatter"])
@pytest.mark.parametrize("kind", ["grad", "sparse_jac"])
def test_sparse_movement_derivative_buffers_scale_with_stored_entries(operation, kind):
  n = 512
  source = sparse.diags_array([-np.ones(n - 1), 4 * np.ones(n), -np.ones(n - 1)], offsets=[-1, 0, 1], format="coo")
  pattern = _pattern(source)
  leaf = sc.arg("a", sc.TensorType(source.shape, pattern=pattern), layout="csc")
  reverse = np.arange(n - 1, -1, -1)
  logical = np.asarray(pattern.rows) * n + pattern.cols

  def body(a):
    moved = {
      "slice": lambda: a[::2, :],
      "gather_matrix": lambda: sc.gather(a, (reverse[:, None] * n + reverse[None, :])[: n // 2]),
      "gather_vector": lambda: sc.gather(a, np.concatenate((logical[::2], logical[:3]))),
      "scatter": lambda: _scatter_zero(a, np.arange(n * n - 1, -1, -1), source.shape),
    }[operation]()
    values = sc.values(moved).sin()
    return values.sum() if kind == "grad" else values

  fn = sc.function(leaf, name=f"stored_movement_{operation}_{kind}")(body)
  derivative = sc.gradient(fn, wrt="a") if kind == "grad" else sc.sparse_jacobian(fn, wrt="a")
  observed = {}
  lower_function(derivative, observe=observed.__setitem__)
  for program in (observed["lowered"], observed["pass:pack_workspace"]):
    sizes = [np.prod(b.attrs["shape"]) for b in walk_program(program) if b.op == ProgramOp.BUFFER and b.attrs["address_space"] == "private"]
    assert max(sizes, default=0) <= 8 * pattern.nnz


@pytest.mark.parametrize("operation", ["neg", "sin", "tan", "asin", "atan", "sinh", "tanh", "erf", "sqrt", "abs"])
def test_sparse_unary_partials_match_numpy_differences_on_stored_coordinates(operation):
  pattern = CanonicalPattern((17, 19), (0, 4, 9, 16), (18, 2, 8, 0))

  @sc.function(sc.arg("v", 4), name=f"stored_partial_{operation}")
  def fn(v):
    x = sc.pack(v, pattern)
    result = -x if operation == "neg" else getattr(x, operation)()
    assert result.type.pattern is pattern
    return sc.values(result.block())

  v = np.array([-0.75, -0.15, 0.35, 0.55]) if operation == "abs" else np.array([0.15, 0.35, 0.55, 0.75])
  numpy = ELEMENTWISE[ExprOp(operation)].numpy
  reference = finite_difference(numpy, v)
  np.testing.assert_allclose(fn(v), numpy(v), atol=1e-12)
  derivative = sc.jacobian(fn)
  np.testing.assert_allclose(derivative(v), reference, rtol=1e-7, atol=1e-7)
  loss = sc.function(sc.arg("v", 4), name=f"stored_partial_loss_{operation}")(lambda v: fn(v).sum())
  np.testing.assert_allclose(sc.gradient(loss)(v), reference.sum(axis=0), rtol=1e-7, atol=1e-7)
  for function in (fn, derivative):
    for buffer in walk_program(lower_function(function)):
      if buffer.op == ProgramOp.BUFFER and buffer.attrs["address_space"] == "private":
        assert np.prod(buffer.attrs["shape"]) <= 4 * pattern.nnz


@pytest.mark.parametrize("operation", ["add", "sub", "mul"])
def test_sparse_binary_regions_match_numpy_differences(operation):
  a = CanonicalPattern((5, 7), (0, 1, 3, 4), (6, 0, 2, 4))
  b = CanonicalPattern(a.shape, (0, 2, 4), (6, 3, 0))
  output = a.intersection(b) if operation == "mul" else a.union(b)

  @sc.function(sc.arg("v", 7), name=f"stored_binary_{operation}")
  def fn(v):
    x, y = sc.pack(v[:4], a), sc.pack(v[4:], b)
    result = x + y if operation == "add" else x - y if operation == "sub" else x * y
    assert result.type.pattern is output
    return sc.values(result.block())

  def numpy(v):
    x, y = np.zeros(a.shape), np.zeros(b.shape)
    x[a.rows, a.cols], y[b.rows, b.cols] = v[:4], v[4:]
    return getattr(np, {"add": "add", "sub": "subtract", "mul": "multiply"}[operation])(x, y)[output.rows, output.cols]

  v = np.array([0.2, -0.7, 0.9, 1.3, -0.4, 0.8, 1.1])
  reference = finite_difference(numpy, v)
  np.testing.assert_allclose(fn(v), numpy(v), atol=1e-12)
  np.testing.assert_allclose(sc.jacobian(fn)(v), reference, rtol=1e-7, atol=1e-7)
  loss = sc.function(sc.arg("v", 7), name=f"stored_binary_loss_{operation}")(lambda v: fn(v).sum())
  np.testing.assert_allclose(sc.gradient(loss)(v), reference.sum(axis=0), rtol=1e-7, atol=1e-7)


def test_sparse_multiseeds_keep_one_pattern_and_one_index_table():
  pattern = CanonicalPattern((4, 5), (0, 0, 1, 2, 2, 2, 3, 3, 3), (0, 3, 1, 0, 2, 4, 1, 2, 4))
  nseed = 5
  leaf = sc.arg("a", sc.TensorType(pattern.shape, pattern=pattern), layout="csr")
  seeds_leaf = sc.arg("seeds", sc.TensorType((nseed, *pattern.shape), pattern=pattern), layout="csr")

  @sc.function(leaf, seeds_leaf, sc.arg("b", 5), name="shared_sparse_seed_table")
  def fn(a, seeds, b):
    tangent = jvp_many(a.sin(), a, seeds)
    assert tangent.type.pattern is pattern and tangent.type.storage_shape == (nseed, pattern.nnz)
    return jvp_many(a @ b, a, seeds).block()

  seed_values = np.random.default_rng(46).normal(size=(nseed, pattern.nnz))
  b = np.arange(1.0, 6.0)
  expected = np.stack([sparse.coo_array((values, (pattern.rows, pattern.cols)), shape=pattern.shape) @ b for values in seed_values])
  np.testing.assert_allclose(fn(np.ones(pattern.nnz), seed_values, b), expected, atol=1e-12)
  seen = {}
  lower_function(fn, observe=seen.__setitem__)
  tables = [n for n in walk_program(seen["lowered"]) if n.op == ProgramOp.BUFFER and np.array_equal(n.attrs.get("values", ()), pattern.cols)]
  assert len(tables) == 1


@pytest.mark.parametrize("operation", ["matmul", "gather", "scatter"])
def test_sparse_data_jacobian_has_bounded_loop_and_source_growth(operation):
  loop_counts = []
  for n in (4, 8, 16):
    rows = tuple(np.repeat(np.arange(n), 2))
    cols = tuple(np.column_stack((np.arange(n), (np.arange(n) + 1) % n)).reshape(-1))
    pattern = CanonicalPattern((n, n), rows, cols)

    @sc.function(sc.arg("data", pattern.nnz), sc.arg("x", n), name=f"sparse_seed_growth_{operation}_{n}")
    def fn(data, x):
      a = sc.pack(data, pattern)
      if operation == "matmul":
        result = a @ x
      elif operation == "gather":
        result = sc.gather(a, np.asarray(pattern.rows) * n + pattern.cols)
      else:
        result = sc.scatter(a, np.arange(n * n), (n, n))
      return sc.values(result).sin().block()

    jac = as_concrete(sc.jacobian(fn, wrt="data"))
    jac = jac._replace(outputs=tuple(out.block() for out in jac.outputs))
    module = render_c_module(jac, lanes=1)
    loop_counts.append(sum(node.op == ProgramOp.FOR for node in walk_program(module.program)))
    assert loop_counts[-1] <= 12
    assert len(module.body.encode()) < 16_000
    rng = np.random.default_rng(225)
    data, x = rng.normal(size=pattern.nnz), rng.normal(size=n)
    if operation == "matmul":
      a = sparse.coo_array((data, (pattern.rows, pattern.cols)), shape=pattern.shape).toarray()
      expected = np.zeros((n, pattern.nnz))
      expected[pattern.rows, np.arange(pattern.nnz)] = x[list(pattern.cols)]
      expected *= np.cos(a @ x)[:, None]
    else:
      expected = np.diag(np.cos(data))
    np.testing.assert_allclose(jac(data, x), expected, atol=1e-12)
  assert max(loop_counts) == min(loop_counts)


@pytest.mark.parametrize("operation", ["gather_vector", "gather_matrix", "scatter_vector", "scatter_matrix"])
@pytest.mark.parametrize("empty", [False, True])
def test_batched_sparse_movement_jvp_matches_numpy(operation, empty):
  pattern = CanonicalPattern((3, 4), () if empty else (0, 1, 2, 2), () if empty else (3, 0, 1, 3))
  shape = (2, *pattern.shape)
  nseed = 5
  leaf = sc.arg("a", sc.TensorType(shape, pattern=pattern), layout="csr")
  seeds_leaf = sc.arg("seeds", sc.TensorType((nseed, *shape), pattern=pattern), layout="csr")
  gather_indices = np.array([0, 3, 11, 12, 23, 3])
  indices = gather_indices if operation.endswith("vector") else gather_indices.reshape((2, 3))
  output_shape = (7,) if operation.endswith("vector") else (3, 5)
  scatter_indices = np.arange(np.prod(shape)) % np.prod(output_shape)

  @sc.function(leaf, seeds_leaf, name=f"batched_sparse_{operation}_{empty}")
  def fn(a, seeds):
    moved = sc.gather(a, indices) if operation.startswith("gather") else sc.scatter(a, scatter_indices, output_shape)
    return sc.to_dense(jvp_many(moved, a, seeds)).block()

  rng = np.random.default_rng(225)
  av = rng.normal(size=(2, pattern.nnz))
  seed_values = rng.normal(size=(nseed, 2, pattern.nnz))
  dense = np.zeros((nseed, *shape))
  dense[:, :, list(pattern.rows), list(pattern.cols)] = seed_values
  if operation.startswith("gather"):
    expected = dense.reshape((nseed, -1))[:, indices]
  else:
    expected = np.zeros((nseed, *output_shape))
    for i in range(nseed):
      np.add.at(expected[i].reshape(-1), scatter_indices, dense[i].reshape(-1))
  np.testing.assert_allclose(fn(av, seed_values), expected, atol=1e-12)


@pytest.mark.parametrize("form", ["straight", "dense", "affine", "table"])
def test_forward_map_regions_and_reverse_scatter_match_finite_differences(form):
  indices = {
    "straight": np.array([9, 2, 4]),
    "dense": np.arange(13),
    "affine": np.arange(12, -1, -1),
    "table": np.array([4, 7, 0, 1, 11, 6, 10, 8, 2, 12, 3, 9, 5]),
  }[form]
  pattern = CanonicalPattern((13, 17), tuple(range(13)), tuple(i * 5 % 17 for i in range(13)))
  logical = np.asarray(pattern.rows) * 17 + pattern.cols

  @sc.function(sc.arg("v", 13), name=f"region_ad_{form}")
  def fn(v):
    return sc.gather(sc.pack(v, pattern), logical[indices])

  values = np.random.default_rng(46).normal(size=13)
  reference = finite_difference(fn, values)
  np.testing.assert_allclose(sc.jacobian(fn)(values), reference, rtol=1e-7, atol=1e-7)
  weights = np.random.default_rng(47).normal(size=len(indices))

  @sc.function(sc.arg("v", 13), name=f"region_loss_{form}")
  def loss(v):
    return fn(v) @ sc.const(weights)

  np.testing.assert_allclose(sc.gradient(loss)(values), weights @ reference, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("left", [True, False])
def test_sampled_adjoint_does_not_build_an_outer_product(left):
  pattern = CanonicalPattern((20_000, 17_000), (0, 1, 1, 19_999), (16_999, 0, 2, 1))
  input_type = sc.TensorType(pattern.shape, pattern=pattern)
  width, height = (17_000, 20_000) if left else (20_000, 17_000)

  @sc.function(sc.arg("a", input_type, layout="csr"), sc.arg("x", width), sc.arg("cot", height), name=f"sampled_outer_{left}")
  def fn(a, x, cot):
    product = a @ x if left else x @ a
    assert product.type.pattern is None
    abar, xbar = vjp((product,), (a, x), (cot,))
    assert abar.type.pattern is pattern
    return sc.values(abar).block(), xbar

  program = lower_function(fn)
  for buffer in walk_program(program):
    if buffer.op == ProgramOp.BUFFER and buffer.attrs["address_space"] == "private":
      assert np.prod(buffer.attrs["shape"]) <= 2 * (width + height)
  x = np.random.default_rng(46).normal(size=width)
  cot = np.random.default_rng(47).normal(size=height)
  av = np.array([0.2, -0.7, 0.9, 1.3])
  actual, dense_cot = fn(av, x, cot)
  row, col = (pattern.rows, pattern.cols) if left else (pattern.cols, pattern.rows)
  np.testing.assert_allclose(actual, cot[list(row)] * x[list(col)])
  matrix = sparse.coo_array((av, (pattern.rows, pattern.cols)), shape=pattern.shape)
  np.testing.assert_allclose(dense_cot, (matrix.T if left else matrix) @ cot, atol=1e-12)
