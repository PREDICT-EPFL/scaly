"""Sparse derivatives retain stored coordinates through calls and mapped helpers."""

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.ad.forward import jvp, jvp_many
from scaly.ad.derivatives import gradient
from scaly.ad.reverse import vjp
from scaly.ad.sparse import sparse_hessian, sparse_jacobian
from scaly.ad.sparsity import jacobian_sparsity
from scaly.function.concrete import ConcreteFunction
from scaly.function.sugar import _mapped_call, vmap
from scaly.ir import expr as e
from scaly.ir.layout import SparseLayout
from scaly.ir.types import CanonicalPattern, TensorType, stored_indices


def _input(name, pattern, batch=()):
  return e.Expr(e.ExprOp.INPUT, type=TensorType((*batch, *pattern.shape), pattern=pattern), name=name)


def _evaluate(name, inputs, outputs, values):
  fn = ConcreteFunction._from_exprs(name, inputs, outputs, [x.name for x in inputs], [f"out{i}" for i in range(len(outputs))])
  actuals = [
    sparse.coo_array((value.reshape(-1), np.unravel_index(stored_indices(x.type), x.shape)), shape=x.shape) if x.type.pattern is not None else value
    for x, value in zip(inputs, values, strict=True)
  ]
  return fn._flat_numerical_call(*actuals)


def test_pack_values_derivatives_match_numpy():
  pattern = CanonicalPattern((3, 4), (0, 1, 2), (3, 0, 2))
  x = e.Expr.sym("sparse_ad_values", 3)
  sparse = e.pack(x * x, pattern)
  result = e.values(sparse.sin())
  forward = jvp(result, x, e.Expr.const([1.0, -2.0, 0.5]))
  reverse = vjp((result,), (x,), (e.Expr.const([2.0, -1.0, 3.0]),))[0]
  xval = np.array([0.2, -0.5, 1.2])
  got = _evaluate("sparse_ad_pack", [x], [forward, reverse], [xval])
  derivative = 2 * xval * np.cos(xval * xval)
  np.testing.assert_allclose(got[0], derivative * [1, -2, 0.5])
  np.testing.assert_allclose(got[1], derivative * [2, -1, 3])
  np.testing.assert_array_equal(jacobian_sparsity(result, x).to_mask(), np.eye(3, dtype=bool))


@pytest.mark.parametrize("mapped", [False, True])
def test_sparse_call_seed_and_adjoint_storage_match_numpy(mapped):
  pattern = CanonicalPattern((3, 4), (0, 1, 2), (3, 0, 2))
  formal = _input("sparse_ad_formal", pattern)
  callee = ConcreteFunction._from_exprs("sparse_ad_body", [formal], [formal.sin()], [formal.name], ["y"])
  shape = (2,) if mapped else ()
  x = _input("sparse_ad_outer", pattern, shape)
  call = vmap(callee, 2).symbolic_call(x) if mapped else callee.symbolic_call(x)
  seed_values = e.Expr.const(np.arange(1.0, x.type.storage_size + 1).reshape(x.type.storage_shape))
  seed = e.pack(seed_values, pattern, shape=x.shape)
  tangent = jvp(call, x, seed)
  cotangent = e.pack(seed_values * 2, pattern, shape=x.shape)
  adjoint = vjp((call,), (x,), (cotangent,))[0]
  assert tangent.type.pattern is pattern and adjoint.type.pattern is pattern
  xval = np.arange(x.type.storage_size, dtype=float).reshape(x.type.storage_shape) / 5
  got = _evaluate(f"sparse_ad_call_{mapped}", [x], [e.values(tangent), e.values(adjoint)], [xval])
  expected = np.cos(xval) * np.arange(1.0, x.type.storage_size + 1).reshape(x.type.storage_shape)
  np.testing.assert_allclose(got[0], expected)
  np.testing.assert_allclose(got[1], expected * 2)


def test_sparse_seed_batches_and_compact_jacobian_use_stored_columns():
  pattern = CanonicalPattern((100_000, 100_000), (0, 20, 99_999), (1, 50, 99_998))
  x = _input("sparse_ad_large", pattern)
  result = e.values(x.sin())
  seeds = e.pack(e.Expr.const(np.eye(3)), pattern, shape=(3, *x.shape))
  tangent = jvp_many(result, x, seeds)
  compact = sparse_jacobian(result, x)
  assert compact.sparsity.shape == (3, 10_000_000_000)
  assert compact.sparsity.nnz == 3 and compact.coloring_width == 1
  got = _evaluate("sparse_ad_large", [x], [tangent, compact.values], [np.array([0.2, -0.5, 1.2])])
  np.testing.assert_allclose(got[0], np.diag(np.cos([0.2, -0.5, 1.2])))
  np.testing.assert_allclose(got[1], np.cos([0.2, -0.5, 1.2]))


def test_sparse_mapped_multiseeds_and_incidence_use_stored_windows():
  pattern = CanonicalPattern((3, 4), (0, 1, 2), (3, 0, 2))
  formal = _input("sparse_ad_many_formal", pattern)
  callee = ConcreteFunction._from_exprs("sparse_ad_many_body", [formal], [formal.sin()], [formal.name], ["y"])
  x = _input("sparse_ad_many_outer", pattern, (2,))
  result = vmap(callee, 2).symbolic_call(x)
  seed_values = np.array([[[1, -2, 0.5], [3, 2, 1]], [[0, 0, 0], [0, 0, 0]], [[-1, 4, 2], [-3, 1, 0.5]]], dtype=float)
  seeds = e.pack(e.Expr.const(seed_values), pattern, shape=(3, *x.shape))
  tangent = jvp_many(result, x, seeds)
  compact = sparse_jacobian(e.values(result), x)
  xval = np.arange(6, dtype=float).reshape(2, 3) / 5
  got = _evaluate("sparse_ad_many", [x], [e.values(tangent), compact.values], [xval])
  np.testing.assert_allclose(got[0], seed_values * np.cos(xval))
  np.testing.assert_allclose(got[1], np.cos(xval).reshape(-1))
  assert compact.sparsity.rows == tuple(range(6))
  assert compact.sparsity.cols == (3, 4, 10, 15, 16, 22)


def test_sparse_union_intersection_and_broadcast_adjoint_match_numpy():
  pattern = CanonicalPattern((2, 3), (0, 1, 1), (2, 0, 2))
  other = CanonicalPattern((2, 3), (0, 1), (0, 2))
  x, z = _input("sparse_ad_union_x", pattern), _input("sparse_ad_union_z", other)
  scale = e.Expr.sym("sparse_ad_scale")
  result = (x.sin() + z) * x * scale
  cot = e.pack(e.Expr.const([1.0, 2.0, 3.0]), pattern)
  dx, dz, ds = vjp((result,), (x, z, scale), (cot,))
  assert dx.type.pattern is pattern and dz.type.pattern is other
  xv, zv = np.array([0.2, -0.5, 1.2]), np.array([2.0, 3.0])
  got = _evaluate("sparse_ad_union", [x, z, scale], [e.values(dx), e.values(dz), ds], [xv, zv, np.array(0.7)])
  coefficient = np.array([0.0, 0.0, zv[1]])
  expected_dx = (np.sin(xv) + coefficient + xv * np.cos(xv)) * [1, 2, 3] * 0.7
  np.testing.assert_allclose(got[0], expected_dx)
  np.testing.assert_allclose(got[1], [0, xv[2] * 3 * 0.7])
  np.testing.assert_allclose(got[2], np.sum((np.sin(xv) + coefficient) * xv * [1, 2, 3]))


@pytest.mark.parametrize("left", [True, False])
def test_mixed_sparse_matmul_derivatives_match_numpy(left):
  pattern = CanonicalPattern((2, 3), (0, 1, 1), (2, 0, 2))
  x = _input("sparse_ad_matmul", pattern)
  coefficient = np.arange(1.0, (3 if left else 2) + 1)
  result = x @ e.Expr.const(coefficient) if left else e.Expr.const(coefficient) @ x
  seed = e.pack(e.Expr.const([0.5, -1.0, 2.0]), pattern)
  tangent = jvp(result, x, seed)
  cot = np.arange(1.0, result.size + 1)
  adjoint = vjp((result,), (x,), (e.Expr.const(cot),))[0]
  xv = np.array([0.2, -0.5, 1.2])
  got = _evaluate(f"sparse_ad_matmul_{left}", [x], [tangent, e.values(adjoint)], [xv])
  dense_seed = np.zeros(pattern.shape)
  dense_seed[pattern.rows, pattern.cols] = [0.5, -1, 2]
  reference_tangent = dense_seed @ coefficient if left else coefficient @ dense_seed
  reference_adjoint = np.outer(cot, coefficient) if left else np.outer(coefficient, cot)
  np.testing.assert_allclose(got[0], reference_tangent)
  np.testing.assert_allclose(got[1], reference_adjoint[pattern.rows, pattern.cols])


def test_sparse_gradient_and_hessian_keep_logical_coordinates_without_dense_storage():
  pattern = CanonicalPattern((100_000, 100_000), (0, 20, 99_999), (1, 50, 99_998))
  x = _input("sparse_ad_quadratic", pattern)
  objective = (x * x).sum()
  grad = gradient(objective, x)
  hess = sparse_hessian(objective, x)
  assert grad.type.pattern is pattern
  entries = [1, 2_000_050, 9_999_999_998]
  assert hess.sparsity.rows == hess.sparsity.cols == tuple(entries)
  got = _evaluate("sparse_ad_quadratic", [x], [e.values(grad), hess.values], [np.array([0.2, -0.5, 1.2])])
  np.testing.assert_allclose(got[0], [0.4, -1, 2.4])
  np.testing.assert_allclose(got[1], [2, 2, 2])


@pytest.mark.parametrize("shape", [(), (3,), (1, 3), (2, 1)])
def test_sparse_forward_broadcast_keeps_seed_axis_separate(shape):
  import scaly as sc

  pattern = CanonicalPattern((2, 3), (0, 1), (0, 1))
  leaf = sc.arg("v", shape)
  fn = sc.function(leaf, name="sparse_broadcast_seed")(lambda v: sc.const([3.0, 7.0], pattern=pattern) * v)
  actual = np.arange(np.prod(shape), dtype=float).reshape(shape) if shape else np.array(2.0)
  dense = np.zeros(pattern.shape)
  dense[pattern.rows, pattern.cols] = [3.0, 7.0]
  expected = np.stack([(dense * direction.reshape(shape)).reshape(-1) for direction in np.eye(actual.size)], axis=1)
  np.testing.assert_allclose(sc.jacobian(fn)(actual), expected)


@pytest.mark.parametrize("shape", [(2, 3), (100_000, 100_000)])
@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("declared", [False, True])
def test_custom_sparse_rules_bound_relations_by_storage_without_losing_zero_rows(mapped, declared, shape):
  import scaly as sc

  pattern = CanonicalPattern(shape, (0, 1), (0, 1))
  leaf = sc.arg("x", TensorType(pattern.shape, pattern=pattern), layout="csr")
  primal = sc.function(leaf, name="sparse_custom_primal")(lambda x: x * sc.const([1.0, 0.0], pattern=pattern))
  rule = sc.function(leaf, leaf.relabel("d"), name="sparse_custom_rule")(lambda x, dx: 3 * x * dx)
  domain = (0, shape[1] + 1)
  size = int(np.prod(shape))
  declared_pattern = sc.SparsityPattern((size, size), domain, domain)
  ruled = sc.custom_derivative(primal, jvp=rule, sparsity=(lambda *_: declared_pattern) if declared else None)
  fn = sc.vmap(ruled, 2) if mapped else ruled
  data = np.array([[2.0, 5.0], [7.0, 11.0]]) if mapped else np.array([2.0, 5.0])
  batch = (np.repeat([0, 1], 2),) if mapped else ()
  coords = (*batch, np.tile(pattern.rows, 2 if mapped else 1), np.tile(pattern.cols, 2 if mapped else 1))
  source = sparse.coo_array((data.reshape(-1), coords), shape=(*((2,) if mapped else ()), *pattern.shape))
  derivative = sc.sparse_jacobian(fn)
  result = derivative(source)
  output_pattern = derivative.sparsity()
  assert output_pattern is not None
  entries = np.array([*domain, *(i + size for i in domain)]) if mapped else np.array(domain)
  assert output_pattern.shape == ((2 * size, 2 * size) if mapped else (size, size))
  assert np.isin(output_pattern.rows, entries).all() and np.isin(output_pattern.cols, entries).all()
  rows = np.searchsorted(entries, output_pattern.rows)
  cols = np.searchsorted(entries, output_pattern.cols)
  actual = sparse.coo_array((result, (rows, cols)), shape=(len(entries), len(entries))).toarray()
  np.testing.assert_allclose(actual, np.diag(3 * data.reshape(-1)))


def test_empty_sparse_mapped_windows_keep_outer_gradient_and_hessian_shapes():
  import scaly as sc

  pattern = CanonicalPattern((2, 3), (), ())
  stage = sc.function(sc.arg("x", TensorType(pattern.shape, pattern=pattern)), name="empty_sparse_stage")(lambda x: x.sin())
  loss = sc.function(sc.arg("raw", 6), name="empty_sparse_window_loss")(lambda raw: sc.vmap(stage, 2)(sc.window(raw, 0, 3)).sum())
  data = np.arange(6.0)
  grad = sc.gradient(loss)(data)
  assert grad.shape == data.shape
  np.testing.assert_array_equal(grad, np.zeros(6))
  np.testing.assert_array_equal(sc.hessian(loss)(data), np.zeros((6, 6)))


@pytest.mark.parametrize("nseed", [2, 5])
def test_sparse_tangent_broadcast_preserves_seeds_and_shared_pattern_batches(nseed):
  pattern = CanonicalPattern((2, 3), (0, 1), (0, 1))
  x = _input("sparse_broadcast_tangent", pattern)
  result = x * e.Expr.const(np.array([2.0, -3.0]).reshape(2, 1, 1))
  seeds = np.arange(nseed * 2, dtype=float).reshape(nseed, 2)
  tangent = jvp_many(result, x, e.pack(e.Expr.const(seeds), pattern))
  got = _evaluate("sparse_broadcast_batches", [x], [e.values(tangent)], [np.array([3.0, 7.0])])[0]
  expected = seeds[:, None, :] * np.array([2.0, -3.0])[None, :, None]
  np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize("layout", ["csc", "coordinate"])
@pytest.mark.parametrize(
  "batch,length,stride,empty",
  [((), 20, 3, False), ((), 3, 0, False), ((2, 3), 3, 7, False), ((2, 3), 3, 0, False), ((), 0, 3, False), ((), 3, 3, True), ((0,), 3, 3, False)],
)
def test_declared_layout_window_jacobian_keeps_local_coloring(monkeypatch, layout, batch, length, stride, empty):
  pattern = CanonicalPattern((3, 4), (), ()) if empty else CanonicalPattern((3, 4), (0, 0, 1, 2, 2), (0, 3, 1, 0, 2))
  order = sc.SparsityPattern(pattern.shape, (2, 0, 2, 0, 1), (2, 0, 0, 3, 1)) if not empty else sc.SparsityPattern(pattern.shape, (), ())
  leaf = sc.arg("x", TensorType((*batch, *pattern.shape), pattern=pattern), layout=order if layout == "coordinate" else layout)
  declared = leaf.layouts[0]
  assert isinstance(declared, SparseLayout)
  if pattern.nnz:
    assert not np.array_equal(declared.permutation, declared.inverse_permutation)
  weights = np.arange(1.0, 13.0).reshape(pattern.shape)
  stage = sc.function(leaf, name="layout_window_stage")(lambda x: (sc.to_dense(x.sin()) * weights).sum())
  width = leaf.types[0].storage_size
  size = 1 + max(0, length - 1) * stride + width
  raw = sc.sym("layout_window_raw", size)
  result = sc.vmap(stage, length)(sc.window(raw, 1, stride)).vec()

  def reject_global_coloring(*args, **kwargs):
    pytest.fail("declared-layout window fell back to global coloring")

  monkeypatch.setattr("scaly.ad.sparse.sparse_jacobian_colored", reject_global_coloring)
  compact = sparse_jacobian(result, raw)
  data = np.linspace(-0.8, 0.9, size)
  expected = np.zeros((length, size))
  coefficients = np.broadcast_to(weights[declared.rows, declared.cols], leaf.types[0].storage_shape).reshape(-1)
  for iteration in range(length):
    columns = 1 + iteration * stride + np.arange(width)
    expected[iteration, columns] = coefficients * np.cos(data[columns])
  np.testing.assert_array_equal(compact.sparsity.to_mask(), expected != 0)
  actual, dense = _evaluate("layout_window_jac", [raw], [compact.to_dense(), sc.jacobian(result, raw)], [data])
  np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
  np.testing.assert_allclose(actual, dense, rtol=1e-13, atol=1e-13)
  assert compact.coloring_width == (width if length and width else 0)


def test_gathered_mapped_columns_accumulate_within_and_across_formals(monkeypatch):
  stage = sc.function(sc.arg("x", 3), sc.arg("y", 3), name="gathered_columns_stage")(
    lambda x, y: sc.stack([x.sin().sum() + y.sin().sum(), x[0] * y[1] + x[2] * y[0]])
  )
  raw = sc.sym("gathered_columns_raw", 6)
  indices = np.array([[4, 1, 4], [2, 0, 3]])
  actual = raw.reshape((2, 3)).gather(indices).vec()
  result = _mapped_call(stage, 2, [(actual, 0, 3), (raw, 1, 2)])
  reference = sc.stack([stage(raw.gather(columns), raw[1 + i * 2 : 4 + i * 2]) for i, columns in enumerate(indices)])

  def reject_global_coloring(*args, **kwargs):
    pytest.fail("constant gather fell back to global coloring")

  monkeypatch.setattr("scaly.ad.sparse.sparse_jacobian_colored", reject_global_coloring)
  compact = sparse_jacobian(result, raw)
  np.testing.assert_array_equal(compact.sparsity.to_mask(), jacobian_sparsity(reference, raw).to_mask())
  data = np.linspace(-0.8, 0.9, raw.size)
  actual, dense = _evaluate("gathered_columns_jac", [raw], [compact.to_dense(), sc.jacobian(reference, raw)], [data])
  np.testing.assert_allclose(actual, dense, rtol=1e-13, atol=1e-13)
