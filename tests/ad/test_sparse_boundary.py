"""Sparse derivatives retain stored coordinates through calls and mapped helpers."""

import numpy as np
import pytest
from scipy import sparse

from scaly.ad.forward import jvp, jvp_many
from scaly.ad.derivatives import gradient
from scaly.ad.reverse import vjp
from scaly.ad.sparse import sparse_hessian, sparse_jacobian
from scaly.ad.sparsity import jacobian_sparsity
from scaly.function.concrete import ConcreteFunction
from scaly.function.sugar import vmap
from scaly.ir import expr as e
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
