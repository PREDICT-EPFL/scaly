"""Sparse SELECT values, stored-coordinate derivatives and masked singular partials."""

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.ad.derivatives import finite_difference
from scaly.ir.program import ProgramOp, walk_program
from scaly.ir.types import CanonicalPattern
from scaly.passes.lowering import lower_function


@pytest.mark.parametrize('batch,condition_shape', [((), ()), ((), (1, 300)), ((2,), (2, 1, 300)), ((0,), (0, 1, 300))])
def test_sparse_select_union_broadcast_and_derivatives(batch, condition_shape):
  a = CanonicalPattern((200, 300), (0, 1, 4, 9, 10, 32, 100, 199), (2, 0, 299, 17, 42, 201, 2, 299))
  b = CanonicalPattern(a.shape, (0, 2, 4, 10, 51, 100, 199), (2, 3, 12, 42, 101, 2, 0))
  output = a.union(b)
  width = a.nnz + b.nnz
  v_leaf = sc.arg('v', (*batch, width))
  c_leaf = sc.arg('c', condition_shape)

  def expression(v, c):
    result = sc.where(c > 0, sc.pack(v[..., :a.nnz], a).sin(), sc.pack(v[..., a.nnz:], b).tanh())
    assert result.type.pattern is output
    assert result.type.storage_shape == (*batch, output.nnz)
    return result.block()

  fn = sc.function(v_leaf, c_leaf, name='sparse_select_union')(lambda v, c: sc.values(expression(v, c)))
  loss = sc.function(v_leaf, c_leaf, name='sparse_select_union_loss')(lambda v, c: expression(v, c).sum())
  random = np.random.default_rng(46)
  v = random.normal(size=(*batch, width))
  condition = np.asarray(random.random(size=condition_shape) > .4)

  def numpy(v):
    x, y = np.zeros((*batch, *a.shape)), np.zeros((*batch, *b.shape))
    x[..., a.rows, a.cols], y[..., b.rows, b.cols] = v[..., :a.nnz], v[..., a.nnz:]
    return np.where(condition, np.sin(x), np.tanh(y))[..., output.rows, output.cols]

  np.testing.assert_allclose(fn(v, condition), numpy(v), atol=1e-12)
  for node in walk_program(lower_function(fn)):
    if node.op == ProgramOp.BUFFER and node.attrs['address_space'] == 'private':
      assert np.prod(node.attrs['shape']) <= 4 * max(1, np.prod(batch)) * (width + output.nnz)
  if v.size:
    reference = finite_difference(numpy, v)
    np.testing.assert_allclose(sc.jacobian(fn, wrt='v')(v, condition), reference, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(sc.gradient(loss, wrt='v')(v, condition), reference.sum(axis=0).reshape(v.shape), atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize('movement', ['identity', 'transpose', 'reshape', 'slice', 'vector', 'gather', 'scatter', 'stack', 'concat', 'sum', 'matvec', 'rmatvec', 'matmat', 'rmatmat'])
def test_sparse_select_mask_reaches_singular_partial_through_movement(movement):
  pattern = CanonicalPattern((4, 5), (0, 0, 1, 2, 3, 3), (0, 4, 1, 3, 0, 2))
  v_leaf = sc.arg('v', pattern.nnz)
  c_leaf = sc.arg('c')

  def body(v, c):
    x = sc.pack(v, pattern).sqrt()
    branch = {
      'identity': lambda: x,
      'transpose': lambda: x.T,
      'reshape': lambda: x.reshape((2, 10)),
      'slice': lambda: x[::-1, ::2],
      'vector': lambda: x[0],
      'gather': lambda: sc.gather(x, [[19, 0, 4], [10, 16, 9]]),
      'scatter': lambda: sc.scatter(x, np.arange(20) % 9, (3, 3)),
      'stack': lambda: sc.stack((x, 2 * x)),
      'concat': lambda: sc.concat((x, 2 * x), axis=1),
      'sum': lambda: x.sum(),
      'matvec': lambda: x @ sc.const(np.ones(5)),
      'rmatvec': lambda: sc.const(np.ones(4)) @ x,
      'matmat': lambda: x @ sc.const(np.ones((5, 2))),
      'rmatmat': lambda: sc.const(np.ones((2, 4))) @ x,
    }[movement]()
    ones = sc.const(np.ones(branch.type.storage_shape))
    other = sc.pack(ones, branch.type.pattern, shape=branch.shape) if branch.type.pattern is not None else ones
    return sc.where(c > 0, branch, other).sum().block()

  loss = sc.function(v_leaf, c_leaf, name=f'sparse_select_mask_{movement}')(body)
  grad = sc.gradient(loss, wrt='v')
  zero = np.zeros(pattern.nnz)
  np.testing.assert_array_equal(grad(zero, np.asarray(False)), zero)
  positive = np.linspace(.2, 1.4, pattern.nnz)
  for condition in (False, True):
    c = np.asarray(condition)
    reference = finite_difference(lambda v: loss(v, c), positive).reshape(positive.shape)
    np.testing.assert_allclose(grad(positive, c), reference, atol=1e-7, rtol=1e-7)


def test_sparse_select_stored_bool_condition_and_hessian_without_logical_matrix_buffer():
  pattern = CanonicalPattern((1000, 1000), (0, 1, 7, 30, 103, 999), (999, 0, 40, 3, 2, 998))
  condition_pattern = CanonicalPattern(pattern.shape, (0, 30, 999), (999, 3, 998))
  c_leaf = sc.arg('c', condition_pattern.nnz)
  v_leaf = sc.arg('v', pattern.nnz)

  def body(v, c):
    x = sc.pack(v, pattern)
    c = sc.pack(sc.cast(c, 'bool'), condition_pattern)
    return sc.where(c, x.sqrt(), sc.pack(sc.const(np.ones(pattern.nnz)), pattern)).sum().block()

  loss = sc.function(v_leaf, c_leaf, name='sparse_select_stored_bool')(body)
  grad, hess = sc.gradient(loss, wrt='v'), sc.sparse_hessian(loss, wrt='v')
  v = np.array([.25, 0, 0, 0, 0, 1.0])
  condition = np.array([True, False, True])
  expected_grad = np.array([1., 0, 0, 0, 0, .5])
  np.testing.assert_array_equal(grad(v, condition), expected_grad)
  hp = hess.sparsity()
  assert hp is not None
  actual_hess = sparse.coo_array((hess(v, condition), (hp.rows, hp.cols)), shape=hp.shape).toarray()
  np.testing.assert_array_equal(actual_hess, np.diag([-2., 0, 0, 0, 0, -.25]))
  difference = finite_difference(lambda z: grad(z, condition), v)
  np.testing.assert_allclose(actual_hess, difference, atol=1e-7, rtol=1e-7)
  for function in (loss, grad, hess):
    observed = {}
    lower_function(function, observe=observed.__setitem__)
    for program in (observed['lowered'], observed['pass:pack_workspace']):
      for node in walk_program(program):
        if node.op == ProgramOp.BUFFER and node.attrs['address_space'] == 'private':
          assert np.prod(node.attrs['shape']) <= 20 * pattern.nnz ** 2

