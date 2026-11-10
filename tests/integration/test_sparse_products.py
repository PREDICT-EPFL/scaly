"""Normal equations and finite-element restriction ported from devrush's sparse notebooks."""

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.ad.derivatives import finite_difference
from scaly.ir.types import CanonicalPattern


@pytest.mark.parametrize("layout", ["csc", "csr", "coordinates"])
def test_mpc_weighted_normal_equations_and_gram_derivatives(layout):
  n, nx, nu = 4, 2, 1
  a, b = np.array([[1.0, 0.1], [0.0, 1.0]]), np.array([[0.01], [0.1]])
  g = np.zeros((n * nx, n * (nx + nu)))
  for k in range(n):
    g[k * nx : (k + 1) * nx, k * (nx + nu) : k * (nx + nu) + nu] = -b
    g[k * nx : (k + 1) * nx, k * (nx + nu) + nu : (k + 1) * (nx + nu)] = np.eye(nx)
    if k:
      g[k * nx : (k + 1) * nx, (k - 1) * (nx + nu) + nu : k * (nx + nu)] = -a
  g = sparse.coo_array(g)
  pg = CanonicalPattern(g.shape, tuple(map(int, g.row)), tuple(map(int, g.col)))
  pw = CanonicalPattern((n * nx, n * nx), tuple(range(n * nx)), tuple(range(n * nx)))
  pc = CanonicalPattern(pg.shape[::-1], pg.cols, pg.rows).matmul(pg)
  order = sc.SparsityPattern(pc.shape, pc.rows[::-1], pc.cols[::-1]) if layout == "coordinates" else layout

  @sc.function(
    sc.arg("v", pg.nnz), sc.arg("w", n * nx), outputs=sc.arg("c", sc.TensorType(pc.shape, pattern=pc), layout=order), name=f"mpc_normal_{layout}"
  )
  def normal(v, w):
    g = sc.pack(v, pg)
    return g.T @ sc.pack(w, pw) @ g

  @sc.function(sc.arg("v", pg.nnz), outputs=sc.arg("c", sc.TensorType(pc.shape, pattern=pc), layout=order), name=f"mpc_gram_{layout}")
  def gram(v):
    a = sc.pack(v, pg)
    return a.T @ a

  @sc.function(sc.arg("v", pg.nnz), sc.arg("w", n * nx), name=f"mpc_normal_loss_{layout}")
  def loss(v, w):
    return sc.values(normal(v, w)).sin().sum()

  @sc.function(sc.arg("v", pg.nnz), name=f"mpc_gram_loss_{layout}")
  def gram_loss(v):
    return sc.values(gram(v)).sin().sum()

  v, w = g.tocsr().data, np.linspace(0.5, 1.5, n * nx)

  def reference(v, w):
    g = sparse.coo_array((v, (pg.rows, pg.cols)), shape=pg.shape)
    return (g.T @ sparse.diags_array(w) @ g).toarray()

  expected = reference(v, w)
  np.testing.assert_allclose(sparse.coo_array(normal(v, w)).toarray(), expected, atol=1e-12)
  np.testing.assert_allclose(gram(v).toarray(), (g.T @ g).toarray(), atol=1e-12)
  ref_loss = lambda v, w: np.sin(reference(v, w)[pc.rows, pc.cols]).sum()
  np.testing.assert_allclose(sc.gradient(loss, wrt="v")(v, w), finite_difference(lambda x: ref_loss(x, w), v).ravel(), atol=1e-7)
  np.testing.assert_allclose(sc.gradient(loss, wrt="w")(v, w), finite_difference(lambda x: ref_loss(v, x), w).ravel(), atol=1e-7)
  np.testing.assert_allclose(sc.jacobian(normal, wrt="w")(v, w), finite_difference(lambda x: reference(v, x), w), atol=1e-7)
  np.testing.assert_allclose(sc.gradient(gram_loss)(v), finite_difference(lambda x: ref_loss(x, np.ones(n * nx)), v).ravel(), atol=1e-7)
  grad = sc.gradient(gram_loss)
  np.testing.assert_allclose(sc.hessian(gram_loss)(v), finite_difference(grad, v), atol=1e-6)


def test_fem_assembly_and_sparse_restriction_derivatives():
  nodes = np.arange(9).reshape((3, 3))
  triangles = np.concatenate(
    (
      np.stack((nodes[:-1, :-1], nodes[:-1, 1:], nodes[1:, 1:]), -1).reshape(-1, 3),
      np.stack((nodes[:-1, :-1], nodes[1:, 1:], nodes[1:, :-1]), -1).reshape(-1, 3),
    )
  )
  xy = np.array([(j / 2, i / 2) for i in range(3) for j in range(3)])
  matrices = []
  for triangle in triangles:
    p = xy[triangle]
    basis = np.linalg.inv(np.column_stack((np.ones(3), p)))[1:]
    area = abs(np.linalg.det(np.column_stack((np.ones(3), p)))) / 2
    matrices.append(area * basis.T @ basis)
  matrices = np.array(matrices)
  rows, cols = np.repeat(triangles, 3, axis=1).ravel(), np.tile(triangles, (1, 3)).ravel()
  entries, slots = np.unique(rows * 9 + cols, return_inverse=True)
  pk = CanonicalPattern((9, 9), tuple(map(int, entries // 9)), tuple(map(int, entries % 9)))
  free = np.array([1, 2, 4, 5, 7, 8])
  r = sparse.coo_array((np.ones(len(free)), (np.arange(len(free)), free)), shape=(len(free), 9))

  @sc.function(sc.arg("k", len(triangles)), name="fem_sparse_restriction")
  def conductance(k):
    element_values = sc.gather(k, np.repeat(np.arange(len(triangles)), 9)) * sc.const(matrices.ravel())
    assembled = sc.pack(sc.scatter(element_values, slots, (pk.nnz,)), pk)
    restriction = sc.const(r)
    return restriction @ assembled @ restriction.T

  @sc.function(sc.arg("k", len(triangles)), name="fem_restriction_loss")
  def loss(k):
    return sc.sumsqr(sc.values(conductance(k)))

  def reference(k):
    full = sparse.coo_array(((k[:, None, None] * matrices).ravel(), (rows, cols)), shape=(9, 9)).tocsr()
    return full[free][:, free].toarray()

  k = np.random.default_rng(47).uniform(0.2, 1.0, len(triangles))
  np.testing.assert_allclose(conductance(k).toarray(), reference(k), atol=1e-12)
  np.testing.assert_allclose(sc.jacobian(conductance)(k), finite_difference(reference, k), atol=1e-7)
  np.testing.assert_allclose(sc.gradient(loss)(k), finite_difference(lambda x: (reference(x) ** 2).sum(), k).ravel(), atol=1e-7)
