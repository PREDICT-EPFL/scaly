"""Sparse-sparse tangents and sampled adjoints agree with SciPy and finite differences."""

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.ad.forward import jvp_many
from scaly.ad.reverse import vjp
from scaly.function.model import as_concrete
from scaly.ir.expr import ExprOp, matmul
from scaly.ir.program import ProgramOp, walk_program
from scaly.ir.types import CanonicalPattern
from scaly.passes.lowering import lower_function
from scaly.passes.lowering.plans import SampledProductPlan


@pytest.mark.parametrize("case", ["rectangular", "banded", "empty_product", "empty_operand", "empty_dimension"])
@pytest.mark.parametrize("layout", ["csc", "csr", "coordinates"])
def test_sparse_product_forward_and_reverse(case, layout):
  rng = np.random.default_rng(47)
  a = sparse.coo_array(rng.random((5, 7)) < 0.3)
  b = sparse.coo_array(rng.random((7, 4)) < 0.4)
  if case == "banded":
    a = sparse.diags_array([np.ones(6), np.ones(6)], offsets=[0, 1], shape=(6, 7), format="coo")
    b = sparse.diags_array([np.ones(6), np.ones(6)], offsets=[0, -1], shape=(7, 6), format="coo")
  elif case == "empty_product":
    a = sparse.coo_array(([1.0], ([0], [0])), shape=(5, 7))
    b = sparse.coo_array(([1.0], ([6], [3])), shape=(7, 4))
  elif case == "empty_operand":
    a = sparse.coo_array((5, 7))
  elif case == "empty_dimension":
    a, b = sparse.coo_array((5, 0)), sparse.coo_array((0, 4))
  pa = CanonicalPattern(a.shape, tuple(map(int, a.row)), tuple(map(int, a.col)))
  pb = CanonicalPattern(b.shape, tuple(map(int, b.row)), tuple(map(int, b.col)))
  pc = pa.matmul(pb)

  def leaf(name, pattern, batch=()):
    order = sc.SparsityPattern(pattern.shape, pattern.rows[::-1], pattern.cols[::-1]) if layout == "coordinates" else layout
    return sc.arg(name, sc.TensorType((*batch, *pattern.shape), pattern=pattern), layout=order)

  @sc.function(
    leaf("a", pa),
    leaf("b", pb),
    leaf("da", pa, (2,)),
    leaf("db", pb, (2,)),
    leaf("cot", pc),
    outputs=sc.group(leaf("dc", pc, (2,)), leaf("abar", pa), leaf("bbar", pb)),
    name=f"sparse_product_ad_{case}_{layout}",
  )
  def derivatives(a, b, da, db, cot):
    c = a @ b
    tangent = jvp_many(c, a, da) + jvp_many(c, b, db)
    abar, bbar = vjp((c,), (a, b), (cot,))
    assert tangent.type.pattern is c.type.pattern
    assert abar.type.pattern is a.type.pattern and bbar.type.pattern is b.type.pattern
    return tangent, abar, bbar

  def matrix(pattern, data):
    return sparse.coo_array((data, (pattern.rows, pattern.cols)), shape=pattern.shape)

  av, bv, cv = (rng.normal(size=p.nnz) for p in (pa, pb, pc))
  a, b, cot = matrix(pa, av), matrix(pb, bv), matrix(pc, cv)
  dav, dbv = rng.normal(size=(2, pa.nnz)), rng.normal(size=(2, pb.nnz))

  def batch(pattern, data):
    return sparse.coo_array(
      (data.ravel(), (np.repeat(np.arange(2), pattern.nnz), np.tile(pattern.rows, 2), np.tile(pattern.cols, 2))), shape=(2, *pattern.shape)
    )

  dc, abar, bbar = derivatives(a, b, batch(pa, dav), batch(pb, dbv), cot)
  eps = 1e-5
  for i in range(2):
    da, db = matrix(pa, dav[i]), matrix(pb, dbv[i])
    reference = da @ b + a @ db
    difference = ((a + eps * da) @ (b + eps * db) - (a - eps * da) @ (b - eps * db)) / (2 * eps)
    np.testing.assert_allclose(dc.toarray()[i], reference.toarray(), atol=1e-12)
    np.testing.assert_allclose(dc.toarray()[i], difference.toarray(), atol=1e-8)
    dual = (abar.toarray() * da.toarray()).sum() + (bbar.toarray() * db.toarray()).sum()
    np.testing.assert_allclose(dual, (cot.toarray() * difference.toarray()).sum(), atol=1e-8)
  expected_a, expected_b = (cot @ b.T).toarray(), (a.T @ cot).toarray()
  np.testing.assert_allclose(abar.toarray()[pa.rows, pa.cols], expected_a[pa.rows, pa.cols], atol=1e-12)
  np.testing.assert_allclose(bbar.toarray()[pb.rows, pb.cols], expected_b[pb.rows, pb.cols], atol=1e-12)
  assert abar.nnz == pa.nnz and bbar.nnz == pb.nnz


def test_sampled_sparse_adjoint_does_not_materialize_the_unsampled_product(monkeypatch):
  n = 97
  pa = CanonicalPattern((n, n), tuple(range(n)), tuple(range(n)))
  pb = CanonicalPattern((n, 2), tuple(np.repeat(np.arange(n), 2)), (0, 1) * n)
  pc = pa.matmul(pb)
  matmul = CanonicalPattern.matmul

  def guarded(a, b):
    assert (a.shape[0], b.shape[1]) != (n, n), "the sampled adjoint must not build its full Boolean pattern"
    return matmul(a, b)

  monkeypatch.setattr(CanonicalPattern, "matmul", guarded)

  @sc.function(
    sc.arg("a", sc.TensorType(pa.shape, pattern=pa)),
    sc.arg("b", sc.TensorType(pb.shape, pattern=pb)),
    sc.arg("cot", sc.TensorType(pc.shape, pattern=pc)),
    name="sampled_sparse_adjoint",
  )
  def fn(a, b, cot):
    return sc.values(vjp((a @ b,), (a,), (cot,))[0]).block()

  observed = {}
  lower_function(fn, observe=observed.__setitem__)
  for program in (observed["lowered"], observed["pass:pack_workspace"]):
    buffers = [b for b in walk_program(program) if b.op == ProgramOp.BUFFER and b.attrs["address_space"] == "private"]
    assert max((np.prod(b.attrs["shape"]) for b in buffers), default=0) <= 2 * n
  a = sparse.coo_array((np.ones(n), (pa.rows, pa.cols)), shape=pa.shape)
  b = sparse.coo_array((np.arange(1.0, 2 * n + 1), (pb.rows, pb.cols)), shape=pb.shape)
  cot = sparse.coo_array((np.arange(pc.nnz, dtype=float), (pc.rows, pc.cols)), shape=pc.shape)
  np.testing.assert_allclose(fn(a, b, cot), (cot @ b.T).diagonal())


@pytest.mark.parametrize("n", [32, 64])
def test_weighted_gram_with_a_dense_column_has_stored_size_reverse_and_second_order_work(n, monkeypatch):
  rows = np.tile(np.arange(2 * n), 3)
  cols = np.concatenate((np.arange(2 * n) // 2, np.minimum(np.arange(2 * n) // 2 + 1, n - 1), np.full(2 * n, n - 1)))
  g = sparse.coo_array((np.ones(len(rows)), (rows, cols)), shape=(2 * n, n))
  g.sum_duplicates()
  pg = CanonicalPattern(g.shape, tuple(map(int, g.row)), tuple(map(int, g.col)))
  pw = CanonicalPattern((2 * n, 2 * n), tuple(range(2 * n)), tuple(range(2 * n)))
  matmul = CanonicalPattern.matmul

  def guarded(a, b):
    result = matmul(a, b)
    assert result.nnz <= 12 * n, "an adjoint built the unsampled Boolean product"
    return result

  monkeypatch.setattr(CanonicalPattern, "matmul", guarded)

  @sc.function(sc.arg("v", pg.nnz), sc.arg("w", 2 * n), sc.arg("seeds", (3, 2 * n)), name=f"sparse_dense_column_ad_{n}")
  def fn(v, w, seeds):
    matrix = sc.pack(v, pg)
    c = matrix.T @ sc.pack(w, pw) @ matrix
    loss = sc.values(c).sin().sum()
    grad = vjp((loss,), (w,), (sc.const(1.0),))[0]
    return grad, jvp_many(grad, w, seeds)

  concrete = as_concrete(fn)
  sampled_work = 0
  for node in concrete.nodes:
    if node.op == ExprOp.MATMUL:
      assert node.type.storage_size <= 12 * n
      if node.attrs.get("sampled") is not None:
        a, b = node.args[0].type.pattern, node.args[1].type.pattern
        assert a is not None and b is not None and node.type.pattern is not None
        plan = SampledProductPlan.build(a, b, node.type.pattern)
        sampled_work += (plan.right_stops - plan.right_starts).sum()
        assert sum(v.size for v in vars(plan).values()) <= 6 * (a.nnz + b.nnz + node.type.pattern.nnz)
  assert sampled_work <= 160 * n
  observed = {}
  lower_function(fn, observe=observed.__setitem__)
  for program in (observed["lowered"], observed["pass:pack_workspace"]):
    sizes = [np.prod(b.attrs["shape"]) for b in walk_program(program) if b.op == ProgramOp.BUFFER and b.attrs["address_space"] == "private"]
    assert max(sizes, default=0) <= 36 * n
  v, w = np.linspace(0.2, 0.7, pg.nnz), np.linspace(0.4, 1.2, 2 * n)
  seeds = np.random.default_rng(47).normal(size=(3, 2 * n))
  matrix = sparse.coo_array((v, (pg.rows, pg.cols)), shape=pg.shape).toarray()

  def reference(w):
    c = matrix.T @ (w[:, None] * matrix)
    return np.einsum("ij,jk,ik->i", matrix, np.cos(c), matrix)

  grad, tangent = fn(v, w, seeds)
  np.testing.assert_allclose(grad, reference(w), atol=1e-11)
  eps = 1e-5
  for direction, actual in zip(seeds, tangent, strict=True):
    expected = (reference(w + eps * direction) - reference(w - eps * direction)) / (2 * eps)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_sparse_product_hessian_keeps_sampled_patterns():
  n = 24
  pattern = CanonicalPattern((n, n), tuple(range(n)), tuple(range(n)))
  b = sparse.coo_array(np.linspace(0.1, 1.0, 2 * n).reshape(n, 2))

  @sc.function(sc.arg("w", n), name="sampled_sparse_hessian")
  def loss(w):
    return sc.values(sc.pack(w, pattern) @ sc.const(b)).sin().sum()

  hess = sc.hessian(loss)
  observed = {}
  lower_function(hess, observe=observed.__setitem__)
  for program in (observed["lowered"], observed["pass:pack_workspace"]):
    sizes = [np.prod(b.attrs["shape"]) for b in walk_program(program) if b.op == ProgramOp.BUFFER and b.attrs["address_space"] == "private"]
    assert max(sizes, default=0) <= 2 * n * n
  w = np.linspace(0.2, 0.8, n)
  expected = np.diag(-(np.sin(w[:, None] * b.toarray()) * b.toarray() ** 2).sum(axis=1))
  np.testing.assert_allclose(hess(w), expected, atol=1e-12)


def test_sampled_product_both_operand_derivatives_and_hessian_match_differences():
  from scaly.ad.derivatives import finite_difference

  a = CanonicalPattern((3, 5), (0, 0, 1, 2, 2), (0, 2, 1, 0, 4))
  b = CanonicalPattern((5, 4), (0, 1, 2, 3, 4), (0, 2, 3, 1, 1))
  requested = CanonicalPattern((3, 4), (0, 1, 1, 2, 2), (3, 0, 2, 0, 1))

  @sc.function(sc.arg("v", a.nnz + b.nnz), name="sampled_sparse_derivatives")
  def loss(v):
    product = matmul(sc.pack(v[: a.nnz], a), sc.pack(v[a.nnz :], b), sampled=requested)
    return sc.values(product).sin().sum()

  def reference(v):
    x = sparse.coo_array((v[: a.nnz], (a.rows, a.cols)), shape=a.shape)
    y = sparse.coo_array((v[a.nnz :], (b.rows, b.cols)), shape=b.shape)
    return np.sin((x @ y).toarray()[requested.rows, requested.cols]).sum()

  v = np.linspace(0.2, 0.9, a.nnz + b.nnz)
  grad = sc.gradient(loss)
  np.testing.assert_allclose(grad(v), finite_difference(reference, v).ravel(), atol=1e-8)
  expected = finite_difference(grad, v)
  np.testing.assert_allclose(sc.hessian(loss)(v), expected, atol=1e-8)
  reverse_grad = sc.function(sc.arg("v", len(v)), name="sampled_sparse_reverse_twice")(lambda v: grad(v).sum())
  np.testing.assert_allclose(sc.gradient(reverse_grad)(v), expected.sum(axis=0), atol=1e-8)
  hess = sc.sparse_hessian(loss)
  pattern = hess.sparsity()
  assert pattern is not None
  actual = sparse.coo_array((hess(v), (pattern.rows, pattern.cols)), shape=pattern.shape).toarray()
  np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_sampled_product_derivative_sparsity_does_not_join_unsampled_rows(monkeypatch):
  from scaly.ad import sparsity

  n = 128
  pattern = CanonicalPattern((n, n), tuple(range(n)), tuple(range(n)))
  b = sparse.coo_array(np.linspace(0.1, 1.0, 2 * n).reshape(n, 2))

  @sc.function(sc.arg("w", n), name="sampled_sparse_analysis")
  def loss(w):
    return sc.values(sc.pack(w, pattern) @ sc.const(b)).sin().sum()

  incidence = sparsity._incidence

  def guarded(shape, rows, cols):
    assert rows.size <= 16 * n, "sampled derivative analysis joined unsampled output rows"
    return incidence(shape, rows, cols)

  monkeypatch.setattr(sparsity, "_incidence", guarded)
  hess = sc.sparse_hessian(loss)
  hp = hess.sparsity()
  assert hp is not None and hp.nnz == n
  w = np.linspace(0.2, 0.8, n)
  expected = -(np.sin(w[:, None] * b.toarray()) * b.toarray() ** 2).sum(axis=1)
  np.testing.assert_allclose(hess(w), expected[np.asarray(hp.rows)], atol=1e-12)
