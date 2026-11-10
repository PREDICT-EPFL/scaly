"""Sparse-sparse tangents and sampled adjoints agree with SciPy and finite differences."""

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.ad.forward import jvp_many
from scaly.ad.reverse import vjp
from scaly.ir.program import ProgramOp, walk_program
from scaly.ir.types import CanonicalPattern
from scaly.passes.lowering import lower_function


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


def test_sampled_sparse_adjoint_does_not_materialize_the_unsampled_product():
  n = 97
  pa = CanonicalPattern((n, n), tuple(range(n)), tuple(range(n)))
  pb = CanonicalPattern((n, 2), tuple(np.repeat(np.arange(n), 2)), (0, 1) * n)
  pc = pa.matmul(pb)

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
