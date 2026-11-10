"""Sparse products use stored-entry tables and one dense accumulator column."""

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.ir.program import ProgramOp, walk_program
from scaly.ir.types import CanonicalPattern
from scaly.passes.lowering import lower_function


def _pattern(a):
  a = sparse.coo_array(a)
  return CanonicalPattern(a.shape, tuple(map(int, a.row)), tuple(map(int, a.col)))


def _layout(pattern, layout):
  return sc.SparsityPattern(pattern.shape, pattern.rows[::-1], pattern.cols[::-1]) if layout == "coordinates" else layout


def _operands(case):
  rng = np.random.default_rng(47)
  if case == "empty_product":
    return sparse.coo_array(([2.0], ([1], [0])), shape=(4, 3)), sparse.coo_array(([3.0], ([2], [1])), shape=(3, 5))
  if case == "empty_operand":
    return sparse.coo_array((4, 3)), sparse.eye_array(3, 5)
  if case == "empty_dimension":
    return sparse.coo_array((4, 0)), sparse.coo_array((0, 5))
  if case == "banded":
    a = sparse.diags_array([np.ones(8), np.arange(1.0, 10), -np.ones(8)], offsets=[-1, 0, 1])
    return a, a.T
  if case == "gram":
    a = sparse.coo_array(rng.normal(size=(7, 5)) * (rng.random((7, 5)) < 0.35))
    return a.T, a
  if case == "singleton":
    return sparse.coo_array([[2.0]]), sparse.coo_array([[3.0]])
  a = rng.normal(size=(6, 7)) * (rng.random((6, 7)) < 0.35)
  b = rng.normal(size=(7, 8)) * (rng.random((7, 8)) < 0.3)
  a[2] = 0
  b[:, 4] = 0
  a[0, 3] = 1
  a = sparse.coo_array(a)
  a.data[0] = 0
  return a, sparse.coo_array(b)


@pytest.mark.parametrize("case", ["rectangular", "banded", "gram", "empty_product", "empty_operand", "empty_dimension", "singleton"])
@pytest.mark.parametrize("layout", ["csc", "csr", "coordinates"])
@pytest.mark.parametrize("lowering", ["block", "scalar"])
def test_sparse_product_matches_scipy_in_declared_layout(case, layout, lowering):
  a, b = _operands(case)
  pa, pb = _pattern(a), _pattern(b)
  pc = pa.matmul(pb)
  fn = sc.function(
    sc.arg("a", sc.TensorType(a.shape, pattern=pa), layout=_layout(pa, layout)),
    sc.arg("b", sc.TensorType(b.shape, pattern=pb), layout=_layout(pb, layout)),
    outputs=sc.arg("c", sc.TensorType(pc.shape, pattern=pc), layout=_layout(pc, layout)),
    name=f"sparse_product_{case}_{layout}_{lowering}",
  )(lambda a, b: (a @ b).with_lowering(lowering))
  actual = fn(a, b)
  assert actual.nnz == pc.nnz
  np.testing.assert_allclose(actual.toarray(), (a @ b).toarray(), atol=1e-12)
  observed = {}
  lower_function(fn, observe=observed.__setitem__)
  private = [n for n in walk_program(observed["lowered"]) if n.op == ProgramOp.BUFFER and n.attrs["address_space"] == "private"]
  assert [n.attrs["shape"] for n in private] == ([(a.shape[0],)] if pc.nnz else [])


@pytest.mark.parametrize("n,degree", [(64, 5), (256, 5), (128, 32)])
def test_weighted_gram_tables_and_workspace_follow_fill(n, degree):
  rows = np.repeat(np.arange(2 * n), degree)
  cols = ((np.arange(2 * n)[:, None] // 2 + np.arange(degree)) % n).reshape(-1)
  g = sparse.coo_array((np.ones(len(rows)), (rows, cols)), shape=(2 * n, n))
  pg = _pattern(g)
  pw = CanonicalPattern((2 * n, 2 * n), tuple(range(2 * n)), tuple(range(2 * n)))
  fn = sc.function(sc.arg("g", sc.TensorType(g.shape, pattern=pg)), sc.arg("w", 2 * n), name=f"weighted_gram_{n}_{degree}")(
    lambda g, w: (g.T @ sc.pack(w, pw) @ g).block()
  )
  observed = {}
  lower_function(fn, observe=observed.__setitem__)
  fill = _pattern(g.T @ g).nnz
  for program in (observed["lowered"], observed["pass:pack_workspace"]):
    buffers = [b for b in walk_program(program) if b.op == ProgramOp.BUFFER]
    tables = sum(np.prod(b.attrs["shape"]) for b in buffers if "values" in b.attrs)
    assert tables <= 6 * (2 * pg.nnz + 2 * n + fill)
    private = [b for b in buffers if b.attrs["address_space"] == "private" and "alias_of" not in b.attrs]
    assert max((np.prod(b.attrs["shape"]) for b in private), default=0) <= max(pg.nnz, n)
    assert sum(np.prod(b.attrs["shape"]) for b in private) <= 2 * pg.nnz + 2 * n
  w = np.linspace(0.5, 1.5, 2 * n)
  np.testing.assert_allclose(fn(g, w).toarray(), (g.T @ sparse.diags_array(w) @ g).toarray(), atol=1e-12)


def test_sparse_product_ignores_empty_logical_columns_in_tables():
  a = CanonicalPattern((10000, 13000), (1, 55, 9999), (12000, 7, 12000))
  b = CanonicalPattern((13000, 17000), (7, 12000), (16000, 3))
  fn = sc.function(sc.arg("a", sc.TensorType(a.shape, pattern=a)), sc.arg("b", sc.TensorType(b.shape, pattern=b)), name="wide_sparse_product")(
    lambda a, b: (a @ b).block()
  )
  observed = {}
  lower_function(fn, observe=observed.__setitem__)
  for program in (observed["lowered"], observed["pass:pack_workspace"]):
    buffers = [n for n in walk_program(program) if n.op == ProgramOp.BUFFER]
    assert sum(np.prod(n.attrs["shape"]) for n in buffers if "values" in n.attrs) < 50
    assert max(np.prod(n.attrs["shape"]) for n in buffers if n.attrs["address_space"] == "private") == a.shape[0]
  av = sparse.coo_array(([2.0, -3.0, 4.0], (a.rows, a.cols)), shape=a.shape)
  bv = sparse.coo_array(([5.0, 6.0], (b.rows, b.cols)), shape=b.shape)
  actual = fn(av, bv).tocsr()
  np.testing.assert_array_equal(actual.data, [12.0, -15.0, 24.0])


def test_sparse_product_skips_absent_infinite_factors_but_keeps_stored_zeros():
  pattern = CanonicalPattern((2, 2), (0, 1), (0, 1))
  fn = sc.function(
    sc.arg("a", sc.TensorType(pattern.shape, pattern=pattern)), sc.arg("b", sc.TensorType(pattern.shape, pattern=pattern)), name="sparse_inf"
  )(lambda a, b: (a @ b).block())
  np.testing.assert_array_equal(fn(np.array([1.0, 0.0]), np.array([np.inf, np.inf])).toarray(), [[np.inf, 0.0], [0.0, np.nan]])


@pytest.mark.parametrize("also_return_product", [False, True])
def test_sampled_product_handles_absent_and_repeated_entries_and_shared_products(also_return_product):
  a = sparse.coo_array([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 3.0, 4.0]])
  b = sparse.coo_array([[0.0, 1.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0]])
  pa, pb = _pattern(a), _pattern(b)
  indices = np.array([0, 1, 7, 11, 1, 8])

  @sc.function(
    sc.arg("a", sc.TensorType(a.shape, pattern=pa)),
    sc.arg("b", sc.TensorType(b.shape, pattern=pb)),
    name=f"shared_sparse_product_{also_return_product}",
  )
  def fn(a, b):
    product = a @ b
    sample = sc.gather(product, indices).block()
    return (sample, product) if also_return_product else sample

  reference = (a @ b).toarray()
  actual = fn(a, b)
  if also_return_product:
    actual, product = actual
    np.testing.assert_array_equal(product.toarray(), reference)
  np.testing.assert_array_equal(actual, reference.ravel()[indices])
