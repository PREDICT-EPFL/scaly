"""``SparseMatrix``: static CSC pattern, ``Expr`` values. Every operation is checked against SciPy
on random patterns with symbolic values, and derivatives through the values against finite
differences."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.ad import finite_difference
from scaly.ad.derivatives import gradient, hessian, jacobian
from scaly.linalg import SparseMatrix

RNG = np.random.default_rng(404)


def _random(m: int, n: int, density: float, seed: int) -> sparse.csc_array:
  a = sparse.random_array((m, n), density=density, random_state=seed, format="csc")
  a.sort_indices()
  return sparse.csc_array(a)


def _symbolic(name: str, a: sparse.csc_array) -> tuple[SparseMatrix, np.ndarray]:
  """A matrix with ``a``'s pattern and symbolic values, and the numeric values that make it ``a``."""
  mat = SparseMatrix.symbol(name, a)
  return mat, mat.to_scipy(np.zeros(mat.nnz)).tocsc().astype(float) and _values_in_order(mat, a)


def _values_in_order(mat: SparseMatrix, a: sparse.csc_array) -> np.ndarray:
  rows, cols = mat.coordinates()
  return np.asarray(a.tocsr()[rows, cols]).reshape(-1)


def _eval(inputs: list[sc.Expr], outputs: list[sc.Expr], point: list[np.ndarray]) -> tuple[np.ndarray, ...]:
  fn = sc.Function._from_exprs(
    f"sm{abs(hash(tuple(o.id for o in outputs))) % 10**9}", inputs, outputs, [str(x.name) for x in inputs], [f"o{k}" for k in range(len(outputs))]
  )
  return fn._flat_numerical_call(*point)


def _dense(mat: SparseMatrix, inputs: list[sc.Expr], point: list[np.ndarray]) -> np.ndarray:
  return _eval(inputs, [mat.to_dense()], point)[0]


# --- construction -------------------------------------------------------------------------------


def test_from_coo_sums_duplicates_and_sorts() -> None:
  rows, cols = np.array([2, 0, 2, 1, 0]), np.array([1, 1, 1, 0, 1])
  v = sc.sym("v", 5)
  mat = SparseMatrix.from_coo(rows, cols, v, (3, 2))
  assert mat.nnz == 3 and list(mat.indptr) == [0, 1, 3] and list(mat.indices) == [1, 0, 2]
  vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
  want = sparse.coo_array((vals, (rows, cols)), shape=(3, 2)).toarray()
  np.testing.assert_array_equal(_dense(mat, [v], [vals]), want)
  distinct = SparseMatrix.from_coo([2, 0, 1], [0, 0, 0], v[:3], (3, 1))  # a 3-cycle, not its own inverse
  np.testing.assert_array_equal(_dense(distinct, [v], [vals]), [[2.0], [3.0], [1.0]])


@pytest.mark.parametrize("kind", ["sparsity", "mask", "scipy"])
def test_from_pattern_reads_values_in_the_pattern_order(kind: str) -> None:
  a = _random(5, 4, 0.5, 1).tocsr()
  a.data = np.arange(1.0, a.nnz + 1)
  if kind == "sparsity":
    coo = a.tocoo()
    perm = RNG.permutation(coo.nnz)
    pattern = sc.SparsityType(a.shape, tuple(int(r) for r in coo.row[perm]), tuple(int(c) for c in coo.col[perm]))
    vals = coo.data[perm]
  elif kind == "mask":
    pattern, vals = a.toarray() != 0, a.toarray()[a.toarray() != 0]
  else:
    pattern, vals = a, a.data
  v = sc.sym("v", a.nnz)
  np.testing.assert_array_equal(_dense(SparseMatrix.from_pattern(pattern, v), [v], [vals]), a.toarray())


def test_from_scipy_from_dense_and_constants() -> None:
  a = _random(4, 6, 0.4, 2)
  const = SparseMatrix.from_scipy(a)
  x = sc.sym("x", 1)
  np.testing.assert_array_equal(_dense(const, [x], [np.zeros(1)]), a.toarray())
  dense_const = SparseMatrix.from_dense(sc.const(a.toarray()))
  assert dense_const.nnz == a.nnz
  xm = sc.sym("xm", (4, 6))
  full = SparseMatrix.from_dense(xm)
  on_pattern = SparseMatrix.from_dense(xm, a)
  assert full.nnz == 24 and on_pattern.nnz == a.nnz
  val = RNG.standard_normal((4, 6))
  np.testing.assert_array_equal(_dense(full, [xm], [val]), val)
  np.testing.assert_array_equal(_dense(on_pattern, [xm], [val]), np.where(a.toarray() != 0, val, 0.0))


def test_from_sparse_jacobian_and_hessian() -> None:
  x = sc.sym("x", 6)
  f = (x[:-1] * x[1:]).sum() + (x * x * x).sum() + x[0] * x[5]
  h = SparseMatrix.from_sparse_jacobian(sc.sparse_hessian(f, x))
  xv = RNG.standard_normal(6)
  np.testing.assert_allclose(_dense(h, [x], [xv]), _eval([x], [hessian(f, x)], [xv])[0], rtol=1e-13)
  assert h.nnz == 6 + 2 * 5 + 2


def test_structural_helpers() -> None:
  d = sc.sym("d", 3)
  assert SparseMatrix.diag(d).nnz == 3 and SparseMatrix.identity(4).nnz == 4 and SparseMatrix.zeros((2, 3)).nnz == 0
  a = _random(5, 5, 0.4, 3)
  mat = SparseMatrix.symbol("a", a)
  rows, cols = mat.coordinates()
  assert all(mat.position(int(r), int(c)) == k for k, (r, c) in enumerate(zip(rows, cols, strict=True)))
  assert mat.position(0, 0) in (None, 0)
  sp = mat.sparsity
  assert sp.nnz == mat.nnz and sp.rows == tuple(int(r) for r in rows)


def test_validation() -> None:
  with pytest.raises(ValueError, match="indptr"):
    SparseMatrix((2, 2), [0, 2], [0, 1], sc.sym("v", 2))
  with pytest.raises(ValueError, match="sorted and distinct"):
    SparseMatrix((2, 1), [0, 2], [1, 0], sc.sym("v", 2))
  with pytest.raises(ValueError, match="out of bounds"):
    SparseMatrix((2, 1), [0, 1], [5], sc.sym("v", 1))
  with pytest.raises(ValueError, match="values must have shape"):
    SparseMatrix((2, 1), [0, 1], [0], sc.sym("v", 2))
  with pytest.raises(ValueError, match="distinct"):
    SparseMatrix.from_pattern(sparse.coo_array((np.ones(3), ([0, 1, 0], [0, 0, 0])), shape=(2, 1)), sc.sym("w", 3))
  with pytest.raises(ValueError, match="must have shape"):
    SparseMatrix.from_pattern(np.eye(2), sc.sym("w", 3))
  with pytest.raises(ValueError, match="scalar"):
    SparseMatrix.identity(3) * sc.sym("s", 2)
  with pytest.raises(ValueError, match="no block"):
    SparseMatrix.block([[None, SparseMatrix.identity(2)], [None, SparseMatrix.identity(2)]])
  with pytest.raises(ValueError, match="different sizes"):
    SparseMatrix.block([[SparseMatrix.identity(2), SparseMatrix.identity(3)]])


# --- algebra against SciPy ----------------------------------------------------------------------

A_NP = _random(7, 5, 0.35, 11)
B_NP = _random(7, 5, 0.35, 12)
C_NP = _random(5, 6, 0.4, 13)


def _pair():
  a, b, c = SparseMatrix.symbol("a", A_NP), SparseMatrix.symbol("b", B_NP), SparseMatrix.symbol("c", C_NP)
  inputs = [a.values, b.values, c.values]
  point = [_values_in_order(a, A_NP), _values_in_order(b, B_NP), _values_in_order(c, C_NP)]
  return a, b, c, inputs, point


@pytest.mark.parametrize(
  "name",
  ["add", "sub", "neg", "scale", "scale_expr", "div", "hadamard", "rows", "cols", "transpose", "spgemm", "tril", "triu", "add_diag", "block"],
)
def test_matrix_results_match_scipy(name: str) -> None:
  a, b, c, inputs, point = _pair()
  s = sc.sym("s", ())
  d5, d7 = sc.sym("d5", 5), sc.sym("d7", 7)
  inputs = [*inputs, s, d5, d7]
  sv, d5v, d7v = 1.7, RNG.standard_normal(5), RNG.standard_normal(7)
  point = [*point, np.array(sv), d5v, d7v]
  A, B, C = A_NP.toarray(), B_NP.toarray(), C_NP.toarray()
  sq = SparseMatrix.symbol("q", C_NP[:, :5])
  q = C_NP[:, :5].toarray()
  qv = _values_in_order(sq, C_NP[:, :5].tocsc())
  cases = {
    "add": (lambda: a + b, A + B),
    "sub": (lambda: a - b, A - B),
    "neg": (lambda: -a, -A),
    "scale": (lambda: 2.5 * a, 2.5 * A),
    "scale_expr": (lambda: a * s, A * sv),
    "div": (lambda: a / s, A / sv),
    "hadamard": (lambda: a * b, A * B),
    "rows": (lambda: a.scale_rows(d7), d7v[:, None] * A),
    "cols": (lambda: a.scale_cols(d5), A * d5v[None, :]),
    "transpose": (lambda: a.T, A.T),
    "spgemm": (lambda: a @ c, A @ C),
    "tril": (lambda: sq.tril(-1), np.tril(q, -1)),
    "triu": (lambda: sq.triu(1), np.triu(q, 1)),
    "add_diag": (lambda: sq.add_diagonal(d5), q + np.diag(d5v)),
    "block": (lambda: SparseMatrix.block([[a, None], [c.T, sc.const(np.ones((6, 5)))]]), np.block([[A, np.zeros((7, 5))], [C.T, np.ones((6, 5))]])),
  }
  build, want = cases[name]
  got = _dense(build(), [*inputs, sq.values], [*point, qv])
  np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-15)


def test_products_with_dense_operands() -> None:
  a, _, _, inputs, point = _pair()
  x, y, X, Y = sc.sym("x", 5), sc.sym("y", 7), sc.sym("X", (5, 3)), sc.sym("Y", (2, 7))
  vals = [RNG.standard_normal(5), RNG.standard_normal(7), RNG.standard_normal((5, 3)), RNG.standard_normal((2, 7))]
  got = _eval([*inputs, x, y, X, Y], [a @ x, y @ a, a @ X, Y @ a, a.matvec(x)], [*point, *vals])
  A = A_NP.toarray()
  for g, w in zip(got, [A @ vals[0], vals[1] @ A, A @ vals[2], vals[3] @ A, A @ vals[0]], strict=True):
    np.testing.assert_allclose(g, w, rtol=1e-13, atol=1e-15)


def test_diagonal_and_pattern_moves() -> None:
  sq = SparseMatrix.symbol("q", sparse.csc_array(np.array([[1.0, 0, 2], [0, 0, 3], [4, 0, 5]])))
  qv = np.arange(1.0, 6.0)
  inputs = [sq.values]
  np.testing.assert_array_equal(_eval(inputs, [sq.diagonal()], [qv])[0], [1.0, 0.0, 5.0])
  bigger = sq.with_pattern(SparseMatrix.symbol("full", np.ones((3, 3), dtype=bool)))
  assert bigger.nnz == 9
  np.testing.assert_array_equal(_dense(bigger, inputs, [qv]), _dense(sq, inputs, [qv]))
  with pytest.raises(ValueError, match="does not contain"):
    SparseMatrix.symbol("full", np.ones((3, 3), dtype=bool)).with_pattern(sq)
  kept = sq.select(np.array([True, False, True, False, True]))
  np.testing.assert_array_equal(_dense(kept, inputs, [qv]), [[1.0, 0, 3], [0, 0, 0], [0, 0, 5]])  # values are in CSC order


def test_kkt_assembly_matches_scipy_bmat() -> None:
  p_np, a_np = _random(6, 6, 0.3, 21), _random(3, 6, 0.4, 22)
  p_np = sparse.csc_array(p_np + p_np.T)
  P, A = SparseMatrix.symbol("P", p_np), SparseMatrix.symbol("A", a_np)
  rho, delta = sc.sym("rho", ()), sc.sym("delta", ())
  kkt = SparseMatrix.block([[P.add_diagonal(rho), A.T], [A, SparseMatrix.identity(3) * (-delta)]])
  point = [_values_in_order(P, p_np), _values_in_order(A, a_np), np.array(1e-6), np.array(1e-4)]
  got = _dense(kkt, [P.values, A.values, rho, delta], point)
  want = sparse.bmat([[p_np + 1e-6 * sparse.eye_array(6), a_np.T], [a_np, -1e-4 * sparse.eye_array(3)]]).toarray()
  np.testing.assert_allclose(got, want, rtol=1e-15)
  assert kkt.nnz == sparse.bmat([[p_np + sparse.eye_array(6), a_np.T], [a_np, sparse.eye_array(3)]]).nnz


# --- derivatives and the Function boundary ------------------------------------------------------


def test_derivatives_through_the_values(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  a, _, c, _, point = _pair()
  x = sc.sym("x", 6)
  y = (a @ c) @ x
  f = sc.sumsqr(y) + (a.T @ y.sin()).sum()
  inputs = [a.values, c.values, x]
  pt = [point[0], point[2], RNG.standard_normal(6)]
  grad_a, jac_a, hess_x = _eval(inputs, [gradient(f, a.values), jacobian(f.reshape((1,)), a.values), hessian(f, x)], pt)

  def value(av: np.ndarray, xv: np.ndarray) -> np.ndarray:
    am = a.to_scipy(av)
    cm = c.to_scipy(pt[1])
    yv = am @ (cm @ xv)
    return np.array([yv @ yv + (am.T @ np.sin(yv)).sum()])

  np.testing.assert_allclose(grad_a, finite_difference(lambda z: value(z, pt[2]), pt[0]).reshape(-1), rtol=1e-6, atol=1e-8)
  np.testing.assert_allclose(jac_a.reshape(-1), grad_a, rtol=1e-12, atol=1e-14)
  grad_x = sc.Function._from_exprs("sm_gx", inputs, [gradient(f, x)], ["a", "c", "x"], ["g"])
  fd = finite_difference(lambda z: grad_x._flat_numerical_call(pt[0], pt[1], z)[0].reshape(-1), pt[2])
  np.testing.assert_allclose(hess_x, fd, rtol=1e-6, atol=1e-7)


def test_compact_values_cross_the_function_boundary() -> None:
  a, _, c, _, point = _pair()
  prod = a @ c
  fn = sc.Function._from_exprs("sm_out", [a.values, c.values], [prod.values], ["a", "c"], ["ac"], output_sparsities=[prod.sparsity])
  (vals,) = fn._flat_numerical_call(point[0], point[2])
  assert fn.output_sparsities[0] == prod.sparsity
  got = sparse.coo_array((vals, (prod.sparsity.rows, prod.sparsity.cols)), shape=prod.shape).toarray()
  np.testing.assert_allclose(got, A_NP.toarray() @ C_NP.toarray(), rtol=1e-13, atol=1e-15)
  assert prod.nnz == (sparse.csc_array((A_NP != 0).astype(float)) @ sparse.csc_array((C_NP != 0).astype(float))).nnz


def test_scalars_and_misfit_selections_are_refused() -> None:
  a = SparseMatrix.symbol("A", np.eye(3, dtype=bool))
  for bad in (lambda: 2.0 + a, lambda: a + 2.0, lambda: a - sc.sym("s", ()), lambda: 1 - a):
    with pytest.raises(TypeError, match="add_diagonal"):
      bad()
  with pytest.raises(ValueError, match="one flag per stored entry"):
    a.select(np.ones(2, dtype=bool))
  assert (a + np.eye(3)).nnz == 3 and SparseMatrix.block([[a, None], [None, np.zeros((2, 2))]]).nnz == 3
