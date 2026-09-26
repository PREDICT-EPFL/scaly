"""Sparse matrices as ``Function`` arguments and results: ``sc.S`` puts the pattern in the signature."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.codegen import render_c_module
from scaly.linalg import S, SparseMatrix

RNG = np.random.default_rng(7)
MASK = np.array([[1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 1, 1], [0, 0, 1, 1]], dtype=bool)


def _matrix(mask: np.ndarray = MASK) -> sparse.csc_array:
  """A SciPy matrix storing exactly ``mask``'s entries, with nonzero values."""
  rows, cols = np.nonzero(mask)
  return sparse.csc_array((RNG.uniform(1.0, 2.0, rows.size), (rows, cols)), shape=mask.shape)


def _matvec(name: str) -> sc.Function:
  @sc.function(sc.G(S("A", MASK), sc.L("x", 4)), sc.L("y", ...), name=name)
  def f(inputs):
    a, x = inputs
    return a @ x

  return f


def test_every_pattern_spelling_declares_the_same_signature() -> None:
  spellings = [MASK, sparse.csr_array(MASK.astype(float)), SparseMatrix.symbol("P", MASK), SparseMatrix.symbol("P", MASK).sparsity]
  patterns = [S("A", p).pattern for p in spellings]
  assert all(p == patterns[0] for p in patterns)
  assert S("A", MASK).types == (sc.TensorType((int(MASK.sum()),)),)


def test_evaluation_takes_scipy_matrices_with_the_declared_pattern() -> None:
  f = _matvec("sg_matvec")
  a, x = _matrix(), RNG.standard_normal(4)
  for form in (a, sparse.csr_array(a), sparse.coo_array(a), sparse.csc_matrix(a)):
    np.testing.assert_allclose(f((form, x)), a @ x, rtol=1e-14)


def test_explicit_zeros_are_stored_entries() -> None:
  f = _matvec("sg_zeros")
  a = _matrix()
  a.data[0] = 0.0  # still stored
  x = RNG.standard_normal(4)
  np.testing.assert_allclose(f((a, x)), a @ x, rtol=1e-14)
  with pytest.raises(ValueError, match="different sparsity pattern"):
    a.eliminate_zeros()
    f((a, x))


def test_duplicates_are_summed_without_touching_the_callers_matrix() -> None:
  f = _matvec("sg_dups")
  ref = _matrix()
  # Every entry twice, as halves of its value, rows in reverse within each column: not canonical.
  counts = np.diff(ref.indptr)
  indices = np.concatenate([np.r_[col, col][::-1] for col in np.split(ref.indices, ref.indptr[1:-1])])
  data = np.concatenate([np.r_[col, col][::-1] / 2 for col in np.split(ref.data, ref.indptr[1:-1])])
  a = sparse.csc_array((data, indices, np.r_[0, np.cumsum(2 * counts)]), shape=MASK.shape)
  assert not a.has_canonical_format
  before = (a.data.copy(), a.indices.copy(), a.indptr.copy())
  x = RNG.standard_normal(4)
  np.testing.assert_allclose(f((a, x)), ref @ x, rtol=1e-14)
  assert all(np.array_equal(u, v) for u, v in zip(before, (a.data, a.indices, a.indptr), strict=True))


@pytest.mark.parametrize(
  "bad, match",
  [
    (lambda: sparse.csc_array(np.eye(4)), r"3 stored entries|column 0 stores rows \[0\], expected \[0, 2\]"),
    (lambda: _matrix(MASK | np.eye(4, dtype=bool)[::-1]), "column 0 stores rows"),
    (lambda: _matrix(np.ones((4, 5), dtype=bool)), r"shape \(4, 5\), expected \(4, 4\)"),
    (lambda: _matrix().toarray(), "expected a SciPy sparse matrix"),
    (lambda: _matrix().data, "expected a SciPy sparse matrix"),
  ],
  ids=["identity", "extra_entry", "wrong_shape", "dense", "values_only"],
)
def test_evaluation_refuses_anything_but_the_declared_pattern(bad, match) -> None:
  with pytest.raises(ValueError, match=match):
    _matvec("sg_refuse")((bad(), np.zeros(4)))


def test_symbolic_calls_check_the_pattern() -> None:
  f = _matvec("sg_symbolic")
  x = sc.sym("x", 4)
  assert f((SparseMatrix.symbol("M", MASK), x)).op == sc.ExprOp.CALL
  with pytest.raises(ValueError, match="different sparsity pattern"):
    f((SparseMatrix.symbol("M", MASK | np.eye(4, k=1, dtype=bool)), x))
  with pytest.raises(ValueError, match="expected a SparseMatrix for 'A', got Expr"):
    f((sc.sym("v", int(MASK.sum())), x))
  with pytest.raises(TypeError, match="mix Expr and numerical"):
    f((SparseMatrix.symbol("M", MASK), np.zeros(4)))


def _kkt_builder(name: str, declared) -> sc.Function:
  """``K = [[diag(q) + rho I, A^T], [A, -delta I]]``, the lower triangle stored."""

  @sc.function(sc.G(sc.L("q", 4), S("A", MASK[:2])), S("K", declared), name=name)
  def build(inputs):
    q, a = inputs
    k = SparseMatrix.block([[SparseMatrix.diag(q).add_diagonal(1e-6), None], [a, SparseMatrix.identity(2) * -1e-3]])
    return k

  return build


def test_sparse_results_evaluate_to_scipy_and_carry_their_pattern() -> None:
  inferred = _kkt_builder("sg_kkt_inferred", ...)
  pattern = inferred.output_tree.sparsities[0]
  assert pattern is not None and inferred.output_sparsities == (pattern,)
  explicit = _kkt_builder("sg_kkt_explicit", pattern)
  q, a = RNG.uniform(1.0, 2.0, 4), _matrix(MASK[:2])
  want = np.block([[np.diag(q) + 1e-6 * np.eye(4), np.zeros((4, 2))], [a.toarray(), -1e-3 * np.eye(2)]])
  for fn in (inferred, explicit):
    k = fn((q, a))
    assert isinstance(k, sparse.csc_array) and k.nnz == len(pattern.rows)
    np.testing.assert_allclose(k.toarray(), want, rtol=1e-15)
  with pytest.raises(TypeError, match="different sparsity pattern"):
    _kkt_builder("sg_kkt_wrong", MASK)
  with pytest.raises(TypeError, match="different sparsity pattern"):
    _kkt_builder("sg_kkt_wrong", np.ones((6, 6), dtype=bool))


def test_one_function_builds_a_matrix_another_consumes() -> None:
  """The pattern travels with the value: symbolically as a ``SparseMatrix``, numerically as SciPy."""
  build = _kkt_builder("sg_chain_build", ...)
  pattern = build.output_tree.sparsities[0]

  @sc.function(sc.G(S("K", pattern), sc.L("b", 6)), sc.L("x", ...), name="sg_chain_solve")
  def solve(inputs):
    k, b = inputs
    return sc.linalg.SparseLDL(k, name="sg_chain").solve(b)

  @sc.function(sc.G(sc.L("q", 4), S("A", MASK[:2]), sc.L("b", 6)), sc.L("x", ...), name="sg_chain_both")
  def both(inputs):
    q, a, b = inputs
    k = build((q, a))
    assert isinstance(k, SparseMatrix)
    return solve((k, b))

  q, a, b = RNG.uniform(1.0, 2.0, 4), _matrix(MASK[:2]), RNG.standard_normal(6)
  k = build((q, a))
  full = k.toarray() + np.tril(k.toarray(), -1).T
  np.testing.assert_allclose(solve((k, b)), np.linalg.solve(full, b), rtol=1e-10)
  np.testing.assert_allclose(both((q, a, b)), np.linalg.solve(full, b), rtol=1e-10)


def test_generated_code_is_that_of_the_values_vector() -> None:
  """``S`` is interface only: the kernel is the one a Function over the bare values vector gets."""
  f = _matvec("sg_same_code")
  a, x = SparseMatrix.symbol("A", MASK), sc.sym("x", 4)
  flat = sc.Function._from_exprs("sg_same_code", [a.values, x], [a @ x], ["A", "x"], ["y"])
  assert render_c_module(f).body == render_c_module(flat).body
  assert render_c_module(f).header == render_c_module(flat).header

  build = _kkt_builder("sg_same_header", ...)
  q, am = sc.sym("q", 4), SparseMatrix.symbol("A", MASK[:2])
  k = SparseMatrix.block([[SparseMatrix.diag(q).add_diagonal(1e-6), None], [am, SparseMatrix.identity(2) * -1e-3]])
  flat_build = sc.Function._from_exprs("sg_same_header", [q, am.values], [k.values], ["q", "A"], ["K"], output_sparsities=[k.sparsity])
  assert render_c_module(build).body == render_c_module(flat_build).body
  assert render_c_module(build).header == render_c_module(flat_build).header


def test_derivatives_keep_the_sparse_signature() -> None:
  """Derived Functions take the same tree; a gradient in a sparse input is one entry per stored value."""

  @sc.function(sc.G(S("A", MASK), sc.L("x", 4)), sc.L("f", ...), name="sg_quad")
  def quad(inputs):
    a, x = inputs
    return x @ (a @ x)

  a, x = _matrix(), RNG.standard_normal(4)
  rows, cols = np.nonzero(MASK)
  order = np.lexsort((rows, cols))  # CSC order of the stored entries
  grad_a = sc.gradient(quad, "f", "A")
  np.testing.assert_allclose(grad_a((a, x)), (x[rows] * x[cols])[order], rtol=1e-14)
  grad_x = sc.gradient(quad, "f", "x")
  np.testing.assert_allclose(grad_x((a, x)), (a + a.T) @ x, rtol=1e-14)
  with pytest.raises(ValueError, match="different sparsity pattern"):
    grad_a((sparse.csc_array(np.eye(4)), x))


def test_multipliers_of_a_sparse_output_are_sparse_too() -> None:
  """``lagrangian_hessian`` relabels the output tree, so the weight of a sparse output is a matrix of its pattern."""

  @sc.function(sc.L("x", 3), sc.G(sc.L("s", ...), S("M", np.eye(3, dtype=bool))), name="sg_lag")
  def f(x):
    return (x * x).sum(), SparseMatrix.diag(x * x * x)

  h = sc.lagrangian_hessian(f, "x")
  x = np.array([0.5, -1.0, 2.0])
  lam_m = sparse.csc_array(np.diag([1.0, 2.0, 3.0]))
  np.testing.assert_allclose(h((x, (np.array(1.0), lam_m))), 2 * np.eye(3) + np.diag(6 * x * [1.0, 2.0, 3.0]), rtol=1e-14)
  with pytest.raises(ValueError, match="expected a SciPy sparse matrix"):
    h((x, (np.array(1.0), np.ones(3))))  # ty: ignore[no-matching-overload]


def test_declaration_errors() -> None:
  with pytest.raises(ValueError, match="non-empty name"):
    S("", MASK)
  with pytest.raises(TypeError, match="inferred"):
    S("A", ...).symbols()
  with pytest.raises(TypeError, match="inferred pattern"):
    S("A", ...).unflatten((np.zeros(2),))
  with pytest.raises(ValueError, match="stores 8 values"):
    S("A", MASK).with_types((sc.TensorType((3,)),))


def test_patterns_are_inferred_inside_groups() -> None:
  @sc.function(sc.L("x", 3), sc.G(sc.L("s", ...), S("M", ...)), name="sg_group_infer")
  def f(x):
    return x.sum(), SparseMatrix.diag(x)

  assert f.output_sparsities == (None, SparseMatrix.symbol("d", np.eye(3, dtype=bool)).sparsity)
  s, m = f(np.array([1.0, 2.0, 3.0]))
  assert isinstance(m, sparse.csc_array)
  np.testing.assert_allclose(m.toarray(), np.diag([1.0, 2.0, 3.0]))


def test_symbols_honor_diff() -> None:
  assert S("A", MASK).symbols().values.type.diff
  assert not S("A", MASK).symbols(diff=False).values.type.diff


def test_a_returned_matrix_shares_no_structure_with_the_declaration() -> None:
  """Editing a result's index arrays in place, as SciPy's own in-place methods do, leaves the next call alone."""
  build = _kkt_builder("sg_fresh", ...)
  q, a = RNG.uniform(1.0, 2.0, 4), _matrix(MASK[:2])
  first = build((q, a))
  want = first.toarray()
  first.indices[:] = 0
  first.indptr[:] = 0
  np.testing.assert_array_equal(build((q, a)).toarray(), want)
  assert build.output_tree.sparsities[0] == _kkt_builder("sg_fresh_ref", ...).output_tree.sparsities[0]
