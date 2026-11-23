"""Sparse ``cholesky``, ``ldl`` and ``solve(assume="pos"|"sym")`` against SciPy on random, banded,
arrow and block-tridiagonal patterns under each ordering, the pivot status, and the derivatives of
the solves against finite differences and against the dense solve of the same matrix."""

from __future__ import annotations

from typing import Any, cast

import re

import numpy as np
import pytest
import scipy.linalg
from scipy import sparse
from scipy.sparse.linalg import spsolve

import scaly as sc
from scaly import linalg
from scaly.codegen import render_c_source
from scaly.passes.lowering import calls
from scaly.function.model import as_concrete
from scaly.ir.expr import ExprOp, topo
from scaly.ir.program import ProgramOp, walk_program
from scaly.ir.types import CanonicalPattern
from scaly.linalg import symbolic
from scaly.linalg.sparse import _factorization
from scaly.linalg.symbolic import ORDERINGS, Ordering
from scaly.passes.lowering import lower_function

from .test_symbolic import _arrow, _banded, _multistage, _random

STATUS = sc.TensorType((), sc.dtypes.int32)

PATTERNS = {
  "random": lambda: _random(18, 0.15, 5),
  "banded": lambda: _banded(14, 2),
  "arrow, hub first": lambda: _arrow(10, hub_first=True),
  "arrow, hub last": lambda: _arrow(10, hub_first=False),
  "block-tridiagonal": lambda: _multistage(4, 3, 2, 6),
}


def _values(mask: np.ndarray, rng: np.random.Generator, *, negative: int = 0) -> np.ndarray:
  """A matrix on the lower-triangular ``mask``, diagonally dominant enough that no elimination
  pivots, with the last ``negative`` diagonal entries negative: positive definite without them,
  quasi-definite with them."""
  n = mask.shape[0]
  a = np.where(mask, rng.normal(size=(n, n)), 0.0)
  a[np.diag_indices(n)] = 0.0
  full = a + a.T
  dominance = 2.0 * np.abs(full).sum(axis=1) + 1.0
  a[np.diag_indices(n)] = np.where(np.arange(n) < n - negative, dominance, -dominance)
  return a


def _symmetric(a: np.ndarray) -> np.ndarray:
  return np.tril(a) + np.tril(a, -1).T


def _pattern(a: np.ndarray) -> CanonicalPattern:
  """The pattern of the nonzero entries of ``a``, which a CSC array of it stores."""
  return sc.SparsityPattern.from_mask(a != 0).canonicalize()[0]


def _sparse_type(a: np.ndarray) -> sc.TensorType:
  return sc.TensorType(a.shape, pattern=_pattern(a))


def _factor_and_solve(a: np.ndarray, ordering: Ordering):
  n = a.shape[0]
  K_type = _sparse_type(a)
  outputs = sc.group(
    sc.group(sc.arg("L_pos"), sc.arg("status_pos", STATUS)),
    sc.group(sc.arg("L_sym"), sc.arg("d", n), sc.arg("status_sym", STATUS)),
    sc.group(sc.arg("x_pos", (n, 2)), sc.arg("x_sym", n)),
  )

  @sc.function(sc.arg("K", K_type), sc.arg("b", (n, 2)), outputs=outputs)
  def run(K, b):
    return (
      linalg.cholesky(K, ordering=ordering),
      linalg.ldl(K, ordering=ordering),
      (linalg.solve(K, b, assume="pos", ordering=ordering)[0], linalg.solve(K, b[:, 0], assume="sym", ordering=ordering)[0]),
    )

  return run


@pytest.mark.parametrize("ordering", ORDERINGS)
@pytest.mark.parametrize("name", list(PATTERNS))
def test_factors_and_solves_match_scipy(name: str, ordering: Ordering) -> None:
  rng = np.random.default_rng(1)
  mask = PATTERNS[name]()
  n = mask.shape[0]
  a = _values(mask, rng)
  K = sparse.csc_array(a)
  b = rng.normal(size=(n, 2))
  run = _factor_and_solve(a, ordering)
  analysis = symbolic.ldl(_pattern(a), ordering)
  perm = np.asarray(analysis.perm)
  permuted = _symmetric(a)[np.ix_(perm, perm)]

  (L, status), (L_unit, d, status_sym), (x_pos, x_sym) = run(K, b)
  assert status == 0 and status_sym == 0
  # The factor stores the analysis' pattern of L, with its diagonal, in the reordered numbering.
  expected_pattern = np.zeros((n, n), dtype=bool)
  expected_pattern[np.asarray(analysis.l_rows), np.repeat(np.arange(n), np.diff(analysis.l_ptr))] = True
  expected_pattern[np.diag_indices(n)] = True
  for factor in (L, L_unit):
    stored = np.zeros((n, n), dtype=bool)
    coo = factor.tocoo()
    stored[coo.row, coo.col] = True
    np.testing.assert_array_equal(stored, expected_pattern)

  np.testing.assert_allclose(L.toarray(), scipy.linalg.cholesky(permuted, lower=True), rtol=1e-12, atol=1e-13)
  expected_l, expected_d, ldl_perm = scipy.linalg.ldl(permuted, lower=True)
  np.testing.assert_array_equal(ldl_perm, np.arange(n))  # dominant enough that SciPy does not pivot either
  np.testing.assert_allclose(L_unit.toarray(), expected_l, rtol=1e-12, atol=1e-13)
  np.testing.assert_allclose(np.diag(d), expected_d, rtol=1e-12, atol=1e-13)
  np.testing.assert_allclose(x_pos, spsolve(sparse.csc_array(_symmetric(a)), b), rtol=1e-12, atol=1e-13)
  np.testing.assert_allclose(x_sym, spsolve(sparse.csc_array(_symmetric(a)), b[:, 0]), rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("ordering", ORDERINGS)
@pytest.mark.parametrize("name", list(PATTERNS))
def test_quasi_definite_solves_match_scipy(name: str, ordering: Ordering) -> None:
  rng = np.random.default_rng(2)
  mask = PATTERNS[name]()
  n = mask.shape[0]
  a = _values(mask, rng, negative=n // 3)
  K = sparse.csc_array(a)
  K_type = _sparse_type(a)

  @sc.function(sc.arg("K", K_type), sc.arg("b", n), outputs=sc.group(sc.arg("x", n), sc.arg("d", n), sc.arg("status", STATUS)))
  def run(K, b):
    x, status = linalg.solve(K, b, assume="sym", ordering=ordering)
    return x, linalg.ldl(K, ordering=ordering)[1], status

  b = rng.normal(size=n)
  x, d, status = run(K, b)
  assert status == 0
  assert (d < 0).sum() == n // 3  # Sylvester's law of inertia
  np.testing.assert_allclose(x, spsolve(sparse.csc_array(_symmetric(a)), b), rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("stored", ["upper", "both"])
def test_either_triangle_stands_for_its_mirror(stored: str) -> None:
  """An entry stored only above the diagonal stands for the one below, and of a pair stored in both
  triangles the lower one is read, whatever the upper one holds."""
  rng = np.random.default_rng(3)
  a = _values(PATTERNS["random"](), rng)
  n = a.shape[0]
  if stored == "upper":
    other = a.T
  else:
    garbage = np.where(np.tril(a, -1).T != 0, rng.normal(size=(n, n)), 0.0)
    other = a + garbage
  b = rng.normal(size=n)

  def solve(m: np.ndarray) -> np.ndarray:
    K = sparse.csc_array(m)

    @sc.function(sc.arg("K", _sparse_type(m)), sc.arg("b", n), outputs=sc.arg("x", n))
    def run(K, b):
      return linalg.solve(K, b, assume="sym")[0]

    return run(K, b)

  np.testing.assert_allclose(solve(other), solve(a), rtol=1e-13, atol=1e-14)


def _status_of(m: np.ndarray, routine: str, ordering: Ordering) -> int:
  @sc.function(sc.arg("K", _sparse_type(m)), outputs=sc.arg("status", STATUS))
  def run(K):
    return getattr(linalg, routine)(K, ordering=ordering)[-1]

  return int(np.asarray(run(sparse.csc_array(m))))


@pytest.mark.parametrize("ordering", ORDERINGS)
@pytest.mark.parametrize("routine", ["cholesky", "ldl"])
def test_an_unusable_pivot_sets_the_status_in_the_reordered_numbering(routine: str, ordering: Ordering) -> None:
  rng = np.random.default_rng(4)
  mask = PATTERNS["arrow, hub last"]()
  a = _values(mask, rng)
  n = a.shape[0]
  iperm = symbolic.ldl(_pattern(a), ordering).iperm
  assert _status_of(a, routine, ordering) == 0

  # Column 3 touches only the hub, so its pivot is its own diagonal entry wherever the ordering puts it.
  zero = a.copy()
  zero[3, 3] = 0.0
  assert _status_of(zero, routine, ordering) == iperm[3] + 1

  # Cholesky refuses a negative pivot, which LDL^T accepts.
  negative = a.copy()
  negative[3, 3] = -1.0
  assert _status_of(negative, routine, ordering) == (iperm[3] + 1 if routine == "cholesky" else 0)

  # Two unusable pivots: the status names the first in the reordered numbering.
  both = zero.copy()
  both[5, 5] = 0.0
  assert _status_of(both, routine, ordering) == min(iperm[3], iperm[5]) + 1

  # A diagonal entry the pattern does not store is a zero pivot.
  unstored = np.where(np.eye(n, dtype=bool) & (np.arange(n) == 4)[:, None], 0.0, a)
  assert _pattern(unstored).nnz == _pattern(a).nnz - 1
  assert _status_of(unstored, routine, ordering) == iperm[4] + 1


def test_float32_matrices_factor_and_solve_in_float32() -> None:
  rng = np.random.default_rng(5)
  a = _values(PATTERNS["banded"](), rng)
  n = a.shape[0]
  K = sparse.csc_array(a.astype(np.float32))
  K_type = sc.TensorType((n, n), dtype=sc.dtypes.float32, pattern=_pattern(a))

  @sc.function(
    sc.arg("K", K_type), sc.arg("b", sc.TensorType((n,), sc.dtypes.float32)), outputs=sc.group(sc.arg("x"), sc.arg("L"), sc.arg("status", STATUS))
  )
  def run(K, b):
    x, status = linalg.solve(K, b, assume="pos", ordering="rcm")
    return x, linalg.cholesky(K, ordering="rcm")[0], status

  b = rng.normal(size=n).astype(np.float32)
  x, L, status = cast(tuple[np.ndarray, Any, np.ndarray], run(K, b))
  assert status == 0 and x.dtype == np.float32 and L.dtype == np.float32
  np.testing.assert_allclose(x, np.linalg.solve(_symmetric(a), b), rtol=1e-5, atol=1e-6)


# Derivatives: the matrix's stored values and the right-hand side packed in one input, so that both
# are active, against central differences and against the dense solve of the same matrix.

DERIVATIVE_MASK = _random(7, 0.3, 7)


def _point(negative: int) -> tuple[CanonicalPattern, np.ndarray]:
  rng = np.random.default_rng(8)
  a = _values(DERIVATIVE_MASK, rng, negative=negative)
  pattern = _pattern(a)
  stored = a[np.asarray(pattern.rows), np.asarray(pattern.cols)]
  return pattern, np.concatenate([stored, rng.normal(size=a.shape[0])])


def _cost(x: sc.Expr) -> sc.Expr:
  return (x * x * x).sum() / 3.0 + x[0] * x[-1]


def _cost_function(pattern: CanonicalPattern, assume: str, ordering: Ordering | None):
  n, nnz = pattern.shape[0], pattern.nnz

  @sc.function(sc.arg("p", nnz + n), outputs=sc.arg("c", ()))
  def cost(p):
    K = sc.pack(p[:nnz], pattern)
    if ordering is None:
      x = linalg.solve(K.to_dense(), p[nnz:], assume=assume)[0]
    else:
      x = linalg.solve(K, p[nnz:], assume=assume, ordering=ordering)[0]
    return _cost(x)

  return cost


def _central(f, p: np.ndarray, h: float) -> np.ndarray:
  steps = np.eye(p.size) * h
  return np.stack([(np.asarray(f(p + e)) - np.asarray(f(p - e))) / (2 * h) for e in steps], axis=-1)


@pytest.mark.parametrize("ordering", ["amd", "natural"])
@pytest.mark.parametrize(("assume", "negative"), [("pos", 0), ("sym", 2)])
def test_solve_derivatives_match_finite_differences(assume: str, negative: int, ordering: Ordering) -> None:
  pattern, p = _point(negative)
  cost = _cost_function(pattern, assume, ordering)
  gradient = sc.gradient(cost, "c", "p")
  forward = sc.jacobian(cost, "c", "p")
  hessian = sc.hessian(cost, "c", "p")

  g = gradient(p)
  np.testing.assert_allclose(g, _central(cost, p, 1e-6), rtol=1e-6, atol=1e-8)
  np.testing.assert_allclose(np.asarray(forward(p)).reshape(-1), g, rtol=1e-12, atol=1e-13)
  np.testing.assert_allclose(hessian(p), _central(gradient, p, 1e-6), rtol=1e-6, atol=1e-8)

  # The dense solve reads the same lower triangle, so it is the same function of p.
  dense = _cost_function(pattern, assume, None)
  np.testing.assert_allclose(g, sc.gradient(dense, "c", "p")(p), rtol=1e-11, atol=1e-13)
  np.testing.assert_allclose(hessian(p), sc.hessian(dense, "c", "p")(p), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("routine", ["cholesky", "ldl"])
def test_the_factor_outputs_are_declared_nondifferentiable(routine: str) -> None:
  """As the dense factors are, so that #477's check reaches them: a derivative goes through ``solve``."""
  a = _values(PATTERNS["random"](), np.random.default_rng(12))
  factor = _factorization(routine, "amd", _sparse_type(a))
  assert factor.outputs is not None
  *factors, status = factor.outputs.decls
  assert factors and all(type_ is not None and not type_.diff for type_ in factors)
  assert status is not None and status.dtype == sc.dtypes.int32


@pytest.mark.parametrize("assume", ["pos", "sym"])
def test_reverse_over_reverse_matches_forward_over_reverse(assume: str) -> None:
  pattern, p = _point(0 if assume == "pos" else 2)
  cost = _cost_function(pattern, assume, "amd")
  gradient = sc.gradient(cost, "c", "p")
  weights = np.random.default_rng(9).normal(size=p.size)

  @sc.function(sc.arg("p", p.size), outputs=sc.arg("g", p.size))
  def twice(p):
    return sc.gradient((gradient(p) * sc.const(weights)).sum(), p)

  np.testing.assert_allclose(twice(p), sc.hessian(cost, "c", "p")(p) @ weights, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(("assume", "kind"), [("pos", "cholesky"), ("sym", "ldl")])
def test_a_value_and_its_gradient_share_one_factorization(assume: str, kind: str) -> None:
  pattern, p = _point(0)
  n = pattern.shape[0]

  @sc.function(sc.arg("K", sc.TensorType((n, n), pattern=pattern)), sc.arg("b", n))
  def value_and_gradient(K, b):
    value = _cost(linalg.solve(K, b, assume=assume)[0])
    return value, *sc.vjp((value,), (K, b), (sc.const(1.0),))

  found, todo, factorizations = [], [as_concrete(value_and_gradient)], set()
  while todo:
    graph = todo.pop()
    if graph in found:
      continue
    found.append(graph)
    for node in topo(graph.results):
      if node.op in (ExprOp.CALL, ExprOp.LOOP):
        todo.append(node.attrs["callee"])
        if node.attrs["callee"].name.startswith(f"sparse_{kind}_column"):
          factorizations.add((graph.name, node.args))
  assert len(factorizations) == 1


def test_ordering_applies_to_sparse_matrices_only() -> None:
  K = _values(PATTERNS["banded"](), np.random.default_rng(10))
  n = K.shape[0]
  ones = sc.const(np.ones(n))
  cases = [
    ((n, n), lambda A: linalg.cholesky(A, ordering="amd")[0], TypeError, "ordering applies to a sparse matrix"),
    ((n, n), lambda A: linalg.solve(A, ones, assume="pos", ordering="natural")[0], TypeError, "ordering applies to a sparse matrix"),
    (
      _sparse_type(K),
      lambda K: linalg.ldl(K, ordering="metis")[0],  # ty: ignore[invalid-argument-type]
      ValueError,
      "ordering must be one of",
    ),
    (_sparse_type(K), lambda K: linalg.solve(K, ones, assume="gen")[0], ValueError, "assume 'pos' or 'sym'"),
    (_sparse_type(K), lambda K: linalg.solve(K, K, assume="pos")[0], TypeError, "dense right-hand side"),
    (_sparse_type(K), lambda K: linalg.solve(K, sc.const(np.ones(n + 1)), assume="pos")[0], ValueError, "right-hand side has shape"),
  ]
  for type_, body, error, message in cases:
    with pytest.raises(error, match=message):
      as_concrete(sc.function(sc.arg("A", type_))(body))


@pytest.mark.parametrize(("assume", "kind"), [("pos", "cholesky"), ("sym", "ldl")])
def test_every_loop_updates_its_carries_in_place(assume: str, kind: str, monkeypatch) -> None:
  """The loop structure of a left-looking factorization: one step per column that loads the column,
  updates it from the earlier ones and stores it, every carry in place, so no step copies a vector."""
  # The table-driven solve loops of a small pattern unroll; a zero budget keeps the loops this checks.
  monkeypatch.setattr(calls, "STATIC_LOOP_BUDGET", 0)
  a = _values(PATTERNS["block-tridiagonal"](), np.random.default_rng(11))
  n = a.shape[0]

  @sc.function(sc.arg("K", _sparse_type(a)), sc.arg("b", n), outputs=sc.arg("x", n))
  def solve(K, b):
    return linalg.solve(K, b, assume=assume)[0]

  loops = {
    "sparse_factor_load",
    f"sparse_{kind}_update",
    "sparse_factor_axpy",
    "sparse_factor_store",
    f"sparse_{kind}_column",
    "sparse_solve_axpy",
    f"sparse_{kind}_forward",
    f"sparse_{kind}_backward",
  }
  assert set(re.findall(r"static inline void (sparse_\w+?)_\d\w*?_inplace\w*_raw\(", render_c_source(solve))) == loops
  # Inside the procedures of those loops, no buffer holds more than one value: nothing is copied.
  for proc in lower_function(solve).args:
    if proc.op == ProgramOp.PROC and proc.attrs["name"].startswith(tuple(loops)):
      private = [node for node in walk_program(proc) if node.op == ProgramOp.BUFFER and node.attrs.get("address_space") == "private"]
      assert all(int(np.prod(node.attrs["shape"])) <= 2 for node in private), proc.attrs["name"]


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("assume", ["pos", "sym"])
def test_a_scipy_matrix_takes_any_right_hand_side(assume: str, dtype: type) -> None:
  """A SciPy matrix is converted first, so a Python ``b`` takes its dtype, and a NumPy or symbolic one is accepted."""
  K = sparse.eye_array(2, dtype=dtype)
  ones = np.ones(2, dtype=dtype)
  for b in (ones, sc.const(ones), [1, 1]):
    x, status = linalg.solve(K, b, assume=assume)
    assert x.type.dtype == sc.dtypes.from_name(np.dtype(dtype).name) and status.type.dtype == sc.dtypes.int32


def test_a_subnormal_pivot_scales_its_column_without_overflow() -> None:
  """``1 / pivot`` overflows for a subnormal pivot, so the column is divided by the pivot, as dense LDL^T does."""
  a = 1e-310 * np.array([[2.0, 0.0], [1.0, 2.0]])

  @sc.function(sc.arg("K", _sparse_type(a)), sc.arg("b", 2), outputs=sc.group(sc.arg("x", 2), sc.arg("status", STATUS)))
  def solve(K, b):
    return linalg.solve(K, b, assume="sym", ordering="natural")

  x, status = solve(sparse.csc_array(a), 1e-310 * np.ones(2))
  assert status == 0
  np.testing.assert_allclose(x, [1 / 3, 1 / 3], rtol=1e-12)
