"""The symbolic analysis of sparse factorizations against a dense boolean elimination, SciPy's
reverse Cuthill–McKee ordering and SuperLU's minimum-degree fill."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import splu

from scaly.ir.types import CanonicalPattern, SparsityPattern
from scaly.linalg.symbolic import ORDERINGS, Ordering, SymbolicAnalysis, analyze


def _pattern(mask: np.ndarray) -> CanonicalPattern:
  return SparsityPattern.from_mask(mask).canonicalize()[0]


def _random(n: int, density: float, seed: int) -> np.ndarray:
  rng = np.random.default_rng(seed)
  m = rng.random((n, n)) < density
  return np.tril(m | m.T) | np.eye(n, dtype=bool)


def _banded(n: int, bandwidth: int) -> np.ndarray:
  i, j = np.indices((n, n))
  return (i >= j) & (i - j <= bandwidth)


def _arrow(n: int, *, hub_first: bool) -> np.ndarray:
  m = np.eye(n, dtype=bool)
  hub = 0 if hub_first else n - 1
  m[hub, :] = m[:, hub] = True
  return np.tril(m)


def _multistage(stages: int, nx: int, nu: int, seed: int) -> np.ndarray:
  """A block-tridiagonal OCP Hessian-and-dynamics pattern, stage by stage: dense stage blocks over
  ``(u_k, x_k)`` coupled to the next stage's states through a sparse dynamics block."""
  rng = np.random.default_rng(seed)
  size = nx + nu
  n = stages * size
  m = np.zeros((n, n), dtype=bool)
  for k in range(stages):
    s = slice(k * size, (k + 1) * size)
    m[s, s] = True
    if k + 1 < stages:
      coupling = rng.random((nx, size)) < 0.6
      m[(k + 1) * size + nu : (k + 2) * size, s] = coupling
  return np.tril(m | m.T)


def _grid(k: int) -> np.ndarray:
  n = k * k
  m = np.eye(n, dtype=bool)
  for i in range(n):
    if (i + 1) % k:
      m[i + 1, i] = True
    if i + k < n:
      m[i + k, i] = True
  return m


PATTERNS: dict[str, Callable[[], np.ndarray]] = {
  "random sparse": lambda: _random(30, 0.08, 1),
  "random denser": lambda: _random(25, 0.25, 2),
  "banded": lambda: _banded(20, 3),
  "tridiagonal": lambda: _banded(15, 1),
  "arrow, hub first": lambda: _arrow(12, hub_first=True),
  "arrow, hub last": lambda: _arrow(12, hub_first=False),
  "multistage": lambda: _multistage(5, 4, 2, 3),
  "grid": lambda: _grid(6),
  "dense": lambda: np.tril(np.ones((7, 7), dtype=bool)),
  "diagonal": lambda: np.eye(9, dtype=bool),
  "two blocks": lambda: np.tril(sparse.block_diag([sparse.coo_array(np.ones((4, 4))), _random(5, 0.4, 4)]).toarray() != 0),
  "single entry": lambda: np.ones((1, 1), dtype=bool),
}


def _symmetrized(mask: np.ndarray) -> np.ndarray:
  return mask | mask.T | np.eye(mask.shape[0], dtype=bool)


def _dense_elimination(mask: np.ndarray, perm: tuple[int, ...]) -> np.ndarray:
  """The strictly lower pattern of L, by eliminating the permuted symmetric pattern column by column."""
  m = _symmetrized(mask)[np.ix_(perm, perm)]
  for k in range(m.shape[0]):
    below = np.flatnonzero(m[k + 1 :, k]) + k + 1
    m[np.ix_(below, below)] = True
  return np.tril(m, -1)


def _columns(ptr: tuple[int, ...], rows: tuple[int, ...]) -> list[tuple[int, ...]]:
  return [rows[ptr[j] : ptr[j + 1]] for j in range(len(ptr) - 1)]


@pytest.mark.parametrize("ordering", ORDERINGS)
@pytest.mark.parametrize("name", list(PATTERNS))
def test_pattern_of_l_and_elimination_tree_match_a_dense_elimination(name: str, ordering: Ordering) -> None:
  mask = PATTERNS[name]()
  n = mask.shape[0]
  s = analyze(_pattern(mask), ordering)
  assert s.n == n
  assert sorted(s.perm) == list(range(n)) and [s.perm[s.iperm[i]] for i in range(n)] == list(range(n))

  ref = _dense_elimination(mask, s.perm)
  got = np.zeros((n, n), dtype=bool)
  for j, col in enumerate(_columns(s.l_ptr, s.l_rows)):
    assert list(col) == sorted(col) and all(i > j for i in col)
    got[list(col), j] = True
    # the parent of j is the first row below the diagonal in column j of L
    assert s.parent[j] == (col[0] if col else -1)
  np.testing.assert_array_equal(got, ref)
  assert s.col_counts == tuple(int(c) for c in ref.sum(axis=0))
  assert s.nnz_l == int(ref.sum())

  # a postorder: a permutation listing every node after all of its descendants, subtrees contiguous
  assert sorted(s.postorder) == list(range(n))
  position = {node: k for k, node in enumerate(s.postorder)}
  for j in range(n):
    if s.parent[j] >= 0:
      assert position[j] < position[s.parent[j]]
  size = [1] * n
  for j in s.postorder:
    if s.parent[j] >= 0:
      size[s.parent[j]] += size[j]
  for j in range(n):
    assert set(s.postorder[position[j] - size[j] + 1 : position[j] + 1]) == {d for d in range(n) if _is_descendant(s.parent, d, j)}


def _is_descendant(parent: tuple[int, ...], node: int, ancestor: int) -> bool:
  while node != -1:
    if node == ancestor:
      return True
    node = parent[node]
  return False


@pytest.mark.parametrize("ordering", ORDERINGS)
@pytest.mark.parametrize("name", ["random denser", "multistage", "arrow, hub last"])
def test_permuted_pattern_reads_the_lower_stored_entry_of_each_pair(name: str, ordering: Ordering) -> None:
  lower = PATTERNS[name]()
  # store the lower triangle and part of the upper one, so some entries are stored twice
  rng = np.random.default_rng(7)
  mask = lower | (lower.T & (rng.random(lower.shape) < 0.5))
  pattern = _pattern(mask)
  s = analyze(pattern, ordering)
  n = mask.shape[0]
  permuted = np.tril((mask | mask.T)[np.ix_(s.perm, s.perm)])
  assert len(s.a_ptr) == n + 1 and s.a_ptr[-1] == len(s.a_rows) == len(s.a_source) == int(permuted.sum())
  for j, col in enumerate(_columns(s.a_ptr, s.a_rows)):
    assert list(col) == sorted(col)
    assert col == tuple(np.flatnonzero(permuted[:, j]).tolist())
  for k, (i, j) in enumerate((i, j) for j, col in enumerate(_columns(s.a_ptr, s.a_rows)) for i in col):
    r, c = pattern.rows[s.a_source[k]], pattern.cols[s.a_source[k]]
    assert {r, c} == {s.perm[i], s.perm[j]}
    assert r >= c or not mask[c, r]


def test_upper_lower_and_full_storage_give_one_factor_pattern() -> None:
  lower = _multistage(4, 3, 2, 5)
  analyses = [analyze(_pattern(m), "amd") for m in (lower, lower.T, lower | lower.T)]
  assert len({(s.perm, s.parent, s.l_ptr, s.l_rows) for s in analyses}) == 1


@pytest.mark.parametrize("name", list(PATTERNS))
def test_rcm_is_scipys_reverse_cuthill_mckee(name: str) -> None:
  mask = PATTERNS[name]()
  graph = sparse.csr_array((mask | mask.T).astype(np.float64))
  expected = tuple(int(i) for i in reverse_cuthill_mckee(graph, symmetric_mode=True))
  assert analyze(_pattern(mask), "rcm").perm == expected


def _splu_strict_lower_fill(mask: np.ndarray) -> int:
  """Entries of L below the diagonal in SuperLU's factorization of a diagonally dominant matrix with
  this pattern, ordered by its minimum degree on ``A + A^T``."""
  sym = _symmetrized(mask)
  n = sym.shape[0]
  a = sparse.csc_array(np.where(sym, -1.0, 0.0) + np.eye(n) * (n + 1.0))
  lu = splu(a, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options={"SymmetricMode": True})
  return int(lu.L.nnz) - n


@pytest.mark.parametrize(
  ("name", "mask"),
  [
    ("grid", _grid(15)),
    ("larger grid", _grid(40)),
    ("random", _random(200, 0.015, 8)),
    ("multistage", _multistage(20, 6, 3, 9)),
    ("banded", _banded(150, 4)),
    ("arrow", _arrow(80, hub_first=True)),
  ],
)
def test_amd_fill_is_within_superlus_minimum_degree_fill(name: str, mask: np.ndarray) -> None:
  # a sanity bound, not a match: the two minimum-degree variants break ties differently
  amd, superlu = analyze(_pattern(mask), "amd").nnz_l, _splu_strict_lower_fill(mask)
  assert amd <= 1.1 * superlu, (name, amd, superlu)


def test_amd_leaves_the_hub_of_an_arrow_for_last() -> None:
  # the hub ties with the last leaf at degree 1, either order fills nothing
  s = analyze(_pattern(_arrow(40, hub_first=True)), "amd")
  assert 0 in s.perm[-2:] and s.nnz_l == 39
  assert analyze(_pattern(_arrow(40, hub_first=True)), "natural").nnz_l == 40 * 39 // 2


def test_natural_ordering_of_an_arrow_matches_davis_ldl_symbolic() -> None:
  # column 0 couples to every row, so eliminating it first fills the whole trailing block
  s = analyze(_pattern(_arrow(5, hub_first=True)), "natural")
  assert s.perm == (0, 1, 2, 3, 4)
  assert s.parent == (1, 2, 3, 4, -1)
  assert s.l_ptr == (0, 4, 7, 9, 10, 10)
  # a diagonal matrix needs no factor entries at all
  s = analyze(_pattern(np.eye(5, dtype=bool)), "natural")
  assert s.parent == (-1,) * 5 and s.l_ptr == (0,) * 6


def test_equal_patterns_share_one_immutable_analysis() -> None:
  mask = _multistage(3, 3, 1, 10)
  s = analyze(_pattern(mask), "amd")
  assert analyze(_pattern(mask.copy()), "amd") is s
  assert isinstance(s.digest, str) and len(s.digest) == 64
  assert analyze(_pattern(mask), "natural").digest != s.digest
  assert analyze(_pattern(mask | np.tri(*mask.shape, -2, dtype=bool)), "amd").digest != s.digest
  with pytest.raises(dataclasses.FrozenInstanceError):
    s.perm = ()  # ty: ignore[invalid-assignment]
  assert all(isinstance(getattr(s, f.name), tuple) for f in dataclasses.fields(SymbolicAnalysis) if f.name not in ("n", "digest"))


def test_orderings_with_equal_tables_give_the_same_analysis() -> None:
  # a dense matrix has one structure whatever the order of RCM's breadth-first search
  mask = np.tril(np.ones((4, 4), dtype=bool))
  assert analyze(_pattern(mask), "natural") is analyze(_pattern(mask), "amd")


def test_refuses_a_rectangular_pattern_and_an_unknown_ordering() -> None:
  with pytest.raises(ValueError, match="square"):
    analyze(_pattern(np.ones((2, 3), dtype=bool)))
  with pytest.raises(ValueError, match="ordering"):
    analyze(_pattern(np.eye(3, dtype=bool)), "mmd")  # ty: ignore[invalid-argument-type]
