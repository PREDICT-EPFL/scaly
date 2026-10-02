"""``opt.ipm.stages``: the block-tridiagonal partition a QP's sparsity gives, and the dense arrays of its constraint matrices."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from scaly.opt.ipm import QPStructure
from scaly.opt.ipm.stages import Stages, beside, condensed_pattern, dense_blocks, pairs_in_rows, stages
from scaly.testing.qp import mpc_qp
from tests.opt.ipm.problems import ipm_inputs


def _structure(n: int, P: np.ndarray, A: np.ndarray, G: np.ndarray) -> QPStructure:
  free = np.full(n, np.inf)
  return QPStructure.from_patterns(P != 0, A != 0, G != 0, h_l=np.full(G.shape[0], -1.0), h_u=np.full(G.shape[0], 1.0), x_l=-free, x_u=free)


def _check(s: QPStructure, st: Stages) -> None:
  """``st`` holds every variable once, and the condensed matrix is block tridiagonal in its slots,
  the blocks below the diagonal zero outside their last ``c`` columns."""
  assert st.order.shape == (st.K * st.B,)
  assert sorted(st.order[st.order >= 0]) == list(range(s.n))
  slot = st.slots
  assert np.array_equal(st.order[slot], np.arange(s.n))
  coo = condensed_pattern(s).tocoo()
  row, col = slot[coo.row] // st.B, slot[coo.col] // st.B
  assert np.all(np.abs(row - col) <= 1)
  below = row == col + 1
  assert np.all(slot[coo.col[below]] % st.B >= st.B - st.c)
  assert (st.c == 0) == (st.K == 1) and st.c <= st.B


@pytest.mark.parametrize(("nx", "nu", "horizon"), [(4, 2, 10), (6, 2, 12), (3, 3, 5), (2, 5, 6), (1, 1, 4)])
@pytest.mark.parametrize("staged", [False, True], ids=["states first", "stage order"])
def test_a_multistage_problem_gives_its_stages_in_any_variable_order(nx: int, nu: int, horizon: int, staged: bool) -> None:
  """An MPC problem with its states before its inputs, as ``mpc_qp`` writes it, and with its
  variables in stage order: either way the blocks are the stages, one more than the horizon, as
  large as a state and an input, coupled through the states."""
  s, _ = ipm_inputs(mpc_qp(nx, nu, horizon))
  if staged:
    order = np.concatenate(
      [np.r_[k * nx + np.arange(nx), (horizon + 1) * nx + k * nu + np.arange(nu)] for k in range(horizon)] + [horizon * nx + np.arange(nx)]
    )
    back = np.empty(s.n, dtype=np.int64)
    back[order] = np.arange(s.n)
    P, A = np.zeros((s.n, s.n), dtype=bool), np.zeros((s.p, s.n), dtype=bool)
    P[back[s.P_rows], back[s.P_cols]] = True
    A[s.A_rows, back[s.A_cols]] = True
    s = _structure(s.n, P, A, np.zeros((0, s.n), dtype=bool))
  st = stages(s)
  _check(s, st)
  assert (st.K, st.B, st.c) == (horizon + 1, nx + nu, nx)
  assert stages(s) is st  # found once per structure


@pytest.mark.parametrize("seed", range(30))
def test_random_patterns_are_partitioned_block_tridiagonally(seed: int) -> None:
  """Whatever the pattern, the level sets are a valid partition: banded problems, problems in
  several unconnected parts, variables no row reads, a dense row."""
  rng = np.random.default_rng(seed)
  n = int(rng.integers(1, 30))
  width = int(rng.integers(1, 5))
  P = np.diag(rng.integers(0, 2, n)).astype(float)
  rows = int(rng.integers(0, 2 * n))
  A = np.zeros((rows, n))
  for i in range(rows):
    first = int(rng.integers(0, n))
    A[i, first : first + int(rng.integers(1, width + 1))] = 1.0
  if seed % 5 == 0 and n > 2:
    A = np.vstack([A, np.ones((1, n))])  # a row that couples everything: one block
  if seed % 3 == 0:
    A[:, rng.integers(0, n)] = 0.0  # a variable only the diagonal reads
  G = np.zeros((int(rng.integers(0, 3)), n))
  for i in range(G.shape[0]):
    G[i, rng.choice(n, min(2, n), replace=False)] = 1.0
  perm = rng.permutation(n)
  s = _structure(n, P[np.ix_(perm, perm)], A[:, perm], G[:, perm])
  st = stages(s)
  _check(s, st)
  if seed % 5 == 0 and n > 2:
    assert st.K == 1 and st.B == n


def test_unconnected_parts_are_laid_beside_each_other_where_the_largest_block_stays_smallest() -> None:
  """Three chains of variables no row connects: of five variables in a line, of two, and one on its
  own. The longest gives the blocks, and each other part's levels join the ones with the most
  room, so no block grows past the size the first part gave it and none is added."""
  n = 8
  A = np.zeros((5, n))
  for i, (a, b) in enumerate([(0, 1), (1, 2), (2, 3), (3, 4), (5, 6)]):
    A[i, [a, b]] = 1.0
  P = np.eye(n)
  s = _structure(n, P, A, np.zeros((0, n)))
  st = stages(s)
  _check(s, st)
  assert (st.K, st.B) == (5, 2)
  block = st.slots // st.B
  assert abs(block[5] - block[6]) == 1 and len({block[k] for k in range(5)}) == 5
  # Parts given as levels: one variable joins the first level, where the largest stays three, and
  # two then join the second, the only place that keeps it three.
  parts = [[np.array([0]), np.array([1]), np.array([2, 3, 4])], [np.array([5])], [np.array([6, 7])]]
  assert [level.size for level in beside(parts)] == [2, 3, 3]


def test_the_cells_of_the_block_storage_are_the_lower_triangles_and_the_coupling_columns() -> None:
  st = Stages(3, 4, 2, np.arange(12))
  assert st.cells == 3 * 16 + 2 * 4 * 2
  # Inside a block: row and column in either order give the lower triangle's cell.
  assert st.cell(np.array([5]), np.array([6]))[0] == st.cell(np.array([6]), np.array([5]))[0] == 16 + 2 * 4 + 1
  # Between neighbouring blocks: the row of the later block, the coupling column of the earlier.
  assert st.cell(np.array([3]), np.array([5]))[0] == st.cell(np.array([5]), np.array([3]))[0] == 48 + 1 * 2 + 1
  with pytest.raises(AssertionError):
    st.cell(np.array([0]), np.array([8]))  # two blocks apart
  with pytest.raises(AssertionError):
    st.cell(np.array([1]), np.array([4]))  # not a coupling slot


def test_pairs_in_rows_are_the_products_of_a_gram_matrix_each_once() -> None:
  rng = np.random.default_rng(1)
  m = sparse.random_array((7, 9), density=0.4, rng=rng, format="coo")
  first, second = pairs_in_rows(m.row, 7)
  assert np.array_equal(m.row[first], m.row[second])
  gram = np.zeros((9, 9))
  np.add.at(gram, (np.maximum(m.col[first], m.col[second]), np.minimum(m.col[first], m.col[second])), m.data[first] * m.data[second])
  want = (m.T @ m).toarray()
  np.testing.assert_allclose(gram, np.tril(want), rtol=1e-13, atol=1e-13)
  assert pairs_in_rows(np.zeros(0, dtype=np.int64), 3)[0].size == 0


@pytest.mark.parametrize(("nx", "nu", "horizon"), [(8, 2, 6), (5, 5, 4), (12, 4, 5)])
def test_the_dense_arrays_and_the_other_entries_make_up_the_gram_matrix(nx: int, nu: int, horizon: int) -> None:
  """The dynamics rows of an MPC problem: each block's array holds the stage's ``[A B]``, the
  identity and the initial-state rows stay out, and the arrays' products, placed by ``source`` and
  ``cell``, with the products of the other pairs of entries, are the lower triangle of ``A^T A``
  in the block storage."""
  qp = mpc_qp(nx, nu, horizon)
  s, _ = ipm_inputs(qp)
  st = stages(s)
  found = dense_blocks(st, s.A_rows, s.A_cols, s.p)
  assert found is not None
  K, r, nd = found.entries.shape
  assert (K, r, nd) == (horizon + 1, nx, nx + nu)
  per_row = np.bincount(s.A_rows[found.taken], minlength=s.p)
  assert set(per_row) <= {0, nx + nu} and (per_row == 0).sum() == nx  # the initial-state rows are in no array
  rng = np.random.default_rng(nx)
  values = rng.normal(size=s.A_rows.size)
  arrays = np.where(found.entries >= 0, np.r_[values, 0.0][found.entries], 0.0)
  cells = np.zeros(st.cells)
  products = np.einsum("kri,krj->kij", arrays, arrays).reshape(-1)
  np.add.at(cells, found.cell, products[found.source])
  first, second = pairs_in_rows(s.A_rows, s.p)
  apart = ~(found.taken[first] & found.taken[second])
  slot = st.slots[s.A_cols]
  np.add.at(cells, st.cell(slot[first[apart]], slot[second[apart]]), values[first[apart]] * values[second[apart]])
  a = sparse.csr_array((values, (s.A_rows, s.A_cols)), shape=(s.p, s.n))
  gram = (a.T @ a).toarray()
  want = np.zeros(st.cells)
  i, j = np.nonzero(np.tril(gram[np.ix_(st.order.clip(0), st.order.clip(0))] * np.outer(st.order >= 0, st.order >= 0)))
  want[st.cell(i, j)] = gram[st.order[i], st.order[j]]
  np.testing.assert_allclose(cells, want, rtol=1e-12, atol=1e-12)
  # The rows' places: each row taken is in one array, at the block of its last variable.
  assert sorted(found.rows[found.rows >= 0]) == sorted(np.flatnonzero(per_row))


def test_a_matrix_with_no_column_mostly_filled_has_no_dense_arrays() -> None:
  """Bounds on single variables and rows of two entries: every product stays in the index tables."""
  n = 12
  A = np.zeros((n - 1, n))
  A[np.arange(n - 1), np.arange(n - 1)] = 1.0
  A[np.arange(n - 1), np.arange(1, n)] = -1.0
  s = _structure(n, np.eye(n), A, np.zeros((0, n)))
  st = stages(s)
  _check(s, st)
  found = dense_blocks(st, s.A_rows, s.A_cols, s.p)
  assert found is None or found.entries.shape[2] <= 2
  assert dense_blocks(st, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), 0) is None
