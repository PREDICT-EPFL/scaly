"""Symbolic ``LDL^T`` analysis: checked against a dense boolean elimination, a dense numerical
factorization run through the left-looking tables, and brute-force segment selection."""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest
from scipy import sparse

from scaly.linalg.symbolic import CostModel, Segment, analyze, ordering

RNG = np.random.default_rng(505)


def _random_symmetric(n: int, density: float, seed: int) -> sparse.csr_array:
  a = sparse.random_array((n, n), density=density, random_state=seed, format="csr")
  return sparse.csr_array(abs(a) + abs(a).T + sparse.eye_array(n))


def _grid(k: int) -> sparse.csr_array:
  t = sparse.diags_array([-np.ones(k - 1), 4 * np.ones(k), -np.ones(k - 1)], offsets=[-1, 0, 1])
  e = sparse.diags_array([np.ones(k - 1), np.ones(k - 1)], offsets=[-1, 1])
  return sparse.csr_array(sparse.kron(sparse.eye_array(k), t) - sparse.kron(e, sparse.eye_array(k)))


def _arrow(n: int) -> sparse.csr_array:
  a = sparse.lil_array((n, n))
  a.setdiag(n + 1.0)
  a[0, :] = 1.0
  a[:, 0] = 1.0
  a[0, 0] = n + 1.0
  return sparse.csr_array(a)


def _coords(a: sparse.sparray, triangle: str = "lower") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  coo = sparse.coo_array(a)
  keep = {"lower": coo.row >= coo.col, "upper": coo.row <= coo.col, "full": np.ones(coo.nnz, dtype=bool)}[triangle]
  return coo.row[keep].astype(np.int64), coo.col[keep].astype(np.int64), coo.data[keep]


def _reference_pattern(a: sparse.sparray, perm: np.ndarray) -> np.ndarray:
  """Strictly lower pattern of L by boolean elimination of the permuted pattern."""
  m = (a.toarray() != 0)[np.ix_(perm, perm)]
  m = m | m.T
  n = m.shape[0]
  for k in range(n):
    below = np.flatnonzero(m[k + 1 :, k]) + k + 1
    m[np.ix_(below, below)] = True
  return np.tril(m, -1)


MATRICES = {
  "random": lambda: _random_symmetric(30, 0.08, 1),
  "grid": lambda: _grid(6),
  "arrow": lambda: _arrow(12),
  "tridiagonal": lambda: sparse.csr_array(sparse.diags_array([np.ones(19), 3 * np.ones(20), np.ones(19)], offsets=[-1, 0, 1])),
  "dense": lambda: sparse.csr_array(np.ones((7, 7)) + 7 * np.eye(7)),
  "blocks": lambda: sparse.csr_array(sparse.block_diag([_random_symmetric(5, 0.3, 2), _random_symmetric(4, 0.5, 3)])),
}


@pytest.mark.parametrize("method", ["natural", "rcm", "mmd"])
@pytest.mark.parametrize("name", list(MATRICES))
def test_pattern_tree_and_tables(name: str, method: str) -> None:
  a = MATRICES[name]()
  rows, cols, _ = _coords(a)
  s = analyze(a.shape, rows, cols, method)
  n = a.shape[0]
  assert np.array_equal(np.sort(s.perm), np.arange(n)) and np.array_equal(s.iperm[s.perm], np.arange(n))
  ref = _reference_pattern(a, s.perm)
  got = np.zeros((n, n), dtype=bool)
  got[s.l_rows, np.repeat(np.arange(n), s.col_counts)] = True
  np.testing.assert_array_equal(got, ref)
  for j in range(n):
    col = s.l_rows[s.l_ptr[j] : s.l_ptr[j + 1]]
    assert np.all(np.diff(col) > 0)
    assert s.parent[j] == (col[0] if col.size else -1)
  seen = np.zeros(n, dtype=bool)
  for j in s.postorder:  # children first
    assert all(seen[c] for c in np.flatnonzero(s.parent == j))
    seen[j] = True
  for j in range(n):  # row j of L, and where each entry sits
    ks = s.r_cols[s.r_ptr[j] : s.r_ptr[j + 1]]
    np.testing.assert_array_equal(ks, np.flatnonzero(ref[j]))
    np.testing.assert_array_equal(s.l_rows[s.r_pos[s.r_ptr[j] : s.r_ptr[j + 1]]], np.full(ks.size, j))


def _left_looking(s, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """The factorization T2-7 generates, run in NumPy on the tables alone."""
  n = s.n
  lv, d, w = np.zeros(s.nnz_l), np.zeros(n), np.zeros(n)
  for j in range(n):
    a = slice(s.a_ptr[j], s.a_ptr[j + 1])
    w[s.a_rows[a]] = values[s.a_source[a]]
    u = slice(s.u_ptr[j], s.u_ptr[j + 1])
    np.subtract.at(w, s.u_rows[u], lv[s.u_lik[u]] * d[s.u_k[u]] * lv[s.u_ljk[u]])
    d[j] = w[j]
    c = slice(s.l_ptr[j], s.l_ptr[j + 1])
    lv[c] = w[s.l_rows[c]] / d[j]
    w[s.l_rows[c]] = 0.0
    w[j] = 0.0
  return lv, d


@pytest.mark.parametrize("triangle", ["lower", "upper", "full"])
@pytest.mark.parametrize("name", ["random", "grid", "arrow", "blocks"])
def test_left_looking_tables_factor_the_permuted_matrix(name: str, triangle: str) -> None:
  a = MATRICES[name]()
  n = a.shape[0]
  signs = np.where(np.arange(n) % 3 == 2, -1.0, 1.0)  # quasi-definite: a negative block too
  k = sparse.csr_array(a.multiply(signs[:, None] * signs[None, :] > 0) * (RNG.random((n, n)) + 0.5))
  k = sparse.csr_array((k + k.T) / 2 + sparse.diags_array(signs * (n + 2.0)))
  rows, cols, vals = _coords(k, triangle)
  s = analyze(k.shape, rows, cols, "mmd")
  lv, d = _left_looking(s, vals)
  lower = np.eye(n)
  lower[s.l_rows, np.repeat(np.arange(n), s.col_counts)] = lv
  permuted = k.toarray()[np.ix_(s.perm, s.perm)]
  np.testing.assert_allclose(lower @ np.diag(d) @ lower.T, permuted, rtol=1e-12, atol=1e-12)


def test_every_triangle_gives_the_same_analysis() -> None:
  a = MATRICES["grid"]()
  runs = [analyze(a.shape, *_coords(a, t)[:2], "rcm") for t in ("lower", "upper", "full")]
  for s in runs[1:]:
    for attr in ("perm", "l_ptr", "l_rows", "u_ptr", "u_rows", "u_lik", "u_ljk", "u_k", "a_rows"):
      np.testing.assert_array_equal(getattr(s, attr), getattr(runs[0], attr))


def test_orderings_reduce_fill_and_bandwidth() -> None:
  arrow = _arrow(40)
  rows, cols, _ = _coords(arrow)
  natural, mmd = analyze(arrow.shape, rows, cols, "natural"), analyze(arrow.shape, rows, cols, "mmd")
  assert natural.nnz_l == 40 * 39 // 2 and mmd.nnz_l == 39
  grid = _grid(12)
  gr, gc, _ = _coords(grid)
  assert analyze(grid.shape, gr, gc, "mmd").nnz_l < analyze(grid.shape, gr, gc, "natural").nnz_l
  shuffled = RNG.permutation(144)
  bandwidth = lambda p: int(np.max(np.abs(np.argsort(p)[gr] - np.argsort(p)[gc])))  # noqa: E731
  assert bandwidth(ordering(144, gr, gc, "rcm")) <= bandwidth(shuffled)
  given = analyze(grid.shape, gr, gc, perm=shuffled)
  assert given.method == "given" and np.array_equal(given.perm, shuffled)


def test_stats() -> None:
  tri = MATRICES["tridiagonal"]()
  s = analyze(tri.shape, *_coords(tri)[:2], "natural")
  st = s.stats()
  assert st["nnz_l"] == 19 and st["fill"] == 0 and st["height"] == 20 and st["max_col"] == 1
  assert st["supernodes"] == 19  # a path: only the last two columns share a structure
  dense = analyze((7, 7), *_coords(MATRICES["dense"]())[:2], "natural").stats()
  assert dense["supernodes"] == 1 and dense["nnz_l"] == 21 and dense["height"] == 7
  blocks = analyze((9, 9), *_coords(MATRICES["blocks"]())[:2], "natural").stats()
  assert blocks["height"] < 9


def test_validation() -> None:
  with pytest.raises(ValueError, match="diagonal"):
    analyze((3, 3), np.array([1, 2]), np.array([0, 1]))
  with pytest.raises(ValueError, match="square"):
    analyze((3, 2), np.array([0]), np.array([0]))
  with pytest.raises(ValueError, match="permutation"):
    analyze((2, 2), np.array([0, 1]), np.array([0, 1]), perm=np.array([0, 0]))
  with pytest.raises(ValueError, match="ordering"):
    ordering(3, np.array([0]), np.array([0]), "amd")  # type: ignore[arg-type]
  arrow = _arrow(40)
  with pytest.raises(ValueError, match="over the limit"):
    analyze(arrow.shape, *_coords(arrow)[:2], "natural", max_update_lanes=1000)
  assert analyze(arrow.shape, *_coords(arrow)[:2], "mmd", max_update_lanes=1000).nnz_l == 39


def _brute_force(s, cost: CostModel) -> float:
  n, w = s.n, s.widths()
  best = np.inf
  for cuts in itertools.product([False, True], repeat=n - 1):
    bounds = [0, *[i + 1 for i, c in enumerate(cuts) if c], n]
    segs = [Segment(lo, hi, *(int(x) for x in w[lo:hi].max(axis=0))) for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)]
    best = min(best, s.padded_work(segs, cost))
  return best


@pytest.mark.parametrize("price", [0.0, 5.0, 40.0, 1e6])
def test_segments_are_optimal_and_cover_the_columns(price: float) -> None:
  a = _random_symmetric(9, 0.3, 7)
  s = analyze(a.shape, *_coords(a)[:2], "natural")
  cost = CostModel(segment=price)
  segs = s.segments(cost)
  assert segs[0].start == 0 and segs[-1].stop == 9 and all(x.stop == y.start for x, y in zip(segs[:-1], segs[1:], strict=True))
  w = s.widths()
  for seg in segs:
    assert (seg.a, seg.u, seg.c) == tuple(int(x) for x in w[seg.start : seg.stop].max(axis=0))
  assert s.padded_work(segs, cost) == pytest.approx(_brute_force(s, cost))
  if price == 1e6:
    assert len(segs) == 1


def test_chunked_segments_for_many_columns() -> None:
  grid = _grid(40)
  s = analyze(grid.shape, *_coords(grid)[:2], "mmd")
  segs = s.segments(max_chunks=64)
  assert segs[0].start == 0 and segs[-1].stop == 1600 and len(segs) <= 64
  one = [Segment(0, 1600, *(int(x) for x in s.widths().max(axis=0)))]
  assert s.padded_work(segs) <= s.padded_work(one)


def test_analysis_time_is_near_linear() -> None:
  grid = _grid(100)  # n = 10 000
  rows, cols, _ = _coords(grid)
  start = time.perf_counter()
  s = analyze(grid.shape, rows, cols, "mmd")
  s.segments()
  elapsed = time.perf_counter() - start
  assert s.nnz_l > 50_000
  assert elapsed < 10.0, f"{elapsed:.1f} s for nnz(L) = {s.nnz_l}"


def test_a_mirrored_pair_is_read_from_its_lower_entry() -> None:
  a = MATRICES["random"]()
  rows, cols, _ = _coords(a, "full")
  s = analyze(a.shape, rows, cols, "rcm")
  assert np.all(rows[s.a_source] >= cols[s.a_source])


def test_auto_ordering_keeps_the_least_work() -> None:
  for a in (_arrow(30), _grid(10), sparse.csr_array(sparse.diags_array([np.ones(29), 3 * np.ones(30), np.ones(29)], offsets=[-1, 0, 1]))):
    rows, cols, _ = _coords(a)
    auto = analyze(a.shape, rows, cols, "auto")
    each = [analyze(a.shape, rows, cols, m) for m in ("natural", "rcm", "mmd")]
    assert auto.method.startswith("auto:")
    assert auto.u_rows.size == min(s.u_rows.size for s in each)
