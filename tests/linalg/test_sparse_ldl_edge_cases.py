"""Edge cases of the sparse ``L D L^T``: ``SparseLDL`` in every schedule and the ``sparse_ldl_factor`` and
``sparse_ldl_solve`` ops under it. The smallest and most lopsided patterns, supernode chains on either
side of the chunk width, zero, signed-zero and non-finite pivots and how far a NaN spreads, exact
power-of-two scaling, the looped ops bit for bit against the scan schedule, every table check of the
two builders, and the implicit derivatives at the edges."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from scipy.linalg import block_diag

import scaly as sc
from scaly.ad import finite_difference
from scaly.ad.derivatives import gradient, hessian, jacobian
from scaly.ad.forward import jvp
from scaly.codegen import render_c_module
from scaly.linalg.ops import SPARSE_LDL_MAX_WIDTH, sparse_ldl_factor, sparse_ldl_solve
from scaly.linalg import SparseLDL, SparseMatrix
from scaly.linalg.sparse_factor import Schedule, _call
from scaly.linalg.symbolic import Ordering
from scaly.passes.lowering import LoweringError

W = SPARSE_LDL_MAX_WIDTH


def _fn(name: str, inputs: list[sc.Expr], outputs: list[sc.Expr]) -> sc.Function:
  return sc.Function.from_exprs(name, list(inputs), list(outputs), [str(x.name) for x in inputs], [f"o{k}" for k in range(len(outputs))])


def _band(n: int) -> np.ndarray:
  return np.eye(n, dtype=bool) | np.eye(n, k=1, dtype=bool) | np.eye(n, k=-1, dtype=bool)


def _arrow(n: int, hub: int) -> np.ndarray:
  m = np.eye(n, dtype=bool)
  m[hub, :] = m[:, hub] = True
  return m


def _far(n: int) -> np.ndarray:
  """The diagonal and one entry in the last row of the first column."""
  m = np.eye(n, dtype=bool)
  m[n - 1, 0] = m[0, n - 1] = True
  return m


def _blocks() -> np.ndarray:
  """Four components: a dense block, a path, a lone entry and a star."""
  return block_diag(np.ones((3, 3)), _band(4), np.ones((1, 1)), _arrow(5, 0)) != 0


PATTERNS: dict[str, Callable[[], np.ndarray]] = {
  "n1": lambda: np.ones((1, 1), dtype=bool),
  "diag5": lambda: np.eye(5, dtype=bool),
  **{f"dense{n}": (lambda n=n: np.ones((n, n), dtype=bool)) for n in (7, 8, 9, 16, 17)},
  "band20": lambda: _band(20),
  "arrow_down12": lambda: _arrow(12, 11),
  "arrow_up12": lambda: _arrow(12, 0),
  "far10": lambda: _far(10),
  "blocks": _blocks,
}


def _signs(n: int) -> np.ndarray:
  return np.where(np.arange(n) % 3 == 1, -1.0, 1.0)


def _quasi_definite(mask: np.ndarray, signs: np.ndarray, seed: int) -> np.ndarray:
  """A symmetric matrix on ``mask``, diagonally dominant with sign ``signs[i]`` on row ``i``: its
  positive rows and its negative rows are definite blocks, so it is quasi-definite and every
  ordering factors it without a zero pivot."""
  rng = np.random.default_rng(seed)
  n = mask.shape[0]
  off = np.tril(np.where(mask, rng.uniform(0.5, 1.5, (n, n)) * rng.choice([-1.0, 1.0], (n, n)), 0.0), -1)
  off = off + off.T
  return off + np.diag(signs * (np.abs(off).sum(axis=1) + 1.0))


def _setup(mask: np.ndarray, seed: int = 0, signs: np.ndarray | None = None) -> tuple[SparseMatrix, np.ndarray, np.ndarray]:
  """The lower triangle of ``mask`` as a matrix symbol, a quasi-definite matrix on it and its values."""
  n = mask.shape[0]
  k = _quasi_definite(mask, _signs(n) if signs is None else signs, seed)
  mat = SparseMatrix.symbol("K", np.tril(mask))
  return mat, k, k[mat.coordinates()]


def _canonical(x: Any) -> np.ndarray:
  """The bits of ``x`` with every zero made +0.0 and every NaN one NaN: what the looped ops promise
  to share with the scan schedule."""
  x = np.asarray(x, dtype=np.float64)
  return np.where(np.isnan(x), np.nan, np.where(x == 0.0, 0.0, x)).view(np.int64)


def _assert_bits(got: Any, want: Any) -> None:
  np.testing.assert_array_equal(_canonical(got), _canonical(want))


def _unpack(fact: SparseLDL, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """The unit lower ``L`` (dense, permuted order) and ``D`` of a factor."""
  s = fact.symbolic
  lower = np.eye(s.n)
  lower[s.l_rows, np.repeat(np.arange(s.n), s.col_counts)] = values[: s.nnz_l]
  return lower, values[s.nnz_l :]


def _ancestors(fact: SparseLDL, j: int) -> list[int]:
  """``j`` and its ancestors in the elimination tree, in the permuted order."""
  out = [j]
  while fact.symbolic.parent[out[-1]] >= 0:
    out.append(int(fact.symbolic.parent[out[-1]]))
  return out


# --- patterns and schedules ---------------------------------------------------------------------


def _dense_chunks(n: int) -> tuple[list[int], list[list[int]]]:
  """The solve and factor chunk widths of a dense matrix: one chain of ``n`` columns, and row ``j``
  of ``L`` holding ``j`` columns that share their rows, both cut at the width cap."""
  cut = lambda m: [W] * (m // W) + ([m % W] if m % W else [])  # noqa: E731
  return cut(n), [cut(j) for j in range(n)]


def _row_chunks(fact: SparseLDL) -> list[list[int]]:
  ck = fact.tables()
  return [ck["ck_width"][ck["ck_ptr"][j] : ck["ck_ptr"][j + 1]].tolist() for j in range(fact.n)]


@pytest.mark.parametrize("n", [1, 7, 8, 9, 16, 17])
def test_dense_chains_are_cut_at_the_chunk_width(n: int) -> None:
  mat = SparseMatrix.symbol("K", np.tril(np.ones((n, n), dtype=bool)))
  fact = SparseLDL(mat, ordering="natural", schedule="loop")
  sn, rows = _dense_chunks(n)
  assert fact.solve_tables()["sn_width"].tolist() == sn
  assert _row_chunks(fact) == rows
  ck = fact.tables()
  for j in range(n):  # every chunk of row j updates rows j .. n-1
    assert np.all(ck["ck_len"][ck["ck_ptr"][j] : ck["ck_ptr"][j + 1]] == n - j)


def test_chunk_tables_of_lopsided_patterns() -> None:
  def fact(mask: np.ndarray, ordering: Ordering = "natural") -> SparseLDL:
    return SparseLDL(SparseMatrix.symbol("K", np.tril(mask)), ordering=ordering, schedule="loop")

  diag = fact(np.eye(5, dtype=bool))
  assert diag.symbolic.nnz_l == 0 and diag.tables()["ck_q"].size == 0 and np.all(diag.tables()["ck_ptr"] == 0)
  assert diag.solve_tables()["sn_width"].tolist() == [1] * 5
  # A path: one entry per column, and no two columns share their rows but the last two.
  band = fact(_band(20))
  assert band.symbolic.nnz_l == 19 and band.solve_tables()["sn_width"].tolist() == [1] * 18 + [2]
  assert _row_chunks(band) == [[]] + [[1]] * 19
  # Pointing down, the hub is last: no fill, and its row of L is 11 columns sharing one row.
  down = fact(_arrow(12, 11))
  assert down.symbolic.nnz_l == 11 and _row_chunks(down) == [[]] * 11 + [[W, 11 - W]]
  assert down.solve_tables()["sn_width"].tolist() == [1] * 10 + [2]
  # Pointing up, the hub first fills L completely; a fill-reducing ordering moves it last.
  assert fact(_arrow(12, 0)).symbolic.nnz_l == 66
  assert fact(_arrow(12, 0), "mmd").symbolic.nnz_l == 11 and fact(_arrow(12, 0), "auto").symbolic.nnz_l == 11
  # Column 0 has one row, the last; column 1 has none. Their counts differ by one, but column 0's
  # parent is 9, not 1: no chain.
  far = fact(_far(10))
  assert far.symbolic.nnz_l == 1 and far.symbolic.parent[0] == 9
  assert far.solve_tables()["sn_width"].tolist() == [1] * 10


def _cases() -> list[Any]:
  return [pytest.param(name, ordering, id=f"{name}-{ordering}") for name in PATTERNS for ordering in ("natural", "mmd")]


@pytest.mark.parametrize("name, ordering", _cases())
def test_every_schedule_on_edge_patterns(name: str, ordering: Ordering) -> None:
  """The looped factor and solve are the scan's bit for bit; the unrolled ones agree to rounding;
  all factor ``P K P^T`` to a componentwise backward error at rounding, keep the quasi-definite
  sign pattern and solve like NumPy."""
  mask = PATTERNS[name]()
  n = mask.shape[0]
  mat, k, kv = _setup(mask, seed=n)
  tag = f"sle_pat_{name}_{ordering}"
  scan = SparseLDL(mat, ordering=ordering, schedule="scan", name=f"{tag}_s")
  loop = SparseLDL(mat, ordering=ordering, schedule="loop", name=f"{tag}_l")
  facts = [scan, loop]
  if scan.work <= 400:  # straight-line code costs about a millisecond per operation to generate
    facts.append(SparseLDL(mat, ordering=ordering, schedule="unroll", name=f"{tag}_u"))
  b = sc.sym("b", n)
  bv = np.random.default_rng(n).standard_normal(n)
  outs = [f.values for f in facts] + [f.solve(b) for f in facts] + [f.health(signs=_signs(n)) for f in facts]
  out = _fn(tag, [mat.values, b], outs)._flat_numerical_call(kv, bv)
  values, xs, healthy = out[: len(facts)], out[len(facts) : 2 * len(facts)], out[2 * len(facts) :]
  assert all(bool(h) for h in healthy)  # the sign pattern in K's own order, whatever the ordering
  _assert_bits(values[1], values[0])
  _assert_bits(xs[1], xs[0])
  s = scan.symbolic
  lower, d = _unpack(scan, values[0])
  bound = np.abs(lower) @ np.diag(np.abs(d)) @ np.abs(lower.T)
  assert np.all(np.abs(lower @ np.diag(d) @ lower.T - k[np.ix_(s.perm, s.perm)]) <= (2 * n + 2) * np.finfo(float).eps * bound)
  np.testing.assert_array_equal(np.sign(d), _signs(n)[s.perm])  # the quasi-definite sign pattern, whatever the ordering
  ref = np.linalg.solve(k, bv)
  for f, x, v in zip(facts, xs, values, strict=True):
    np.testing.assert_allclose(v, values[0], rtol=1e-13, atol=1e-15, err_msg=f.schedule)
    np.testing.assert_allclose(x, ref, rtol=1e-11, atol=1e-13 * np.abs(ref).max(), err_msg=f.schedule)


def test_auto_schedule_on_either_side_of_the_option() -> None:
  """``auto`` unrolls exactly when the work is at most ``sparse_unroll``, a zero-work matrix even at
  0, and gives that schedule's values."""
  mat, _, kv = _setup(np.ones((9, 9), dtype=bool), seed=1)
  work = SparseLDL(mat, schedule="loop").work
  assert work == 156  # 120 update multiply-adds and 36 entries of L
  picks = {}
  for limit in (0, work - 1, work, 10 * work):
    with sc.options(linalg=dict(sparse_unroll=limit)):
      picks[limit] = SparseLDL(mat, name=f"sle_auto_{limit}")
  assert {k: f.schedule for k, f in picks.items()} == {0: "loop", work - 1: "loop", work: "unroll", 10 * work: "unroll"}
  unroll, loop = SparseLDL(mat, schedule="unroll", name="sle_auto_u"), SparseLDL(mat, schedule="loop", name="sle_auto_l")
  b = sc.sym("b", 9)
  facts = [*picks.values(), unroll, loop]
  out = _fn("sle_auto", [mat.values, b], [f.solve(b) for f in facts])._flat_numerical_call(kv, np.random.default_rng(1).standard_normal(9))
  for x, f in zip(out[:4], picks.values(), strict=True):
    _assert_bits(x, out[4] if f.schedule == "unroll" else out[5])
  for pattern in (np.eye(4, dtype=bool), np.ones((1, 1), dtype=bool)):  # nothing below the diagonal: no work
    with sc.options(linalg=dict(sparse_unroll=0)):
      assert SparseLDL(SparseMatrix.symbol("D", pattern)).schedule == "unroll"
  with pytest.raises(ValueError, match="non-negative integer"), sc.options(linalg=dict(sparse_unroll=-1)):
    pass


# --- values: zero, tiny and non-finite pivots, scaling --------------------------------------------

SCHEDULES: tuple[Schedule, ...] = ("scan", "loop", "unroll")


def _all_schedules(mat: SparseMatrix, tag: str, **opts: Any) -> list[SparseLDL]:
  return [SparseLDL(mat, schedule=s, name=f"{tag}_{s}", **opts) for s in SCHEDULES]


def test_zero_pivots_give_inf_and_nan_in_every_schedule() -> None:
  """No pivoting and no run-time check: a zero pivot divides by zero. ``[[0, 1], [1, 0]]`` gives
  ``L = 1 / 0`` and then ``D[1] = 0 - inf * 0 * inf``; a negative zero is a zero pivot too. The
  signs of the infinities follow the sign of a zero, which the schedules need not agree on."""
  mat = SparseMatrix.symbol("K", np.ones((2, 2), dtype=bool))  # both triangles: the lower entry is read
  facts = _all_schedules(mat, "sle_zp", ordering="natural")
  b = sc.sym("b", 2)
  fn = _fn("sle_zp", [mat.values, b], [out for f in facts for out in (f.values, f.solve(b), f.inertia(), f.health())])
  cases = [  # a list: the first two keys would be one dict key, since -0.0 == 0.0
    ((0.0, 1.0, 1.0, 0.0), [np.inf, 0.0, np.nan]),
    ((-0.0, 1.0, 1.0, 0.0), [-np.inf, -0.0, np.nan]),
    ((4.0, 2.0, 2.0, 1.0), [0.5, 4.0, 0.0]),  # the last pivot is exactly zero
  ]
  for kv, want in cases:
    out = fn._flat_numerical_call(np.array(kv), np.array([1.0, 2.0]))
    for s, (values, x, inertia, ok) in zip(SCHEDULES, (out[i : i + 4] for i in range(0, len(out), 4)), strict=True):
      np.testing.assert_array_equal(np.abs(values), np.abs(want), err_msg=s)
      assert not np.any(np.isfinite(x)), s
      d = np.array(want[1:])
      np.testing.assert_array_equal(inertia, [np.sum(d > 0), np.sum(d < 0), np.sum(~(d > 0) & ~(d < 0))], err_msg=s)
      assert not ok, s


def test_one_by_one() -> None:
  """``x = b / k``, including ``k`` zero, negative zero and NaN, in every schedule."""
  mat = SparseMatrix.symbol("K", np.ones((1, 1), dtype=bool))
  facts = _all_schedules(mat, "sle_one")
  b = sc.sym("b", 1)
  fn = _fn("sle_one", [mat.values, b], [out for f in facts for out in (f.values, f.solve(b), f.inertia(), f.health())])
  for k, bv, inertia in ((2.0, 3.0, [1, 0, 0]), (-4.0, 3.0, [0, 1, 0]), (0.0, 3.0, [0, 0, 1]), (-0.0, 3.0, [0, 0, 1]), (np.nan, 1.0, [0, 0, 1])):
    out = fn._flat_numerical_call(np.array([k]), np.array([bv]))
    with np.errstate(divide="ignore", invalid="ignore"):
      want = np.array([bv]) / np.array([k])
    for i in range(0, len(out), 4):
      _assert_bits(out[i], [k])
      _assert_bits(np.abs(out[i + 1]), np.abs(want))
      assert out[i + 2].dtype == np.float64
      np.testing.assert_array_equal(out[i + 2], inertia)
      assert bool(out[i + 3]) == (k != 0.0 and k == k)


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_a_nan_spreads_to_its_ancestors_and_its_component_only(schedule: Schedule) -> None:
  """A NaN on the diagonal of a column reaches exactly that column and its ancestors in the
  elimination tree, every entry of their columns of ``L``, and every unknown of its connected
  component; everything else is the clean run's bit for bit. An infinite right-hand side entry
  stays in its component too."""
  mask = _blocks()
  mat, _, kv = _setup(mask, seed=3)
  fact = SparseLDL(mat, ordering="mmd", schedule=schedule, name=f"sle_nan_{schedule}")
  s = fact.symbolic
  assert not np.array_equal(s.perm, np.arange(13))
  b = sc.sym("b", 13)
  x = fact.solve(b)
  fn = _fn(f"sle_nan_{schedule}", [mat.values, b], [fact.values, x, fact.inertia(), fact.health(x=x), fact.solve(b, refine=3, tol=1e-12)])
  bv = np.random.default_rng(3).standard_normal(13)
  clean = fn._flat_numerical_call(kv, bv)
  component = np.repeat(np.arange(4), [3, 4, 1, 5])
  rows, cols = mat.coordinates()
  leaf = next(c for c in range(9, 13) if not np.any(s.parent == s.iperm[c]))  # a spoke of the star with no children
  for c in (leaf, 3, 7, 8):  # a spoke, an end of the path, the lone entry, the star's hub
    poisoned = kv.copy()
    poisoned[np.flatnonzero((rows == c) & (cols == c))] = np.nan
    values, xv, inertia, ok, refined = fn._flat_numerical_call(poisoned, bv)
    up = _ancestors(fact, int(s.iperm[c]))
    bad_d = np.zeros(13, dtype=bool)
    bad_d[up] = True
    bad_l = np.isin(np.repeat(np.arange(13), s.col_counts), up)
    bad = np.r_[bad_l, bad_d]
    assert np.all(np.isnan(values[bad]))
    _assert_bits(values[~bad], clean[0][~bad])
    mine = component == component[c]
    assert np.all(np.isnan(xv[mine])) and np.all(np.isnan(refined[mine]))
    _assert_bits(xv[~mine], clean[1][~mine])
    np.testing.assert_allclose(refined[~mine], clean[1][~mine], rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(inertia, [np.sum(clean[0][s.nnz_l :][~bad_d] > 0), np.sum(clean[0][s.nnz_l :][~bad_d] < 0), len(up)])
    assert not ok
  for c in (0, 7):
    bb = bv.copy()
    bb[c] = np.inf
    _, xv, _, ok, refined = fn._flat_numerical_call(kv, bb)
    mine = component == component[c]
    assert not np.any(np.isfinite(xv[mine])) and not np.any(np.isfinite(refined[mine])) and not ok
    _assert_bits(xv[~mine], clean[1][~mine])


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_power_of_two_scaling_is_exact(schedule: Schedule) -> None:
  """``S K S`` with ``S`` a diagonal of powers of two from 2^-150 to 2^150 (pivots across 1e-90 to
  1e90) factors to ``S L S^-1`` and ``S^2 D`` and solves ``S b`` to ``S^-1 x``, all exactly:
  nothing in any schedule depends on the scale. So does a uniform scale of 2^-900 or 2^900."""
  mat, k, kv = _setup(_blocks(), seed=4)
  fact = SparseLDL(mat, ordering="mmd", schedule=schedule, name=f"sle_scale_{schedule}")
  s = fact.symbolic
  b = sc.sym("b", 13)
  fn = _fn(f"sle_scale_{schedule}", [mat.values, b], [fact.values, fact.solve(b), fact.inertia()])
  rng = np.random.default_rng(4)
  bv = rng.standard_normal(13)
  base = fn._flat_numerical_call(kv, bv)
  rows, cols = mat.coordinates()
  exps = [rng.integers(-150, 151, 13), np.full(13, -450), np.full(13, 450)]
  for e in exps:
    scale = np.ldexp(1.0, e)
    values, x, inertia = fn._flat_numerical_call(kv * scale[rows] * scale[cols], bv * scale)
    ep = e[s.perm]
    l_scale = np.ldexp(1.0, ep[s.l_rows] - ep[np.repeat(np.arange(13), s.col_counts)])
    _assert_bits(values[: s.nnz_l], base[0][: s.nnz_l] * l_scale)
    _assert_bits(values[s.nnz_l :], base[0][s.nnz_l :] * np.ldexp(1.0, 2 * ep))
    _assert_bits(x, base[1] / scale)
    np.testing.assert_array_equal(inertia, base[2])
  assert np.abs(np.linalg.solve(k, bv) - base[1]).max() < 1e-12


def test_explicit_zeros_are_stored_entries() -> None:
  """Stored entries whose value is zero keep their place in the pattern: the looped factor is the
  scan's but for the sign of a zero (``0 * (1 / D)`` with ``D < 0``), and the implicit gradient in
  a zero entry is the derivative of the solution in it, not zero."""
  mask = np.ones((6, 6), dtype=bool)
  mat, k, kv = _setup(mask, seed=5, signs=-np.ones(6))
  rows, cols = mat.coordinates()
  zero = (rows > cols) & ((rows + cols) % 2 == 0)
  kv[zero] = 0.0
  k[rows[zero], cols[zero]] = k[cols[zero], rows[zero]] = 0.0
  scan, loop = SparseLDL(mat, schedule="scan", name="sle_ez_s"), SparseLDL(mat, schedule="loop", name="sle_ez_l")
  b = sc.sym("b", 6)
  x = loop.solve(b)
  fn = _fn("sle_ez", [mat.values, b], [scan.values, loop.values, scan.solve(b), x, gradient(sc.sumsqr(x), mat.values)])
  fs, fl, xs, xl, g = fn._flat_numerical_call(kv, bv := np.random.default_rng(5).standard_normal(6))
  _assert_bits(fl, fs)
  _assert_bits(xl, xs)
  kinv = np.linalg.inv(k)
  ref = np.linalg.solve(k, bv)
  np.testing.assert_allclose(xl, ref, rtol=1e-12)
  bbar = 2 * kinv @ ref
  want = -(bbar[rows] * ref[cols] + np.where(rows != cols, bbar[cols] * ref[rows], 0.0))
  np.testing.assert_allclose(g, want, rtol=1e-10, atol=1e-13 * np.abs(want).max())
  assert np.all(g[zero] != 0.0)


def _mixed() -> tuple[SparseMatrix, np.ndarray, np.ndarray, np.ndarray]:
  """A pattern storing some pairs in the lower triangle only, some in the upper only and some in
  both; the matrix, its values and which stored entries are upper duplicates of a stored lower one."""
  rng = np.random.default_rng(13)
  n = 10
  pairs = np.tril(rng.random((n, n)) < 0.35, -1)
  where = rng.integers(0, 3, (n, n))  # 0: lower, 1: upper, 2: both
  mask = np.eye(n, dtype=bool) | (pairs & (where != 1)) | (pairs & (where != 0)).T
  k = _quasi_definite(pairs | pairs.T | np.eye(n, dtype=bool), _signs(n), 13)
  mat = SparseMatrix.symbol("K", mask)
  rows, cols = mat.coordinates()
  dup = (rows < cols) & mask.T[rows, cols]
  assert dup.any() and np.any(mask & ~mask.T)
  return mat, k, k[rows, cols], dup


def test_mixed_triangles_read_the_lower_entry_only() -> None:
  """Pairs stored in either triangle or both: a NaN in every upper duplicate changes nothing, not the
  factor, the solve, its refinement, nor the gradient, which is exactly zero there."""
  mat, k, kv, dup = _mixed()
  facts = _all_schedules(mat, "sle_mix", ordering="mmd")
  b = sc.sym("b", 10)
  outs = [o for f in facts for o in (f.values, f.solve(b), f.solve(b, refine=2))] + [gradient(sc.sumsqr(facts[1].solve(b)), mat.values)]
  fn = _fn("sle_mix", [mat.values, b], outs)
  bv = np.random.default_rng(13).standard_normal(10)
  poisoned = kv.copy()
  poisoned[dup] = np.nan
  clean, dirty = fn._flat_numerical_call(kv, bv), fn._flat_numerical_call(poisoned, bv)
  for c, d in zip(clean, dirty, strict=True):
    _assert_bits(d, c)
  np.testing.assert_allclose(dirty[1], np.linalg.solve(k, bv), rtol=1e-12)
  assert np.all(dirty[-1][dup] == 0.0) and np.all(dirty[-1][~dup] != 0.0)


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_a_batch_of_factorizations_under_vmap(schedule: Schedule) -> None:
  """A Function that factors and solves, mapped over three matrices ``K``, ``2 K`` and ``-K``: each
  lane its own factorization, and the gradient in every lane's entries."""
  mat, k, kv, dup = _mixed()
  nnz = mat.nnz
  ik, ib = sc.sym("ik", nnz), sc.sym("ib", 10)
  fact = SparseLDL(mat.with_values(ik), ordering="mmd", schedule=schedule, name=f"sle_vm_{schedule}")
  body = _fn(f"sle_vm_body_{schedule}", [ik, ib], [fact.solve(ib)])
  kvs, bs = sc.sym("kvs", (3 * nnz,)), sc.sym("bs", (30,))
  mapped = sc.vmap(body, 3, [(kvs, 0, nnz), (bs, 0, 10)])
  bv = np.random.default_rng(14).standard_normal(10)
  scales = [1.0, 2.0, -1.0]
  x, g = _fn(f"sle_vm_{schedule}", [kvs, bs], [mapped, gradient(sc.sumsqr(mapped), kvs)])._flat_numerical_call(
    np.concatenate([c * kv for c in scales]), np.tile(bv, 3)
  )
  rows, cols = mat.coordinates()
  for lane, c in enumerate(scales):
    ref = np.linalg.solve(c * k, bv)
    np.testing.assert_allclose(x[10 * lane : 10 * (lane + 1)], ref, rtol=1e-12)
    bbar = 2 * np.linalg.solve(c * k, ref)
    want = np.where(dup, 0.0, -(bbar[rows] * ref[cols] + np.where(rows != cols, bbar[cols] * ref[rows], 0.0)))
    np.testing.assert_allclose(g[nnz * lane : nnz * (lane + 1)], want, rtol=1e-10, atol=1e-13 * np.abs(want).max())


# --- solves ---------------------------------------------------------------------------------------


def test_empty_zero_and_single_column_right_hand_sides() -> None:
  """``b = 0`` solves to exactly zero with and without refinement (an adaptive loop takes no step);
  a matrix of one column is the vector solve, and one of no columns is empty."""
  mat, k, kv = _setup(_blocks(), seed=6)
  facts = _all_schedules(mat, "sle_rhs", ordering="mmd")
  b, bm = sc.sym("b", 13), sc.sym("B", (13, 1))
  outs = []
  for f in facts:
    outs += [f.solve(b), f.solve(b, refine=2), f.solve(b, refine=4, tol=1e-14), f.solve(bm), f.solve(sc.const(np.zeros((13, 0))))]
  fn = _fn("sle_rhs", [mat.values, b, bm], outs)
  zeros = fn._flat_numerical_call(kv, np.zeros(13), np.zeros((13, 1)))
  for x in zeros:
    assert np.all(x == 0.0)
  bv = np.random.default_rng(6).standard_normal(13)
  out = fn._flat_numerical_call(kv, bv, bv[:, None])
  for i in range(0, len(out), 5):
    x, _, _, xm, empty = out[i : i + 5]
    _assert_bits(xm[:, 0], x)
    assert empty.shape == (13, 0)
    np.testing.assert_allclose(x, np.linalg.solve(k, bv), rtol=1e-11)


def test_the_empty_matrix() -> None:
  mat = SparseMatrix.symbol("K", np.zeros((0, 0), dtype=bool))
  facts = [*_all_schedules(mat, "sle_empty"), SparseLDL(mat, schedule="loop", name="sle_empty_loop")]
  assert SparseLDL(mat).schedule == "unroll"
  b = sc.sym("b", 0)
  out = _fn("sle_empty", [mat.values, b], [o for f in facts for o in (f.values, f.solve(b), f.inertia(), f.health())])._flat_numerical_call(
    np.zeros(0), np.zeros(0)
  )
  for i in range(0, len(out), 4):
    assert out[i].shape == (0,) and out[i + 1].shape == (0,)
    np.testing.assert_array_equal(out[i + 2], [0.0, 0.0, 0.0])
    assert bool(out[i + 3])


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_adaptive_refinement_edges(schedule: Schedule) -> None:
  """An infinite tolerance takes no step; a tolerance of 1e-300 takes all of its one step, as the
  fixed refinement does. A NaN in ``b`` ends the loop at its cap at the latest and leaves the other
  components accurate."""
  mat, k, kv = _setup(_blocks(), seed=7)
  fact = SparseLDL(mat, ordering="mmd", schedule=schedule, name=f"sle_ref_{schedule}")
  b = sc.sym("b", 13)
  outs = [fact.solve(b), fact.solve(b, refine=5, tol=np.inf), fact.solve(b, refine=1, tol=1e-300), fact.solve(b, refine=1)]
  fn = _fn(f"sle_ref_{schedule}", [mat.values, b], outs)
  bv = np.random.default_rng(7).standard_normal(13) * 1e3
  x0, inf_tol, one_adaptive, one = fn._flat_numerical_call(kv, bv)
  _assert_bits(inf_tol, x0)
  np.testing.assert_allclose(one_adaptive, one, rtol=1e-14, atol=1e-16 * np.abs(one).max())
  np.testing.assert_allclose(one, np.linalg.solve(k, bv), rtol=1e-12)
  bb = bv.copy()
  bb[0] = np.nan
  x0, inf_tol, one_adaptive, one = fn._flat_numerical_call(kv, bb)
  healthy = np.arange(13) >= 3
  for x in (x0, inf_tol, one_adaptive, one):
    assert np.all(np.isnan(x[~healthy]))
    np.testing.assert_allclose(x[healthy], np.linalg.solve(k[3:, 3:], bb[3:]), rtol=1e-11)
  with pytest.raises(ValueError, match="tol must be positive"):
    fact.solve(b, refine=2, tol=float("nan"))
  with pytest.raises(ValueError, match="refine must be a non-negative integer"):
    fact.solve(b, refine=True)
  with pytest.raises(ValueError, match="refine must be a non-negative integer"):
    fact.solve(b, refine=1.0)  # ty: ignore[invalid-argument-type]
  with pytest.raises(ValueError, match="of 13 rows"):
    fact.solve(sc.sym("B3", (13, 1, 1)))
  with pytest.raises(ValueError, match="of 13 rows"):
    fact.solve(sc.sym("B14", (14, 2)))


@pytest.mark.parametrize("schedule", ["scan", "loop"])
def test_adaptive_refinement_step_counts_at_the_edges(schedule: Schedule) -> None:
  """Refining against ``K`` with the factor of ``1.2 K`` leaves ``(1 - 6^-(k+1)) K^-1 b`` after ``k``
  steps, so the result shows the step count: an infinite tolerance takes none, a tolerance of 1e-300
  stops at the cap however small the cap, and ``b = 0`` takes none."""
  mat, k, kv = _setup(_blocks(), seed=12, signs=np.ones(13))
  fact = SparseLDL(mat, ordering="mmd", schedule=schedule, name=f"sle_cnt_{schedule}")
  inexact = SparseLDL(mat.with_values(mat.values * 1.2), symbolic=fact.symbolic, schedule=schedule, name=f"sle_cnt_{schedule}_x")
  b = sc.sym("b", 13)
  cases = {(4, np.inf): 0, (1, 1e-300): 1, (3, 1e-300): 3}
  outs = [_call(fact._solver_function(refine, tol), inexact.values, mat.values, b) for refine, tol in cases]
  fn = _fn(f"sle_cnt_{schedule}", [mat.values, b], outs)
  bv = np.random.default_rng(12).standard_normal(13)
  for x, steps in zip(fn._flat_numerical_call(kv, bv), cases.values(), strict=True):
    np.testing.assert_allclose(x, (1.0 - 6.0 ** -(steps + 1)) * np.linalg.solve(k, bv), rtol=1e-10)
  assert all(np.all(x == 0.0) for x in fn._flat_numerical_call(kv, np.zeros(13)))


def test_inertia_and_health_edges() -> None:
  """Definite matrices of either sign, and ``health``'s strict inequalities at the tolerance."""
  mat, k, kv = _setup(np.ones((9, 9), dtype=bool), seed=8, signs=np.ones(9))
  fact = SparseLDL(mat, schedule="loop", name="sle_in")
  b = sc.sym("b", 9)
  x = fact.solve(b)
  flags = [fact.health(), fact.health(signs=np.ones(9)), fact.health(signs=-np.ones(9)), fact.health(x=x)]
  fn = _fn("sle_in", [mat.values, b], [fact.inertia(), *flags])
  bv = np.random.default_rng(8).standard_normal(9)
  inertia, *ok = fn._flat_numerical_call(kv, bv)
  np.testing.assert_array_equal(inertia, [9.0, 0.0, 0.0])
  assert [bool(f) for f in ok] == [True, True, False, True]
  inertia, *ok = fn._flat_numerical_call(-kv, bv)
  np.testing.assert_array_equal(inertia, [0.0, 9.0, 0.0])
  assert [bool(f) for f in ok] == [True, False, True, True]
  _, *ok = fn._flat_numerical_call(kv, np.r_[bv[:8], np.inf])
  assert [bool(f) for f in ok] == [True, True, False, False]
  dg = SparseMatrix.symbol("Dg", np.eye(3, dtype=bool))
  fd = SparseLDL(dg, schedule="loop", name="sle_in_d")
  signs = [1.0, -1.0, 1.0]
  tols = [0.5, np.nextafter(0.5, 0.0)]
  outs = [fd.inertia(), *(fd.health(pivot_tol=t) for t in tols), *(fd.health(signs=signs, pivot_tol=t) for t in tols)]
  inertia, *ok = _fn("sle_in_d", [dg.values], outs)._flat_numerical_call(np.array([2.0, -3.0, 0.5]))
  np.testing.assert_array_equal(inertia, [2.0, 1.0, 0.0])
  assert [bool(f) for f in ok] == [False, True, False, True]
  inertia, *ok = _fn("sle_in_d", [dg.values], outs)._flat_numerical_call(np.array([-0.0, -3.0, 0.5]))
  np.testing.assert_array_equal(inertia, [1.0, 1.0, 1.0])
  assert not any(bool(f) for f in ok)
  with pytest.raises(ValueError, match="signs"):
    fd.health(signs=[1.0, 0.0, 1.0])


def test_kkt_assembled_from_blocks() -> None:
  """``[[P + rho I, A^T], [A, -delta I]]`` from ``SparseMatrix.block``: both triangles of the
  coupling stored, the lower one read. Solves, inertia ``(n, m, 0)`` and the derivatives in
  ``rho``, ``delta``, ``P`` and ``A`` against a dense reference; a zero block stores no diagonal,
  and stored zeros there factor in the natural order."""
  n, m = 6, 3
  rng = np.random.default_rng(9)
  pmask = _band(n)
  amask = rng.random((m, n)) < 0.5
  amask[np.arange(m), np.arange(m)] = True
  pm, am = SparseMatrix.symbol("P", np.tril(pmask)), SparseMatrix.symbol("A", amask)
  rho, delta = sc.sym("rho", ()), sc.sym("delta", ())
  full_p = pm + pm.tril(-1).T
  kkt = SparseMatrix.block([[full_p.add_diagonal(rho), am.T], [am, SparseMatrix.identity(m) * -delta]])
  fact = SparseLDL(kkt, schedule="loop", name="sle_kkt")
  b = sc.sym("b", n + m)
  x = fact.solve(b)
  f = sc.sumsqr(x)
  outs = [x, fact.inertia(), jacobian(x, rho), jacobian(x, delta), gradient(f, pm.values), gradient(f, am.values)]
  fn = _fn("sle_kkt", [pm.values, am.values, rho, delta, b], outs)
  p = _quasi_definite(pmask, np.ones(n), 9)
  pv, av = p[pm.coordinates()], rng.standard_normal(am.nnz)
  a = np.zeros((m, n))
  a[am.coordinates()] = av
  rv, dv, bv = 1e-6, 1e-4, rng.standard_normal(n + m)
  xv, inertia, dx_rho, dx_delta, gp, ga = fn._flat_numerical_call(pv, av, np.array(rv), np.array(dv), bv)
  dense = np.block([[p + rv * np.eye(n), a.T], [a, -dv * np.eye(m)]])
  ref = np.linalg.solve(dense, bv)
  np.testing.assert_allclose(xv, ref, rtol=1e-9)
  np.testing.assert_array_equal(inertia, [n, m, 0])
  kinv = np.linalg.inv(dense)
  np.testing.assert_allclose(dx_rho[:, 0], -kinv @ np.r_[ref[:n], np.zeros(m)], rtol=1e-8)
  np.testing.assert_allclose(dx_delta[:, 0], kinv @ np.r_[np.zeros(n), ref[n:]], rtol=1e-8)
  # A symmetric perturbation per stored entry: P's lower entries, and A's in the lower block.
  bbar = 2 * kinv @ ref
  pr, pc = pm.coordinates()
  ar, ac = am.coordinates()
  want_p = -(bbar[pr] * ref[pc] + np.where(pr != pc, bbar[pc] * ref[pr], 0.0))
  np.testing.assert_allclose(gp, want_p, rtol=1e-8, atol=1e-12 * np.abs(want_p).max())
  np.testing.assert_allclose(ga, -(bbar[n + ar] * ref[ac] + bbar[ac] * ref[n + ar]), rtol=1e-8)
  with pytest.raises(ValueError, match="every diagonal entry stored"):
    SparseLDL(SparseMatrix.block([[full_p.add_diagonal(rho), am.T], [am, None]]))
  zero_block = SparseMatrix.block([[full_p.add_diagonal(rho), am.T], [am, SparseMatrix.zeros((m, m)).add_diagonal(0.0)]])
  nat = SparseLDL(zero_block, ordering="natural", name="sle_kkt0")
  x0, inertia = _fn("sle_kkt0", [pm.values, am.values, rho, b], [nat.solve(b), nat.inertia()])._flat_numerical_call(pv, av, np.array(rv), bv)
  dense[n:, n:] = 0.0
  np.testing.assert_allclose(x0, np.linalg.solve(dense, bv), rtol=1e-9)
  np.testing.assert_array_equal(inertia, [n, m, 0])


# --- derivatives ----------------------------------------------------------------------------------


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_one_by_one_derivatives(monkeypatch: pytest.MonkeyPatch, schedule: Schedule) -> None:
  """``x = b / k``: first and second derivatives in ``k`` and ``b``, forward, multi-seed forward and
  reverse, exactly as the formulas give them."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  mat = SparseMatrix.symbol("K", np.ones((1, 1), dtype=bool))
  kk, b = mat.values, sc.sym("b", 1)
  fact = SparseLDL(mat, schedule=schedule, name=f"sle_d1_{schedule}")
  x = fact.solve(b)
  xs = x.sum()
  outs = [
    jvp(x, kk, sc.const(np.ones(1))),
    jvp(x, b, sc.const(np.ones(1))),
    jacobian(x, kk),
    gradient(xs, kk),
    gradient(xs, b),
    hessian(xs, kk),
    jacobian(gradient(xs, kk), b),
    fact.solve(sc.const(np.ones((1, 3))) * b.reshape((1, 1))),
  ]
  fn = _fn(f"sle_d1_{schedule}", [kk, b], outs)
  for kv, bv in ((2.0, 3.0), (-0.5, 1.5)):
    tk, tb, jk, gk, gb, hk, hkb, xm = fn._flat_numerical_call(np.array([kv]), np.array([bv]))
    for got in (tk, jk, gk):
      np.testing.assert_allclose(np.ravel(got), [-bv / kv**2], rtol=1e-15)
    for got in (tb, gb):
      np.testing.assert_allclose(np.ravel(got), [1.0 / kv], rtol=1e-15)
    np.testing.assert_allclose(np.ravel(hk), [2.0 * bv / kv**3], rtol=1e-15)
    np.testing.assert_allclose(np.ravel(hkb), [-1.0 / kv**2], rtol=1e-15)
    np.testing.assert_allclose(xm, np.full((1, 3), bv / kv), rtol=1e-15)


def _solve_jac_ref(mat: SparseMatrix, k: np.ndarray, x: np.ndarray) -> np.ndarray:
  """``dx / dK`` for the stored lower entries of a symmetric ``K``: ``-K^-1 (e_r e_c^T + e_c e_r^T) x``."""
  kinv = np.linalg.inv(k)
  rows, cols = mat.coordinates()
  return -(kinv[:, rows] * x[cols] + np.where(rows != cols, kinv[:, cols] * x[rows], 0.0))


@pytest.mark.parametrize("n, schedule", [(9, "loop"), (9, "scan"), (17, "loop")])
def test_implicit_derivatives_across_the_chunk_boundary(monkeypatch: pytest.MonkeyPatch, n: int, schedule: Schedule) -> None:
  """A dense chain of 9 or 17 columns (solve chunks 8 + 1, 8 + 8 + 1): the Jacobian of ``x`` in
  ``K``'s entries and in ``b``, one seed and many, reverse, and the second derivative, against the
  dense formula and finite differences."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  mat, k, kv = _setup(np.ones((n, n), dtype=bool), seed=n + 10)
  fact = SparseLDL(mat, ordering="natural", schedule=schedule, name=f"sle_dc_{n}_{schedule}")
  assert fact.solve_tables()["sn_width"].tolist() == _dense_chunks(n)[0]
  b = sc.sym("b", n)
  x = fact.solve(b)
  f = 0.5 * sc.sumsqr(x) + x.sin().sum()
  rng = np.random.default_rng(n)
  seed = rng.standard_normal(mat.nnz)
  outs = [x, jacobian(x, mat.values), jacobian(x, b), jvp(x, mat.values, sc.const(seed)), gradient(f, mat.values), gradient(f, b)]
  fn = _fn(f"sle_dc_{n}_{schedule}", [mat.values, b], [*outs, jacobian(gradient(f, b), mat.values)])
  bv = rng.standard_normal(n)
  xv, jk, jb, tk, gk, _, hbk = fn._flat_numerical_call(kv, bv)
  ref = _solve_jac_ref(mat, k, xv)
  np.testing.assert_allclose(jk, ref, rtol=1e-9, atol=1e-12 * np.abs(ref).max())
  np.testing.assert_allclose(jb, np.linalg.inv(k), rtol=1e-9, atol=1e-13)
  np.testing.assert_allclose(tk, ref @ seed, rtol=1e-9, atol=1e-12 * np.abs(ref).max())
  np.testing.assert_allclose(gk, (xv + np.cos(xv)) @ ref, rtol=1e-9, atol=1e-12 * np.abs(gk).max())
  fd = finite_difference(lambda z: fn._flat_numerical_call(z, bv)[5], kv)
  np.testing.assert_allclose(hbk, fd, rtol=1e-5, atol=1e-6 * np.abs(hbk).max())


def test_scan_loops_differentiated_agree_with_the_implicit_rule(monkeypatch: pytest.MonkeyPatch) -> None:
  """The scan schedule's factor and sweeps differentiated through their loops (``solve_with`` has no
  rule of its own) give the implicit rule's derivatives; the unrolled factor, straight-line code,
  has the same Jacobian as the scan's loops."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  mat, k, kv = _setup(np.ones((9, 9), dtype=bool), seed=21)
  scan = SparseLDL(mat, ordering="natural", schedule="scan", name="sle_thru_s")
  unroll = SparseLDL(mat, ordering="natural", schedule="unroll", name="sle_thru_u")
  b = sc.sym("b", 9)
  looped, implicit = scan.solve_with(scan.values, b), scan.solve(b)
  outs = [
    jacobian(looped, mat.values),
    jacobian(implicit, mat.values),
    gradient(sc.sumsqr(looped), mat.values),
    gradient(sc.sumsqr(implicit), mat.values),
    jacobian(scan.values, mat.values),
    jacobian(unroll.values, mat.values),
  ]
  jl, ji, gl, gi, fs, fu = _fn("sle_thru", [mat.values, b], outs)._flat_numerical_call(kv, np.random.default_rng(21).standard_normal(9))
  np.testing.assert_allclose(jl, ji, rtol=1e-10, atol=1e-13 * np.abs(ji).max())
  np.testing.assert_allclose(gl, gi, rtol=1e-10, atol=1e-13 * np.abs(gi).max())
  np.testing.assert_allclose(fs, fu, rtol=1e-10, atol=1e-13 * np.abs(fu).max())
  values = _fn("sle_thru_v", [mat.values], [unroll.values])
  fd = finite_difference(lambda z: values._flat_numerical_call(z)[0], kv)
  np.testing.assert_allclose(fu, fd, rtol=1e-6, atol=1e-8 * np.abs(fu).max())


def test_the_factor_op_refuses_every_derivative() -> None:
  """``sparse_ldl_factor`` has no derivative: forward mode with one seed or many, and reverse mode,
  refuse it, directly or through a solve with it, naming the scan schedule."""
  mat, _, _ = _setup(np.ones((4, 4), dtype=bool))
  loop = SparseLDL(mat, schedule="loop", name="sle_nod")
  bare = sparse_ldl_factor(mat.values * 2.0, loop.tables())
  solved = sparse_ldl_solve(bare, sc.sym("b", 4), loop.solve_tables())
  for build in (
    lambda: jacobian(loop.values, mat.values),
    lambda: jvp(bare, mat.values, sc.const(np.ones(mat.nnz))),
    lambda: gradient(bare.sum(), mat.values),
    lambda: jacobian(solved, mat.values),
  ):
    with pytest.raises(NotImplementedError, match="schedule='scan'"):
      build()


# --- names ----------------------------------------------------------------------------------------


def test_two_factorizations_in_one_graph() -> None:
  """Two factorizations of one matrix under different names coexist. One name serves one
  factorization: two with different orderings, of different matrices, the same one built twice, or
  two names spelled alike in C are refused at lowering."""
  mat, k, kv = _setup(_blocks(), seed=11)
  b = sc.sym("b", 13)
  one, two = (
    SparseLDL(mat, ordering="natural", schedule="scan", name="sle_pair_a"),
    SparseLDL(mat, ordering="mmd", schedule="scan", name="sle_pair_b"),
  )
  xa, xb = _fn("sle_pair", [mat.values, b], [one.solve(b), two.solve(b)])._flat_numerical_call(
    kv, bv := np.random.default_rng(11).standard_normal(13)
  )
  np.testing.assert_allclose(xa, np.linalg.solve(k, bv), rtol=1e-11)
  np.testing.assert_allclose(xb, xa, rtol=1e-11)
  clash = SparseLDL(mat, ordering="mmd", schedule="scan", name="sle_pair_a")
  with pytest.raises(LoweringError, match="give them distinct names"):
    render_c_module(_fn("sle_clash", [mat.values, b], [one.solve(b), clash.solve(b)]))
  other = SparseMatrix.symbol("K2", np.eye(13, dtype=bool))
  with pytest.raises(LoweringError, match="give them distinct names"):
    render_c_module(_fn("sle_clash2", [mat.values, other.values, b], [one.solve(b), SparseLDL(other, schedule="scan", name="sle_pair_a").solve(b)]))
  twin = SparseLDL(mat, ordering="natural", schedule="scan", name="sle_pair_a")
  with pytest.raises(LoweringError, match="give them distinct names"):
    render_c_module(_fn("sle_twin", [mat.values, b], [one.solve(b), twin.solve(b)]))
  dotted = SparseLDL(other, schedule="scan", name="sle.pair_b")  # two names, one C symbol
  with pytest.raises(LoweringError, match="both 'sle_pair_b_s[bf]' in C"):
    render_c_module(_fn("sle_clash3", [mat.values, other.values, b], [two.solve(b), dotted.solve(b)]))


# --- table validation -----------------------------------------------------------------------------


def _tables() -> tuple[SparseLDL, dict[str, np.ndarray], dict[str, np.ndarray]]:
  """A dense block (a chain, chunks up to width 3) beside a path, natural order."""
  mask = block_diag(np.ones((4, 4)), _band(4)) != 0
  fact = SparseLDL(SparseMatrix.symbol("K", np.tril(mask)), ordering="natural", schedule="loop")
  return fact, {k: v.copy() for k, v in fact.tables().items()}, {k: v.copy() for k, v in fact.solve_tables().items()}


def _set(table: np.ndarray, index: int, value: int) -> np.ndarray:
  out = table.copy()
  out[index] = value
  return out


FACTOR_CORRUPTIONS: dict[str, tuple[Callable[[dict[str, np.ndarray], int], dict[str, np.ndarray]], str]] = {
  "no_a_ptr": (lambda t, nl: {**t, "a_ptr": t["a_ptr"][:0]}, "disagree on the order"),
  "short_l_ptr": (lambda t, nl: {**t, "l_ptr": t["l_ptr"][:-1]}, "disagree on the order"),
  "negative_a_src": (lambda t, nl: {**t, "a_src": _set(t["a_src"], 0, -1)}, "outside its"),
  "width_0": (lambda t, nl: {**t, "ck_width": _set(t["ck_width"], 0, 0)}, "chunks cover 1 to 8"),
  "width_9": (lambda t, nl: {**t, "ck_width": _set(t["ck_width"], 0, 9)}, "chunks cover 1 to 8"),
  "a_ptr_start": (lambda t, nl: {**t, "a_ptr": t["a_ptr"] + 1}, "columns of the matrix"),
  "a_ptr_decreasing": (lambda t, nl: {**t, "a_ptr": _set(t["a_ptr"], 2, 0)}, "columns of the matrix"),
  "a_src_length": (lambda t, nl: {**t, "a_src": t["a_src"][:-1]}, "columns of the matrix"),
  "a_rows_negative": (lambda t, nl: {**t, "a_rows": _set(t["a_rows"], 0, -1)}, "columns of the matrix"),
  "l_ptr_end": (lambda t, nl: {**t, "l_ptr": _set(t["l_ptr"], -1, nl - 1)}, "l_ptr and l_rows"),
  "l_rows_past_n": (lambda t, nl: {**t, "l_rows": _set(t["l_rows"], -1, 8)}, "l_ptr and l_rows"),
  "l_row_on_diagonal": (lambda t, nl: {**t, "l_rows": _set(t["l_rows"], 0, 0)}, "rows below the diagonal, sorted"),
  "l_row_above_diagonal": (lambda t, nl: {**t, "l_rows": _set(t["l_rows"], 3, 0)}, "rows below the diagonal, sorted"),
  "l_rows_unsorted": (lambda t, nl: {**t, "l_rows": _set(_set(t["l_rows"], 0, 2), 1, 1)}, "rows below the diagonal, sorted"),
  "l_rows_repeated": (lambda t, nl: {**t, "l_rows": _set(t["l_rows"], 1, 1)}, "rows below the diagonal, sorted"),
  "r_cols_past_n": (lambda t, nl: {**t, "r_cols": _set(t["r_cols"], 0, 8)}, "one entry per entry of L"),
  "r_cols_length": (lambda t, nl: {**t, "r_cols": t["r_cols"][:-1]}, "one entry per entry of L"),
  "r_pos_past_l": (lambda t, nl: {**t, "r_pos": _set(t["r_pos"], 0, nl)}, "one entry per entry of L"),
  "ck_ptr_end": (lambda t, nl: {**t, "ck_ptr": _set(t["ck_ptr"], -1, t["ck_q"].size - 1)}, "chunk tables disagree"),
  "ck_width_length": (lambda t, nl: {**t, "ck_width": t["ck_width"][:-1]}, "chunk tables disagree"),
  "ck_len_length": (lambda t, nl: {**t, "ck_len": t["ck_len"][:-1]}, "chunk tables disagree"),
  "ck_q_past_l": (lambda t, nl: {**t, "ck_q": _set(t["ck_q"], 0, nl)}, "chunk tables disagree"),
  "chunk_past_l": (lambda t, nl: {**t, "ck_q": _set(t["ck_q"], -1, nl - 1), "ck_width": _set(t["ck_width"], -1, 2)}, "a chunk runs past"),
  "ck_len_negative": (lambda t, nl: {**t, "ck_len": _set(t["ck_len"], 0, -1)}, "a chunk runs past"),
}


@pytest.mark.parametrize("case", list(FACTOR_CORRUPTIONS))
def test_sparse_ldl_factor_refuses_each_corrupted_table(case: str) -> None:
  fact, tables, _ = _tables()
  corrupt, match = FACTOR_CORRUPTIONS[case]
  kv = sc.sym("kv", fact.matrix.nnz)
  sparse_ldl_factor(kv, tables)  # the valid tables pass
  with pytest.raises(ValueError, match=match):
    sparse_ldl_factor(kv, corrupt(tables, fact.symbolic.nnz_l))


def test_sparse_ldl_factor_checks_the_rows_of_every_lane_of_a_chunk() -> None:
  """A chunk's later columns may run past ``L`` when its first does not."""
  fact, tables, _ = _tables()
  nl = fact.symbolic.nnz_l
  wide = np.flatnonzero((tables["ck_width"] >= 2) & (tables["ck_len"] >= 2))
  q = int(tables["ck_q"][wide[0]])
  assert tables["r_pos"][q] + tables["ck_len"][wide[0]] <= nl
  with pytest.raises(ValueError, match="rows run past"):
    sparse_ldl_factor(sc.sym("kv", fact.matrix.nnz), {**tables, "r_pos": _set(tables["r_pos"], q + 1, nl - 1)})
  with pytest.raises(ValueError, match="floating-point vector"):
    sparse_ldl_factor(sc.sym("kv_int", fact.matrix.nnz, dtype="int64"), tables)


SOLVE_CORRUPTIONS: dict[str, tuple[Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]], str]] = {
  "short_l_ptr": (lambda t: {**t, "l_ptr": t["l_ptr"][:-1]}, "of order 8 needs a factor"),
  "sn_lengths": (lambda t: {**t, "sn_first": t["sn_first"][:-1]}, "every column once"),
  "sn_width_0": (lambda t: {**t, "sn_first": np.array([0, 4, 4, 5, 6]), "sn_width": np.array([4, 0, 1, 1, 2])}, "chunks cover 1 to 8"),
  "perm_past_n": (lambda t: {**t, "perm": t["perm"] + 1}, "permutation"),
  "l_ptr_end": (lambda t: {**t, "l_ptr": _set(t["l_ptr"], -1, t["l_rows"].size - 1)}, "l_ptr and l_rows"),
  "l_row_on_diagonal": (lambda t: {**t, "l_rows": _set(t["l_rows"], 0, 0)}, "rows below the diagonal"),
  # Column 3 ends the dense block: it has no rows at all.
  "empty_column_in_a_chain": (
    lambda t: {**t, "sn_first": np.array([0, 3, 5, 6]), "sn_width": np.array([3, 2, 1, 2])},
    "columns 3..4 are not a chain",
  ),
  # Column 4's row is 5, but column 5's rows are not the rest of column 4's.
  "chain_rows_differ": (lambda t: {**t, "sn_first": np.array([0, 4, 6]), "sn_width": np.array([4, 2, 2])}, "columns 4..5 are not a chain"),
  # Column 4's row moved to 6: its first row is not the next column.
  "chain_first_row": (
    lambda t: {**t, "l_rows": _set(t["l_rows"], 6, 6), "sn_first": np.array([0, 4, 6]), "sn_width": np.array([4, 2, 2])},
    "columns 4..5 are not a chain",
  ),
}


@pytest.mark.parametrize("case", list(SOLVE_CORRUPTIONS))
def test_sparse_ldl_solve_refuses_each_corrupted_table(case: str) -> None:
  fact, _, tables = _tables()
  f, b = sc.sym("f", fact.values.shape), sc.sym("b", 8)
  sparse_ldl_solve(f, b, tables)
  corrupt, match = SOLVE_CORRUPTIONS[case]
  with pytest.raises(ValueError, match=match):
    sparse_ldl_solve(f, b, corrupt(tables))
  with pytest.raises(ValueError, match="needs a factor of"):
    sparse_ldl_solve(sc.sym("f_short", fact.values.size - 1), b, tables)
