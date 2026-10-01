"""A loop carry updated in place through ``put``/``put_add`` at run-time indices, proven safe at the
loop from the constant index tables it slices (``in_place_steps``).

Every in-place loop is compared bit for bit with the same loop lowered with two carry slots, and
checked against NumPy or SciPy. Each case where a read would see a value an earlier write in the
same step already changed must keep the two-slot carry, and still compute the right thing.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular

import scaly as sc
from scaly.ad import finite_difference
from scaly.ad.derivatives import gradient
from scaly.codegen import render_c_module
from scaly.passes import lowering
from scaly.passes.lowering import _normalize_function, in_place_steps, update_chain

RNG = np.random.default_rng(303)


def _fn(name, inputs, outputs):
  return sc.Function.from_exprs(name, list(inputs), list(outputs), [str(x.name) for x in inputs], [f"o{k}" for k in range(len(outputs))])


def _run_both(monkeypatch: pytest.MonkeyPatch, build, point, *, expect_in_place: bool):
  """The function built in place (when proven) and with two carry slots: results must agree exactly."""
  fn = build("ip")
  src = str(render_c_module(fn).source)
  assert ("_inplace_raw" in src) == expect_in_place
  got = fn._flat_numerical_call(*point)
  monkeypatch.setattr(lowering, "DONATE_CARRIES", False)
  plain = build("two_slot")
  assert "_inplace_raw" not in str(render_c_module(plain).source)
  ref = plain._flat_numerical_call(*point)
  monkeypatch.setattr(lowering, "DONATE_CARRIES", True)
  for a, b in zip(got, ref, strict=True):
    np.testing.assert_array_equal(a, b)
  return got


def _lower_triangle(n: int, density: float, seed: int) -> sparse.csr_array:
  a = sparse.random_array((n, n), density=density, random_state=seed, format="csr")
  lower = sparse.tril(a, k=-1, format="csr") + sparse.diags_array(2.0 + np.arange(n) % 3, format="csr")
  lower = sparse.csr_array(lower)
  lower.sort_indices()
  return lower


def _row_tables(m: sparse.csr_array, *, strict_lower: bool) -> tuple[np.ndarray, np.ndarray, int]:
  n = m.shape[0]
  rows = []
  for r in range(n):
    lo, hi = m.indptr[r], m.indptr[r + 1]
    keep = [(c, q) for c, q in zip(m.indices[lo:hi], range(lo, hi), strict=True) if not strict_lower or c < r]
    rows.append(keep)
  width = max(1, max(len(r) for r in rows))
  cols = np.full((n, width), n, dtype=np.int64)
  pos = np.full((n, width), m.nnz, dtype=np.int64)
  for r, keep in enumerate(rows):
    for j, (c, q) in enumerate(keep):
      cols[r, j], pos[r, j] = c, q
  return cols.reshape(-1), pos.reshape(-1), width


def _forward_substitution(m: sparse.csr_array, tag: str, *, grads: bool) -> sc.Function:
  """``L x = b`` row by row: step ``k`` reads ``x`` at the columns of row ``k`` (all ``< k``) and
  writes ``x[k]``, a run-time-index read and write on one carry."""
  n = m.shape[0]
  cols_t, pos_t, width = _row_tables(m, strict_lower=True)
  diag_pos = np.array([m.indptr[r] + list(m.indices[m.indptr[r] : m.indptr[r + 1]]).index(r) for r in range(n)], dtype=np.int64)
  x, k = sc.sym("x", n), sc.sym("k", (), dtype="int64")
  cols, pos, dpos = sc.sym("cols", width, dtype="int64"), sc.sym("pos", width, dtype="int64"), sc.sym("dpos", 1, dtype="int64")
  vals, bk = sc.sym("vals", m.nnz), sc.sym("bk", ())
  row = (sc.take(vals, pos) * sc.take(x, cols)).sum()
  xk = (bk - row) / sc.take(vals, dpos)
  body = sc.Function.from_exprs(
    f"fsub_{tag}", [x, k, cols, pos, dpos, vals, bk], [sc.put(x, k.reshape((1,)), xk)], ["x", "k", "cols", "pos", "dpos", "vals", "bk"], ["n"]
  )
  v, b = sc.sym("v", m.nnz), sc.sym("b", n)
  tables = [(sc.const(cols_t, dtype="int64"), 0, width), (sc.const(pos_t, dtype="int64"), 0, width), (sc.const(diag_pos, dtype="int64"), 0, 1)]
  (sol,) = sc.scan(body, sc.const(np.zeros(n)), [*tables, (v, 0, 0), (b, 0, 1)], length=n, index=True)
  outs = [sol, gradient(sc.sumsqr(sol), v), gradient(sc.sumsqr(sol), b)] if grads else [sol]
  return _fn(f"trsv_{tag}", [v, b], outs)


def test_sparse_forward_substitution_runs_in_place_and_matches_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
  m = _lower_triangle(40, 0.15, 1)
  b = RNG.standard_normal(40)
  (sol,) = _run_both(monkeypatch, lambda tag: _forward_substitution(m, tag, grads=False), (m.data, b), expect_in_place=True)
  np.testing.assert_allclose(sol, spsolve_triangular(m.tocsr(), b, lower=True), rtol=1e-12)
  # Reverse mode reads the stored carries, so that loop keeps every step's carry instead.
  sol2, gv, gb = _run_both(monkeypatch, lambda tag: _forward_substitution(m, f"g{tag}", grads=True), (m.data, b), expect_in_place=False)
  np.testing.assert_array_equal(sol2, sol)

  def value(data: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    s = spsolve_triangular(sparse.csr_array((data, m.indices, m.indptr), shape=m.shape).tocsr(), rhs, lower=True)
    return np.array([s @ s])

  np.testing.assert_allclose(gv, finite_difference(lambda d: value(d, b), m.data).reshape(-1), rtol=1e-6, atol=1e-8)
  np.testing.assert_allclose(gb, finite_difference(lambda r: value(m.data, r), b).reshape(-1), rtol=1e-6, atol=1e-8)


def _accumulate(tag: str, *, read: str, n: int = 12, steps: int = 8, width: int = 3):
  """Step ``k`` adds values into the carry at ``write[k]`` read from the carry at ``read[k]``."""
  rng = np.random.default_rng(7)
  write_t = rng.integers(0, n + 2, size=(steps, width))  # some padded lanes (n, n + 1)
  if read == "disjoint":
    read_t = (write_t + n // 2) % n  # never a written column: n // 2 apart, and n is even
    read_t = np.where(write_t >= n, write_t, read_t)
    for k in range(steps):  # make sure the step's reads avoid all of its writes
      bad = np.isin(read_t[k], write_t[k])
      read_t[k][bad] = n  # a padded read
  elif read == "same":
    read_t = write_t.copy()
  else:  # overlapping at one step only
    read_t = np.full((steps, width), n)
    read_t[5, 1] = write_t[5, 0] if write_t[5, 0] < n else 0
    write_t[5, 0] = read_t[5, 1]
  c, w_idx, r_idx, u = sc.sym("c", n), sc.sym("wi", width, dtype="int64"), sc.sym("ri", width, dtype="int64"), sc.sym("u", width)
  body = sc.Function.from_exprs(
    f"acc_{read}_{tag}", [c, w_idx, r_idx, u], [sc.put_add(c, w_idx, u * sc.take(c, r_idx, fill=1.0))], ["c", "wi", "ri", "u"], ["n"]
  )
  c0, us = sc.sym("c0", n), sc.sym("us", steps * width)
  tables = [(sc.const(write_t.reshape(-1), dtype="int64"), 0, width), (sc.const(read_t.reshape(-1), dtype="int64"), 0, width)]
  (fin,) = sc.scan(body, c0, [*tables, (us, 0, width)], length=steps)
  fn = _fn(f"acc_run_{read}_{tag}", [c0, us], [fin])

  def numpy(c0v: np.ndarray, usv: np.ndarray) -> np.ndarray:
    cv = c0v.copy()
    for k in range(steps):
      vals = usv[k * width : (k + 1) * width] * np.where(read_t[k] < n, cv[np.minimum(read_t[k], n - 1)], 1.0)
      for j in range(width):
        if write_t[k, j] < n:
          cv[write_t[k, j]] += vals[j]
    return cv

  return fn, numpy


@pytest.mark.parametrize(("read", "in_place"), [("disjoint", True), ("same", False), ("one_step", False)])
def test_proof_follows_the_tables(monkeypatch: pytest.MonkeyPatch, read: str, in_place: bool) -> None:
  point = (RNG.standard_normal(12), RNG.standard_normal(24))
  _, numpy = _accumulate("np", read=read)
  (got,) = _run_both(monkeypatch, lambda tag: _accumulate(tag, read=read)[0], point, expect_in_place=in_place)
  np.testing.assert_allclose(got, numpy(*point), rtol=1e-13)


def test_reads_of_an_earlier_link_must_miss_every_later_write(monkeypatch: pytest.MonkeyPatch) -> None:
  """``u2`` reads the carry as it entered the step, after ``u1`` wrote: allowed while ``u1`` and
  ``u2`` write elsewhere, refused once ``u1`` writes what ``u2`` reads. (A value may never read what
  its own update writes, even the same lane: fusion may interleave the lanes.)"""
  n, steps = 10, 6

  def build(tag: str, *, clash: bool) -> sc.Function:
    c, k = sc.sym("c", n), sc.sym("k", (), dtype="int64")
    first = k.reshape((1,))  # the step's own entry
    u1 = sc.put(c, first, sc.take(c, (k + 2).reshape((1,))) * 0.5)
    source = first if clash else sc.const(np.array([9]), dtype="int64")
    u2 = sc.put_add(u1, (k + 1).reshape((1,)), sc.take(c, source))
    body = sc.Function.from_exprs(f"two_{int(clash)}_{tag}", [c, k], [u2], ["c", "k"], ["n"])
    c0 = sc.sym("c0", n)
    (fin,) = sc.scan(body, c0, [], length=steps, index=True)
    return _fn(f"two_run_{int(clash)}_{tag}", [c0], [fin])

  c0 = RNG.standard_normal(n)
  for clash in (False, True):
    ref = c0.copy()
    for k in range(steps):
      old = ref.copy()
      ref[k] = old[k + 2] * 0.5
      ref[k + 1] += old[k] if clash else old[9]
    (got,) = _run_both(monkeypatch, lambda tag, clash=clash: build(tag, clash=clash), (c0,), expect_in_place=not clash)
    np.testing.assert_allclose(got, ref, rtol=1e-14)


def test_data_dependent_indices_keep_two_slots(monkeypatch: pytest.MonkeyPatch) -> None:
  n = 6
  c, k = sc.sym("c", n), sc.sym("k", (), dtype="int64")
  target = sc.stack([c[0].abs().floor()]).cast("int64")  # an index read from the data

  def build(tag: str) -> sc.Function:
    body = sc.Function.from_exprs(f"dd_{tag}", [c, k], [sc.put_add(c, target, sc.stack([1.0]))], ["c", "k"], ["n"])
    c0 = sc.sym("c0", n)
    (fin,) = sc.scan(body, c0, [], length=4, index=True)
    return _fn(f"dd_run_{tag}", [c0], [fin])

  (got,) = _run_both(monkeypatch, build, (np.array([2.5, 0, 0, 0, 0, 0]),), expect_in_place=False)
  np.testing.assert_array_equal(got, [2.5, 0, 4, 0, 0, 0])


def test_another_output_reading_the_carry_keeps_two_slots(monkeypatch: pytest.MonkeyPatch) -> None:
  n = 5
  c, k = sc.sym("c", n), sc.sym("k", (), dtype="int64")

  def build(tag: str) -> sc.Function:
    nxt = sc.put(c, k.reshape((1,)), sc.stack([7.0]))
    body = sc.Function.from_exprs(f"oo_{tag}", [c, k], [nxt, sc.stack([c.sum()])], ["c", "k"], ["n", "s"])
    c0 = sc.sym("c0", n)
    fin, sums = sc.scan(body, c0, [], length=n, index=True)
    return _fn(f"oo_run_{tag}", [c0], [fin, sums])

  got = _run_both(monkeypatch, build, (np.ones(n),), expect_in_place=False)
  np.testing.assert_array_equal(got[0], np.full(n, 7.0))
  np.testing.assert_array_equal(got[1], [5.0, 11.0, 17.0, 23.0, 29.0])


def test_batched_carry_and_mixed_chain(monkeypatch: pytest.MonkeyPatch) -> None:
  """A rank-2 carry updated by ``index_add`` then ``put_add`` with padded lanes: the scratch slots
  sit after all rows."""
  rows, n, steps, width = 3, 7, 5, 4
  table = np.array([[0, 7, 3, 7], [1, 2, -1, 7], [6, 6, 5, 7], [4, 7, 7, 7], [2, 3, 0, 1]])

  def build(tag: str) -> sc.Function:
    c, idx = sc.sym("c", (rows, n)), sc.sym("idx", width, dtype="int64")
    u1 = sc.index_add(c, [0], sc.const([1.0]))
    c0, us = sc.sym("c0", (rows, n)), sc.sym("us", steps * rows * width)
    uf = sc.sym("uf", rows * width)
    flat_body = sc.Function.from_exprs(f"bmf_{tag}", [c, idx, uf], [sc.put_add(u1, idx, uf.reshape((rows, width)))], ["c", "idx", "uf"], ["n"])
    (fin,) = sc.scan(flat_body, c0, [(sc.const(table.reshape(-1), dtype="int64"), 0, width), (us, 0, rows * width)], length=steps)
    return _fn(f"bm_run_{tag}", [c0, us], [fin])

  c0v, usv = RNG.standard_normal((rows, n)), RNG.standard_normal(steps * rows * width)
  (got,) = _run_both(monkeypatch, build, (c0v, usv), expect_in_place=True)
  ref = c0v.copy()
  for k in range(steps):
    ref.reshape(-1)[0] += 1.0
    u = usv[k * rows * width : (k + 1) * rows * width].reshape(rows, width)
    for j, i in enumerate(table[k]):
      if 0 <= i < n:
        ref[:, i] += u[:, j]
  np.testing.assert_allclose(got, ref, rtol=1e-14)


def test_while_loop_with_index_updates_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
  n = 8
  c, k = sc.sym("c", n), sc.sym("k", (), dtype="int64")
  cc = sc.sym("cc", n)

  def build(tag: str) -> sc.Function:
    body = sc.Function.from_exprs(f"wi_{tag}", [c, k], [sc.put(c, k.reshape((1,)), sc.take(c, (k + 1).reshape((1,))) + 1.0)], ["c", "k"], ["n"])
    cond = sc.Function.from_exprs(f"wc_{tag}", [cc], [sc.less(cc.sum(), 40.0)], ["cc"], ["go"])
    c0 = sc.sym("c0", n)
    fin, count = sc.while_loop(cond, body, c0, max_iter=n, index=True)
    return _fn(f"wi_run_{tag}", [c0], [fin, count])

  fin, count = _run_both(monkeypatch, build, (np.arange(8.0),), expect_in_place=True)
  ref, steps = np.arange(8.0), 0
  while steps < n and ref.sum() < 40.0:
    ref[steps] = (ref[steps + 1] if steps + 1 < n else 0.0) + 1.0
    steps += 1
  np.testing.assert_array_equal(fin, ref)
  assert count == steps


def test_proof_helpers_directly() -> None:
  """``update_chain`` finds the chain structurally; ``in_place_steps`` needs the tables."""
  c, i, j = sc.sym("c", 4), sc.sym("i", 1, dtype="int64"), sc.sym("j", 1, dtype="int64")
  body = _normalize_function(sc.Function.from_exprs("ph", [c, i, j], [sc.put(c, i, sc.take(c, j))], ["c", "i", "j"], ["n"]))
  chain = update_chain(body)
  assert chain is not None and [e.op for e in chain] == [sc.ExprOp.PUT]
  assert in_place_steps(body, {1: np.array([[0], [1]]), 2: np.array([[1], [2]])})
  assert not in_place_steps(body, {1: np.array([[0], [1]]), 2: np.array([[1], [1]])})
  assert not in_place_steps(body, {1: np.array([[0], [1]])})  # j unknown
  assert in_place_steps(body, {1: np.array([[0], [4]]), 2: np.array([[4], [4]])})  # both padded at step 1


def test_whole_link_reads_and_row_positions_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
  """A value that reads the whole carry (here through a sum) and one that reads a fixed entry of the
  second row, which a step's column write later reaches, both keep two slots."""
  rows, n = 2, 5
  c, k = sc.sym("c", (rows, n)), sc.sym("k", (), dtype="int64")

  def build(tag: str, kind: str) -> sc.Function:
    col = k.reshape((1,))
    if kind == "sum":
      values = sc.stack([c.sum() * 0.1, c.sum() * 0.2]).reshape((rows, 1))
    else:
      values = sc.stack([c[1, 3], c[1, 3]]).reshape((rows, 1)) + 1.0
    body = sc.Function.from_exprs(f"wl_{kind}_{tag}", [c, k], [sc.put(c, col, values)], ["c", "k"], ["n"])
    c0 = sc.sym("c0", (rows, n))
    (fin,) = sc.scan(body, c0, [], length=n, index=True)
    return _fn(f"wl_run_{kind}_{tag}", [c0], [fin])

  c0 = np.arange(10.0).reshape(rows, n)
  for kind in ("sum", "entry"):
    ref = c0.copy()
    for step in range(n):
      value = [ref.sum() * 0.1, ref.sum() * 0.2] if kind == "sum" else [ref[1, 3] + 1.0] * 2
      ref[:, step] = value
    (got,) = _run_both(monkeypatch, lambda tag, kind=kind: build(tag, kind), (c0,), expect_in_place=False)
    np.testing.assert_allclose(got, ref, rtol=1e-14)


@pytest.mark.parametrize(
  ("op", "read", "in_place"),
  [
    ("floor", (1, 0), False),
    ("ceil", (1, 0), False),
    ("abs", (1, 0), False),
    ("abs", (2, 2), True),
    ("floor", (2, 2), False),
    ("ceil", (2, 2), False),
  ],
)
def test_a_read_through_floor_or_ceil_counts_as_a_read_of_every_entry(
  monkeypatch: pytest.MonkeyPatch, op: str, read: tuple[int, int], in_place: bool
) -> None:
  """``floor`` and ``ceil`` have a zero derivative, so their pattern is empty and says nothing of
  what they read. A body that writes ``floor`` of entry 1 into entry 0 and of entry 0 into entry 1
  reads what it writes; taken for one that reads nothing, it ran in place, and the second write
  read the first one's result. A path through either keeps two slots, wherever it reads; through
  ``abs``, whose pattern is what it reads, the proof follows the entries."""

  def build(tag: str) -> sc.Function:
    x = sc.sym("x", 3)
    body = _fn(f"fl_body_{op}_{read[0]}_{tag}", [x], [sc.index_set(x, [0, 1], getattr(x, op)().gather(np.array(read)) + 0.5)])
    x0 = sc.sym("x0", 3)
    (out,) = sc.scan(body, x0, length=2)
    return _fn(f"fl_{op}_{read[0]}_{tag}", [x0], [out])

  point = (np.array([1.5, -2.5, 3.25]),)
  (got,) = _run_both(monkeypatch, build, point, expect_in_place=in_place)
  ref, f = point[0].copy(), getattr(np, op)
  for _ in range(2):
    ref = np.array([f(ref[read[0]]) + 0.5, f(ref[read[1]]) + 0.5, ref[2]])
  np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("op", ["floor", "ceil", "cast", "take", "copysign", "select", "lt"])
def test_an_op_whose_pattern_is_not_what_it_reads_does_not_claim_exact_reads(op: str) -> None:
  """The in-place proof takes a pattern for a read set only through ops with ``exact_reads``. These
  read entries their derivative pattern omits (a zero derivative, an index, a condition), and with
  the trait on any of them a body that overwrites what it reads would run in place."""
  from scaly.ir.expr import ExprOp, has_trait

  assert not has_trait(getattr(ExprOp, op.upper()), "exact_reads")


def test_a_read_through_a_cast_or_a_run_time_index_keeps_two_slots(monkeypatch: pytest.MonkeyPatch) -> None:
  def cast(tag: str) -> sc.Function:
    x = sc.sym("x", 3)
    body = _fn(f"cast_body_{tag}", [x], [sc.index_set(x, [0, 1], sc.cast(sc.cast(x, "int64"), "float64").gather(np.array([1, 0])) + 0.5)])
    x0 = sc.sym("x0", 3)
    return _fn(f"cast_{tag}", [x0], [sc.scan(body, x0, length=2)[0]])

  point = (np.array([1.5, -2.5, 3.25]),)
  (got,) = _run_both(monkeypatch, cast, point, expect_in_place=False)
  ref = point[0].copy()
  for _ in range(2):
    ref = np.array([np.trunc(ref[1]) + 0.5, np.trunc(ref[0]) + 0.5, ref[2]])
  np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("reads_written", [False, True])
def test_constant_index_updates_without_step_inputs(monkeypatch: pytest.MonkeyPatch, reads_written: bool) -> None:
  """A body with no input sliced per step and constant update indices: one step stands for all.
  ``c = [a | x | r]``; ``x += f(a, r)`` then ``r = g(a, x_new)`` runs in place. Computing ``x``'s
  new value from the old ``x`` (a read of an entry the same update writes) keeps two slots."""
  n = 3

  def build(tag: str) -> sc.Function:
    c = sc.sym("c", 3 * n)
    a, x, r = c[:n], c[n : 2 * n], c[2 * n :]
    ix, ir = sc.const(np.arange(n, 2 * n), dtype="int64"), sc.const(np.arange(2 * n, 3 * n), dtype="int64")
    if reads_written:
      u1 = sc.put(c, ix, x + (a * r).sin(), in_range=True)
    else:
      u1 = sc.put_add(c, ix, (a * r).sin(), in_range=True)
    body = _fn(f"ci_body_{tag}_{reads_written}", [c], [sc.put(u1, ir, a - 0.5 * u1[n : 2 * n], in_range=True)])
    cond = _fn(f"ci_cond_{tag}_{reads_written}", [c], [sc.norm_inf(c[2 * n :]) > 1e-9])
    init = sc.sym("init", 3 * n)
    out, count = sc.while_loop(cond, body, init, max_iter=30)
    (scanned,) = sc.scan(body, init, length=7)
    return _fn(f"ci_{tag}_{reads_written}", [init], [out, count, scanned])

  point = (np.r_[np.linspace(0.2, 0.6, n), np.zeros(n), np.ones(n)],)
  got = _run_both(monkeypatch, build, point, expect_in_place=not reads_written)
  c = point[0].copy()
  a = c[:n]
  steps = []
  for _ in range(30):
    if np.abs(c[2 * n :]).max() <= 1e-9:
      break
    c[n : 2 * n] += np.sin(a * c[2 * n :])
    c[2 * n :] = a - 0.5 * c[n : 2 * n]
    steps.append(c.copy())
  np.testing.assert_allclose(got[0], c, rtol=1e-14)
  assert got[1] == len(steps)
  assert len(steps) > 6
  np.testing.assert_allclose(got[2], steps[6], rtol=1e-14)
