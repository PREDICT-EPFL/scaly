"""Sparse ``L D L^T`` of a symmetric quasi-definite matrix, generated as loops over its columns.

``SparseLDL(K)`` analyzes ``K``'s pattern at build time (``linalg.symbolic``) and factors its
values as straight-line code when that is small (``schedule="unroll"``) and otherwise as one loop
nest, ``ir.expr.sparse_ldl_factor`` (``schedule="loop"``), which updates each column from chunks of
columns that share their rows. ``schedule="scan"`` keeps the factorization differentiable through
its loops: one ``scan`` per column segment, where every step is the left-looking column update, written
with run-time-index operations on a single carry vector ``[L values | D | work | 0 | scratch]``:
for each column ``k`` in row ``j`` of ``L``, one ``ragged_add`` run over the contiguous part of
column ``k`` from row ``j`` down, the loop the C reference factorizations have. The loop slices the
analysis tables one padded row per step, so the carry is proven safe to update in place and nothing
is copied between steps; the tables are the size of ``K`` and ``L``, not of the factorization's
work. ``solve`` runs the two triangular sweeps the same way. The factorization is separate from the
solve, so one factorization serves several right-hand sides, and the solve carries the implicit
derivative ``dx = K^{-1} (db - dK x)``: differentiating a solve never differentiates the loops.
"""

from __future__ import annotations

import itertools
from typing import Any, Literal

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from ..function.model import Function
from ..function.sugar import custom_derivative, scan, vmap, while_loop
from ..ir.expr import (
  SPARSE_LDL_MAX_WIDTH,
  Expr,
  as_expr,
  concat,
  gather,
  isfinite,
  logical_and,
  maximum,
  norm_inf,
  put,
  put_add,
  ragged_add,
  ragged_dot,
  scatter,
  segment_sum,
  sparse_ldl_factor,
  stack,
  take,
  where,
)
from ..utils.options import get_options
from .sparse import SparseMatrix
from .symbolic import CostModel, Ordering, Segment, SymbolicLDL, analyze

Schedule = Literal["auto", "loop", "scan", "unroll"]
SCHEDULES: tuple[Schedule, ...] = ("auto", "loop", "scan", "unroll")

_NAMES = itertools.count()


def _table(ptr: np.ndarray, data: np.ndarray, seg: Segment, width: int, *, offset: int = 0, pad: int | np.ndarray = -1) -> np.ndarray:
  """Columns ``seg.start ... seg.stop - 1`` of a ragged table, each padded to ``width``, flattened row
  by row; ``offset`` is added to every real entry. ``pad`` fills the padding: one index, or one per
  lane (a scratch slot each)."""
  width = max(width, 1)
  out = np.empty((seg.length, width), dtype=np.int64)
  out[:] = np.broadcast_to(np.asarray(pad, dtype=np.int64), (width,))
  for t, j in enumerate(range(seg.start, seg.stop)):
    lo, hi = ptr[j], ptr[j + 1]
    out[t, : hi - lo] = data[lo:hi] + offset
  return out.reshape(-1)


def _call(fn: Function, *args: Expr) -> Expr:
  """The single output of ``fn`` applied to ``args``."""
  out = fn.symbolic_call(tuple(args))
  return out[0] if isinstance(out, tuple) else out


def _int_sym(name: str, size: int) -> Expr:
  return Expr.sym(name, (size,), dtype="int64")


class SparseLDL:
  """The ``L D L^T`` factorization of a symmetric ``SparseMatrix`` without pivoting.

  ``K`` must be quasi-definite (or definite): every pivot of the permuted matrix is then nonzero
  whatever the ordering. The pattern of ``K`` may hold its lower triangle, its upper triangle or
  both; a mirrored pair is read from its lower entry. ``values`` is the factor, laid out as
  ``[L below the diagonal (CSC of the permuted matrix) | D]``; ``solve`` uses it.

  The loops index without bounds checks: every range comes from the analysis, and a padded lane
  reads a zero entry, writes a scratch slot of its own or runs an empty range.
  """

  def __init__(
    self,
    matrix: SparseMatrix,
    *,
    ordering: Ordering = "auto",
    schedule: Schedule = "auto",
    symbolic: SymbolicLDL | None = None,
    cost: CostModel | None = None,
    name: str | None = None,
  ) -> None:
    if matrix.shape[0] != matrix.shape[1]:
      raise ValueError(f"SparseLDL needs a square matrix, got {matrix.shape}")
    if schedule not in SCHEDULES:
      raise ValueError(f"schedule must be one of {SCHEDULES}, got {schedule!r}")
    rows, cols = matrix.coordinates()
    self.matrix = matrix
    if symbolic is not None:
      symbolic.check(matrix.shape, rows, cols)
    self.symbolic = symbolic if symbolic is not None else analyze(matrix.shape, rows, cols, ordering)
    s = self.symbolic
    self.n = s.n
    self.name = name or f"sldl{next(_NAMES)}"
    self.l_size = s.nnz_l
    self.d_offset = s.nnz_l
    self.w_offset = s.nnz_l + s.n
    if schedule == "auto":
      schedule = "unroll" if self.work <= get_options().sparse_unroll else "loop"
    self.schedule: Schedule = schedule
    self._sweeps: dict[bool, Function] = {}
    self._solvers: dict[tuple[int, float | None], Function] = {}
    self.segments: list[Segment] = []
    if schedule == "unroll":
      self.values = self._factor_unrolled(matrix.values)
      return
    if schedule == "loop":
      self.values = sparse_ldl_factor(matrix.values, self.tables())
      return
    self.zero = s.nnz_l + 2 * s.n  # an entry that stays zero: padded reads land here
    # Per column: matrix entries, columns in its row of L (one ragged run each), entries of its column.
    widths = np.stack([np.diff(s.a_ptr), np.diff(s.r_ptr), np.diff(s.l_ptr)], axis=1)
    # Every loop copies its carry in once, so a segment costs at least the carry's size.
    self.segments = s.segments(cost or CostModel(step=8.0, a=1.0, u=3.0, c=1.0, segment=256.0 + 0.5 * self.zero), widths=widths)
    lanes = max((max(seg.a, seg.c, 1) for seg in self.segments), default=1)
    self.dump = self.zero + 1  # one scratch slot per lane for padded writes
    self.size = self.dump + lanes
    self.values = self._factor(matrix.values)[: self.w_offset]

  def tables(self) -> dict[str, np.ndarray]:
    """The analysis as ``ir.expr.sparse_ldl_factor`` reads it."""
    s = self.symbolic
    return {
      "a_ptr": s.a_ptr,
      "a_rows": s.a_rows,
      "a_src": s.a_source,
      "l_ptr": s.l_ptr,
      "l_rows": s.l_rows,
      "r_cols": s.r_cols,
      "r_pos": s.r_pos,
      **s.chunks(SPARSE_LDL_MAX_WIDTH),
    }

  @property
  def work(self) -> int:
    """Multiply-adds and divisions of the factorization: what ``schedule="auto"`` compares with
    ``sc.options(sparse_unroll=...)``."""
    return self.symbolic.update_lanes + self.symbolic.nnz_l

  # --- the factorization ----------------------------------------------------------------------

  def _factor_body(self, seg: Segment, k: int) -> Function:
    a, g, c = max(seg.a, 1), max(seg.u, 1), max(seg.c, 1)
    s = self.symbolic
    carry = Expr.sym("c", (self.size,))
    j = Expr.sym("j", (), dtype="int64")
    a_idx, a_src = _int_sym("a_idx", a), _int_sym("a_src", a)
    r_lo, r_hi, r_dk, r_jk = (_int_sym(nm, g) for nm in ("r_lo", "r_hi", "r_dk", "r_jk"))
    col, clear = _int_sym("col", 2), _int_sym("clear", c)
    kv = Expr.sym("kv", (self.matrix.nnz,))
    ok = {"in_range": True}
    # w[rows of column j] = K[., j]
    u1 = put(carry, a_idx, take(kv, a_src, **ok), **ok)
    # For each k in row j of L: w[i] -= L[i, k] * (D[k] L[j, k]) over column k from row j down.
    weight = -(take(carry, r_dk, **ok) * take(carry, r_jk, **ok))
    u2 = ragged_add(u1, carry, r_lo, r_hi, weight, dst_map=self.w_offset + s.l_rows)
    # D[j] = w[j]; L[., j] = w[rows] / D[j]; then clear the work entries the column used.
    wj = take(u2, (j + self.w_offset).reshape((1,)), **ok)
    u3 = put(u2, (j + self.d_offset).reshape((1,)), wj, **ok)
    u4 = ragged_add(u3, u2, col[:1], col[1:], 1.0 / wj, src_map=self.w_offset + s.l_rows)
    u5 = put(u4, clear, Expr.const(np.zeros(c)), **ok)
    inputs = [carry, j, a_idx, a_src, r_lo, r_hi, r_dk, r_jk, col, clear, kv]
    names = ["c", "j", "a_idx", "a_src", "r_lo", "r_hi", "r_dk", "r_jk", "col", "clear", "kv"]
    return Function._from_exprs(f"{self.name}_f{k}", inputs, [u5], names, ["c_next"])

  def _factor_tables(self, seg: Segment) -> list[np.ndarray]:
    s = self.symbolic
    a, g, c = max(seg.a, 1), max(seg.u, 1), max(seg.c, 1)
    j = np.arange(seg.start, seg.stop, dtype=np.int64)
    dump = lambda width: self.dump + np.arange(width)  # noqa: E731
    row_k = _table(s.r_ptr, s.r_cols, seg, g, pad=0).reshape(seg.length, g)
    real = _table(s.r_ptr, np.ones(s.r_cols.size, dtype=np.int64), seg, g, pad=0).reshape(seg.length, g) > 0
    r_pos = _table(s.r_ptr, s.r_pos, seg, g, pad=0).reshape(seg.length, g)
    r_lo = np.where(real, r_pos, 0)
    r_hi = np.where(real, s.l_ptr[row_k + 1], 0)  # an empty range for a padded group
    r_dk = np.where(real, self.d_offset + row_k, self.zero)
    r_jk = np.where(real, r_pos, self.zero)
    col = np.stack([s.l_ptr[j], s.l_ptr[j + 1]], axis=1)
    return [
      j,
      _table(s.a_ptr, s.a_rows, seg, a, offset=self.w_offset, pad=dump(a)),
      _table(s.a_ptr, s.a_source, seg, a, pad=0),
      r_lo.reshape(-1),
      r_hi.reshape(-1),
      r_dk.reshape(-1),
      r_jk.reshape(-1),
      col.reshape(-1),
      _table(s.l_ptr, s.l_rows, seg, c, offset=self.w_offset, pad=dump(c)),
    ]

  def _factor(self, kv: Expr) -> Expr:
    carry: Expr = Expr.const(np.zeros(self.size))
    for k, seg in enumerate(self.segments):
      body = self._factor_body(seg, k)
      tables = self._factor_tables(seg)
      widths = [1, *(t.size // seg.length for t in tables[1:])]
      xs = [(Expr.const(t, dtype="int64"), 0, w) for t, w in zip(tables, widths, strict=True)]
      (carry,) = scan(body, carry, [*xs, (kv, 0, 0)], length=seg.length)
    return carry

  def _factor_unrolled(self, kv: Expr) -> Expr:
    """The same left-looking factorization as straight-line code: one scalar expression per entry
    of ``L`` and ``D``, for a matrix small enough that loop overhead would dominate."""
    s = self.symbolic
    lv: dict[int, Expr] = {}
    dv: list[Expr] = []
    for j in range(s.n):
      w: dict[int, Expr] = {int(s.a_rows[p]): kv[int(s.a_source[p])] for p in range(s.a_ptr[j], s.a_ptr[j + 1])}
      for q in range(s.r_ptr[j], s.r_ptr[j + 1]):
        k, pos = int(s.r_cols[q]), int(s.r_pos[q])
        t = dv[k] * lv[pos]
        for p in range(pos, s.l_ptr[k + 1]):
          i = int(s.l_rows[p])
          w[i] = w[i] - lv[p] * t if i in w else -(lv[p] * t)
      dj = w.get(j, Expr.const(0.0))
      dv.append(dj)
      for p in range(s.l_ptr[j], s.l_ptr[j + 1]):
        lv[p] = w[int(s.l_rows[p])] / dj
    return stack([*(lv[p] for p in range(s.nnz_l)), *dv]) if s.n else Expr.const(np.zeros(0))

  @property
  def l_values(self) -> Expr:
    """The entries of ``L`` below the diagonal, CSC of the permuted matrix (``symbolic.l_ptr``/``l_rows``)."""
    return self.values[: self.l_size]

  @property
  def d(self) -> Expr:
    """The diagonal of ``D``, in the permuted order."""
    return self.values[self.d_offset : self.d_offset + self.n]

  # --- health -------------------------------------------------------------------------------

  def inertia(self) -> Expr:
    """The counts of positive, negative and other (zero or NaN) pivots in ``D``, as a ``float64``
    vector of 3. For a quasi-definite ``K = [[H, A^T], [A, -G]]`` it is ``(n_H, n_G, 0)``."""
    d = self.d
    one, zero = Expr.const(1.0), Expr.const(0.0)
    pos = where(d > 0.0, one, zero).sum()
    neg = where(d < 0.0, one, zero).sum()
    return stack([pos, neg, float(self.n) - pos - neg])

  def health(self, *, signs: Any = None, pivot_tol: float = 0.0, x: Any = None) -> Expr:
    """``True`` when every pivot is finite with ``|D[j]| > pivot_tol`` (with ``signs``, a vector of
    ``+1``/``-1`` per row of ``K`` in its own order: ``signs[i] * D > pivot_tol``, the quasi-definite
    sign pattern), and, given a solution ``x``, every entry of ``x`` is finite. A bool scalar,
    computed in the generated code next to the factorization: no pivoting happens, so this is the
    check that a regularization was large enough."""
    d = self.d
    if signs is None:
      ok = logical_and(isfinite(d), d.abs() > pivot_tol)
    else:
      sgn = np.asarray(signs, dtype=np.float64)
      if sgn.shape != (self.n,) or not np.all(np.abs(sgn) == 1.0):
        raise ValueError(f"signs must be {self.n} entries of +1 or -1")
      ok = logical_and(isfinite(d), d * Expr.const(sgn[self.symbolic.perm]) > pivot_tol)
    bad = where(ok, 0.0, 1.0).sum()
    if x is not None:
      bad = bad + where(isfinite(as_expr(x)), 0.0, 1.0).sum()
    return bad < 0.5

  # --- the solves -----------------------------------------------------------------------------

  def _sweep_body(self, backward: bool) -> Function:
    """One column of the unit lower sweep (forward) or of its transpose (backward); built once per
    factorization, so every solve in a graph calls the same procedure."""
    if backward not in self._sweeps:
      s = self.symbolic
      n = self.n
      yy, j, col, ff = Expr.sym("y", (n,)), Expr.sym("j", (), dtype="int64"), _int_sym("col", 2), Expr.sym("f", (self.w_offset,))
      if backward:
        # x[j] -= sum over the rows i > j of column j of L[i, j] x[i]
        dot = ragged_dot(ff, yy, col[:1], col[1:], b_map=s.l_rows)
        nxt = put_add(yy, j.reshape((1,)), -dot, in_range=True)
      else:
        # y[i] -= L[i, j] y[j] for the rows i > j of column j
        nxt = ragged_add(yy, ff, col[:1], col[1:], -take(yy, j.reshape((1,)), in_range=True), dst_map=s.l_rows)
      tag = "b" if backward else "f"
      self._sweeps[backward] = Function._from_exprs(f"{self.name}_s{tag}", [yy, j, col, ff], [nxt], ["y", "j", "col", "f"], ["y_next"])
    return self._sweeps[backward]

  def _sweep(self, f: Expr, y: Expr, backward: bool) -> Expr:
    """The unit lower sweep (forward) or its transpose (backward), one column per step."""
    s = self.symbolic
    n = self.n
    if n == 0:
      return y
    steps = np.arange(n, dtype=np.int64)
    if backward:
      steps = steps[::-1].copy()
    cols = np.stack([s.l_ptr[steps], s.l_ptr[steps + 1]], axis=1).reshape(-1)
    xs = [(Expr.const(steps, dtype="int64"), 0, 1), (Expr.const(cols, dtype="int64"), 0, 2), (f, 0, 0)]
    (y,) = scan(self._sweep_body(backward), y, xs, length=n)
    return y

  def _raw_solve(self, f: Expr, b: Expr) -> Expr:
    """``K^{-1} b`` from the factor ``f``, differentiated (if at all) through its loops."""
    if self.schedule == "unroll":
      return self._unrolled_solve(f, b)
    s = self.symbolic
    y = self._sweep(f, gather(b, s.perm), backward=False)
    x = self._sweep(f, y / f[self.d_offset : self.d_offset + self.n], backward=True)
    return gather(x, s.iperm)

  def _unrolled_solve(self, f: Expr, b: Expr) -> Expr:
    """The two sweeps as straight-line code, one scalar expression per entry."""
    s = self.symbolic
    n = self.n
    if n == 0:
      return b
    y = [b[int(s.perm[i])] for i in range(n)]
    for j in range(n):
      for p in range(s.l_ptr[j], s.l_ptr[j + 1]):
        i = int(s.l_rows[p])
        y[i] = y[i] - f[p] * y[j]
    y = [y[j] / f[self.d_offset + j] for j in range(n)]
    for j in range(n - 1, -1, -1):
      for p in range(s.l_ptr[j], s.l_ptr[j + 1]):
        y[j] = y[j] - f[p] * y[int(s.l_rows[p])]
    return stack([y[int(s.iperm[i])] for i in range(n)])

  def _refined_solve(self, f: Expr, kv: Expr, b: Expr, refine: int, tol: float | None, tag: str = "") -> Expr:
    """``K^{-1} b`` with iterative refinement: ``x += K^{-1} (b - K x)``, ``refine`` times, or with
    ``tol`` while ``||b - K x||_inf > tol * max(1, ||b||_inf)`` and at most ``refine`` times."""
    x = self._raw_solve(f, b)
    if refine == 0:
      return x
    if tol is None:
      for _ in range(refine):
        x = x + self._raw_solve(f, b - self._k_times(kv, x))
      return x
    # The loop carries [x | r]; the factor, K's values, b and the threshold are its params, read in
    # place every step. x += K^{-1} r, then r = b - K x reading the updated x: each update reads only
    # entries no later update writes, so the loop overwrites its carry in place.
    n = self.n
    c = Expr.sym("c", (2 * n,))
    pf, pk, pb, pt = Expr.sym("f", f.shape), Expr.sym("k", kv.shape), Expr.sym("b", b.shape), Expr.sym("threshold", (1,))
    xs, rs = Expr.const(np.arange(n), dtype="int64"), Expr.const(np.arange(n, 2 * n), dtype="int64")
    u1 = put_add(c, xs, self._raw_solve(pf, c[n:]), in_range=True)
    body_out = put(u1, rs, pb - self._k_times(pk, u1[:n]), in_range=True)
    params = [pf, pk, pb, pt]
    names = ["c", "f", "k", "b", "threshold"]
    body = Function._from_exprs(f"{self.name}{tag}_refine", [c, *params], [body_out], names, ["c_next"])
    cond = Function._from_exprs(f"{self.name}{tag}_refining", [c, *params], [norm_inf(c[n:]) > pt[0]], names, ["go"])
    threshold = (tol * maximum(1.0, norm_inf(b))).reshape((1,))
    out, _ = while_loop(cond, body, concat([x, b - self._k_times(kv, x)]), max_iter=refine, params=(f, kv, b, threshold))
    return out[:n]

  def _k_times(self, kv: Expr, x: Expr) -> Expr:
    """``K x`` for values ``kv`` in ``K``'s pattern, reading each mirrored pair once, as the
    factorization does: the entries it reads, symmetrically completed."""
    rows, cols = self.matrix.coordinates()
    used = self.symbolic.a_source
    r, c = rows[used], cols[used]
    off = np.flatnonzero(r != c)
    v = gather(kv, used)
    terms = [v * gather(x, c)]
    if off.size:
      terms.append(gather(v, off) * gather(x, r[off]))
    ids = np.concatenate([r, c[off]])
    return segment_sum(concat(terms) if len(terms) > 1 else terms[0], ids, self.n)

  def _solver_function(self, refine: int = 0, tol: float | None = None) -> Function:
    """The solve as a ``Function`` of ``(factor, K values, b)`` with implicit derivative rules.

    The factor is a function of ``K``'s values computed outside; its own derivative is taken to be
    zero here and the whole derivative flows through the values: ``dx = K^{-1} (db - dK x)``, and in
    reverse ``bbar = K^{-1} xbar``, ``Kbar = -bbar x^T`` on the entries the factorization reads. The
    rules solve with the same factor through a solve that has rules of its own, so second
    derivatives are implicit too; only third derivatives would go through the loops. Refinement,
    if any, is part of every solve, the rules' included."""
    if refine == 0:
      tol = None
    key = (refine, tol)
    if key not in self._solvers:
      f, kv, b = Expr.sym("f", (self.w_offset,)), Expr.sym("kv", (self.matrix.nnz,)), Expr.sym("b", (self.n,))
      # Every variant, and every rule of it, has a name of its own: several can meet in one graph.
      tag = "" if refine == 0 else f"_r{refine}" + ("" if tol is None else f"a{sum(t is not None for _, t in self._solvers)}")
      x = self._refined_solve(f, kv, b, refine, tol, tag)
      base = Function._from_exprs(f"{self.name}_solve{tag}", [f, kv, b], [x], ["f", "kv", "b"], ["x"])
      inner = base
      pattern = self._solve_sparsity()
      for level in (1, 2):
        inner = custom_derivative(base, jvp=self._jvp_rule(inner, level, tag), vjp=self._vjp_rule(inner, level, tag), sparsity=pattern)
      self._solvers[key] = inner
    return self._solvers[key]

  def _solve_sparsity(self) -> Any:
    """The pattern of the solution: ``x[i]`` depends on ``b[j]`` and on the entries of ``K`` in
    ``j``'s connected component of ``K``'s graph, and not on the factor, whose derivative the
    rules take to be zero. The body's own pattern, through run-time indices, would be dense and slow
    to compute."""
    n = self.n
    rows, cols = self.matrix.coordinates()
    graph = sparse.csr_array((np.ones(rows.size), (rows, cols)), shape=(n, n))
    _, label = connected_components(graph, directed=False)
    used = np.zeros(self.matrix.nnz, dtype=bool)
    used[self.symbolic.a_source] = True
    # member[i, c]: row i lies in component c. Built on request: one component makes both dense.
    member = sparse.csr_array((np.ones(n, dtype=bool), (np.arange(n), label)), shape=(n, int(label.max(initial=-1)) + 1))
    cache: dict[int, Any] = {}

    def pattern(output: int, k: int) -> Any:
      if k == 0:
        return None
      if k not in cache:
        other = member if k == 2 else sparse.csr_array(member[rows] * used[:, None])
        cache[k] = sparse.csr_array(member @ other.T, dtype=bool)
      return cache[k]

    return pattern

  def _jvp_rule(self, inner: Function, level: int, tag: str) -> Function:
    f, kv, b = Expr.sym("f", (self.w_offset,)), Expr.sym("kv", (self.matrix.nnz,)), Expr.sym("b", (self.n,))
    df, dkv, db = Expr.sym("df", (self.w_offset,)), Expr.sym("dkv", (self.matrix.nnz,)), Expr.sym("db", (self.n,))
    x = _call(inner, f, kv, b)
    dx = _call(inner, f, kv, db - self._k_times(dkv, x))
    names = ["f", "kv", "b", "df", "dkv", "db"]
    return Function._from_exprs(f"{self.name}_solve{tag}_jvp{level}", [f, kv, b, df, dkv, db], [dx], names, ["dx"])

  def _vjp_rule(self, inner: Function, level: int, tag: str) -> Function:
    f, kv, b = Expr.sym("f", (self.w_offset,)), Expr.sym("kv", (self.matrix.nnz,)), Expr.sym("b", (self.n,))
    x, xbar = Expr.sym("xo", (self.n,)), Expr.sym("xbar", (self.n,))
    bbar = _call(inner, f, kv, xbar)
    rows, cols = self.matrix.coordinates()
    used = self.symbolic.a_source
    r, c = rows[used], cols[used]
    grad_used = -(gather(bbar, r) * gather(x, c))
    off = np.flatnonzero(r != c)
    if off.size:
      grad_used = grad_used - scatter(gather(bbar, c[off]) * gather(x, r[off]), off, (used.size,))
    kbar = scatter(grad_used, used, (self.matrix.nnz,))
    outs = [Expr.const(np.zeros(self.w_offset)), kbar, bbar]
    return Function._from_exprs(
      f"{self.name}_solve{tag}_vjp{level}", [f, kv, b, x, xbar], outs, ["f", "kv", "b", "xo", "xbar"], ["fbar", "kvbar", "bbar"]
    )

  def solve_with(self, factor: Any, b: Any) -> Expr:
    """``K^{-1} b`` from ``factor``, a factorization in the layout of ``values`` computed elsewhere
    for this analysis (carried out of a loop that retries it, say): the two sweeps alone, with
    neither refinement nor derivative rules."""
    f, b = as_expr(factor), as_expr(b)
    if f.shape != (self.w_offset,):
      raise ValueError(f"solve_with needs a factor of length {self.w_offset}, got {f.shape}")
    if b.shape != (self.n,):
      raise ValueError(f"solve_with needs a right-hand side of length {self.n}, got {b.shape}")
    return self._raw_solve(f, b)

  def solve(self, b: Any, *, refine: int = 0, tol: float | None = None) -> Expr:
    """``K^{-1} b`` for a vector or a matrix of right-hand sides (``(n,)`` or ``(n, m)``).

    Differentiable in ``K``'s values and in ``b`` by the implicit rule: the tangent is one more
    solve with the same factor, and reverse mode is one transposed solve and an outer product on
    ``K``'s pattern. The factorization loops are never differentiated.

    ``refine`` adds steps of iterative refinement, ``x += K^{-1} (b - K x)``, each one more solve
    and one product with ``K``: exactly ``refine`` of them, or, with ``tol``, only while
    ``||b - K x||_inf > tol * max(1, ||b||_inf)`` (a ``while_loop`` of at most ``refine`` steps)."""
    if not isinstance(refine, int) or isinstance(refine, bool) or refine < 0:
      raise ValueError(f"refine must be a non-negative integer, got {refine!r}")
    if tol is not None and not tol > 0.0:
      raise ValueError(f"tol must be positive, got {tol!r}")
    b = as_expr(b)
    fn = self._solver_function(refine, None if tol is None else float(tol))
    if len(b.shape) == 1:
      if b.shape != (self.n,):
        raise ValueError(f"solve needs a right-hand side of length {self.n}, got {b.shape}")
      return _call(fn, self.values, self.matrix.values, b)
    if len(b.shape) != 2 or b.shape[0] != self.n:
      raise ValueError(f"solve needs a right-hand side of {self.n} rows, got {b.shape}")
    m = b.shape[1]
    stacked = b.T.reshape((m * self.n,))
    cols = vmap(fn, m, [(self.values, 0, 0), (self.matrix.values, 0, 0), (stacked, 0, self.n)])
    return cols.reshape((m, self.n)).T


def sparse_ldl(matrix: SparseMatrix, **options: Any) -> SparseLDL:
  """``SparseLDL(matrix, **options)``: the factorization of a symmetric quasi-definite matrix."""
  return SparseLDL(matrix, **options)
