"""Sparse ``L D L^T`` of a symmetric quasi-definite matrix, generated as loops over its columns.

``SparseLDL(K)`` analyzes ``K``'s pattern at build time (``linalg.symbolic``) and factors its
values with one ``scan`` per column segment. Every step is the left-looking column update, written
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
from typing import Any

import numpy as np

from ..function.model import Function
from ..function.sugar import custom_derivative, scan, vmap
from ..ir.expr import Expr, as_expr, concat, gather, put, put_add, ragged_add, ragged_dot, scatter, segment_sum, take
from .sparse import SparseMatrix
from .symbolic import CostModel, Ordering, Segment, SymbolicLDL, analyze

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
    symbolic: SymbolicLDL | None = None,
    cost: CostModel | None = None,
    name: str | None = None,
  ) -> None:
    if matrix.shape[0] != matrix.shape[1]:
      raise ValueError(f"SparseLDL needs a square matrix, got {matrix.shape}")
    rows, cols = matrix.coordinates()
    self.matrix = matrix
    self.symbolic = symbolic if symbolic is not None else analyze(matrix.shape, rows, cols, ordering)
    s = self.symbolic
    self.n = s.n
    self.name = name or f"sldl{next(_NAMES)}"
    self.l_size = s.nnz_l
    self.d_offset = s.nnz_l
    self.w_offset = s.nnz_l + s.n
    self.zero = s.nnz_l + 2 * s.n  # an entry that stays zero: padded reads land here
    # Per column: matrix entries, columns in its row of L (one ragged run each), entries of its column.
    widths = np.stack([np.diff(s.a_ptr), np.diff(s.r_ptr), np.diff(s.l_ptr)], axis=1)
    # Every loop copies its carry in once, so a segment costs at least the carry's size.
    self.segments = s.segments(cost or CostModel(step=8.0, a=1.0, u=3.0, c=1.0, segment=256.0 + 0.5 * self.zero), widths=widths)
    lanes = max((max(seg.a, seg.c, 1) for seg in self.segments), default=1)
    self.dump = self.zero + 1  # one scratch slot per lane for padded writes
    self.size = self.dump + lanes
    self.values = self._factor(matrix.values)[: self.w_offset]
    self._solver: Function | None = None

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

  @property
  def l_values(self) -> Expr:
    """The entries of ``L`` below the diagonal, CSC of the permuted matrix (``symbolic.l_ptr``/``l_rows``)."""
    return self.values[: self.l_size]

  @property
  def d(self) -> Expr:
    """The diagonal of ``D``, in the permuted order."""
    return self.values[self.d_offset : self.d_offset + self.n]

  # --- the solves -----------------------------------------------------------------------------

  def _sweep(self, f: Expr, y: Expr, backward: bool) -> Expr:
    """The unit lower sweep (forward) or its transpose (backward), one column per step."""
    s = self.symbolic
    n = self.n
    if n == 0:
      return y
    yy, j, col, ff = Expr.sym("y", (n,)), Expr.sym("j", (), dtype="int64"), _int_sym("col", 2), Expr.sym("f", (self.w_offset,))
    if backward:
      # x[j] -= sum over the rows i > j of column j of L[i, j] x[i]
      dot = ragged_dot(ff, yy, col[:1], col[1:], b_map=s.l_rows)
      nxt = put_add(yy, j.reshape((1,)), -dot, in_range=True)
    else:
      # y[i] -= L[i, j] y[j] for the rows i > j of column j
      nxt = ragged_add(yy, ff, col[:1], col[1:], -take(yy, j.reshape((1,)), in_range=True), dst_map=s.l_rows)
    tag = "b" if backward else "f"
    body = Function._from_exprs(f"{self.name}_s{tag}", [yy, j, col, ff], [nxt], ["y", "j", "col", "f"], ["y_next"])
    steps = np.arange(n, dtype=np.int64)
    if backward:
      steps = steps[::-1].copy()
    cols = np.stack([s.l_ptr[steps], s.l_ptr[steps + 1]], axis=1).reshape(-1)
    xs = [(Expr.const(steps, dtype="int64"), 0, 1), (Expr.const(cols, dtype="int64"), 0, 2), (f, 0, 0)]
    (y,) = scan(body, y, xs, length=n)
    return y

  def _raw_solve(self, f: Expr, b: Expr) -> Expr:
    """``K^{-1} b`` from the factor ``f``, differentiated (if at all) through its loops."""
    s = self.symbolic
    y = self._sweep(f, gather(b, s.perm), backward=False)
    x = self._sweep(f, y / f[self.d_offset : self.d_offset + self.n], backward=True)
    return gather(x, s.iperm)

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

  def _solver_function(self) -> Function:
    """The solve as a ``Function`` of ``(factor, K values, b)`` with implicit derivative rules.

    The factor is a function of ``K``'s values computed outside; its own derivative is taken to be
    zero here and the whole derivative flows through the values: ``dx = K^{-1} (db - dK x)``, and in
    reverse ``bbar = K^{-1} xbar``, ``Kbar = -bbar x^T`` on the entries the factorization reads. The
    rules solve with the same factor through a solve that has rules of its own, so second
    derivatives are implicit too; only third derivatives would go through the loops."""
    if self._solver is not None:
      return self._solver
    f, kv, b = Expr.sym("f", (self.w_offset,)), Expr.sym("kv", (self.matrix.nnz,)), Expr.sym("b", (self.n,))
    base = Function._from_exprs(f"{self.name}_solve", [f, kv, b], [self._raw_solve(f, b)], ["f", "kv", "b"], ["x"])
    inner = base
    for level in (1, 2):
      inner = custom_derivative(base, jvp=self._jvp_rule(inner, level), vjp=self._vjp_rule(inner, level))
    self._solver = inner
    return inner

  def _jvp_rule(self, inner: Function, level: int) -> Function:
    f, kv, b = Expr.sym("f", (self.w_offset,)), Expr.sym("kv", (self.matrix.nnz,)), Expr.sym("b", (self.n,))
    df, dkv, db = Expr.sym("df", (self.w_offset,)), Expr.sym("dkv", (self.matrix.nnz,)), Expr.sym("db", (self.n,))
    x = _call(inner, f, kv, b)
    dx = _call(inner, f, kv, db - self._k_times(dkv, x))
    names = ["f", "kv", "b", "df", "dkv", "db"]
    return Function._from_exprs(f"{self.name}_solve_jvp{level}", [f, kv, b, df, dkv, db], [dx], names, ["dx"])

  def _vjp_rule(self, inner: Function, level: int) -> Function:
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
    return Function._from_exprs(f"{self.name}_solve_vjp{level}", [f, kv, b, x, xbar], outs, ["f", "kv", "b", "xo", "xbar"], ["fbar", "kvbar", "bbar"])

  def solve(self, b: Any) -> Expr:
    """``K^{-1} b`` for a vector or a matrix of right-hand sides (``(n,)`` or ``(n, m)``).

    Differentiable in ``K``'s values and in ``b`` by the implicit rule: the tangent is one more
    solve with the same factor, and reverse mode is one transposed solve and an outer product on
    ``K``'s pattern. The factorization loops are never differentiated."""
    b = as_expr(b)
    fn = self._solver_function()
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
