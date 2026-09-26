"""Sparse ``L D L^T`` of a symmetric quasi-definite matrix, generated as loops over its columns.

``SparseLDL(K)`` analyzes ``K``'s pattern at build time (``linalg.symbolic``) and factors its
values with one ``scan`` per column segment. Every step is the left-looking column update, written
with the run-time-index operations on a single carry vector ``[L values | D | work]``; the loop
slices the analysis tables one padded row per step, so the carry is proven safe to update in place
and nothing is copied between steps. ``solve`` runs the two triangular sweeps the same way. The
factorization is separate from the solve, so one factorization serves several right-hand sides,
and the solve carries the implicit derivative ``dx = K^{-1} (db - dK x)``: differentiating a solve
never differentiates the factorization loops.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from ..function.model import Function
from ..function.sugar import custom_derivative, scan, vmap
from ..ir.expr import Expr, as_expr, concat, gather, put, put_add, scatter, segment_sum, take
from .sparse import SparseMatrix
from .symbolic import CostModel, Ordering, Segment, SymbolicLDL, analyze

_NAMES = itertools.count()
DROP = -1  # an index a padded lane uses: outside every array, so it reads 0 and writes nowhere


def _table(ptr: np.ndarray, data: np.ndarray, seg: Segment, width: int, *, offset: int = 0, pad: int | np.ndarray = DROP) -> np.ndarray:
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
  ``[L below the diagonal (CSC of the permuted matrix) | D | n work entries | 0]``; ``solve`` uses it.

  The loops index without bounds checks: a padded lane reads the zero entry or writes a scratch slot
  of its own after it, so every index in the tables is in range by construction.
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
    # Every loop copies its carry in once, so a segment costs at least the carry's size.
    self.segments = s.segments(cost or CostModel(segment=256.0 + 0.5 * self.zero))
    self.solve_segments = s.segments(CostModel(step=2.0, a=0.0, u=0.0, c=1.0, segment=256.0 + 0.5 * s.n))
    lanes = max((max(seg.a, seg.u, seg.c, 1) for seg in self.segments), default=1)
    self.dump = self.zero + 1  # one scratch slot per lane for padded writes
    self.size = self.dump + lanes
    self.solve_lanes = max((max(seg.c, 1) for seg in self.solve_segments), default=1)
    self.values = self._factor(matrix.values)[: self.zero + 1]
    self._solver: Function | None = None

  # --- the factorization ----------------------------------------------------------------------

  def _factor_body(self, seg: Segment, k: int) -> Function:
    a, u, c = max(seg.a, 1), max(seg.u, 1), max(seg.c, 1)
    carry = Expr.sym("c", (self.size,))
    j = Expr.sym("j", (), dtype="int64")
    a_idx, a_src = _int_sym("a_idx", a), _int_sym("a_src", a)
    u_w, u_lik, u_ljk, u_dk = (_int_sym(nm, u) for nm in ("u_w", "u_lik", "u_ljk", "u_dk"))
    c_pos, c_w, clear = _int_sym("c_pos", c), _int_sym("c_w", c), _int_sym("clear", c)
    kv = Expr.sym("kv", (self.matrix.nnz,))
    ok = {"in_range": True}
    # w[rows of column j] = K[., j]
    u1 = put(carry, a_idx, take(kv, a_src, **ok), **ok)
    # w[i] -= L[i, k] D[k] L[j, k] for every k in row j of L and i >= j in column k
    u2 = put_add(u1, u_w, -(take(carry, u_lik, **ok) * take(carry, u_dk, **ok) * take(carry, u_ljk, **ok)), **ok)
    wj = take(u2, (j + self.w_offset).reshape((1,)), **ok)
    # D[j] = w[j]; L[., j] = w[rows] / D[j]; then clear the work entries this column used
    u3 = put(u2, (j + self.d_offset).reshape((1,)), wj, **ok)
    u4 = put(u3, c_pos, take(u2, c_w, **ok) * (1.0 / wj), **ok)
    u5 = put(u4, clear, Expr.const(np.zeros(c)), **ok)
    inputs = [carry, j, a_idx, a_src, u_w, u_lik, u_ljk, u_dk, c_pos, c_w, clear, kv]
    names = ["c", "j", "a_idx", "a_src", "u_w", "u_lik", "u_ljk", "u_dk", "c_pos", "c_w", "clear", "kv"]
    return Function._from_exprs(f"{self.name}_f{k}", inputs, [u5], names, ["c_next"])

  def _factor_tables(self, seg: Segment) -> list[np.ndarray]:
    s = self.symbolic
    a, u, c = max(seg.a, 1), max(seg.u, 1), max(seg.c, 1)
    j = np.arange(seg.start, seg.stop, dtype=np.int64)
    dump = lambda width: self.dump + np.arange(width)  # noqa: E731
    c_w = _table(s.l_ptr, s.l_rows, seg, c, offset=self.w_offset, pad=self.zero)
    # The entries of w this column's rows used; w[j] itself is never read again, so it is left.
    clear = _table(s.l_ptr, s.l_rows, seg, c, offset=self.w_offset, pad=dump(c))
    return [
      j,
      _table(s.a_ptr, s.a_rows, seg, a, offset=self.w_offset, pad=dump(a)),
      _table(s.a_ptr, s.a_source, seg, a, pad=0),
      _table(s.u_ptr, s.u_rows, seg, u, offset=self.w_offset, pad=dump(u)),
      _table(s.u_ptr, s.u_lik, seg, u, pad=self.zero),
      _table(s.u_ptr, s.u_ljk, seg, u, pad=self.zero),
      _table(s.u_ptr, s.u_k, seg, u, offset=self.d_offset, pad=self.zero),
      _table(s.l_ptr, np.arange(s.nnz_l), seg, c, pad=dump(c)),
      c_w,
      clear,
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

  def _sweep_body(self, seg: Segment, k: int, backward: bool) -> Function:
    c = max(seg.c, 1)
    ok = {"in_range": True}
    y, j = Expr.sym("y", (self.n + 1 + self.solve_lanes,)), Expr.sym("j", (), dtype="int64")
    rows, pos, f = _int_sym("rows", c), _int_sym("pos", c), Expr.sym("f", (self.zero + 1,))
    lcol = take(f, pos, **ok)
    if backward:
      # x[j] -= sum over i > j of L[i, j] x[i]: reads only entries below j, writes j
      nxt = put_add(y, j.reshape((1,)), -((lcol * take(y, rows, **ok)).sum()).reshape((1,)), **ok)
    else:
      # y[i] -= L[i, j] y[j] for the rows i > j of column j
      nxt = put_add(y, rows, -(lcol * take(y, j.reshape((1,)), **ok)), **ok)
    tag = "b" if backward else "f"
    return Function._from_exprs(f"{self.name}_s{tag}{k}", [y, j, rows, pos, f], [nxt], ["y", "j", "rows", "pos", "f"], ["y_next"])

  def _sweep(self, f: Expr, y: Expr, backward: bool) -> Expr:
    s = self.symbolic
    segments = self.solve_segments[::-1] if backward else self.solve_segments
    for k, seg in enumerate(segments):
      c = max(seg.c, 1)
      j = np.arange(seg.start, seg.stop, dtype=np.int64)
      # Padding: the forward sweep writes a scratch slot per lane after y's zero entry; the
      # backward sweep reads that zero entry; both read the factor's zero entry for L.
      row_pad = self.n if backward else self.n + 1 + np.arange(c)
      rows = _table(s.l_ptr, s.l_rows, seg, c, pad=row_pad).reshape(seg.length, c)
      pos = _table(s.l_ptr, np.arange(s.nnz_l), seg, c, pad=self.zero).reshape(seg.length, c)
      if backward:
        j, rows, pos = j[::-1], rows[::-1], pos[::-1]
      xs = [
        (Expr.const(j.copy(), dtype="int64"), 0, 1),
        (Expr.const(rows.reshape(-1).copy(), dtype="int64"), 0, c),
        (Expr.const(pos.reshape(-1).copy(), dtype="int64"), 0, c),
        (f, 0, 0),
      ]
      (y,) = scan(self._sweep_body(seg, k, backward), y, xs, length=seg.length)
    return y

  def _raw_solve(self, f: Expr, b: Expr) -> Expr:
    """``K^{-1} b`` from the factor ``f``, differentiated (if at all) through its loops."""
    s = self.symbolic
    extra = Expr.const(np.zeros(1 + self.solve_lanes))
    y = self._sweep(f, concat([gather(b, s.perm), extra]), backward=False)
    z = y[: self.n] / f[self.d_offset : self.d_offset + self.n]
    x = self._sweep(f, concat([z, extra]), backward=True)
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
    f, kv, b = Expr.sym("f", (self.zero + 1,)), Expr.sym("kv", (self.matrix.nnz,)), Expr.sym("b", (self.n,))
    base = Function._from_exprs(f"{self.name}_solve", [f, kv, b], [self._raw_solve(f, b)], ["f", "kv", "b"], ["x"])
    inner = base
    for level in (1, 2):
      inner = custom_derivative(base, jvp=self._jvp_rule(inner, level), vjp=self._vjp_rule(inner, level))
    self._solver = inner
    return inner

  def _jvp_rule(self, inner: Function, level: int) -> Function:
    f, kv, b = Expr.sym("f", (self.zero + 1,)), Expr.sym("kv", (self.matrix.nnz,)), Expr.sym("b", (self.n,))
    df, dkv, db = Expr.sym("df", (self.zero + 1,)), Expr.sym("dkv", (self.matrix.nnz,)), Expr.sym("db", (self.n,))
    x = _call(inner, f, kv, b)
    dx = _call(inner, f, kv, db - self._k_times(dkv, x))
    names = ["f", "kv", "b", "df", "dkv", "db"]
    return Function._from_exprs(f"{self.name}_solve_jvp{level}", [f, kv, b, df, dkv, db], [dx], names, ["dx"])

  def _vjp_rule(self, inner: Function, level: int) -> Function:
    f, kv, b = Expr.sym("f", (self.zero + 1,)), Expr.sym("kv", (self.matrix.nnz,)), Expr.sym("b", (self.n,))
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
    outs = [Expr.const(np.zeros(self.zero + 1)), kbar, bbar]
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
