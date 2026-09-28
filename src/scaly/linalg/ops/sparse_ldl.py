"""``sparse_ldl_factor`` and ``sparse_ldl_solve``, the looped sparse ``L D L^T`` expression ops over
the tables ``linalg.symbolic`` makes, with their verification, structural sparsity and loop
lowering. ``SparseLDL.solve`` differentiates implicitly, so neither op differentiates in the factor."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse

from ...ad.forward import JVPManyUnsupported, is_zero_const
from ...ad.reverse import NoAdjoint
from ...ad.sparsity import incidence, mask_compose
from ...ir import program as p
from ...ir.expr import Expr, as_expr, common_lowering, diff_any, promote_dtype, register_op, stack
from ...ir.program import ProgramNode, ProgramOp
from ...ir.spec import Rule
from ...ir.types import TensorType

if TYPE_CHECKING:
  from ...passes.lowering import LowerCtx

SPARSE_LDL, SPARSE_LDL_SOLVE = "sparse_ldl", "sparse_ldl_solve"

SPARSE_LDL_NO_DERIVATIVE = (
  "the derivative of a looped sparse LDL^T factorization is not implemented: SparseLDL.solve differentiates "
  "implicitly without it, and SparseLDL(..., schedule='scan') differentiates the factorization through its loops"
)
"""Why a ``sparse_ldl_factor`` node refuses a nonzero tangent or cotangent."""


SPARSE_LDL_TABLES = ("a_ptr", "a_rows", "a_src", "l_ptr", "l_rows", "r_cols", "r_pos", "ck_ptr", "ck_q", "ck_width", "ck_len")
"""The analysis tables a ``sparse_ldl_factor`` node carries, as ``linalg.symbolic`` names them (``a_src`` is
its ``a_source``; the ``ck_*`` tables are ``SymbolicLDL.chunks``)."""


SPARSE_LDL_MAX_WIDTH = 8
"""The most columns one chunk of a ``sparse_ldl_factor`` update covers."""


def _pointers_ok(ptr: np.ndarray, size: int) -> bool:
  """``ptr`` a pointer array into ``size`` entries: from 0, not decreasing, ending at ``size``."""
  return bool(ptr.size and ptr[0] == 0 and ptr[-1] == size and np.all(np.diff(ptr) >= 0))


def _in_range(values: np.ndarray, stop: int) -> bool:
  return bool(values.size == 0 or (values.min() >= 0 and values.max() < stop))


def _check_l_pattern(a: dict[str, np.ndarray], n: int, what: str) -> None:
  """The pattern of ``L``: every row inside the matrix and every column's rows below it, sorted."""
  l_ptr, l_rows = a["l_ptr"], a["l_rows"]
  if not _pointers_ok(l_ptr, l_rows.size) or not _in_range(l_rows, n):
    raise ValueError(f"{what}: l_ptr and l_rows do not describe the columns of an order-{n} L")
  col = np.repeat(np.arange(n), np.diff(l_ptr))
  if np.any(l_rows <= col) or np.any((np.diff(l_rows) <= 0) & (col[1:] == col[:-1])):
    raise ValueError(f"{what}: each column of L needs its rows below the diagonal, sorted")


def sparse_ldl_factor(values: Any, tables: dict[str, Any]) -> Expr:
  """The ``L D L^T`` factor of a symmetric matrix with a fixed sparsity pattern, without pivoting, as
  one vector ``[L below the diagonal, CSC | D]`` of the permuted matrix.

  ``values`` holds the matrix entries, and ``tables`` the analysis ``linalg.symbolic`` made of their
  pattern (``SPARSE_LDL_TABLES``): column ``j`` of the permuted lower triangle is
  ``values[a_src[p]]`` at rows ``a_rows[p]``, ``a_ptr[j] <= p < a_ptr[j + 1]``; column ``j`` of ``L``
  has rows ``l_rows[l_ptr[j]:l_ptr[j + 1]]``; row ``j`` of ``L`` lists its columns ``r_cols`` and the
  positions ``r_pos`` of its entries, cut into chunks of consecutive entries whose columns have the
  same rows from ``j`` down (``ck_*``: per column a range of chunks, each its first entry, width and
  number of rows). The generated code is a left-looking factorization that updates a work column
  from each chunk in one pass, keeping the update order of one column at a time, so the result is
  the same as ``SparseLDL(schedule="scan")``'s, bit for bit but for the sign of a zero or a NaN.
  The builder checks every table the generated code indexes; nothing is checked at run time: a zero
  pivot gives inf or NaN, as for ``ldl``. The derivative is not implemented: ``SparseLDL.solve``
  differentiates implicitly and never needs it, and ``SparseLDL(schedule="scan")`` differentiates
  the factorization through its loops."""
  values = as_expr(values)
  if len(values.shape) != 1 or not values.type.dtype.is_floating:
    raise ValueError(f"sparse_ldl_factor needs a floating-point vector of matrix entries, got {values.type.dtype}{values.shape}")
  missing = [k for k in SPARSE_LDL_TABLES if k not in tables]
  if missing:
    raise ValueError(f"sparse_ldl_factor needs the tables {missing}")
  attrs = {k: np.ascontiguousarray(np.asarray(tables[k], dtype=np.int64).reshape(-1)) for k in SPARSE_LDL_TABLES}
  n = attrs["a_ptr"].size - 1
  if n < 0 or any(attrs[k].size != n + 1 for k in ("l_ptr", "ck_ptr")):
    raise ValueError("sparse_ldl_factor tables disagree on the order of the matrix")
  if attrs["a_src"].size and (attrs["a_src"].min() < 0 or attrs["a_src"].max() >= values.size):
    raise ValueError(f"sparse_ldl_factor reads entries outside its {values.size} values")
  if attrs["ck_width"].size and (attrs["ck_width"].min() < 1 or attrs["ck_width"].max() > SPARSE_LDL_MAX_WIDTH):
    raise ValueError(f"sparse_ldl_factor chunks cover 1 to {SPARSE_LDL_MAX_WIDTH} columns")
  # Everything the generated code indexes stays inside its table: nothing is checked at run time.
  nnz_l, a = attrs["l_rows"].size, attrs
  if not _pointers_ok(a["a_ptr"], a["a_rows"].size) or a["a_src"].size != a["a_rows"].size or not _in_range(a["a_rows"], n):
    raise ValueError("sparse_ldl_factor: a_ptr, a_rows and a_src do not describe the columns of the matrix")
  _check_l_pattern(a, n, "sparse_ldl_factor")
  if a["r_cols"].size != nnz_l or a["r_pos"].size != nnz_l or not _in_range(a["r_cols"], n) or not _in_range(a["r_pos"], nnz_l):
    raise ValueError("sparse_ldl_factor: r_cols and r_pos need one entry per entry of L")
  chunks = a["ck_q"].size
  if not _pointers_ok(a["ck_ptr"], chunks) or a["ck_width"].size != chunks or a["ck_len"].size != chunks or not _in_range(a["ck_q"], nnz_l):
    raise ValueError("sparse_ldl_factor: the chunk tables disagree")
  last = a["ck_q"] + a["ck_width"]
  if np.any(last > nnz_l) or np.any(a["ck_len"] < 0):
    raise ValueError("sparse_ldl_factor: a chunk runs past the entries of L")
  for k in range(SPARSE_LDL_MAX_WIDTH):
    live = a["ck_width"] > k
    if np.any(a["r_pos"][a["ck_q"][live] + k] + a["ck_len"][live] > nnz_l):
      raise ValueError("sparse_ldl_factor: a chunk's rows run past the entries of L")
  size = attrs["l_rows"].size + n
  return Expr(SPARSE_LDL, (values,), TensorType((size,), dtype=values.type.dtype, diff=values.type.diff), attrs=attrs, lowering=values.lowering)


SPARSE_LDL_SOLVE_TABLES = ("perm", "l_ptr", "l_rows", "sn_first", "sn_width")
"""The tables a ``sparse_ldl_solve`` node carries: the ordering (``perm[new] = old``), the pattern of
``L`` and its chunks of columns (``SymbolicLDL.solve_chunks``)."""


def sparse_ldl_solve(factor: Any, b: Any, tables: dict[str, Any]) -> Expr:
  """``K^{-1} b`` from the ``[L | D]`` of ``sparse_ldl_factor`` (or of ``SparseLDL``), for the
  analysis in ``tables`` (``SPARSE_LDL_SOLVE_TABLES``).

  ``b`` permuted, the unit lower sweep, then the diagonal and the transposed sweep, and the result
  permuted back as each unknown is found. The forward sweep takes consecutive columns that form a
  chain, each column's rows the next column followed by that column's rows (a supernode), in chunks
  of up to ``SPARSE_LDL_MAX_WIDTH`` (``sn_first``, ``sn_width``): the chain's own rows column by
  column, then each shared row once for the whole chunk, the sum in a register. Every entry sees
  its updates in the order of one column at a time, and the transposed sweep sums as the ``scan``
  schedule does, so the result is that schedule's, bit for bit but for the sign of a zero or a NaN.
  Linear in ``b``, with that derivative; the derivative in the factor is not implemented
  (``SparseLDL.solve`` differentiates implicitly and never needs it)."""
  factor, b = as_expr(factor), as_expr(b)
  missing = [k for k in SPARSE_LDL_SOLVE_TABLES if k not in tables]
  if missing:
    raise ValueError(f"sparse_ldl_solve needs the tables {missing}")
  attrs = {k: np.ascontiguousarray(np.asarray(tables[k], dtype=np.int64).reshape(-1)) for k in SPARSE_LDL_SOLVE_TABLES}
  n = attrs["perm"].size
  if attrs["l_ptr"].size != n + 1 or factor.shape != (attrs["l_rows"].size + n,) or b.shape != (n,):
    raise ValueError(
      f"sparse_ldl_solve of order {n} needs a factor of {attrs['l_rows'].size + n} and a right-hand side of {n}, got {factor.shape} and {b.shape}"
    )
  first, width = attrs["sn_first"], attrs["sn_width"]
  if width.size != first.size or int(width.sum()) != n or not np.array_equal(first, np.cumsum(width) - width):
    raise ValueError("sparse_ldl_solve chunks must cover every column once, in order")
  if width.size and (width.min() < 1 or width.max() > SPARSE_LDL_MAX_WIDTH):
    raise ValueError(f"sparse_ldl_solve chunks cover 1 to {SPARSE_LDL_MAX_WIDTH} columns")
  if not np.array_equal(np.sort(attrs["perm"]), np.arange(n)):
    raise ValueError("sparse_ldl_solve: perm must be a permutation of the columns")
  _check_l_pattern(attrs, n, "sparse_ldl_solve")
  l_ptr, l_rows = attrs["l_ptr"], attrs["l_rows"]
  for f, w in zip(first.tolist(), width.tolist(), strict=True):
    for c in range(f, f + w - 1):  # a chain: each column's rows the next column, then that column's rows
      rows = l_rows[l_ptr[c] : l_ptr[c + 1]]
      if rows.size == 0 or rows[0] != c + 1 or not np.array_equal(rows[1:], l_rows[l_ptr[c + 1] : l_ptr[c + 2]]):
        raise ValueError(f"sparse_ldl_solve: columns {f}..{f + w - 1} are not a chain")
  return Expr(
    SPARSE_LDL_SOLVE,
    (factor, b),
    TensorType((n,), dtype=promote_dtype(factor, b), diff=diff_any(factor, b)),
    attrs=attrs,
    lowering=common_lowering(factor, b),
  )


def _sparse_ldl_tables(expr: Expr) -> str | None:
  missing = [k for k in SPARSE_LDL_TABLES if k not in expr.attrs]
  if missing:
    return f"SPARSE_LDL needs the tables {missing}"
  n = expr.attrs["a_ptr"].size - 1
  if expr.shape != (expr.attrs["l_rows"].size + n,) or len(expr.args[0].shape) != 1:
    return f"SPARSE_LDL of {expr.args[0].shape} values gives [L | D], not {expr.shape}"
  if any(expr.attrs[k].size != n + 1 for k in ("l_ptr", "ck_ptr")) or expr.attrs["r_cols"].size != expr.attrs["l_rows"].size:
    return "SPARSE_LDL tables disagree on the order of the matrix or the entries of L"
  return None


def _sparse_ldl_solve_tables(expr: Expr) -> str | None:
  missing = [k for k in SPARSE_LDL_SOLVE_TABLES if k not in expr.attrs]
  if missing:
    return f"SPARSE_LDL_SOLVE needs the tables {missing}"
  n = expr.attrs["perm"].size
  factor, b = expr.args
  if expr.shape != (n,) or b.shape != (n,) or factor.shape != (expr.attrs["l_rows"].size + n,) or expr.attrs["l_ptr"].size != n + 1:
    return f"SPARSE_LDL_SOLVE of order {n} with a factor of {factor.shape} and b of {b.shape} gives {expr.shape}"
  return None


def _refuse(*_: Any) -> Any:
  # reached only with a tangent or cotangent: without one, the dependence check gave zero
  raise NotImplementedError(SPARSE_LDL_NO_DERIVATIVE)


def _jvp_sparse_ldl_solve(expr: Expr, d: list[Expr]) -> Expr:
  # linear in b; in the factor, not implemented
  args = expr.args
  if not is_zero_const(d[0]):
    raise NotImplementedError(SPARSE_LDL_NO_DERIVATIVE)
  return sparse_ldl_solve(args[0], d[1], dict(expr.attrs))


def _jvp_many_sparse_ldl_solve(expr: Expr, tan: Callable[[Expr], Expr], nseed: int) -> Expr:
  args = expr.args
  d = [tan(arg) for arg in args]
  if not is_zero_const(d[0]):
    raise NotImplementedError(SPARSE_LDL_NO_DERIVATIVE)
  if nseed > 64:
    raise JVPManyUnsupported(str(expr.op))  # one solve per seed would outgrow the per-seed fallback
  return stack([sparse_ldl_solve(args[0], d[1][k], dict(expr.attrs)) for k in range(nseed)], axis=0)


def _vjp_sparse_ldl_solve(expr: Expr, cot: Expr) -> tuple[NoAdjoint | Expr, ...]:
  # K is symmetric: b's cotangent is one more solve. The factor's is not implemented.
  return (NoAdjoint(SPARSE_LDL_NO_DERIVATIVE), sparse_ldl_solve(expr.args[0], cot, dict(expr.attrs)))


def _sparse_ldl_reads(expr: Expr) -> sparse.csr_array:
  """Which matrix entries each entry of a ``sparse_ldl_factor`` result reads: column ``j`` of ``L`` and
  ``D[j]`` come from the columns of the elimination subtree of ``j`` (``j`` and every column whose
  path up the tree, parent = first row below the diagonal, passes through ``j``)."""
  a = expr.attrs
  n = a["a_ptr"].size - 1
  l_ptr, l_rows = a["l_ptr"], a["l_rows"]
  counts = np.diff(l_ptr)
  parent = np.full(n, -1, dtype=np.int64)
  parent[counts > 0] = l_rows[l_ptr[:-1][counts > 0]]
  anc_rows, anc_cols = [], []
  for k in range(n):  # k lies in the subtree of each of its ancestors, itself included
    j = k
    while j >= 0:
      anc_rows.append(j)
      anc_cols.append(k)
      j = int(parent[j])
  subtree = incidence((n, n), np.asarray(anc_rows, dtype=np.int64), np.asarray(anc_cols, dtype=np.int64))
  reads = incidence((n, expr.args[0].size), np.repeat(np.arange(n), np.diff(a["a_ptr"])), a["a_src"])
  column = np.concatenate([np.repeat(np.arange(n), counts), np.arange(n)])  # the column of each factor entry
  return mask_compose(mask_compose(incidence((expr.size, n), np.arange(expr.size), column), subtree), reads)


def _sparsity_sparse_ldl(expr: Expr, mask: Callable[[Expr], sparse.csr_array], ncols: int) -> sparse.csr_array:
  return mask_compose(_sparse_ldl_reads(expr), mask(expr.args[0]))


def _lower_sparse_ldl(ctx: LowerCtx, node: Expr) -> None:
  """Left-looking sparse ``L D L^T`` over the node's analysis tables, into ``[L | D]``.

  Per column ``j``: the matrix column goes into a dense work column; each chunk of row ``j`` of
  ``L`` (up to ``SPARSE_LDL_MAX_WIDTH`` columns ``k`` whose rows from ``j`` down are the same)
  updates it in one pass over those rows, ``w[i] += L[i, k] * (-D[k] L[j, k])`` for its columns in
  order, the sum held in a register; then ``D[j] = w[j]`` and ``L[i, j] = w[i] / D[j]``, clearing
  each ``w[i]`` read. Every entry sees its updates in the order of one column at a time, so the
  rounding is that of the column-by-column factorization. A chunk's width picks one of the loops
  for the widths the analysis has, each run zero or one times, so the widths need no branch
  statement."""
  (kv,) = node.args
  a = node.attrs
  n, nnz_l = a["a_ptr"].size - 1, a["l_rows"].size
  out, src, dt = ctx.alloc_tmp(node), ctx.buf_of(kv), node.type.dtype
  if n == 0:
    return
  c = p.const_int
  nm = out.attrs["name"]
  tables = {
    k: ctx.new_const_index(a[k]) for k in ("a_ptr", "a_rows", "a_src", "l_ptr", "l_rows", "r_cols", "r_pos", "ck_ptr", "ck_q", "ck_width", "ck_len")
  }

  def at(table: str, i: ProgramNode) -> ProgramNode:
    return p.load(p.view(tables[table], [i]))

  def scalar() -> ProgramNode:
    return p.view(ctx.new_private(dt, ()), [c(0)])

  work = ctx.new_private(dt, (n,))
  zero = p.const_float(0.0, dtype=dt)
  i0 = p.var(f"lz_{nm}")
  ctx.emit(p.for_(p.range_(i0.attrs["name"], 0, n), [p.store(p.view(work, [i0]), zero)]))
  j = p.var(f"lj_{nm}")
  j1 = p.add(j, c(1))
  pa = p.var(f"la_{nm}")
  column = p.for_(
    p.range_(pa.attrs["name"], at("a_ptr", j), at("a_ptr", j1)), [p.store(p.view(work, [at("a_rows", pa)]), p.load(p.view(src, [at("a_src", pa)])))]
  )
  r = p.var(f"lr_{nm}")
  q = at("ck_q", r)
  widths = []
  for width in sorted(set(a["ck_width"].tolist())):  # a loop for each width the analysis has
    pos = [at("r_pos", p.add(q, c(k)) if k else q) for k in range(width)]
    scales = [scalar() for _ in range(width)]
    # -D[k] L[j, k] per column of the chunk, before the pass over its rows
    head = [
      p.store(s, p.neg(p.mul(p.load(p.view(out, [p.add(c(nnz_l), at("r_cols", p.add(q, c(k)) if k else q))])), p.load(p.view(out, [pos[k]])))))
      for k, s in enumerate(scales)
    ]
    t = p.var(f"lt{width}_{nm}")
    dst = p.view(work, [at("l_rows", p.add(pos[0], t))])
    total = p.load(dst)
    for k, s in enumerate(scales):
      total = p.add(total, p.mul(p.load(p.view(out, [p.add(pos[k], t)])), p.load(s)))
    rows = p.for_(p.range_(t.attrs["name"], 0, at("ck_len", r)), [p.store(dst, total)])
    once = p.var(f"lw{width}_{nm}")
    taken = p.select(p.compare(ProgramOp.EQ, at("ck_width", r), c(width)), c(1), c(0))
    widths.append(p.for_(p.range_(once.attrs["name"], 0, taken), [*head, rows]))
  updates = p.for_(p.range_(r.attrs["name"], at("ck_ptr", j), at("ck_ptr", j1)), widths)
  pivot, inverse = scalar(), scalar()
  pl = p.var(f"ll_{nm}")
  entry = p.view(work, [at("l_rows", pl)])
  finish = [
    p.store(pivot, p.load(p.view(work, [j]))),
    p.store(p.view(out, [p.add(c(nnz_l), j)]), p.load(pivot)),
    p.store(inverse, p.div(p.const_float(1.0, dtype=dt), p.load(pivot))),
    p.for_(
      p.range_(pl.attrs["name"], at("l_ptr", j), at("l_ptr", j1)),
      [p.store(p.view(out, [pl]), p.mul(p.load(entry), p.load(inverse))), p.store(entry, zero)],
    ),
  ]
  ctx.emit(p.for_(p.range_(j.attrs["name"], 0, n), [column, updates, *finish]))


def _lower_sparse_ldl_solve(ctx: LowerCtx, node: Expr) -> None:
  """``K^{-1} b`` from ``[L | D]``: ``b`` permuted into a work vector, the unit lower sweep by
  chunks of chained columns, then the transposed sweep, which divides by ``D``, subtracts its
  blocked dot product and writes each unknown to its place in the output as it is found.

  A chunk of ``w`` chained columns ``j .. j + w - 1`` (each column's rows the next column followed
  by that column's rows) first updates its own rows column by column, then every row below it once
  with all ``w`` terms, the sum in a register: each entry takes its updates in column order, as the
  column-by-column sweep gives them. A chunk's width picks one of the loops for the widths the
  analysis has, run zero or one times."""
  factor, b = node.args
  a = node.attrs
  n, nnz_l = a["perm"].size, a["l_rows"].size
  out, fb, dt = ctx.alloc_tmp(node), ctx.buf_of(factor), node.type.dtype
  if n == 0:
    return
  c = p.const_int
  nm = out.attrs["name"]
  tables = {k: ctx.new_const_index(a[k]) for k in ("perm", "l_ptr", "l_rows", "sn_first", "sn_width")}

  def at(table: str, i: ProgramNode) -> ProgramNode:
    return p.load(p.view(tables[table], [i]))

  def scalar() -> ProgramNode:
    return p.view(ctx.new_private(dt, ()), [c(0)])

  def entry(k: ProgramNode) -> ProgramNode:
    return p.load(p.view(fb, [k]))

  y = ctx.new_private(dt, (n,))
  i0 = p.var(f"sp_{nm}")
  ctx.emit(p.for_(p.range_(i0.attrs["name"], 0, n), [p.store(p.view(y, [i0]), p.load(p.view(ctx.buf_of(b), [at("perm", i0)])))]))
  r = p.var(f"sr_{nm}")
  j = at("sn_first", r)
  widths = []
  for width in sorted(set(a["sn_width"].tolist())):  # a loop for each width the analysis has
    col = [p.add(j, c(k)) if k else j for k in range(width)]
    body: list[ProgramNode] = []
    for k in range(width - 1):  # the chain's own rows, column by column
      s = scalar()
      body.append(p.store(s, p.neg(p.load(p.view(y, [col[k]])))))
      start = at("l_ptr", col[k])
      for k2 in range(k + 1, width):
        dst = p.view(y, [col[k2]])
        body.append(p.store(dst, p.add(p.load(dst), p.mul(entry(p.add(start, c(k2 - k - 1)) if k2 > k + 1 else start), p.load(s)))))
    scales = [scalar() for _ in range(width)]
    body += [p.store(s, p.neg(p.load(p.view(y, [col[k]])))) for k, s in enumerate(scales)]
    below = at("l_ptr", col[-1])  # the rows under the chunk, shared by its columns
    t = p.var(f"st{width}_{nm}")
    dst = p.view(y, [at("l_rows", p.add(below, t))])
    total = p.load(dst)
    for k, s in enumerate(scales):
      offset = width - 1 - k
      first = at("l_ptr", col[k])
      total = p.add(total, p.mul(entry(p.add(p.add(first, c(offset)) if offset else first, t)), p.load(s)))
    body.append(p.for_(p.range_(t.attrs["name"], 0, p.sub(at("l_ptr", p.add(col[-1], c(1))), below)), [p.store(dst, total)]))
    once = p.var(f"sw{width}_{nm}")
    taken = p.select(p.compare(ProgramOp.EQ, at("sn_width", r), c(width)), c(1), c(0))
    widths.append(p.for_(p.range_(once.attrs["name"], 0, taken), body))
  ctx.emit(p.for_(p.range_(r.attrs["name"], 0, a["sn_first"].size), widths))
  step = p.var(f"sb_{nm}")
  jb = p.sub(c(n - 1), step)
  sums, total = ctx.blocked_sum(
    f"sd_{nm}", at("l_ptr", jb), at("l_ptr", p.add(jb, c(1))), lambda q: p.mul(entry(q), p.load(p.view(y, [at("l_rows", q)]))), dt
  )
  unknown = scalar()
  back = [
    *sums,
    p.store(unknown, p.sub(p.div(p.load(p.view(y, [jb])), entry(p.add(c(nnz_l), jb))), total)),
    p.store(p.view(y, [jb]), p.load(unknown)),
    p.store(p.view(out, [at("perm", jb)]), p.load(unknown)),
  ]
  ctx.emit(p.for_(p.range_(step.attrs["name"], 0, n), back))


register_op(
  SPARSE_LDL,
  arity=1,
  jvp=_refuse,
  jvp_many=_refuse,
  vjp=_refuse,
  sparsity=_sparsity_sparse_ldl,
  verify=(Rule(SPARSE_LDL, "sparse-ldl-tables", _sparse_ldl_tables),),
  lower=_lower_sparse_ldl,
  traits={"runtime_index": True},
)
# Every unknown of a solve may depend on every entry of the factor and of the right-hand side: the
# default pattern, so the solve has no sparsity rule of its own.
register_op(
  SPARSE_LDL_SOLVE,
  arity=2,
  jvp=_jvp_sparse_ldl_solve,
  jvp_many=_jvp_many_sparse_ldl_solve,
  vjp=_vjp_sparse_ldl_solve,
  verify=(Rule(SPARSE_LDL_SOLVE, "sparse-ldl-solve-tables", _sparse_ldl_solve_tables),),
  lower=_lower_sparse_ldl_solve,
  traits={"runtime_index": True},
)
