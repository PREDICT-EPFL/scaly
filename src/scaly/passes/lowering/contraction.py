"""Lower vector and matrix products into ordered reduction loops."""

from __future__ import annotations

import numpy as np

from ...ir import program as p
from ...ir.expr import Expr, ExprOp
from ...ir.program import ProgramNode, RangeKind
from ...ir.types import DType, stored_indices
from .ctx import LowerCtx, lowers, LoweringError


def _nest(ranges: list[ProgramNode], body: list[ProgramNode]) -> list[ProgramNode]:
  for rng in reversed(ranges):
    body = [p.for_(rng, body)]
  return body


def _mm_init(out: ProgramNode, idx: ProgramNode, dtype: DType) -> ProgramNode:
  return p.store(p.view(out, [idx]), p.const_float(0.0, dtype=dtype))


def _mm_accum(out: ProgramNode, idx: ProgramNode, a_load: ProgramNode, b_load: ProgramNode) -> ProgramNode:
  return p.store(p.view(out, [idx]), p.add(p.load(p.view(out, [idx])), p.mul(a_load, b_load)))


def _mm_accumulate(
  ctx: LowerCtx,
  out: ProgramNode,
  out_idx: ProgramNode,
  a_load: ProgramNode,
  b_load: ProgramNode,
  dtype: DType,
  outer: list[ProgramNode],
  k_rng: ProgramNode,
) -> None:
  """Zero ``out[out_idx]`` over the ``outer`` loops, then accumulate ``a*b`` with the REDUCE-k loop outermost.

  A dot product per output is a serial add chain the C compiler cannot break without reassociation. With
  k outermost the inner loop runs over independent outputs and vectorizes, and each output still sums its
  terms in the same order, so the result is bit-identical to the dot form. Use it when the reduction axis
  is the matrix's slow axis; a contiguous reduction axis would make the compiler gather under -march=native.
  """
  ctx.statements.extend(_nest(outer, [_mm_init(out, out_idx, dtype)]))
  ctx.statements.extend(_nest([k_rng, *outer], [_mm_accum(out, out_idx, a_load, b_load)]))


@lowers(ExprOp.MATMUL)
def _lower_matmul(ctx: LowerCtx, node: Expr) -> None:
  a, b = node.args
  if a.type.pattern is not None or b.type.pattern is not None:
    _lower_stored_matmul(ctx, node)
    return
  a_buf, b_buf, out = ctx.buf_of(a), ctx.buf_of(b), ctx.alloc_tmp(node)
  dt = node.type.dtype
  sa, sb = a.shape, b.shape
  nm = out.attrs["name"]
  indices = {prefix: ctx.names.allocate(f"{prefix}_{nm}") for prefix in ("i", "j", "k", "ib", "it")}
  if len(sa) == 1 and len(sb) == 1:  # dot
    k = p.var(indices["k"])
    krng = p.range_(indices["k"], 0, sa[0], kind=RangeKind.REDUCE)
    _mm_accumulate(ctx, out, p.const_int(0), p.load(p.view(a_buf, [k])), p.load(p.view(b_buf, [k])), dt, [], krng)
  elif len(sa) == 2 and len(sb) == 1:  # mat @ vec: the reduction axis is contiguous in ``a``
    m, kk = sa
    i = p.var(indices["i"])
    irng = p.range_(indices["i"], 0, m, kind=RangeKind.GLOBAL)
    k = p.var(indices["k"])
    krng = p.range_(indices["k"], 0, kk, kind=RangeKind.REDUCE)
    row_dot = lambda row: _mm_accum(out, row, p.load(p.view(a_buf, [p.add(p.mul(row, p.const_int(kk)), k)])), p.load(p.view(b_buf, [k])))
    ctx.statements.extend(_nest([irng], [_mm_init(out, i, dt)]))
    # Four output rows per pass as four unrolled statements, so the compiler keeps four independent
    # accumulators; a four-trip inner loop over the rows becomes gathers under -march=native.
    blocks, tail = divmod(m, 4)
    if blocks:
      ib = p.var(indices["ib"])
      ibrng = p.range_(indices["ib"], 0, blocks, kind=RangeKind.GLOBAL)
      rows = [p.add(p.mul(ib, p.const_int(4)), p.const_int(r)) for r in range(4)]
      ctx.statements.extend(_nest([ibrng, krng], [row_dot(row) for row in rows]))
    if tail:
      it = p.var(indices["it"])
      itrng = p.range_(indices["it"], 4 * blocks, m, kind=RangeKind.GLOBAL)
      ctx.statements.extend(_nest([itrng, krng], [row_dot(it)]))
  elif len(sa) == 1 and len(sb) == 2:  # vec @ mat
    kk, n = sb
    j = p.var(indices["j"])
    jrng = p.range_(indices["j"], 0, n, kind=RangeKind.GLOBAL)
    k = p.var(indices["k"])
    krng = p.range_(indices["k"], 0, kk, kind=RangeKind.REDUCE)
    b_idx = p.add(p.mul(k, p.const_int(n)), j)
    _mm_accumulate(ctx, out, j, p.load(p.view(a_buf, [k])), p.load(p.view(b_buf, [b_idx])), dt, [jrng], krng)
  elif len(sa) == 2 and len(sb) == 2:  # mat @ mat
    m, kk = sa
    n = sb[1]
    i = p.var(indices["i"])
    irng = p.range_(indices["i"], 0, m, kind=RangeKind.GLOBAL)
    j = p.var(indices["j"])
    k = p.var(indices["k"])
    krng = p.range_(indices["k"], 0, kk, kind=RangeKind.REDUCE)

    def panel(rows: list[ProgramNode], columns: list[ProgramNode]) -> list[ProgramNode]:
      coordinates = [(row, column) for row in rows for column in columns]
      names = [ctx.names.allocate(f"acc_{nm}_{c}") for c in range(len(coordinates))]
      accumulators = [p.var(name, dt) for name in names]
      initial = [p.assign(name, p.const_float(0.0, dtype=dt), declare=True) for name in names]
      products = [
        p.mul(p.load(p.view(a_buf, [p.add(p.mul(row, p.const_int(kk)), k)])), p.load(p.view(b_buf, [p.add(p.mul(k, p.const_int(n)), column)])))
        for row, column in coordinates
      ]
      updates = [p.assign(name, p.add(acc, product)) for name, acc, product in zip(names, accumulators, products, strict=True)]
      stores = [
        p.store(p.view(out, [p.add(p.mul(row, p.const_int(n)), column)]), acc) for (row, column), acc in zip(coordinates, accumulators, strict=True)
      ]
      return [*initial, p.for_(krng, updates), *stores]

    if n <= 4:
      blocks, tail = divmod(m, 4)
      if blocks:
        ibrng = p.range_(indices["ib"], 0, blocks, kind=RangeKind.GLOBAL)
        rows = [p.add(p.mul(p.var(indices["ib"]), p.const_int(4)), p.const_int(r)) for r in range(4)]
        ctx.statements.extend(_nest([ibrng], panel(rows, [p.const_int(c) for c in range(n)])))
      if tail:
        tail_range = p.range_(indices["i"], 4 * blocks, m, kind=RangeKind.GLOBAL)
        ctx.statements.extend(_nest([tail_range], panel([i], [p.const_int(c) for c in range(n)])))
      return

    blocks, tail = divmod(n, 8)
    if blocks:
      panel_range = p.range_(indices["j"], 0, blocks, kind=RangeKind.GLOBAL)
      columns = [p.add(p.mul(j, p.const_int(8)), p.const_int(c)) for c in range(8)]
      ctx.statements.extend(_nest([irng, panel_range], panel([i], columns)))
    if tail:
      ctx.statements.extend(_nest([irng], panel([i], [p.const_int(c) for c in range(8 * blocks, n)])))
  else:
    raise LoweringError(f"matmul shapes {sa}@{sb} not lowered (batched / higher-rank deferred)")


def _lower_stored_matmul(ctx: LowerCtx, node: Expr) -> None:
  a, b = node.args
  if a.type.pattern is not None and b.type.pattern is not None:
    raise LoweringError("sparse-sparse products require sparse product kernels")
  out = ctx.alloc_tmp(node)
  m, width = (1, a.shape[0]) if len(a.shape) == 1 else a.shape
  n = 1 if len(b.shape) == 1 else b.shape[1]
  a_rows: dict[int, list[tuple[int, int]]] = {}
  b_cols: dict[int, dict[int, int]] = {}
  if a.type.pattern is not None:
    for slot, (row, k) in enumerate(zip(a.type.pattern.rows, a.type.pattern.cols, strict=True)):
      a_rows.setdefault(row, []).append((k, slot))
  if b.type.pattern is not None:
    for slot, (k, column) in enumerate(zip(b.type.pattern.rows, b.type.pattern.cols, strict=True)):
      b_cols.setdefault(column, {})[k] = slot
  outputs, left, right = [], [], []
  for slot, logical in enumerate(stored_indices(node.type)):
    row, column = divmod(int(logical), n)
    if a.type.pattern is not None:
      candidates = a_rows.get(row, ())
    else:
      candidates = ((k, row * width + k) for k in sorted(b_cols.get(column, {})))
    for k, a_slot in candidates:
      b_slot = b_cols.get(column, {}).get(k) if b.type.pattern is not None else k * n + column
      if b_slot is not None:
        outputs.append(slot)
        left.append(a_slot)
        right.append(b_slot)
  ranges, (i,) = ctx.nest((node.type.storage_size,), f"zero_{out.attrs['name']}")
  ctx.emit(ranges, [p.store(p.view(out, [i]), p.const_float(0.0, dtype=node.type.dtype))])
  ranges, (k,) = ctx.reduce((len(outputs),), f"product_{out.attrs['name']}")
  dst = ctx.index_at(np.asarray(outputs, dtype=np.int64), k)
  av = p.load(p.view(ctx.buf_of(a), [ctx.index_at(np.asarray(left, dtype=np.int64), k)]))
  bv = p.load(p.view(ctx.buf_of(b), [ctx.index_at(np.asarray(right, dtype=np.int64), k)]))
  ctx.emit(ranges, [_mm_accum(out, dst, av, bv)])
