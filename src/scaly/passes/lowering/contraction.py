"""Lower vector and matrix products through the ordered reduction scheduler."""

from __future__ import annotations

import numpy as np

from ...ir import program as p
from ...ir.expr import Expr, ExprOp
from ...ir.program import ProgramNode
from ...ir.types import stored_indices
from .ctx import LowerCtx, lowers, LoweringError, _flat
from .schedule import emit_reduction


def _mm_accum(out: ProgramNode, idx: ProgramNode, a_load: ProgramNode, b_load: ProgramNode) -> ProgramNode:
  return p.store(p.view(out, [idx]), p.add(p.load(p.view(out, [idx])), p.mul(a_load, b_load)))


@lowers(ExprOp.MATMUL)
def _lower_matmul(ctx: LowerCtx, node: Expr) -> None:
  a, b = node.args
  if a.type.pattern is not None or b.type.pattern is not None:
    _lower_stored_matmul(ctx, node)
    return
  if len(a.shape) not in (1, 2) or len(b.shape) not in (1, 2):
    raise LoweringError(f"matmul shapes {a.shape}@{b.shape} not lowered (batched / higher-rank deferred)")
  out = ctx.alloc_tmp(node)
  name = out.attrs["name"]
  outputs, coords = ctx.nest(node.shape, f"i_{name}")
  reductions, (k,) = ctx.reduce((a.shape[-1],), f"k_{name}")
  a_coords = [coords[0], k] if len(a.shape) == 2 else [k]
  b_coords = [k, coords[-1]] if len(b.shape) == 2 else [k]
  target = p.view(out, [_flat(node.shape, coords)])
  emit_reduction(ctx, outputs, reductions, target, p.mul(ctx.read(a, a_coords), ctx.read(b, b_coords)))


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
