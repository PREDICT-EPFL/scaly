"""Lower vector and matrix products through the ordered reduction scheduler."""

from __future__ import annotations

import numpy as np

from ...ir import program as p
from ...ir.expr import Expr, ExprOp
from ..program.scalarize import AUTO_SCALAR_OPS_PER_PROC
from .ctx import LowerCtx, lowers, LoweringError, _flat
from .schedule import emit_reduction


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
  left = a.type.pattern is not None
  src, dense = (a, b) if left else (b, a)
  pattern = src.type.pattern
  assert pattern is not None
  rows = np.asarray(pattern.rows if left else pattern.cols, dtype=np.int64)
  columns = np.asarray(pattern.cols if left else pattern.rows, dtype=np.int64)
  order = np.lexsort((columns, rows))
  active, counts = np.unique(rows[order], return_counts=True)
  pointers = np.concatenate(([0], np.cumsum(counts)))
  out = ctx.alloc_tmp(node)
  name = out.attrs["name"]
  vector = len(dense.shape) == 1
  width = 1 if vector else dense.shape[1] if left else dense.shape[0]
  if vector and len(active) < node.size:
    zero, (z,) = ctx.nest((node.size,), f"zero_{name}")
    ctx.emit(zero, [p.store(p.view(out, [z]), p.const_float(0, dtype=node.type.dtype))])

  def emit_row(outputs, row, column, entries, extent):
    reductions, (k,) = ctx.reduce((extent,), f"stored_{name}")
    inner = ctx.index_at(columns[entries], k)
    sparse_value = ctx.read_stored(src, entries, k)
    dense_coords = [inner] if vector else [inner, column] if left else [column, inner]
    if vector:
      slot = ctx.index_at(active, row)
    elif left:
      slot = p.add(p.mul(row, p.const_int(width)), column)
    else:
      slot = p.add(p.mul(column, p.const_int(len(active))), row)
    emit_reduction(ctx, outputs, reductions, ctx.stored_view(node, slot), p.mul(sparse_value, ctx.read(dense, dense_coords)))

  if ctx.scalarize and (ctx.fun._effective_lowering == "scalar" or 2 * pattern.nnz * width <= AUTO_SCALAR_OPS_PER_PROC):
    for row, (start, stop) in enumerate(zip(pointers[:-1], pointers[1:], strict=True)):
      outputs, (column,) = ctx.nest((width,), f"column_{name}")
      emit_row(outputs, p.const_int(row), column, order[start:stop], int(stop - start))
  else:
    outputs, (row, column) = ctx.nest((len(active), width), f"row_{name}")
    bounds = ctx.new_const_index(pointers)
    start = p.load(p.view(bounds, [row]))
    stop = p.load(p.view(bounds, [p.add(row, p.const_int(1))]))
    emit_row(outputs, row, column, order, (start, stop))
