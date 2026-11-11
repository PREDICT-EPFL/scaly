"""Lower vector and matrix products through the ordered reduction scheduler."""

from __future__ import annotations

import numpy as np

from ...ir import program as p
from ...ir.expr import Expr, ExprOp
from ...ir.program import RangeKind
from ...ir.support import support_indices
from ..program.scalarize import AUTO_SCALAR_OPS_PER_PROC
from .ctx import LowerCtx, lowers, _flat
from .schedule import emit_reduction
from .plans import lower_sparse_product


@lowers(ExprOp.MATMUL)
def _lower_matmul(ctx: LowerCtx, node: Expr) -> None:
  a, b = node.args
  if a.type.pattern is not None or b.type.pattern is not None:
    _lower_stored_matmul(ctx, node)
    return
  if ctx.seed_pairs and len(a.shape) <= 2 and len(b.shape) <= 2 and _lower_supported_matmul(ctx, node):
    return
  out = ctx.alloc_tmp(node)
  name = out.attrs["name"]
  outputs, coords = ctx.nest(node.shape, f"i_{name}")
  reductions, (k,) = ctx.reduce((a.shape[-1],), f"k_{name}")
  batch_rank = len(node.shape) - (len(a.shape) > 1) - (len(b.shape) > 1)

  def batch_coords(arg):
    return [p.const_int(0) if dim == 1 else coords[batch_rank - len(arg.shape[:-2]) + axis] for axis, dim in enumerate(arg.shape[:-2])]

  a_coords = [*batch_coords(a), coords[batch_rank], k] if len(a.shape) > 1 else [k]
  b_coords = [*batch_coords(b), k, coords[-1]] if len(b.shape) > 1 else [k]
  target = p.view(out, [_flat(node.shape, coords)])
  emit_reduction(
    ctx, outputs, reductions, target, p.mul(ctx.read(a, a_coords), ctx.read(b, b_coords)), batch_axes=sum(dim != 1 for dim in node.shape[:batch_rank])
  )


def _lower_supported_matmul(ctx: LowerCtx, node: Expr) -> bool:
  a, b = node.args
  supports = [ctx.support_of(arg) for arg in node.args]
  if all(s is None or len(s) == arg.size for arg, s in zip(node.args, supports, strict=True)):
    return False
  ai, bi = [np.arange(arg.size, dtype=np.int64) if s is None else support_indices(s) for arg, s in zip(node.args, supports, strict=True)]
  inner = a.shape[-1]
  width = b.shape[-1] if len(b.shape) == 2 else 1
  ak, bk = ai % inner, bi // width
  left, right = [], []
  for k in np.intersect1d(ak, bk):
    aa, bb = ai[ak == k], bi[bk == k]
    left.append(np.repeat(aa, len(bb)))
    right.append(np.tile(bb, len(aa)))
  reads_a = np.concatenate(left) if left else np.empty(0, dtype=np.int64)
  reads_b = np.concatenate(right) if right else np.empty(0, dtype=np.int64)
  writes = reads_a // inner * width + reads_b % width
  out = ctx.alloc_tmp(node)
  zero, (z,) = ctx.nest((node.size,), f"zero_{out.attrs['name']}")
  ctx.emit(zero, [p.store(p.view(out, [z]), p.const_float(0, dtype=node.type.dtype))])
  ranges, (k,) = ctx.nest((len(writes),), f"pairs_{out.attrs['name']}", kind=RangeKind.REDUCE)
  target = p.view(out, [ctx.index_at(writes, k, shared=True)])
  value = p.mul(ctx.read_stored(a, reads_a, k), ctx.read_stored(b, reads_b, k))
  ctx.emit(ranges, [p.store(target, p.add(p.load(target), value))])
  return True


def _lower_stored_matmul(ctx: LowerCtx, node: Expr) -> None:
  a, b = node.args
  if a.type.pattern is not None and b.type.pattern is not None:
    lower_sparse_product(ctx, node)
    return
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
    inner = ctx.index_at(columns[entries], k, shared=True)
    sparse_value = ctx.read_stored(src, entries, k)
    dense_coords = [inner] if vector else [inner, column] if left else [column, inner]
    if vector:
      slot = ctx.index_at(active, row, shared=True)
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
    bounds = ctx.new_const_index(pointers, shared=True)
    start = p.load(p.view(bounds, [row]))
    stop = p.load(p.view(bounds, [p.add(row, p.const_int(1))]))
    emit_row(outputs, row, column, order, (start, stop))
