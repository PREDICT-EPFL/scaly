"""Lower vector and matrix products through the ordered reduction scheduler."""

from __future__ import annotations

import numpy as np

from ...ad.rules import ELEMENTWISE
from ...ir import program as p
from ...ir.expr import Expr, ExprOp
from ...ir.program import ProgramNode
from ...ir.support import support_indices
from ..program.scalarize import AUTO_SCALAR_OPS_PER_PROC
from .ctx import LowerCtx, lowers, _broadcast, _flat
from .fusion import SeedEpilogue
from .schedule import emit_reduction
from .plans import lower_sparse_product


@lowers(ExprOp.MATMUL)
def _lower_matmul(ctx: LowerCtx, node: Expr) -> None:
  a, b = node.args[:2]
  if a.type.pattern is not None or b.type.pattern is not None:
    _lower_stored_matmul(ctx, node)
    return
  if ctx.seed_pairs and len(a.shape) <= 2 and len(b.shape) <= 2 and _lower_supported_matmul(ctx, node):
    return
  _emit_dense_matmul(ctx, node)


def lower_seed_epilogue(ctx: LowerCtx, plan: SeedEpilogue) -> None:
  _emit_dense_matmul(ctx, plan.chain[0], plan)


def _emit_dense_matmul(ctx: LowerCtx, node: Expr, epilogue: SeedEpilogue | None = None) -> None:
  a, b = node.args[:2]
  shape = node.shape
  packed = epilogue is not None and epilogue.packed_consumer is not None
  if epilogue is None:
    out = ctx.alloc_tmp(node)
  elif epilogue.packed_consumer is not None:
    out = ctx.new_private(node.type.dtype, (len(epilogue.seeds), shape[1]))
    ctx.packed_seed_inputs[epilogue.packed_consumer] = out
  else:
    out = ctx.alloc_tmp(epilogue.chain[-1])
  name = out.attrs["name"]
  extents = shape if epilogue is None else (len(epilogue.seeds), *shape[1:])
  entries = ctx.domains.get(node.id) if epilogue is None else None
  if entries is None:
    outputs, coords = ctx.nest(extents, f"i_{name}")
  else:
    outputs, (entry,) = ctx.nest((len(entries),), f"i_{name}")
    coords = ctx.domain_coords(shape, entries, entry)
  packed_coords = coords[:]
  if epilogue is not None:
    coords[0] = ctx.index_at(epilogue.seeds, coords[0], shared=True)
  reductions, (k,) = ctx.reduce((a.shape[-1],), f"k_{name}")
  batch_rank = len(node.shape) - (len(a.shape) > 1) - (len(b.shape) > 1)

  def batch_coords(arg):
    return [p.const_int(0) if dim == 1 else coords[batch_rank - len(arg.shape[:-2]) + axis] for axis, dim in enumerate(arg.shape[:-2])]

  a_coords = [*batch_coords(a), coords[batch_rank], k] if len(a.shape) > 1 else [k]
  b_coords = [*batch_coords(b), k, coords[-1]] if len(b.shape) > 1 else [k]
  slot = _flat(extents, packed_coords) if packed else _flat(shape, coords) if entries is None else ctx.index_at(entries, entry, shared=True)
  target = p.view(out, [slot])
  packed_input = ctx.packed_seed_inputs.get(node.id)
  rhs = ctx.read(b, b_coords) if packed_input is None else p.load(p.view(packed_input, [_flat((b.shape[0], b.shape[1]), (packed_coords[0], k))]))

  def finish(value: ProgramNode) -> ProgramNode:
    assert epilogue is not None
    source = epilogue.chain[1]
    for expr in epilogue.chain[2:]:
      args = tuple(value if arg is source else ctx.read(arg, _broadcast(coords[:-1], arg.shape)) for arg in expr.args)
      value = ProgramNode(ELEMENTWISE[ExprOp(expr.op)].program, args, dtype=expr.type.dtype)
      source = expr
    return value

  emit_reduction(
    ctx,
    outputs,
    reductions,
    target,
    p.mul(ctx.read(a, a_coords), rhs),
    batch_axes=sum(dim != 1 for dim in node.shape[:batch_rank]) if entries is None else 0,
    delinearize=node.id in ctx.permutes,
    finish=finish if epilogue else None,
  )
  if epilogue is not None and not packed:
    skipped = np.setdiff1d(np.arange(shape[0], dtype=np.int64), epilogue.seeds, assume_unique=True)
    zeros, (seed, row) = ctx.nest((len(skipped), shape[1]), f"zero_{name}")
    slot = _flat(shape, [ctx.index_at(skipped, seed, shared=True), row, p.const_int(0)])
    ctx.emit(zeros, [p.store(p.view(out, [slot]), p.const_float(0, dtype=node.type.dtype))])


def _lower_supported_matmul(ctx: LowerCtx, node: Expr) -> bool:
  a, b = node.args
  supports = [ctx.support_of(arg) for arg in node.args]
  if all(s is None or len(s) == arg.size for arg, s in zip(node.args, supports, strict=True)):
    return False
  inner = a.shape[-1]
  height = a.shape[0] if len(a.shape) == 2 else 1
  width = b.shape[-1] if len(b.shape) == 2 else 1
  out = ctx.alloc_tmp(node)
  zero, (z,) = ctx.nest((node.size,), f"zero_{out.attrs['name']}")
  ctx.emit(zero, [p.store(p.view(out, [z]), p.const_float(0, dtype=node.type.dtype))])
  ranges, (k,) = ctx.reduce((inner,), f"inner_{out.attrs['name']}")
  coords, values = [], []
  for left, (arg, support) in zip((True, False), zip(node.args, supports, strict=True), strict=True):
    if support is None or len(support) == arg.size:
      extent = height if left else width
      loops, (coord,) = ctx.nest((extent,), f"free_{out.attrs['name']}_{int(left)}")
      slot = p.add(p.mul(coord, p.const_int(inner)), k) if left else p.add(p.mul(k, p.const_int(width)), coord)
    else:
      entries = support_indices(support)
      contracted = entries % inner if left else entries // width
      order = np.argsort(contracted, kind="stable")
      entries = entries[order]
      pointers = np.concatenate(([0], np.cumsum(np.bincount(contracted, minlength=inner))))
      start = ctx.index_at(pointers, k, shared=True)
      stop = ctx.index_at(pointers, p.add(k, p.const_int(1)), shared=True)
      loops, (i,) = ctx.nest(((start, stop),), f"stored_{out.attrs['name']}_{int(left)}")
      coord = ctx.index_at(entries // inner if left else entries % width, i, shared=True)
      slot = ctx.index_at(entries, i, shared=True)
    ranges.extend(loops)
    coords.append(coord)
    values.append(p.load(ctx.stored_view(arg, slot)))
  target = p.view(out, [p.add(p.mul(coords[0], p.const_int(width)), coords[1])])
  value = p.mul(*values)
  ctx.emit(ranges, [p.store(target, p.add(p.load(target), value))])
  return True


def _lower_stored_matmul(ctx: LowerCtx, node: Expr) -> None:
  a, b = node.args[:2]
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
  layout = ctx.product_outputs.get(node.id)
  target_node = node if layout is None else layout[0]
  out = ctx.alloc_tmp(target_node)
  permutation = None if layout is None else ctx.physical_slots(target_node, layout[1])
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
    target = ctx.stored_view(node, slot) if permutation is None else p.view(out, [ctx.index_at(permutation, slot, shared=True)])
    emit_reduction(ctx, outputs, reductions, target, p.mul(sparse_value, ctx.read(dense, dense_coords)), delinearize=dense.id in ctx.permutes)

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
