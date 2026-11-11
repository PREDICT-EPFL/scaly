"""Lower axis reductions into dense nests or groups of stored coordinates."""

from __future__ import annotations

import numpy as np

from ...ir import program as p
from ...ir.expr import Expr, ExprOp
from ...ir.program import ProgramNode, ProgramOp
from ...ir.support import reduce_entries, support_indices
from ...ir.types import stored_indices
from ..program.scalarize import AUTO_SCALAR_OPS_PER_PROC
from .ctx import LowerCtx, lowers, _flat
from .schedule import emit_reduction


@lowers(ExprOp.REDUCE)
def _lower_reduce(ctx: LowerCtx, node: Expr) -> None:
  src = node.args[0]
  axes, kind = node.attrs["axes"], node.attrs["kind"]
  out = ctx.alloc_tmp(node)
  name = out.attrs["name"]
  if src.type.pattern is not None and len(axes) != len(src.shape):
    _lower_stored_reduce(ctx, node)
    return
  if len(axes) != len(src.shape) and src.type.pattern is None and (support := src.support) is not None and len(support) < src.size:
    _lower_supported_reduce(ctx, node, support_indices(support))
    return
  if len(axes) == len(src.shape):
    reductions, (i,) = ctx.reduce((src.type.storage_size,), f"i_{name}")
    value = ctx.read_stored(src, np.arange(src.type.storage_size, dtype=np.int64), i)
    identity = p.const_float(0, dtype=node.type.dtype) if kind != "add" and src.type.storage_size < src.size else None
    emit_reduction(ctx, [], reductions, p.view(out, [p.const_int(0)]), value, kind=kind, identity=identity)
    return
  outputs, output_coords = ctx.nest(node.shape, f"i_{name}")
  reductions, reduce_coords = ctx.reduce(tuple(src.shape[axis] for axis in axes), f"k_{name}")
  kept, reduced = iter(output_coords), iter(reduce_coords)
  coords = [next(reduced) if axis in axes else next(kept) for axis in range(len(src.shape))]
  emit_reduction(ctx, outputs, reductions, p.view(out, [_flat(node.shape, output_coords)]), ctx.read(src, coords), kind=kind)


def _lower_supported_reduce(ctx: LowerCtx, node: Expr, entries: np.ndarray) -> None:
  src = node.args[0]
  axes, kind = node.attrs["axes"], node.attrs["kind"]
  groups = reduce_entries(entries, src.shape, axes)
  domain = int(np.prod([src.shape[axis] for axis in axes], dtype=np.int64))
  counts = np.bincount(groups, minlength=node.size)
  out = ctx.raw_buf_of(node)
  name = out.attrs["name"]
  identity = {"add": 0.0, "max": -np.inf, "min": np.inf}[kind]
  for missing in (False, True) if kind != "add" else (False,):
    selected = np.flatnonzero((counts < domain) == missing) if kind != "add" else np.arange(node.size)
    outputs, (i,) = ctx.nest((len(selected),), f"zero_{name}")
    ctx.emit(outputs, [p.store(p.view(out, [ctx.index_at(selected, i)]), p.const_float(0 if missing else identity, dtype=node.type.dtype))])
  reductions, (k,) = ctx.reduce((len(entries),), f"support_{name}")
  target = p.view(out, [ctx.index_at(groups, k)])
  value = ctx.read_stored(src, entries, k)
  operation = {"add": ProgramOp.ADD, "max": ProgramOp.MAXIMUM, "min": ProgramOp.MINIMUM}[kind]
  ctx.emit(reductions, [p.store(target, ProgramNode(operation, (p.load(target), value), dtype=node.type.dtype))])


def _lower_stored_reduce(ctx: LowerCtx, node: Expr) -> None:
  src = node.args[0]
  axes, kind = node.attrs["axes"], node.attrs["kind"]
  projected = reduce_entries(stored_indices(src.type), src.shape, axes)
  writes = stored_indices(node.type)
  groups = np.searchsorted(writes, projected)
  order = np.argsort(groups, kind="stable")
  counts = np.bincount(groups, minlength=len(writes))
  pointers = np.concatenate(([0], np.cumsum(counts)))
  domain = int(np.prod([src.shape[axis] for axis in axes], dtype=np.int64))
  name = ctx.raw_buf_of(node).attrs["name"]

  def emit_group(outputs, row, entries, extent, initial):
    reductions, (k,) = ctx.reduce((extent,), f"stored_{name}")
    emit_reduction(ctx, outputs, reductions, ctx.stored_view(node, row), ctx.read_stored(src, entries, k), kind=kind, identity=initial)

  identity = {"add": 0.0, "max": -np.inf, "min": np.inf}[kind]
  if ctx.scalarize and (ctx.fun._effective_lowering == "scalar" or src.type.storage_size + len(writes) <= AUTO_SCALAR_OPS_PER_PROC):
    for row, count in enumerate(counts):
      initial = 0.0 if count < domain else identity
      emit_group([], p.const_int(row), order[pointers[row] : pointers[row + 1]], int(count), p.const_float(initial, dtype=node.type.dtype))
    return
  bounds = ctx.new_const_index(pointers)
  # An absent coordinate contributes zero once, while an empty domain keeps its identity.
  extrema = kind != "add" and domain > 0
  for missing in (False, True) if extrema else (False,):
    selected = np.flatnonzero((counts < domain) == missing) if extrema else np.arange(len(writes))
    outputs, (i,) = ctx.nest((len(selected),), f"group_{name}")
    row = ctx.index_at(selected, i)
    start = p.load(p.view(bounds, [row]))
    stop = p.load(p.view(bounds, [p.add(row, p.const_int(1))]))
    emit_group(outputs, row, order, (start, stop), p.const_float(0 if missing else identity, dtype=node.type.dtype))
