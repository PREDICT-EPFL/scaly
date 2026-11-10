"""Plan stored-coordinate maps and emit their straight, table, affine or dense regions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ...ir import program as p
from ...ir.expr import Expr, ExprOp
from ...ir.program import ProgramNode, RangeKind
from ...ir.support import project_entries, slice_indices, transpose_entries
from ...ir.types import stored_indices
from ..affine import affine_index_map
from .ctx import LowerCtx


@dataclass(frozen=True)
class MapRegion:
  """One output region with fixed operand membership and physical stored-slot access maps."""

  writes: np.ndarray
  reads: tuple[np.ndarray, ...]

  @property
  def form(self) -> Literal["straight", "table", "affine", "dense"]:
    if len(self.writes) <= 4:
      return "straight"
    maps = (self.writes, *self.reads)
    if all(np.all(np.diff(a) == 1) or np.all(a == a[0]) for a in maps):
      return "dense"
    return "affine" if all(len(affine_index_map(a).residual) == 1 for a in maps) else "table"


def stored_slots(expr: Expr, logical: np.ndarray) -> np.ndarray:
  """Map logical coordinates to canonical slots, with -1 for structural zeros."""
  if expr.type.pattern is None:
    return logical
  entries = stored_indices(expr.type)
  slots = np.searchsorted(entries, logical)
  valid = slots < len(entries)
  valid[valid] &= entries[slots[valid]] == logical[valid]
  return np.where(valid, slots, -1)


def map_regions(writes: np.ndarray, reads: Sequence[np.ndarray]) -> tuple[MapRegion, ...]:
  """Build union and intersection regions before emission, with no run-time index comparisons."""
  if not len(writes):
    return ()
  membership = np.stack([r >= 0 for r in reads], axis=1)
  _, labels = np.unique(membership, axis=0, return_inverse=True)
  return tuple(MapRegion(writes[labels == label], tuple(r[labels == label] for r in reads)) for label in range(int(labels.max()) + 1))


def emit_map(
  ctx: LowerCtx,
  node: Expr,
  reads: Sequence[np.ndarray],
  body: Callable[[tuple[ProgramNode, ...]], ProgramNode],
  *,
  writes: np.ndarray | None = None,
  operands: Sequence[Expr] | None = None,
) -> None:
  """Emit a compact map directly into the output's physical order through the loop compiler."""
  out = ctx.alloc_tmp(node)
  operands = node.args if operands is None else operands
  writes = np.arange(node.type.storage_size, dtype=np.int64) if writes is None else writes
  writes = ctx.physical_slots(node, writes)
  reads = tuple(ctx.physical_slots(arg, slots) for arg, slots in zip(operands, reads, strict=True))
  for region in map_regions(writes, reads):

    def statements(k: ProgramNode) -> list[ProgramNode]:
      values = tuple(
        p.const_float(0, dtype=arg.type.dtype) if indices[0] < 0 else p.load(p.view(ctx.raw_buf_of(arg), [ctx.index_at(indices, k)]))
        for arg, indices in zip(operands, region.reads, strict=True)
      )
      dst = ctx.index_at(region.writes, k)
      return [p.store(p.view(out, [dst]), body(values))]

    if region.form == "straight":
      for k in range(len(region.writes)):
        ctx.statements.extend(statements(p.const_int(k)))
    else:
      ranges, (k,) = ctx.nest((len(region.writes),), f"map_{out.attrs['name']}")
      ctx.emit(ranges, statements(k))


def lower_movement(ctx: LowerCtx, node: Expr) -> None:
  """Use the compact map for every movement involving sparse storage."""
  logical = stored_indices(node.type)
  op = node.op
  if op == ExprOp.SCATTER:
    src = node.args[0]
    source = stored_indices(src.type)
    destinations = node.attrs["indices"].reshape(-1)[source]
    dst = stored_slots(node, destinations)
    out = ctx.alloc_tmp(node)
    ranges, (i,) = ctx.nest((node.type.storage_size,), f"zero_{out.attrs['name']}")
    ctx.emit(ranges, [ctx.store_stored(node, i, p.const_float(0, dtype=node.type.dtype))])
    repeated = len(np.unique(dst)) != len(dst)
    ranges, (i,) = ctx.nest((len(source),), f"scatter_{out.attrs['name']}", kind=RangeKind.REDUCE if repeated else RangeKind.GLOBAL)
    target = ctx.stored_view(node, ctx.index_at(dst, i))
    value = ctx.read_stored(src, np.arange(len(source), dtype=np.int64), i)
    ctx.emit(ranges, [p.store(target, p.add(p.load(target), value) if repeated else value)])
    return
  if op in (ExprOp.STACK, ExprOp.CONCAT):
    axis = node.attrs["axis"]
    coords = list(np.unravel_index(logical, node.shape))
    offset = 0
    for i, src in enumerate(node.args):
      selected = coords[axis] == i if op == ExprOp.STACK else (coords[axis] >= offset) & (coords[axis] < offset + src.shape[axis])
      source_coords = [c[selected] for c in coords]
      if op == ExprOp.STACK:
        source_coords.pop(axis)
      else:
        source_coords[axis] -= offset
        offset += src.shape[axis]
      source = np.ravel_multi_index(tuple(source_coords), src.shape)
      emit_map(ctx, node, [stored_slots(src, source)], lambda v: v[0], writes=np.flatnonzero(selected), operands=(src,))
    return
  src = node.args[0]
  if op == ExprOp.TRANSPOSE:
    logical = transpose_entries(logical, node.shape, tuple(map(int, np.argsort(node.attrs["axes"]))))
  elif op == ExprOp.SLICE:
    logical = slice_indices(src.shape, node.attrs["index"], logical)
  elif op == ExprOp.GATHER:
    logical = node.attrs["indices"].reshape(-1)[logical]
  emit_map(ctx, node, [stored_slots(src, logical)], lambda v: v[0])


def lower_elementwise(ctx: LowerCtx, node: Expr, program_op) -> None:
  """Sample broadcast operands on the output pattern and emit its compact scalar body."""
  logical = stored_indices(node.type)
  reads = [stored_slots(arg, project_entries(logical, arg.shape, node.shape)) for arg in node.args]
  emit_map(ctx, node, reads, lambda values: ProgramNode(program_op, values, dtype=node.type.dtype))
