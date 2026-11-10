"""Plan stored-coordinate maps and emit their straight, table, affine or dense regions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ...ir import program as p
from ...ir.expr import Expr, ExprOp
from ...ir.program import ProgramNode, RangeKind
from ...ir.support import project_entries, slice_indices, transpose_entries, _constant_condition
from ...ir.types import CanonicalPattern, stored_indices
from ..arith import constant
from .ctx import LowerCtx


@dataclass(frozen=True)
class MapRegion:
  """One output region with fixed operand membership and physical stored-slot access maps."""

  writes: np.ndarray
  reads: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class SparseProductPlan:
  """Compressed column traversals, with no table of scalar multiplication pairs."""

  left_slots: np.ndarray
  left_rows: np.ndarray
  left_starts: np.ndarray
  left_stops: np.ndarray
  right_slots: np.ndarray
  right_ptr: np.ndarray
  output_ptr: np.ndarray
  output: MapRegion

  @staticmethod
  def build(a: CanonicalPattern, b: CanonicalPattern, logical: np.ndarray, writes: np.ndarray) -> SparseProductPlan:
    rows, cols = logical // b.shape[1], logical % b.shape[1]
    order = np.lexsort((rows, cols))
    columns, counts = np.unique(cols[order], return_counts=True)
    output_ptr = np.concatenate(([0], np.cumsum(counts)))
    left_slots = np.lexsort((a.rows, a.cols))
    left_columns, counts = np.unique(np.asarray(a.cols)[left_slots], return_counts=True)
    left_ptr = np.concatenate(([0], np.cumsum(counts)))
    right_slots = np.lexsort((b.rows, b.cols))
    br, bc = np.asarray(b.rows, dtype=np.int64), np.asarray(b.cols, dtype=np.int64)
    right_slots = right_slots[np.isin(br[right_slots], left_columns) & np.isin(bc[right_slots], columns)]
    left_column = np.searchsorted(left_columns, br[right_slots])
    right_ptr = np.concatenate((np.searchsorted(bc[right_slots], columns), [len(right_slots)]))
    return SparseProductPlan(
      left_slots,
      np.asarray(a.rows, dtype=np.int64)[left_slots],
      left_ptr[left_column],
      left_ptr[left_column + 1],
      right_slots,
      right_ptr,
      output_ptr,
      MapRegion(writes[order], (rows[order],)),
    )


def lower_sparse_product(ctx: LowerCtx, node: Expr, product: Expr | None = None) -> None:
  """Accumulate one column and write only requested entries, directly in their physical order."""
  sampled = product is not None
  product = node if product is None else product
  a, b = product.args
  assert a.type.pattern is not None and b.type.pattern is not None
  out = ctx.alloc_tmp(node)
  if not node.type.storage_size:
    return
  logical = stored_indices(node.type)
  if sampled:
    logical = node.attrs["indices"].reshape(-1)[logical]
  plan = SparseProductPlan.build(a.type.pattern, b.type.pattern, logical, ctx.physical_slots(node, np.arange(node.type.storage_size)))
  acc = ctx.new_private(node.type.dtype, (a.shape[0],))
  name = out.attrs["name"]
  if sampled:
    # Unsampled rows receive updates too; initialise them once so those additions never read uninitialised memory.
    rows = np.unique(plan.left_rows)
    ranges, (i,) = ctx.nest((len(rows),), f"init_{name}")
    ctx.emit(ranges, [p.store(p.view(acc, [ctx.index_at(rows, i, shared=True)]), p.const_float(0, dtype=node.type.dtype))])
  columns, (column,) = ctx.nest((len(plan.output_ptr) - 1,), f"column_{name}", kind=RangeKind.SERIAL)

  def bounds(ptr: np.ndarray, index: ProgramNode) -> tuple[ProgramNode, ProgramNode]:
    return ctx.index_at(ptr, index, shared=True), ctx.index_at(ptr, p.add(index, p.const_int(1)), shared=True)

  outputs, (i,) = ctx.nest((bounds(plan.output_ptr, column),), f"entry_{name}")
  row = ctx.index_at(plan.output.reads[0], i, shared=True)
  target = p.view(acc, [row])
  clear = ctx.nest_body(outputs, [p.store(target, p.const_float(0, dtype=node.type.dtype))])
  right, (j,) = ctx.reduce((bounds(plan.right_ptr, column),), f"right_{name}")
  left, (k,) = ctx.reduce(((ctx.index_at(plan.left_starts, j, shared=True), ctx.index_at(plan.left_stops, j, shared=True)),), f"left_{name}")
  dst = p.view(acc, [ctx.index_at(plan.left_rows, k, shared=True)])
  term = p.mul(ctx.read_stored(a, plan.left_slots, k), ctx.read_stored(b, plan.right_slots, j))
  updates = ctx.nest_body(right, ctx.nest_body(left, [p.store(dst, p.add(p.load(dst), term))]))
  write = ctx.nest_body(outputs, [p.store(p.view(out, [ctx.index_at(plan.output.writes, i, shared=True)]), p.load(target))])
  ctx.emit(columns, [*clear, *updates, *write])


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
  order = np.argsort(writes)
  writes, reads = writes[order], tuple(r[order] for r in reads)
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
        constant(0, arg.type.dtype) if indices[0] < 0 else p.load(p.view(ctx.raw_buf_of(arg), [ctx.index_at(indices, k, shared=True)]))
        for arg, indices in zip(operands, region.reads, strict=True)
      )
      dst = ctx.index_at(region.writes, k, shared=True)
      return [p.store(p.view(out, [dst]), body(values))]

    if len(region.writes) <= 4:
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
    ctx.emit(ranges, [p.store(p.view(out, [i]), p.const_float(0, dtype=node.type.dtype))])
    repeated = len(np.unique(dst)) != len(dst)
    ranges, (i,) = ctx.nest((len(source),), f"scatter_{out.attrs['name']}", kind=RangeKind.REDUCE if repeated else RangeKind.GLOBAL)
    target = p.view(out, [ctx.index_at(ctx.physical_slots(node, dst), i, shared=True)])
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
  if program_op == p.ProgramOp.SELECT:
    chosen = _constant_condition(node.args[0], logical, node.shape)
    if chosen is not None:
      reads[1] = np.where(chosen, reads[1], -1)
      reads[2] = np.where(chosen, -1, reads[2])
  emit_map(ctx, node, reads, lambda values: ProgramNode(program_op, values, dtype=node.type.dtype))
