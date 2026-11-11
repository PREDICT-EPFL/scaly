"""Plan stored-coordinate maps and emit their straight, table, affine or dense regions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ...ad.rules import _mask_of
from ...ir import program as p
from ...ir.expr import Expr, ExprOp
from ...ir.program import ProgramNode, ProgramOp, RangeKind
from ...ir.support import project_entries, slice_indices, transpose_entries, support_indices, _constant_condition
from ...ir.types import CanonicalPattern, dtypes, stored_indices
from .ctx import LowerCtx
from ..arith import constant, fold_program


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
  output_writes: np.ndarray
  output_rows: np.ndarray

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
      writes[order],
      rows[order],
    )


def lower_sparse_product(ctx: LowerCtx, node: Expr) -> None:
  """Accumulate one column and write stored entries directly in their physical order."""
  if node.attrs.get("sampled") is not None:
    _lower_sampled_product(ctx, node)
    return
  a, b = node.args
  assert a.type.pattern is not None and b.type.pattern is not None
  out = ctx.alloc_tmp(node)
  if not node.type.storage_size:
    return
  logical = stored_indices(node.type)
  plan = SparseProductPlan.build(a.type.pattern, b.type.pattern, logical, ctx.physical_slots(node, np.arange(node.type.storage_size)))
  acc = ctx.new_private(node.type.dtype, (a.shape[0],))
  name = out.attrs["name"]
  columns, (column,) = ctx.nest((len(plan.output_ptr) - 1,), f"column_{name}", kind=RangeKind.SERIAL)

  def bounds(ptr: np.ndarray, index: ProgramNode) -> tuple[ProgramNode, ProgramNode]:
    return ctx.index_at(ptr, index, shared=True), ctx.index_at(ptr, p.add(index, p.const_int(1)), shared=True)

  outputs, (i,) = ctx.nest((bounds(plan.output_ptr, column),), f"entry_{name}")
  row = ctx.index_at(plan.output_rows, i, shared=True)
  target = p.view(acc, [row])
  clear = ctx.nest_body(outputs, [p.store(target, p.const_float(0, dtype=node.type.dtype))])
  right, (j,) = ctx.reduce((bounds(plan.right_ptr, column),), f"right_{name}")
  left, (k,) = ctx.reduce(((ctx.index_at(plan.left_starts, j, shared=True), ctx.index_at(plan.left_stops, j, shared=True)),), f"left_{name}")
  dst = p.view(acc, [ctx.index_at(plan.left_rows, k, shared=True)])
  term = p.mul(ctx.read_stored(a, plan.left_slots, k), ctx.read_stored(b, plan.right_slots, j))
  updates = ctx.nest_body(right, ctx.nest_body(left, [p.store(dst, p.add(p.load(dst), term))]))
  write = ctx.nest_body(outputs, [p.store(p.view(out, [ctx.index_at(plan.output_writes, i, shared=True)]), p.load(target))])
  ctx.emit(columns, [*clear, *updates, *write])


@dataclass(frozen=True)
class SampledProductPlan:
  """Requested rows and dot-product bounds, without forming the full product pattern."""

  output_ptr: np.ndarray
  left_starts: np.ndarray
  left_stops: np.ndarray
  right_slots: np.ndarray
  right_rows: np.ndarray
  right_starts: np.ndarray
  right_stops: np.ndarray

  @staticmethod
  def build(a: CanonicalPattern, b: CanonicalPattern, output: CanonicalPattern) -> SampledProductPlan:
    rows, counts = np.unique(output.rows, return_counts=True)
    right_slots = np.lexsort((b.rows, b.cols))
    right_cols = np.asarray(b.cols, dtype=np.int64)[right_slots]
    return SampledProductPlan(
      np.concatenate(([0], np.cumsum(counts))),
      np.searchsorted(a.rows, rows),
      np.searchsorted(a.rows, rows, side="right"),
      right_slots,
      np.asarray(b.rows, dtype=np.int64)[right_slots],
      np.searchsorted(right_cols, output.cols),
      np.searchsorted(right_cols, output.cols, side="right"),
    )


def _lower_sampled_product(ctx: LowerCtx, node: Expr) -> None:
  a, b = node.args
  assert a.type.pattern is not None and b.type.pattern is not None and node.type.pattern is not None
  out = ctx.alloc_tmp(node)
  if not node.type.storage_size:
    return
  plan = SampledProductPlan.build(a.type.pattern, b.type.pattern, node.type.pattern)
  slots = ctx.new_private(dtypes.int64, (a.shape[1],))
  name = out.attrs["name"]
  active = np.unique(plan.right_rows)
  initial, (i,) = ctx.nest((len(active),), f"init_{name}")
  ctx.emit(initial, [p.store(p.view(slots, [ctx.index_at(active, i, shared=True)]), p.const_int(-1))])
  rows, (row,) = ctx.nest((len(plan.left_starts),), f"row_{name}", kind=RangeKind.SERIAL)
  scatter, (k,) = ctx.nest(((ctx.index_at(plan.left_starts, row, shared=True), ctx.index_at(plan.left_stops, row, shared=True)),), f"scatter_{name}")
  target = p.view(slots, [ctx.index_at(np.asarray(a.type.pattern.cols, dtype=np.int64), k, shared=True)])
  physical = ctx.physical_slots(a, np.arange(a.type.storage_size))
  fill = ctx.nest_body(scatter, [p.store(target, ctx.index_at(physical, k, shared=True))])
  clear = ctx.nest_body(scatter, [p.store(target, p.const_int(-1))])
  outputs, (entry,) = ctx.nest(
    ((ctx.index_at(plan.output_ptr, row, shared=True), ctx.index_at(plan.output_ptr, p.add(row, p.const_int(1)), shared=True)),), f"entry_{name}"
  )
  right, (j,) = ctx.reduce(
    ((ctx.index_at(plan.right_starts, entry, shared=True), ctx.index_at(plan.right_stops, entry, shared=True)),), f"dot_{name}"
  )
  slot = p.load(p.view(slots, [ctx.index_at(plan.right_rows, j, shared=True)]))
  # A zero-trip range skips absent factors, including 0*inf; a stored zero still executes its product.
  start = ProgramNode(ProgramOp.MAXIMUM, (slot, p.const_int(0)), dtype=dtypes.int64)
  stop = ProgramNode(ProgramOp.MAXIMUM, (p.add(slot, p.const_int(1)), p.const_int(0)), dtype=dtypes.int64)
  present, (pos,) = ctx.reduce(((start, stop),), f"present_{name}")
  acc_name = ctx.names.allocate(f"acc_{name}")
  acc = p.var(acc_name, node.type.dtype)
  value = p.mul(p.load(p.view(ctx.raw_buf_of(a), [pos])), ctx.read_stored(b, plan.right_slots, j))
  value = _guard_sampled_term(ctx, a, pos, value)
  value = _guard_sampled_term(ctx, b, ctx.index_at(ctx.physical_slots(b, plan.right_slots), j, shared=True), value)
  dot = ctx.nest_body(right, ctx.nest_body(present, [p.assign(acc_name, p.add(acc, value))]))
  writes = ctx.physical_slots(node, np.arange(node.type.storage_size))
  body = [
    p.assign(acc_name, p.const_float(0, dtype=node.type.dtype), declare=True),
    *dot,
    p.store(p.view(out, [ctx.index_at(writes, entry, shared=True)]), acc),
  ]
  ctx.emit(rows, [*fill, *ctx.nest_body(outputs, body), *clear])


def _guard_sampled_term(ctx: LowerCtx, operand: Expr, slot: ProgramNode, term: ProgramNode) -> ProgramNode:
  entries = stored_indices(operand.type)
  logical = entries[np.argsort(ctx.physical_slots(operand, np.arange(len(entries))))]
  # Guard the multiply itself: selecting a zero operand still evaluates 0 * inf.
  current = operand
  while True:
    slots = stored_slots(current, logical)
    valid = slots >= 0
    if not np.any(valid):
      break
    logical[~valid] = -1
    if current.op == ExprOp.PACK:
      logical = slots
    elif current.op == ExprOp.VALUES:
      logical[valid] = stored_indices(current.args[0].type)[logical[valid]]
    elif current.op == ExprOp.SLICE:
      logical[valid] = slice_indices(current.args[0].shape, current.attrs["index"], logical[valid])
    elif current.op == ExprOp.TRANSPOSE:
      logical[valid] = transpose_entries(logical[valid], current.shape, tuple(map(int, np.argsort(current.attrs["axes"]))))
    elif current.op != ExprOp.RESHAPE:
      mask = _mask_of(current)
      if mask is None:
        break
      condition, inner = mask
      reads = np.full(len(logical), -1, dtype=np.int64)
      reads[valid] = stored_slots(condition, project_entries(logical[valid], condition.shape, current.shape))
      reads = ctx.physical_slots(condition, reads)
      index = ctx.index_at(reads, slot, shared=True)
      keep = p.load(p.view(ctx.raw_buf_of(condition), [index]))
      if np.any(reads < 0):
        present = ProgramNode(ProgramOp.LE, (p.const_int(0), index), dtype=dtypes.bool_)
        keep = ProgramNode(ProgramOp.SELECT, (present, keep, constant(False, dtypes.bool_)), dtype=dtypes.bool_)
      if not np.all(valid):
        applies = ProgramNode(ProgramOp.NE, (ctx.index_at(valid.astype(np.int64), slot, shared=True), p.const_int(0)), dtype=dtypes.bool_)
        keep = ProgramNode(ProgramOp.SELECT, (applies, keep, constant(True, dtypes.bool_)), dtype=dtypes.bool_)
      term = ProgramNode(ProgramOp.SELECT, (keep, term, p.const_float(0, dtype=operand.type.dtype)), dtype=operand.type.dtype)
      current = inner
      continue
    current = current.args[0]
  return term


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


def seed_regions(writes: np.ndarray, reads: Sequence[np.ndarray], width: int) -> tuple[tuple[MapRegion, int], ...]:
  """Group entries with equally long seed lists, retaining each entry's own seed coordinates."""
  result = []
  for region in map_regions(writes, reads):
    entries = region.writes % width
    order = np.lexsort((region.writes // width, entries))
    _, counts = np.unique(entries[order], return_counts=True)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    for count in np.unique(counts):
      groups = np.flatnonzero(counts == count)
      positions = order[(offsets[groups, None] + np.arange(count)).reshape(-1)]
      result.append((MapRegion(region.writes[positions], tuple(r[positions] for r in region.reads)), int(count)))
  return tuple(result)


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
    ctx.emit(ranges, [p.store(p.view(out, [i]), constant(0, node.type.dtype))])
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


def lower_supported_elementwise(ctx: LowerCtx, node: Expr, program_op) -> bool:
  """Plan nonzero seed pairs by entry, splitting regions at known-zero operands."""
  if not ctx.seed_pairs or len(node.shape) < 2:
    return False
  supports = [ctx.support_of(arg) for arg in node.args]
  support = ctx.support_of(node)
  if (support is None or len(support) == node.size) and all(s is None or len(s) == arg.size for arg, s in zip(node.args, supports, strict=True)):
    return False
  logical = np.arange(node.size, dtype=np.int64) if support is None else support_indices(support)
  reads = []
  for arg, known in zip(node.args, supports, strict=True):
    slots = project_entries(logical, arg.shape, node.shape)
    reads.append(slots if known is None else np.where(np.isin(slots, support_indices(known)), slots, -1))
  out = ctx.alloc_tmp(node)
  if len(logical) < node.size:
    ranges, (k,) = ctx.nest((node.size,), f"zero_{out.attrs['name']}")
    ctx.emit(ranges, [p.store(p.view(out, [k]), constant(0, node.type.dtype))])
  width = int(np.prod(node.shape[1:], dtype=np.int64))
  for region, count in seed_regions(logical, reads, width):
    entries = len(region.writes) // count
    grid = region.writes.reshape(entries, count)
    if np.all(grid // width == grid[0] // width):
      region = MapRegion(grid.T.reshape(-1), tuple(slots.reshape(entries, count).T.reshape(-1) for slots in region.reads))
      extents = count, entries
    else:
      extents = entries, count
    ranges, (outer, inner) = ctx.nest(extents, f"seed_{out.attrs['name']}")
    k = p.add(p.mul(outer, p.const_int(extents[1])), inner)
    values = []
    for arg, slots in zip(node.args, region.reads, strict=True):
      if slots[0] < 0:
        values.append(constant(0, arg.type.dtype))
      elif arg.op == ExprOp.CONST and arg.value is not None and np.all((sample := arg.value.reshape(-1)[slots]) == sample[0]):
        values.append(constant(sample[0].item(), arg.type.dtype))
      else:
        values.append(ctx.read_stored(arg, slots, k))
    value = fold_program(ProgramNode(program_op, tuple(values), dtype=node.type.dtype))
    ctx.emit(ranges, [p.store(p.view(out, [ctx.index_at(region.writes, k, shared=True)]), value)])
  return True
