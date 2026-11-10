"""Lower constants, reshapes, slices, transposes, stacks, and concatenations."""

from __future__ import annotations

import numpy as np

from ...ir import program as p
from ...ir.expr import Expr, ExprOp
from ...ir.program import ProgramNode
from ...ir.types import _uniform_value
from .ctx import LowerCtx, lowers, _flat_coordinates, _shift, _shape_or_scalar, _stride
from .plans import lower_movement, emit_map


def _sparse_movement(ctx: LowerCtx, node: Expr) -> bool:
  if node.type.pattern is not None or any(a.type.pattern is not None for a in node.args):
    lower_movement(ctx, node)
    return True
  return False


@lowers(ExprOp.RESHAPE, ExprOp.STOP_GRADIENT, ExprOp.PACK, ExprOp.VALUES)
def _lower_reshape(ctx: LowerCtx, node: Expr) -> None:
  if node.op in (ExprOp.PACK, ExprOp.VALUES):
    if node.id in ctx._output_alias:
      emit_map(ctx, node, [np.arange(node.type.storage_size, dtype=np.int64)], lambda v: v[0])
    else:
      ctx.value_buffers[node.id] = ctx.buf_of(node.args[0]).attrs["name"]
    return
  if _sparse_movement(ctx, node):
    return
  # Metadata-only: the result aliases its source buffer (no copy).
  ctx.value_buffers[node.id] = ctx.value_buffers[node.args[0].id]


@lowers(ExprOp.CONST)
def _lower_const(ctx: LowerCtx, node: Expr) -> None:
  value = node.value
  assert value is not None
  # CONST outputs never alias output buffers. emit_outputs copies the table or filled private buffer.
  uniform = node.size > 1024 and _uniform_value(value) is not None
  name = ctx.names.allocate(f"k{ctx._tmp}", generated=True)
  ctx._tmp += 1
  shape = (1,) if uniform else _shape_or_scalar(node.shape)
  values = [value.flat[0].item()] if uniform else value.reshape(-1).tolist()
  buf = p.const_buffer(name, node.type.dtype, shape, values)
  ctx.buffers[name] = buf
  ctx.statements.append(buf)
  if uniform:
    out = ctx.alloc_tmp(node)
    ranges, (i,) = ctx.nest((node.size,), f"i_{out.attrs['name']}")
    ctx.emit(ranges, [p.store(p.view(out, [i]), p.load(p.view(buf, [p.const_int(0)])))])
  else:
    ctx.value_buffers[node.id] = name


def _contiguous_slice_offset(index: tuple[object, ...], in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> int | None:
  """If the slice selects a contiguous sub-block of ``src`` at a constant flat offset (no stride,
  at most one partial leading slice with full trailing dims), return that offset; else None. Port
  of ``codegen/c.py::_contiguous_slice_offset`` — the precondition for pointer aliasing."""
  if not out_shape:
    offset = 0
    for dim, item in enumerate(index):
      if not isinstance(item, int):
        return None
      offset += (item if item >= 0 else in_shape[dim] + item) * _stride(in_shape, dim)
    return offset
  first_slice = None
  offset = 0
  out_dim = 0
  for dim, item in enumerate(index):
    stride = _stride(in_shape, dim)
    if isinstance(item, int):
      if first_slice is not None and in_shape[dim] != 1:
        return None
      offset += (item if item >= 0 else in_shape[dim] + item) * stride
      continue
    assert isinstance(item, slice)
    start, stop, step = item.indices(int(in_shape[dim]))
    if step != 1:
      return None
    if first_slice is None:
      first_slice = dim
      offset += start * stride
      out_dim += 1
      continue
    if start != 0 or stop != int(in_shape[dim]) or out_shape[out_dim] != int(in_shape[dim]):
      return None
    out_dim += 1
  return offset


@lowers(ExprOp.SLICE)
def _lower_slice(ctx: LowerCtx, node: Expr) -> None:
  """General SLICE: integer indices drop a dim, slices keep one. A contiguous slice (constant
  flat offset, no stride) becomes a zero-copy pointer alias of its source; otherwise one flat loop
  over the output reads source coordinate ``start + step * c``. Covers rank-1, multi-dim, integer,
  strided."""
  if _sparse_movement(ctx, node):
    return
  src = node.args[0]
  src_shape = src.shape
  index = node.attrs["index"]
  # Contiguous + not an output: alias the source pointer instead of copying (legacy parity).
  if node.id not in ctx._output_alias and (offset := _contiguous_slice_offset(index, src_shape, node.shape)) is not None:
    alias = ctx.new_alias(node.type.dtype, node.shape, ctx.value_buffers[src.id], offset)
    ctx.value_buffers[node.id] = alias.attrs["name"]
    return
  out = ctx.alloc_tmp(node)
  # One flat loop, its coordinates as quotients and remainders, rather than a nest of short rows:
  # lanes over such a nest split each row's vector store across cache lines, and the loads that
  # read the copy back stall (chain M=9 ran 1.4 times slower). Until lanes handle nests (#71).
  ranges, (k,) = ctx.nest((node.size,), f"i_{out.attrs['name']}")
  coords = _flat_coordinates(k, node.shape)
  source: list[ProgramNode] = []
  out_dim = 0
  for dim, item in enumerate(index):
    if isinstance(item, int):
      source.append(p.const_int(item if item >= 0 else int(src_shape[dim]) + item))
      continue
    start, _stop, step = item.indices(int(src_shape[dim]))
    source.append(_shift(coords[out_dim], start, step))
    out_dim += 1
  ctx.emit(ranges, [p.store(p.view(out, [k]), ctx.read(src, source))])


@lowers(ExprOp.TRANSPOSE)
def _lower_transpose(ctx: LowerCtx, node: Expr) -> None:
  """Permuted copy: output coordinate ``i`` is source coordinate ``axes[i]``."""
  if _sparse_movement(ctx, node):
    return
  src = node.args[0]
  axes = tuple(int(a) for a in node.attrs["axes"])
  out = ctx.alloc_tmp(node)
  ranges, coords = ctx.nest(node.shape, f"i_{out.attrs['name']}")
  source = [coords[axes.index(dim)] for dim in range(len(axes))]
  ctx.emit(ranges, [ctx.write(node, coords, ctx.read(src, source))])


@lowers(ExprOp.STACK)
def _lower_stack(ctx: LowerCtx, node: Expr) -> None:
  """Copy input ``i`` to index ``i`` along the new ``axis``."""
  if _sparse_movement(ctx, node):
    return
  axis = int(node.attrs.get("axis", 0))
  out = ctx.alloc_tmp(node)
  for i, src in enumerate(node.args):
    ranges, coords = ctx.nest(src.shape, f"j_{out.attrs['name']}_{i}")
    ctx.emit(ranges, [ctx.write(node, [*coords[:axis], p.const_int(i), *coords[axis:]], ctx.read(src, coords))])


@lowers(ExprOp.CONCAT)
def _lower_concat(ctx: LowerCtx, node: Expr) -> None:
  """Copy each input, shifting only the concatenated coordinate."""
  if _sparse_movement(ctx, node):
    return
  axis = int(node.attrs.get("axis", 0))
  out = ctx.alloc_tmp(node)
  offset = 0
  for i, src in enumerate(node.args):
    ranges, coords = ctx.nest(src.shape, f"j_{out.attrs['name']}_{i}")
    shifted = list(coords)
    shifted[axis] = _shift(coords[axis], offset)
    ctx.emit(ranges, [ctx.write(node, shifted, ctx.read(src, coords))])
    offset += int(src.shape[axis])
