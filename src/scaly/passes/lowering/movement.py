"""Lower constants, reshapes, slices, transposes, stacks, and concatenations."""

from __future__ import annotations

from ...ir import program as p
from ...ir.expr import Expr, ExprOp
from ...ir.program import ProgramNode
from .ctx import LowerCtx, lowers, _shift, _shape_or_scalar, _stride


@lowers(ExprOp.RESHAPE, ExprOp.STOP_GRADIENT)
def _lower_reshape(ctx: LowerCtx, node: Expr) -> None:
  # Metadata-only: the result aliases its source buffer (no copy).
  ctx.value_buffers[node.id] = ctx.value_buffers[node.args[0].id]


@lowers(ExprOp.CONST)
def _lower_const(ctx: LowerCtx, node: Expr) -> None:
  value = node.value
  assert value is not None
  # A constant of any size materializes as a read-only ``constant``-space buffer
  # (rendered ``static const``). Output-aliasing never applies to CONST, so
  # emit_outputs inserts a copy when a CONST is itself an output.
  name = ctx.names.allocate(f"k{ctx._tmp}", generated=True)
  ctx._tmp += 1
  buf = p.const_buffer(name, node.type.dtype, _shape_or_scalar(node.shape), [float(v) for v in value.reshape(-1)])
  ctx.buffers[name] = buf
  ctx.value_buffers[node.id] = name
  ctx.statements.append(buf)


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
  flat offset, no stride) becomes a zero-copy pointer alias of its source; otherwise a nest over the
  output reads source coordinate ``start + step * c``. Covers rank-1, multi-dim, integer, strided."""
  src = node.args[0]
  src_shape = src.shape
  index = node.attrs["index"]
  # Contiguous + not an output: alias the source pointer instead of copying (legacy parity).
  if node.id not in ctx._output_alias and (offset := _contiguous_slice_offset(index, src_shape, node.shape)) is not None:
    alias = ctx.new_alias(node.type.dtype, node.shape, ctx.value_buffers[src.id], offset)
    ctx.value_buffers[node.id] = alias.attrs["name"]
    return
  out = ctx.alloc_tmp(node)
  ranges, coords = ctx.nest(node.shape, f"i_{out.attrs['name']}")
  source: list[ProgramNode] = []
  out_dim = 0
  for dim, item in enumerate(index):
    if isinstance(item, int):
      source.append(p.const_int(item if item >= 0 else int(src_shape[dim]) + item))
      continue
    start, _stop, step = item.indices(int(src_shape[dim]))
    source.append(_shift(coords[out_dim], start, step))
    out_dim += 1
  ctx.emit(ranges, [ctx.write(node, coords, ctx.read(src, source))])


@lowers(ExprOp.TRANSPOSE)
def _lower_transpose(ctx: LowerCtx, node: Expr) -> None:
  """Permuted copy: output coordinate ``i`` is source coordinate ``axes[i]``."""
  src = node.args[0]
  axes = tuple(int(a) for a in node.attrs["axes"])
  out = ctx.alloc_tmp(node)
  ranges, coords = ctx.nest(node.shape, f"i_{out.attrs['name']}")
  source = [coords[axes.index(dim)] for dim in range(len(axes))]
  ctx.emit(ranges, [ctx.write(node, coords, ctx.read(src, source))])


@lowers(ExprOp.STACK)
def _lower_stack(ctx: LowerCtx, node: Expr) -> None:
  """Copy input ``i`` to index ``i`` along the new ``axis``."""
  axis = int(node.attrs.get("axis", 0))
  out = ctx.alloc_tmp(node)
  for i, src in enumerate(node.args):
    ranges, coords = ctx.nest(src.shape, f"j_{out.attrs['name']}_{i}")
    ctx.emit(ranges, [ctx.write(node, [*coords[:axis], p.const_int(i), *coords[axis:]], ctx.read(src, coords))])


@lowers(ExprOp.CONCAT)
def _lower_concat(ctx: LowerCtx, node: Expr) -> None:
  """Copy each input, shifting only the concatenated coordinate."""
  axis = int(node.attrs.get("axis", 0))
  out = ctx.alloc_tmp(node)
  offset = 0
  for i, src in enumerate(node.args):
    ranges, coords = ctx.nest(src.shape, f"j_{out.attrs['name']}_{i}")
    shifted = list(coords)
    shifted[axis] = _shift(coords[axis], offset)
    ctx.emit(ranges, [ctx.write(node, shifted, ctx.read(src, coords))])
    offset += int(src.shape[axis])
