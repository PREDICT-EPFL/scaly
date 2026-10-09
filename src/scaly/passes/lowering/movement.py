"""Lower constants, reshapes, slices, transposes, stacks, and concatenations."""

from __future__ import annotations

from ...ir import program as p
from ...ir.expr import Expr, ExprOp
from ...ir.program import ProgramNode, RangeKind
from .ctx import LowerCtx, lowers, _shape_or_scalar, _size_of, _stride, _coord_p, _flat_index_p, _row_major_strides, _affine_sum


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
  flat offset, no stride) becomes a zero-copy pointer alias of its source; otherwise each output
  element reads the source via flat-index arithmetic. Covers rank-1, multi-dim, integer, strided."""
  src = node.args[0]
  src_shape = src.shape
  index = node.attrs["index"]
  # Contiguous + not an output: alias the source pointer instead of copying (legacy parity).
  if node.id not in ctx._output_alias and (offset := _contiguous_slice_offset(index, src_shape, node.shape)) is not None:
    alias = ctx.new_alias(node.type.dtype, node.shape, ctx.value_buffers[src.id], offset)
    ctx.value_buffers[node.id] = alias.attrs["name"]
    return
  out = ctx.alloc_tmp(node)
  vname = ctx.names.allocate(f"i_{out.attrs['name']}")
  rng = p.range_(vname, 0, _size_of(node.shape), kind=RangeKind.GLOBAL)
  k = p.var(vname)
  coords: list[ProgramNode] = []
  out_dim = 0
  for dim, item in enumerate(index):
    if isinstance(item, int):
      coords.append(p.const_int(item if item >= 0 else int(src_shape[dim]) + item))
      continue
    start, _stop, step = item.indices(int(src_shape[dim]))
    c = _coord_p(k, node.shape, out_dim)
    if step != 1:
      c = p.mul(c, p.const_int(step))
    coords.append(c if (start == 0 and step == 1) else p.add(p.const_int(start), c))
    out_dim += 1
  src_idx = _flat_index_p(coords, src_shape)
  ctx.statements.append(p.for_(rng, [p.store(p.view(out, [k]), p.load(p.view(ctx.buf_of(src), [src_idx])))]))


@lowers(ExprOp.TRANSPOSE)
def _lower_transpose(ctx: LowerCtx, node: Expr) -> None:
  """Permuted copy: ``out[Σ o_i·out_stride_i] = src[Σ o_i·src_stride_{axes[i]}]``, one loop per output axis."""
  src = node.args[0]
  axes = tuple(int(a) for a in node.attrs["axes"])
  src_shape, out_shape = src.shape, node.shape
  out = ctx.alloc_tmp(node)
  src_strides = _row_major_strides(src_shape)
  out_strides = _row_major_strides(out_shape)
  ranges, loop_vars = [], []
  for i, d in enumerate(out_shape):
    name = ctx.names.allocate(f"d{i}_{out.attrs['name']}")
    ranges.append(p.range_(name, 0, int(d), kind=RangeKind.GLOBAL))
    loop_vars.append(p.var(name))
  out_idx = _affine_sum(loop_vars, out_strides)
  src_idx = _affine_sum(loop_vars, [src_strides[axes[i]] for i in range(len(out_shape))])
  stmt: ProgramNode = p.store(p.view(out, [out_idx]), p.load(p.view(ctx.buf_of(src), [src_idx])))
  for rng in reversed(ranges):
    stmt = p.for_(rng, [stmt])
  ctx.statements.append(stmt)


@lowers(ExprOp.STACK)
def _lower_stack(ctx: LowerCtx, node: Expr) -> None:
  """Stack ``n`` rank-r inputs along a new ``axis`` into a rank-(r+1) output: each input
  occupies index ``i`` along the new axis. Per element, decompose the input flat index into
  its coords, insert ``i`` at ``axis``, recombine against the output shape."""
  axis = int(node.attrs.get("axis", 0))
  out_shape = node.shape
  out = ctx.alloc_tmp(node)
  for i, src in enumerate(node.args):
    src_shape = src.shape
    name = ctx.names.allocate(f"j_{out.attrs['name']}_{i}")
    rng = p.range_(name, 0, _size_of(src_shape), kind=RangeKind.GLOBAL)
    j = p.var(name)
    src_coords = [_coord_p(j, src_shape, d) for d in range(len(src_shape))]
    out_coords = src_coords[:axis] + [p.const_int(i)] + src_coords[axis:]
    dst = _flat_index_p(out_coords, out_shape)
    ctx.statements.append(p.for_(rng, [p.store(p.view(out, [dst]), p.load(p.view(ctx.buf_of(src), [j])))]))


@lowers(ExprOp.CONCAT)
def _lower_concat(ctx: LowerCtx, node: Expr) -> None:
  """Concatenate inputs along ``axis``: each input keeps its shape but its ``axis`` coordinate is
  shifted by the running offset. Per element, decompose / shift / recombine against the output."""
  axis = int(node.attrs.get("axis", 0))
  out_shape = node.shape
  out = ctx.alloc_tmp(node)
  offset = 0
  for i, src in enumerate(node.args):
    src_shape = src.shape
    name = ctx.names.allocate(f"j_{out.attrs['name']}_{i}")
    rng = p.range_(name, 0, _size_of(src_shape), kind=RangeKind.GLOBAL)
    j = p.var(name)
    coords = [_coord_p(j, src_shape, d) for d in range(len(src_shape))]
    if offset:
      coords[axis] = p.add(p.const_int(offset), coords[axis])
    dst = _flat_index_p(coords, out_shape)
    ctx.statements.append(p.for_(rng, [p.store(p.view(out, [dst]), p.load(p.view(ctx.buf_of(src), [j])))]))
    offset += int(src_shape[axis])
