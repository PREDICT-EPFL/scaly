"""Lower gathers and scatters using affine indices and residual tables."""

from __future__ import annotations

from ...ir import program as p
from ...ir.expr import _index_values, Expr, ExprOp
from ...ir.program import ProgramNode, ProgramOp, RangeKind
from ...ir.types import dtypes
from ..arith import constant
from .ctx import LowerCtx, lowers, _size_of
from .plans import lower_movement
from .in_place import fused_value, scatter_reuse


def _address(ctx: LowerCtx, node: Expr, k: ProgramNode, *, clamp: bool = False) -> tuple[ProgramNode, ProgramNode]:
  base, index = node.args[:2]
  axis = node.attrs.get("axis")
  if index.value is not None:
    address = ctx.index_at(_index_values(node).reshape(-1), k)
    valid = ProgramNode(
      ProgramOp.AND,
      (
        ProgramNode(ProgramOp.LE, (p.const_int(0), address), dtype=dtypes.bool_),
        ProgramNode(ProgramOp.LT, (address, p.const_int(base.size)), dtype=dtypes.bool_),
      ),
      dtype=dtypes.bool_,
    )
    if clamp:
      lower = ProgramNode(
        ProgramOp.SELECT, (ProgramNode(ProgramOp.LE, (p.const_int(0), address), dtype=dtypes.bool_), address, p.const_int(0)), dtype=dtypes.int64
      )
      address = ProgramNode(
        ProgramOp.SELECT,
        (ProgramNode(ProgramOp.LT, (address, p.const_int(base.size)), dtype=dtypes.bool_), lower, p.const_int(base.size - 1)),
        dtype=dtypes.int64,
      )
    return address, valid
  after = 1 if axis is None else _size_of(base.shape[axis + 1 :])
  position = k if axis is None else p.mod(p.div(k, p.const_int(after)), p.const_int(index.size))
  raw = p.load(p.view(ctx.buf_of(index), [position]))
  bound = base.size if axis is None else base.shape[axis]
  low = ProgramNode(ProgramOp.LE, (p.const_int(0), raw), dtype=dtypes.bool_)
  high = ProgramNode(ProgramOp.LT, (raw, p.const_int(bound)), dtype=dtypes.bool_)
  valid = ProgramNode(ProgramOp.AND, (low, high), dtype=dtypes.bool_)
  if clamp:
    raw = ProgramNode(ProgramOp.SELECT, (low, raw, p.const_int(0)), dtype=dtypes.int64)
    raw = ProgramNode(ProgramOp.SELECT, (high, raw, p.const_int(bound - 1)), dtype=dtypes.int64)
  if axis is not None:
    before = p.div(k, p.const_int(index.size * after))
    raw = p.add(p.mul(p.add(p.mul(before, p.const_int(bound)), raw), p.const_int(after)), p.mod(k, p.const_int(after)))
  return raw, valid


@lowers(ExprOp.GATHER)
def _lower_gather(ctx: LowerCtx, node: Expr) -> None:
  """``out[k] = src[indices[k]]`` over the flat output, the index as arithmetic or a ``static const`` table."""
  if node.args[0].type.pattern is not None:
    lower_movement(ctx, node)
    return
  src = node.args[0]
  if node.args[1].value is None or node.attrs.get("mode") == "fill":
    out = ctx.alloc_tmp(node)
    ranges, (k,) = ctx.nest((node.size,), f"i_{out.attrs['name']}")
    fill = constant(node.attrs.get("fill_value", 0), node.type.dtype)
    if src.size == 0:
      value = fill
    else:
      address, valid = _address(ctx, node, k, clamp=node.attrs.get("mode") == "fill")
      value = p.load(p.view(ctx.buf_of(src), [address]))
      if node.attrs.get("mode") == "fill":
        value = ProgramNode(ProgramOp.SELECT, (valid, value, fill), dtype=node.type.dtype)
    ctx.emit(ranges, [p.store(p.view(out, [k]), value)])
    return
  idx = _index_values(node).reshape(-1)
  out = ctx.alloc_tmp(node)
  ranges, (k,) = ctx.nest((len(idx),), f"i_{out.attrs['name']}")
  ctx.emit(ranges, [p.store(p.view(out, [k]), p.load(p.view(ctx.buf_of(src), [ctx.index_at(idx, k)])))])


@lowers(ExprOp.SCATTER)
def _lower_scatter(ctx: LowerCtx, node: Expr) -> None:
  """Zero the output, then store unique indices or accumulate repeated indices in order."""
  if node.args[2].type.pattern is not None and node.args[0].support == frozenset() and node.args[1].value is not None:
    lower_movement(ctx, node)
    return
  src = node.args[2]
  base, index, _ = node.args
  reuse = node.id in ctx.value_buffers or scatter_reuse(ctx, node)
  if index.value is None or base.support != frozenset() or node.attrs["combine"] != "add" or node.attrs["mode"] == "fill":
    out = ctx.alloc_tmp(node)
    fill = node.attrs["mode"] == "fill"
    work = ctx.new_private(node.type.dtype, (node.size + 1,)) if fill else out
    ranges, (k,) = ctx.nest((node.size + int(fill),), f"zero_{work.attrs['name']}", kind=RangeKind.SERIAL)
    initial = constant(0, node.type.dtype)
    if base.support != frozenset() and node.size:
      address = (
        ProgramNode(
          ProgramOp.SELECT, (ProgramNode(ProgramOp.LT, (k, p.const_int(node.size)), dtype=dtypes.bool_), k, p.const_int(0)), dtype=dtypes.int64
        )
        if fill
        else k
      )
      initial = p.load(p.view(ctx.buf_of(base), [address]))
    if not reuse and base.support == frozenset():
      loop = p.for_(ranges[0], [p.store(p.view(work, [k]), initial)])
      ctx.statements.append(ProgramNode(loop.op, loop.args, {**loop.attrs, "zero_fill": True}, dtype=loop.dtype))
    elif not reuse:
      ctx.emit(ranges, [p.store(p.view(work, [k]), initial)])
    ranges, (i,) = ctx.nest((src.size,), f"scatter_{work.attrs['name']}", kind=RangeKind.REDUCE)
    dst, valid = _address(ctx, node, i, clamp=fill)
    if fill:
      dst = ProgramNode(ProgramOp.SELECT, (valid, dst, p.const_int(node.size)), dtype=dtypes.int64)
    target = p.view(work, [dst])
    value = fused_value(ctx, src, node, i, dst) if node.id in ctx.fused_scatter else p.load(p.view(ctx.buf_of(src), [i]))
    combine = node.attrs["combine"]
    if combine == "add":
      value = p.add(p.load(target), value)
    elif combine in ("max", "min"):
      old = p.load(target)
      reduced = ProgramNode(ProgramOp.MAXIMUM if combine == "max" else ProgramOp.MINIMUM, (old, value), dtype=node.type.dtype)
      reduced = ProgramNode(ProgramOp.SELECT, (ProgramNode(ProgramOp.NE, (value, value), dtype=dtypes.bool_), value, reduced), dtype=node.type.dtype)
      value = ProgramNode(ProgramOp.SELECT, (ProgramNode(ProgramOp.NE, (old, old), dtype=dtypes.bool_), old, reduced), dtype=node.type.dtype)
    ctx.emit(ranges, [p.store(target, value)])
    if fill:
      ranges, (k,) = ctx.nest((node.size,), f"copy_{out.attrs['name']}")
      ctx.emit(ranges, [p.store(p.view(out, [k]), p.load(p.view(work, [k])))])
    return
  idx = _index_values(node).reshape(-1)
  out = ctx.alloc_tmp(node)
  zname = ctx.names.allocate(f"z_{out.attrs['name']}")
  zrng = p.range_(zname, 0, _size_of(node.shape), kind=RangeKind.GLOBAL)
  z = p.var(zname)
  ctx.statements.append(p.for_(zrng, [p.store(p.view(out, [z]), constant(0, node.type.dtype))]))
  iname = ctx.names.allocate(f"i_{out.attrs['name']}")
  repeated = len(set(idx.tolist())) != len(idx)
  irng = p.range_(iname, 0, len(idx), kind=RangeKind.REDUCE if repeated else RangeKind.GLOBAL)
  i = p.var(iname)
  dst = ctx.index_at(idx, i)
  target = p.view(out, [dst])
  value = p.load(p.view(ctx.buf_of(src), [i]))
  ctx.statements.append(p.for_(irng, [p.store(target, p.add(p.load(target), value) if repeated else value)]))
