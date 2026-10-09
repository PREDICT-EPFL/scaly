"""Lower gathers and scatters using affine indices and residual tables."""

from __future__ import annotations

from ...ir import program as p
from ...ir.expr import Expr, ExprOp
from ...ir.program import RangeKind
from .ctx import LowerCtx, lowers, _size_of


@lowers(ExprOp.GATHER)
def _lower_gather(ctx: LowerCtx, node: Expr) -> None:
  """``out[k] = src[indices[k]]`` via a ``static const`` index table + one GLOBAL loop (any size)."""
  src = node.args[0]
  idx = node.attrs["indices"].reshape(-1)
  out = ctx.alloc_tmp(node)
  vname = ctx.names.allocate(f"i_{out.attrs['name']}")
  rng = p.range_(vname, 0, _size_of(node.shape), kind=RangeKind.GLOBAL)
  k = p.var(vname)
  src_idx = ctx.index_at(idx, k)
  ctx.statements.append(p.for_(rng, [p.store(p.view(out, [k]), p.load(p.view(ctx.buf_of(src), [src_idx])))]))


@lowers(ExprOp.SCATTER)
def _lower_scatter(ctx: LowerCtx, node: Expr) -> None:
  """Zero the output, then store unique indices or accumulate repeated indices in order."""
  src = node.args[0]
  idx = node.attrs["indices"].reshape(-1)
  out = ctx.alloc_tmp(node)
  zname = ctx.names.allocate(f"z_{out.attrs['name']}")
  zrng = p.range_(zname, 0, _size_of(node.shape), kind=RangeKind.GLOBAL)
  z = p.var(zname)
  ctx.statements.append(p.for_(zrng, [p.store(p.view(out, [z]), p.const_float(0.0, dtype=node.type.dtype))]))
  iname = ctx.names.allocate(f"i_{out.attrs['name']}")
  repeated = len(set(idx.tolist())) != len(idx)
  irng = p.range_(iname, 0, len(idx), kind=RangeKind.REDUCE if repeated else RangeKind.GLOBAL)
  i = p.var(iname)
  dst = ctx.index_at(idx, i)
  target = p.view(out, [dst])
  value = p.load(p.view(ctx.buf_of(src), [i]))
  ctx.statements.append(p.for_(irng, [p.store(target, p.add(p.load(target), value) if repeated else value)]))
