"""Lower full tensor reductions into accumulator loops."""

from __future__ import annotations

from ...ir import program as p
from ...ir.expr import Expr, ExprOp
from ...ir.program import RangeKind
from .ctx import LowerCtx, lowers, _size_of


@lowers(ExprOp.SUM)
def _lower_sum(ctx: LowerCtx, node: Expr) -> None:
  """Full reduction to a scalar: zero the accumulator, then a REDUCE loop adds every element."""
  src = node.args[0]
  acc = ctx.alloc_tmp(node)
  z = p.const_int(0)
  ctx.statements.append(p.store(p.view(acc, [z]), p.const_float(0.0, dtype=node.type.dtype)))
  name = ctx.names.allocate(f"i_{acc.attrs['name']}")
  rng = p.range_(name, 0, _size_of(src.shape), kind=RangeKind.REDUCE)
  i = p.var(name)
  ctx.statements.append(p.for_(rng, [p.store(p.view(acc, [z]), p.add(p.load(p.view(acc, [z])), p.load(p.view(ctx.buf_of(src), [i]))))]))
