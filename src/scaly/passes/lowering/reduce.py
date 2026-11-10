"""Lower full tensor reductions into accumulator loops."""

from __future__ import annotations

from ...ir import program as p
from ...ir.expr import Expr, ExprOp
from .ctx import LowerCtx, lowers
from .schedule import emit_reduction


@lowers(ExprOp.SUM)
def _lower_sum(ctx: LowerCtx, node: Expr) -> None:
  """Full reduction to one scalar accumulator, initialized even for an empty input."""
  src = node.args[0]
  acc = ctx.alloc_tmp(node)
  reductions, (i,) = ctx.reduce((src.type.storage_size,), f"i_{acc.attrs['name']}")
  emit_reduction(ctx, [], reductions, p.view(acc, [p.const_int(0)]), p.load(p.view(ctx.buf_of(src), [i])))
