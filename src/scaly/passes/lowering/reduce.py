"""Lower full tensor reductions into accumulator loops."""

from __future__ import annotations

import numpy as np

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
  value = (
    ctx.read_stored(src, np.arange(src.type.storage_size, dtype=np.int64), i)
    if src.type.pattern is not None
    else p.load(p.view(ctx.buf_of(src), [i]))
  )
  emit_reduction(ctx, [], reductions, p.view(acc, [p.const_int(0)]), value)
