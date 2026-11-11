"""Lower elementwise unary and binary expressions with broadcasting."""

from __future__ import annotations

from ...ir import program as p
from ...ir.expr import Expr, ExprOp
from ...ad.rules import ELEMENTWISE
from .ctx import LowerCtx, lowers
from .gather import lower_scatter_sum
from .plans import lower_elementwise, lower_supported_elementwise


@lowers(*ELEMENTWISE)
def _lower_elementwise(ctx: LowerCtx, node: Expr) -> None:
  """One flat loop over the output. An operand of the output's shape reads at the loop variable, and
  a broadcast one at its trailing coordinates as quotients and remainders, zero on a broadcast axis.

  A broadcast stays flat rather than a nest of rows: lanes over such a nest store each row apart, and
  the gathers reading the result back stall (unbumpercars C=8 ran 7% slower). Until lanes handle
  nests (#71). An operand read through a transpose makes the loop a nest over the output, merged where
  contiguous, so the remap divides only on the axes a reshape merges."""
  if node.type.pattern is not None:
    lower_elementwise(ctx, node, ELEMENTWISE[ExprOp(node.op)].program)
    return
  if (scatters := ctx.sums.get(node.id)) is not None:
    lower_scatter_sum(ctx, node, scatters)
    return
  if lower_supported_elementwise(ctx, node, ELEMENTWISE[ExprOp(node.op)].program):
    return
  out = ctx.alloc_tmp(node)
  if node.id in ctx.permutes:
    ranges, coords = ctx.nest(node.shape, f"i_{out.attrs['name']}")
    ctx.emit(ranges, [ctx.write(node, coords, ctx.compute(node, coords))], delinearize=True)
    return
  ranges, (flat,) = ctx.nest((node.size,), f"i_{out.attrs['name']}")
  ctx.emit(ranges, [p.store(p.view(out, [flat]), ctx.compute_flat(node, flat))])
