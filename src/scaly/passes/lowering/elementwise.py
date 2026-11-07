"""Lower elementwise unary and binary expressions with broadcasting."""

from __future__ import annotations

from ...ir import program as p
from ...ir.expr import Expr, ExprOp
from ...ir.program import ProgramNode
from ...ad.rules import ELEMENTWISE
from .ctx import LowerCtx, lowers


@lowers(*ELEMENTWISE)
def _lower_elementwise(ctx: LowerCtx, node: Expr) -> None:
  """One nest over the output; an operand reads its trailing coordinates, and zero on a broadcast axis."""
  out = ctx.alloc_tmp(node)
  ranges, coords = ctx.nest(node.shape, f"i_{out.attrs['name']}")
  loads = tuple(
    ctx.read(a, [c if d != 1 else p.const_int(0) for c, d in zip(coords[len(coords) - len(a.shape) :], a.shape, strict=True)]) for a in node.args
  )
  computed = ProgramNode(ELEMENTWISE[ExprOp(node.op)].program, loads, dtype=node.type.dtype)
  ctx.emit(ranges, [ctx.write(node, coords, computed)])
