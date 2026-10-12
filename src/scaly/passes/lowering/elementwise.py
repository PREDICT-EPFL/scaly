"""Lower elementwise unary and binary expressions with broadcasting."""

from __future__ import annotations

from ...ir import program as p
from ...ir.expr import Expr, ExprOp
from ...ir.program import ProgramNode, RangeKind
from ...ad.rules import ELEMENTWISE
from .ctx import LowerCtx, lowers, _broadcast_index_p, _size_of


@lowers(*ELEMENTWISE)
def _lower_elementwise(ctx: LowerCtx, node: Expr) -> None:
  out = ctx.alloc_tmp(node)
  vname = ctx.names.allocate(f"i_{out.attrs['name']}")
  rng = p.range_(vname, 0, _size_of(node.shape), kind=RangeKind.GLOBAL)
  i = p.var(vname)
  loads = tuple(p.load(p.view(ctx.buf_of(a), [_broadcast_index_p(i, a.shape, node.shape)])) for a in node.args)
  computed = ProgramNode(ELEMENTWISE[ExprOp(node.op)].program, loads, dtype=node.type.dtype)
  ctx.statements.append(p.for_(rng, [p.store(p.view(out, [i]), computed)]))
