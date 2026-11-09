"""Lower elementwise unary and binary expressions with broadcasting."""

from __future__ import annotations

from ...ir import program as p
from ...ir.expr import Expr, ExprOp
from ...ir.program import ProgramNode
from ...ad.rules import ELEMENTWISE
from .ctx import LowerCtx, lowers, _flat_coordinates


@lowers(*ELEMENTWISE)
def _lower_elementwise(ctx: LowerCtx, node: Expr) -> None:
  """One loop over the output; an operand reads its trailing coordinates, and zero on a broadcast axis.

  An op that broadcasts stays one flat loop with quotient and remainder coordinates, an operand of
  the output's shape reading at the flat position. Lanes over its nest of rows store each row apart,
  and the gathers reading the result back stall (unbumpercars C=8 ran 7% slower). Until lanes handle
  nests (#71)."""
  out = ctx.alloc_tmp(node)
  name = f"i_{out.attrs['name']}"
  if all(a.size == 1 or a.shape == node.shape for a in node.args):
    ranges, coords = ctx.nest(node.shape, name)
    flat = None
  else:
    ranges, (flat,) = ctx.nest((node.size,), name)
    coords = _flat_coordinates(flat, node.shape)

  def load(a: Expr) -> ProgramNode:
    if flat is not None and a.shape == node.shape:
      return p.load(p.view(ctx.buf_of(a), [flat]))
    return ctx.read(a, [c if d != 1 else p.const_int(0) for c, d in zip(coords[len(coords) - len(a.shape) :], a.shape, strict=True)])

  computed = ProgramNode(ELEMENTWISE[ExprOp(node.op)].program, tuple(load(a) for a in node.args), dtype=node.type.dtype)
  ctx.emit(ranges, [ctx.write(node, coords, computed) if flat is None else p.store(p.view(out, [flat]), computed)])
