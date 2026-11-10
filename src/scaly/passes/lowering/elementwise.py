"""Lower elementwise unary and binary expressions with broadcasting."""

from __future__ import annotations

from ...ir import program as p
from ...ir.expr import Expr, ExprOp
from ...ir.program import ProgramNode
from ...ad.rules import ELEMENTWISE
from ...ir.types import stored_indices
from ...ir.support import project_entries
from .ctx import LowerCtx, lowers, _flat_coordinates


@lowers(*ELEMENTWISE)
def _lower_elementwise(ctx: LowerCtx, node: Expr) -> None:
  """One flat loop over the output. An operand of the output's shape reads at the loop variable, and
  a broadcast one at its trailing coordinates as quotients and remainders, zero on a broadcast axis.

  A broadcast stays flat rather than a nest of rows: lanes over such a nest store each row apart, and
  the gathers reading the result back stall (unbumpercars C=8 ran 7% slower). Until lanes handle
  nests (#71)."""
  out = ctx.alloc_tmp(node)
  if node.type.pattern is not None:
    ranges, (flat,) = ctx.nest((node.type.storage_size,), f"i_{out.attrs['name']}")
    logical = stored_indices(node.type)
    operands = tuple(ctx.read_indexed(a, project_entries(logical, a.shape, node.shape), flat) for a in node.args)
    computed = ProgramNode(ELEMENTWISE[ExprOp(node.op)].program, operands, dtype=node.type.dtype)
    ctx.emit(ranges, [p.store(p.view(out, [flat]), computed)])
    return
  ranges, (flat,) = ctx.nest((node.size,), f"i_{out.attrs['name']}")
  coords = _flat_coordinates(flat, node.shape)

  def load(a: Expr) -> ProgramNode:
    if a.shape == node.shape:
      return p.load(p.view(ctx.buf_of(a), [flat]))
    return ctx.read(a, [c if d != 1 else p.const_int(0) for c, d in zip(coords[len(coords) - len(a.shape) :], a.shape, strict=True)])

  computed = ProgramNode(ELEMENTWISE[ExprOp(node.op)].program, tuple(load(a) for a in node.args), dtype=node.type.dtype)
  ctx.emit(ranges, [p.store(p.view(out, [flat]), computed)])
