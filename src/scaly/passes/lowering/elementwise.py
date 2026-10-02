"""Lower elementwise unary and binary expressions with broadcasting."""

from __future__ import annotations

from ...ir import program as p
from ...ir.expr import Expr, ExprOp
from ...ir.program import ProgramNode, ProgramOp, RangeKind
from .ctx import LowerCtx, lowers, _broadcast_index_p, _size_of


_UNARY: dict[ExprOp, ProgramOp] = {
  ExprOp.NEG: ProgramOp.NEG,
  ExprOp.SIN: ProgramOp.SIN,
  ExprOp.COS: ProgramOp.COS,
  ExprOp.TAN: ProgramOp.TAN,
  ExprOp.ASIN: ProgramOp.ASIN,
  ExprOp.ACOS: ProgramOp.ACOS,
  ExprOp.ATAN: ProgramOp.ATAN,
  ExprOp.SINH: ProgramOp.SINH,
  ExprOp.COSH: ProgramOp.COSH,
  ExprOp.TANH: ProgramOp.TANH,
  ExprOp.ERF: ProgramOp.ERF,
  ExprOp.EXP: ProgramOp.EXP,
  ExprOp.LOG: ProgramOp.LOG,
  ExprOp.SQRT: ProgramOp.SQRT,
  ExprOp.ABS: ProgramOp.ABS,
  ExprOp.FLOOR: ProgramOp.FLOOR,
  ExprOp.CEIL: ProgramOp.CEIL,
}

_BINARY: dict[ExprOp, ProgramOp] = {
  ExprOp.ADD: ProgramOp.ADD,
  ExprOp.SUB: ProgramOp.SUB,
  ExprOp.MUL: ProgramOp.MUL,
  ExprOp.DIV: ProgramOp.DIV,
  ExprOp.POW: ProgramOp.POW,
  ExprOp.ATAN2: ProgramOp.ATAN2,
  ExprOp.MINIMUM: ProgramOp.MINIMUM,
  ExprOp.MAXIMUM: ProgramOp.MAXIMUM,
}


def _emit_elementwise(ctx: LowerCtx, node: Expr, pop: ProgramOp, *, arity: int) -> None:
  out = ctx.alloc_tmp(node)
  vname = f"i_{out.attrs['name']}"
  rng = p.range_(vname, 0, _size_of(node.shape), kind=RangeKind.GLOBAL)
  i = p.var(vname)
  loads = tuple(p.load(p.view(ctx.buf_of(a), [_broadcast_index_p(i, a.shape, node.shape)])) for a in node.args[:arity])
  computed = ProgramNode(pop, loads, dtype=node.type.dtype)
  ctx.statements.append(p.for_(rng, [p.store(p.view(out, [i]), computed)]))


@lowers(*_UNARY)
def _lower_unary(ctx: LowerCtx, node: Expr) -> None:
  _emit_elementwise(ctx, node, _UNARY[ExprOp(node.op)], arity=1)


@lowers(*_BINARY)
def _lower_binary(ctx: LowerCtx, node: Expr) -> None:
  _emit_elementwise(ctx, node, _BINARY[ExprOp(node.op)], arity=2)
