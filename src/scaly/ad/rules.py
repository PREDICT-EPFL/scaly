"""The elementwise table: each op's local partials, NumPy fold, program op and C spelling, read by both AD modes."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..ir.expr import Expr, ExprOp, scatter
from ..ir.program import ProgramOp

type Partial = Callable[[Expr], Expr | int]


@dataclass(frozen=True, slots=True)
class Elementwise:
  """One row of the elementwise table.

  ``partials[i]`` builds the partial of a node with respect to operand ``i``, and is called only
  when that operand is active. A partial of ``1`` or ``-1`` passes the tangent through or negates
  it. ``partials`` is ``None`` for an op without a derivative. ``jvp`` replaces the sum of partials
  times tangents with a prescribed form, more stable or cheaper; it receives the tangents aligned
  with the output.
  ``c`` is the C spelling, a libm function or an operator, and ``c_integer`` a different spelling
  for integer operands. ``expensive`` marks a libm call that fusion must not duplicate.
  """

  program: ProgramOp
  numpy: Callable[..., np.ndarray | np.generic]
  c: str
  partials: tuple[Partial, ...] | None = None
  jvp: Callable[[Expr, Sequence[Expr | None]], Expr] | None = None
  c_integer: str | None = None
  expensive: bool = False


def _const(value: float, like: Expr) -> Expr:
  return Expr.const(value, dtype=like.type.dtype)


def _inverse_sqrt_one_minus_square(e: Expr) -> Expr:
  x = e.args[0]
  return _const(1, e) / (_const(1, e) - x ** _const(2, e)).sqrt()


def _atan2_denominator(e: Expr) -> Expr:
  y, x = e.args
  return x * x + y * y


def _mul_jvp(e: Expr, tangents: Sequence[Expr | None]) -> Expr:
  (x, y), (dx, dy) = e.args, tangents
  if x is y:
    return _seed_axis(_const(2, e) * x) * dx
  left = None if dx is None else dx * _seed_axis(y, e)
  right = None if dy is None else _seed_axis(x, e) * dy
  if left is None:
    assert right is not None
    return right
  return left if right is None else left + right


def _div_jvp(e: Expr, tangents: Sequence[Expr | None]) -> Expr:
  dx, db = tangents
  quotient = _seed_axis(e)
  numerator = dx if db is None else -(quotient * db) if dx is None else dx - quotient * db
  return numerator / _seed_axis(e.args[1], e)


# NumPy has no erf; frompyfunc keeps constant folding vectorized without adding SciPy.
def _erf(x: np.ndarray) -> np.ndarray:
  return np.asarray(np.frompyfunc(math.erf, 1, 1)(x), dtype=np.float64)


ELEMENTWISE: dict[ExprOp, Elementwise] = {
  ExprOp.NEG: Elementwise(ProgramOp.NEG, np.negative, "-", (lambda e: -1,)),
  ExprOp.SIN: Elementwise(ProgramOp.SIN, np.sin, "sin", (lambda e: e.args[0].cos(),), expensive=True),
  ExprOp.COS: Elementwise(ProgramOp.COS, np.cos, "cos", (lambda e: -e.args[0].sin(),), expensive=True),
  ExprOp.TAN: Elementwise(ProgramOp.TAN, np.tan, "tan", (lambda e: _const(1, e) / e.args[0].cos() ** _const(2, e),), expensive=True),
  ExprOp.ASIN: Elementwise(ProgramOp.ASIN, np.arcsin, "asin", (_inverse_sqrt_one_minus_square,), expensive=True),
  ExprOp.ACOS: Elementwise(ProgramOp.ACOS, np.arccos, "acos", (lambda e: -_inverse_sqrt_one_minus_square(e),), expensive=True),
  ExprOp.ATAN: Elementwise(ProgramOp.ATAN, np.arctan, "atan", (lambda e: _const(1, e) / (_const(1, e) + e.args[0] ** _const(2, e)),), expensive=True),
  ExprOp.SINH: Elementwise(ProgramOp.SINH, np.sinh, "sinh", (lambda e: e.args[0].cosh(),), expensive=True),
  ExprOp.COSH: Elementwise(ProgramOp.COSH, np.cosh, "cosh", (lambda e: e.args[0].sinh(),), expensive=True),
  ExprOp.TANH: Elementwise(ProgramOp.TANH, np.tanh, "tanh", (lambda e: _const(1, e) - e * e,), expensive=True),
  ExprOp.ERF: Elementwise(
    ProgramOp.ERF, _erf, "erf", (lambda e: _const(2 / math.sqrt(math.pi), e) * (-(e.args[0] ** _const(2, e))).exp(),), expensive=True
  ),
  ExprOp.EXP: Elementwise(ProgramOp.EXP, np.exp, "exp", (lambda e: e,), expensive=True),
  ExprOp.LOG: Elementwise(ProgramOp.LOG, np.log, "log", (lambda e: _const(1, e) / e.args[0],), expensive=True),
  ExprOp.SQRT: Elementwise(ProgramOp.SQRT, np.sqrt, "sqrt", (lambda e: _const(0.5, e) / e,), expensive=True),
  ExprOp.ABS: Elementwise(ProgramOp.ABS, np.abs, "fabs", (lambda e: e.args[0] / e,)),
  ExprOp.FLOOR: Elementwise(ProgramOp.FLOOR, np.floor, "floor"),
  ExprOp.CEIL: Elementwise(ProgramOp.CEIL, np.ceil, "ceil"),
  ExprOp.ADD: Elementwise(ProgramOp.ADD, np.add, "+", (lambda e: 1, lambda e: 1)),
  ExprOp.SUB: Elementwise(ProgramOp.SUB, np.subtract, "-", (lambda e: 1, lambda e: -1)),
  ExprOp.MUL: Elementwise(ProgramOp.MUL, np.multiply, "*", (lambda e: e.args[1], lambda e: e.args[0]), _mul_jvp),
  ExprOp.DIV: Elementwise(ProgramOp.DIV, np.divide, "/", (lambda e: _const(1, e) / e.args[1], lambda e: -(e / e.args[1])), _div_jvp),
  ExprOp.POW: Elementwise(
    ProgramOp.POW,
    np.power,
    "pow",
    (lambda e: e.args[1] * e.args[0] ** (e.args[1] - _const(1, e)), lambda e: e * e.args[0].log()),
    expensive=True,
  ),
  ExprOp.ATAN2: Elementwise(
    ProgramOp.ATAN2,
    np.arctan2,
    "atan2",
    (lambda e: e.args[1] / _atan2_denominator(e), lambda e: -e.args[0] / _atan2_denominator(e)),
    expensive=True,
  ),
  ExprOp.MINIMUM: Elementwise(ProgramOp.MINIMUM, np.fmin, "fmin", c_integer="({0} < {1} ? {0} : {1})"),
  ExprOp.MAXIMUM: Elementwise(ProgramOp.MAXIMUM, np.fmax, "fmax", c_integer="({0} > {1} ? {0} : {1})"),
}


def tangent(expr: Expr, tangents: Sequence[Expr | None], nseed: int) -> Expr:
  """The tangent of an elementwise ``expr``, shaped ``(nseed, ...)``, from its operands' tangents.

  Each tangent is ``(nseed, *operand.shape)``, or ``None`` for an inactive operand, whose partial
  is never built.
  """
  row = ELEMENTWISE[ExprOp(expr.op)]
  assert row.partials is not None
  aligned = [None if t is None else _broadcast_tangent(t, arg, expr, nseed) for arg, t in zip(expr.args, tangents, strict=True)]
  if row.jvp is not None:
    return row.jvp(expr, aligned)
  total: Expr | None = None
  for partial, dx in zip(row.partials, aligned, strict=True):
    if dx is None:
      continue
    local = partial(expr)
    negate = isinstance(local, int) and local < 0
    term = dx if isinstance(local, int) else _seed_axis(local, expr) * dx
    total = (-term if negate else term) if total is None else total - term if negate else total + term
  assert total is not None
  return total


def cotangent(expr: Expr, cot: Expr, position: int) -> Expr:
  """The cotangent of operand ``position`` of an elementwise ``expr``: ``cot`` times the partial,
  summed over the axes the operand was broadcast along."""
  partials = ELEMENTWISE[ExprOp(expr.op)].partials
  assert partials is not None
  local = partials[position](expr)
  # A masked partial, a select's where(c, cot, 0), would apply to cot here instead of a product.
  scaled = (-cot if local < 0 else cot) if isinstance(local, int) else cot * local
  return _unbroadcast(scaled, expr.args[position].shape, expr.shape)


def _seed_axis(expr: Expr, output: Expr | None = None) -> Expr:
  missing = 0 if output is None else len(output.shape) - len(expr.shape)
  return expr.reshape((1, *(1,) * missing, *expr.shape))


def _broadcast_tangent(tangent: Expr, operand: Expr, output: Expr, nseed: int) -> Expr:
  missing = len(output.shape) - len(operand.shape)
  return tangent if missing == 0 else tangent.reshape((nseed, *(1,) * missing, *operand.shape))


def _unbroadcast(cot: Expr, in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> Expr:
  if in_shape == out_shape:
    return cot
  if not in_shape:
    return cot.sum()
  source = np.arange(int(np.prod(in_shape, dtype=int))).reshape(in_shape)
  source = np.broadcast_to(source, out_shape).reshape(-1)
  return scatter(cot, source, in_shape)
