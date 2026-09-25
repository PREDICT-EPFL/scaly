"""Arithmetic identities and scalar constant evaluation shared by both dialects, over one small adapter."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Any, Callable, Protocol, cast

import numpy as np

from ..ir import program as p
from ..ir.expr import Expr, ExprOp
from ..ir.program import ProgramNode, ProgramOp
from ..ir.types import DType

_KINDS = frozenset({"add", "sub", "mul", "div", "neg", "pow"})


class _HasArgs(Protocol):
  @property
  def args(self) -> tuple[Any, ...]: ...


@dataclass(frozen=True, slots=True)
class Arith[Node: _HasArgs]:
  """What a rule may ask of a node: its arithmetic kind, a uniform constant's value, and how to build results.

  ``fits(arg, result)`` says whether ``arg`` may stand in for ``result`` (same shape and dtype).
  ``scalar`` makes a coefficient of the result's dtype; ``full`` makes a constant that takes the
  result's place, so tensor and scalar constants materialize differently under one policy.
  """

  kind: Callable[[Node], str | None]
  const: Callable[[Node], int | float | None]
  fits: Callable[[Node, Node], bool]
  scalar: Callable[[int | float, Node], Node]
  full: Callable[[int | float, Node], Node]
  build: Callable[[str, tuple[Node, ...], Node], Node]


def simplify_arith[Node: _HasArgs](d: Arith[Node], node: Node) -> Node | None:
  """One rewrite step of the shared identities on ``node``, or None when none applies."""
  kind = d.kind(node)
  if kind is None:
    return None
  x = node.args[0]
  if kind == "neg":
    return x.args[0] if d.kind(x) == "neg" else None
  y = node.args[1]
  cx, cy = d.const(x), d.const(y)
  if kind == "add":
    if x is y:
      return d.build("mul", (d.scalar(2, node), x), node)
    if cx == 0 and d.fits(y, node):
      return y
    if cy == 0 and d.fits(x, node):
      return x
    if d.kind(y) == "neg":
      return d.build("sub", (x, y.args[0]), node)
    if d.kind(x) == "neg":
      return d.build("sub", (y, x.args[0]), node)
  elif kind == "sub":
    if x is y:
      return d.full(0, node)
    if cy == 0 and d.fits(x, node):
      return x
    if cx == 0 and d.fits(y, node):
      return d.build("neg", (y,), node)
    if d.kind(y) == "neg":
      return d.build("add", (x, y.args[0]), node)
  elif kind == "mul":
    if cx == 0 or cy == 0:
      return d.full(0, node)
    if cx == 1 and d.fits(y, node):
      return y
    if cy == 1 and d.fits(x, node):
      return x
    if cx == -1 and d.fits(y, node):
      return d.build("neg", (y,), node)
    if cy == -1 and d.fits(x, node):
      return d.build("neg", (x,), node)
    if d.kind(x) == d.kind(y) == "neg":
      return d.build("mul", (x.args[0], y.args[0]), node)
    if d.kind(x) == "neg":
      return d.build("neg", (d.build("mul", (x.args[0], y), node),), node)
    if d.kind(y) == "neg":
      return d.build("neg", (d.build("mul", (x, y.args[0]), node),), node)
  elif kind == "div":
    if x is y:
      return d.full(1, node)
    if cx == 0:
      return d.full(0, node)
    if cy == 1 and d.fits(x, node):
      return x
    if d.kind(x) == "neg":
      return d.build("neg", (d.build("div", (x.args[0], y), node),), node)
    if d.kind(y) == "neg":
      return d.build("neg", (d.build("div", (x, y.args[0]), node),), node)
  elif kind == "pow":
    if cy == 0 or cx == 1:
      return d.full(1, node)
    if cy == 1 and d.fits(x, node):
      return x
    if cy == 2 and d.fits(x, node):
      return d.build("mul", (x, x), node)
  return None


def fold[Node: _HasArgs](d: Arith[Node], node: Node) -> Node:
  """Apply the shared identities at ``node`` until none fires. Operands are assumed already folded."""
  while (new := simplify_arith(d, node)) is not None:
    node = new
  return node


# --- expression dialect -------------------------------------------------------


def _expr_const(e: Expr) -> int | float | None:
  if e.op != ExprOp.CONST or e.value is None or e.value.size == 0:
    return None
  first = e.value.reshape(-1)[0]
  return first.item() if bool(np.all(e.value == first)) else None


def _expr_full(value: int | float, like: Expr) -> Expr:
  return Expr.const(np.full(like.shape, value), dtype=like.type.dtype, lowering=like.lowering)


ARITH_EXPR: Arith[Expr] = Arith(
  kind=lambda e: str(e.op) if str(e.op) in _KINDS else None,
  const=_expr_const,
  fits=lambda arg, result: arg.shape == result.shape and arg.type.dtype == result.type.dtype,
  scalar=lambda value, like: Expr.const(value, dtype=like.type.dtype, lowering=like.lowering),
  full=_expr_full,
  build=lambda kind, args, like: Expr(ExprOp(kind), args, like.type, lowering=like.lowering),
)


# --- program dialect ----------------------------------------------------------

CONSTANTS = frozenset({ProgramOp.CONST_INT, ProgramOp.CONST_FLOAT})
_ARITHMETIC: dict[ProgramOp, Callable[..., Any]] = {
  ProgramOp.ADD: operator.add,
  ProgramOp.SUB: operator.sub,
  ProgramOp.MUL: operator.mul,
  ProgramOp.DIV: operator.truediv,
}
_MATH = {op: getattr(math, op.value) for op in (*p.UNARY_FN_OPS, ProgramOp.POW, ProgramOp.ATAN2, ProgramOp.COPYSIGN) if op != ProgramOp.ABS}
_MATH[ProgramOp.ABS] = abs
_PREDICATES: dict[ProgramOp, Callable[..., bool]] = {
  ProgramOp.LT: operator.lt,
  ProgramOp.LE: operator.le,
  ProgramOp.EQ: operator.eq,
  ProgramOp.NE: operator.ne,
  ProgramOp.AND: lambda x, y: bool(x) and bool(y),
  ProgramOp.OR: lambda x, y: bool(x) or bool(y),
  ProgramOp.NOT: lambda x: not x,
  ProgramOp.ISFINITE: math.isfinite,
}


def constant(value: int | float, dtype: DType) -> ProgramNode:
  """A program constant of ``dtype``, truncating to int for integer dtypes; a bool is a 0/1 ``CONST_INT``."""
  if dtype.is_bool:
    return ProgramNode(ProgramOp.CONST_INT, attrs={"value": int(bool(value))}, dtype=dtype)
  return ProgramNode(
    ProgramOp.CONST_INT if dtype.is_integer else ProgramOp.CONST_FLOAT, attrs={"value": int(value) if dtype.is_integer else float(value)}, dtype=dtype
  )


def evaluate(op: ProgramOp, values: list[int | float], dtype: DType) -> int | float | None:
  """Evaluate a scalar operation on constants with ``dtype`` semantics, or None when the result is not
  faithfully representable (division by zero, domain errors, overflow): those stay runtime operations."""
  value = _evaluate_scalar(op, values, dtype)
  if value is not None and dtype.is_integer and not -(1 << (dtype.bits - 1)) <= value < (1 << (dtype.bits - 1)):
    return None
  return value


def _evaluate_scalar(op: ProgramOp, values: list[int | float], dtype: DType) -> int | float | None:
  try:
    if op in {ProgramOp.DIV, ProgramOp.MOD} and dtype.is_integer:
      x, y = values
      q = (abs(x) // abs(y)) * (-1 if (x < 0) != (y < 0) else 1)
      return q if op == ProgramOp.DIV else x - q * y
    if op == ProgramOp.NEG:
      return -values[0]
    if op in _PREDICATES:
      return int(_PREDICATES[op](*values))
    if op == ProgramOp.SELECT:
      return values[1] if values[0] else values[2]
    if op == ProgramOp.CAST:
      return int(values[0]) if dtype.is_integer else float(values[0])
    if op in _ARITHMETIC:
      return cast(int | float, _ARITHMETIC[op](*values))
    if op in _MATH:
      return _MATH[op](*values)
    if op in {ProgramOp.MINIMUM, ProgramOp.MAXIMUM} and all(math.isfinite(v) for v in values):
      return (min if op == ProgramOp.MINIMUM else max)(values)
  except (ValueError, OverflowError, ZeroDivisionError):
    return None
  return None


ARITH_PROGRAM: Arith[ProgramNode] = Arith(
  kind=lambda n: str(n.op) if str(n.op) in _KINDS else None,
  const=lambda n: n.attrs["value"] if n.op in CONSTANTS else None,
  fits=lambda arg, result: arg.dtype == result.dtype,
  scalar=lambda value, like: constant(value, like.dtype),
  full=lambda value, like: constant(value, like.dtype),
  build=lambda kind, args, like: ProgramNode(ProgramOp(kind), args, dtype=like.dtype),
)


def fold_program(node: ProgramNode) -> ProgramNode:
  """Constant-evaluate ``node`` when every operand is a constant, else apply the shared identities.

  An all-constant node that evaluation refuses stays as it is: ``0 / 0`` is a known invalid
  operation, not a case of ``0 / x``.
  """
  if node.args and all(a.op in CONSTANTS for a in node.args):
    value = evaluate(node.op, [a.attrs["value"] for a in node.args], node.dtype)
    return node if value is None else constant(value, node.dtype)
  if node.op == ProgramOp.SELECT:
    cond, x, y = node.args
    if cond.op in CONSTANTS:
      return x if cond.attrs["value"] else y
    return x if x is y else node
  if node.op == ProgramOp.NOT and node.args[0].op == ProgramOp.NOT:
    return node.args[0].args[0]
  return fold(ARITH_PROGRAM, node)
