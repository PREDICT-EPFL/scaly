from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

import numpy as np


class Ops(StrEnum):
  INPUT = "input"
  CONST = "const"
  NEG = "neg"
  SIN = "sin"
  COS = "cos"
  TAN = "tan"
  ASIN = "asin"
  ACOS = "acos"
  ATAN = "atan"
  SINH = "sinh"
  COSH = "cosh"
  TANH = "tanh"
  EXP = "exp"
  LOG = "log"
  SQRT = "sqrt"
  ABS = "abs"
  FLOOR = "floor"
  CEIL = "ceil"
  ADD = "add"
  SUB = "sub"
  MUL = "mul"
  DIV = "div"
  POW = "pow"
  ATAN2 = "atan2"
  MINIMUM = "minimum"
  MAXIMUM = "maximum"
  SUM = "sum"
  RESHAPE = "reshape"
  TRANSPOSE = "transpose"
  SLICE = "slice"
  GATHER = "gather"
  SCATTER = "scatter"
  STACK = "stack"
  CONCAT = "concat"
  MATMUL = "matmul"
  CALL = "call"
  MAP = "map"
  SOLVER_CALL = "solver_call"


COMMON_ELEMENTWISE_UNARY = {
  Ops.NEG,
  Ops.SIN,
  Ops.COS,
  Ops.TAN,
  Ops.ASIN,
  Ops.ACOS,
  Ops.ATAN,
  Ops.SINH,
  Ops.COSH,
  Ops.TANH,
  Ops.EXP,
  Ops.LOG,
  Ops.SQRT,
  Ops.ABS,
  Ops.FLOOR,
  Ops.CEIL,
}

COMMON_ELEMENTWISE_BINARY = {
  Ops.ADD,
  Ops.SUB,
  Ops.MUL,
  Ops.DIV,
  Ops.POW,
  Ops.ATAN2,
  Ops.MINIMUM,
  Ops.MAXIMUM,
}

COMMON_STRUCTURAL = {
  Ops.INPUT,
  Ops.CONST,
  Ops.SUM,
  Ops.RESHAPE,
  Ops.TRANSPOSE,
  Ops.SLICE,
  Ops.GATHER,
  Ops.SCATTER,
  Ops.STACK,
  Ops.CONCAT,
  Ops.MATMUL,
  Ops.CALL,
  Ops.MAP,
  Ops.SOLVER_CALL,
}

# Deliberately not in the MVP set: expm1/log1p (nice but low priority), splines/interpolants
# (important but require carefully specified extrapolation, knots, derivatives, and codegen tables),
# matrix exponentials/decompositions, and solver/control-flow ops.
COMMON_OPS = COMMON_STRUCTURAL | COMMON_ELEMENTWISE_UNARY | COMMON_ELEMENTWISE_BINARY


@dataclass(frozen=True, slots=True)
class OpInfo:
  op: Ops
  arity: int | None
  numpy: Callable[..., np.ndarray | np.generic] | None = None
  differentiable: bool = True

  @property
  def name(self) -> str:
    return self.op.value


OP_INFO: dict[Ops, OpInfo] = {
  Ops.INPUT: OpInfo(Ops.INPUT, 0, None),
  Ops.CONST: OpInfo(Ops.CONST, 0, None, False),
  Ops.NEG: OpInfo(Ops.NEG, 1, np.negative),
  Ops.SIN: OpInfo(Ops.SIN, 1, np.sin),
  Ops.COS: OpInfo(Ops.COS, 1, np.cos),
  Ops.TAN: OpInfo(Ops.TAN, 1, np.tan),
  Ops.ASIN: OpInfo(Ops.ASIN, 1, np.arcsin),
  Ops.ACOS: OpInfo(Ops.ACOS, 1, np.arccos),
  Ops.ATAN: OpInfo(Ops.ATAN, 1, np.arctan),
  Ops.SINH: OpInfo(Ops.SINH, 1, np.sinh),
  Ops.COSH: OpInfo(Ops.COSH, 1, np.cosh),
  Ops.TANH: OpInfo(Ops.TANH, 1, np.tanh),
  Ops.EXP: OpInfo(Ops.EXP, 1, np.exp),
  Ops.LOG: OpInfo(Ops.LOG, 1, np.log),
  Ops.SQRT: OpInfo(Ops.SQRT, 1, np.sqrt),
  Ops.ABS: OpInfo(Ops.ABS, 1, np.abs),
  Ops.FLOOR: OpInfo(Ops.FLOOR, 1, np.floor, False),
  Ops.CEIL: OpInfo(Ops.CEIL, 1, np.ceil, False),
  Ops.ADD: OpInfo(Ops.ADD, 2, np.add),
  Ops.SUB: OpInfo(Ops.SUB, 2, np.subtract),
  Ops.MUL: OpInfo(Ops.MUL, 2, np.multiply),
  Ops.DIV: OpInfo(Ops.DIV, 2, np.divide),
  Ops.POW: OpInfo(Ops.POW, 2, np.power),
  Ops.ATAN2: OpInfo(Ops.ATAN2, 2, np.arctan2),
  Ops.MINIMUM: OpInfo(Ops.MINIMUM, 2, np.minimum, False),
  Ops.MAXIMUM: OpInfo(Ops.MAXIMUM, 2, np.maximum, False),
  Ops.SUM: OpInfo(Ops.SUM, 1, np.sum),
  Ops.RESHAPE: OpInfo(Ops.RESHAPE, 1, np.reshape),
  Ops.TRANSPOSE: OpInfo(Ops.TRANSPOSE, 1, np.transpose),
  Ops.SLICE: OpInfo(Ops.SLICE, 1, None),
  Ops.GATHER: OpInfo(Ops.GATHER, 1, None),
  Ops.SCATTER: OpInfo(Ops.SCATTER, 1, None),
  Ops.STACK: OpInfo(Ops.STACK, None, np.stack),
  Ops.CONCAT: OpInfo(Ops.CONCAT, None, np.concatenate),
  Ops.MATMUL: OpInfo(Ops.MATMUL, 2, np.matmul),
  Ops.CALL: OpInfo(Ops.CALL, None, None),
  Ops.MAP: OpInfo(Ops.MAP, None, None),
  Ops.SOLVER_CALL: OpInfo(Ops.SOLVER_CALL, None, None, differentiable=False),
}
