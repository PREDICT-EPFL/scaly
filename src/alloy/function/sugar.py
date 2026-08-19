"""Expression builders that need a ``Function``.

``ir/expr.py`` owns the expression vocabulary and every builder that only needs a node; a builder
that has to look inside a callee belongs to the frontend instead. Today that is ``map_``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..ir.expr import Expr, ExprOp, as_expr, common_lowering
from ..ir.types import TensorType
from .model import Function


def map_(callee: Any, length: int, inputs: Any, output: int = 0) -> Expr:
  """Create an ``ExprOp.MAP`` node: ``length`` independent calls of ``callee`` whose i-th argument list is
  sliced out of outer tensors with per-input ``(start, stride)`` strides.

  ``inputs`` is either a sequence of ``(outer_tensor, start, stride)`` tuples ordered to match
  ``callee.inputs``, or a mapping from callee input name to the same tuple. The i-th iteration reads
  ``outer[start + i*stride : start + i*stride + callee.inputs[k].size]`` for callee input ``k``.
  Iterations are independent: ``stride=0`` broadcasts the same slice every iteration.

  Only the outer tensors must be rank-1. Callee formals and outputs may be rank-2 (as well as scalar
  or rank-1); each iteration reads a flat slice of ``formal.size`` values and the produced node has
  shape ``(length * callee.outputs[output].size,)``, with iteration outputs concatenated flat.
  """
  if not isinstance(callee, Function):
    raise TypeError(f"map callee must be an alloy Function, got {type(callee).__name__}")
  length = int(length)
  if length < 0:
    raise ValueError(f"map length must be non-negative, got {length}")
  if not 0 <= output < len(callee.outputs):
    raise ValueError(f"map output index {output} out of range for callee with {len(callee.outputs)} outputs")
  out_expr = callee.outputs[output]

  if isinstance(inputs, Mapping):
    extra = [n for n in inputs if n not in callee.input_names]
    if extra:
      raise ValueError(f"map inputs reference unknown callee input names {extra}; callee accepts {list(callee.input_names)}")
    missing = [n for n in callee.input_names if n not in inputs]
    if missing:
      raise ValueError(f"map inputs missing entries for callee inputs {missing}")
    specs = tuple(inputs[n] for n in callee.input_names)
  else:
    specs = tuple(inputs)
    if len(specs) != len(callee.inputs):
      raise ValueError(f"map expects {len(callee.inputs)} input specs, got {len(specs)}")
  outers: list[Expr] = []
  starts: list[int] = []
  strides: list[int] = []
  for i, spec in enumerate(specs):
    outer, start, stride = spec
    outer = as_expr(outer)
    formal = callee.inputs[i]
    if len(outer.shape) != 1:
      raise NotImplementedError(f"map currently requires rank-1 outer tensors, got {outer.shape} for input {i}")
    start = int(start)
    stride = int(stride)
    if start < 0:
      raise ValueError(f"map input {i} start must be non-negative, got {start}")
    if stride < 0:
      raise ValueError(f"map input {i} stride must be non-negative, got {stride}")
    if length > 0:
      end = start + (length - 1) * stride + formal.size
      if end > outer.size:
        raise ValueError(
          f"map input {i} reads past outer tensor of size {outer.size}: start={start}, stride={stride}, length={length}, slice_size={formal.size}"
        )
    outers.append(outer)
    starts.append(start)
    strides.append(stride)

  diff = out_expr.type.diff and any(o.type.diff for o in outers)
  return Expr(
    ExprOp.MAP,
    tuple(outers),
    TensorType((length * out_expr.size,), out_expr.type.dtype, diff=diff),
    attrs={
      "callee": callee,
      "output": int(output),
      "length": length,
      "starts": tuple(starts),
      "strides": tuple(strides),
      "slice_size": int(out_expr.size),
    },
    lowering=common_lowering(*outers) if outers else "auto",
  )
