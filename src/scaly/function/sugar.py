"""Expression builders that need a ``Function``.

``ir/expr.py`` owns the expression vocabulary and every builder that only needs a node; a builder
that has to look inside a callee belongs to the frontend instead. Today that is ``vmap``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..ir.expr import Expr, ExprOp, as_expr, common_lowering
from ..ir.types import TensorType
from .model import Function


def vmap(callee: Any, length: int, inputs: Any, output: int = 0) -> Expr:
  """Create an ``ExprOp.VMAP`` node: ``length`` independent calls of ``callee`` whose i-th argument list is
  sliced out of outer tensors with per-input ``(start, stride)`` strides.

  ``inputs`` is a sequence ordered to match ``callee.inputs``, or a mapping from callee input name
  to the same entries. An entry is usually a bare outer tensor: one of size ``length * formal.size``
  is cut into ``length`` contiguous chunks, one of size ``formal.size`` is broadcast to every
  iteration. The explicit ``(outer_tensor, start, stride)`` tuple covers overlapping or offset
  windows: the i-th iteration reads ``outer[start + i*stride : start + i*stride + formal.size]``,
  and ``stride=0`` broadcasts the same slice every iteration.

  Only the outer tensors must be rank-1. Callee formals and outputs may be rank-2 (as well as scalar
  or rank-1). Each iteration reads a flat slice of ``formal.size`` values and the produced node has
  shape ``(length * callee.outputs[output].size,)``, with iteration outputs concatenated flat.
  """
  if not isinstance(callee, Function):
    raise TypeError(f"vmap callee must be an scaly Function, got {type(callee).__name__}")
  length = int(length)
  if length < 0:
    raise ValueError(f"vmap length must be non-negative, got {length}")
  if not 0 <= output < len(callee.outputs):
    raise ValueError(f"vmap output index {output} out of range for callee with {len(callee.outputs)} outputs")
  out_expr = callee.outputs[output]

  if isinstance(inputs, Mapping):
    extra = [n for n in inputs if n not in callee.input_names]
    if extra:
      raise ValueError(f"vmap inputs reference unknown callee input names {extra}; callee accepts {list(callee.input_names)}")
    missing = [n for n in callee.input_names if n not in inputs]
    if missing:
      raise ValueError(f"vmap inputs missing entries for callee inputs {missing}")
    specs = tuple(inputs[n] for n in callee.input_names)
  else:
    specs = tuple(inputs)
    if len(specs) != len(callee.inputs):
      raise ValueError(f"vmap expects {len(callee.inputs)} input specs, got {len(specs)}")
  outers: list[Expr] = []
  starts: list[int] = []
  strides: list[int] = []
  for i, spec in enumerate(specs):
    formal = callee.inputs[i]
    if isinstance(spec, tuple):
      outer, start, stride = spec
      outer = as_expr(outer)
    else:
      outer = as_expr(spec)
      start = 0
      if outer.size == length * formal.size:
        stride = formal.size
      elif outer.size == formal.size:
        stride = 0
      else:
        raise ValueError(
          f"vmap input {callee.input_names[i]!r} has size {outer.size}; expected {length * formal.size} "
          f"({length} chunks of {formal.size}) or {formal.size} (broadcast)"
        )
    if len(outer.shape) != 1:
      raise NotImplementedError(f"vmap currently requires rank-1 outer tensors, got {outer.shape} for input {i}")
    start = int(start)
    stride = int(stride)
    if start < 0:
      raise ValueError(f"vmap input {i} start must be non-negative, got {start}")
    if stride < 0:
      raise ValueError(f"vmap input {i} stride must be non-negative, got {stride}")
    if length > 0:
      end = start + (length - 1) * stride + formal.size
      if end > outer.size:
        raise ValueError(
          f"vmap input {i} reads past outer tensor of size {outer.size}: start={start}, stride={stride}, length={length}, slice_size={formal.size}"
        )
    outers.append(outer)
    starts.append(start)
    strides.append(stride)

  diff = out_expr.type.diff and any(o.type.diff for o in outers)
  return Expr(
    ExprOp.VMAP,
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
