"""Expression builders that need a ``Function``.

``ir/expr.py`` owns the expression vocabulary and every builder that only needs a node; a builder
that has to look inside a callee belongs to the frontend instead: ``vmap`` and ``scan``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..ir.expr import Expr, ExprOp, as_expr, common_lowering
from ..ir.types import TensorType
from .model import Function


def vmap(callee: Any, length: int, inputs: Any, output: int = 0) -> Expr:
  """Create an ``ExprOp.VMAP`` node: ``length`` independent calls of ``callee`` whose i-th argument list is
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
    outer, start, stride = spec
    outer = as_expr(outer)
    formal = callee.inputs[i]
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


def scan(body: Any, init: Any, xs: Sequence[tuple[Any, int, int]] = (), *, length: int) -> tuple[Expr, ...]:
  """Run ``body`` ``length`` times in sequence, threading a carry: a loop in the generated C, not an unrolling.

  ``body`` is a ``Function`` whose first input is the carry and whose first output is the next
  carry, with the same shape and dtype. Its other inputs are sliced from ``xs`` exactly as ``vmap``
  slices: step ``k`` reads ``outer[start + k*stride : start + k*stride + formal.size]``, and a
  ``stride`` of zero passes the same slice at every step. Its other outputs are stacked flat, one
  slice per step. Returns ``(final_carry, *ys)``; ``final_carry`` is ``init`` when ``length`` is zero.

  The number of steps is fixed when the graph is built, which is what makes the code size and the
  derivative's workspace (reverse mode stores the carry at every step) known ahead of time.
  """
  if not isinstance(body, Function):
    raise TypeError(f"scan body must be an scaly Function, got {type(body).__name__}")
  if not body.inputs or not body.outputs:
    raise ValueError("scan body needs the carry as its first input and the next carry as its first output")
  init = as_expr(init)
  carry, nxt = body.inputs[0], body.outputs[0]
  if nxt.shape != carry.shape or nxt.type.dtype != carry.type.dtype:
    raise ValueError(f"scan body must return a carry like its input: {carry.type.dtype}{carry.shape} -> {nxt.type.dtype}{nxt.shape}")
  if init.shape != carry.shape or init.type.dtype != carry.type.dtype:
    raise ValueError(f"scan init {init.type.dtype}{init.shape} does not match the carry {carry.type.dtype}{carry.shape}")
  specs = tuple(xs)
  if len(specs) != len(body.inputs) - 1:
    raise ValueError(f"scan body takes {len(body.inputs) - 1} sliced inputs after the carry, got {len(specs)}")
  length = int(length)
  if length < 0:
    raise ValueError(f"scan length must be non-negative, got {length}")
  outers = tuple(as_expr(outer) for outer, _, _ in specs)
  starts = tuple(int(start) for _, start, _ in specs)
  strides = tuple(int(stride) for _, _, stride in specs)
  return tuple(_scan_node(body, init, outers, starts, strides, length, k) for k in range(len(body.outputs)))


def _scan_node(
  body: Function, init: Expr, outers: tuple[Expr, ...], starts: tuple[int, ...], strides: tuple[int, ...], length: int, output: int
) -> Expr:
  """One output of a scan: the final carry (0), a stacked output (1..), or with ``output=-1`` the carry
  entering every step, stacked, which reverse mode reads backwards. A negative stride walks backwards."""
  for i, (outer, start, stride) in enumerate(zip(outers, starts, strides, strict=True)):
    formal = body.inputs[i + 1]
    if len(outer.shape) != 1:
      raise NotImplementedError(f"scan requires rank-1 outer tensors, got {outer.shape} for input {i + 1}")
    last = start + (length - 1) * stride
    if length and (min(start, last) < 0 or max(start, last) + formal.size > outer.size):
      raise ValueError(f"scan input {i + 1} reads outside its outer tensor of size {outer.size}: start={start}, stride={stride}, length={length}")
  if output == 0:
    shape: tuple[int, ...] = body.inputs[0].shape
    produced = body.outputs[0]
  elif output == -1:
    shape, produced = (length * body.inputs[0].size,), body.outputs[0]
  else:
    produced = body.outputs[output]
    shape = (length * produced.size,)
  diff = produced.type.diff and (init.type.diff or any(o.type.diff for o in outers))
  args = (init, *outers)
  return Expr(
    ExprOp.SCAN,
    args,
    TensorType(shape, produced.type.dtype, diff=diff),
    attrs={"callee": body, "output": int(output), "length": int(length), "starts": starts, "strides": strides},
    lowering=common_lowering(*args),
  )
