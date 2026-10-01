"""Prototype: a map whose body has batched forms for every op as one batched expression."""
from __future__ import annotations
import numpy as np
from scaly.ir.expr import (COMMON_ELEMENTWISE_BINARY, COMMON_ELEMENTWISE_UNARY, Expr, ExprOp, concat, matmul, stack, topo)
from scaly.ir.types import TensorType


def _typed(n: Expr, args, B: int) -> Expr:
  return Expr(n.op, tuple(args), TensorType((B, *n.shape), dtype=n.type.dtype, diff=n.type.diff), attrs=n.attrs, lowering=n.lowering)


def _matmul(n, a, ab, b, bb, B):
  if ab and bb:
    return None
  if ab:
    sa = a.shape[1:]
    if len(sa) == 1:
      return matmul(a, b)
    k, nn = sa
    out = matmul(a.reshape((B * k, nn)), b)
    return out.reshape((B, k, *b.shape[1:]))
  sb = b.shape[1:]
  at = a if len(a.shape) == 1 else a.T
  if len(sb) == 1:
    return matmul(b, at)
  nn, k = sb
  rows = b.transpose((0, 2, 1)).reshape((B * k, nn))
  out = matmul(rows, at)
  return out.reshape((B, k)) if len(a.shape) == 1 else out.reshape((B, k, a.shape[0])).transpose((0, 2, 1))


def batched_map(node: Expr) -> Expr | None:
  callee = node.attrs["callee"]
  B, starts, strides = int(node.attrs["length"]), node.attrs["starts"], node.attrs["strides"]
  env: dict[int, tuple[Expr, bool]] = {}
  for k, inp in enumerate(callee.inputs):
    flat = node.args[k].reshape((node.args[k].size,))
    if strides[k] == 0:
      env[inp.id] = (flat[starts[k] : starts[k] + inp.size].reshape(inp.shape), False)
    elif strides[k] == inp.size:
      env[inp.id] = (flat[starts[k] : starts[k] + B * inp.size].reshape((B, *inp.shape)), True)
    else:
      return None
  for n in topo(callee.outputs):
    if n.id in env:
      continue
    args = [env[a.id] for a in n.args]
    if not any(b for _, b in args):
      same = all(x is a for (x, _), a in zip(args, n.args))
      env[n.id] = (n if same else Expr(n.op, tuple(x for x, _ in args), n.type, name=n.name, value=n.value, attrs=n.attrs, lowering=n.lowering), False)
      continue
    out = None
    if n.op in COMMON_ELEMENTWISE_UNARY:
      out = _typed(n, [args[0][0]], B)
    elif n.op in COMMON_ELEMENTWISE_BINARY:
      rank = len(n.shape)
      ops = [x.reshape((B, *([1] * (rank - (len(x.shape) - 1))), *x.shape[1:])) if b else x for x, b in args]
      out = _typed(n, ops, B)
    elif n.op == ExprOp.MATMUL:
      out = _matmul(n, args[0][0], args[0][1], args[1][0], args[1][1], B)
    elif n.op == ExprOp.RESHAPE:
      out = args[0][0].reshape((B, *n.shape))
    elif n.op == ExprOp.TRANSPOSE:
      axes = n.attrs.get("axes") or tuple(reversed(range(len(n.args[0].shape))))
      out = args[0][0].transpose((0, *(a + 1 for a in axes)))
    elif n.op in (ExprOp.STACK, ExprOp.CONCAT):
      if not all(b for _, b in args):
        return None
      axis = int(n.attrs.get("axis", 0)) % len(n.shape)
      out = (stack if n.op == ExprOp.STACK else concat)([x for x, _ in args], axis=axis + 1)
    if out is None:
      print("no rule:", n.op, [a.shape for a in n.args])
      return None
    assert out.shape == (B, *n.shape), (n.op, out.shape, n.shape)
    env[n.id] = (out, True)
  out, batched = env[callee.outputs[int(node.attrs["output"])].id]
  return out.reshape(node.shape) if batched else None
