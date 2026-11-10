"""Reverse-mode AD: ``vjp`` and the transpose rules of the structural ops."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..ir.expr import Expr, ExprOp, _uniform, from_dense, gather, pack, scatter, topo, to_dense, values, zeros_like
from ..ir.support import slice_indices, support_indices
from .calls import _call_vjp, _vmap_vjp
from .rules import ELEMENTWISE, accumulate, carry_mask, cotangent
from .sparsity import _check_wrt, _depends_on, _prunable_support


def vjp(outputs: Sequence[Expr], wrts: Sequence[Expr], cotangents: Sequence[Expr]) -> tuple[Expr, ...]:
  """Reverse-mode derivative: one adjoint per entry of ``wrts``, seeded by ``cotangents``.

  Each cotangent is shaped like its output. One sweep computes the derivative of a scalar with
  respect to every requested input at once, which is why gradients go through here.
  Every entry of ``wrts`` must be an independent input from ``sc.sym`` or a Function declaration.
  Results follow ``wrts`` order.
  """
  if len(outputs) != len(cotangents):
    raise ValueError(f"expected {len(outputs)} cotangents, got {len(cotangents)}")
  if not all(out.type.dtype.is_floating for out in outputs):
    raise TypeError("integer and boolean values have no tangent")
  for wrt in wrts:
    _check_wrt(wrt)
  adjoints: dict[int, Expr] = {}
  nodes = topo(outputs)
  expr_ids = {e.id for e in nodes}
  dep_memo: dict[tuple[int, int], bool] = {}

  def needed(expr: Expr) -> bool:
    return any(_depends_on(expr, wrt, dep_memo, through_stops=False) for wrt in wrts)

  for out, cot in zip(outputs, cotangents, strict=True):
    if out.shape != cot.shape:
      raise ValueError(f"cotangent for output shape {out.shape} has shape {cot.shape}")
    if out.type.pattern is not None and cot.type.pattern is not out.type.pattern:
      cot = from_dense(to_dense(cot), out.type.pattern)
    adjoints[out.id] = cot if out.id not in adjoints else adjoints[out.id] + cot

  for expr in reversed(nodes):
    cot = adjoints.get(expr.id)
    if cot is None or expr.op in {ExprOp.INPUT, ExprOp.CONST, ExprOp.STOP_GRADIENT} or not needed(expr):
      continue
    if expr.op == ExprOp.VMAP:
      for arg, arg_cot in _vmap_vjp(expr, cot, wrts, dep_memo, pullback=vjp):
        if arg.id in expr_ids:
          adjoints[arg.id] = arg_cot if arg.id not in adjoints else adjoints[arg.id] + arg_cot
      continue
    active = (
      tuple(i for i, arg in enumerate(expr.args) if arg.type.dtype.is_floating and needed(arg)) if expr.op in {ExprOp.CALL, *ELEMENTWISE} else None
    )
    transpose = lambda c: _local_vjp(expr, c, active)  # noqa: E731
    arg_cots = carry_mask(cot, transpose) if expr.op in _SELECTIONS else transpose(cot)
    for arg, arg_cot in zip(expr.args, arg_cots, strict=True):
      if arg_cot is not None and arg.id in expr_ids:
        if arg.type.pattern is not None and arg_cot.type.pattern is not arg.type.pattern:
          arg_cot = from_dense(to_dense(arg_cot), arg.type.pattern)
        adjoints[arg.id] = arg_cot if arg.id not in adjoints else accumulate(adjoints[arg.id], arg_cot)

  return tuple(adjoints.get(wrt.id, zeros_like(wrt)) for wrt in wrts)


# Structural ops whose transposes weigh each entry by 0 or 1, so a select's mask passes through them.
_SELECTIONS = {
  ExprOp.SUM,
  ExprOp.RESHAPE,
  ExprOp.TRANSPOSE,
  ExprOp.SLICE,
  ExprOp.GATHER,
  ExprOp.SCATTER,
  ExprOp.STACK,
  ExprOp.CONCAT,
}


def _local_vjp(expr: Expr, cot: Expr, active: tuple[int, ...] | None = None) -> tuple[Expr | None, ...]:
  args = expr.args
  if expr.op == ExprOp.PACK:
    return (values(cot),)
  if expr.op == ExprOp.VALUES:
    assert args[0].type.pattern is not None
    return (pack(cot, args[0].type.pattern, shape=args[0].shape),)
  if expr.op in ELEMENTWISE:
    if ELEMENTWISE[expr.op].partials is None:
      raise NotImplementedError(f"VJP for nonsmooth op {expr.op!r} is not implemented")
    assert active is not None
    return tuple(cotangent(expr, cot, i) if i in active else None for i in range(len(args)))
  if expr.op == ExprOp.SUM:
    return (cot * _ones_like(args[0]),)
  if expr.op == ExprOp.RESHAPE:
    return (cot.reshape(args[0].shape),)
  if expr.op == ExprOp.PRINT:
    return (cot, *(zeros_like(arg) if arg.type.dtype.is_floating else None for arg in args[1:]))
  if expr.op == ExprOp.TRANSPOSE:
    axes = expr.attrs["axes"]
    inv = tuple(int(np.argsort(axes)[i]) for i in range(len(axes)))
    return (cot.transpose(inv),)
  if expr.op == ExprOp.SLICE:
    support = _prunable_support(cot)
    entries = support_indices(support) if support is not None and len(support) < cot.size else None
    indices = slice_indices(args[0].shape, expr.attrs["index"], entries)
    selected = gather(cot, entries) if entries is not None else cot
    return (scatter(selected, indices, args[0].shape),)
  if expr.op == ExprOp.GATHER:
    return (_gather_vjp(cot, expr.attrs["indices"], args[0].shape),)
  if expr.op == ExprOp.SCATTER:
    return (gather(cot, expr.attrs["indices"]),)
  if expr.op == ExprOp.STACK:
    return _stack_vjp(cot, len(args), expr.attrs.get("axis", 0))
  if expr.op == ExprOp.CONCAT:
    return _concat_vjp(cot, args, expr.attrs.get("axis", 0))
  if expr.op == ExprOp.MATMUL:
    if any(arg.type.pattern is not None for arg in args):
      return _matmul_vjp(to_dense(args[0]), to_dense(args[1]), to_dense(cot))
    return _matmul_vjp(args[0], args[1], cot)
  if expr.op == ExprOp.CALL:
    return _call_vjp(expr, cot, active, pullback=vjp)
  if expr.op == ExprOp.EXTERN:
    raise NotImplementedError(f"active derivative through extern {expr.attrs['callee'].name!r} needs sc.custom_derivative on a wrapping Function")
  if expr.op == ExprOp.SOLVER_CALL:
    if cot.op == ExprOp.CONST and cot.value is not None and not np.any(cot.value):
      return tuple(zeros_like(arg) for arg in args)
    raise NotImplementedError("active derivative through SOLVER_CALL is not implemented")
  raise NotImplementedError(f"VJP for op {expr.op!r} is not implemented")


def _ones_like(expr: Expr) -> Expr:
  if expr.type.pattern is not None:
    return pack(_uniform(1, expr.type.storage_shape, dtype=expr.type.dtype, lowering=expr.lowering), expr.type.pattern, shape=expr.shape)
  return _uniform(1, expr.shape, dtype=expr.type.dtype, lowering=expr.lowering)


def _gather_vjp(cot: Expr, indices: np.ndarray, shape: tuple[int, ...]) -> Expr:
  return scatter(cot, indices, shape)


def _stack_vjp(cot: Expr, nargs: int, axis: int) -> tuple[Expr, ...]:
  ret: list[Expr] = []
  for i in range(nargs):
    index = tuple(i if dim == axis else slice(None) for dim in range(len(cot.shape)))
    ret.append(cot[index])
  return tuple(ret)


def _concat_vjp(cot: Expr, args: Sequence[Expr], axis: int) -> tuple[Expr, ...]:
  ret: list[Expr] = []
  start = 0
  for arg in args:
    index: list[Any] = [slice(None)] * len(cot.shape)
    index[axis] = slice(start, start + arg.shape[axis])
    ret.append(cot[tuple(index)])
    start += arg.shape[axis]
  return tuple(ret)


def _matmul_vjp(x: Expr, y: Expr, cot: Expr) -> tuple[Expr, Expr]:
  if len(x.shape) == 1 and len(y.shape) == 1:
    return cot * y, cot * x
  if len(x.shape) == 2 and len(y.shape) == 1:
    return cot.reshape((x.shape[0], 1)) @ y.reshape((1, x.shape[1])), x.T @ cot
  if len(x.shape) == 1 and len(y.shape) == 2:
    return y @ cot, x.reshape((x.shape[0], 1)) @ cot.reshape((1, y.shape[1]))
  if len(x.shape) == 2 and len(y.shape) == 2:
    return cot @ y.T, x.T @ cot
  raise NotImplementedError(f"matmul VJP for {x.shape} @ {y.shape} is not implemented")
