"""Reverse-mode AD: ``vjp`` and the transpose rules of the structural ops."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..ir.expr import Expr, ExprOp, _independent, substitute, gather, scatter, topo, zeros_like
from .calls import _call_vjp, _vmap_vjp
from .rules import ELEMENTWISE, cotangent
from .sparsity import _depends_on


def vjp(outputs: Sequence[Expr], wrts: Sequence[Expr], cotangents: Sequence[Expr]) -> tuple[Expr, ...]:
  """Reverse-mode derivative: one adjoint per entry of ``wrts``, seeded by ``cotangents``.

  Each cotangent is shaped like its output. One sweep computes the derivative of a scalar with
  respect to every selected expression at once, which is why gradients go through here.
  Intermediate selections act as independent inputs. Calculations that use their inputs without
  using a selected expression stay fixed. When one selection uses another, differentiating a use
  of the former does not also contribute to the derivative of the latter. Results follow ``wrts`` order.
  """
  if len(outputs) != len(cotangents):
    raise ValueError(f"expected {len(outputs)} cotangents, got {len(cotangents)}")
  outputs, wrts, restore = _independent(outputs, wrts)
  adjoints: dict[int, Expr] = {}
  nodes = topo(outputs)
  expr_ids = {e.id for e in nodes}
  dep_memo: dict[tuple[int, int], bool] = {}

  def needed(expr: Expr) -> bool:
    return any(_depends_on(expr, wrt, dep_memo, through_stops=False) for wrt in wrts)

  for out, cot in zip(outputs, cotangents, strict=True):
    if out.shape != cot.shape:
      raise ValueError(f"cotangent for output shape {out.shape} has shape {cot.shape}")
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
    active = tuple(i for i, arg in enumerate(expr.args) if needed(arg)) if expr.op in {ExprOp.CALL, *ELEMENTWISE} else None
    for arg, arg_cot in zip(expr.args, _local_vjp(expr, cot, active), strict=True):
      if arg_cot is not None and arg.id in expr_ids:
        adjoints[arg.id] = arg_cot if arg.id not in adjoints else adjoints[arg.id] + arg_cot

  result = tuple(adjoints.get(wrt.id, zeros_like(wrt)) for wrt in wrts)
  return tuple(substitute(expr, restore) for expr in result) if restore else result


def _local_vjp(expr: Expr, cot: Expr, active: tuple[int, ...] | None = None) -> tuple[Expr | None, ...]:
  args = expr.args
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
    return (cot, *(zeros_like(arg) for arg in args[1:]))
  if expr.op == ExprOp.TRANSPOSE:
    axes = expr.attrs["axes"]
    inv = tuple(int(np.argsort(axes)[i]) for i in range(len(axes)))
    return (cot.transpose(inv),)
  if expr.op == ExprOp.SLICE:
    indices = np.arange(args[0].size).reshape(args[0].shape)[expr.attrs["index"]]
    return (scatter(cot, indices, args[0].shape),)
  if expr.op == ExprOp.GATHER:
    return (_gather_vjp(cot, expr.attrs["indices"], args[0].shape),)
  if expr.op == ExprOp.SCATTER:
    return (gather(cot, expr.attrs["indices"]),)
  if expr.op == ExprOp.STACK:
    return _stack_vjp(cot, len(args), expr.attrs.get("axis", 0))
  if expr.op == ExprOp.CONCAT:
    return _concat_vjp(cot, args, expr.attrs.get("axis", 0))
  if expr.op == ExprOp.MATMUL:
    return _matmul_vjp(args[0], args[1], cot)
  if expr.op == ExprOp.CALL:
    return _call_vjp(expr, cot, active, pullback=vjp)
  if expr.op == ExprOp.SOLVER_CALL:
    if cot.op == ExprOp.CONST and cot.value is not None and not np.any(cot.value):
      return tuple(zeros_like(arg) for arg in args)
    raise NotImplementedError("active derivative through SOLVER_CALL is not implemented")
  raise NotImplementedError(f"VJP for op {expr.op!r} is not implemented")


def _ones_like(expr: Expr) -> Expr:
  return Expr.const(np.ones(expr.shape, dtype=np.float64), lowering=expr.lowering)


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
