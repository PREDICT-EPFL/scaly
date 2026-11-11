"""Reverse-mode AD: ``vjp`` and the transpose rules of the structural ops."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..ir.expr import _index_values, Expr, ExprOp, _uniform, gather, matmul, pack, _scatter_zero, topo, to_dense, values, zeros_like

from ..ir.support import slice_indices, support_indices
from ..ir.types import stored_indices
from .calls import _call_vjp, _cond_vjp, _vmap_vjp
from .rules import (
  ELEMENTWISE,
  _masked,
  accumulate,
  carry_mask,
  cotangent,
  _unbroadcast,
  reduction_partial,
  times_partial,
  zero_partial,
  scatter_weights,
)
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
      cot = _on_pattern(cot, out)
    adjoints[out.id] = cot if out.id not in adjoints else adjoints[out.id] + cot

  for expr in reversed(nodes):
    cot = adjoints.get(expr.id)
    if (
      cot is None
      or expr.op in {ExprOp.INPUT, ExprOp.CONST, ExprOp.STOP_GRADIENT}
      or expr.op in ELEMENTWISE
      and ELEMENTWISE[expr.op].partials == (zero_partial,)
      or not needed(expr)
    ):
      continue
    if expr.op == ExprOp.VMAP:
      for arg, arg_cot in _vmap_vjp(expr, cot, wrts, dep_memo, pullback=vjp):
        if arg.id in expr_ids:
          adjoints[arg.id] = arg_cot if arg.id not in adjoints else accumulate(adjoints[arg.id], arg_cot)
      continue
    active = (
      tuple(i for i, arg in enumerate(expr.args) if arg.type.dtype.is_floating and needed(arg))
      if expr.op in {ExprOp.CALL, ExprOp.COND, *ELEMENTWISE} or (expr.op == ExprOp.MATMUL and any(arg.type.pattern is not None for arg in expr.args))
      else None
    )
    transpose = lambda c: _local_vjp(expr, c, active)  # noqa: E731
    arg_cots = carry_mask(cot, transpose) if expr.op in _SELECTIONS else transpose(cot)
    for arg, arg_cot in zip(expr.args, arg_cots, strict=True):
      if arg_cot is not None and arg.id in expr_ids:
        if arg.type.pattern is not None and arg_cot.type.pattern is not arg.type.pattern:
          arg_cot = _on_pattern(arg_cot, arg)
        elif arg.type.pattern is None and arg_cot.type.pattern is not None:
          (arg_cot,) = carry_mask(arg_cot, lambda c: (to_dense(c),))
          assert arg_cot is not None
        adjoints[arg.id] = arg_cot if arg.id not in adjoints else accumulate(adjoints[arg.id], arg_cot)

  return tuple(adjoints.get(wrt.id, zeros_like(wrt)) for wrt in wrts)


# Transpose rules with nonnegative weights let a select's mask pass through them.
_SELECTIONS = {
  ExprOp.PACK,
  ExprOp.VALUES,
  ExprOp.REDUCE,
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
  if expr.op == ExprOp.REDUCE:
    shape = tuple(1 if i in expr.attrs["axes"] else n for i, n in enumerate(args[0].shape))
    if expr.attrs["kind"] != "add":
      (seed,) = carry_mask(cot, lambda c: (to_dense(c).reshape(shape),))
      assert seed is not None
      return (times_partial(reduction_partial(expr), seed),)
    return (cot.reshape(shape) * _ones_like(args[0]),)
  if expr.op == ExprOp.RESHAPE:
    return (cot.reshape(args[0].shape),)
  if expr.op == ExprOp.PRINT:
    return (cot, *(zeros_like(arg) if arg.type.dtype.is_floating else None for arg in args[1:]))
  if expr.op == ExprOp.TRANSPOSE:
    axes = expr.attrs["axes"]
    inv = tuple(int(np.argsort(axes)[i]) for i in range(len(axes)))
    return (cot.transpose(inv),)
  if expr.op == ExprOp.SLICE:
    if args[0].type.pattern is not None:
      return (_stored_movement_vjp(expr, cot),)
    support = _prunable_support(cot)
    entries = support_indices(support) if support is not None and len(support) < cot.size else None
    indices = slice_indices(args[0].shape, expr.attrs["index"], entries)
    selected = gather(cot, entries) if entries is not None else cot
    return (_scatter_zero(selected, indices, args[0].shape),)
  if expr.op == ExprOp.GATHER:
    if args[0].type.pattern is not None:
      return (_stored_movement_vjp(expr, cot), None)
    if expr.args[1].value is None or expr.attrs.get("mode") == "fill":
      return (_scatter_zero(cot, expr.args[1], args[0].shape, axis=expr.attrs.get("axis"), mode=expr.attrs.get("mode", "promise_in_bounds")), None)
    return (_gather_vjp(cot, _index_values(expr), args[0].shape), None)
  if expr.op == ExprOp.SCATTER:
    if args[2].type.pattern is not None:
      return (cot, None, _stored_movement_vjp(expr, cot))
    if expr.attrs["combine"] != "add":
      base_weight, update_weight = scatter_weights(expr)
      selected = gather(cot, args[1], axis=expr.attrs.get("axis"), mode=expr.attrs["mode"]).reshape(args[2].shape)
      return (cot * base_weight, None, selected * update_weight)
    return (cot, None, gather(cot, expr.args[1], axis=expr.attrs.get("axis"), mode=expr.attrs["mode"]).reshape(args[2].shape))
  if expr.op == ExprOp.STACK:
    return _stack_vjp(cot, len(args), expr.attrs.get("axis", 0))
  if expr.op == ExprOp.CONCAT:
    return _concat_vjp(cot, args, expr.attrs.get("axis", 0))
  if expr.op == ExprOp.MATMUL:
    if any(arg.type.pattern is not None for arg in args):
      return _stored_matmul_vjp(args[0], args[1], cot, active)
    return _matmul_vjp(args[0], args[1], cot)
  if expr.op == ExprOp.CALL:
    return _call_vjp(expr, cot, active, pullback=vjp)
  if expr.op == ExprOp.EXTERN:
    raise NotImplementedError(f"active derivative through extern {expr.attrs['callee'].name!r} needs sc.custom_derivative on a wrapping Function")
  if expr.op == ExprOp.COND:
    assert active is not None
    return _cond_vjp(expr, cot, active, pullback=vjp)
  if expr.op == ExprOp.LOOP:
    raise NotImplementedError("derivatives through scan and while_loop are not implemented yet")
  if expr.op == ExprOp.SOLVER_CALL:
    if cot.op == ExprOp.CONST and cot.value is not None and not np.any(cot.value):
      return tuple(zeros_like(arg) for arg in args)
    raise NotImplementedError("active derivative through SOLVER_CALL is not implemented")
  raise NotImplementedError(f"VJP for op {expr.op!r} is not implemented")


def _on_pattern(value: Expr, primal: Expr) -> Expr:
  pattern = primal.type.pattern
  assert pattern is not None
  (result,) = carry_mask(
    value, lambda c: (pack(gather(c, stored_indices(primal.type)).reshape(primal.type.storage_shape), pattern, shape=primal.shape),)
  )
  assert result is not None
  return result


def _stored_movement_vjp(expr: Expr, cot: Expr) -> Expr:
  src = expr.args[2] if expr.op == ExprOp.SCATTER else expr.args[0]
  pattern = src.type.pattern
  assert pattern is not None
  entries = stored_indices(src.type)
  if expr.op == ExprOp.SCATTER:
    selected = gather(cot, _index_values(expr).reshape(-1)[entries]).reshape(src.type.storage_shape)
  else:
    outputs = stored_indices(expr.type)
    inputs = slice_indices(src.shape, expr.attrs["index"], outputs) if expr.op == ExprOp.SLICE else _index_values(expr).reshape(-1)[outputs]
    slots = np.searchsorted(entries, inputs)
    keep = slots < len(entries)
    keep[keep] &= entries[slots[keep]] == inputs[keep]
    selected = _scatter_zero(gather(cot, outputs[keep]), slots[keep], src.type.storage_shape)
  return pack(selected, pattern, shape=src.shape)


def _stored_matmul_vjp(x: Expr, y: Expr, cot: Expr, active: tuple[int, ...] | None) -> tuple[Expr | None, ...]:
  if x.type.pattern is not None and y.type.pattern is not None:
    active = (0, 1) if active is None else active
    return matmul(cot, y.T, sampled=x.type.pattern) if 0 in active else None, matmul(x.T, cot, sampled=y.type.pattern) if 1 in active else None
  left = x.type.pattern is not None
  src, dense = (x, y) if left else (y, x)
  pattern = src.type.pattern
  assert pattern is not None
  row, col = np.asarray(pattern.rows, dtype=np.int64), np.asarray(pattern.cols, dtype=np.int64)
  active = (0, 1) if active is None else active
  sparse_cot = dense_cot = None
  if len(dense.shape) == 1:
    outer, inner = (row, col) if left else (col, row)
    (sampled,) = carry_mask(cot, lambda c: (gather(c, outer),))
    assert sampled is not None
    if (0 if left else 1) in active:
      contribution = _masked(sampled, lambda c: c * gather(dense, inner))
      (sparse_cot,) = carry_mask(contribution, lambda c: (pack(c, pattern),))
    if (1 if left else 0) in active:
      contribution = _masked(sampled, lambda c: values(src) * c)
      (dense_cot,) = carry_mask(contribution, lambda c: (_scatter_zero(c, inner, dense.shape),))
  else:
    width = dense.shape[1] if left else dense.shape[0]
    axis = np.arange(width, dtype=np.int64)
    cot_indices = row[:, None] * width + axis if left else col[:, None] + axis * src.shape[1]
    dense_indices = col[:, None] * width + axis if left else row[:, None] + axis * dense.shape[1]
    (sampled,) = carry_mask(cot, lambda c: (gather(c, cot_indices.reshape(-1)).reshape((pattern.nnz, width)),))
    assert sampled is not None
    if (0 if left else 1) in active:
      coefficients = gather(dense, dense_indices.reshape(-1)).reshape((pattern.nnz, width))
      contribution = _masked(sampled, lambda c: c * coefficients)
      (sparse_cot,) = carry_mask(contribution, lambda c: (pack(c @ _uniform(1, (width,), dtype=cot.type.dtype), pattern),))
    if (1 if left else 0) in active:
      contribution = _masked(sampled, lambda c: values(src).reshape((pattern.nnz, 1)) * c)
      (dense_cot,) = carry_mask(contribution, lambda c: (_scatter_zero(c.vec(), dense_indices.reshape(-1), dense.shape),))
  return (sparse_cot, dense_cot) if left else (dense_cot, sparse_cot)


def _ones_like(expr: Expr) -> Expr:
  if expr.type.pattern is not None:
    return pack(_uniform(1, expr.type.storage_shape, dtype=expr.type.dtype, lowering=expr.lowering), expr.type.pattern, shape=expr.shape)
  return _uniform(1, expr.shape, dtype=expr.type.dtype, lowering=expr.lowering)


def _gather_vjp(cot: Expr, indices: np.ndarray, shape: tuple[int, ...]) -> Expr:
  return _scatter_zero(cot, indices, shape)


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
  a = x.reshape((1, *x.shape)) if len(x.shape) == 1 else x
  b = y.reshape((*y.shape, 1)) if len(y.shape) == 1 else y
  batch = (a @ b).shape[:-2]
  c = cot.reshape((*batch, a.shape[-2], b.shape[-1]))

  def transpose(value):
    axes = list(range(len(value.shape)))
    axes[-2:] = axes[-2:][::-1]
    return value.transpose(tuple(axes))

  da, db = c @ transpose(b), transpose(a) @ c
  return _unbroadcast(da, a.shape, da.shape).reshape(x.shape), _unbroadcast(db, b.shape, db.shape).reshape(y.shape)
