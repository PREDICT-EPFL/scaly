"""Forward-mode AD through one iterative traversal with a leading seed axis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..ir.expr import Expr, ExprOp, _uniform, callees_of, concat, from_dense, gather, pack, scatter, stack, to_dense, values
from ..ir.types import TensorType, stored_indices
from ..passes.expr import simplify_cse_fixpoint
from .calls import _call_jvp_many, _is_zero_const, _vmap_jvp_many, read_args
from .rules import ELEMENTWISE, tangent as _elementwise_tangent
from .sparsity import _check_wrt, _prunable_support


def jvp(expr: Expr, wrt: Expr, seed: Expr) -> Expr:
  """Forward-mode derivative: ``J(expr, wrt) @ seed``, with ``seed`` shaped like ``wrt``.

  ``wrt`` must be an independent input from ``sc.sym`` or a Function declaration.
  """
  if seed.shape != wrt.shape:
    raise ValueError(f"JVP expects seed shape {wrt.shape}, got {seed.shape}")
  if seed.type.pattern is not wrt.type.pattern:
    if seed.type.pattern is not None or wrt.type.pattern is None:
      raise ValueError("JVP seeds must have the primal's stored pattern")
    seed = from_dense(seed, wrt.type.pattern)
  tangent = jvp_many(expr, wrt, _reshape_tangent(seed, wrt, 1))
  if expr.type.pattern is not None:
    return pack(values(tangent).reshape(expr.type.storage_shape), expr.type.pattern, shape=expr.shape)
  if tangent.op == ExprOp.CONST:
    assert tangent.value is not None
    return Expr.const(tangent.value.reshape(expr.shape), dtype=tangent.type.dtype, lowering=tangent.lowering)
  return tangent.reshape(expr.shape)


def jvp_many(expr: Expr, wrt: Expr, seeds: Expr) -> Expr:
  """Forward mode over seeds shaped ``(nseed, *wrt.shape)``, returning ``(nseed, *expr.shape)``.

  ``wrt`` must be an independent input from ``sc.sym`` or a Function declaration. One traversal
  shares primal work across all seeds. A missing derivative rule raises.
  """
  _check_wrt(wrt)
  if len(seeds.shape) < 1 or seeds.shape[1:] != wrt.shape:
    raise ValueError(f"multi-seed JVP expects seeds shape (nseed, *{wrt.shape}), got {seeds.shape}")
  if not wrt.type.dtype.is_floating or not expr.type.dtype.is_floating:
    raise TypeError("integer and boolean values have no tangent")
  if seeds.type.dtype != wrt.type.dtype:
    raise TypeError(f"JVP seeds must have primal dtype {wrt.type.dtype}, got {seeds.type.dtype}")
  if seeds.type.pattern is not wrt.type.pattern:
    if seeds.type.pattern is not None or wrt.type.pattern is None:
      raise ValueError("JVP seeds must have the primal's stored pattern")
    seeds = from_dense(seeds, wrt.type.pattern)
  seeds = simplify_cse_fixpoint(seeds)
  (ret,) = _pushforward((expr,), {wrt: seeds}, seeds.shape[0])
  assert ret is not None
  return ret


def _zero_tangent(expr: Expr, nseed: int) -> Expr:
  zero = _uniform(0, (nseed, *expr.type.storage_shape), dtype=expr.type.dtype, lowering=expr.lowering)
  return pack(zero, expr.type.pattern, shape=(nseed, *expr.shape)) if expr.type.pattern is not None else zero


def _reshape_tangent(tangent: Expr, primal: Expr, nseed: int) -> Expr:
  if primal.type.pattern is not None:
    return pack(values(tangent).reshape((nseed, *primal.type.storage_shape)), primal.type.pattern, shape=(nseed, *primal.shape))
  return tangent.reshape((nseed, *primal.shape))


def _pushforward(outputs: Sequence[Expr], seeds: dict[Expr, Expr], nseed: int, callee: str = "<top-level>") -> tuple[Expr | None, ...]:
  """Keep floating zeros implicit internally; nonfloating outputs have no tangent."""
  tangents: dict[int, Expr | None] = {}
  pending = [(out, False) for out in reversed(outputs)]
  while pending:
    expr, visited = pending.pop()
    if expr.id in tangents:
      continue
    args = () if expr.op == ExprOp.STOP_GRADIENT else expr.args[:1] if expr.op == ExprOp.PRINT else read_args(expr) if callees_of(expr) else expr.args
    if expr.type.dtype.is_floating and nseed and args and not visited:
      pending.append((expr, True))
      pending.extend((arg, False) for arg in reversed(args) if arg.id not in tangents)
      continue
    if not expr.type.dtype.is_floating or not nseed or expr.op in {ExprOp.CONST, ExprOp.STOP_GRADIENT}:
      tangent = None
    elif expr.op == ExprOp.INPUT:
      tangent = seeds.get(expr)
    else:
      d = [tangents[arg.id] for arg in args]
      if all(t is None for t in d):
        tangent = None
      elif expr.op == ExprOp.PRINT:
        tangent = d[0]
      elif expr.op == ExprOp.EXTERN:
        raise NotImplementedError(f"active derivative through extern {expr.attrs['callee'].name!r} needs sc.custom_derivative on a wrapping Function")
      elif callees_of(expr):
        read = {arg.id for arg in args}
        call_tangents = [simplify_cse_fixpoint(t) if arg.id in read and (t := tangents[arg.id]) is not None else None for arg in expr.args]
        tangent = (
          _call_jvp_many(expr, call_tangents, nseed, pushforward=_pushforward)
          if expr.op == ExprOp.CALL
          else _vmap_jvp_many(expr, call_tangents, nseed, pushforward=_pushforward)
        )
      elif expr.op in ELEMENTWISE:
        if ELEMENTWISE[expr.op].partials is None:
          raise NotImplementedError(f"JVP for nonsmooth op {expr.op!r} in callee {callee!r} is not implemented")
        tangent = _elementwise_tangent(expr, d, nseed)
      else:
        assert all(arg.type.dtype.is_floating for arg in args)
        d = [t if t is not None else _zero_tangent(arg, nseed) for arg, t in zip(args, d, strict=True)]
        tangent = _pushforward_rule(expr, d, nseed, callee)
        if expr.op in {
          ExprOp.REDUCE,
          ExprOp.RESHAPE,
          ExprOp.TRANSPOSE,
          ExprOp.SLICE,
          ExprOp.GATHER,
          ExprOp.SCATTER,
          ExprOp.STACK,
          ExprOp.CONCAT,
        } and all(t.op == ExprOp.CONST for t in d):
          tangent = simplify_cse_fixpoint(tangent)
      if tangent is not None and tangent.shape != (nseed, *expr.shape):
        tangent = tangent * _uniform(1, (1, *expr.shape), dtype=expr.type.dtype)
      if tangent is not None and tangent.type.pattern is not expr.type.pattern:
        if expr.type.pattern is None:
          tangent = to_dense(tangent)
        else:
          type_ = TensorType((nseed, *expr.shape), expr.type.dtype, pattern=expr.type.pattern)
          tangent = pack(gather(tangent, stored_indices(type_)).reshape(type_.storage_shape), expr.type.pattern, shape=type_.shape)
    tangents[expr.id] = None if tangent is not None and _prunable_support(tangent) == frozenset() else tangent
  return tuple(tangents[out.id] if tangents[out.id] is not None or not out.type.dtype.is_floating else _zero_tangent(out, nseed) for out in outputs)


def _pushforward_rule(expr: Expr, d: Sequence[Expr], nseed: int, callee: str) -> Expr:
  args = expr.args
  op = expr.op
  if op == ExprOp.PACK:
    assert expr.type.pattern is not None
    return pack(d[0], expr.type.pattern, shape=(nseed, *expr.shape))
  if op == ExprOp.VALUES:
    return values(d[0])
  if op == ExprOp.REDUCE:
    if expr.attrs["kind"] != "add":
      raise NotImplementedError(f"JVP for REDUCE {expr.attrs['kind']} awaits #65")
    tangent = d[0].sum(tuple(axis + 1 for axis in expr.attrs["axes"]))
    return to_dense(tangent) if expr.type.pattern is None else tangent
  if op == ExprOp.RESHAPE:
    if args[0].type.pattern is not None:
      return d[0].reshape((nseed, *expr.shape))
    return _reshape_tangent(d[0], expr, nseed)
  if op == ExprOp.TRANSPOSE:
    return d[0].transpose((0, *(axis + 1 for axis in expr.attrs["axes"])))
  if op == ExprOp.SLICE:
    return d[0][(slice(None), *expr.attrs["index"])]
  if op == ExprOp.GATHER:
    indices = expr.attrs["indices"].reshape(-1)
    if args[0].type.pattern is not None:
      return stack(gather(d[0][i], expr.attrs["indices"]) for i in range(nseed))
    full = (np.arange(nseed, dtype=np.int64)[:, None] * args[0].size + indices[None, :]).reshape(-1)
    return gather(d[0], full).reshape((nseed, *expr.shape))
  if op == ExprOp.SCATTER:
    indices = expr.attrs["indices"].reshape(-1)
    if args[0].type.pattern is not None:
      return stack(scatter(d[0][i], indices, expr.shape) for i in range(nseed))
    full = (np.arange(nseed, dtype=np.int64)[:, None] * expr.size + indices[None, :]).reshape(-1)
    return scatter(d[0].reshape((nseed * args[0].size,)), full, (nseed * expr.size,)).reshape((nseed, *expr.shape))
  if op == ExprOp.STACK:
    return stack(d, axis=expr.attrs.get("axis", 0) + 1)
  if op == ExprOp.CONCAT:
    return concat(d, axis=expr.attrs.get("axis", 0) + 1)
  if op == ExprOp.MATMUL:
    if any(arg.type.pattern is not None for arg in args):
      terms = []
      if not _is_zero_const(d[0]):
        terms.append(
          stack(d[0][i] @ args[1] for i in range(nseed)) if args[0].type.pattern is not None else _jvp_many_matmul_left(args[0], args[1], d[0], nseed)
        )
      if not _is_zero_const(d[1]):
        terms.append(
          stack(args[0] @ d[1][i] for i in range(nseed))
          if args[1].type.pattern is not None
          else _jvp_many_matmul_right(args[0], args[1], d[1], nseed)
        )
      return terms[0] if len(terms) == 1 else terms[0] + terms[1]
    if args[0] is args[1] and len(args[0].shape) == 1:
      return 2 * (d[0] @ args[0])
    if _is_zero_const(d[0]):
      return _jvp_many_matmul_right(args[0], args[1], d[1], nseed)
    if _is_zero_const(d[1]):
      return _jvp_many_matmul_left(args[0], args[1], d[0], nseed)
    return _jvp_many_matmul_left(args[0], args[1], d[0], nseed) + _jvp_many_matmul_right(args[0], args[1], d[1], nseed)
  if op == ExprOp.SOLVER_CALL:
    raise NotImplementedError(f"active derivative through SOLVER_CALL in callee {callee!r} is not implemented")
  raise NotImplementedError(f"JVP for op {op!r} in callee {callee!r} is not implemented")


def _jvp_many_matmul_left(x: Expr, y: Expr, dx: Expr, nseed: int) -> Expr:
  batch = max(len(x.shape) - 2, len(y.shape) - 2, 0)
  left = dx.reshape((nseed, *((1,) * (batch - len(x.shape[:-2]))), *((1, x.shape[0]) if len(x.shape) == 1 else x.shape)))
  right = y.reshape((*y.shape, 1)) if len(y.shape) == 1 else y
  return (left @ right).reshape((nseed, *(x @ y).shape))


def _jvp_many_matmul_right(x: Expr, y: Expr, dy: Expr, nseed: int) -> Expr:
  batch = max(len(x.shape) - 2, len(y.shape) - 2, 0)
  left = x.reshape((1, *x.shape)) if len(x.shape) == 1 else x
  right = dy.reshape((nseed, *((1,) * (batch - len(y.shape[:-2]))), *y.shape, *((1,) if len(y.shape) == 1 else ())))
  return (left @ right).reshape((nseed, *(x @ y).shape))
