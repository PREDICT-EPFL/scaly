"""Forward-mode AD through one iterative traversal with a leading seed axis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..ir.expr import (
  _index_values,
  Expr,
  ExprOp,
  _scatter_zero,
  _uniform,
  callees_of,
  concat,
  from_dense,
  gather,
  pack,
  scatter,
  stack,
  to_dense,
  values,
)
from ..passes.expr import simplify_cse_fixpoint
from .calls import _call_jvp_many, _is_zero_const, _vmap_jvp_many, read_args
from .rules import ELEMENTWISE, scatter_weights, tangent as _elementwise_tangent
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
        if expr.op == ExprOp.GATHER:
          args, d = args[:1], d[:1]
        elif expr.op == ExprOp.SCATTER:
          args, d = (args[0], args[2]), (d[0], d[2])
        assert all(arg.type.dtype.is_floating for arg in args)
        d = [t if t is not None else _zero_tangent(arg, nseed) for arg, t in zip(args, d, strict=True)]
        tangent = _pushforward_rule(expr, d, nseed, callee)
        if expr.op in {
          ExprOp.SUM,
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
  if op == ExprOp.SUM:
    width = args[0].type.storage_size
    ones = _uniform(1, (width,), dtype=expr.type.dtype)
    return values(d[0]).reshape((nseed, width)) @ ones
  if op == ExprOp.RESHAPE:
    return _reshape_tangent(d[0], expr, nseed)
  if op == ExprOp.TRANSPOSE:
    return d[0].transpose((0, *(axis + 1 for axis in expr.attrs["axes"])))
  if op == ExprOp.SLICE:
    return d[0][(slice(None), *expr.attrs["index"])]
  if op == ExprOp.GATHER:
    if expr.args[1].value is None or expr.attrs.get("mode") == "fill":
      axis = expr.attrs.get("axis")
      source = d[0].reshape((nseed, args[0].size)) if axis is None else d[0]
      return gather(source, args[1], axis=1 if axis is None else axis + 1, mode=expr.attrs["mode"]).reshape((nseed, *expr.shape))
    indices = _index_values(expr).reshape(-1)
    full = (np.arange(nseed, dtype=np.int64)[:, None] * args[0].size + indices[None, :]).reshape(-1)
    return gather(d[0], full).reshape((nseed, *expr.shape))
  if op == ExprOp.SCATTER:
    if expr.args[1].value is None or expr.attrs["mode"] == "fill" or expr.attrs["combine"] != "add":
      base, updates = d
      if expr.attrs["combine"] != "add":
        base_weight, update_weight = scatter_weights(expr)
        base, updates = base * base_weight, updates * update_weight
      axis = expr.attrs.get("axis")
      if axis is None:
        base = base.reshape((nseed, expr.size))
        updates = updates.reshape((nseed, *args[1].shape))
      return scatter(base, args[1], updates, axis=1 if axis is None else axis + 1, mode=expr.attrs["mode"]).reshape((nseed, *expr.shape))
    indices = _index_values(expr).reshape(-1)
    full = (np.arange(nseed, dtype=np.int64)[:, None] * expr.size + indices[None, :]).reshape(-1)
    scattered = _scatter_zero(d[1].reshape((nseed * args[2].size,)), full, (nseed * expr.size,)).reshape((nseed, *expr.shape))
    return scattered if d[0].support == frozenset() else d[0] + scattered
  if op == ExprOp.STACK:
    return stack(d, axis=expr.attrs.get("axis", 0) + 1)
  if op == ExprOp.CONCAT:
    return concat(d, axis=expr.attrs.get("axis", 0) + 1)
  if op == ExprOp.MATMUL:
    if any(arg.type.pattern is not None for arg in args):
      dense_expr = to_dense(args[0]) @ to_dense(args[1])
      dense_tangent = _pushforward_rule(dense_expr, [to_dense(t) for t in d], nseed, callee)
      return from_dense(dense_tangent, expr.type.pattern) if expr.type.pattern is not None else dense_tangent
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
  if len(x.shape) == 1:
    return dx @ y
  m, k = x.shape
  return (dx.reshape((nseed * m, k)) @ y).reshape((nseed, m, *y.shape[1:]))


def _jvp_many_matmul_right(x: Expr, y: Expr, dy: Expr, nseed: int) -> Expr:
  if len(y.shape) == 1:
    return (x @ dy.T).T if len(x.shape) == 2 else dy @ x
  k, n = y.shape
  product = x @ dy.transpose((1, 0, 2)).reshape((k, nseed * n))
  if len(x.shape) == 1:
    return product.reshape((nseed, n))
  return product.reshape((x.shape[0], nseed, n)).transpose((1, 0, 2))
