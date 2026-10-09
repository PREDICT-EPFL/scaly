"""Forward-mode AD through one iterative traversal with a leading seed axis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..ir.expr import Expr, ExprOp, concat, gather, scatter, stack
from ..passes.expr import simplify_cse_fixpoint
from .calls import _call_jvp_many, _is_zero_const, _vmap_jvp_many


def jvp(expr: Expr, wrt: Expr, seed: Expr) -> Expr:
  """Forward-mode derivative: ``J(expr, wrt) @ seed``, with ``seed`` shaped like ``wrt``."""
  tangent = jvp_many(expr, wrt, seed.reshape((1, *seed.shape)))
  if tangent.op == ExprOp.CONST:
    assert tangent.value is not None
    return Expr.const(tangent.value.reshape(expr.shape), dtype=tangent.type.dtype, lowering=tangent.lowering)
  return tangent.reshape(expr.shape)


def jvp_many(expr: Expr, wrt: Expr, seeds: Expr) -> Expr:
  """Forward mode over seeds shaped ``(nseed, *wrt.shape)``, returning ``(nseed, *expr.shape)``.

  One traversal shares primal work across all seeds. A missing derivative rule raises.
  """
  if len(seeds.shape) < 1 or seeds.shape[1:] != wrt.shape:
    raise ValueError(f"multi-seed JVP expects seeds shape (nseed, *{wrt.shape}), got {seeds.shape}")
  if not wrt.type.dtype.is_floating or not expr.type.dtype.is_floating:
    raise TypeError("integer and boolean values have no tangent")
  if seeds.type.dtype != wrt.type.dtype:
    raise TypeError(f"JVP seeds must have primal dtype {wrt.type.dtype}, got {seeds.type.dtype}")
  seeds = simplify_cse_fixpoint(seeds)
  (ret,) = _pushforward((expr,), {wrt: seeds}, seeds.shape[0])
  assert ret is not None
  return ret


def _zero_tangent(expr: Expr, nseed: int) -> Expr:
  return Expr.const(np.zeros((nseed, *expr.shape), dtype=expr.type.dtype.numpy()), dtype=expr.type.dtype, lowering=expr.lowering)


def _pushforward(outputs: Sequence[Expr], seeds: dict[Expr, Expr], nseed: int, callee: str = "<top-level>") -> tuple[Expr | None, ...]:
  """Keep floating zeros implicit internally; nonfloating outputs have no tangent."""
  tangents: dict[int, Expr | None] = {}
  pending = [(out, False) for out in reversed(outputs)]
  while pending:
    expr, visited = pending.pop()
    if expr.id in tangents:
      continue
    args = expr.args[:1] if expr.op == ExprOp.PRINT else expr.args
    if expr.type.dtype.is_floating and nseed and args and not visited:
      pending.append((expr, True))
      pending.extend((arg, False) for arg in reversed(args) if arg.id not in tangents)
      continue
    if not expr.type.dtype.is_floating or not nseed or expr.op == ExprOp.CONST:
      tangent = None
    elif expr.op == ExprOp.INPUT:
      tangent = seeds.get(expr)
    else:
      d = [tangents[arg.id] for arg in args]
      if all(t is None for t in d):
        tangent = None
      elif expr.op == ExprOp.PRINT:
        tangent = d[0]
      elif expr.op in {ExprOp.CALL, ExprOp.VMAP}:
        call_tangents = [None if t is None else simplify_cse_fixpoint(t) for t in d]
        tangent = (
          _call_jvp_many(expr, call_tangents, nseed, pushforward=_pushforward)
          if expr.op == ExprOp.CALL
          else _vmap_jvp_many(expr, call_tangents, nseed, pushforward=_pushforward)
        )
      else:
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
        tangent = tangent * Expr.const(np.ones((1, *expr.shape), dtype=expr.type.dtype.numpy()), dtype=expr.type.dtype)
    tangents[expr.id] = None if tangent is not None and _is_zero_const(tangent) else tangent
  return tuple(tangents[out.id] if tangents[out.id] is not None or not out.type.dtype.is_floating else _zero_tangent(out, nseed) for out in outputs)


def _pushforward_rule(expr: Expr, d: Sequence[Expr], nseed: int, callee: str) -> Expr:
  args = expr.args
  op = expr.op
  if op == ExprOp.NEG:
    return -d[0]
  if op in {ExprOp.ADD, ExprOp.SUB}:
    dx = _broadcast_tangent(d[0], args[0], expr, nseed)
    dy = _broadcast_tangent(d[1], args[1], expr, nseed)
    if _is_zero_const(d[0]):
      return dy if op == ExprOp.ADD else -dy
    if _is_zero_const(d[1]):
      return dx
    return dx + dy if op == ExprOp.ADD else dx - dy
  if op == ExprOp.MUL:
    if args[0] is args[1]:
      return _seed_axis(2 * args[0]) * d[0]
    if _is_zero_const(d[0]):
      return _seed_axis(args[0], expr) * _broadcast_tangent(d[1], args[1], expr, nseed)
    if _is_zero_const(d[1]):
      return _broadcast_tangent(d[0], args[0], expr, nseed) * _seed_axis(args[1], expr)
    return _broadcast_tangent(d[0], args[0], expr, nseed) * _seed_axis(args[1], expr) + _seed_axis(args[0], expr) * _broadcast_tangent(
      d[1], args[1], expr, nseed
    )
  if op == ExprOp.DIV:
    dx = _broadcast_tangent(d[0], args[0], expr, nseed)
    dy = _broadcast_tangent(d[1], args[1], expr, nseed)
    numerator = dx if _is_zero_const(d[1]) else -_seed_axis(expr) * dy if _is_zero_const(d[0]) else dx - _seed_axis(expr) * dy
    return numerator * _seed_axis(1.0 / args[1], expr)
  if op == ExprOp.POW:
    dx = _broadcast_tangent(d[0], args[0], expr, nseed)
    if args[1].op == ExprOp.CONST:
      return _seed_axis(args[1] * (args[0] ** (args[1] - 1)), expr) * dx
    x, y = _seed_axis(args[0], expr), _seed_axis(args[1], expr)
    dy = _broadcast_tangent(d[1], args[1], expr, nseed)
    term = y * dx / x if _is_zero_const(d[1]) else dy * x.log() if _is_zero_const(d[0]) else dy * x.log() + y * dx / x
    return _seed_axis(expr) * term
  if op == ExprOp.SIN:
    return _seed_axis(args[0].cos()) * d[0]
  if op == ExprOp.COS:
    return -_seed_axis(args[0].sin()) * d[0]
  if op == ExprOp.TAN:
    return d[0] / (_seed_axis(args[0].cos()) ** 2)
  if op == ExprOp.ASIN:
    return d[0] / _seed_axis((1 - args[0] ** 2).sqrt())
  if op == ExprOp.ACOS:
    return -d[0] / _seed_axis((1 - args[0] ** 2).sqrt())
  if op == ExprOp.ATAN:
    return d[0] / _seed_axis(1 + args[0] ** 2)
  if op == ExprOp.ATAN2:
    y, x = args
    dy = _broadcast_tangent(d[0], y, expr, nseed)
    dx = _broadcast_tangent(d[1], x, expr, nseed)
    left, right = _seed_axis(x, expr), _seed_axis(y, expr)
    numerator = -right * dx if _is_zero_const(d[0]) else left * dy if _is_zero_const(d[1]) else left * dy - right * dx
    return numerator / _seed_axis(x * x + y * y, expr)
  if op == ExprOp.SINH:
    return _seed_axis(args[0].cosh()) * d[0]
  if op == ExprOp.COSH:
    return _seed_axis(args[0].sinh()) * d[0]
  if op == ExprOp.TANH:
    return d[0] * (1 - _seed_axis(expr * expr))
  if op == ExprOp.ERF:
    return _seed_axis((2 / np.sqrt(np.pi)) * (-(args[0] ** 2)).exp()) * d[0]
  if op == ExprOp.EXP:
    return _seed_axis(expr) * d[0]
  if op == ExprOp.LOG:
    return d[0] / _seed_axis(args[0])
  if op == ExprOp.SQRT:
    return d[0] * _seed_axis(0.5 / expr)
  if op == ExprOp.ABS:
    return _seed_axis(args[0] / args[0].abs()) * d[0]
  if op == ExprOp.SUM:
    ones = Expr.const(np.ones(args[0].size, dtype=expr.type.dtype.numpy()), dtype=expr.type.dtype)
    return d[0].reshape((nseed, args[0].size)) @ ones
  if op == ExprOp.RESHAPE:
    return d[0].reshape((nseed, *expr.shape))
  if op == ExprOp.TRANSPOSE:
    return d[0].transpose((0, *(axis + 1 for axis in expr.attrs["axes"])))
  if op == ExprOp.SLICE:
    return d[0][(slice(None), *expr.attrs["index"])]
  if op == ExprOp.GATHER:
    indices = expr.attrs["indices"].reshape(-1)
    full = (np.arange(nseed, dtype=np.int64)[:, None] * args[0].size + indices[None, :]).reshape(-1)
    return gather(d[0], full).reshape((nseed, *expr.shape))
  if op == ExprOp.SCATTER:
    indices = expr.attrs["indices"].reshape(-1)
    full = (np.arange(nseed, dtype=np.int64)[:, None] * expr.size + indices[None, :]).reshape(-1)
    return scatter(d[0].reshape((nseed * args[0].size,)), full, (nseed * expr.size,)).reshape((nseed, *expr.shape))
  if op == ExprOp.STACK:
    return stack(d, axis=expr.attrs.get("axis", 0) + 1)
  if op == ExprOp.CONCAT:
    return concat(d, axis=expr.attrs.get("axis", 0) + 1)
  if op == ExprOp.MATMUL:
    if args[0] is args[1] and len(args[0].shape) == 1:
      return 2 * (d[0] @ args[0])
    if _is_zero_const(d[0]):
      return _jvp_many_matmul_right(args[0], args[1], d[1], nseed)
    if _is_zero_const(d[1]):
      return _jvp_many_matmul_left(args[0], args[1], d[0], nseed)
    return _jvp_many_matmul_left(args[0], args[1], d[0], nseed) + _jvp_many_matmul_right(args[0], args[1], d[1], nseed)
  if op == ExprOp.SOLVER_CALL:
    raise NotImplementedError(f"active derivative through SOLVER_CALL in callee {callee!r} is not implemented")
  if op in {ExprOp.FLOOR, ExprOp.CEIL, ExprOp.MINIMUM, ExprOp.MAXIMUM}:
    raise NotImplementedError(f"JVP for nonsmooth op {op!r} in callee {callee!r} is not implemented")
  raise NotImplementedError(f"JVP for op {op!r} in callee {callee!r} is not implemented")


def _seed_axis(expr: Expr, output: Expr | None = None) -> Expr:
  missing = 0 if output is None else len(output.shape) - len(expr.shape)
  return expr.reshape((1, *(1,) * missing, *expr.shape))


def _broadcast_tangent(tangent: Expr, operand: Expr, output: Expr, nseed: int) -> Expr:
  missing = len(output.shape) - len(operand.shape)
  return tangent if missing == 0 else tangent.reshape((nseed, *(1,) * missing, *operand.shape))


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
