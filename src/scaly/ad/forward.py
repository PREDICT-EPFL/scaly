"""Forward-mode AD: single-seed ``jvp`` and multi-seed ``jvp_many``.

``jvp_many`` has structural rules that share work across seeds; when an op has none it falls back
to unrolling ``jvp`` per seed unless ``SCALY_STRICT_JVP_MANY`` forbids it.
"""

from __future__ import annotations

import numpy as np

from ..ir.expr import Expr, ExprOp, concat, gather, scatter, stack, zeros_like
from ..passes.expr import simplify_cse_fixpoint
from ..utils.env import env_bool
from .calls import _call_jvp, _call_jvp_many, _is_zero_const, _vmap_jvp_many
from .sparsity import _depends_on


class _JVPManyUnsupported(Exception):
  def __init__(self, op: str):
    super().__init__(op)
    self.op = op


def jvp(expr: Expr, wrt: Expr, seed: Expr) -> Expr:
  """Forward-mode derivative: ``J(expr, wrt) @ seed``, with ``seed`` shaped like ``wrt``.

  One pass per seed. For many seeds at once use ``jvp_many``, which shares the expensive work.
  """
  return _jvp(expr, {wrt: seed}, {}, {})


def _jvp(expr: Expr, seeds: dict[Expr, Expr], memo: dict[int, Expr], dep_memo: dict[tuple[int, int], bool]) -> Expr:
  if expr.id in memo:
    return memo[expr.id]
  if not any(not _is_zero_const(seed) and _depends_on(expr, wrt, dep_memo) for wrt, seed in seeds.items()):
    memo[expr.id] = ret = zeros_like(expr)
    return ret
  if expr.op == ExprOp.INPUT:
    memo[expr.id] = ret = seeds.get(expr, zeros_like(expr))
    return ret
  if expr.op == ExprOp.CONST:
    memo[expr.id] = ret = zeros_like(expr)
    return ret
  if expr.op in (ExprOp.CALL, ExprOp.VMAP):
    tangents = [_jvp(arg, seeds, memo, dep_memo) for arg in expr.args]
    memo[expr.id] = ret = _call_jvp(expr, tangents, pushforward=_jvp)
    return ret
  if expr.op == ExprOp.SOLVER_CALL:
    raise NotImplementedError("active derivative through SOLVER_CALL is not implemented")
  if expr.op == ExprOp.PRINT:
    memo[expr.id] = ret = _jvp(expr.args[0], seeds, memo, dep_memo)
    return ret

  def save(ret: Expr) -> Expr:
    memo[expr.id] = ret
    return ret

  args = expr.args
  d = [_jvp(a, seeds, memo, dep_memo) for a in args]
  if expr.op == ExprOp.NEG:
    return save(-d[0])
  if expr.op == ExprOp.ADD:
    return save(d[0] + d[1])
  if expr.op == ExprOp.SUB:
    return save(d[0] - d[1])
  if expr.op == ExprOp.MUL:
    return save((2 * args[0]) * d[0] if args[0] is args[1] else d[0] * args[1] + args[0] * d[1])
  if expr.op == ExprOp.DIV:
    return save((d[0] - expr * d[1]) * (1.0 / args[1]))
  if expr.op == ExprOp.POW:
    if args[1].op == ExprOp.CONST:
      return save(args[1] * (args[0] ** (args[1] - 1)) * d[0])
    return save(expr * (d[1] * args[0].log() + args[1] * d[0] / args[0]))
  if expr.op == ExprOp.SIN:
    return save(args[0].cos() * d[0])
  if expr.op == ExprOp.COS:
    return save(-args[0].sin() * d[0])
  if expr.op == ExprOp.TAN:
    return save(d[0] / (args[0].cos() ** 2))
  if expr.op == ExprOp.ASIN:
    return save(d[0] / (1 - args[0] ** 2).sqrt())
  if expr.op == ExprOp.ACOS:
    return save(-d[0] / (1 - args[0] ** 2).sqrt())
  if expr.op == ExprOp.ATAN:
    return save(d[0] / (1 + args[0] ** 2))
  if expr.op == ExprOp.ATAN2:
    y, x = args
    dy, dx = d
    return save((x * dy - y * dx) / (x * x + y * y))
  if expr.op == ExprOp.SINH:
    return save(args[0].cosh() * d[0])
  if expr.op == ExprOp.COSH:
    return save(args[0].sinh() * d[0])
  if expr.op == ExprOp.TANH:
    return save(d[0] * (1 - expr * expr))
  if expr.op == ExprOp.ERF:
    return save((2 / np.sqrt(np.pi)) * (-(args[0] ** 2)).exp() * d[0])
  if expr.op == ExprOp.EXP:
    return save(expr * d[0])
  if expr.op == ExprOp.LOG:
    return save(d[0] / args[0])
  if expr.op == ExprOp.SQRT:
    return save(d[0] * (0.5 / expr))
  if expr.op == ExprOp.ABS:
    return save(args[0] / args[0].abs() * d[0])
  if expr.op in {ExprOp.FLOOR, ExprOp.CEIL, ExprOp.MINIMUM, ExprOp.MAXIMUM}:
    raise NotImplementedError(f"JVP for nonsmooth op {expr.op!r} is not implemented")
  if expr.op == ExprOp.SUM:
    return save(d[0].sum())
  if expr.op == ExprOp.RESHAPE:
    return save(d[0].reshape(expr.shape))
  if expr.op == ExprOp.TRANSPOSE:
    return save(d[0].transpose(expr.attrs["axes"]))
  if expr.op == ExprOp.SLICE:
    return save(Expr.const(d[0].value[expr.attrs["index"]]) if d[0].op == ExprOp.CONST and d[0].value is not None else d[0][expr.attrs["index"]])
  if expr.op == ExprOp.GATHER:
    return save(
      Expr.const(np.take(d[0].value.reshape(-1), expr.attrs["indices"]).reshape(expr.attrs["indices"].shape))
      if d[0].op == ExprOp.CONST and d[0].value is not None
      else gather(d[0], expr.attrs["indices"])
    )
  if expr.op == ExprOp.SCATTER:
    return save(scatter(d[0], expr.attrs["indices"], expr.shape))
  if expr.op == ExprOp.STACK:
    return save(stack(d, axis=expr.attrs.get("axis", 0)))
  if expr.op == ExprOp.CONCAT:
    return save(concat(d, axis=expr.attrs.get("axis", 0)))
  if expr.op == ExprOp.MATMUL:
    if args[0] is args[1] and len(args[0].shape) == 1:
      return save(2 * (args[0] @ d[0]))
    return save(d[0] @ args[1] + args[0] @ d[1])
  raise NotImplementedError(f"JVP for op {expr.op!r} is not implemented")


def jvp_many(expr: Expr, wrt: Expr, seeds: Expr) -> Expr:
  """Forward mode over several seeds in one pass. ``seeds`` has shape ``(n, *wrt.shape)``.

  Structural rules share work across seeds. One ``cos`` serves every column of a ``sin``'s
  derivative, so this is much cheaper than ``n`` separate ``jvp`` calls. Ops without a
  multi-seed rule fall back to per-seed evaluation. Set ``SCALY_STRICT_JVP_MANY=1`` to raise
  instead of falling back. Returns shape ``(n, *expr.shape)``.
  """
  if len(seeds.shape) < 1 or seeds.shape[1:] != wrt.shape:
    raise ValueError(f"multi-seed JVP expects seeds shape (nseed, *{wrt.shape}), got {seeds.shape}")
  if seeds.shape[0] == 0:
    return Expr.const(np.zeros((0, *expr.shape), dtype=np.float64))
  strict = env_bool("SCALY_STRICT_JVP_MANY", False)
  try:
    ret = _jvp_many_structural(expr, wrt, seeds, {}, {})
  except _JVPManyUnsupported as unsupported:
    if strict:
      raise NotImplementedError(
        f"structural jvp_many does not support {unsupported.op!r}; SCALY_STRICT_JVP_MANY=1 forbids the unrolled fallback"
      ) from unsupported
    return _jvp_many_unrolled(expr, wrt, seeds)
  expected = (seeds.shape[0], *expr.shape)
  if ret.shape == expected:
    return ret
  if strict:
    raise NotImplementedError(
      f"structural jvp_many returned shape {ret.shape} for {expr.op!r}, expected {expected}; SCALY_STRICT_JVP_MANY=1 forbids the unrolled fallback"
    )
  return _jvp_many_unrolled(expr, wrt, seeds)


def _jvp_many_unrolled(expr: Expr, wrt: Expr, seeds: Expr) -> Expr:
  return stack([jvp(expr, wrt, seeds[i]) for i in range(seeds.shape[0])], axis=0)


def _jvp_many_structural(expr: Expr, wrt: Expr, seeds: Expr, memo: dict[int, Expr], dep_memo: dict[tuple[int, int], bool]) -> Expr:
  if expr.id in memo:
    return memo[expr.id]
  nseed = seeds.shape[0]
  if _is_zero_const(seeds) or not _depends_on(expr, wrt, dep_memo):
    memo[expr.id] = ret = Expr.const(np.zeros((nseed, *expr.shape), dtype=np.float64))
    return ret
  if expr.op == ExprOp.INPUT:
    memo[expr.id] = ret = seeds if expr.id == wrt.id else Expr.const(np.zeros((nseed, *expr.shape), dtype=np.float64))
    return ret
  if expr.op == ExprOp.CONST:
    memo[expr.id] = ret = Expr.const(np.zeros((nseed, *expr.shape), dtype=np.float64))
    return ret
  if expr.op == ExprOp.SOLVER_CALL:
    raise NotImplementedError("active derivative through SOLVER_CALL is not implemented")
  if expr.op == ExprOp.SLICE:
    memo[expr.id] = ret = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo)[(slice(None), *expr.attrs["index"])]
    return ret
  if expr.op == ExprOp.RESHAPE:
    memo[expr.id] = ret = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo).reshape((nseed, *expr.shape))
    return ret
  if expr.op == ExprOp.PRINT:
    memo[expr.id] = ret = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo)
    return ret
  if expr.op == ExprOp.TRANSPOSE:
    if len(expr.shape) > 3:
      raise _JVPManyUnsupported(str(expr.op))  # seed axis would make a rank-5 TRANSPOSE, beyond the rank-4 lowering limit
    d0 = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo)
    memo[expr.id] = ret = d0.transpose((0, *(axis + 1 for axis in expr.attrs["axes"])))
    return ret
  if expr.op == ExprOp.ADD:
    d0 = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo)
    d1 = _jvp_many_structural(expr.args[1], wrt, seeds, memo, dep_memo)
    if _is_zero_const(d0) and expr.args[1].shape == expr.shape:
      memo[expr.id] = ret = d1
    elif _is_zero_const(d1) and expr.args[0].shape == expr.shape:
      memo[expr.id] = ret = d0
    else:
      memo[expr.id] = ret = _broadcast_tangent(d0, expr.args[0], expr, nseed) + _broadcast_tangent(d1, expr.args[1], expr, nseed)
    return ret
  if expr.op == ExprOp.SUB:
    d0 = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo)
    d1 = _jvp_many_structural(expr.args[1], wrt, seeds, memo, dep_memo)
    if _is_zero_const(d0) and expr.args[1].shape == expr.shape:
      memo[expr.id] = ret = -d1
    elif _is_zero_const(d1) and expr.args[0].shape == expr.shape:
      memo[expr.id] = ret = d0
    else:
      memo[expr.id] = ret = _broadcast_tangent(d0, expr.args[0], expr, nseed) - _broadcast_tangent(d1, expr.args[1], expr, nseed)
    return ret
  if expr.op == ExprOp.NEG:
    memo[expr.id] = ret = -_jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo)
    return ret
  if expr.op == ExprOp.CONCAT:
    memo[expr.id] = ret = concat([_jvp_many_structural(arg, wrt, seeds, memo, dep_memo) for arg in expr.args], axis=expr.attrs.get("axis", 0) + 1)
    return ret
  if expr.op == ExprOp.STACK:
    memo[expr.id] = ret = stack([_jvp_many_structural(arg, wrt, seeds, memo, dep_memo) for arg in expr.args], axis=expr.attrs.get("axis", 0) + 1)
    return ret
  if expr.op == ExprOp.GATHER:
    d0 = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo)
    indices = expr.attrs["indices"].reshape(-1)
    full = (np.arange(nseed, dtype=np.int64)[:, None] * expr.args[0].size + indices[None, :]).reshape(-1)
    memo[expr.id] = ret = gather(d0.reshape((nseed * expr.args[0].size,)), full).reshape((nseed, *expr.shape))
    return ret
  if expr.op == ExprOp.SCATTER:
    d0 = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo)
    indices = expr.attrs["indices"].reshape(-1)
    full = (np.arange(nseed, dtype=np.int64)[:, None] * expr.size + indices[None, :]).reshape(-1)
    memo[expr.id] = ret = scatter(d0.reshape((nseed * expr.args[0].size,)), full, (nseed * expr.size,)).reshape((nseed, *expr.shape))
    return ret
  if expr.op == ExprOp.SUM:
    d0 = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo)
    ones = Expr.const(np.ones(expr.args[0].size), dtype=d0.type.dtype)
    memo[expr.id] = ret = d0.reshape((nseed, expr.args[0].size)) @ ones
    return ret
  if expr.op == ExprOp.VMAP:
    memo[expr.id] = ret = _vmap_jvp_many(
      expr, wrt, seeds, memo, dep_memo, pushforward=_jvp, pushforward_many=_jvp_many_structural, unroll=_jvp_many_unrolled
    )
    return ret
  if expr.op == ExprOp.CALL:
    tangents = [simplify_cse_fixpoint(_jvp_many_structural(arg, wrt, seeds, memo, dep_memo)) for arg in expr.args]
    memo[expr.id] = ret = _call_jvp_many(expr, tangents, nseed, pushforward=_jvp, unroll=_jvp_many_unrolled)
    return ret

  args = expr.args
  d = [_jvp_many_structural(arg, wrt, seeds, memo, dep_memo) for arg in args]
  if expr.op == ExprOp.MUL:
    if args[0] is args[1]:
      memo[expr.id] = ret = _seed_axis(2 * args[0], nseed) * d[0]
      return ret
    memo[expr.id] = ret = _broadcast_tangent(d[0], args[0], expr, nseed) * _seed_axis(args[1], nseed, expr) + _seed_axis(
      args[0], nseed, expr
    ) * _broadcast_tangent(d[1], args[1], expr, nseed)
    return ret
  if expr.op == ExprOp.DIV:
    inv = _seed_axis(1.0 / args[1], nseed, expr)
    memo[expr.id] = ret = (
      _broadcast_tangent(d[0], args[0], expr, nseed) - _seed_axis(expr, nseed) * _broadcast_tangent(d[1], args[1], expr, nseed)
    ) * inv
    return ret
  if expr.op == ExprOp.POW:
    if args[1].op == ExprOp.CONST:
      memo[expr.id] = ret = _seed_axis(args[1] * (args[0] ** (args[1] - 1)), nseed, expr) * _broadcast_tangent(d[0], args[0], expr, nseed)
    else:
      x = _seed_axis(args[0], nseed, expr)
      y = _seed_axis(args[1], nseed, expr)
      dx = _broadcast_tangent(d[0], args[0], expr, nseed)
      dy = _broadcast_tangent(d[1], args[1], expr, nseed)
      memo[expr.id] = ret = _seed_axis(expr, nseed) * (dy * x.log() + y * dx / x)
    return ret
  if expr.op == ExprOp.SIN:
    memo[expr.id] = ret = _seed_axis(args[0].cos(), nseed) * d[0]
    return ret
  if expr.op == ExprOp.COS:
    memo[expr.id] = ret = -_seed_axis(args[0].sin(), nseed) * d[0]
    return ret
  if expr.op == ExprOp.TAN:
    memo[expr.id] = ret = d[0] / (_seed_axis(args[0].cos(), nseed) ** 2)
    return ret
  if expr.op == ExprOp.EXP:
    memo[expr.id] = ret = _seed_axis(expr, nseed) * d[0]
    return ret
  if expr.op == ExprOp.LOG:
    memo[expr.id] = ret = d[0] / _seed_axis(args[0], nseed)
    return ret
  if expr.op == ExprOp.SQRT:
    memo[expr.id] = ret = d[0] * _seed_axis(0.5 / expr, nseed)
    return ret
  if expr.op == ExprOp.TANH:
    memo[expr.id] = ret = d[0] * (1 - _seed_axis(expr * expr, nseed))
    return ret
  if expr.op == ExprOp.ERF:
    memo[expr.id] = ret = _seed_axis((2 / np.sqrt(np.pi)) * (-(args[0] ** 2)).exp(), nseed) * d[0]
    return ret
  if expr.op == ExprOp.COSH:
    memo[expr.id] = ret = _seed_axis(args[0].sinh(), nseed) * d[0]
    return ret
  if expr.op == ExprOp.SINH:
    memo[expr.id] = ret = _seed_axis(args[0].cosh(), nseed) * d[0]
    return ret
  if expr.op == ExprOp.MATMUL:
    if args[0] is args[1] and len(args[0].shape) == 1:
      memo[expr.id] = ret = 2 * (d[0] @ args[0])
      return ret
    ret: Expr | None = None
    if not _is_zero_const(d[0]):
      ret = _jvp_many_matmul_left(args[0], args[1], d[0], nseed)
    if not _is_zero_const(d[1]):
      term = _jvp_many_matmul_right(args[0], args[1], d[1], nseed)
      ret = term if ret is None else ret + term
    memo[expr.id] = ret = Expr.const(np.zeros((nseed, *expr.shape), dtype=np.float64)) if ret is None else ret
    return ret
  raise _JVPManyUnsupported(str(expr.op))


def _seed_axis(expr: Expr, nseed: int, output: Expr | None = None) -> Expr:
  # tangents carry a leading seed axis, so a stacked primal must also be rank-aligned against (nseed, *output.shape) when its rank is lower
  if expr.shape == ():
    return expr
  missing = 0 if output is None else len(output.shape) - len(expr.shape)
  t = stack([expr] * nseed, axis=0)
  return t if missing == 0 else t.reshape((nseed, *(1,) * missing, *expr.shape))


def _broadcast_tangent(tangent: Expr, operand: Expr, output: Expr, nseed: int) -> Expr:
  missing = len(output.shape) - len(operand.shape)
  return tangent if missing == 0 else tangent.reshape((nseed, *(1,) * missing, *operand.shape))


def _jvp_many_matmul_left(x: Expr, y: Expr, dx: Expr, nseed: int) -> Expr:
  if len(x.shape) == 2 and len(y.shape) == 1:
    return stack([dx[i] @ y for i in range(nseed)], axis=0)
  if len(x.shape) == 1 and len(y.shape) == 2:
    return dx @ y
  if len(x.shape) == 1 and len(y.shape) == 1:
    return dx @ y  # (nseed, n) @ (n,) -> (nseed,); a plain .sum() would also contract the seed axis
  return stack([dx[i] @ y for i in range(nseed)], axis=0)


def _jvp_many_matmul_right(x: Expr, y: Expr, dy: Expr, nseed: int) -> Expr:
  if len(x.shape) == 2 and len(y.shape) == 1:
    return dy @ x.T
  if len(x.shape) == 1 and len(y.shape) == 2:
    return stack([x @ dy[i] for i in range(nseed)], axis=0)
  if len(x.shape) == 1 and len(y.shape) == 1:
    return dy @ x  # (nseed, n) @ (n,) -> (nseed,); a plain .sum() would also contract the seed axis
  return stack([x @ dy[i] for i in range(nseed)], axis=0)
