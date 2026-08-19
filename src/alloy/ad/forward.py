"""Forward-mode AD: single-seed ``jvp`` and multi-seed ``jvp_many``.

``jvp_many`` has structural rules that share work across seeds; when an op has none it falls back
to unrolling ``jvp`` per seed unless ``ALLOY_STRICT_JVP_MANY`` forbids it.
"""

from __future__ import annotations

import hashlib
import weakref
from typing import Any

import numpy as np

from ..function import Function
from ..function.sugar import map_
from ..ir.expr import Expr, ExprOp, concat, gather, scatter, stack, zeros_like
from ..ir.types import SparsityType
from ..passes.expr import simplify_cse_fixpoint
from ..utils.env import env_bool
from .sparsity import _depends_on, _jac_mask, column_coloring


# Cache derivative helper Functions per live callee object. Do not key by ``id(callee)``:
# CPython may reuse ids after a short-lived Function is collected, which can splice a stale
# call-JVP helper into a different graph under xdist/CI-sized test runs.
_CALL_JVP_CACHE: weakref.WeakKeyDictionary[Any, dict[tuple[int, int], tuple[Any, tuple[int, ...], bool]]] = weakref.WeakKeyDictionary()
_CALL_JVP_MANY_CACHE: weakref.WeakKeyDictionary[Any, dict[tuple[int, int, int], tuple[Any, tuple[int, ...], bool]]] = weakref.WeakKeyDictionary()
_CALL_JVP_MANY_CONST_CACHE: weakref.WeakKeyDictionary[
  Any, dict[tuple[int, int, tuple[int, ...], bytes], tuple[Any, tuple[int, ...], tuple[int, ...]]]
] = weakref.WeakKeyDictionary()


class _JVPManyUnsupported(Exception):
  def __init__(self, op: str):
    super().__init__(op)
    self.op = op


def _is_zero_const(expr: Expr) -> bool:
  return expr.op == ExprOp.CONST and expr.value is not None and bool(np.all(expr.value == 0))


def jvp(expr: Expr, wrt: Expr, seed: Expr) -> Expr:
  """Forward-mode derivative: ``J(expr, wrt) @ seed``, with ``seed`` shaped like ``wrt``.

  One pass per seed. For many seeds at once use ``jvp_many``, which shares the expensive work.
  """
  return _jvp(expr, wrt, seed, {}, {})


def _jvp(expr: Expr, wrt: Expr, seed: Expr, memo: dict[int, Expr], dep_memo: dict[tuple[int, int], bool]) -> Expr:
  if expr.id in memo:
    return memo[expr.id]
  if _is_zero_const(seed):
    memo[expr.id] = ret = zeros_like(expr)
    return ret
  if not _depends_on(expr, wrt, dep_memo):
    memo[expr.id] = ret = zeros_like(expr)
    return ret
  if expr.op == ExprOp.INPUT:
    memo[expr.id] = ret = seed if expr.id == wrt.id else zeros_like(expr)
    return ret
  if expr.op == ExprOp.CONST:
    memo[expr.id] = ret = zeros_like(expr)
    return ret
  if expr.op == ExprOp.CALL:
    callee = expr.attrs["callee"]
    ret: Expr | None = None
    for formal_idx, actual in enumerate(expr.args):
      actual_tan = _jvp(actual, wrt, seed, memo, dep_memo)
      if _is_zero_const(actual_tan):
        continue
      jvp_fn, arg_indices, takes_seed = _call_jvp_function(callee, expr.attrs["output"], formal_idx)
      call_args = [expr.args[i] for i in arg_indices]
      if takes_seed:
        call_args.append(actual_tan)
      term = jvp_fn.call(call_args)[0]
      ret = term if ret is None else ret + term
    memo[expr.id] = ret = zeros_like(expr) if ret is None else ret
    return ret
  if expr.op == ExprOp.MAP:
    callee = expr.attrs["callee"]
    output_idx = expr.attrs["output"]
    length = expr.attrs["length"]
    starts = expr.attrs["starts"]
    strides = expr.attrs["strides"]
    ret: Expr | None = None
    for formal_idx, actual_outer in enumerate(expr.args):
      actual_tan = _jvp(actual_outer, wrt, seed, memo, dep_memo)
      if _is_zero_const(actual_tan):
        continue
      jvp_fn, arg_indices, takes_seed = _call_jvp_function(callee, output_idx, formal_idx)
      if not takes_seed:
        continue
      primal_specs = [(expr.args[i], starts[i], strides[i]) for i in arg_indices]
      seed_spec = (actual_tan, starts[formal_idx], strides[formal_idx])
      term = map_(jvp_fn, length, [*primal_specs, seed_spec])
      ret = term if ret is None else ret + term
    memo[expr.id] = ret = zeros_like(expr) if ret is None else ret
    return ret
  if expr.op == ExprOp.SOLVER_CALL:
    # Solver outputs are treated as non-differentiable today. Implicit
    # function theorem AD (e.g. cyipopt-style adjoint through KKT residuals)
    # is future work; for now any JVP through a solver returns zero.
    memo[expr.id] = ret = zeros_like(expr)
    return ret

  def save(ret: Expr) -> Expr:
    memo[expr.id] = ret
    return ret

  args = expr.args
  d = [_jvp(a, wrt, seed, memo, dep_memo) for a in args]
  if expr.op == ExprOp.NEG:
    return save(-d[0])
  if expr.op == ExprOp.ADD:
    return save(d[0] + d[1])
  if expr.op == ExprOp.SUB:
    return save(d[0] - d[1])
  if expr.op == ExprOp.MUL:
    return save(d[0] * args[1] + args[0] * d[1])
  if expr.op == ExprOp.DIV:
    return save((d[0] * args[1] - args[0] * d[1]) / (args[1] ** 2))
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
  if expr.op == ExprOp.EXP:
    return save(expr * d[0])
  if expr.op == ExprOp.LOG:
    return save(d[0] / args[0])
  if expr.op == ExprOp.SQRT:
    return save(d[0] / (2 * expr))
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
    return save(d[0] @ args[1] + args[0] @ d[1])
  raise NotImplementedError(f"JVP for op {expr.op!r} is not implemented")


def _call_jvp_many_const_function(
  callee: Any, output_index: int, formal_index: int, seed_value: np.ndarray
) -> tuple[Any, tuple[int, ...], tuple[int, ...]]:
  key = (output_index, formal_index, tuple(seed_value.shape), seed_value.tobytes())
  cache = _CALL_JVP_MANY_CONST_CACHE.setdefault(callee, {})
  if key not in cache:
    formal = callee.inputs[formal_index]
    out = callee.outputs[output_index]
    flat_seed = seed_value.reshape((seed_value.shape[0], -1))
    active = tuple(int(i) for i in np.nonzero(np.any(flat_seed != 0, axis=1))[0])
    seed = Expr.const(seed_value[list(active)])
    deriv = simplify_cse_fixpoint(_jvp_many_unrolled(out, formal, seed))
    dep_memo: dict[tuple[int, int], bool] = {}
    arg_indices = tuple(i for i, inp in enumerate(callee.inputs) if _depends_on(deriv, inp, dep_memo))
    inputs = tuple(callee.inputs[i] for i in arg_indices)
    input_names = tuple(callee.input_names[i] for i in arg_indices)
    seed_hash = hashlib.sha1(seed_value.tobytes()).hexdigest()[:10]
    name = f"{callee.name}_fwd{seed_value.shape[0]}c{seed_hash}_{callee.output_names[output_index]}_{callee.input_names[formal_index]}"
    fn = Function(name, inputs, [deriv], input_names, [f"fwd:{callee.output_names[output_index]}:{callee.input_names[formal_index]}"])
    cache[key] = (fn, arg_indices, active)
  return cache[key]


def _call_jvp_many_function(callee: Any, output_index: int, formal_index: int, nseed: int) -> tuple[Any, tuple[int, ...], bool]:
  key = (output_index, formal_index, nseed)
  cache = _CALL_JVP_MANY_CACHE.setdefault(callee, {})
  if key not in cache:
    formal = callee.inputs[formal_index]
    seed = Expr.sym(f"fwd:{callee.input_names[formal_index]}", (nseed, *formal.shape))
    out = callee.outputs[output_index]
    deriv = simplify_cse_fixpoint(_jvp_many_unrolled(out, formal, seed))
    dep_memo: dict[tuple[int, int], bool] = {}
    arg_indices = tuple(i for i, inp in enumerate(callee.inputs) if _depends_on(deriv, inp, dep_memo))
    takes_seed = _depends_on(deriv, seed, dep_memo)
    inputs = tuple(callee.inputs[i] for i in arg_indices) + ((seed,) if takes_seed else ())
    input_names = tuple(callee.input_names[i] for i in arg_indices) + ((seed.name,) if takes_seed else ())
    name = f"{callee.name}_fwd{nseed}_{callee.output_names[output_index]}_{callee.input_names[formal_index]}"
    fn = Function(name, inputs, [deriv], input_names, [f"fwd:{callee.output_names[output_index]}:{callee.input_names[formal_index]}"])
    cache[key] = (fn, arg_indices, takes_seed)
  return cache[key]


def _call_jvp_function(callee: Any, output_index: int, formal_index: int) -> tuple[Any, tuple[int, ...], bool]:
  key = (output_index, formal_index)
  cache = _CALL_JVP_CACHE.setdefault(callee, {})
  if key not in cache:
    formal = callee.inputs[formal_index]
    seed = Expr.sym(f"fwd:{callee.input_names[formal_index]}", formal.shape)
    out = callee.outputs[output_index]
    deriv = jvp(out, formal, seed)
    dep_memo: dict[tuple[int, int], bool] = {}
    arg_indices = tuple(i for i, inp in enumerate(callee.inputs) if _depends_on(deriv, inp, dep_memo))
    takes_seed = _depends_on(deriv, seed, dep_memo)
    inputs = tuple(callee.inputs[i] for i in arg_indices) + ((seed,) if takes_seed else ())
    input_names = tuple(callee.input_names[i] for i in arg_indices) + ((seed.name,) if takes_seed else ())
    name = f"{callee.name}_fwd_{callee.output_names[output_index]}_{callee.input_names[formal_index]}"
    fn = Function(name, inputs, [deriv], input_names, [f"fwd:{callee.output_names[output_index]}:{callee.input_names[formal_index]}"])
    cache[key] = (fn, arg_indices, takes_seed)
  return cache[key]


def jvp_many(expr: Expr, wrt: Expr, seeds: Expr) -> Expr:
  """Forward mode over several seeds in one pass. ``seeds`` has shape ``(n, *wrt.shape)``.

  Structural rules share work across seeds — one ``cos`` serves every column of a ``sin``'s
  derivative — so this is much cheaper than ``n`` separate ``jvp`` calls. Ops without a
  multi-seed rule fall back to per-seed evaluation; set ``ALLOY_STRICT_JVP_MANY=1`` to raise
  instead of falling back. Returns shape ``(n, *expr.shape)``.
  """
  if len(seeds.shape) < 1 or seeds.shape[1:] != wrt.shape:
    raise ValueError(f"multi-seed JVP expects seeds shape (nseed, *{wrt.shape}), got {seeds.shape}")
  if seeds.shape[0] == 0:
    return Expr.const(np.zeros((0, *expr.shape), dtype=np.float64))
  strict = env_bool("ALLOY_STRICT_JVP_MANY", False)
  try:
    ret = _jvp_many_structural(expr, wrt, seeds, {}, {})
  except _JVPManyUnsupported as unsupported:
    if strict:
      raise NotImplementedError(
        f"structural jvp_many does not support {unsupported.op!r}; ALLOY_STRICT_JVP_MANY=1 forbids the unrolled fallback"
      ) from unsupported
    return _jvp_many_unrolled(expr, wrt, seeds)
  expected = (seeds.shape[0], *expr.shape)
  if ret.shape == expected:
    return ret
  if strict:
    raise NotImplementedError(
      f"structural jvp_many returned shape {ret.shape} for {expr.op!r}, expected {expected}; ALLOY_STRICT_JVP_MANY=1 forbids the unrolled fallback"
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
  if expr.op == ExprOp.SLICE:
    memo[expr.id] = ret = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo)[(slice(None), *expr.attrs["index"])]
    return ret
  if expr.op == ExprOp.RESHAPE:
    memo[expr.id] = ret = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo).reshape((nseed, *expr.shape))
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
    memo[expr.id] = ret = stack([d0[i].sum() for i in range(nseed)], axis=0)
    return ret
  if expr.op == ExprOp.MAP:
    callee = expr.attrs["callee"]
    output_idx = expr.attrs["output"]
    length = expr.attrs["length"]
    starts = expr.attrs["starts"]
    strides = expr.attrs["strides"]
    slice_size = expr.attrs["slice_size"]
    callee_out = callee.outputs[output_idx]

    ret: Expr | None = None
    for formal_idx, actual_outer in enumerate(expr.args):
      actual_tan = simplify_cse_fixpoint(_jvp_many_structural(actual_outer, wrt, seeds, memo, dep_memo))
      if _is_zero_const(actual_tan):
        continue
      formal = callee.inputs[formal_idx]
      formal_size = formal.size
      outer_size = actual_outer.size
      start = starts[formal_idx]
      stride = strides[formal_idx]

      # Local coloring of the callee's Jacobian tile w.r.t. this formal. When this gives c_f < nseed
      # local colors, baking those local seeds into the per-iteration JVP callee yields a body of
      # size O(c_f) instead of O(nseed). The global compressed JVP is then assembled back by, for
      # each local color `c_local`, gathering the unique nonzero column of `actual_tan` for each
      # output row and multiplying with the per-iter compressed-JVP slice.
      local_mask = _jac_mask(callee_out, formal, {})
      local_colors = column_coloring(SparsityType.from_mask(local_mask)) if local_mask.any() else ()
      use_local = bool(local_colors) and (max(local_colors) + 1) < nseed
      if use_local:
        c_f = max(local_colors) + 1
        seed_f = np.zeros((c_f, formal_size), dtype=np.float64)
        for j, c in enumerate(local_colors):
          seed_f[c, j] = 1.0
        seed_f_shaped = seed_f.reshape((c_f, *formal.shape)) if formal.shape != (formal.size,) else seed_f
        inner_fn, primal_arg_indices, active = _call_jvp_many_const_function(callee, output_idx, formal_idx, seed_f_shaped)
        active_count = len(active)
        if active_count == 0:
          continue
        # For each local color, pick a representative formal column j whose Jacobian row k has a
        # nonzero. By the coloring property, that nonzero j is unique within the color group, so the
        # column's value in actual_tan is the full contribution at row k. If no j in the color group
        # has a nonzero at row k, the per-iter JVP entry is zero and the choice of j is irrelevant.
        local_colors_arr = np.asarray(local_colors, dtype=np.int64)
        unique_j = np.zeros((c_f, slice_size), dtype=np.int64)
        for c in range(c_f):
          cols = np.flatnonzero(local_colors_arr == c)
          for k in range(slice_size):
            for j in cols:
              if local_mask[k, j]:
                unique_j[c, k] = int(j)
                break
        primal_specs = [(expr.args[i], starts[i], strides[i]) for i in primal_arg_indices]
        mapped_flat = map_(inner_fn, length, primal_specs)
        mapped_3d = mapped_flat.reshape((length, active_count, slice_size))
        c_arr = np.arange(nseed, dtype=np.int64).reshape(nseed, 1, 1)
        it_arr = np.arange(length, dtype=np.int64).reshape(1, length, 1)
        for pos, c_local in enumerate(active):
          uj_arr = unique_j[c_local].reshape(1, 1, slice_size)
          flat_idx = (c_arr * outer_size + start + it_arr * stride + uj_arr).reshape(-1)
          gathered = gather(actual_tan, flat_idx).reshape((nseed, length, slice_size))
          mapped_slice = mapped_3d[:, pos, :].reshape((1, length, slice_size))
          contribution = (gathered * mapped_slice).reshape((nseed, length * slice_size))
          ret = contribution if ret is None else ret + contribution
        continue

      inner_fn, primal_arg_indices, takes_seed = _call_jvp_many_function(callee, output_idx, formal_idx, nseed)
      if not takes_seed:
        continue
      # Generic-seed path: build a per-iter seed buffer tile out of actual_tan and pass it as the
      # callee's seed input. The JVP callee body is unrolled across all nseed colors.
      it_arr = np.arange(length, dtype=np.int64)[:, None, None]
      c_arr = np.arange(nseed, dtype=np.int64)[None, :, None]
      j_arr = np.arange(formal_size, dtype=np.int64)[None, None, :]
      tile_indices = (c_arr * outer_size + start + it_arr * stride + j_arr).reshape(-1)
      seed_buffer = gather(actual_tan, tile_indices)
      primal_specs = [(expr.args[i], starts[i], strides[i]) for i in primal_arg_indices]
      seed_spec = (seed_buffer, 0, nseed * formal_size)
      mapped_flat = map_(inner_fn, length, [*primal_specs, seed_spec])
      term = mapped_flat.reshape((length, nseed, slice_size)).transpose((1, 0, 2)).reshape((nseed, length * slice_size))
      ret = term if ret is None else ret + term
    memo[expr.id] = ret = Expr.const(np.zeros((nseed, *expr.shape), dtype=np.float64)) if ret is None else ret
    return ret
  if expr.op == ExprOp.CALL:
    ret: Expr | None = None
    for formal_idx, actual in enumerate(expr.args):
      actual_tan = simplify_cse_fixpoint(_jvp_many_structural(actual, wrt, seeds, memo, dep_memo))
      if _is_zero_const(actual_tan):
        continue
      if actual_tan.op == ExprOp.CONST and actual_tan.value is not None:
        jvp_fn, arg_indices, active = _call_jvp_many_const_function(expr.attrs["callee"], expr.attrs["output"], formal_idx, actual_tan.value)
        active_term = jvp_fn.call([expr.args[i] for i in arg_indices])[0]
        if len(active) == nseed:
          term = active_term
        else:
          rows: list[Expr] = []
          active_pos = {row: i for i, row in enumerate(active)}
          zero = zeros_like(expr)
          for row in range(nseed):
            rows.append(active_term[active_pos[row]] if row in active_pos else zero)
          term = stack(rows, axis=0)
      else:
        jvp_fn, arg_indices, takes_seed = _call_jvp_many_function(expr.attrs["callee"], expr.attrs["output"], formal_idx, nseed)
        call_args = [expr.args[i] for i in arg_indices]
        if takes_seed:
          call_args.append(actual_tan)
        term = jvp_fn.call(call_args)[0]
      ret = term if ret is None else ret + term
    memo[expr.id] = ret = Expr.const(np.zeros((nseed, *expr.shape), dtype=np.float64)) if ret is None else ret
    return ret

  args = expr.args
  d = [_jvp_many_structural(arg, wrt, seeds, memo, dep_memo) for arg in args]
  if expr.op == ExprOp.MUL:
    memo[expr.id] = ret = _broadcast_tangent(d[0], args[0], expr, nseed) * _seed_axis(args[1], nseed, expr) + _seed_axis(
      args[0], nseed, expr
    ) * _broadcast_tangent(d[1], args[1], expr, nseed)
    return ret
  if expr.op == ExprOp.DIV:
    y = _seed_axis(args[1], nseed, expr)
    memo[expr.id] = ret = (
      _broadcast_tangent(d[0], args[0], expr, nseed) * y - _seed_axis(args[0], nseed, expr) * _broadcast_tangent(d[1], args[1], expr, nseed)
    ) / (y**2)
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
    memo[expr.id] = ret = d[0] / (2 * _seed_axis(expr, nseed))
    return ret
  if expr.op == ExprOp.TANH:
    memo[expr.id] = ret = d[0] * (1 - _seed_axis(expr * expr, nseed))
    return ret
  if expr.op == ExprOp.COSH:
    memo[expr.id] = ret = _seed_axis(args[0].sinh(), nseed) * d[0]
    return ret
  if expr.op == ExprOp.SINH:
    memo[expr.id] = ret = _seed_axis(args[0].cosh(), nseed) * d[0]
    return ret
  if expr.op == ExprOp.MATMUL:
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
