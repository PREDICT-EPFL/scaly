from __future__ import annotations

import hashlib
from typing import Any, Iterable, Sequence

import numpy as np

from .expr import Expr, as_expr, concat, dot, gather, map_, scatter, stack, topo, zeros_like
from .ops import Ops


_CALL_JVP_CACHE: dict[tuple[int, int, int], tuple[Any, tuple[int, ...], bool]] = {}
_CALL_JVP_MANY_CACHE: dict[tuple[int, int, int, int], tuple[Any, tuple[int, ...], bool]] = {}
_CALL_JVP_MANY_CONST_CACHE: dict[tuple[int, int, int, tuple[int, ...], bytes], tuple[Any, tuple[int, ...], tuple[int, ...]]] = {}


class _JVPManyUnsupported(Exception):
  pass


def _is_zero_const(expr: Expr) -> bool:
  return expr.op == Ops.CONST and expr.value is not None and bool(np.all(expr.value == 0))


def _depends_on(expr: Expr, wrt: Expr, memo: dict[tuple[int, int], bool]) -> bool:
  key = (expr.id, wrt.id)
  if key not in memo:
    memo[key] = expr.id == wrt.id or any(_depends_on(arg, wrt, memo) for arg in expr.args)
  return memo[key]


def jvp(expr: Expr, wrt: Expr, seed: Expr) -> Expr:
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
  if expr.op == Ops.INPUT:
    memo[expr.id] = ret = seed if expr.id == wrt.id else zeros_like(expr)
    return ret
  if expr.op == Ops.CONST:
    memo[expr.id] = ret = zeros_like(expr)
    return ret
  if expr.op == Ops.CALL:
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
  if expr.op == Ops.MAP:
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
  if expr.op == Ops.SOLVER_CALL:
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
  if expr.op == Ops.NEG:
    return save(-d[0])
  if expr.op == Ops.ADD:
    return save(d[0] + d[1])
  if expr.op == Ops.SUB:
    return save(d[0] - d[1])
  if expr.op == Ops.MUL:
    return save(d[0] * args[1] + args[0] * d[1])
  if expr.op == Ops.DIV:
    return save((d[0] * args[1] - args[0] * d[1]) / (args[1] ** 2))
  if expr.op == Ops.POW:
    if args[1].op == Ops.CONST:
      return save(args[1] * (args[0] ** (args[1] - 1)) * d[0])
    return save(expr * (d[1] * args[0].log() + args[1] * d[0] / args[0]))
  if expr.op == Ops.SIN:
    return save(args[0].cos() * d[0])
  if expr.op == Ops.COS:
    return save(-args[0].sin() * d[0])
  if expr.op == Ops.TAN:
    return save(d[0] / (args[0].cos() ** 2))
  if expr.op == Ops.ASIN:
    return save(d[0] / (1 - args[0] ** 2).sqrt())
  if expr.op == Ops.ACOS:
    return save(-d[0] / (1 - args[0] ** 2).sqrt())
  if expr.op == Ops.ATAN:
    return save(d[0] / (1 + args[0] ** 2))
  if expr.op == Ops.ATAN2:
    y, x = args
    dy, dx = d
    return save((x * dy - y * dx) / (x * x + y * y))
  if expr.op == Ops.SINH:
    return save(args[0].cosh() * d[0])
  if expr.op == Ops.COSH:
    return save(args[0].sinh() * d[0])
  if expr.op == Ops.TANH:
    return save(d[0] * (1 - expr * expr))
  if expr.op == Ops.EXP:
    return save(expr * d[0])
  if expr.op == Ops.LOG:
    return save(d[0] / args[0])
  if expr.op == Ops.SQRT:
    return save(d[0] / (2 * expr))
  if expr.op == Ops.ABS:
    return save(args[0] / args[0].abs() * d[0])
  if expr.op in {Ops.FLOOR, Ops.CEIL, Ops.MINIMUM, Ops.MAXIMUM}:
    raise NotImplementedError(f"JVP for nonsmooth op {expr.op!r} is not implemented")
  if expr.op == Ops.SUM:
    return save(d[0].sum())
  if expr.op == Ops.RESHAPE:
    return save(d[0].reshape(expr.shape))
  if expr.op == Ops.TRANSPOSE:
    return save(d[0].transpose(expr.attrs["axes"]))
  if expr.op == Ops.SLICE:
    return save(Expr.const(d[0].value[expr.attrs["index"]]) if d[0].op == Ops.CONST and d[0].value is not None else d[0][expr.attrs["index"]])
  if expr.op == Ops.GATHER:
    return save(
      Expr.const(np.take(d[0].value.reshape(-1), expr.attrs["indices"]).reshape(expr.attrs["indices"].shape))
      if d[0].op == Ops.CONST and d[0].value is not None
      else gather(d[0], expr.attrs["indices"])
    )
  if expr.op == Ops.SCATTER:
    return save(scatter(d[0], expr.attrs["indices"], expr.shape))
  if expr.op == Ops.STACK:
    return save(stack(d, axis=expr.attrs.get("axis", 0)))
  if expr.op == Ops.CONCAT:
    return save(concat(d, axis=expr.attrs.get("axis", 0)))
  if expr.op == Ops.MATMUL:
    return save(d[0] @ args[1] + args[0] @ d[1])
  raise NotImplementedError(f"JVP for op {expr.op!r} is not implemented")


def _call_jvp_many_const_function(
  callee: Any, output_index: int, formal_index: int, seed_value: np.ndarray
) -> tuple[Any, tuple[int, ...], tuple[int, ...]]:
  key = (id(callee), output_index, formal_index, tuple(seed_value.shape), seed_value.tobytes())
  if key not in _CALL_JVP_MANY_CONST_CACHE:
    from .function import Function
    from .rewrite import simplify_cse_fixpoint

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
    _CALL_JVP_MANY_CONST_CACHE[key] = (fn, arg_indices, active)
  return _CALL_JVP_MANY_CONST_CACHE[key]


def _call_jvp_many_function(callee: Any, output_index: int, formal_index: int, nseed: int) -> tuple[Any, tuple[int, ...], bool]:
  key = (id(callee), output_index, formal_index, nseed)
  if key not in _CALL_JVP_MANY_CACHE:
    from .function import Function
    from .rewrite import simplify_cse_fixpoint

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
    _CALL_JVP_MANY_CACHE[key] = (fn, arg_indices, takes_seed)
  return _CALL_JVP_MANY_CACHE[key]


def _call_jvp_function(callee: Any, output_index: int, formal_index: int) -> tuple[Any, tuple[int, ...], bool]:
  key = (id(callee), output_index, formal_index)
  if key not in _CALL_JVP_CACHE:
    from .function import Function

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
    _CALL_JVP_CACHE[key] = (fn, arg_indices, takes_seed)
  return _CALL_JVP_CACHE[key]


def jvp_many(expr: Expr, wrt: Expr, seeds: Expr) -> Expr:
  if len(seeds.shape) < 1 or seeds.shape[1:] != wrt.shape:
    raise ValueError(f"multi-seed JVP expects seeds shape (nseed, *{wrt.shape}), got {seeds.shape}")
  if seeds.shape[0] == 0:
    return Expr.const(np.zeros((0, *expr.shape), dtype=np.float64))
  try:
    return _jvp_many_structural(expr, wrt, seeds, {}, {})
  except _JVPManyUnsupported:
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
  if expr.op == Ops.INPUT:
    memo[expr.id] = ret = seeds if expr.id == wrt.id else Expr.const(np.zeros((nseed, *expr.shape), dtype=np.float64))
    return ret
  if expr.op == Ops.CONST:
    memo[expr.id] = ret = Expr.const(np.zeros((nseed, *expr.shape), dtype=np.float64))
    return ret
  if expr.op == Ops.SLICE:
    memo[expr.id] = ret = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo)[(slice(None), *expr.attrs["index"])]
    return ret
  if expr.op == Ops.RESHAPE:
    memo[expr.id] = ret = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo).reshape((nseed, *expr.shape))
    return ret
  if expr.op == Ops.ADD:
    d0 = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo)
    d1 = _jvp_many_structural(expr.args[1], wrt, seeds, memo, dep_memo)
    memo[expr.id] = ret = d1 if _is_zero_const(d0) else d0 if _is_zero_const(d1) else d0 + d1
    return ret
  if expr.op == Ops.SUB:
    d0 = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo)
    d1 = _jvp_many_structural(expr.args[1], wrt, seeds, memo, dep_memo)
    memo[expr.id] = ret = -d1 if _is_zero_const(d0) else d0 if _is_zero_const(d1) else d0 - d1
    return ret
  if expr.op == Ops.NEG:
    memo[expr.id] = ret = -_jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo)
    return ret
  if expr.op == Ops.CONCAT:
    memo[expr.id] = ret = concat([_jvp_many_structural(arg, wrt, seeds, memo, dep_memo) for arg in expr.args], axis=expr.attrs.get("axis", 0) + 1)
    return ret
  if expr.op == Ops.STACK:
    memo[expr.id] = ret = stack([_jvp_many_structural(arg, wrt, seeds, memo, dep_memo) for arg in expr.args], axis=expr.attrs.get("axis", 0) + 1)
    return ret
  if expr.op == Ops.GATHER:
    d0 = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo)
    memo[expr.id] = ret = stack([gather(d0[i], expr.attrs["indices"]) for i in range(nseed)], axis=0)
    return ret
  if expr.op == Ops.SUM:
    d0 = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo)
    memo[expr.id] = ret = stack([d0[i].sum() for i in range(nseed)], axis=0)
    return ret
  if expr.op == Ops.MAP:
    from .rewrite import simplify_cse_fixpoint
    from .sparsity import _jac_mask, column_coloring
    from .types import SparsityType

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
  if expr.op == Ops.CALL:
    from .rewrite import simplify_cse_fixpoint

    ret: Expr | None = None
    for formal_idx, actual in enumerate(expr.args):
      actual_tan = simplify_cse_fixpoint(_jvp_many_structural(actual, wrt, seeds, memo, dep_memo))
      if _is_zero_const(actual_tan):
        continue
      if actual_tan.op == Ops.CONST and actual_tan.value is not None:
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
  if expr.op == Ops.MUL:
    memo[expr.id] = ret = d[0] * _seed_axis(args[1], nseed) + _seed_axis(args[0], nseed) * d[1]
    return ret
  if expr.op == Ops.DIV:
    y = _seed_axis(args[1], nseed)
    memo[expr.id] = ret = (d[0] * y - _seed_axis(args[0], nseed) * d[1]) / (y**2)
    return ret
  if expr.op == Ops.POW:
    if args[1].op == Ops.CONST:
      memo[expr.id] = ret = _seed_axis(args[1] * (args[0] ** (args[1] - 1)), nseed) * d[0]
    else:
      x = _seed_axis(args[0], nseed)
      y = _seed_axis(args[1], nseed)
      memo[expr.id] = ret = _seed_axis(expr, nseed) * (d[1] * x.log() + y * d[0] / x)
    return ret
  if expr.op == Ops.SIN:
    memo[expr.id] = ret = _seed_axis(args[0].cos(), nseed) * d[0]
    return ret
  if expr.op == Ops.COS:
    memo[expr.id] = ret = -_seed_axis(args[0].sin(), nseed) * d[0]
    return ret
  if expr.op == Ops.TAN:
    memo[expr.id] = ret = d[0] / (_seed_axis(args[0].cos(), nseed) ** 2)
    return ret
  if expr.op == Ops.EXP:
    memo[expr.id] = ret = _seed_axis(expr, nseed) * d[0]
    return ret
  if expr.op == Ops.LOG:
    memo[expr.id] = ret = d[0] / _seed_axis(args[0], nseed)
    return ret
  if expr.op == Ops.SQRT:
    memo[expr.id] = ret = d[0] / (2 * _seed_axis(expr, nseed))
    return ret
  if expr.op == Ops.TANH:
    memo[expr.id] = ret = d[0] * (1 - _seed_axis(expr * expr, nseed))
    return ret
  if expr.op == Ops.COSH:
    memo[expr.id] = ret = _seed_axis(args[0].sinh(), nseed) * d[0]
    return ret
  if expr.op == Ops.SINH:
    memo[expr.id] = ret = _seed_axis(args[0].cosh(), nseed) * d[0]
    return ret
  if expr.op == Ops.MATMUL:
    ret: Expr | None = None
    if not _is_zero_const(d[0]):
      ret = _jvp_many_matmul_left(args[0], args[1], d[0], nseed)
    if not _is_zero_const(d[1]):
      term = _jvp_many_matmul_right(args[0], args[1], d[1], nseed)
      ret = term if ret is None else ret + term
    memo[expr.id] = ret = Expr.const(np.zeros((nseed, *expr.shape), dtype=np.float64)) if ret is None else ret
    return ret
  raise _JVPManyUnsupported


def _seed_axis(expr: Expr, nseed: int) -> Expr:
  return expr if expr.shape == () else stack([expr] * nseed, axis=0)


def _jvp_many_matmul_left(x: Expr, y: Expr, dx: Expr, nseed: int) -> Expr:
  if len(x.shape) == 2 and len(y.shape) == 1:
    return stack([dx[i] @ y for i in range(nseed)], axis=0)
  if len(x.shape) == 1 and len(y.shape) == 2:
    return dx @ y
  if len(x.shape) == 1 and len(y.shape) == 1:
    return (dx * y).sum()
  return stack([dx[i] @ y for i in range(nseed)], axis=0)


def _jvp_many_matmul_right(x: Expr, y: Expr, dy: Expr, nseed: int) -> Expr:
  if len(x.shape) == 2 and len(y.shape) == 1:
    return dy @ x.T
  if len(x.shape) == 1 and len(y.shape) == 2:
    return stack([x @ dy[i] for i in range(nseed)], axis=0)
  if len(x.shape) == 1 and len(y.shape) == 1:
    return (x * dy).sum()
  return stack([x @ dy[i] for i in range(nseed)], axis=0)


def _substitute(expr: Expr, replacements: dict[int, Expr]) -> Expr:
  memo: dict[int, Expr] = {}
  for node in topo((expr,)):
    if node.id in replacements:
      memo[node.id] = replacements[node.id]
      continue
    args = tuple(memo[arg.id] for arg in node.args)
    memo[node.id] = (
      node
      if all(a is b for a, b in zip(args, node.args, strict=True))
      else Expr(node.op, args, node.type, node.name, node.value, dict(node.attrs), node.lowering)
    )
  return memo[expr.id]


def vjp(outputs: Sequence[Expr], wrts: Sequence[Expr], cotangents: Sequence[Expr]) -> tuple[Expr, ...]:
  if len(outputs) != len(cotangents):
    raise ValueError(f"expected {len(outputs)} cotangents, got {len(cotangents)}")
  adjoints: dict[int, Expr] = {}
  nodes = topo(outputs)
  expr_ids = {e.id for e in nodes}
  dep_memo: dict[tuple[int, int], bool] = {}

  def needed(expr: Expr) -> bool:
    return any(_depends_on(expr, wrt, dep_memo) for wrt in wrts)

  for out, cot in zip(outputs, cotangents, strict=True):
    if out.shape != cot.shape:
      raise ValueError(f"cotangent for output shape {out.shape} has shape {cot.shape}")
    adjoints[out.id] = cot if out.id not in adjoints else adjoints[out.id] + cot

  for expr in reversed(nodes):
    cot = adjoints.get(expr.id)
    if cot is None or expr.op in {Ops.INPUT, Ops.CONST} or not needed(expr):
      continue
    for arg, arg_cot in zip(expr.args, _local_vjp(expr, cot), strict=True):
      if arg.id in expr_ids:
        adjoints[arg.id] = arg_cot if arg.id not in adjoints else adjoints[arg.id] + arg_cot

  return tuple(adjoints.get(wrt.id, zeros_like(wrt)) for wrt in wrts)


def vjp_many(outputs: Sequence[Expr], wrts: Sequence[Expr], cotangents: Sequence[Expr]) -> tuple[Expr, ...]:
  if len(outputs) != len(cotangents):
    raise ValueError(f"expected {len(outputs)} cotangents, got {len(cotangents)}")
  nseed: int | None = None
  for out, cot in zip(outputs, cotangents, strict=True):
    if len(cot.shape) < 1 or cot.shape[1:] != out.shape:
      raise ValueError(f"multi-seed VJP expects cotangent shape (nseed, *{out.shape}), got {cot.shape}")
    if nseed is None:
      nseed = cot.shape[0]
    elif cot.shape[0] != nseed:
      raise ValueError(f"all VJP cotangents must have the same leading seed axis, got {nseed} and {cot.shape[0]}")
  nseed = 0 if nseed is None else nseed
  if nseed == 0:
    return tuple(Expr.const(np.zeros((0, *wrt.shape), dtype=np.float64)) for wrt in wrts)

  per_seed = [vjp(outputs, wrts, tuple(cot[i] for cot in cotangents)) for i in range(nseed)]
  return tuple(stack([seed_grads[i] for seed_grads in per_seed], axis=0) for i in range(len(wrts)))


def _local_vjp(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  if expr.op == Ops.NEG:
    return (-cot,)
  if expr.op == Ops.ADD:
    return (_unbroadcast(cot, args[0].shape, expr.shape), _unbroadcast(cot, args[1].shape, expr.shape))
  if expr.op == Ops.SUB:
    return (_unbroadcast(cot, args[0].shape, expr.shape), _unbroadcast(-cot, args[1].shape, expr.shape))
  if expr.op == Ops.MUL:
    return (_unbroadcast(cot * args[1], args[0].shape, expr.shape), _unbroadcast(cot * args[0], args[1].shape, expr.shape))
  if expr.op == Ops.DIV:
    return (
      _unbroadcast(cot / args[1], args[0].shape, expr.shape),
      _unbroadcast(-cot * args[0] / (args[1] ** 2), args[1].shape, expr.shape),
    )
  if expr.op == Ops.POW:
    return (
      _unbroadcast(cot * args[1] * (args[0] ** (args[1] - 1)), args[0].shape, expr.shape),
      _unbroadcast(cot * expr * args[0].log(), args[1].shape, expr.shape),
    )
  if expr.op == Ops.SIN:
    return (cot * args[0].cos(),)
  if expr.op == Ops.COS:
    return (-cot * args[0].sin(),)
  if expr.op == Ops.TAN:
    return (cot / (args[0].cos() ** 2),)
  if expr.op == Ops.ASIN:
    return (cot / (1 - args[0] ** 2).sqrt(),)
  if expr.op == Ops.ACOS:
    return (-cot / (1 - args[0] ** 2).sqrt(),)
  if expr.op == Ops.ATAN:
    return (cot / (1 + args[0] ** 2),)
  if expr.op == Ops.ATAN2:
    y, x = args
    denom = x * x + y * y
    return (_unbroadcast(cot * x / denom, y.shape, expr.shape), _unbroadcast(-cot * y / denom, x.shape, expr.shape))
  if expr.op == Ops.SINH:
    return (cot * args[0].cosh(),)
  if expr.op == Ops.COSH:
    return (cot * args[0].sinh(),)
  if expr.op == Ops.TANH:
    return (cot * (1 - expr * expr),)
  if expr.op == Ops.EXP:
    return (cot * expr,)
  if expr.op == Ops.LOG:
    return (cot / args[0],)
  if expr.op == Ops.SQRT:
    return (cot / (2 * expr),)
  if expr.op == Ops.ABS:
    return (cot * args[0] / args[0].abs(),)
  if expr.op in {Ops.FLOOR, Ops.CEIL, Ops.MINIMUM, Ops.MAXIMUM}:
    raise NotImplementedError(f"VJP for nonsmooth op {expr.op!r} is not implemented")
  if expr.op == Ops.SUM:
    return (cot * _ones_like(args[0]),)
  if expr.op == Ops.RESHAPE:
    return (cot.reshape(args[0].shape),)
  if expr.op == Ops.TRANSPOSE:
    axes = expr.attrs["axes"]
    inv = tuple(int(np.argsort(axes)[i]) for i in range(len(axes)))
    return (cot.transpose(inv),)
  if expr.op == Ops.SLICE:
    indices = np.arange(args[0].size).reshape(args[0].shape)[expr.attrs["index"]]
    return (scatter(cot, indices, args[0].shape),)
  if expr.op == Ops.GATHER:
    return (_gather_vjp(cot, expr.attrs["indices"], args[0].shape),)
  if expr.op == Ops.SCATTER:
    return (gather(cot, expr.attrs["indices"]),)
  if expr.op == Ops.STACK:
    return _stack_vjp(cot, len(args), expr.attrs.get("axis", 0))
  if expr.op == Ops.CONCAT:
    return _concat_vjp(cot, args, expr.attrs.get("axis", 0))
  if expr.op == Ops.MATMUL:
    return _matmul_vjp(args[0], args[1], cot)
  if expr.op == Ops.CALL:
    callee = expr.attrs["callee"]
    callee_out = callee.outputs[expr.attrs["output"]]
    replacements = dict(zip((inp.id for inp in callee.inputs), args, strict=True))
    return tuple(_substitute(g, replacements) for g in vjp((callee_out,), callee.inputs, (cot,)))
  if expr.op == Ops.SOLVER_CALL:
    # Non-differentiable: every arg cotangent is zero. See the matching JVP rule.
    return tuple(zeros_like(arg) for arg in args)
  raise NotImplementedError(f"VJP for op {expr.op!r} is not implemented")


def _ones_like(expr: Expr) -> Expr:
  return Expr.const(np.ones(expr.shape, dtype=np.float64), lowering=expr.lowering)


def _sum_exprs(exprs: Iterable[Expr]) -> Expr:
  ret: Expr | None = None
  for expr in exprs:
    ret = expr if ret is None else ret + expr
  return as_expr(0.0) if ret is None else ret


def _unbroadcast(cot: Expr, in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> Expr:
  if in_shape == out_shape:
    return cot
  if not in_shape:
    return cot.sum()
  source = np.arange(int(np.prod(in_shape, dtype=int))).reshape(in_shape)
  source = np.broadcast_to(source, out_shape).reshape(-1)
  vals = []
  for i in range(int(np.prod(in_shape, dtype=int))):
    vals.append(gather(cot, np.nonzero(source == i)[0]).sum())
  return stack(vals).reshape(in_shape)


def _gather_vjp(cot: Expr, indices: np.ndarray, shape: tuple[int, ...]) -> Expr:
  flat = indices.reshape(-1)
  vals = []
  for i in range(int(np.prod(shape, dtype=int))):
    positions = np.nonzero(flat == i)[0]
    vals.append(gather(cot, positions).sum() if positions.size else as_expr(0.0))
  return stack(vals).reshape(shape)


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
    gx = stack([stack([cot[i] * y[j] for j in range(x.shape[1])]) for i in range(x.shape[0])])
    gy = stack([_sum_exprs(cot[i] * x[i, j] for i in range(x.shape[0])) for j in range(y.shape[0])])
    return gx, gy
  if len(x.shape) == 1 and len(y.shape) == 2:
    gx = stack([_sum_exprs(cot[j] * y[i, j] for j in range(y.shape[1])) for i in range(x.shape[0])])
    gy = stack([stack([x[i] * cot[j] for j in range(y.shape[1])]) for i in range(y.shape[0])])
    return gx, gy
  if len(x.shape) == 2 and len(y.shape) == 2:
    gx = stack([stack([_sum_exprs(cot[i, j] * y[k, j] for j in range(y.shape[1])) for k in range(x.shape[1])]) for i in range(x.shape[0])])
    gy = stack([stack([_sum_exprs(x[i, k] * cot[i, j] for i in range(x.shape[0])) for j in range(y.shape[1])]) for k in range(y.shape[0])])
    return gx, gy
  raise NotImplementedError(f"matmul VJP for {x.shape} @ {y.shape} is not implemented")


def basis(shape: tuple[int, ...], index: int) -> Expr:
  arr = np.zeros(shape, dtype=np.float64).reshape(-1)
  arr[index] = 1.0
  return Expr.const(arr.reshape(shape))


def jacobian(expr: Expr, wrt: Expr) -> Expr:
  if wrt.size == 0:
    return Expr.const(np.zeros((expr.size, 0), dtype=np.float64))
  # Batched forward AD: stack the wrt.size identity columns as a (wrt.size, *wrt.shape) seed and
  # push them through jvp_many. The structural multi-seed rules share cos/sin/exp across columns
  # and turn per-column chain-rule unrolls into small matmuls. Falls back to column-by-column jvp
  # only if jvp_many hits an unsupported op. Output is reshaped from (wrt.size, expr.size) →
  # (expr.size, wrt.size) so column j of the Jacobian = partial expr / partial wrt[j].
  from .rewrite import simplify_cse_fixpoint

  seed_arr = np.eye(wrt.size, dtype=np.float64).reshape((wrt.size, *wrt.shape))
  return simplify_cse_fixpoint(jvp_many(expr, wrt, Expr.const(seed_arr)).reshape((wrt.size, expr.size)).transpose((1, 0)))


def gradient(expr: Expr, wrt: Expr) -> Expr:
  if expr.size != 1:
    raise ValueError("gradient expects a scalar expression")
  return vjp((expr,), (wrt,), (_ones_like(expr),))[0]


def hessian(expr: Expr, wrt: Expr) -> Expr:
  return jacobian(gradient(expr, wrt).reshape((wrt.size,)), wrt)


def linear_combination(terms: dict[str, Expr], multipliers: dict[str, Expr], names: list[str]) -> Expr:
  out: Expr | None = None
  for name in names:
    term = dot(multipliers[name], terms[name])
    out = term if out is None else out + term
  return as_expr(0.0) if out is None else out


def finite_difference(fun: Any, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
  x = np.asarray(x, dtype=np.float64)
  y0 = np.asarray(fun(x), dtype=np.float64).reshape(-1)
  jac = np.empty((y0.size, x.size), dtype=np.float64)
  flat = x.reshape(-1)
  for i in range(flat.size):
    xp = flat.copy()
    xm = flat.copy()
    xp[i] += eps
    xm[i] -= eps
    jac[:, i] = (np.asarray(fun(xp.reshape(x.shape))).reshape(-1) - np.asarray(fun(xm.reshape(x.shape))).reshape(-1)) / (2 * eps)
  return jac
