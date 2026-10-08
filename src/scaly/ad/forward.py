"""Forward-mode AD: single-seed ``jvp`` and multi-seed ``jvp_many``.

``jvp_many`` has structural rules that share work across seeds; when an op has none it falls back
to unrolling ``jvp`` per seed unless ``SCALY_STRICT_JVP_MANY`` forbids it.
"""

from __future__ import annotations

import hashlib
import weakref
from typing import Any

import numpy as np

from ..function.concrete import ConcreteFunction
from ..function.sugar import _mapped_call
from ..ir.expr import Expr, ExprOp, concat, gather, scatter, stack, substitute, zeros_like
from ..passes.expr import simplify_cse_fixpoint
from ..utils.env import env_bool
from .sparsity import _depends_on, _jac_mask, _mask_sparsity, column_coloring


# Cache derivative helper Functions per live callee object. Do not key by ``id(callee)``:
# CPython may reuse ids after a short-lived ConcreteFunction is collected, which can splice a stale
# call-JVP helper into a different graph under xdist/CI-sized test runs.
_CALL_JVP_CACHE: weakref.WeakKeyDictionary[Any, dict[tuple[int, tuple[int, ...]], tuple[Any, tuple[int, ...], tuple[int, ...]]]] = (
  weakref.WeakKeyDictionary()
)
_CALL_JVP_MANY_CACHE: weakref.WeakKeyDictionary[Any, dict[tuple[Any, ...], tuple[Any, tuple[int, ...], tuple[int, ...], tuple[int, ...]]]] = (
  weakref.WeakKeyDictionary()
)
_CALL_JVP_PACK_CACHE: weakref.WeakKeyDictionary[Any, dict[tuple[Any, ...], Any]] = weakref.WeakKeyDictionary()


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
    active = tuple(i for i, tangent in enumerate(tangents) if not _is_zero_const(tangent))
    if not active:
      memo[expr.id] = ret = zeros_like(expr)
      return ret
    fn, arg_indices, seed_indices = _call_jvp_function(expr.attrs["callee"], expr.attrs["output"], active)
    if expr.op == ExprOp.CALL:
      call_args = [expr.args[i] for i in arg_indices] + [tangents[i] for i in seed_indices]
      memo[expr.id] = ret = fn.symbolic_call(*fn.input_tree.unflatten(tuple(call_args)))
    elif not seed_indices:
      memo[expr.id] = ret = zeros_like(expr)
    else:
      starts, strides = expr.attrs["starts"], expr.attrs["strides"]
      specs = [(expr.args[i], starts[i], strides[i]) for i in arg_indices] + [(tangents[i], starts[i], strides[i]) for i in seed_indices]
      memo[expr.id] = ret = _mapped_call(fn, expr.attrs["length"], specs)
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


def _call_jvp_many_function(
  callee: Any, output_index: int, formal_indices: tuple[int, ...], nseed: int, constants: tuple[np.ndarray | None, ...]
) -> tuple[Any, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
  key = (output_index, formal_indices, nseed, tuple(None if value is None else value.tobytes() for value in constants))
  cache = _CALL_JVP_MANY_CACHE.setdefault(callee, {})
  if key not in cache:
    active = tuple(range(nseed))
    if all(value is not None for value in constants):
      active = tuple(row for row in range(nseed) if any(value is not None and np.any(value[row] != 0) for value in constants))
    seeds = {
      i: Expr.sym(f"fwd:{callee.input_names[i]}", (nseed, *callee.inputs[i].shape)) if value is None else Expr.const(value[list(active)])
      for i, value in zip(formal_indices, constants, strict=True)
    }
    out = callee.outputs[output_index]
    single_constant = len(formal_indices) == 1 and constants[0] is not None
    if single_constant:
      formal_index = formal_indices[0]
      deriv = _jvp_many_unrolled(out, callee.inputs[formal_index], seeds[formal_index])
    else:
      deriv = stack([_jvp(out, {callee.inputs[i]: seed[row] for i, seed in seeds.items()}, {}, {}) for row in range(len(active))], axis=0)
    deriv = callee._inherit_lowering(simplify_cse_fixpoint(deriv))
    dep_memo: dict[tuple[int, int], bool] = {}
    arg_indices = tuple(i for i, inp in enumerate(callee.inputs) if _depends_on(deriv, inp, dep_memo))
    seed_indices = tuple(i for i, value in zip(formal_indices, constants, strict=True) if value is None and _depends_on(deriv, seeds[i], dep_memo))
    inputs = tuple(callee.inputs[i] for i in arg_indices) + tuple(seeds[i] for i in seed_indices)
    input_names = tuple(callee.input_names[i] for i in arg_indices) + tuple(f"fwd:{callee.input_names[i]}" for i in seed_indices)
    if single_constant:
      formal_index = formal_indices[0]
      seed_hash = hashlib.sha1(constants[0].tobytes()).hexdigest()[:10]  # type: ignore[union-attr]
      name = f"{callee.name}_fwd{nseed}c{seed_hash}_{callee.output_names[output_index]}_{callee.input_names[formal_index]}"
      output_name = f"fwd:{callee.output_names[output_index]}:{callee.input_names[formal_index]}"
    else:
      seed_hash = hashlib.sha1(repr(key).encode()).hexdigest()[:10]
      name = f"{callee.name}_fwd{nseed}j{seed_hash}_{output_index}_" + "_".join(str(i) for i in formal_indices)
      output_name = f"fwd:{callee.output_names[output_index]}"
    fn = ConcreteFunction._from_exprs(name, inputs, [deriv], input_names, [output_name], role="forward")
    cache[key] = (fn, arg_indices, seed_indices, active)
  return cache[key]


def _call_jvp_many_const_function(
  callee: Any, output_index: int, formal_index: int, seed_value: np.ndarray
) -> tuple[Any, tuple[int, ...], tuple[int, ...]]:
  """Specialize the joint JVP helper for one formal with a constant seed."""
  fn, arg_indices, seed_indices, active = _call_jvp_many_function(callee, output_index, (formal_index,), seed_value.shape[0], (seed_value,))
  assert not seed_indices
  return fn, arg_indices, active


def _call_jvp_function(callee: Any, output_index: int, formal_indices: tuple[int, ...]) -> tuple[Any, tuple[int, ...], tuple[int, ...]]:
  key = (output_index, formal_indices)
  cache = _CALL_JVP_CACHE.setdefault(callee, {})
  if key not in cache:
    seeds = {i: Expr.sym(f"fwd:{callee.input_names[i]}", callee.inputs[i].shape) for i in formal_indices}
    deriv = callee._inherit_lowering(_jvp(callee.outputs[output_index], {callee.inputs[i]: seed for i, seed in seeds.items()}, {}, {}))
    dep_memo: dict[tuple[int, int], bool] = {}
    arg_indices = tuple(i for i, inp in enumerate(callee.inputs) if _depends_on(deriv, inp, dep_memo))
    seed_indices = tuple(i for i, seed in seeds.items() if _depends_on(deriv, seed, dep_memo))
    inputs = tuple(callee.inputs[i] for i in arg_indices) + tuple(seeds[i] for i in seed_indices)
    input_names = tuple(callee.input_names[i] for i in arg_indices) + tuple(seeds[i].name for i in seed_indices)
    name = f"{callee.name}_fwd{output_index}_" + "_".join(str(i) for i in formal_indices)
    fn = ConcreteFunction._from_exprs(name, inputs, [deriv], input_names, [f"fwd:{callee.output_names[output_index]}"], role="forward")
    cache[key] = (fn, arg_indices, seed_indices)
  return cache[key]


def _pack_jvp_maps(callee: Any, result: Expr, maps: list[Expr]) -> Expr:
  """Share specialized tangent bodies through one mapped result, preserving each seed layout."""
  groups: list[tuple[int, dict[Expr, tuple[Expr, int, int]], list[Expr]]] = []
  for mapped in maps:
    fn = mapped.attrs["callee"]
    specs = dict(zip(fn.inputs, zip(mapped.args, mapped.attrs["starts"], mapped.attrs["strides"], strict=True), strict=True))
    for length, bindings, members in groups:
      if length == mapped.attrs["length"] and all(formal not in bindings or bindings[formal] == spec for formal, spec in specs.items()):
        bindings.update(specs)
        members.append(mapped)
        break
    else:
      groups.append((mapped.attrs["length"], specs, [mapped]))
  replacements = {}
  cache = _CALL_JVP_PACK_CACHE.setdefault(callee, {})
  for length, bindings, members in groups:
    if len(members) == 1:
      continue
    functions = tuple(mapped.attrs["callee"] for mapped in members)
    inputs = tuple(bindings)
    key = (*functions, inputs)
    if key not in cache:
      outputs = [fn.outputs[0].reshape((fn.outputs[0].size,)) for fn in functions]
      packed = callee._inherit_lowering(simplify_cse_fixpoint(concat(outputs)))
      name_hash = hashlib.sha1(";".join(fn.name for fn in functions).encode()).hexdigest()[:10]
      names = {inp: name for fn in functions for inp, name in zip(fn.inputs, fn.input_names, strict=True)}
      cache[key] = ConcreteFunction._from_exprs(
        f"{callee.name}_fwd_pack_{name_hash}", inputs, [packed], [names[inp] for inp in inputs], ["fwd"], role="forward"
      )
    fn = cache[key]
    mapped = _mapped_call(fn, length, list(bindings.values()))
    width = fn.outputs[0].size
    offset = 0
    for member, function in zip(members, functions, strict=True):
      size = function.outputs[0].size
      indices = (np.arange(length)[:, None] * width + offset + np.arange(size)).reshape(-1)
      replacements[member] = gather(mapped, indices)
      offset += size
  return substitute(result, replacements) if replacements else result


def _periodic_seed_tiles(
  tangent: Expr, nseed: int, outer_size: int, start: int, stride: int, formal_size: int, length: int
) -> tuple[np.ndarray, int] | None:
  if not length or tangent.op != ExprOp.CONST or tangent.value is None:
    return None
  flat = np.asarray(tangent.value, dtype=np.float64).reshape(nseed, outer_size)
  tiles = np.stack([flat[:, start + it * stride : start + it * stride + formal_size] for it in range(length)])
  period = next(
    (k for k in range(1, min(8, length // 2) + 1) if length % k == 0 and np.array_equal(tiles, np.tile(tiles[:k], (length // k, 1, 1)))),
    None,
  )
  return None if period is None else (tiles, period)


def _local_seed_colors(callee_out: Expr, formal: Expr, nseed: int) -> tuple[Any, tuple[int, ...]] | None:
  mask = _jac_mask(callee_out, formal, {})
  colors = column_coloring(_mask_sparsity(mask)) if mask.nnz else ()
  return (mask, colors) if colors and max(colors) + 1 < nseed else None


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
    ones = Expr.const(np.ones(expr.args[0].size), dtype=d0.type.dtype)
    memo[expr.id] = ret = d0.reshape((nseed, expr.args[0].size)) @ ones
    return ret
  if expr.op == ExprOp.VMAP:
    callee = expr.attrs["callee"]
    output_idx = expr.attrs["output"]
    length = expr.attrs["length"]
    starts = expr.attrs["starts"]
    strides = expr.attrs["strides"]
    slice_size = expr.attrs["slice_size"]
    callee_out = callee.outputs[output_idx]

    maps: list[Expr] = []

    def mapped_call(fn: Any, count: int, specs: list[tuple[Expr, int, int]]) -> Expr:
      mapped = _mapped_call(fn, count, specs)
      maps.append(mapped)
      return mapped

    generic_seeds: dict[int, Expr] = {}
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

      # A constant tangent whose per-iteration tiles repeat with a short period (at most 8, and at
      # least twice, so a short horizon of distinct tiles is not unrolled) is baked into one
      # const-seed callee per tile: the 0/1 products fold inside the body and no seed table or
      # gather is emitted. Iteration ``it`` uses tile ``it % period``; residue class ``r`` is mapped
      # with start ``start + r * stride`` and stride ``stride * period``.
      periodic = _periodic_seed_tiles(actual_tan, nseed, outer_size, start, stride, formal_size, length)
      if periodic is not None:
        tiles, period = periodic
        n = length // period
        zero_rows = Expr.const(np.zeros((n, slice_size), dtype=np.float64))
        parts: list[Expr] = []
        for r in range(period):
          inner_fn, primal_arg_indices, active = _call_jvp_many_const_function(
            callee, output_idx, formal_idx, tiles[r].reshape((nseed, *formal.shape))
          )
          if not active:
            parts.append(Expr.const(np.zeros((nseed, n, slice_size), dtype=np.float64)))
            continue
          primal_specs = [(expr.args[i], starts[i] + r * strides[i], strides[i] * period) for i in primal_arg_indices]
          mapped_3d = mapped_call(inner_fn, n, primal_specs).reshape((n, len(active), slice_size))
          if len(active) == nseed:
            parts.append(mapped_3d.transpose((1, 0, 2)))
          else:
            active_pos = {row: i for i, row in enumerate(active)}
            parts.append(stack([mapped_3d[:, active_pos[row], :] if row in active_pos else zero_rows for row in range(nseed)], axis=0))
        if all(_is_zero_const(part) for part in parts):
          continue
        term = (parts[0] if period == 1 else stack(parts, axis=2)).reshape((nseed, length * slice_size))
        ret = term if ret is None else ret + term
        continue

      # Local coloring of the callee's Jacobian tile w.r.t. this formal. When this gives c_f < nseed
      # local colors, baking those local seeds into the per-iteration JVP callee yields a body of
      # size O(c_f) instead of O(nseed). The global compressed JVP is then assembled back by, for
      # each local color `c_local`, gathering the unique nonzero column of `actual_tan` for each
      # output row and multiplying with the per-iter compressed-JVP slice.
      local = _local_seed_colors(callee_out, formal, nseed)
      if local is not None:
        local_mask, local_colors = local
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
        mapped_flat = mapped_call(inner_fn, length, primal_specs)
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

      it_arr = np.arange(length, dtype=np.int64)[:, None, None]
      c_arr = np.arange(nseed, dtype=np.int64)[None, :, None]
      j_arr = np.arange(formal_size, dtype=np.int64)[None, None, :]
      tile_indices = (c_arr * outer_size + start + it_arr * stride + j_arr).reshape(-1)
      seed_buffer = gather(actual_tan, tile_indices)
      generic_seeds[formal_idx] = seed_buffer
    if generic_seeds:
      formals = tuple(generic_seeds)
      inner_fn, primal_arg_indices, seed_indices, _ = _call_jvp_many_function(callee, output_idx, formals, nseed, (None,) * len(formals))
      primal_specs = [(expr.args[i], starts[i], strides[i]) for i in primal_arg_indices]
      seed_specs = [(generic_seeds[i], 0, nseed * callee.inputs[i].size) for i in seed_indices]
      mapped_flat = mapped_call(inner_fn, length, [*primal_specs, *seed_specs])
      term = mapped_flat.reshape((length, nseed, slice_size)).transpose((1, 0, 2)).reshape((nseed, length * slice_size))
      ret = term if ret is None else ret + term
    ret = Expr.const(np.zeros((nseed, *expr.shape), dtype=np.float64)) if ret is None else ret
    memo[expr.id] = ret = _pack_jvp_maps(callee, ret, maps)
    return ret
  if expr.op == ExprOp.CALL:
    tangents = [simplify_cse_fixpoint(_jvp_many_structural(arg, wrt, seeds, memo, dep_memo)) for arg in expr.args]
    formals = tuple(i for i, tangent in enumerate(tangents) if not _is_zero_const(tangent))
    if not formals:
      memo[expr.id] = ret = Expr.const(np.zeros((nseed, *expr.shape), dtype=np.float64))
      return ret
    constants = tuple(tangents[i].value if tangents[i].op == ExprOp.CONST else None for i in formals)
    fn, arg_indices, seed_indices, active = _call_jvp_many_function(expr.attrs["callee"], expr.attrs["output"], formals, nseed, constants)
    call_args = [expr.args[i] for i in arg_indices] + [tangents[i] for i in seed_indices]
    ret = fn.symbolic_call(*fn.input_tree.unflatten(tuple(call_args)))
    if len(active) != nseed:
      active_pos = {row: i for i, row in enumerate(active)}
      ret = stack([ret[active_pos[row]] if row in active_pos else zeros_like(expr) for row in range(nseed)], axis=0)
    memo[expr.id] = ret
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
