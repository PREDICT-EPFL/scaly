"""Call and map derivative rules, helper construction, and mapped seed layout."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import numpy as np

from ..function.concrete import ConcreteFunction
from ..function.sugar import _mapped_call
from ..ir.expr import Expr, ExprOp, as_expr, concat, gather, scatter, stack, substitute, topo, zeros_like
from ..passes.expr import simplify_cse_fixpoint
from .helpers import HelperKey, helper_name
from .sparsity import _callee_mask, _depends_on, _mask_sparsity, column_coloring

_Pushforward = Callable[[Sequence[Expr], dict[Expr, Expr], int, str], tuple[Expr | None, ...]]
_Pullback = Callable[[Sequence[Expr], Sequence[Expr], Sequence[Expr]], tuple[Expr, ...]]


def body_tangents(callee: Any, outputs: Sequence[int], seeds: dict[int, Expr], nseed: int, pushforward: _Pushforward) -> tuple[Expr | None, ...]:
  """Tangents of the selected results of ``callee``, with ``seeds`` keyed by input index.

  An output with a JVP rule maps the single-seed rule over the seeds. Every other result,
  residuals included, differentiates the body with the caller's seed-axis traversal.
  """
  ruled = callee.rules is not None and callee.rules.jvp is not None
  body = [i for i in outputs if not ruled or i >= len(callee.outputs)]
  tangents = dict(
    zip(body, pushforward([callee.results[i] for i in body], {callee.inputs[k]: seed for k, seed in seeds.items()}, nseed, callee.name))
  )
  return tuple(tangents[i] if i in tangents else _rule_tangent(callee, i, seeds, nseed) for i in outputs)


def _rule_tangent(callee: Any, output: int, seeds: dict[int, Expr], nseed: int) -> Expr:
  rule = callee.rules.jvp
  reads = _reads(callee, output)
  specs = [(inp.reshape((inp.size,)), 0, 0) for inp in callee.inputs]
  for k, inp in enumerate(callee.inputs):
    seed = seeds.get(k) if reads[k] else None
    specs.append(
      (Expr.const(np.zeros(inp.size, dtype=inp.type.dtype.numpy()), dtype=inp.type.dtype), 0, 0)
      if seed is None
      else (seed.reshape((nseed * inp.size,)), 0, inp.size)
    )
  return _mapped_call(rule, nseed, specs, output).reshape((nseed, *callee.outputs[output].shape))


def _reads(callee: Any, output: int) -> tuple[bool, ...]:
  """Which input tangents the derivative of ``callee``'s result ``output`` reads."""
  rule = None if callee.rules is None or output >= len(callee.outputs) else callee.rules.jvp
  if rule is None:
    return (True,) * len(callee.inputs)
  memo: dict[tuple[int, int], bool] = {}
  return tuple(_depends_on(rule.outputs[output], tangent, memo) for tangent in rule.inputs[len(callee.inputs) :])


def read_args(expr: Expr) -> tuple[Expr, ...]:
  """The arguments of a call or map whose tangents its callee's derivative reads."""
  return tuple(arg for arg, read in zip(expr.args, _reads(expr.attrs["callee"], expr.attrs["output"]), strict=True) if read)


def body_cotangents(callee: Any, output: int, wrts: Sequence[int], cotangent: Expr, pullback: _Pullback) -> tuple[Expr, ...]:
  """Cotangents of the inputs ``wrts`` of ``callee`` given the cotangent of its result ``output``.

  An output with a reverse rule calls ``bwd`` on the residuals, as results of a call of ``callee``
  on its own inputs, so a caller substituting its actuals shares the invocation with the primal.
  Every other result differentiates the body with the caller's reverse traversal.
  """
  rules = callee.rules
  if rules is None or rules.bwd is None or output >= len(callee.outputs):
    return pullback((callee.results[output],), tuple(callee.inputs[k] for k in wrts), (cotangent,))
  residuals = callee._call(callee.inputs)[len(callee.outputs) :]
  cotangents = tuple(cotangent if j == output else zeros_like(out) for j, out in enumerate(callee.outputs))
  grads = rules.bwd._call((*residuals, *cotangents))
  return tuple(grads[k] for k in wrts)


def _is_zero_const(expr: Expr) -> bool:
  return expr.op == ExprOp.CONST and expr.value is not None and bool(np.all(expr.value == 0))


def _call_jvp_many_function(
  callee: Any,
  output_index: int,
  formal_indices: tuple[int, ...],
  nseed: int,
  constants: tuple[np.ndarray | None, ...],
  *,
  pushforward: _Pushforward,
) -> tuple[Any, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
  key = HelperKey.of(
    callee,
    "forward",
    (output_index,),
    formal_indices,
    nseed=nseed,
    constants=tuple(None if value is None else (value.dtype.str, value.tobytes()) for value in constants),
  )
  cache = callee._memo.helpers
  if key not in cache:
    active = tuple(range(nseed))
    if all(value is not None for value in constants):
      active = tuple(row for row in range(nseed) if any(value is not None and np.any(value[row] != 0) for value in constants))
    seeds = {
      i: Expr.sym(f"fwd:{callee.input_names[i]}", (nseed, *callee.inputs[i].shape), dtype=callee.inputs[i].type.dtype)
      if value is None
      else Expr.const(value[list(active)], dtype=callee.inputs[i].type.dtype)
      for i, value in zip(formal_indices, constants, strict=True)
    }
    single_constant = len(formal_indices) == 1 and constants[0] is not None
    (deriv,) = body_tangents(callee, (output_index,), seeds, len(active), pushforward)
    assert deriv is not None
    deriv = callee._inherit_lowering(simplify_cse_fixpoint(deriv))
    dep_memo: dict[tuple[int, int], bool] = {}
    arg_indices = tuple(i for i, inp in enumerate(callee.inputs) if _depends_on(deriv, inp, dep_memo))
    seed_indices = tuple(i for i, value in zip(formal_indices, constants, strict=True) if value is None and _depends_on(deriv, seeds[i], dep_memo))
    inputs = tuple(callee.inputs[i] for i in arg_indices) + tuple(seeds[i] for i in seed_indices)
    input_names = tuple(callee.input_names[i] for i in arg_indices) + tuple(f"fwd:{callee.input_names[i]}" for i in seed_indices)
    output_name = f"fwd:{callee.result_names[output_index]}"
    if single_constant:
      output_name += f":{callee.input_names[formal_indices[0]]}"
    fn = ConcreteFunction._from_exprs(helper_name(callee, key), inputs, [deriv], input_names, [output_name], role="forward")
    cache[key] = (fn, arg_indices, seed_indices, active)
  return cache[key]


def _call_jvp_many_const_function(
  callee: Any,
  output_index: int,
  formal_index: int,
  seed_value: np.ndarray,
  *,
  pushforward: _Pushforward,
) -> tuple[Any, tuple[int, ...], tuple[int, ...]]:
  """Specialize the joint JVP helper for one formal with a constant seed."""
  fn, arg_indices, seed_indices, active = _call_jvp_many_function(
    callee, output_index, (formal_index,), seed_value.shape[0], (seed_value,), pushforward=pushforward
  )
  assert not seed_indices
  return fn, arg_indices, active


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
  cache = callee._memo.helpers
  for length, bindings, members in groups:
    if len(members) == 1:
      continue
    functions = tuple(mapped.attrs["callee"] for mapped in members)
    inputs = tuple(bindings)
    key = HelperKey.of(callee, "forward", (), (), members=functions)
    if key not in cache:
      outputs = [fn.outputs[0].reshape((fn.outputs[0].size,)) for fn in functions]
      packed = callee._inherit_lowering(simplify_cse_fixpoint(concat(outputs)))
      names = {inp: name for fn in functions for inp, name in zip(fn.inputs, fn.input_names, strict=True)}
      cache[key] = ConcreteFunction._from_exprs(helper_name(callee, key), inputs, [packed], [names[inp] for inp in inputs], ["fwd"], role="forward")
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
  flat = np.asarray(tangent.value).reshape(nseed, outer_size)
  tiles = np.stack([flat[:, start + it * stride : start + it * stride + formal_size] for it in range(length)])
  period = next(
    (k for k in range(1, min(8, length // 2) + 1) if length % k == 0 and np.array_equal(tiles, np.tile(tiles[:k], (length // k, 1, 1)))),
    None,
  )
  return None if period is None else (tiles, period)


def _local_seed_colors(callee: Any, output: int, formal: int, nseed: int) -> tuple[Any, tuple[int, ...]] | None:
  mask = _callee_mask(callee, output, formal, {})
  colors = column_coloring(_mask_sparsity(mask)) if mask.nnz else ()
  return (mask, colors) if colors and max(colors) + 1 < nseed else None


def _vmap_jvp_many(
  expr: Expr,
  tangents: Sequence[Expr | None],
  nseed: int,
  *,
  pushforward: _Pushforward,
) -> Expr:
  callee = expr.attrs["callee"]
  output_idx = expr.attrs["output"]
  length = expr.attrs["length"]
  starts = expr.attrs["starts"]
  strides = expr.attrs["strides"]
  slice_size = expr.attrs["slice_size"]

  maps: list[Expr] = []

  def mapped_call(fn: Any, count: int, specs: list[tuple[Expr, int, int]]) -> Expr:
    mapped = _mapped_call(fn, count, specs)
    maps.append(mapped)
    return mapped

  generic_seeds: dict[int, Expr] = {}
  ret: Expr | None = None
  for formal_idx, (actual_outer, actual_tan) in enumerate(zip(expr.args, tangents, strict=True)):
    if actual_tan is None or _is_zero_const(actual_tan):
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
      zero_rows = Expr.const(np.zeros((n, slice_size), dtype=expr.type.dtype.numpy()), dtype=expr.type.dtype)
      parts: list[Expr] = []
      for r in range(period):
        inner_fn, primal_arg_indices, active = _call_jvp_many_const_function(
          callee,
          output_idx,
          formal_idx,
          tiles[r].reshape((nseed, *formal.shape)),
          pushforward=pushforward,
        )
        if not active:
          parts.append(Expr.const(np.zeros((nseed, n, slice_size), dtype=expr.type.dtype.numpy()), dtype=expr.type.dtype))
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
    local = _local_seed_colors(callee, output_idx, formal_idx, nseed)
    if local is not None:
      local_mask, local_colors = local
      c_f = max(local_colors) + 1
      seed_f = np.zeros((c_f, formal_size), dtype=expr.type.dtype.numpy())
      for j, c in enumerate(local_colors):
        seed_f[c, j] = 1.0
      seed_f_shaped = seed_f.reshape((c_f, *formal.shape)) if formal.shape != (formal.size,) else seed_f
      inner_fn, primal_arg_indices, active = _call_jvp_many_const_function(callee, output_idx, formal_idx, seed_f_shaped, pushforward=pushforward)
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
    inner_fn, primal_arg_indices, seed_indices, _ = _call_jvp_many_function(
      callee, output_idx, formals, nseed, (None,) * len(formals), pushforward=pushforward
    )
    primal_specs = [(expr.args[i], starts[i], strides[i]) for i in primal_arg_indices]
    seed_specs = [(generic_seeds[i], 0, nseed * callee.inputs[i].size) for i in seed_indices]
    mapped_flat = mapped_call(inner_fn, length, [*primal_specs, *seed_specs])
    term = mapped_flat.reshape((length, nseed, slice_size)).transpose((1, 0, 2)).reshape((nseed, length * slice_size))
    ret = term if ret is None else ret + term
  ret = Expr.const(np.zeros((nseed, *expr.shape), dtype=expr.type.dtype.numpy()), dtype=expr.type.dtype) if ret is None else ret
  ret = _pack_jvp_maps(callee, ret, maps)
  return ret


def _call_jvp_many(expr: Expr, tangents: Sequence[Expr | None], nseed: int, *, pushforward: _Pushforward) -> Expr:
  formals = tuple(i for i, tangent in enumerate(tangents) if tangent is not None and not _is_zero_const(tangent))
  if not formals:
    ret = Expr.const(np.zeros((nseed, *expr.shape), dtype=expr.type.dtype.numpy()), dtype=expr.type.dtype)
    return ret
  active_tangents = {i: tangents[i] for i in formals}
  constants = tuple(t.value if t.op == ExprOp.CONST else None for t in active_tangents.values() if t is not None)
  fn, arg_indices, seed_indices, active = _call_jvp_many_function(
    expr.attrs["callee"], expr.attrs["output"], formals, nseed, constants, pushforward=pushforward
  )
  if not active:
    return Expr.const(np.zeros((nseed, *expr.shape), dtype=expr.type.dtype.numpy()), dtype=expr.type.dtype)
  call_args = [expr.args[i] for i in arg_indices] + [tangents[i] for i in seed_indices if tangents[i] is not None]
  ret = fn.symbolic_call(*fn.input_tree.unflatten(tuple(call_args)))
  if len(active) != nseed:
    active_pos = {row: i for i, row in enumerate(active)}
    ret = stack(
      [
        ret[active_pos[row]] if row in active_pos else Expr.const(np.zeros(expr.shape, dtype=expr.type.dtype.numpy()), dtype=expr.type.dtype)
        for row in range(nseed)
      ],
      axis=0,
    )
  return ret


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


def _vmap_adj_function(callee: Any, output_index: int, active_formals: tuple[int, ...], *, pullback: _Pullback) -> tuple[Any, tuple[int, ...]]:
  key = HelperKey.of(callee, "adjoint", (output_index,), active_formals)
  cache = callee._memo.helpers
  if key not in cache:
    lam_name = f"lam:{callee.output_names[output_index]}"
    lam = Expr.sym(lam_name, callee.outputs[output_index].shape)
    grads = body_cotangents(callee, output_index, active_formals, lam, pullback)
    adj = callee._inherit_lowering(simplify_cse_fixpoint(concat([grad.reshape((grad.size,)) for grad in grads])))
    dep_memo: dict[tuple[int, int], bool] = {}
    arg_indices = tuple(i for i, inp in enumerate(callee.inputs) if _depends_on(adj, inp, dep_memo))
    inputs = tuple(callee.inputs[i] for i in arg_indices) + (lam,)
    input_names = tuple(callee.input_names[i] for i in arg_indices) + (lam_name,)
    fn = ConcreteFunction._from_exprs(
      helper_name(callee, key), inputs, [adj], input_names, [f"adj:{callee.output_names[output_index]}"], role="adjoint"
    )
    cache[key] = (fn, arg_indices)
  return cache[key]


def _vmap_vjp(
  vmap_expr: Expr, cot: Expr, wrts: Sequence[Expr], dep_memo: dict[tuple[int, int], bool], *, pullback: _Pullback
) -> list[tuple[Expr, Expr]]:
  callee = vmap_expr.attrs["callee"]
  output_idx = vmap_expr.attrs["output"]
  length = vmap_expr.attrs["length"]
  if length == 0:
    return []
  starts = vmap_expr.attrs["starts"]
  strides = vmap_expr.attrs["strides"]
  slice_size = vmap_expr.attrs["slice_size"]
  active_formals = tuple(k for k, arg in enumerate(vmap_expr.args) if any(_depends_on(arg, wrt, dep_memo, through_stops=False) for wrt in wrts))
  if not active_formals:
    return []

  adj_fn, arg_indices = _vmap_adj_function(callee, output_idx, active_formals, pullback=pullback)
  primal_specs = [(vmap_expr.args[i], starts[i], strides[i]) for i in arg_indices]
  mapped = _mapped_call(adj_fn, length, [*primal_specs, (cot, 0, slice_size)])
  adj_size = sum(callee.inputs[k].size for k in active_formals)
  ret: list[tuple[Expr, Expr]] = []
  offset = 0
  for k in active_formals:
    arg, start, stride = vmap_expr.args[k], starts[k], strides[k]
    formal_size = callee.inputs[k].size
    if stride == 0:
      indices = np.asarray([it * adj_size + offset + j for it in range(length) for j in range(formal_size)], dtype=np.int64)
      segments = gather(mapped, indices).reshape((length, formal_size))
      vbar = Expr.const(np.ones(length, dtype=np.float64)) @ segments
      if start != 0 or formal_size != arg.size:
        vbar = scatter(vbar, start + np.arange(formal_size, dtype=np.int64), arg.shape)
    else:
      pieces: list[Expr] = []
      groups = -(-formal_size // stride)
      for group in range(groups):
        iterations = range(group, length, groups)
        indices = np.asarray([it * adj_size + offset + j for it in iterations for j in range(formal_size)], dtype=np.int64)
        destinations = np.asarray([start + it * stride + j for it in iterations for j in range(formal_size)], dtype=np.int64)
        if indices.size:
          pieces.append(scatter(gather(mapped, indices), destinations, arg.shape))
      vbar = _sum_exprs(pieces)
    ret.append((arg, vbar))
    offset += formal_size
  return ret


def _call_vjp(expr: Expr, cot: Expr, active: tuple[int, ...] | None, *, pullback: _Pullback) -> tuple[Expr, ...]:
  args = expr.args
  callee = expr.attrs["callee"]
  output_idx = expr.attrs["output"]
  callee_out = callee.results[output_idx]
  # Differentiate the callee body against a fresh cotangent symbol, then graft the real ``cot``
  # in via the same substitution that maps formals to actuals. Passing ``cot`` directly into the
  # inner vjp would make it part of the substituted graph: if the caller reuses a callee formal
  # symbol (the usual construction pattern), occurrences of that symbol *inside the cotangent*
  # would be rewritten to this call's actuals, corrupting the adjoint.
  lam = Expr.sym(f"lam:{callee.result_names[output_idx]}", callee_out.shape)
  replacements = dict(zip((inp.id for inp in callee.inputs), args, strict=True))
  replacements[lam.id] = cot
  indices = tuple(range(len(args))) if active is None else active
  grads = body_cotangents(callee, output_idx, indices, lam, pullback)
  adjoints = dict(zip(indices, grads, strict=True))
  return tuple(_substitute(adjoints[i], replacements) if i in adjoints else zeros_like(arg) for i, arg in enumerate(args))


def _sum_exprs(exprs: Iterable[Expr]) -> Expr:
  ret: Expr | None = None
  for expr in exprs:
    ret = expr if ret is None else ret + expr
  return as_expr(0.0) if ret is None else ret
