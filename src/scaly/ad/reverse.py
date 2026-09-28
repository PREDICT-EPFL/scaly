"""Reverse-mode AD: ``vjp``, ``vjp_many``, and the per-op local adjoint rules."""

from __future__ import annotations

import weakref
from typing import Any, Iterable, Sequence

import numpy as np

from ..function import ConcreteFunction
from ..function.sugar import _scan_node, _while_node, vmap, while_parts
from ..ir.expr import (
  PREDICATE_OPS,
  Expr,
  ExprOp,
  independent,
  substitute,
  tangent_dtype,
  as_expr,
  cast,
  concat,
  copysign,
  define_rules,
  equal,
  gather,
  index_set,
  op_def,
  put,
  put_add,
  scatter,
  stack,
  take,
  topo,
  where,
  zeros_like,
)
from ..passes.expr import simplify_cse_fixpoint
from ..utils.options import get_options
from .forward import (
  is_zero_const,
  _minus_one,
  claim_name,
  custom_vjp_call,
  extern_no_derivative,
  extremum_weight,
  floor_tangent,
  options_tag,
  reduce_weights,
  segment_weights,
  sign,
)
from .sparsity import _depends_on


_VMAP_ADJ_CACHE: weakref.WeakKeyDictionary[Any, dict[tuple[Any, ...], tuple[Any, tuple[int, ...], frozenset[int]]]] = weakref.WeakKeyDictionary()


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


def _vmap_adj_function(callee: Any, output_index: int, active_formals: tuple[int, ...]) -> tuple[Any, tuple[int, ...], frozenset[int]]:
  """The adjoint of one map lane for the ``active_formals``, the inputs it reads, and the formals
  whose cotangent is a constant zero (an implicit rule's for a factor it treats as constant): the
  map's adjoint leaves those out, so reverse mode does not walk back into what produced them."""
  key = (output_index, active_formals, get_options().derivative_key())
  cache = _VMAP_ADJ_CACHE.setdefault(callee, {})
  if key not in cache:
    out = callee.outputs[output_index]
    taken = {*callee.input_names, *callee.output_names}
    lam_name = claim_name(f"lam:{callee.output_names[output_index]}", taken)
    lam = Expr.sym(lam_name, out.shape, dtype=tangent_dtype(out))
    grads = body_cotangents(callee, {output_index: lam}, active_formals)
    zero = frozenset(k for k, grad in zip(active_formals, grads, strict=True) if is_zero_const(simplify_cse_fixpoint(grad)))
    adj = callee._inherit_lowering(simplify_cse_fixpoint(concat([grad.reshape((grad.size,)) for grad in grads])))
    dep_memo: dict[tuple[int, int], bool] = {}
    arg_indices = tuple(i for i, inp in enumerate(callee.inputs) if _depends_on(adj, inp, dep_memo))
    inputs = tuple(callee.inputs[i] for i in arg_indices) + (lam,)
    input_names = tuple(callee.input_names[i] for i in arg_indices) + (lam_name,)
    # Suffix by formal index, not name: joined names are not injective ({a_b} vs {a, b}) and
    # lowering dedupes callees by name, so a collision would silently reuse the wrong proc body.
    name = f"{callee.name}_adj{output_index}_" + "_".join(str(i) for i in active_formals) + options_tag()
    fn = ConcreteFunction.from_exprs(name, inputs, [adj], input_names, [claim_name(f"adj:{callee.output_names[output_index]}", taken)])
    cache[key] = (fn, arg_indices, zero)
  return cache[key]


def _vmap_vjp(vmap_expr: Expr, cot: Expr, wrts: Sequence[Expr], dep_memo: dict[tuple[int, int], bool]) -> list[tuple[Expr, Expr]]:
  callee = vmap_expr.attrs["callee"]
  output_idx = vmap_expr.attrs["output"]
  length = vmap_expr.attrs["length"]
  if length == 0:
    return []
  starts = vmap_expr.attrs["starts"]
  strides = vmap_expr.attrs["strides"]
  slice_size = vmap_expr.attrs["slice_size"]
  active_formals = tuple(k for k, arg in enumerate(vmap_expr.args) if any(_depends_on(arg, wrt, dep_memo) for wrt in wrts))
  if not active_formals:
    return []

  adj_fn, arg_indices, zero = _vmap_adj_function(callee, output_idx, active_formals)
  primal_specs = [(vmap_expr.args[i], starts[i], strides[i]) for i in arg_indices]
  mapped = vmap(adj_fn, length, [*primal_specs, (cot, 0, slice_size)])
  adj_size = sum(callee.inputs[k].size for k in active_formals)
  ret: list[tuple[Expr, Expr]] = []
  offset = 0
  for k in active_formals:
    arg, start, stride = vmap_expr.args[k], starts[k], strides[k]
    formal_size = callee.inputs[k].size
    if k in zero:
      offset += formal_size
      continue
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


_SCAN_ADJ_CACHE: weakref.WeakKeyDictionary[Any, dict[tuple[Any, ...], Any]] = weakref.WeakKeyDictionary()


def _scan_adj_function(callee: Any, extras: tuple[int, ...], active: tuple[int, ...]) -> Any:
  """One backward step of a scan: from the cotangent ``lam`` of the carry leaving step ``k``, the
  carry entering it, step ``k``'s slices, and step ``k``'s share of the cotangent of each output in
  ``extras`` (a stacked output ``j > 0``, or ``-1`` for the stored entering carry), return the
  cotangent of the entering carry and of each ``active`` slice. Every output of a scan that has a
  cotangent goes through this one step, so a scan has one backward scan however many of its outputs
  are used."""
  key = (extras, active, get_options().derivative_key())
  cache = _SCAN_ADJ_CACHE.setdefault(callee, {})
  if key not in cache:
    carry, xs = callee.inputs[0], callee.inputs[1:]
    taken = {*callee.input_names, *callee.output_names}
    lam = Expr.sym(claim_name(f"lam:{callee.input_names[0]}", taken), carry.shape, dtype=tangent_dtype(carry))
    cots: dict[int, Expr] = {0: lam}
    extra: list[Expr] = []
    for output in extras:
      if output > 0:
        bar = Expr.sym(
          claim_name(f"lam:{callee.output_names[output]}", taken), callee.outputs[output].shape, dtype=tangent_dtype(callee.outputs[output])
        )
        cots[output] = bar
      else:
        bar = Expr.sym(claim_name(f"lam:{callee.input_names[0]}:t", taken), carry.shape, dtype=tangent_dtype(carry))
      extra.append(bar)
    grads = body_cotangents(callee, cots, (0, *(i + 1 for i in active)))
    lam_in = grads[0]
    if -1 in extras:
      lam_in = lam_in + extra[extras.index(-1)]
    inputs = [lam, carry, *xs, *extra]
    names = [str(lam.name), *callee.input_names, *(str(e.name) for e in extra)]
    body = [callee._inherit_lowering(simplify_cse_fixpoint(g)) for g in (lam_in, *grads[1:])]
    tag = "_".join("t" if output < 0 else str(output) for output in extras) or "0"
    suffix = f"{tag}_" + ("_".join(str(i) for i in active) or "c") + options_tag()
    cache[key] = ConcreteFunction.from_exprs(
      f"{callee.name}_scanadj{suffix}",
      inputs,
      body,
      names,
      [claim_name("adj:carry", taken), *(claim_name(f"adj:{callee.input_names[i + 1]}", taken) for i in active)],
    )
  return cache[key]


def _group_key(expr: Expr) -> tuple[Any, ...] | None:
  """The call or scan an output node belongs to, for nodes whose outputs are differentiated together:
  the outputs of one call share its callee and arguments, those of one scan also its slicing."""
  if expr.op == ExprOp.CALL:
    return ("call", id(expr.attrs["callee"]), tuple(arg.id for arg in expr.args))
  if expr.op == ExprOp.SCAN:
    return ("scan", id(expr.attrs["callee"]), tuple(arg.id for arg in expr.args), expr.attrs["length"], expr.attrs["starts"], expr.attrs["strides"])
  return None


def _call_vjp(expr: Expr, cots: dict[int, Expr], wrts: Sequence[Expr], dep_memo: dict[tuple[int, int], bool]) -> list[tuple[Expr, Expr]]:
  """The adjoint of one call, given the cotangents of its used outputs: one reverse sweep through the
  callee for all of them, so loops inside it get one backward pass each rather than one per output.
  Only the arguments that depend on ``wrts`` are differentiated, as for maps and loops: a callee
  input that has no derivative (a looped sparse factor) is never asked for one it does not need."""
  callee, args = expr.attrs["callee"], expr.args
  if callee.custom_vjp is not None:
    return list(zip(args, custom_vjp_call(callee, args, cots), strict=True))
  active = [i for i, arg in enumerate(args) if any(_depends_on(arg, wrt, dep_memo) for wrt in wrts)]
  # Differentiate the callee body against fresh cotangent symbols, then graft the real cotangents in
  # via the same substitution that maps formals to actuals. Passing them directly into the inner vjp
  # would make them part of the substituted graph: if the caller reuses a callee formal symbol (the
  # usual construction pattern), occurrences of that symbol *inside a cotangent* would be rewritten
  # to this call's actuals, corrupting the adjoint.
  taken = {*callee.input_names, *callee.output_names}
  used = sorted(cots)
  lams = [Expr.sym(claim_name(f"lam:{callee.output_names[k]}", taken), callee.outputs[k].shape, dtype=tangent_dtype(callee.outputs[k])) for k in used]
  replacements = dict(zip((inp.id for inp in callee.inputs), args, strict=True))
  for k, lam in zip(used, lams, strict=True):
    replacements[lam.id] = cots[k]
  grads = vjp(tuple(callee.outputs[k] for k in used), tuple(callee.inputs[i] for i in active), tuple(lams))
  return [(args[i], _substitute(g, replacements)) for i, g in zip(active, grads, strict=True)]


def _scan_vjp(expr: Expr, cots: dict[int, Expr], wrts: Sequence[Expr], dep_memo: dict[tuple[int, int], bool]) -> list[tuple[Expr, Expr]]:
  """The adjoint of one scan, given the cotangent of each of its used outputs (``cots``, by output
  index): one scan over the reversed steps, reading the stored carries backwards and each stacked
  cotangent backwards. The slices' cotangents come back stacked and accumulate into their outer
  tensors by ``scatter``."""
  callee, length = expr.attrs["callee"], int(expr.attrs["length"])
  init, outers = expr.args[0], expr.args[1:]
  starts, strides = expr.attrs["starts"], expr.attrs["strides"]
  if length == 0:
    return [(init, cots[0])] if 0 in cots else []
  cs = init.size
  active = tuple(i for i, outer in enumerate(outers) if any(_depends_on(outer, wrt, dep_memo) for wrt in wrts))
  extras = tuple(sorted(output for output in cots if output != 0))
  fn = _scan_adj_function(callee, extras, active)
  _check_trajectory(callee, length, init.size)
  carries = _scan_node(callee, init, tuple(outers), starts, strides, length, -1)
  rev_outers = [carries, *outers]
  rev_starts = [(length - 1) * cs, *(s + (length - 1) * st for s, st in zip(starts, strides, strict=True))]
  rev_strides = [-cs, *(-st for st in strides)]
  for output in extras:
    size = cs if output == -1 else callee.outputs[output].size
    rev_outers.append(cots[output])
    rev_starts.append((length - 1) * size)
    rev_strides.append(-size)
  lam0 = cots.get(0, zeros_like(init))
  rev = [_scan_node(fn, lam0, tuple(rev_outers), tuple(rev_starts), tuple(rev_strides), length, k) for k in range(1 + len(active))]
  ret = [(init, rev[0])]
  for stacked, i in zip(rev[1:], active, strict=True):
    size = callee.inputs[i + 1].size
    steps = length - 1 - np.arange(length)
    dest = (starts[i] + steps[:, None] * strides[i] + np.arange(size)[None, :]).reshape(-1)
    ret.append((outers[i], scatter(stacked, dest, outers[i].shape)))
  return ret


_WHILE_ADJ_CACHE: weakref.WeakKeyDictionary[Any, dict[tuple[Any, ...], Any]] = weakref.WeakKeyDictionary()


def _while_adj_function(body: Any, index: bool, n_params: int, active: tuple[int, ...]) -> Any:
  """One backward step of a while loop, masked. Inputs: ``lam`` (the carry's cotangent followed by
  the running cotangents of the ``active`` params), the carry entering step ``k``, ``k``, the step
  count ``n`` and the loop's params. Output: when ``k < n`` the entering carry's cotangent and the
  params' cotangents plus this step's share; otherwise ``lam`` unchanged, so steps the loop never
  took contribute nothing."""
  cache = _WHILE_ADJ_CACHE.setdefault(body, {})
  key = (index, n_params, active, get_options().derivative_key())
  if key not in cache:
    carry = body.inputs[0]
    cs = carry.size
    first = 1 + int(index)
    formals = body.inputs[first : first + n_params]
    sizes = [formals[i].size for i in active]
    taken = {*body.input_names, *body.output_names}
    lam = Expr.sym(claim_name(f"lam:{body.input_names[0]}", taken), (cs + sum(sizes),), dtype=tangent_dtype(carry))
    step, count = Expr.sym(claim_name("step", taken), (1,)), Expr.sym(claim_name("count", taken), (1,), diff=False)
    lam_c = lam[:cs].reshape(carry.shape)
    backs = body_cotangents(body, {0: lam_c}, (0, *(first + i for i in active)))
    if index:  # the body's step number is the backward step's ``step``
      backs = tuple(_substitute(b, {body.inputs[1].id: cast(step[0], "int64")}) for b in backs)
    took = step[0] < count[0]
    parts = [where(took, backs[0], lam_c).reshape((cs,))]
    offset = cs
    for back, size in zip(backs[1:], sizes, strict=True):
      parts.append(lam[offset : offset + size] + where(took, back.reshape((size,)), 0.0))
      offset += size
    out = concat(parts) if len(parts) > 1 else parts[0]
    inputs = [lam, carry, step, count, *formals]
    names = [claim_name("lam", taken), body.input_names[0], str(step.name), str(count.name), *body.input_names[first : first + n_params]]
    # Each step-number flag and set of active params is its own Function, so it needs its own name.
    suffix = ("_k" if index else "") + "".join(f"_p{i}" for i in active) + options_tag()
    cache[key] = ConcreteFunction.from_exprs(
      f"{body.name}_whileadj{suffix}", inputs, [body._inherit_lowering(simplify_cse_fixpoint(out))], names, [claim_name("adj:carry", taken)]
    )
  return cache[key]


def _put_winners(base: Expr, idx: Expr, lane_cot: Expr, in_range: bool) -> Expr:
  """``lane_cot`` kept only on the lanes of a ``put`` whose write survives: with repeated indices the
  last lane writing an entry wins. Indices known to be distinct need no mask."""
  lanes = idx.size
  if idx.op == ExprOp.CONST and idx.value is not None:
    known = np.asarray(idx.value).reshape(-1)
    known = known[(known >= 0) & (known < base.shape[-1])]
    if np.unique(known).size == known.size:
      return lane_cot
  order = Expr.const(np.arange(lanes, dtype=np.float64))
  marker = put(Expr.const(np.full(base.shape[-1], -1.0)), idx, order, in_range=in_range)
  won = equal(take(marker, idx, fill=-1.0, in_range=in_range), order)
  return where(won, lane_cot, 0.0)


def _check_trajectory(body: Any, steps: int, carry_size: int) -> None:
  """Reverse mode through a loop stores the carry at every step; refuse a loop whose store would
  exceed ``sc.options(max_trajectory=...)`` values, naming the fix."""
  limit = get_options().max_trajectory
  if steps * carry_size > limit:
    raise ValueError(
      f"reverse mode through the loop over {body.name!r} would store {steps} carries of {carry_size} values "
      f"({steps * carry_size} in all, over max_trajectory={limit}); give the computation an implicit derivative "
      "with sc.custom_derivative, or raise sc.options(max_trajectory=...)"
    )


def _while_vjp(expr: Expr, cot: Expr, wrts: Sequence[Expr], dep_memo: dict[tuple[int, int], bool]) -> list[tuple[Expr, Expr]]:
  """Reverse mode through a while loop: a ``max_iter``-step scan backwards over the stored carries,
  where steps at or beyond the step count leave the cotangent unchanged. The params reach every
  backward step unchanged (stride-0 inputs), and their cotangents accumulate in the scan's carry."""
  output = int(expr.attrs["output"])
  if output == 1:
    return []
  if output == -1:
    raise NotImplementedError("reverse mode through the stored carries of a while_loop (reverse over reverse) is not implemented")
  cond, body, init, params, max_iter, index = while_parts(expr)
  first = 1 + int(index)
  # A param only the condition reads moves the step count, whose derivative is zero: no cotangent.
  active = tuple(
    i
    for i, param in enumerate(params)
    if any(_depends_on(param, wrt, dep_memo) for wrt in wrts) and _depends_on(body.outputs[0], body.inputs[first + i], dep_memo)
  )
  if max_iter == 0:
    return [(init, cot)]
  cs = init.size
  _check_trajectory(body, max_iter, init.size)
  carries = _while_node(cond, body, init, max_iter, -1, params, index)
  count = _while_node(cond, body, init, max_iter, 1, params, index).reshape((1,))
  steps = Expr.const(np.arange(max_iter, dtype=np.float64))
  outers = (carries, steps, count, *(param.reshape((param.size,)) for param in params))
  starts = ((max_iter - 1) * cs, max_iter - 1, 0, *(0 for _ in params))
  strides = (-cs, -1, 0, *(0 for _ in params))
  sizes = [params[i].size for i in active]
  lam0 = concat([cot.reshape((cs,)), *(zeros_like(params[i]).reshape((params[i].size,)) for i in active)])
  back = _scan_node(_while_adj_function(body, index, len(params), active), lam0, outers, starts, strides, max_iter, 0)
  ret = [(init, back[:cs].reshape(init.shape))]
  offset = cs
  for i, size in zip(active, sizes, strict=True):
    ret.append((params[i], back[offset : offset + size].reshape(params[i].shape)))
    offset += size
  return ret


def body_cotangents(fn: Any, cots: dict[int, Expr], wrt: tuple[int, ...]) -> tuple[Expr, ...]:
  """The cotangents of ``fn``'s inputs ``wrt`` (by index) given cotangents of some outputs (by index),
  in terms of its inputs: from its reverse rule when it has one, else by differentiating its body.
  The counterpart of ``forward.body_tangents``, used wherever reverse mode looks inside a Function."""
  if fn.custom_vjp is not None:
    grads = custom_vjp_call(fn, fn.inputs, cots)
    return tuple(grads[i] for i in wrt)
  outs = tuple(fn.outputs[j] for j in cots)
  return vjp(outs, tuple(fn.inputs[i] for i in wrt), tuple(cots.values()))


class NoAdjoint:
  """What a ``vjp`` rule gives for an argument it has no cotangent rule for: reverse mode raises
  ``NotImplementedError(reason)`` when that argument depends on what is being differentiated, and
  otherwise passes it by, so an op can be linear in one argument and refuse the other."""

  __slots__ = ("reason",)

  def __init__(self, reason: str) -> None:
    self.reason = reason


type Adjoint = Expr | NoAdjoint


def vjp(outputs: Sequence[Expr], wrts: Sequence[Expr], cotangents: Sequence[Expr]) -> tuple[Expr, ...]:
  """Reverse-mode derivative: one adjoint per entry of ``wrts``, seeded by ``cotangents``.

  Each cotangent is shaped like its output. One sweep computes the derivative of a scalar with
  respect to every input at once, which is why gradients go through here.
  """
  if len(outputs) != len(cotangents):
    raise ValueError(f"expected {len(outputs)} cotangents, got {len(cotangents)}")
  if any(w.op != ExprOp.INPUT for w in wrts):
    outputs, at, back = independent(outputs, wrts)
    return tuple(substitute(adjoint, back) for adjoint in vjp(outputs, at, cotangents))
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

  def accumulate(arg: Expr, arg_cot: Expr) -> None:
    # A constant zero contributes nothing, and leaving it out keeps reverse mode from walking back
    # through whatever produced ``arg`` (a factorization an implicit rule does not differentiate).
    if arg.id in expr_ids and not is_zero_const(arg_cot):
      adjoints[arg.id] = arg_cot if arg.id not in adjoints else adjoints[arg.id] + arg_cot

  # The output nodes of one call or scan are differentiated together, when the last of them is
  # reached: every node that reads their outputs comes later in ``nodes``, and their arguments earlier.
  group_left: dict[tuple[Any, ...], int] = {}
  for expr in nodes:
    key = _group_key(expr)
    if key is not None:
      group_left[key] = group_left.get(key, 0) + 1
  group_cots: dict[tuple[Any, ...], dict[int, Expr]] = {}

  for expr in reversed(nodes):
    cot = adjoints.get(expr.id)
    key = _group_key(expr)
    if key is not None:
      group_left[key] -= 1
      if cot is not None and needed(expr):
        group_cots.setdefault(key, {})[int(expr.attrs["output"])] = cot
      if group_left[key] or key not in group_cots:
        continue
      cots = group_cots.pop(key)
      pairs = _scan_vjp(expr, cots, wrts, dep_memo) if expr.op == ExprOp.SCAN else _call_vjp(expr, cots, wrts, dep_memo)
      for arg, arg_cot in pairs:
        accumulate(arg, arg_cot)
      continue
    if cot is None or expr.op in {ExprOp.INPUT, ExprOp.CONST} or expr.op in PREDICATE_OPS or not needed(expr):
      continue
    if expr.op in (ExprOp.VMAP, ExprOp.WHILE):
      rule = _vmap_vjp if expr.op == ExprOp.VMAP else _while_vjp
      pairs = rule(expr, cot, wrts, dep_memo)
      for arg, arg_cot in pairs:
        accumulate(arg, arg_cot)
      continue
    for arg, arg_cot in zip(expr.args, _masked_local_vjp(expr, cot), strict=True):
      if isinstance(arg_cot, NoAdjoint):
        if needed(arg):
          raise NotImplementedError(arg_cot.reason)
        continue
      accumulate(arg, arg_cot)

  return tuple(adjoints.get(wrt.id, zeros_like(wrt)) for wrt in wrts)


# Ops whose adjoint in each argument of the output's shape is the cotangent times a local
# derivative, entry by entry.
def _maskable(op: str) -> bool:
  """An elementwise op a derivative flows through (``floor`` and ``ceil`` have none)."""
  definition = op_def(op)
  return "elementwise" in definition.traits and definition.differentiable and op not in (ExprOp.FLOOR, ExprOp.CEIL)


def _masked(cot: Expr) -> tuple[Expr, Expr, bool] | None:
  """``(cond, inner, chosen)`` for a cotangent a ``where`` masked to one branch: ``where(cond, inner, 0)``
  (``chosen``) or ``where(cond, 0, inner)``."""
  if cot.op != ExprOp.SELECT:
    return None
  cond, a, b = cot.args
  if is_zero_const(b) and a.shape == cot.shape:
    return cond, a, True
  if is_zero_const(a) and b.shape == cot.shape:
    return cond, b, False
  return None


def _sum_terms(cot: Expr) -> list[Expr]:
  """The terms of a cotangent accumulated as a sum of same-shaped parts, in order."""
  terms: list[Expr] = []
  stack = [cot]
  while stack:
    e = stack.pop()
    if e.op == ExprOp.ADD and e.args[0].shape == e.args[1].shape == cot.shape:
      stack.extend(reversed(e.args))
    else:
      terms.append(e)
  return terms


def _sum_adjoints(bars: Sequence[Adjoint]) -> Adjoint:
  missing = [bar for bar in bars if isinstance(bar, NoAdjoint)]
  if missing:
    return missing[0]
  live = [bar for bar in bars if isinstance(bar, Expr) and not is_zero_const(bar)]
  return _sum_exprs(live) if live else bars[0]


def _masked_local_vjp(expr: Expr, cot: Expr) -> tuple[Adjoint, ...]:
  """``_local_vjp``, keeping a masked cotangent masked through an elementwise op:
  ``where(c, t, 0) * f'(x)`` becomes ``where(c, t * f'(x), 0)``. An entry the ``where`` did not
  choose then stays zero even where ``f'`` is infinite or NaN (``sqrt``, ``log`` or ``1 / x`` at 0),
  so the derivative flows through the chosen branch only, as it does in forward mode. An argument
  broadcast to the output's shape sums the cotangent over entries, and keeps the plain rule."""
  if not _maskable(expr.op):
    return _local_vjp(expr, cot)
  terms = _sum_terms(cot)
  if len(terms) > 1 and any(_masked(t) is not None for t in terms):
    # A node read under two ``where``s sums two masked cotangents. The adjoint is linear in the
    # cotangent, so each term keeps its own mask through the op.
    per_term = [_masked_local_vjp(expr, t) for t in terms]
    return tuple(_sum_adjoints(bars) for bars in zip(*per_term, strict=True))
  masked = _masked(cot)
  if masked is None:
    return _local_vjp(expr, cot)
  cond, inner, chosen = masked
  inner_bars = _masked_local_vjp(expr, inner)
  plain: tuple[Adjoint, ...] | None = None
  bars: list[Adjoint] = []
  for i, (arg, bar) in enumerate(zip(expr.args, inner_bars, strict=True)):
    if arg.shape != expr.shape:
      plain = _local_vjp(expr, cot) if plain is None else plain
      bars.append(plain[i])
    elif isinstance(bar, NoAdjoint) or is_zero_const(bar):
      bars.append(bar)
    else:
      bars.append(where(cond, bar, 0.0) if chosen else where(cond, 0.0, bar))
  return tuple(bars)


def vjp_many(outputs: Sequence[Expr], wrts: Sequence[Expr], cotangents: Sequence[Expr]) -> tuple[Expr, ...]:
  """Reverse mode over several cotangent seeds at once.

  Each cotangent has a leading seed axis, ``(n, *output.shape)``, and each returned adjoint
  carries the same leading axis.
  """
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


def _local_vjp(expr: Expr, cot: Expr) -> tuple[Adjoint, ...]:
  rule = op_def(expr.op).vjp
  if rule is None:
    raise NotImplementedError(f"VJP for op {expr.op!r} is not implemented")
  return rule(expr, cot)


# The adjoint rules of the builtin ops, each ``(expr, cot) -> one cotangent per argument``, or a
# ``NoAdjoint`` for one it cannot give (``OpDef.vjp``).


def _vjp_neg(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  return (-cot,)


def _vjp_add(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  return (_unbroadcast(cot, args[0].shape, expr.shape), _unbroadcast(cot, args[1].shape, expr.shape))


def _vjp_sub(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  return (_unbroadcast(cot, args[0].shape, expr.shape), _unbroadcast(-cot, args[1].shape, expr.shape))


def _vjp_mul(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  return (_unbroadcast(cot * args[1], args[0].shape, expr.shape), _unbroadcast(cot * args[0], args[1].shape, expr.shape))


def _vjp_div(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  return (
    _unbroadcast(cot / args[1], args[0].shape, expr.shape),
    _unbroadcast(-((cot / args[1]) * expr), args[1].shape, expr.shape),
  )


def _vjp_pow(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  return (
    _unbroadcast(cot * args[1] * (args[0] ** _minus_one(args[1])), args[0].shape, expr.shape),
    _unbroadcast(cot * expr * args[0].log(), args[1].shape, expr.shape),
  )


def _vjp_sin(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  return (cot * args[0].cos(),)


def _vjp_cos(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  return (-cot * args[0].sin(),)


def _vjp_tan(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  return (cot / (args[0].cos() ** 2),)


def _vjp_asin(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  return (cot / (1 - args[0] ** 2).sqrt(),)


def _vjp_acos(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  return (-cot / (1 - args[0] ** 2).sqrt(),)


def _vjp_atan(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  return (cot / (1 + args[0] ** 2),)


def _vjp_atan2(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  y, x = args
  denom = x * x + y * y
  return (_unbroadcast(cot * x / denom, y.shape, expr.shape), _unbroadcast(-cot * y / denom, x.shape, expr.shape))


def _vjp_sinh(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  return (cot * args[0].cosh(),)


def _vjp_cosh(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  return (cot * args[0].sinh(),)


def _vjp_tanh(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  return (cot * (1 - expr * expr),)


def _vjp_erf(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  return (cot * (2 / np.sqrt(np.pi)) * (-(args[0] ** 2)).exp(),)


def _vjp_exp(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  return (cot * expr,)


def _vjp_log(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  return (cot / args[0],)


def _vjp_sqrt(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  return (cot / (2 * expr),)


def _vjp_abs(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  return (cot * sign(args[0]),)


def _vjp_floor_ceil(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  return (floor_tangent(expr),)


def _vjp_minimum_maximum(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  w = extremum_weight(expr)
  return (_unbroadcast(cot * w, args[0].shape, expr.shape), _unbroadcast(cot * (1.0 - w), args[1].shape, expr.shape))


def _vjp_max_min(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  return (cot * reduce_weights(expr),)


def _vjp_segment_max_segment_min(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  return (gather(cot, expr.attrs["indices"]) * segment_weights(expr),)


def _vjp_index_add(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  return (cot, gather(cot, expr.attrs["indices"]))


def _vjp_index_set(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  return (index_set(cot, expr.attrs["indices"], np.zeros(args[1].size)), gather(cot, expr.attrs["indices"]))


def _vjp_take_put_add_put(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  ok = bool(expr.attrs.get("in_range"))
  if expr.op == ExprOp.TAKE:
    return (put_add(zeros_like(args[0]), args[1], cot, in_range=ok), zeros_like(args[1]))
  if expr.op == ExprOp.PUT_ADD:
    return (cot, zeros_like(args[1]), take(cot, args[1], in_range=ok))
  # A put's written entries of the base do not reach the output, and only the last lane writing
  # an entry does.
  return (
    put(cot, args[1], zeros_like(args[2]), in_range=ok),
    zeros_like(args[1]),
    _put_winners(args[0], args[1], take(cot, args[1], in_range=ok), ok),
  )


def _vjp_select(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  cond, a, b = args
  return (
    zeros_like(cond),
    _unbroadcast(where(cond, cot, 0.0), a.shape, expr.shape),
    _unbroadcast(where(cond, 0.0, cot), b.shape, expr.shape),
  )


def _vjp_copysign(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  x, s = args
  return (_unbroadcast(cot * copysign(1.0, x) * copysign(1.0, s), x.shape, expr.shape), zeros_like(s))


def _vjp_cast(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  return (cast(cot, args[0].type.dtype) if expr.type.diff else zeros_like(args[0]),)


def _vjp_sum(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  return (cot * _ones_like(args[0]),)


def _vjp_reshape(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  return (cot.reshape(args[0].shape),)


def _vjp_transpose(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  axes = expr.attrs["axes"]
  inv = tuple(int(np.argsort(axes)[i]) for i in range(len(axes)))
  return (cot.transpose(inv),)


def _vjp_slice(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  indices = np.arange(args[0].size).reshape(args[0].shape)[expr.attrs["index"]]
  return (scatter(cot, indices, args[0].shape),)


def _vjp_gather(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  return (_gather_vjp(cot, expr.attrs["indices"], args[0].shape),)


def _vjp_scatter(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  return (gather(cot, expr.attrs["indices"]),)


def _vjp_stack(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  return _stack_vjp(cot, len(args), expr.attrs.get("axis", 0))


def _vjp_concat(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  return _concat_vjp(cot, args, expr.attrs.get("axis", 0))


def _vjp_matmul(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  return _matmul_vjp(args[0], args[1], cot)


def _vjp_extern_call(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  raise NotImplementedError(extern_no_derivative(expr))  # see the matching JVP rule


_VJP_RULES = {
  ExprOp.NEG: _vjp_neg,
  ExprOp.ADD: _vjp_add,
  ExprOp.SUB: _vjp_sub,
  ExprOp.MUL: _vjp_mul,
  ExprOp.DIV: _vjp_div,
  ExprOp.POW: _vjp_pow,
  ExprOp.SIN: _vjp_sin,
  ExprOp.COS: _vjp_cos,
  ExprOp.TAN: _vjp_tan,
  ExprOp.ASIN: _vjp_asin,
  ExprOp.ACOS: _vjp_acos,
  ExprOp.ATAN: _vjp_atan,
  ExprOp.ATAN2: _vjp_atan2,
  ExprOp.SINH: _vjp_sinh,
  ExprOp.COSH: _vjp_cosh,
  ExprOp.TANH: _vjp_tanh,
  ExprOp.ERF: _vjp_erf,
  ExprOp.EXP: _vjp_exp,
  ExprOp.LOG: _vjp_log,
  ExprOp.SQRT: _vjp_sqrt,
  ExprOp.ABS: _vjp_abs,
  ExprOp.FLOOR: _vjp_floor_ceil,
  ExprOp.CEIL: _vjp_floor_ceil,
  ExprOp.MINIMUM: _vjp_minimum_maximum,
  ExprOp.MAXIMUM: _vjp_minimum_maximum,
  ExprOp.MAX: _vjp_max_min,
  ExprOp.MIN: _vjp_max_min,
  ExprOp.SEGMENT_MAX: _vjp_segment_max_segment_min,
  ExprOp.SEGMENT_MIN: _vjp_segment_max_segment_min,
  ExprOp.INDEX_ADD: _vjp_index_add,
  ExprOp.INDEX_SET: _vjp_index_set,
  ExprOp.TAKE: _vjp_take_put_add_put,
  ExprOp.PUT_ADD: _vjp_take_put_add_put,
  ExprOp.PUT: _vjp_take_put_add_put,
  ExprOp.SELECT: _vjp_select,
  ExprOp.COPYSIGN: _vjp_copysign,
  ExprOp.CAST: _vjp_cast,
  ExprOp.SUM: _vjp_sum,
  ExprOp.RESHAPE: _vjp_reshape,
  ExprOp.TRANSPOSE: _vjp_transpose,
  ExprOp.SLICE: _vjp_slice,
  ExprOp.GATHER: _vjp_gather,
  ExprOp.SCATTER: _vjp_scatter,
  ExprOp.STACK: _vjp_stack,
  ExprOp.CONCAT: _vjp_concat,
  ExprOp.MATMUL: _vjp_matmul,
  ExprOp.EXTERN_CALL: _vjp_extern_call,
}
for _op, _rule in _VJP_RULES.items():
  define_rules(_op, vjp=_rule)


def _ones_like(expr: Expr) -> Expr:
  return Expr.const(np.ones(expr.shape), dtype=tangent_dtype(expr), lowering=expr.lowering)


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
  """The adjoint of a gather: every read's cotangent accumulates back into the entry it read."""
  return scatter(cot.reshape((cot.size,)), indices.reshape(-1), shape)


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
