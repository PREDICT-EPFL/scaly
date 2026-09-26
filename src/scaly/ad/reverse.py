"""Reverse-mode AD: ``vjp``, ``vjp_many``, and the per-op local adjoint rules."""

from __future__ import annotations

import weakref
from typing import Any, Iterable, Sequence

import numpy as np

from ..function import Function
from ..function.sugar import _scan_node, _while_node, vmap
from ..ir.expr import PREDICATE_OPS, Expr, ExprOp, as_expr, cast, concat, copysign, gather, index_set, scatter, stack, topo, where, zeros_like
from ..passes.expr import simplify_cse_fixpoint
from .forward import claim_name, custom_vjp_call, extremum_weight, reduce_weights, segment_weights, sign
from .sparsity import _depends_on


_VMAP_ADJ_CACHE: weakref.WeakKeyDictionary[Any, dict[tuple[int, tuple[int, ...]], tuple[Any, tuple[int, ...]]]] = weakref.WeakKeyDictionary()


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


def _vmap_adj_function(callee: Any, output_index: int, active_formals: tuple[int, ...]) -> tuple[Any, tuple[int, ...]]:
  key = (output_index, active_formals)
  cache = _VMAP_ADJ_CACHE.setdefault(callee, {})
  if key not in cache:
    out = callee.outputs[output_index]
    taken = {*callee.input_names, *callee.output_names}
    lam_name = claim_name(f"lam:{callee.output_names[output_index]}", taken)
    lam = Expr.sym(lam_name, out.shape)
    grads = body_cotangents(callee, {output_index: lam}, active_formals)
    adj = callee._inherit_lowering(simplify_cse_fixpoint(concat([grad.reshape((grad.size,)) for grad in grads])))
    dep_memo: dict[tuple[int, int], bool] = {}
    arg_indices = tuple(i for i, inp in enumerate(callee.inputs) if _depends_on(adj, inp, dep_memo))
    inputs = tuple(callee.inputs[i] for i in arg_indices) + (lam,)
    input_names = tuple(callee.input_names[i] for i in arg_indices) + (lam_name,)
    # Suffix by formal index, not name: joined names are not injective ({a_b} vs {a, b}) and
    # lowering dedupes callees by name, so a collision would silently reuse the wrong proc body.
    name = f"{callee.name}_adj{output_index}_" + "_".join(str(i) for i in active_formals)
    fn = Function._from_exprs(name, inputs, [adj], input_names, [claim_name(f"adj:{callee.output_names[output_index]}", taken)])
    cache[key] = (fn, arg_indices)
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

  adj_fn, arg_indices = _vmap_adj_function(callee, output_idx, active_formals)
  primal_specs = [(vmap_expr.args[i], starts[i], strides[i]) for i in arg_indices]
  mapped = vmap(adj_fn, length, [*primal_specs, (cot, 0, slice_size)])
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


_SCAN_ADJ_CACHE: weakref.WeakKeyDictionary[Any, dict[tuple[int, tuple[int, ...]], Any]] = weakref.WeakKeyDictionary()


def _scan_adj_function(callee: Any, output: int, active: tuple[int, ...]) -> Any:
  """One backward step of a scan: from the cotangent ``lam`` of the carry leaving step ``k``, the
  carry entering it and step ``k``'s slices (and the cotangent of step ``k``'s stacked output or of
  its entering carry, by ``output``), return the cotangent of the entering carry and of each
  ``active`` slice."""
  key = (output, active)
  cache = _SCAN_ADJ_CACHE.setdefault(callee, {})
  if key not in cache:
    carry, xs = callee.inputs[0], callee.inputs[1:]
    taken = {*callee.input_names, *callee.output_names}
    lam = Expr.sym(claim_name(f"lam:{callee.input_names[0]}", taken), carry.shape)
    cots, extra = {0: lam}, []
    if output > 0:
      bar = Expr.sym(claim_name(f"lam:{callee.output_names[output]}", taken), callee.outputs[output].shape)
      cots[output] = bar
      extra.append(bar)
    grads = body_cotangents(callee, cots, (0, *(i + 1 for i in active)))
    lam_in = grads[0]
    if output == -1:
      bar = Expr.sym(claim_name(f"lam:{callee.input_names[0]}:t", taken), carry.shape)
      lam_in = lam_in + bar
      extra.append(bar)
    inputs = [lam, carry, *xs, *extra]
    names = [str(e.name) for e in (lam,)] + list(callee.input_names) + [str(e.name) for e in extra]
    body = [callee._inherit_lowering(simplify_cse_fixpoint(g)) for g in (lam_in, *grads[1:])]
    suffix = f"{'t' if output < 0 else output}_" + ("_".join(str(i) for i in active) or "c")
    cache[key] = Function._from_exprs(
      f"{callee.name}_scanadj{suffix}",
      inputs,
      body,
      names,
      [claim_name("adj:carry", taken), *(claim_name(f"adj:{callee.input_names[i + 1]}", taken) for i in active)],
    )
  return cache[key]


def _scan_vjp(expr: Expr, cot: Expr, wrts: Sequence[Expr], dep_memo: dict[tuple[int, int], bool]) -> list[tuple[Expr, Expr]]:
  """A scan's adjoint is a scan over the reversed steps, reading the stored carries backwards; the
  slices' cotangents come back stacked and accumulate into their outer tensors by ``scatter``."""
  callee, length, output = expr.attrs["callee"], int(expr.attrs["length"]), int(expr.attrs["output"])
  init, outers = expr.args[0], expr.args[1:]
  starts, strides = expr.attrs["starts"], expr.attrs["strides"]
  if length == 0:
    return [(init, cot)] if output == 0 else []
  cs = init.size
  active = tuple(i for i, outer in enumerate(outers) if any(_depends_on(outer, wrt, dep_memo) for wrt in wrts))
  fn = _scan_adj_function(callee, output, active)
  carries = _scan_node(callee, init, tuple(outers), starts, strides, length, -1)
  rev_outers = [carries, *outers]
  rev_starts = [(length - 1) * cs, *(s + (length - 1) * st for s, st in zip(starts, strides, strict=True))]
  rev_strides = [-cs, *(-st for st in strides)]
  if output != 0:
    size = cs if output == -1 else callee.outputs[output].size
    rev_outers.append(cot)
    rev_starts.append((length - 1) * size)
    rev_strides.append(-size)
  lam0 = cot if output == 0 else zeros_like(init)
  rev = [_scan_node(fn, lam0, tuple(rev_outers), tuple(rev_starts), tuple(rev_strides), length, k) for k in range(1 + len(active))]
  ret = [(init, rev[0])]
  for stacked, i in zip(rev[1:], active, strict=True):
    size = callee.inputs[i + 1].size
    steps = length - 1 - np.arange(length)
    dest = (starts[i] + steps[:, None] * strides[i] + np.arange(size)[None, :]).reshape(-1)
    ret.append((outers[i], scatter(stacked, dest, outers[i].shape)))
  return ret


_WHILE_ADJ_CACHE: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()


def _while_adj_function(body: Any) -> Any:
  """One backward step of a while loop, masked: inputs ``lam``, the carry entering step ``k``, ``k``
  and the step count ``n``; output the cotangent of the entering carry when ``k < n``, else ``lam``
  unchanged, so steps the loop never took pass the cotangent through."""
  if body not in _WHILE_ADJ_CACHE:
    carry = body.inputs[0]
    taken = {*body.input_names, *body.output_names}
    lam = Expr.sym(claim_name(f"lam:{body.input_names[0]}", taken), carry.shape)
    step, count = Expr.sym(claim_name("step", taken), (1,)), Expr.sym(claim_name("count", taken), (1,), diff=False)
    (back,) = body_cotangents(body, {0: lam}, (0,))
    out = where(step[0] < count[0], back, lam)
    _WHILE_ADJ_CACHE[body] = Function._from_exprs(
      f"{body.name}_whileadj",
      [lam, carry, step, count],
      [body._inherit_lowering(simplify_cse_fixpoint(out))],
      [claim_name("lam", taken), *body.input_names, str(step.name), str(count.name)],
      [claim_name("adj:carry", taken)],
    )
  return _WHILE_ADJ_CACHE[body]


def _while_vjp(expr: Expr, cot: Expr, wrts: Sequence[Expr], dep_memo: dict[tuple[int, int], bool]) -> list[tuple[Expr, Expr]]:
  """Reverse mode through a while loop: a ``max_iter``-step scan backwards over the stored carries,
  where steps at or beyond the step count leave the cotangent unchanged."""
  output = int(expr.attrs["output"])
  if output == 1:
    return []
  if output == -1:
    raise NotImplementedError("reverse mode through the stored carries of a while_loop (reverse over reverse) is not implemented")
  cond, body, max_iter = expr.attrs["cond"], expr.attrs["callee"], int(expr.attrs["max_iter"])
  init = expr.args[0]
  if max_iter == 0:
    return [(init, cot)]
  cs = init.size
  carries = _while_node(cond, body, init, max_iter, -1)
  count = _while_node(cond, body, init, max_iter, 1).reshape((1,))
  steps = Expr.const(np.arange(max_iter, dtype=np.float64))
  specs = ((carries, steps, count), ((max_iter - 1) * cs, max_iter - 1, 0), (-cs, -1, 0))
  (lam0,) = (_scan_node(_while_adj_function(body), cot, specs[0], specs[1], specs[2], max_iter, 0),)
  return [(init, lam0)]


def body_cotangents(fn: Any, cots: dict[int, Expr], wrt: tuple[int, ...]) -> tuple[Expr, ...]:
  """The cotangents of ``fn``'s inputs ``wrt`` (by index) given cotangents of some outputs (by index),
  in terms of its inputs: from its reverse rule when it has one, else by differentiating its body.
  The counterpart of ``forward.body_tangents``, used wherever reverse mode looks inside a Function."""
  if fn.custom_vjp is not None:
    grads = custom_vjp_call(fn, fn.inputs, cots)
    return tuple(grads[i] for i in wrt)
  outs = tuple(fn.outputs[j] for j in cots)
  return vjp(outs, tuple(fn.inputs[i] for i in wrt), tuple(cots.values()))


def vjp(outputs: Sequence[Expr], wrts: Sequence[Expr], cotangents: Sequence[Expr]) -> tuple[Expr, ...]:
  """Reverse-mode derivative: one adjoint per entry of ``wrts``, seeded by ``cotangents``.

  Each cotangent is shaped like its output. One sweep computes the derivative of a scalar with
  respect to every input at once, which is why gradients go through here.
  """
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
    if cot is None or expr.op in {ExprOp.INPUT, ExprOp.CONST} or expr.op in PREDICATE_OPS or not needed(expr):
      continue
    if expr.op in (ExprOp.VMAP, ExprOp.SCAN, ExprOp.WHILE):
      rule = {ExprOp.VMAP: _vmap_vjp, ExprOp.SCAN: _scan_vjp, ExprOp.WHILE: _while_vjp}[ExprOp(expr.op)]
      pairs = rule(expr, cot, wrts, dep_memo)
      for arg, arg_cot in pairs:
        if arg.id in expr_ids:
          adjoints[arg.id] = arg_cot if arg.id not in adjoints else adjoints[arg.id] + arg_cot
      continue
    for arg, arg_cot in zip(expr.args, _local_vjp(expr, cot), strict=True):
      if arg.id in expr_ids:
        adjoints[arg.id] = arg_cot if arg.id not in adjoints else adjoints[arg.id] + arg_cot

  return tuple(adjoints.get(wrt.id, zeros_like(wrt)) for wrt in wrts)


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


def _local_vjp(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  if expr.op == ExprOp.NEG:
    return (-cot,)
  if expr.op == ExprOp.ADD:
    return (_unbroadcast(cot, args[0].shape, expr.shape), _unbroadcast(cot, args[1].shape, expr.shape))
  if expr.op == ExprOp.SUB:
    return (_unbroadcast(cot, args[0].shape, expr.shape), _unbroadcast(-cot, args[1].shape, expr.shape))
  if expr.op == ExprOp.MUL:
    return (_unbroadcast(cot * args[1], args[0].shape, expr.shape), _unbroadcast(cot * args[0], args[1].shape, expr.shape))
  if expr.op == ExprOp.DIV:
    return (
      _unbroadcast(cot / args[1], args[0].shape, expr.shape),
      _unbroadcast(-((cot / args[1]) * expr), args[1].shape, expr.shape),
    )
  if expr.op == ExprOp.POW:
    return (
      _unbroadcast(cot * args[1] * (args[0] ** (args[1] - 1)), args[0].shape, expr.shape),
      _unbroadcast(cot * expr * args[0].log(), args[1].shape, expr.shape),
    )
  if expr.op == ExprOp.SIN:
    return (cot * args[0].cos(),)
  if expr.op == ExprOp.COS:
    return (-cot * args[0].sin(),)
  if expr.op == ExprOp.TAN:
    return (cot / (args[0].cos() ** 2),)
  if expr.op == ExprOp.ASIN:
    return (cot / (1 - args[0] ** 2).sqrt(),)
  if expr.op == ExprOp.ACOS:
    return (-cot / (1 - args[0] ** 2).sqrt(),)
  if expr.op == ExprOp.ATAN:
    return (cot / (1 + args[0] ** 2),)
  if expr.op == ExprOp.ATAN2:
    y, x = args
    denom = x * x + y * y
    return (_unbroadcast(cot * x / denom, y.shape, expr.shape), _unbroadcast(-cot * y / denom, x.shape, expr.shape))
  if expr.op == ExprOp.SINH:
    return (cot * args[0].cosh(),)
  if expr.op == ExprOp.COSH:
    return (cot * args[0].sinh(),)
  if expr.op == ExprOp.TANH:
    return (cot * (1 - expr * expr),)
  if expr.op == ExprOp.ERF:
    return (cot * (2 / np.sqrt(np.pi)) * (-(args[0] ** 2)).exp(),)
  if expr.op == ExprOp.EXP:
    return (cot * expr,)
  if expr.op == ExprOp.LOG:
    return (cot / args[0],)
  if expr.op == ExprOp.SQRT:
    return (cot / (2 * expr),)
  if expr.op == ExprOp.ABS:
    return (cot * sign(args[0]),)
  if expr.op in {ExprOp.FLOOR, ExprOp.CEIL}:
    raise NotImplementedError(f"VJP for nonsmooth op {expr.op!r} is not implemented")
  if expr.op in {ExprOp.MINIMUM, ExprOp.MAXIMUM}:
    w = extremum_weight(expr)
    return (_unbroadcast(cot * w, args[0].shape, expr.shape), _unbroadcast(cot * (1.0 - w), args[1].shape, expr.shape))
  if expr.op in {ExprOp.MAX, ExprOp.MIN}:
    return (cot * reduce_weights(expr),)
  if expr.op in {ExprOp.SEGMENT_MAX, ExprOp.SEGMENT_MIN}:
    return (gather(cot, expr.attrs["indices"]) * segment_weights(expr),)
  if expr.op == ExprOp.INDEX_ADD:
    return (cot, gather(cot, expr.attrs["indices"]))
  if expr.op == ExprOp.INDEX_SET:
    return (index_set(cot, expr.attrs["indices"], np.zeros(args[1].size)), gather(cot, expr.attrs["indices"]))
  if expr.op == ExprOp.SELECT:
    cond, a, b = args
    zero = as_expr(0.0)
    return (
      zeros_like(cond),
      _unbroadcast(where(cond, cot, zero), a.shape, expr.shape),
      _unbroadcast(where(cond, zero, cot), b.shape, expr.shape),
    )
  if expr.op == ExprOp.COPYSIGN:
    x, s = args
    return (_unbroadcast(cot * copysign(1.0, x) * copysign(1.0, s), x.shape, expr.shape), zeros_like(s))
  if expr.op == ExprOp.CAST:
    return (cast(cot, args[0].type.dtype) if expr.type.diff else zeros_like(args[0]),)
  if expr.op == ExprOp.SUM:
    return (cot * _ones_like(args[0]),)
  if expr.op == ExprOp.RESHAPE:
    return (cot.reshape(args[0].shape),)
  if expr.op == ExprOp.TRANSPOSE:
    axes = expr.attrs["axes"]
    inv = tuple(int(np.argsort(axes)[i]) for i in range(len(axes)))
    return (cot.transpose(inv),)
  if expr.op == ExprOp.SLICE:
    indices = np.arange(args[0].size).reshape(args[0].shape)[expr.attrs["index"]]
    return (scatter(cot, indices, args[0].shape),)
  if expr.op == ExprOp.GATHER:
    return (_gather_vjp(cot, expr.attrs["indices"], args[0].shape),)
  if expr.op == ExprOp.SCATTER:
    return (gather(cot, expr.attrs["indices"]),)
  if expr.op == ExprOp.STACK:
    return _stack_vjp(cot, len(args), expr.attrs.get("axis", 0))
  if expr.op == ExprOp.CONCAT:
    return _concat_vjp(cot, args, expr.attrs.get("axis", 0))
  if expr.op == ExprOp.MATMUL:
    return _matmul_vjp(args[0], args[1], cot)
  if expr.op == ExprOp.CALL and expr.attrs["callee"].custom_vjp is not None:
    return custom_vjp_call(expr.attrs["callee"], args, {expr.attrs["output"]: cot})
  if expr.op == ExprOp.CALL:
    callee = expr.attrs["callee"]
    output_idx = expr.attrs["output"]
    callee_out = callee.outputs[output_idx]
    # Differentiate the callee body against a fresh cotangent symbol, then graft the real ``cot``
    # in via the same substitution that maps formals to actuals. Passing ``cot`` directly into the
    # inner vjp would make it part of the substituted graph: if the caller reuses a callee formal
    # symbol (the usual construction pattern), occurrences of that symbol *inside the cotangent*
    # would be rewritten to this call's actuals, corrupting the adjoint.
    lam = Expr.sym(claim_name(f"lam:{callee.output_names[output_idx]}", {*callee.input_names, *callee.output_names}), callee_out.shape)
    replacements = dict(zip((inp.id for inp in callee.inputs), args, strict=True))
    replacements[lam.id] = cot
    return tuple(_substitute(g, replacements) for g in vjp((callee_out,), callee.inputs, (lam,)))
  if expr.op == ExprOp.SOLVER_CALL:
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
