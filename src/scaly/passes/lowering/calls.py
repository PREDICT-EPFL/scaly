"""Lower calls, maps, carried loops and conditionals, external functions included."""

from __future__ import annotations

import numpy as np

from ...ir import program as p
from ...ir.expr import Expr, ExprOp, invocation_key
from ...ir.program import ProgramNode, ProgramOp, RangeKind
from ...ir.types import DType, dtypes
from ...function.concrete import ConcreteFunction
from ...ir.support import Support, _mapped_support_groups
from ..program.scalarize import _DataDependentAddress, expand_static_loop
from .ctx import LowerCtx, LoweringError, lowers, _lower_to_proc, _normalize_function
from .in_place import carry_reuse


STATIC_LOOP_BUDGET = 256


def _ensure_callee(ctx: LowerCtx, callee: ConcreteFunction, arguments: tuple[Support, ...] | None = None) -> str:
  if arguments is not None:
    arguments = tuple(
      arg.support if support is not None and len(support) == arg.size else support for arg, support in zip(callee.inputs, arguments, strict=True)
    )
  if arguments == tuple(arg.support for arg in callee.inputs):
    arguments = None
  key = callee if arguments is None else (callee, arguments)
  if key not in ctx.callees and arguments is not None and sum(isinstance(k, tuple) and k[0] is callee for k in ctx.callees) >= 32:
    return _ensure_callee(ctx, callee)
  if key not in ctx.callees:
    symbol = ctx.symbols[callee] if arguments is None else ctx.program_names.procedure(f"{ctx.symbols[callee]}_support")
    ctx.callees[key] = _lower_to_proc(
      callee,
      ctx.callees,
      ctx.program_names,
      ctx.symbols,
      ctx.residuals,
      observe_expr=ctx.observe_expr,
      argument_support=arguments,
      symbol=symbol,
    )
  return ctx.callees[key].attrs["name"]


@lowers(ExprOp.CALL, ExprOp.EXTERN)
def _lower_call(ctx: LowerCtx, node: Expr) -> None:
  """Write all used siblings with one call; unused results get scratch storage."""
  callee: ConcreteFunction = node.attrs["callee"]
  symbol = ctx.symbols[callee] if node.op == ExprOp.EXTERN else _ensure_callee(ctx, callee, tuple(ctx.support_of(arg) for arg in node.args))
  results = callee.results if callee in ctx.residuals else callee.outputs
  layouts = (*callee.output_tree.layouts, *(None for _ in results[len(callee.outputs) :]))
  buffers, used = _result_buffers(ctx, node)
  out_bufs = [
    ctx.new_private(result.type.dtype, result.type.storage_shape) if i in used and layouts[i] is not None else buf
    for i, (buf, result) in enumerate(zip(buffers, results, strict=True))
  ]
  in_bufs = [
    ctx.buf_of(actual) if layout is None else ctx.reorder(ctx.buf_of(actual), actual, layout.permutation)
    for actual, layout in zip(node.args, callee.input_tree.layouts, strict=True)
  ]
  ctx.statements.append(
    ProgramNode(
      ProgramOp.CALL,
      tuple(in_bufs + out_bufs),
      attrs={
        "callee": symbol,
        "n_in": len(in_bufs),
        "n_out": len(out_bufs),
        "returns": (),
        **({"external": True} if node.op == ExprOp.EXTERN else {}),
      },
      dtype=dtypes.float64,
    )
  )
  for i in used:
    if (layout := layouts[i]) is not None:
      ctx.reorder(out_bufs[i], results[i], layout.inverse_permutation, buffers[i])


def _result_buffers(ctx: LowerCtx, node: Expr) -> tuple[list[ProgramNode], dict[int, Expr]]:
  callee = node.attrs["callee"]
  results = callee.results if callee in ctx.residuals else callee.outputs
  siblings = ctx.used_results[invocation_key(node)]
  used = {int(sibling.attrs["output"]): sibling for sibling in siblings}
  buffers = [
    ctx.alloc_tmp(used[i]) if i in used else ctx.new_private(result.type.dtype, result.type.storage_shape) for i, result in enumerate(results)
  ]
  for sibling in siblings:
    ctx.value_buffers[sibling.id] = buffers[int(sibling.attrs["output"])].attrs["name"]
  return buffers, used


@lowers(ExprOp.VMAP)
def _lower_vmap(ctx: LowerCtx, node: Expr) -> None:
  callee = node.attrs["callee"]
  groups = _mapped_support_groups(node, ctx.support_of)
  for arguments, stages in groups.items():
    _lower_vmap_group(ctx, node, np.asarray(stages, dtype=np.int64), _ensure_callee(ctx, callee, arguments))
  if not groups:
    _result_buffers(ctx, node)


def _lower_vmap_group(ctx: LowerCtx, node: Expr, stages: np.ndarray, symbol: str) -> None:
  """A ``length``-iteration loop calling the callee with pointer-offset VIEW args. Iteration ``it``
  reads ``outer_k[start_k + it·stride_k ...]`` and writes all used outputs into their mapped buffers."""
  callee: ConcreteFunction = node.attrs["callee"]
  length = len(stages)
  starts = tuple(int(s) for s in node.attrs["starts"])
  strides = tuple(int(s) for s in node.attrs["strides"])
  results = callee.results if callee in ctx.residuals else callee.outputs
  layouts = (*callee.output_tree.layouts, *(None for _ in results[len(callee.outputs) :]))
  buffers, used = _result_buffers(ctx, node)
  scratch = [
    ctx.new_private(result.type.dtype, result.type.storage_shape) if i in used and layouts[i] is not None else buf
    for i, (buf, result) in enumerate(zip(buffers, results, strict=True))
  ]
  if length == 0:
    return
  loop = ctx.names.allocate(f"it_{ctx.buf_of(node).attrs['name']}")
  rng = p.range_(loop, 0, length, kind=RangeKind.GLOBAL)
  rng = ProgramNode(rng.op, rng.args, {**rng.attrs, "mapped": True}, dtype=rng.dtype)
  it = p.var(loop) if np.array_equal(stages, np.arange(length)) else ctx.index_at(stages, p.var(loop), shared=True)
  in_args = []
  before = len(ctx.statements)
  for k, outer in enumerate(node.args):
    off = p.add(p.const_int(starts[k]), p.mul(p.const_int(strides[k]), it)) if strides[k] else p.const_int(starts[k])
    view = p.view(ctx.buf_of(outer), [off])
    layout = callee.input_tree.layouts[k]
    in_args.append(view if layout is None else ctx.reorder(view, callee.inputs[k], layout.permutation))
  out_args = []
  for i, buf in enumerate(scratch):
    if i in used and layouts[i] is None:
      slice_size = int(used[i].attrs["slice_size"])
      off = p.mul(it, p.const_int(slice_size)) if slice_size != 1 else it
      out_args.append(p.view(buf, [off]))
    else:
      out_args.append(buf)
  call = ProgramNode(
    ProgramOp.CALL,
    tuple(in_args + out_args),
    attrs={"callee": symbol, "n_in": len(in_args), "n_out": len(out_args), "returns": ()},
    dtype=dtypes.float64,
  )
  prologue = ctx.statements[before:]
  del ctx.statements[before:]
  body = [*prologue, call]
  for i in used:
    if (layout := layouts[i]) is not None:
      off = p.mul(it, p.const_int(int(used[i].attrs["slice_size"])))
      ctx.reorder(scratch[i], results[i], layout.inverse_permutation, p.view(buffers[i], [off]))
  body.extend(ctx.statements[before:])
  del ctx.statements[before:]
  ctx.statements.extend(statement for statement in body if statement.op == ProgramOp.BUFFER)
  ctx.statements.append(p.for_(rng, [statement for statement in body if statement.op != ProgramOp.BUFFER]))


@lowers(ExprOp.COND)
def _lower_cond(ctx: LowerCtx, node: Expr) -> None:
  branches = node.attrs["branches"]
  used = {int(sibling.attrs["output"]): sibling for sibling in ctx.used_results[invocation_key(node)]}
  buffers = [
    ctx.alloc_tmp(used[i]) if i in used else ctx.new_private(out.type.dtype, out.type.storage_shape) for i, out in enumerate(branches[0].outputs)
  ]
  operands = [ctx.buf_of(arg) for arg in node.args[1:]]
  bodies = []
  for branch in branches:
    _ensure_callee(ctx, branch)
    before = len(ctx.statements)
    ins = [
      buf if layout is None else ctx.reorder(buf, arg, layout.permutation)
      for buf, arg, layout in zip(operands, node.args[1:], branch.input_tree.layouts, strict=True)
    ]
    outs = [
      buf if layout is None else ctx.new_private(out.type.dtype, out.type.storage_shape)
      for buf, out, layout in zip(buffers, branch.outputs, branch.output_tree.layouts, strict=True)
    ]
    ctx.statements.append(_call(ctx, branch, ins, _proc_results(ctx, branch, outs)))
    for i, layout in enumerate(branch.output_tree.layouts):
      if layout is not None:
        ctx.reorder(outs[i], branch.outputs[i], layout.inverse_permutation, buffers[i])
    body = ctx.statements[before:]
    del ctx.statements[before:]
    ctx.statements.extend(stmt for stmt in body if stmt.op == ProgramOp.BUFFER)
    bodies.append([stmt for stmt in body if stmt.op != ProgramOp.BUFFER])
  selector = p.load(p.view(ctx.buf_of(node.args[0]), [p.const_int(0)]))
  body = bodies[-1]
  for i in reversed(range(len(branches) - 1)):
    if node.args[0].type.dtype.is_bool:
      flag = ProgramNode(ProgramOp.NOT, (selector,), dtype=dtypes.bool_)
    else:
      flag = ProgramNode(ProgramOp.LE, (selector, ProgramNode(ProgramOp.CONST_INT, attrs={"value": i}, dtype=selector.dtype)), dtype=dtypes.bool_)
    body = [p.if_(flag, bodies[i], body)]
  ctx.statements.extend(body)


def _zero(dtype: DType) -> ProgramNode:
  if dtype.is_floating:
    return p.const_float(0.0, dtype)
  return p.const_int(0) if dtype == dtypes.int64 else ProgramNode(ProgramOp.CAST, (p.const_int(0),), dtype=dtype)


def _select(cond: ProgramNode, a: ProgramNode, b: ProgramNode) -> ProgramNode:
  return ProgramNode(ProgramOp.SELECT, (cond, a, b), dtype=a.dtype)


def _proc_results(ctx: LowerCtx, fn: ConcreteFunction, buffers: list[ProgramNode]) -> list[ProgramNode]:
  """Scratch storage for the results a procedure writes beyond ``buffers``: residuals another call reads."""
  results = fn.results if fn in ctx.residuals else fn.outputs
  return buffers + [ctx.new_private(r.type.dtype, r.type.storage_shape) for r in results[len(buffers) :]]


@lowers(ExprOp.LOOP)
def _lower_loop(ctx: LowerCtx, node: Expr) -> None:
  """One carried range ``[call cond; break_if !flag;] call body`` that writes every used result.

  A carry lives in two alternating slots, or in ``length + 1`` slots, slot ``j`` entering step ``j``
  (``j + 1`` in reverse), when its trajectory is used. Stacked outputs and trajectory slots past
  the trip count are zeroed only when a condition or a trip operand can end the loop early.
  """
  callee: ConcreteFunction = node.attrs["callee"]
  cond: ConcreteFunction | None = node.attrs["cond"]
  n_carry, length, index, reverse = node.attrs["n_carry"], node.attrs["length"], node.attrs["index"], node.attrs["reverse"]
  starts, strides = node.attrs["starts"], node.attrs["strides"]
  carries, outers = node.args[:n_carry], node.args[n_carry : n_carry + len(starts)]
  trip = node.args[n_carry + len(starts)] if len(node.args) > n_carry + len(starts) else None
  n_ys = len(callee.outputs) - n_carry
  for fn in (callee, *(() if cond is None else (cond,))):
    _ensure_callee(ctx, fn)
    layouts = (*fn.input_tree.layouts[:n_carry], *(fn.output_tree.layouts[:n_carry] if fn is callee else ()))
    if any(layout is not None and not np.array_equal(layout.permutation, np.arange(len(layout.permutation))) for layout in layouts):
      raise LoweringError(f"loop carries of {fn.name!r} must use the canonical sparse layout")
  siblings = ctx.used_results[invocation_key(node)]
  used = {int(sibling.attrs["output"]): sibling for sibling in siblings}
  traj = {c for c in range(n_carry) if n_carry + n_ys + 1 + c in used}
  symbol = ctx.symbols[callee]
  normalized = _normalize_function(callee, residuals=callee in ctx.residuals)
  chains = carry_reuse(normalized, n_carry, length, index, traj)
  reuse = set(chains)
  for j in node.attrs["in_place"]:
    if j not in reuse:
      reason = " because its trajectory is used" if j in traj else ""
      raise LoweringError(f"cannot prove in_place carry {callee.input_names[j]!r} of {callee.name!r}{reason}")
  if reuse:
    symbol = ctx.program_names.procedure(f"{callee.name}_inplace")
    variant = normalized._replace(name=symbol)
    ctx.symbols[variant] = symbol
    ctx.callees[variant] = _lower_to_proc(
      variant,
      ctx.callees,
      ctx.program_names,
      ctx.symbols,
      ctx.residuals,
      observe_expr=ctx.observe_expr,
      inout_carries=chains,
    )
  early = cond is not None or trip is not None
  name = ctx.names.allocate("i_loop")
  i = p.var(name)
  if trip is None:
    count = p.const_int(length)
  else:
    t = p.load(p.view(ctx.buf_of(trip), [p.const_int(0)]))
    t = _select(ProgramNode(ProgramOp.LT, (t, p.const_int(0)), dtype=dtypes.bool_), p.const_int(0), t)
    count = _select(ProgramNode(ProgramOp.LT, (t, p.const_int(length)), dtype=dtypes.bool_), t, p.const_int(length))
  last = p.sub(count, p.const_int(1))
  k = p.sub(last, i) if reverse else i
  sizes = [c.type.storage_size for c in callee.inputs[:n_carry]]
  for arg in (*carries, *outers):
    ctx.buf_of(arg)
  before_loop = len(ctx.statements)
  existing_buffers = list(ctx.buffers.values())
  slots = [
    ctx.new_private(c.type.dtype, ((length + 1 if j in traj else 1 if j in reuse else 2) * sizes[j],)) for j, c in enumerate(callee.inputs[:n_carry])
  ]

  def slot(j: int, at: ProgramNode) -> ProgramNode:
    """The offset of carry ``j``'s slot after ``at`` executed steps."""
    if j in traj:
      at = p.sub(count, at) if reverse else at
    elif j in reuse:
      at = p.const_int(0)
    else:
      at = p.mod(at, p.const_int(2))
    return p.mul(at, p.const_int(sizes[j]))

  def copy(dst: ProgramNode, dst_at: ProgramNode, src: ProgramNode, src_at: ProgramNode, size: int) -> None:
    if size:
      ranges, (e,) = ctx.nest((size,), f"c_{dst.attrs['name']}")
      ctx.emit(ranges, [p.store(p.view(dst, [p.add(dst_at, e)]), p.load(p.view(src, [p.add(src_at, e)])))])

  for j, init in enumerate(carries):
    copy(slots[j], slot(j, p.const_int(0)), ctx.buf_of(init), p.const_int(0), sizes[j])
  ys = []
  for y in range(n_ys):
    out = callee.outputs[n_carry + y].type
    ys.append(ctx.alloc_tmp(used[n_carry + y]) if n_carry + y in used else ctx.new_private(out.dtype, out.storage_shape))
  before = len(ctx.statements)
  args = [p.view(slots[j], [slot(j, i)]) for j in range(n_carry)]
  if index:
    step = ctx.new_private(dtypes.int64, (1,))
    ctx.statements.append(p.store(p.view(step, [p.const_int(0)]), k))
    args.append(step)
  views = [
    p.view(ctx.buf_of(outer), [p.add(p.const_int(start), p.mul(p.const_int(stride), k)) if stride else p.const_int(start)])
    for outer, start, stride in zip(outers, starts, strides, strict=True)
  ]

  def inputs(fn: ConcreteFunction) -> list[ProgramNode]:
    """The step's arguments in ``fn``'s declared input layouts."""
    formals = zip(views, fn.inputs[n_carry + index :], fn.input_tree.layouts[n_carry + index :], strict=True)
    return args + [view if layout is None else ctx.reorder(view, formal, layout.permutation) for view, formal, layout in formals]

  if cond is not None:
    flag = ctx.new_private(dtypes.bool_, (1,))
    ctx.statements.append(_call(ctx, cond, inputs(cond), _proc_results(ctx, cond, [flag])))
    flag_value = p.load(p.view(flag, [p.const_int(0)]))
    ctx.statements.append(p.break_if(ProgramNode(ProgramOp.NOT, (flag_value,), dtype=dtypes.bool_)))
  outs = [p.view(slots[j], [slot(j, p.add(i, p.const_int(1)))]) for j in range(n_carry)]
  scratch = []
  for y, buf in enumerate(ys):
    result, layout = callee.outputs[n_carry + y], callee.output_tree.layouts[n_carry + y]
    if n_carry + y not in used:
      outs.append(buf)
    elif layout is None:
      outs.append(p.view(buf, [p.mul(k, p.const_int(result.type.storage_size))]))
    else:
      scratch.append((y, ctx.new_private(result.type.dtype, result.type.storage_shape)))
      outs.append(scratch[-1][1])
  ins = inputs(callee)
  results = _proc_results(ctx, callee, outs)
  results = [out for j, out in enumerate(results) if j not in reuse]
  ctx.statements.append(
    ProgramNode(
      ProgramOp.CALL,
      (*ins, *results),
      attrs={"callee": symbol, "n_in": len(ins), "n_out": len(results), "inout": tuple(sorted(reuse)), "returns": ()},
      dtype=dtypes.float64,
    )
  )
  for y, buf in scratch:
    result = callee.outputs[n_carry + y]
    ctx.reorder(
      buf, result, callee.output_tree.layouts[n_carry + y].inverse_permutation, p.view(ys[y], [p.mul(k, p.const_int(result.type.storage_size))])
    )
  body = ctx.statements[before:]
  del ctx.statements[before:]
  ctx.statements.extend(statement for statement in body if statement.op == ProgramOp.BUFFER)
  rng = p.range_(name, 0, count, carried=True)
  ctx.statements.append(p.for_(rng, [statement for statement in body if statement.op != ProgramOp.BUFFER], exit_var=cond is not None))
  taken = i if cond is not None else count
  for j in range(n_carry):
    if j in used:
      copy(ctx.alloc_tmp(used[j]), p.const_int(0), slots[j], slot(j, taken), sizes[j])
  if n_carry + n_ys in used:
    ctx.statements.append(p.store(p.view(ctx.alloc_tmp(used[n_carry + n_ys]), [p.const_int(0)]), taken))
  for j in traj:
    sibling = used[n_carry + n_ys + 1 + j]
    alias = ctx.new_alias(sibling.type.dtype, sibling.shape, slots[j].attrs["name"], sizes[j] if reverse else 0)
    ctx.value_buffers[sibling.id] = alias.attrs["name"]
    ctx.statements.append(alias)
  if early:
    fills = [(ys[y], 0, callee.outputs[n_carry + y].type) for y in range(n_ys) if n_carry + y in used]
    fills += [(slots[j], sizes[j] if reverse else 0, callee.inputs[j].type) for j in traj]
    for buf, offset, type_ in fills:
      if type_.storage_size and length:
        e = ctx.names.allocate(f"z_{buf.attrs['name']}")
        start = p.add(p.const_int(offset), p.mul(taken, p.const_int(type_.storage_size)))
        stop = offset + length * type_.storage_size
        ctx.statements.append(p.for_(p.range_(e, start, stop), [p.store(p.view(buf, [p.var(e)]), _zero(type_.dtype))]))

  hints = {sibling.lowering for sibling in siblings}
  if not early and not hints & {"block", "opaque"}:
    procs = {proc.attrs["name"]: proc for proc in ctx.callees.values()}
    cost = _loop_cost(p.proc("loop_step", (), body), procs)
    if cost is not None and ("scalar" in hints or length * cost <= STATIC_LOOP_BUDGET):
      results = [(ctx.raw_buf_of(sibling), sibling.type.storage_size) for sibling in used.values()]
      try:
        expanded = expand_static_loop(existing_buffers, ctx.statements[before_loop:], results, procs, ctx.names)
      except _DataDependentAddress:
        pass
      else:
        ctx.statements[before_loop:] = expanded

  for sibling in siblings:
    if sibling.id not in ctx.value_buffers:
      ctx.value_buffers[sibling.id] = ctx.buf_of(used[int(sibling.attrs["output"])]).attrs["name"]


def _loop_cost(proc: ProgramNode, procs: dict[str, ProgramNode]) -> int | None:
  """Count scalar right-hand sides through every static range and callee, or reject unsafe substitution."""
  if proc.attrs.get("lowering") in ("block", "opaque"):
    return None
  total = 0
  stack = [(stmt, 1) for stmt in proc.args[proc.attrs["param_count"] :]]
  while stack:
    stmt, factor = stack.pop()
    if stmt.dtype is not None and stmt.dtype not in (dtypes.float64, dtypes.int64, dtypes.bool_):
      return None
    if stmt.op == ProgramOp.CALL:
      callee = procs.get(stmt.attrs["callee"])
      if callee is None or callee.attrs.get("lowering") in ("block", "opaque"):
        return None
      stack.extend((child, factor) for child in callee.args[callee.attrs["param_count"] :])
    elif stmt.op == ProgramOp.FOR:
      rng = stmt.args[0]
      if any(arg.op != ProgramOp.CONST_INT for arg in rng.args):
        return None
      start, stop, step = (arg.attrs["value"] for arg in rng.args)
      if step <= 0:
        return None
      stack.extend((child, factor * len(range(start, stop, step))) for child in stmt.args[1:])
    elif stmt.op in (ProgramOp.STORE, ProgramOp.ASSIGN):
      total += factor
    elif stmt.op != ProgramOp.BUFFER:
      return None
  return max(1, total)


def _call(ctx: LowerCtx, fn: ConcreteFunction, ins: list[ProgramNode], outs: list[ProgramNode]) -> ProgramNode:
  return ProgramNode(
    ProgramOp.CALL, (*ins, *outs), attrs={"callee": ctx.symbols[fn], "n_in": len(ins), "n_out": len(outs), "returns": ()}, dtype=dtypes.float64
  )
