"""Lower ConcreteFunction calls, mapped calls and carried loops, including solver oracle dependencies."""

from __future__ import annotations

import numpy as np

from ...ir import program as p
from ...ir.expr import Expr, ExprOp, invocation_key
from ...ir.program import ProgramNode, ProgramOp, RangeKind
from ...ir.types import DType, dtypes
from ...function.concrete import ConcreteFunction
from .ctx import LowerCtx, LoweringError, lowers, _lower_to_proc


def _ensure_callee(ctx: LowerCtx, callee: ConcreteFunction) -> None:
  from ...solvers.graph import is_solver_function, solver_callees

  if is_solver_function(callee):
    # Opaque: the solver wrapper is rendered by codegen/solver (rule 6), not lowered. Its body is
    # SOLVER_CALL (no lowering rule). We still lower the oracle Functions the wrapper drives.
    ctx.solver_fns[callee] = callee
    for oracle in solver_callees(callee):
      _ensure_callee(ctx, oracle)
    return
  if callee not in ctx.callees:
    ctx.callees[callee] = _lower_to_proc(
      callee, ctx.callees, ctx.solver_fns, ctx.program_names, ctx.symbols, ctx.residuals, observe_expr=ctx.observe_expr
    )


@lowers(ExprOp.CALL, ExprOp.EXTERN)
def _lower_call(ctx: LowerCtx, node: Expr) -> None:
  """Write all used siblings with one call; unused results get scratch storage."""
  callee: ConcreteFunction = node.attrs["callee"]
  if node.op != ExprOp.EXTERN:
    _ensure_callee(ctx, callee)
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
        "callee": ctx.symbols[callee],
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
  """A ``length``-iteration loop calling the callee with pointer-offset VIEW args. Iteration ``it``
  reads ``outer_k[start_k + it·stride_k ...]`` and writes all used outputs into their mapped buffers."""
  callee: ConcreteFunction = node.attrs["callee"]
  length = int(node.attrs["length"])
  starts = tuple(int(s) for s in node.attrs["starts"])
  strides = tuple(int(s) for s in node.attrs["strides"])
  _ensure_callee(ctx, callee)
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
  it = p.var(loop)
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
    attrs={"callee": ctx.symbols[callee], "n_in": len(in_args), "n_out": len(out_args), "returns": ()},
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
  used = {int(sibling.attrs["output"]): sibling for sibling in ctx.used_results[invocation_key(node)]}
  traj = {c for c in range(n_carry) if n_carry + n_ys + 1 + c in used}
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
  slots = [ctx.new_private(c.type.dtype, ((length + 1 if j in traj else 2) * sizes[j],)) for j, c in enumerate(callee.inputs[:n_carry])]

  def slot(j: int, at: ProgramNode) -> ProgramNode:
    """The offset of carry ``j``'s slot after ``at`` executed steps."""
    if j in traj:
      at = p.sub(count, at) if reverse else at
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
  ctx.statements.append(_call(ctx, callee, inputs(callee), _proc_results(ctx, callee, outs)))
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


def _call(ctx: LowerCtx, fn: ConcreteFunction, ins: list[ProgramNode], outs: list[ProgramNode]) -> ProgramNode:
  return ProgramNode(ProgramOp.CALL, (*ins, *outs), attrs={"callee": ctx.symbols[fn], "n_in": len(ins), "n_out": len(outs), "returns": ()})
