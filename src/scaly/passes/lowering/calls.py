"""Lower ConcreteFunction calls and mapped calls, including solver oracle dependencies."""

from __future__ import annotations

from ...ir import program as p
from ...ir.expr import Expr, ExprOp, invocation_key
from ...ir.program import ProgramNode, ProgramOp, RangeKind
from ...function.concrete import ConcreteFunction
from .ctx import LowerCtx, lowers, _lower_to_proc


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


@lowers(ExprOp.CALL)
def _lower_call(ctx: LowerCtx, node: Expr) -> None:
  """Write all used siblings with one call; unused results get scratch storage."""
  callee: ConcreteFunction = node.attrs["callee"]
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
      ProgramOp.CALL, tuple(in_bufs + out_bufs), attrs={"callee": ctx.symbols[callee], "n_in": len(in_bufs), "n_out": len(out_bufs), "returns": ()}
    )
  )
  for i in used:
    if (layout := layouts[i]) is not None:
      ctx.reorder(out_bufs[i], results[i], layout.inverse_permutation, buffers[i])


def _result_buffers(ctx: LowerCtx, node: Expr) -> tuple[list[ProgramNode], dict[int, Expr]]:
  callee = node.attrs["callee"]
  results = callee.results if callee in ctx.residuals else callee.outputs
  key = invocation_key(node)
  assert key is not None
  siblings = ctx.used_results[key]
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
  rng = ProgramNode(rng.op, rng.args, {**rng.attrs, "mapped": True}, rng.dtype)
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
    ProgramOp.CALL, tuple(in_args + out_args), attrs={"callee": ctx.symbols[callee], "n_in": len(in_args), "n_out": len(out_args), "returns": ()}
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
