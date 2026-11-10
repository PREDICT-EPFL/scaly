"""Lower ConcreteFunction calls and mapped calls, including solver oracle dependencies."""

from __future__ import annotations

from ...ir import program as p
from ...ir.expr import Expr, ExprOp
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
  """An expression CALL output: emit one Program-IR CALL writing all callee outputs into scratch
  buffers (deduped per unique invocation), then map this node to the selected output buffer."""
  callee: ConcreteFunction = node.attrs["callee"]
  out_idx = int(node.attrs["output"])
  arg_names = tuple(ctx.value_buffers[a.id] for a in node.args)
  key = (callee, arg_names)
  if key not in ctx.call_invocations:
    _ensure_callee(ctx, callee)
    results = callee.results if callee in ctx.residuals else callee.outputs
    out_bufs = [ctx.new_private(o.type.dtype, o.type.storage_shape) for o in results]
    in_bufs = [
      ctx.buf_of(actual) if layout is None else ctx.reorder(ctx.buf_of(actual), actual, layout.permutation)
      for actual, layout in zip(node.args, callee.input_tree.layouts, strict=True)
    ]
    ctx.statements.append(
      ProgramNode(
        ProgramOp.CALL, tuple(in_bufs + out_bufs), attrs={"callee": ctx.symbols[callee], "n_in": len(in_bufs), "n_out": len(out_bufs), "returns": ()}
      )
    )
    layouts = (*callee.output_tree.layouts, *(None for _ in results[len(callee.outputs) :]))
    canonical = [
      buf if layout is None else ctx.reorder(buf, expr, layout.inverse_permutation)
      for buf, expr, layout in zip(out_bufs, results, layouts, strict=True)
    ]
    ctx.call_invocations[key] = tuple(b.attrs["name"] for b in canonical)
  ctx.value_buffers[node.id] = ctx.call_invocations[key][out_idx]


@lowers(ExprOp.VMAP)
def _lower_vmap(ctx: LowerCtx, node: Expr) -> None:
  """A ``length``-iteration loop calling the callee with pointer-offset VIEW args. Iteration ``it``
  reads ``outer_k[start_k + it·stride_k ...]`` and writes the selected output into ``out[it·slice_size ...]``."""
  callee: ConcreteFunction = node.attrs["callee"]
  out_idx = int(node.attrs["output"])
  length = int(node.attrs["length"])
  starts = tuple(int(s) for s in node.attrs["starts"])
  strides = tuple(int(s) for s in node.attrs["strides"])
  slice_size = int(node.attrs["slice_size"])
  _ensure_callee(ctx, callee)
  out = ctx.alloc_tmp(node)
  # Other callee results are written every iteration but discarded: one reused scratch each.
  results = callee.results if callee in ctx.residuals else callee.outputs
  layouts = (*callee.output_tree.layouts, *(None for _ in results[len(callee.outputs) :]))
  scratch = [out if i == out_idx and layouts[i] is None else ctx.new_private(o.type.dtype, o.type.storage_shape) for i, o in enumerate(results)]
  if length == 0:
    return
  loop = ctx.names.allocate(f"it_{out.attrs['name']}")
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
  for i, sbuf in enumerate(scratch):
    if i == out_idx and layouts[i] is None:
      off = p.mul(it, p.const_int(slice_size)) if slice_size != 1 else it
      out_args.append(p.view(out, [off]))
    else:
      out_args.append(sbuf)
  call = ProgramNode(
    ProgramOp.CALL, tuple(in_args + out_args), attrs={"callee": ctx.symbols[callee], "n_in": len(in_args), "n_out": len(out_args), "returns": ()}
  )
  prologue = ctx.statements[before:]
  del ctx.statements[before:]
  body = [*prologue, call]
  if (layout := layouts[out_idx]) is not None:
    off = p.mul(it, p.const_int(slice_size))
    ctx.reorder(scratch[out_idx], results[out_idx], layout.inverse_permutation, p.view(out, [off]))
    body.extend(ctx.statements[before:])
    del ctx.statements[before:]
  ctx.statements.extend(statement for statement in body if statement.op == ProgramOp.BUFFER)
  ctx.statements.append(p.for_(rng, [statement for statement in body if statement.op != ProgramOp.BUFFER]))
