"""Static empty-loop removal and unit-loop unrolling for the Program IR."""

from __future__ import annotations

from ...ir.program import ProgramNode, ProgramOp
from ._common import _map_procs, _proc_parts, _rebuild_proc, prune_dead_buffers, substitute_var as _subst_var, trip_count as _trip_count


def unroll_unit_loops(prog: ProgramNode) -> ProgramNode:
  """Remove static zero-trip loops and inline static one-trip loops.

  Lowering intentionally emits uniform loops even for scalar buffers (shape ``(1,)``). Keeping
  that shape through ``fuse_elementwise`` preserves its producer matcher; after fusion, this pass
  erases the leftover ``for (... < 1)`` noise by substituting the loop variable with its sole value.
  """
  return _map_procs(prog, _unroll_unit_loops_proc)


def _unroll_unit_loops_proc(proc: ProgramNode) -> ProgramNode:
  params, body = _proc_parts(proc)
  new_body: list[ProgramNode] = []
  changed = False
  for stmt in body:
    repl = _unroll_unit_loop_stmt(stmt)
    changed = changed or len(repl) != 1 or repl[0] is not stmt
    new_body.extend(repl)
  return prune_dead_buffers(_rebuild_proc(proc, params, new_body)) if changed else proc


def _unroll_unit_loop_stmt(stmt: ProgramNode) -> list[ProgramNode]:
  if stmt.op != ProgramOp.FOR:
    return [stmt]

  rng, *body = stmt.args
  new_body: list[ProgramNode] = []
  changed = False
  for sub in body:
    repl = _unroll_unit_loop_stmt(sub)
    changed = changed or len(repl) != 1 or repl[0] is not sub
    new_body.extend(repl)

  trip_count = _trip_count(rng)
  if trip_count == 0:
    return []
  if trip_count == 1 and not rng.attrs.get("mapped"):
    vname = rng.attrs["name"]
    only_value = rng.args[0]
    out: list[ProgramNode] = []
    for sub in new_body:
      out.extend(_unroll_unit_loop_stmt(_subst_var(sub, vname, only_value)))
    return out
  if changed:
    return [ProgramNode(ProgramOp.FOR, (rng, *new_body), {**stmt.attrs, "body_len": len(new_body)}, stmt.dtype)] if new_body else []
  return [stmt]


# ---------------------------------------------------------------------------
# Pass 3: workspace lifetime packing + spilling.
# ---------------------------------------------------------------------------


__all__ = ["unroll_unit_loops"]
