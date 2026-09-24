"""Shrink read-only periodic constant tables without changing their observable storage uses."""

from __future__ import annotations

import math

import numpy as np

from ...ir import program as p
from ...ir.match import Pattern, rewrite
from ...ir.program import walk_program, ProgramNode, ProgramOp
from ..arith import constant
from ._common import _map_procs, _proc_parts, _rebuild_proc, prune_dead_buffers, rebuild_program


def fold_tiles(prog: ProgramNode) -> ProgramNode:
  """Replace complete repeated constant tiles by modulo-indexed copies of one tile."""
  return _map_procs(prog, _fold_proc)


def _period(decl: ProgramNode) -> int:
  values = np.asarray(decl.attrs["values"], dtype=decl.dtype.numpy()).view(np.uint8).reshape(-1, decl.dtype.itemsize)
  for period in range(1, len(values) // 2 + 1):
    if len(values) % period == 0 and np.array_equal(values[period:], values[:-period]):
      return period
  return len(values)


def _scalar_safe(value: int | float) -> bool:
  return not isinstance(value, (float, np.floating)) or (math.isfinite(value) and (value != 0 or math.copysign(1, value) > 0))


def _fold_proc(proc: ProgramNode) -> ProgramNode:
  params, body = _proc_parts(proc)
  tables = {
    s.attrs["name"]: s for s in body if s.op == ProgramOp.BUFFER and "values" in s.attrs and "alias_of" not in s.attrs and len(s.attrs["shape"]) == 1
  }
  blocked: set[str] = set()
  for node in walk_program(proc):
    if node.op == ProgramOp.BUFFER and "alias_of" in node.attrs:
      blocked.add(node.attrs["alias_of"])
    for arg in node.args:
      if arg.op == ProgramOp.VIEW and (node.op != ProgramOp.LOAD or len(arg.args) != 1):
        blocked.add(arg.attrs["buffer"])
      elif arg.op == ProgramOp.BUFFER and node.op not in (ProgramOp.PROC, ProgramOp.BLOCK):
        blocked.add(arg.attrs["name"])
  periods = {name: period for name, decl in tables.items() if name not in blocked and 0 < (period := _period(decl)) < len(decl.attrs["values"])}
  if not periods:
    return proc

  def load(node: ProgramNode) -> ProgramNode:
    view = node.args[0]
    name = view.attrs["buffer"]
    decl, period = tables[name], periods[name]
    if period == 1 and _scalar_safe(decl.attrs["values"][0]):
      return constant(decl.attrs["values"][0], node.dtype)
    index = p.mod(view.args[0], p.const_int(period))
    return p.load(ProgramNode(ProgramOp.VIEW, (index,), view.attrs, view.dtype))

  pattern = Pattern(ProgramOp.LOAD, lambda n: n.args[0].attrs["buffer"] in periods, load)
  out = []
  for stmt in body:
    if stmt.op == ProgramOp.BUFFER and (period := periods.get(stmt.attrs["name"])) is not None:
      out.append(ProgramNode(stmt.op, stmt.args, {**stmt.attrs, "shape": (period,), "values": stmt.attrs["values"][:period]}, stmt.dtype))
    else:
      out.append(rewrite(stmt, [pattern], rebuild=rebuild_program, fixpoint=False))
  return prune_dead_buffers(_rebuild_proc(proc, params, out))
