"""Combine safe adjacent float64 stores into explicit two-lane stores."""

from __future__ import annotations

from ...ir import program as p
from ...ir.program import ProgramNode, ProgramOp
from ._common import _alias_sources, _resolve_alias, buffer_refs


def _constant_store(stmt: ProgramNode) -> tuple[str, int] | None:
  if stmt.op != ProgramOp.STORE or stmt.dtype.c_type != "double":
    return None
  view = stmt.args[0]
  if len(view.args) != 1 or view.args[0].op != ProgramOp.CONST_INT:
    return None
  return view.attrs["buffer"], int(view.args[0].attrs["value"])


def coalesce_stores(prog: ProgramNode) -> ProgramNode:
  """Represent safe adjacent stores to consecutive float64 elements as STORE_PAIR."""

  def rewrite_body(body: tuple[ProgramNode, ...], aliases: dict[str, str]) -> tuple[ProgramNode, ...]:
    aliases = {**aliases, **_alias_sources(body)}
    out: list[ProgramNode] = []
    i = 0
    while i < len(body):
      stmt = body[i]
      if stmt.op == ProgramOp.FOR:
        stmt = ProgramNode(stmt.op, (stmt.args[0], *rewrite_body(stmt.args[1:], aliases)), stmt.attrs, stmt.dtype)
      current = _constant_store(stmt)
      following = _constant_store(body[i + 1]) if i + 1 < len(body) else None
      if current is not None and following == (current[0], current[1] + 1):
        second = body[i + 1]
        root = _resolve_alias(current[0], aliases)
        if root not in buffer_refs(second.args[1], aliases).loads:
          out.append(p.store_pair(stmt.args[0], stmt.args[1], second.args[1]))
          i += 2
          continue
      out.append(stmt)
      i += 1
    return tuple(out)

  args: list[ProgramNode] = []
  for node in prog.args:
    if node.op in {ProgramOp.PROC, ProgramOp.KERNEL}:
      pc = node.attrs["param_count"]
      node = ProgramNode(node.op, (*node.args[:pc], *rewrite_body(node.args[pc:], {})), node.attrs, node.dtype)
    args.append(node)
  return ProgramNode(prog.op, tuple(args), prog.attrs, prog.dtype)


__all__ = ["coalesce_stores"]
