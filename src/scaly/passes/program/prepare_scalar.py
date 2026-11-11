"""Prepare statement-local scalar expressions for syntax-only renderers."""

from __future__ import annotations

from ...ir import program as p
from ...ir.program import ProgramNode, ProgramOp, walk_program
from ...utils.names import c_ident
from .scheduling import ScalarNameAllocator, schedule_values


def prepare_scalar_expressions(prog: ProgramNode) -> ProgramNode:
  """Insert local assignments that bound scalar depth without moving loads across statements."""

  def rewrite_body(body: tuple[ProgramNode, ...], names: ScalarNameAllocator) -> tuple[ProgramNode, ...]:
    out: list[ProgramNode] = []
    for stmt in body:
      if stmt.op == ProgramOp.BUFFER:
        out.append(stmt)
        continue
      if stmt.op == ProgramOp.FOR:
        out.append(ProgramNode(stmt.op, (stmt.args[0], *rewrite_body(stmt.args[1:], names)), stmt.attrs, dtype=stmt.dtype))
        continue
      if stmt.op == ProgramOp.IF:
        declarations, (condition,) = schedule_values((stmt.args[0],), names)
        branches = tuple(p.block(*rewrite_body(branch.args, names)) for branch in stmt.args[1:])
        out.extend((*declarations, ProgramNode(stmt.op, (condition, *branches), stmt.attrs, dtype=stmt.dtype)))
        continue
      if stmt.op == ProgramOp.STORE:
        declarations, roots = schedule_values((*stmt.args[0].args, stmt.args[1]), names)
        index = roots[:-1]
        target = ProgramNode(ProgramOp.VIEW, index, stmt.args[0].attrs, dtype=stmt.args[0].dtype)
        out.extend((*declarations, p.store(target, roots[-1])))
        continue
      if stmt.op == ProgramOp.STORE_PAIR:
        declarations, roots = schedule_values((*stmt.args[0].args, *stmt.args[1:]), names)
        index_count = len(stmt.args[0].args)
        target = ProgramNode(ProgramOp.VIEW, roots[:index_count], stmt.args[0].attrs, dtype=stmt.args[0].dtype)
        out.extend((*declarations, p.store_pair(target, roots[-2], roots[-1])))
        continue
      if stmt.op == ProgramOp.ASSIGN:
        declarations, roots = schedule_values((stmt.args[0],), names)
        target = stmt.attrs["target"]
        out.extend((*declarations, p.assign(target, roots[0], stmt.dtype, declare=stmt.attrs.get("declare", False))))
        continue
      if stmt.op == ProgramOp.CALL:
        roots = tuple(arg for node in stmt.args if node.op == ProgramOp.VIEW for arg in node.args)
        declarations, prepared = schedule_values(roots, names)
        values = iter(prepared)
        args = tuple(
          ProgramNode(arg.op, tuple(next(values) for _ in arg.args), arg.attrs, dtype=arg.dtype) if arg.op == ProgramOp.VIEW else arg
          for arg in stmt.args
        )
        out.extend((*declarations, ProgramNode(stmt.op, args, stmt.attrs, dtype=stmt.dtype)))
        continue
      out.append(stmt)
    return tuple(out)

  args: list[ProgramNode] = []
  for node in prog.args:
    if node.op == ProgramOp.PROC:
      pc = node.attrs["param_count"]
      reserved = {arg.attrs["name"] for arg in node.args[:pc]}
      for current in walk_program(ProgramNode(ProgramOp.BLOCK, node.args[pc:], dtype=node.dtype)):
        if current.op == ProgramOp.BUFFER:
          reserved.add(current.attrs["name"])
        elif current.op == ProgramOp.ASSIGN:
          reserved.add(current.attrs["target"])
        elif current.op == ProgramOp.FOR:
          reserved.add(current.args[0].attrs["name"])
        elif current.op == ProgramOp.VAR:
          reserved.add(current.attrs["name"])
      names = ScalarNameAllocator({c_ident(name) for name in reserved})
      node = ProgramNode(node.op, (*node.args[:pc], *rewrite_body(node.args[pc:], names)), node.attrs, dtype=node.dtype)
    args.append(node)
  return ProgramNode(prog.op, tuple(args), prog.attrs, dtype=prog.dtype)
