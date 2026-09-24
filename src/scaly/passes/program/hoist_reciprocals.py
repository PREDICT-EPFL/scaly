"""Hoist explicitly permitted reciprocal evaluations out of nonempty loops."""

from __future__ import annotations

from collections import Counter

from ...ir import program as p
from ...ir.match import Pattern, rewrite
from ...ir.program import walk_program, ProgramNode, ProgramOp
from ...utils.names import c_ident
from ._common import _map_procs, _proc_parts, _rebuild_proc, allocated_name, buffer_refs, rebuild_program, trip_count


def hoist_reciprocals(prog: ProgramNode) -> ProgramNode:
  """Replace invariant divisors with reciprocals, allowing rounding, overflow, and underflow changes."""
  return _map_procs(prog, _hoist_proc)


def _hoist_proc(proc: ProgramNode) -> ProgramNode:
  params, body = _proc_parts(proc)
  nodes = list(walk_program(proc))
  spellings = {c_ident(n.attrs[key]) for n in nodes for key in ("name", "target") if key in n.attrs}
  aliases = {n.attrs["name"]: n.attrs["alias_of"] for n in nodes if n.op == ProgramOp.BUFFER and "alias_of" in n.attrs}

  def transform(stmt: ProgramNode) -> list[ProgramNode]:
    if stmt.op == ProgramOp.BLOCK:
      return [p.block(*(n for child in stmt.args for n in transform(child)))]
    if stmt.op != ProgramOp.FOR:
      return [stmt]
    rng, *children = stmt.args
    children = [n for child in children for n in transform(child)]
    loop = ProgramNode(stmt.op, (rng, *children), {**stmt.attrs, "body_len": len(children)}, stmt.dtype)
    if (trip_count(rng) or 0) < 1:
      return [loop]
    assignments: Counter[str] = Counter()
    pending = list(children)
    while pending:
      current = pending.pop()
      if current.op == ProgramOp.ASSIGN:
        assignments[current.attrs["target"]] += 1
      elif current.op == ProgramOp.FOR:
        pending.extend(current.args[1:])
      elif current.op == ProgramOp.BLOCK:
        pending.extend(current.args)
    assigned = set(assignments) | {n.attrs["name"] for n in walk_program(loop) if n.op == ProgramOp.RANGE}
    definitions: dict[str, ProgramNode] = {}
    refs = buffer_refs(loop, aliases)
    mutated = refs.writes | refs.call_inputs
    local_buffers = {n.attrs["name"] for n in walk_program(loop) if n.op == ProgramOp.BUFFER}
    reciprocals: dict[ProgramNode, ProgramNode] = {}
    prologue: list[ProgramNode] = []

    def invariant(value: ProgramNode) -> ProgramNode | None:
      if any(n.op == ProgramOp.VAR and n.attrs["name"] in assigned - definitions.keys() for n in walk_program(value)):
        return None
      expanded = rewrite(
        value,
        [Pattern(ProgramOp.VAR, lambda n: n.attrs["name"] in definitions, lambda n: definitions[n.attrs["name"]])],
        rebuild=rebuild_program,
        fixpoint=False,
      )
      if buffer_refs(expanded, aliases).reads & mutated or buffer_refs(expanded).reads & local_buffers:
        return None
      return expanded

    divisors: dict[ProgramNode, ProgramNode] = {}

    def eligible(node: ProgramNode) -> bool:
      if not node.dtype.is_floating or (value := invariant(node.args[1])) is None:
        return False
      divisors[node] = value
      return True

    def replace(node: ProgramNode) -> ProgramNode:
      divisor = divisors[node]
      if divisor not in reciprocals:
        name = allocated_name("inv", spellings)
        reciprocals[divisor] = p.var(name, divisor.dtype)
        prologue.append(p.assign(name, p.div(p.const_float(1, divisor.dtype), divisor), declare=True))
      return p.mul(node.args[0], reciprocals[divisor])

    pattern = Pattern(ProgramOp.DIV, eligible, replace)
    rewritten = []
    for child in children:
      value = child if child.op in (ProgramOp.FOR, ProgramOp.BLOCK) else rewrite(child, [pattern], rebuild=rebuild_program, fixpoint=False)
      rewritten.append(value)
      if value.op == ProgramOp.ASSIGN and value.attrs.get("declare") and assignments[value.attrs["target"]] == 1:
        if (definition := invariant(value.args[0])) is not None:
          definitions[value.attrs["target"]] = definition

    return [*prologue, ProgramNode(stmt.op, (rng, *rewritten), {**stmt.attrs, "body_len": len(rewritten)}, stmt.dtype)]

  return _rebuild_proc(proc, params, [n for stmt in body for n in transform(stmt)])
