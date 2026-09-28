"""Schedule bounded scalar expression trees into Program assignments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ...ir import program as p
from ...ir.program import ProgramNode, ProgramOp
from ..arith import CONSTANTS

MAX_SCALAR_DEPTH = 32


@dataclass(slots=True)
class ScalarNameAllocator:
  """Allocate scalar locals against reserved C identifier spellings."""

  occupied: set[str]
  serial: int = 0

  def fresh(self) -> str:
    while (name := f"v{self.serial}") in self.occupied:
      self.serial += 1
    self.serial += 1
    self.occupied.add(name)
    return name


def schedule_values(
  roots: Sequence[ProgramNode],
  names: ScalarNameAllocator,
) -> tuple[list[ProgramNode], tuple[ProgramNode, ...]]:
  """Return declarations and rewritten roots with shared or deep scalar nodes named."""
  order: list[ProgramNode] = []
  seen: set[ProgramNode] = set()
  pending = [(root, False) for root in reversed(roots)]
  while pending:
    node, ready = pending.pop()
    if node in seen:
      continue
    if not ready:
      pending.append((node, True))
      pending.extend((arg, False) for arg in reversed(node.args))
      continue
    seen.add(node)
    order.append(node)

  uses = Counter(arg for node in order for arg in node.args)
  uses.update(roots)
  values: dict[ProgramNode, ProgramNode] = {}
  depth: dict[ProgramNode, int] = {}
  declarations: list[ProgramNode] = []
  for node in order:
    args = tuple(values[arg] for arg in node.args)
    value = ProgramNode(node.op, args, node.attrs, node.dtype)
    d = 1 + max((depth[arg] for arg in node.args), default=0)
    can_name = node.op in p.SCALAR_OPS - {*CONSTANTS, ProgramOp.VAR}
    if can_name and (uses[node] > 1 or d >= MAX_SCALAR_DEPTH):
      name = names.fresh()
      declarations.append(p.assign(name, value, declare=True))
      value, d = p.var(name, node.dtype), 0
    values[node], depth[node] = value, d
  return declarations, tuple(values[root] for root in roots)
