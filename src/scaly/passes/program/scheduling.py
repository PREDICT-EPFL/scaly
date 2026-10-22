"""Schedule bounded scalar expression trees into Program assignments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from ...ir import program as p
from ...ir.program import ProgramNode, ProgramOp
from ..arith import CONSTANTS
from ...utils.names import NameScope

MAX_SCALAR_DEPTH = 32


class ScalarNameAllocator:
  """Allocate scalar locals through the procedure's name authority."""

  def __init__(self, occupied: set[str]) -> None:
    self.scope = NameScope(occupied)
    self.serial = 0

  def fresh(self) -> str:
    while self.scope.contains(name := f"v{self.serial}"):
      self.serial += 1
    self.serial += 1
    return self.scope.allocate(name)


def schedule_values(
  roots: Sequence[ProgramNode],
  names: ScalarNameAllocator,
) -> tuple[list[ProgramNode], tuple[ProgramNode, ...]]:
  """Name shared and deep work, leaving cheap two-use products at their uses."""
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
    local_product = (
      node.op == ProgramOp.MUL and node.dtype.is_floating and uses[node] == 2 and all(arg.op in {*CONSTANTS, ProgramOp.VAR} for arg in args)
    )
    if can_name and ((uses[node] > 1 and not local_product) or d >= MAX_SCALAR_DEPTH):
      name = names.fresh()
      declarations.append(p.assign(name, value, declare=True))
      value, d = p.var(name, node.dtype), 0
    values[node], depth[node] = value, d
  return declarations, tuple(values[root] for root in roots)
