"""Schedule bounded scalar expression trees into Program assignments."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ...ir import program as p
from ...ir.expr import registry_version
from ...ir.program import ProgramNode, ProgramOp
from ..arith import CONSTANTS
from ._common import expensive_ops

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


def _speculable(node: ProgramNode, expensive: frozenset[ProgramOp]) -> bool:
  """Whether ``node`` may be computed where C would have skipped it: it cannot fault (an integer
  division by zero, a float out of an integer's range) and is no libm call."""
  if node.op in expensive:
    return False
  if node.op in (ProgramOp.DIV, ProgramOp.MOD) and not node.dtype.is_floating:
    return False
  return not (node.op == ProgramOp.CAST and not node.dtype.is_floating and node.args[0].dtype.is_floating)


def _conditional_operands(node: ProgramNode) -> tuple[ProgramNode, ...]:
  """The operands C evaluates only on one side of a condition: a select's branches, and the right
  operand of ``&&`` and ``||``."""
  if node.op == ProgramOp.SELECT:
    return node.args[1:]
  return node.args[1:] if node.op in (ProgramOp.AND, ProgramOp.OR) else ()


def schedule_values(
  roots: Sequence[ProgramNode],
  names: ScalarNameAllocator,
  *,
  ahead: bool = False,
) -> tuple[list[ProgramNode], tuple[ProgramNode, ...]]:
  """Return declarations and rewritten roots with shared or deep scalar nodes named. With ``ahead``
  (a statement in a loop), the operands C would evaluate conditionally are named too when they
  load: a load under a condition is one the C compiler may not hoist, so it keeps the branch and
  the loop stays scalar, where a value computed first is a select it vectorizes. A graph evaluates
  every operand anyway. Straight-line code keeps its branches, which skip the work."""
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
  expensive = expensive_ops(registry_version())
  conditional = {arg for node in order for arg in _conditional_operands(node)} if ahead else set()
  values: dict[ProgramNode, ProgramNode] = {}
  depth: dict[ProgramNode, int] = {}
  loads: dict[ProgramNode, bool] = {}  # whether the rewritten node still holds a load
  free: dict[ProgramNode, bool] = {}
  declarations: list[ProgramNode] = []
  for node in order:
    args = tuple(values[arg] for arg in node.args)
    value = ProgramNode(node.op, args, node.attrs, node.dtype)
    d = 1 + max((depth[arg] for arg in node.args), default=0)
    loads[node] = node.op == ProgramOp.LOAD or any(loads[arg] for arg in node.args)
    free[node] = _speculable(node, expensive) and all(free[arg] for arg in node.args)
    can_name = node.op in p.SCALAR_OPS - {*CONSTANTS, ProgramOp.VAR}
    if can_name and (uses[node] > 1 or d >= MAX_SCALAR_DEPTH or (node in conditional and loads[node] and free[node])):
      name = names.fresh()
      declarations.append(p.assign(name, value, declare=True))
      value, d, loads[node], free[node] = p.var(name, node.dtype), 0, False, True
    values[node], depth[node] = value, d
  return declarations, tuple(values[root] for root in roots)


__all__ = ["MAX_SCALAR_DEPTH", "ScalarNameAllocator", "schedule_values"]
