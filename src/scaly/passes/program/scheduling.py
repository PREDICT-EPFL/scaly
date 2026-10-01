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


# What may be computed where C would have skipped it: one instruction with every compiler and on
# every target, and nothing that can fault. A libm call is as slow as the branch (``fmin``, ``floor``
# and ``sqrt`` are calls under some compilers and flags), an integer division traps on a zero
# divisor, and a float outside an integer's range has no defined conversion.
_AHEAD_OPS = frozenset(
  {
    ProgramOp.CONST_INT,
    ProgramOp.CONST_FLOAT,
    ProgramOp.VAR,
    ProgramOp.VIEW,
    ProgramOp.LOAD,
    ProgramOp.ADD,
    ProgramOp.SUB,
    ProgramOp.MUL,
    ProgramOp.DIV,
    ProgramOp.NEG,
    ProgramOp.ABS,
    ProgramOp.LT,
    ProgramOp.LE,
    ProgramOp.EQ,
    ProgramOp.NE,
    ProgramOp.AND,
    ProgramOp.OR,
    ProgramOp.NOT,
    ProgramOp.SELECT,
    ProgramOp.CAST,
  }
)


def _speculable(node: ProgramNode) -> bool:
  """Whether ``node`` itself may be computed where C would have skipped it (``_AHEAD_OPS``)."""
  if node.op not in _AHEAD_OPS:
    return False
  if node.op == ProgramOp.DIV and not node.dtype.is_floating:
    return False
  return not (node.op == ProgramOp.CAST and not node.dtype.is_floating and node.args[0].dtype.is_floating)


def schedule_values(
  roots: Sequence[ProgramNode],
  names: ScalarNameAllocator,
  *,
  ahead: bool = False,
) -> tuple[list[ProgramNode], tuple[ProgramNode, ...]]:
  """Return declarations and rewritten roots with shared or deep scalar nodes named.

  With ``ahead`` (a statement in a loop), a value C would compute only under a condition is named
  too, so that it is computed first. C evaluates one branch of ``c ? a : b`` and skips the right
  operand of ``&&`` and ``||``, and a load it would skip is one the C compiler may not move ahead
  of the condition: it keeps a branch per element and the loop stays scalar, where a value computed
  first is a select it vectorizes. A graph evaluates every operand anyway.

  It is a trade, since the branch's work is then done for every element, where a branch the
  processor predicts would have skipped it. So a select's float branches that load are computed
  ahead only where the trade was measured to win: when one of the two branches is no work at all
  (a constant, a value already named, a load, a select between such), as in ``mask ? 1 / x : 0``.
  A piecewise function, with work in every piece, keeps its branches (1.1-1.4x slower with every
  piece computed, on data the processor predicts); so does a select on a condition that no
  variable enters, a flag the C compiler can test once outside the loop; so does what could fault
  or cost a call (``_AHEAD_OPS``); and so does straight-line code, where nothing is vectorized."""
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
  loads: dict[ProgramNode, bool] = {}  # whether the rewritten node still holds a load
  free: dict[ProgramNode, bool] = {}  # whether it may be computed ahead of a condition
  work: dict[ProgramNode, int] = {}  # the operations left in the rewritten node, loads and selects aside
  varies: dict[ProgramNode, bool] = {}  # whether a variable enters it: the loop's, or a value named in it
  declarations: list[ProgramNode] = []
  nameable = p.SCALAR_OPS - {*CONSTANTS, ProgramOp.VAR}

  def name(node: ProgramNode) -> None:
    target = names.fresh()
    declarations.append(p.assign(target, values[node], declare=True))
    values[node], depth[node], loads[node], free[node], work[node] = p.var(target, node.dtype), 0, False, True, 0

  def first(node: ProgramNode) -> bool:
    return node.op in nameable and loads[node] and free[node]

  for node in order:
    if ahead and node.op == ProgramOp.SELECT and varies[node.args[0]] and min(work[arg] for arg in node.args[1:]) == 0:
      for branch in node.args[1:]:
        if first(branch) and branch.dtype.is_floating:
          name(branch)
    elif ahead and node.op in (ProgramOp.AND, ProgramOp.OR) and first(node.args[1]):
      name(node.args[1])
    args = tuple(values[arg] for arg in node.args)
    values[node] = ProgramNode(node.op, args, node.attrs, node.dtype)
    depth[node] = 1 + max((depth[arg] for arg in node.args), default=0)
    loads[node] = node.op == ProgramOp.LOAD or any(loads[arg] for arg in node.args)
    free[node] = _speculable(node) and all(free[arg] for arg in node.args)
    varies[node] = node.op == ProgramOp.VAR or any(varies[arg] for arg in node.args)
    if node.op in (ProgramOp.LOAD, ProgramOp.VIEW, ProgramOp.VAR, *CONSTANTS):
      work[node] = 0
    else:
      work[node] = sum(work[arg] for arg in node.args[1:]) if node.op == ProgramOp.SELECT else 1 + sum(work[arg] for arg in node.args)
    if node.op in nameable and (uses[node] > 1 or depth[node] >= MAX_SCALAR_DEPTH):
      name(node)
  return declarations, tuple(values[root] for root in roots)


__all__ = ["MAX_SCALAR_DEPTH", "ScalarNameAllocator", "schedule_values"]
