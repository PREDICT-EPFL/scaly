"""The verifier machinery both dialects share: one ``Rule``, one ``Spec`` table, one ``VerifyError``.

The dialect-specific rule tables and entry points are ``ir/expr_spec.py`` and ``ir/program_spec.py``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

Node = TypeVar("Node")
Op = TypeVar("Op", bound=Hashable)


class VerifyError(Exception):
  """Raised by ``verify_expr`` and ``verify_program`` at the first node failing a rule.

  The message names the node, its op and the rule it broke. Both dialects raise this one type.
  """


@dataclass(frozen=True, slots=True)
class Rule(Generic[Node, Op]):
  op: Op | None
  description: str
  check: Callable[[Node], str | None]

  def applies(self, node: Node) -> bool:
    return self.op is None or getattr(node, "op") == self.op


class Spec(Generic[Node, Op]):
  """A verifier: a set of ``Rule``s, indexed by op so checking a node touches only its own.

  Rules with ``op=None`` apply to every node. Both dialects build their specs from this class.
  """

  def __init__(self, rules: Iterable[Rule[Node, Op]]) -> None:
    self.any: list[Rule[Node, Op]] = []
    self.by_op: dict[Op, list[Rule[Node, Op]]] = defaultdict(list)
    for rule in rules:
      if rule.op is None:
        self.any.append(rule)
      else:
        self.by_op[rule.op].append(rule)

  def candidates(self, op: Op) -> Iterable[Rule[Node, Op]]:
    yield from self.any
    yield from self.by_op.get(op, ())

  def check(self, node: Node) -> tuple[Rule[Node, Op], str] | None:
    for rule in self.candidates(getattr(node, "op")):
      diag = rule.check(node)
      if diag is not None:
        return rule, diag
    return None

  def merge(self, *others: Spec[Node, Op]) -> Spec[Node, Op]:
    rules = list(self.any)
    for op_rules in self.by_op.values():
      rules.extend(op_rules)
    for other in others:
      rules.extend(other.any)
      for op_rules in other.by_op.values():
        rules.extend(op_rules)
    return Spec(rules)


__all__ = ["Rule", "Spec", "VerifyError"]
