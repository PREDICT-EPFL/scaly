"""How an expression-dialect pass is defined and applied: patterns, matching, walk-rebuild.

``ir/`` owns the machinery; the concrete rewrites built on it live in ``passes/expr.py``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

from .expr import Expr, ExprOp, topo


@dataclass(frozen=True, slots=True)
class Pattern:
  op: ExprOp | None
  predicate: Callable[[Expr], bool]
  replacement: Callable[[Expr], Expr]

  def matches(self, expr: Expr) -> bool:
    return (self.op is None or expr.op == self.op) and self.predicate(expr)


class PatternMatcher:
  """An op-indexed set of rewrite ``Pattern``s.

  Indexing by op means a node is only tested against patterns that could match it, which is what
  keeps rewriting cheap as the pattern set grows.
  """

  def __init__(self, patterns: Iterable[Pattern]):
    self.any: list[Pattern] = []
    self.by_op: dict[ExprOp, list[Pattern]] = defaultdict(list)
    for pattern in patterns:
      if pattern.op is None:
        self.any.append(pattern)
      else:
        self.by_op[pattern.op].append(pattern)

  def candidates(self, op: ExprOp) -> Iterable[Pattern]:
    yield from self.any
    yield from self.by_op.get(op, ())

  def rewrite(self, expr: Expr) -> Expr | None:
    for pattern in self.candidates(ExprOp(expr.op)):
      if pattern.matches(expr):
        ret = pattern.replacement(expr)
        if ret is not expr:
          return ret
    return None


def rewrite(expr: Expr, patterns: Iterable[Pattern] | PatternMatcher) -> Expr:
  """Apply ``patterns`` across a graph, bottom up, and return the rewritten expression.

  Each node is visited once in topological order with its already-rewritten children cached, so
  shared subgraphs stay shared instead of being expanded into a tree.
  """
  matcher = patterns if isinstance(patterns, PatternMatcher) else PatternMatcher(patterns)
  replacements: dict[int, Expr] = {}
  for node in topo((expr,)):
    cur = _replace_args(node, replacements)
    # tinygrad-style graph rewrite invariant: each original node is processed once
    # in topological order, with already-rewritten children cached in replacements.
    while (new := matcher.rewrite(cur)) is not None:
      cur = new
    replacements[node.id] = cur
  return replacements[expr.id]


def _replace_args(expr: Expr, replacements: dict[int, Expr]) -> Expr:
  args = tuple(replacements.get(arg.id, arg) for arg in expr.args)
  return (
    expr
    if all(a is b for a, b in zip(args, expr.args, strict=True))
    else Expr(expr.op, args, expr.type, expr.name, expr.value, dict(expr.attrs), expr.lowering)
  )
