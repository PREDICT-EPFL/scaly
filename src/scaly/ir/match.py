"""How a rewrite is defined and applied: patterns, matching, and one iterative walk-rebuild driver.

``ir/`` owns the machinery; it is generic over any node with ``.op`` and ``.args`` so both the
expression dialect (``Expr``) and the program dialect (``ProgramNode``) drive their passes through
it. The concrete rewrites live in ``passes/``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Iterable, Protocol, cast

from .expr import Expr, ExprOp
from .types import Lowering


class _HasOpArgs(Protocol):
  @property
  def op(self) -> Any: ...
  @property
  def args(self) -> tuple[Any, ...]: ...


@dataclass(frozen=True, slots=True)
class Pattern[Node: _HasOpArgs]:
  op: Hashable | None
  predicate: Callable[[Node], bool]
  replacement: Callable[[Node], Node]

  def matches(self, node: Node) -> bool:
    return (self.op is None or node.op == self.op) and self.predicate(node)


class PatternMatcher[Node: _HasOpArgs]:
  """An op-indexed set of rewrite ``Pattern``s.

  Indexing by op means a node is only tested against patterns that could match it, which is what
  keeps rewriting cheap as the pattern set grows.
  """

  def __init__(self, patterns: Iterable[Pattern[Node]]):
    self.any: list[Pattern[Node]] = []
    self.by_op: dict[Hashable, list[Pattern[Node]]] = defaultdict(list)
    for pattern in patterns:
      if pattern.op is None:
        self.any.append(pattern)
      else:
        self.by_op[pattern.op].append(pattern)

  def candidates(self, op: Hashable) -> Iterable[Pattern[Node]]:
    yield from self.any
    yield from self.by_op.get(op, ())

  def rewrite(self, node: Node) -> Node | None:
    for pattern in self.candidates(node.op):
      if pattern.matches(node):
        ret = pattern.replacement(node)
        if ret is not node:
          return ret
    return None


def _combined_lowering(own: Lowering, children: Iterable[Lowering]) -> Lowering:
  hints = {own, *children} - {"auto"}
  if "block" in hints or "opaque" in hints and len(hints) > 1:
    return "block"
  if "opaque" in hints:
    return "opaque"
  return "scalar" if "scalar" in hints else "auto"


def _apply_lowering(expr: Expr, lowering: Lowering, *, preserve_identity: bool = False) -> Expr:
  if lowering == "auto" or expr.lowering == lowering:
    return expr
  if expr.op == ExprOp.INPUT or preserve_identity and expr.op != ExprOp.CONST:
    return Expr(
      ExprOp.RESHAPE,
      (expr,),
      expr.type,
      attrs={"shape": expr.shape, "lowering_identity": True},
      lowering=lowering,
    )
  return expr.with_lowering(lowering)


def rebuild_expr(expr: Expr, args: tuple[Expr, ...]) -> Expr:
  """The expression-dialect adapter: ``expr`` with new ``args`` and everything else kept."""
  return Expr(expr.op, args, expr.type, expr.name, expr.value, dict(expr.attrs), expr.lowering)


def rewrite[Node: _HasOpArgs](
  root: Node,
  patterns: Iterable[Pattern[Node]] | PatternMatcher[Node],
  *,
  rebuild: Callable[[Node, tuple[Node, ...]], Node] | None = None,
  fixpoint: bool = True,
  revisit: bool = False,
  max_steps: int = 1_000_000,
  memo: dict[Node, Node] | None = None,
) -> Node:
  """Apply ``patterns`` across a graph, bottom up, and return the rewritten root.

  Iterative over an explicit stack, so depth in the graph never becomes depth on the Python
  stack. Each node is visited once with its already-rewritten children memoized by identity, so
  shared subgraphs stay shared. ``fixpoint`` retries the matcher on a node until nothing fires;
  ``revisit`` instead walks into a replacement's subgraph so nested rewrites collapse in one
  pass. ``max_steps`` bounds the total number of replacements. ``rebuild`` defaults to the
  expression adapter ``rebuild_expr``. Program adapters may reuse ``memo`` across roots with
  identical patterns, rebuild, and options. Expression nodes do not support caller memoization
  because their lowering hints depend on traversal provenance.
  """
  if memo is not None and isinstance(root, Expr):
    raise ValueError("caller memoization is only supported for Program nodes")
  matcher = patterns if isinstance(patterns, PatternMatcher) else PatternMatcher(patterns)
  rebuild = rebuild or cast(Callable[[Node, tuple[Node, ...]], Node], rebuild_expr)
  done: dict[int, Node] = {}
  lowerings: dict[int, Lowering] = {}
  forward: dict[int, Node] = {}
  alive: list[Node] = []
  steps = 0
  stack: list[tuple[Node, bool]] = [(root, False)]
  while stack:
    node, ready = stack.pop()
    if id(node) in done:
      continue
    if memo is not None and node in memo:
      done[id(node)] = memo[node]
      continue
    if not ready:
      stack.append((node, True))
      stack.extend((a, False) for a in reversed(node.args) if id(a) not in done)
      continue
    if (target := forward.pop(id(node), None)) is not None:
      if id(target) not in done:
        raise RuntimeError(f"rewrite replacement at {node.op} contains the node it replaces")
      done[id(node)] = done[id(target)]
      if isinstance(node, Expr):
        lowerings[id(node)] = lowerings[id(target)]
      if memo is not None:
        memo[node] = done[id(node)]
      continue
    args = tuple(done[id(a)] for a in node.args)
    cur = node if all(a is b for a, b in zip(args, node.args, strict=True)) else rebuild(node, args)
    lowering = _combined_lowering(node.lowering, (lowerings[id(arg)] for arg in node.args)) if isinstance(node, Expr) else "auto"
    new = matcher.rewrite(cur)
    while new is not None:
      steps += 1
      if steps > max_steps:
        names = sorted({getattr(p.replacement, "__qualname__", repr(p.replacement)) for p in matcher.candidates(cur.op)})
        raise RuntimeError(f"rewrite exceeded {max_steps} steps at {cur.op} with patterns {names}")
      if isinstance(new, Expr):
        new = cast(Node, _apply_lowering(new, lowering, preserve_identity=id(new) in lowerings))
      alive.append(new)
      if revisit:
        forward[id(node)] = new
        stack.extend(((node, True), (new, False)))
        break
      cur = new
      new = matcher.rewrite(cur) if fixpoint else None
    else:
      done[id(node)] = cur
      if isinstance(node, Expr):
        lowerings[id(node)] = lowering
        lowerings[id(cur)] = lowering
      if memo is not None:
        memo[node] = cur
  return done[id(root)]


def _replace_args(expr: Expr, replacements: dict[int, Expr]) -> Expr:
  args = tuple(replacements.get(arg.id, arg) for arg in expr.args)
  return expr if all(a is b for a, b in zip(args, expr.args, strict=True)) else rebuild_expr(expr, args)
