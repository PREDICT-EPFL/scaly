"""Decide which expression nodes their consumers read in place instead of from a stored buffer."""

from __future__ import annotations

from functools import reduce
from operator import or_

import numpy as np

from ...ad.rules import ELEMENTWISE
from ...function.concrete import ConcreteFunction
from ...ir.expr import Expr, ExprOp
from ...ir.types import stored_indices

ALIASES = frozenset({ExprOp.RESHAPE, ExprOp.STOP_GRADIENT, ExprOp.PACK, ExprOp.VALUES})
MOVEMENT = ALIASES | {ExprOp.TRANSPOSE, ExprOp.SLICE, ExprOp.GATHER, ExprOp.CONST}
_ONCE, _MANY, _BUFFER = 0, 1, 2


def scatter_targets(node: Expr) -> np.ndarray:
  """The flat position each stored entry of a scatter's source lands at."""
  indices = node.attrs["indices"].reshape(-1)
  src = node.args[0]
  return indices if src.type.pattern is None else indices[stored_indices(src.type)]


def _reads(node: Expr) -> list[tuple[Expr, int]]:
  """How ``node`` reads each argument: each element once, some elements several times, or only whole
  from a buffer, as calls, prints and sparse plans do."""
  op = ExprOp(node.op)
  if node.type.pattern is not None or any(a.type.pattern is not None for a in node.args):
    return [(a, _BUFFER) for a in node.args]
  if op in ELEMENTWISE:
    return [(a, _ONCE if a.size == node.size else _MANY) for a in node.args]
  if op == ExprOp.MATMUL:
    trips = node.size * node.args[0].shape[-1]
    return [(a, _ONCE if trips == a.size else _MANY) for a in node.args]
  if op == ExprOp.GATHER:
    indices = node.attrs["indices"].reshape(-1)
    return [(node.args[0], _ONCE if len(np.unique(indices)) == len(indices) else _MANY)]
  if op in ALIASES | {ExprOp.TRANSPOSE, ExprOp.SLICE, ExprOp.REDUCE, ExprOp.SCATTER, ExprOp.STACK, ExprOp.CONCAT}:
    return [(a, _ONCE) for a in node.args]
  return [(a, _BUFFER) for a in node.args]


def plan_fusion(fun: ConcreteFunction, invariant: frozenset[frozenset[int]], *, scalar: bool) -> tuple[frozenset[int], dict[int, tuple[Expr, ...]]]:
  """The nodes lowering evaluates at their consumers' coordinates rather than storing, and the sums
  of scatters, by the id of the sum each one writes.

  A movement op or a constant is an index remap, deferred unless it is an output or a consumer needs
  it whole. A dense elementwise node is deferred when one consumer reads each of its elements once,
  through any chain of deferred movement ops; one that holds a libm call is deferred only into a
  single read, so the call is not duplicated. Everything else is stored: reductions, products,
  scatters, concatenations, calls, outputs and sparse values.

  ``invariant`` holds, for a mapped function, the input positions each of its maps broadcasts. The
  map's trip is a range that a node depending only on those inputs does not depend on, so such a node
  is not deferred into a consumer that depends on the others, and stays where ``hoist_invariant``
  can lift it out of the map. A procedure lowered as scalars keeps a node with known zeros stored,
  since its known zeros stay zero where the arithmetic would give a NaN.

  A left-associated sum whose operands are dense scatters it alone reads, directly or through a
  reshape, is stored once: it zeroes its buffer and each scatter accumulates into it in order. A
  scatter that repeats an index joins only if no earlier operand writes that index, so each element
  still adds its terms in the order of the expression."""
  nodes, outputs = fun.nodes, frozenset(o.id for o in fun.outputs)
  readers: dict[int, list[tuple[Expr, int]]] = {}
  for node in nodes:
    for arg, kind in _reads(node):
      readers.setdefault(arg.id, []).append((node, kind))
  deferred: set[int] = set()
  # Each node's reads as seen through deferred movement: (consumer id, how it reads, read directly).
  terminal: dict[int, list[tuple[int, int, bool]]] = {}
  for node in reversed(nodes):
    edges: list[tuple[int, int, bool]] = []
    for consumer, kind in readers.get(node.id, ()):
      if consumer.id in deferred and ExprOp(consumer.op) in MOVEMENT:
        edges += [(c, max(kind, k), False) for c, k, _ in terminal[consumer.id]]
      else:
        # A stored alias that is not an output shares its source's buffer.
        shares = ExprOp(consumer.op) in ALIASES and consumer.id not in outputs
        edges.append((consumer.id, _BUFFER if shares else kind, True))
    terminal[node.id] = edges
    dense = all(e.type.pattern is None for e in (node, *node.args))
    if node.id not in outputs and dense and ExprOp(node.op) in MOVEMENT and all(k != _BUFFER for _, k, _ in edges):
      deferred.add(node.id)
  position = {x.id: 1 << k for k, x in enumerate(fun.inputs)}
  depends: dict[int, int] = {}
  for node in nodes:
    depends[node.id] = position.get(node.id, 0) | reduce(or_, (depends[a.id] for a in node.args), 0)
  fixed = [reduce(or_, (1 << k for k in positions), 0) for positions in invariant]

  def hoistable(producer: int, consumer: int) -> bool:
    return any(depends[producer] & ~mask == 0 and depends[consumer] & ~mask for mask in fixed)

  def scatter(arg: Expr, parent: Expr) -> Expr | None:
    while arg.id in deferred and ExprOp(arg.op) in ALIASES:
      arg = arg.args[0]
    edges = terminal[arg.id]
    if ExprOp(arg.op) != ExprOp.SCATTER or arg.id in outputs or arg.size != parent.size or len(edges) != 1 or edges[0][:2] != (parent.id, _ONCE):
      return None
    return arg if arg.type.pattern is None else None

  chains: dict[int, tuple[Expr, ...]] = {}
  extended: set[int] = set()
  for node in nodes:
    if ExprOp(node.op) != ExprOp.ADD or node.type.pattern is not None or any(a.size != node.size for a in node.args):
      continue
    left, right = node.args
    if (last := scatter(right, node)) is None:
      continue
    if left.id in chains and left.id not in outputs and terminal[left.id] == [(node.id, _ONCE, True)]:
      leaves = chains[left.id]
    elif (first := scatter(left, node)) is not None:
      leaves = (first,)
    else:
      continue
    written = {int(i) for leaf in leaves for i in scatter_targets(leaf)}
    indices = scatter_targets(last)
    if len(np.unique(indices)) != len(indices) and written.intersection(indices.tolist()):
      continue
    if left.id in chains:
      extended.add(left.id)
    chains[node.id] = (*leaves, last)
  sums = {root: leaves for root, leaves in chains.items() if root not in extended}
  deferred |= extended

  expensive: set[int] = set()
  for node in nodes:
    op = ExprOp(node.op)
    if node.id in sums:
      continue
    if node.id in deferred:
      if any(a.id in expensive for a in node.args):
        expensive.add(node.id)
      continue
    if node.id in outputs or op not in ELEMENTWISE or any(e.type.pattern is not None for e in (node, *node.args)):
      continue
    if scalar and node.support is not None and len(node.support) < node.size:
      continue
    edges = terminal[node.id]
    costly = ELEMENTWISE[op].expensive or any(a.id in expensive for a in node.args)
    if (
      len({c for c, _, _ in edges}) == 1
      and all(k == _ONCE for _, k, _ in edges)
      and (len(edges) == 1 or not costly and all(d for *_, d in edges))
      and not hoistable(node.id, edges[0][0])
    ):
      deferred.add(node.id)
      if costly:
        expensive.add(node.id)
  return frozenset(deferred), sums
