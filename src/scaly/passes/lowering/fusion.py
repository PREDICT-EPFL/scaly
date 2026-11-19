"""Decide which expression nodes their consumers read in place instead of from a stored buffer."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import reduce
from operator import or_

import numpy as np

from ...ad.rules import ELEMENTWISE
from ...function.concrete import ConcreteFunction
from ...ir.expr import Expr, ExprOp, _index_values, topo
from ...ir import program as p
from ...ir.program import ProgramNode
from ...ir.support import Support, support_indices
from ...ir.types import stored_indices
from .plans import compact_reads

ALIASES = frozenset({ExprOp.RESHAPE, ExprOp.STOP_GRADIENT, ExprOp.PACK, ExprOp.VALUES})
MOVEMENT = ALIASES | {ExprOp.TRANSPOSE, ExprOp.SLICE, ExprOp.GATHER, ExprOp.CONST}
_ONCE, _MANY, _BUFFER = 0, 1, 2


@dataclass(frozen=True)
class SeedEpilogue:
  chain: tuple[Expr, ...]
  seeds: np.ndarray
  packed_consumer: int | None = None


def _contiguous_slice_offset(index: tuple[object, ...], in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> int | None:
  """If the slice selects a contiguous sub-block of ``src`` at a constant flat offset (no stride,
  at most one partial leading slice with full trailing dims), return that offset; else None. Port
  of ``codegen/c.py::_contiguous_slice_offset`` — the precondition for pointer aliasing."""
  if not out_shape:
    offset = 0
    for dim, item in enumerate(index):
      if not isinstance(item, int):
        return None
      offset += (item if item >= 0 else in_shape[dim] + item) * math.prod(in_shape[dim + 1 :])
    return offset
  first_slice = None
  offset = 0
  out_dim = 0
  for dim, item in enumerate(index):
    stride = math.prod(in_shape[dim + 1 :])
    if isinstance(item, int):
      if first_slice is not None and in_shape[dim] != 1:
        return None
      offset += (item if item >= 0 else in_shape[dim] + item) * stride
      continue
    assert isinstance(item, slice)
    start, stop, step = item.indices(int(in_shape[dim]))
    if step != 1:
      return None
    if first_slice is None:
      first_slice = dim
      offset += start * stride
      out_dim += 1
      continue
    if start != 0 or stop != int(in_shape[dim]) or out_shape[out_dim] != int(in_shape[dim]):
      return None
    out_dim += 1
  return offset


def _free(node: Expr) -> bool:
  """Whether reading ``node`` in place costs nothing over reading its source: an alias, a contiguous
  slice or a constant."""
  op = ExprOp(node.op)
  return (
    op in ALIASES
    or op == ExprOp.CONST
    or op == ExprOp.SLICE
    and _contiguous_slice_offset(node.attrs["index"], node.args[0].shape, node.shape) is not None
  )


def scatter_targets(node: Expr) -> np.ndarray:
  """The flat position each stored entry of a scatter's source lands at."""
  indices = _index_values(node).reshape(-1)
  src = node.args[2]
  return indices if src.type.pattern is None else indices[stored_indices(src.type)]


def product_outputs(roots: Sequence[Expr], outputs: dict[int, str]) -> dict[int, tuple[Expr, np.ndarray, tuple[Expr, ...]]]:
  """Find one-reader layouts that a stored matrix product can write directly into an output."""
  layout_ops = (ExprOp.RESHAPE, ExprOp.VALUES, ExprOp.TRANSPOSE, ExprOp.SCATTER)
  candidates = [root for root in roots if root.id in outputs and root.op in layout_ops]
  if not candidates:
    return {}
  uses: dict[int, int] = {}
  for node in topo(roots):
    for arg in node.args:
      uses[arg.id] = uses.get(arg.id, 0) + 1
  result = {}
  root_ids = {root.id for root in roots}
  for root in candidates:
    node = root
    chain = []
    size = root.type.storage_size
    while node.op in layout_ops:
      src = node.args[0]
      if node.op == ExprOp.SCATTER:
        if (
          node.type.pattern is not None
          or node.args[0].support != frozenset()
          or node.args[1].value is None
          or node.attrs["combine"] != "add"
          or node.attrs["mode"] == "fill"
        ):
          break
        src = node.args[2]
        if src.type.pattern is not None:
          break
      if src.id in root_ids or uses[src.id] != 1 or src.type.storage_size != size:
        break
      chain.append(node)
      node = src
    if not chain or node.op != ExprOp.MATMUL or len(node.args) != 2 or sum(arg.type.pattern is not None for arg in node.args) != 1:
      continue
    identity = np.arange(size, dtype=np.int64)
    slots = identity
    for move in chain:
      if move.op == ExprOp.SCATTER and not np.array_equal(scatter_targets(move), identity):
        break
      if move.op == ExprOp.TRANSPOSE or move.op == ExprOp.RESHAPE and move.type.pattern is not None:
        (reads,) = compact_reads(move)
        if np.any(reads < 0) or len(np.unique(reads)) != size:
          break
        slots = reads[slots]
    else:
      result[node.id] = root, np.argsort(slots), tuple(chain)
  return result


def _reads(node: Expr) -> list[tuple[Expr, int]]:
  """How ``node`` reads each argument: each element once, some elements several times, or only whole
  from a buffer, as calls, prints and sparse plans do."""
  op = ExprOp(node.op)
  if op == ExprOp.MATMUL and len(node.args) == 2 and sum(a.type.pattern is not None for a in node.args) == 1:
    return [(a, _MANY if a.type.pattern is None else _BUFFER) for a in node.args]
  if node.type.pattern is not None or any(a.type.pattern is not None for a in node.args):
    return [(a, _BUFFER) for a in node.args]
  if op in ELEMENTWISE:
    return [(a, _ONCE if a.size == node.size else _MANY) for a in node.args]
  if op == ExprOp.MATMUL:
    trips = node.size * node.args[0].shape[-1]
    return [(a, _ONCE if trips == a.size else _MANY) for a in node.args]
  if op == ExprOp.GATHER:
    if node.args[1].value is None or node.attrs["mode"] == "fill":
      return [(a, _BUFFER) for a in node.args]
    indices = _index_values(node).reshape(-1)
    return [(node.args[0], _ONCE if len(np.unique(indices)) == len(indices) else _MANY)]
  if op == ExprOp.SCATTER:
    if node.args[0].support == frozenset() and node.args[1].value is not None and node.attrs["combine"] == "add" and node.attrs["mode"] != "fill":
      return [(node.args[2], _ONCE)]
    return [(a, _BUFFER) for a in node.args]
  if op in ALIASES | {ExprOp.TRANSPOSE, ExprOp.SLICE, ExprOp.REDUCE, ExprOp.STACK, ExprOp.CONCAT}:
    return [(a, _ONCE) for a in node.args]
  return [(a, _BUFFER) for a in node.args]


def plan_fusion(
  fun: ConcreteFunction,
  invariant: frozenset[frozenset[int]],
  *,
  roots: Sequence[Expr],
  scalar: bool,
  seed_pairs: bool,
  support_of: Callable[[Expr], Support],
  reads_at: Callable[[Expr, ProgramNode], Sequence[tuple[Expr, ProgramNode]]],
  shares_traversal: Callable[[Expr, Expr], bool],
) -> tuple[frozenset[int], dict[int, tuple[Expr, ...]], dict[int, tuple[Expr, ...]], dict[int, SeedEpilogue]]:
  """The deferred node ids, scatter sums keyed by their output id, and consumer loop groups keyed
  by each member id, and supported matrix-vector epilogues keyed by their final operation.

  A movement op or a constant is an index remap, deferred unless it is an output or a consumer needs
  it whole. A strided slice or a gather is also stored when one read visits entries repeatedly.
  A remap is stored when several reads would duplicate a libm call in the deferred chain beneath it.
  Several reads of a stored input stay index remaps. A dense elementwise node is deferred when one
  consumer reads each of its elements once, through any chain of deferred movement ops; one that holds a libm call is deferred only into a
  single read, so the call is not duplicated. Agreeing reads can share a deferred value and merge
  independent consumer loops over the same domain. Everything else is stored: reductions, products,
  scatters, calls and outputs. A static gather can read a vector concatenation piecewise. A sparse
  or seed-pair map can defer into its sole reader when their traversals agree. A read stays before
  a scatter that may overwrite its source.

  ``invariant`` holds, for a function mapped or run as a loop body, the input positions each map or
  loop holds the same at every trip. The trip is a range that a node depending only on those inputs
  does not depend on, so such a node is not deferred into a consumer that depends on the others, and
  stays where ``hoist_invariant`` can lift it out of the loop. Everything else in a loop body fuses
  as in any procedure, within one trip: the body is its own procedure, and the loop reads its
  operands from buffers, so no value is evaluated inside the carried range on the loop's behalf. A procedure lowered as scalars keeps a node with known zeros stored,
  since its known zeros stay zero where the arithmetic would give a NaN.

  A left-associated sum whose operands are dense scatters it alone reads, directly or through a
  reshape, is stored once: it zeroes its buffer and each scatter accumulates into it in order. A
  scatter that repeats an index joins only if no earlier operand writes that index, so each element
  still adds its terms in the order of the expression."""
  nodes, outputs = tuple(topo(roots)), frozenset(o.id for o in roots)
  readers: dict[int, list[tuple[Expr, int]]] = {}
  for node in nodes:
    for arg, kind in _reads(node):
      readers.setdefault(arg.id, []).append((node, kind))
  position = {x.id: 1 << k for k, x in enumerate(fun.inputs)}
  depends: dict[int, int] = {}
  for node in nodes:
    depends[node.id] = position.get(node.id, 0) | reduce(or_, (depends[a.id] for a in node.args), 0)
  fixed = [reduce(or_, (1 << k for k in positions), 0) for positions in invariant]

  def hoistable(producer: int, consumer: int) -> bool:
    return any(depends[producer] & ~mask == 0 and depends[consumer] & ~mask for mask in fixed)

  updates = {node.id: node for node in nodes if node.op == ExprOp.SCATTER and node.attrs["mode"] != "fill" and node.args[0].support != frozenset()}
  aliases: dict[int, int] = {}
  for node in nodes:
    # A scatter may overwrite its base, so successive versions share the same write barrier.
    aliases[node.id] = aliases[node.args[0].id] if node.op in ALIASES | {ExprOp.SLICE, ExprOp.PRINT} or node.id in updates else node.id
  storage = {source: 1 << k for k, source in enumerate(dict.fromkeys(aliases[node.args[0].id] for node in updates.values()))}
  schedule = {node.id: k for k, node in enumerate(nodes)}
  writes = [(schedule[node.id], storage[aliases[node.args[0].id]]) for node in updates.values()]
  reads_storage: dict[int, int] = {}
  if writes:
    for node in nodes:
      reads_storage[node.id] = storage.get(aliases[node.id], 0) | reduce(or_, (reads_storage[a.id] for a in node.args), 0)

  def crosses_write(producer: int, consumer: int) -> bool:
    return any(schedule[producer] < at < schedule[consumer] and reads_storage[producer] & source for at, source in writes)

  def keep_zeros(node: Expr) -> bool:
    return (scalar and (support := support_of(node)) is not None and len(support) < node.size) or (
      seed_pairs and len(node.shape) >= 2 and any((support := support_of(e)) is not None and len(support) < e.size for e in (node, *node.args))
    )

  inline_costly: set[int] = set()
  for node in nodes:
    op, uses = ExprOp(node.op), readers.get(node.id, ())
    if (
      node.id not in outputs
      and (op in ELEMENTWISE or op in MOVEMENT)
      and len(uses) == 1
      and uses[0][1] == _ONCE
      and all(e.type.pattern is None for e in (node, *node.args))
      and not hoistable(node.id, uses[0][0].id)
      and not crosses_write(node.id, uses[0][0].id)
      and not (op in ELEMENTWISE and keep_zeros(node))
      and (op in ELEMENTWISE and ELEMENTWISE[op].expensive or any(a.id in inline_costly for a in node.args))
    ):
      inline_costly.add(node.id)
  deferred: set[int] = set()
  # Each node's reads as seen through deferred movement: (consumer id, how it reads, read directly).
  terminal: dict[int, list[tuple[int, int, bool]]] = {}
  for node in reversed(nodes):
    op = ExprOp(node.op)
    edges: list[tuple[int, int, bool]] = []
    for consumer, kind in readers.get(node.id, ()):
      if consumer.id in deferred and (consumer.op in MOVEMENT or consumer.op == ExprOp.CONCAT):
        # A selected concat can read a safe placeholder from an unselected segment.
        kind = _MANY if consumer.op == ExprOp.CONCAT else kind
        edges += [(c, max(kind, k), False) for c, k, _ in terminal[consumer.id]]
      else:
        # A stored alias that is not an output shares its source's buffer.
        shares = ExprOp(consumer.op) in ALIASES and consumer.id not in outputs
        edges.append((consumer.id, _BUFFER if shares else kind, True))
    terminal[node.id] = edges
    dense = all(e.type.pattern is None for e in (node, *node.args))
    if (
      node.id not in outputs
      and dense
      and op in MOVEMENT
      and (node.op != ExprOp.GATHER or node.args[1].value is not None and node.attrs["mode"] != "fill")
      and (len(edges) <= 1 or _free(node) or not any(a.id in inline_costly for a in node.args))
      and all(k == _ONCE or k == _MANY and (_free(node) or op == ExprOp.TRANSPOSE) for _, k, _ in edges)
      and not any(crosses_write(node.id, consumer) for consumer, _, _ in edges)
    ):
      deferred.add(node.id)
    if (
      op == ExprOp.CONCAT
      and dense
      and len(node.shape) == 1
      and all(arg.type.dtype == node.type.dtype for arg in node.args)
      and node.id not in outputs
      and len(uses := readers.get(node.id, ())) == 1
      and uses[0][0].op == ExprOp.GATHER
      and uses[0][0].args[1].value is not None
      and uses[0][0].attrs["mode"] != "fill"
      and all(k != _BUFFER and not crosses_write(node.id, consumer) for consumer, k, _ in edges)
    ):
      deferred.add(node.id)

  def scatter(arg: Expr, parent: Expr) -> Expr | None:
    while arg.id in deferred and ExprOp(arg.op) in ALIASES:
      arg = arg.args[0]
    edges = terminal[arg.id]
    if ExprOp(arg.op) != ExprOp.SCATTER or arg.id in outputs or arg.size != parent.size or len(edges) != 1 or edges[0][:2] != (parent.id, _ONCE):
      return None
    return (
      arg
      if arg.type.pattern is None
      and arg.args[0].support == frozenset()
      and arg.args[1].value is not None
      and arg.attrs["combine"] == "add"
      and arg.attrs["mode"] != "fill"
      else None
    )

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
    if any(crosses_write(leaf.id, node.id) for leaf in (*leaves, last)):
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
    if keep_zeros(node):
      continue
    edges = terminal[node.id]
    costly = ELEMENTWISE[op].expensive or any(a.id in expensive for a in node.args)
    if (
      len({c for c, _, _ in edges}) == 1
      and all(k == _ONCE for _, k, _ in edges)
      and (len(edges) == 1 or not costly and all(d for *_, d in edges))
      and not hoistable(node.id, edges[0][0])
      and not crosses_write(node.id, edges[0][0])
    ):
      deferred.add(node.id)
      if costly:
        expensive.add(node.id)
  groups = _merge_consumers(
    nodes,
    outputs,
    deferred,
    sums,
    readers,
    reads_at,
    lambda node, consumer: hoistable(node.id, consumer) or crosses_write(node.id, consumer),
    keep_zeros,
  )
  mapped_terminals: dict[int, int] = {}
  for node in reversed(nodes):
    uses = readers.get(node.id, ())
    if (node.type.pattern is None and not (seed_pairs and node.op in ELEMENTWISE)) or node.id in outputs or len(uses) != 1:
      continue
    consumer = uses[0][0]
    terminal_id = mapped_terminals.get(consumer.id, consumer.id)
    if hoistable(node.id, terminal_id) or crosses_write(node.id, terminal_id) or not shares_traversal(node, consumer):
      continue
    if node.type.pattern is not None and scalar and (support := support_of(node)) is not None and len(support) < node.type.storage_size:
      continue
    deferred.add(node.id)
    mapped_terminals[node.id] = terminal_id

  epilogues: dict[int, SeedEpilogue] = {}
  if seed_pairs:
    for view in nodes:
      if view.op != ExprOp.RESHAPE or len(view.shape) != 2 or view.id not in deferred:
        continue
      product = view.args[0]
      if (
        product.op != ExprOp.MATMUL
        or product.shape != (*view.shape, 1)
        or any(arg.type.pattern is not None for arg in product.args)
        or product.id in outputs
        or len(readers[product.id]) != 1
        or (support := support_of(product)) is None
        or not 0 < len(support) < product.size
      ):
        continue
      seeds = np.unique(support_indices(support) // view.shape[1])
      if len(seeds) * view.shape[1] != len(support):
        continue
      chain = [product, view]
      while len(uses := readers.get(chain[-1].id, ())) == 1:
        consumer, kind = uses[0]
        if (
          kind != _ONCE
          or consumer.op not in ELEMENTWISE
          or consumer.shape != view.shape
          or consumer.type.dtype != product.type.dtype
          or support_of(consumer) != support
          or any(
            arg.id in mapped_terminals or ((known := support_of(arg)) is not None and len(known) != arg.size)
            for arg in consumer.args
            if arg is not chain[-1]
          )
        ):
          break
        chain.append(consumer)
        if consumer.id not in deferred:
          if not hoistable(product.id, consumer.id) and not crosses_write(product.id, consumer.id):
            epilogues[consumer.id] = SeedEpilogue(tuple(chain), seeds)
          break
    products = {plan.chain[0].id: plan for plan in epilogues.values()}
    for result, plan in epilogues.items():
      if result in outputs or len(uses := readers.get(result, ())) != 1:
        continue
      view = uses[0][0]
      if view.op != ExprOp.RESHAPE or view.id not in deferred or view.shape != (*plan.chain[-1].shape, 1) or len(readers.get(view.id, ())) != 1:
        continue
      consumer = readers[view.id][0][0]
      following = products.get(consumer.id)
      if following is not None and consumer.args[1] is view and np.array_equal(plan.seeds, following.seeds):
        epilogues[result] = replace(plan, packed_consumer=consumer.id)
  return frozenset(deferred), sums, groups, epilogues


def _merge_consumers(
  nodes: Sequence[Expr],
  outputs: frozenset[int],
  deferred: set[int],
  sums: dict[int, tuple[Expr, ...]],
  readers: dict[int, list[tuple[Expr, int]]],
  reads_at: Callable[[Expr, ProgramNode], Sequence[tuple[Expr, ProgramNode]]],
  barrier: Callable[[Expr, int], bool],
  keep_zeros: Callable[[Expr], bool],
) -> dict[int, tuple[Expr, ...]]:
  """Propagate consumer domains backwards and merge loops only where every read agrees."""
  eligible = {
    node.id
    for node in nodes
    if (node.op in ELEMENTWISE or node.op in MOVEMENT - {ExprOp.CONST})
    and all(e.type.pattern is None for e in (node, *node.args))
    and node.id not in sums
    and not (node.op == ExprOp.GATHER and (node.args[1].value is None or node.attrs["mode"] == "fill"))
    and not (node.op in ELEMENTWISE and keep_zeros(node))
  }
  roots = {node.id: node for node in nodes if node.id in eligible and node.id not in deferred}
  bits = {root: 1 << k for k, root in enumerate(roots)}
  upstream: dict[int, int] = {}
  for node in nodes:
    upstream[node.id] = bits.get(node.id, 0) | reduce(or_, (upstream[a.id] for a in node.args), 0)
  positions = {node.id: k for k, node in enumerate(nodes)}
  stops = [k for k, node in enumerate(nodes) if node.op in (ExprOp.SCATTER, ExprOp.PRINT, ExprOp.LOOP, ExprOp.COND, ExprOp.EXTERN)]
  groups = {root: {root} for root in roots}
  merged_dependencies: dict[int, tuple[int, int]] = {}
  owners = {root: root for root in roots}
  requests: dict[int, set[tuple[int, ProgramNode, int] | None]] = {}
  index = p.var("fusion_index")
  for node in reversed(nodes):
    if not node.args:
      continue
    uses = requests.get(node.id, set())
    known = [use for use in uses if use is not None]
    members = set().union(*(groups[owners[root]] for root, _, _ in known))
    masks = reduce(or_, (bits[root] for root in members), 0)
    dependencies = reduce(or_, (upstream[root] ^ bits[root] for root in members), 0)
    # Independent roots can become cyclic through two previously merged consumer groups.
    while True:
      previous = dependencies
      for mask, ancestors in merged_dependencies.values():
        if dependencies & mask:
          dependencies |= ancestors
      if dependencies == previous:
        break
    same = (
      node.id in eligible
      and (node.op in ELEMENTWISE or len(readers.get(node.id, ())) > 1)
      and node.id not in outputs
      and known
      and len(known) == len(uses)
      and all(at is known[0][1] and kind == _ONCE and roots[root].size == roots[known[0][0]].size for root, at, kind in known)
      and not masks & dependencies
      and not any(barrier(node, root) for root in members)
      and not any(min(positions[root] for root in members) < stop < max(positions[root] for root in members) for stop in stops)
    )
    if same:
      owner = owners[known[0][0]]
      for root in members:
        merged_dependencies.pop(owners[root], None)
      for root in members:
        owners[root] = owner
      groups[owner] = members
      if len(members) > 1:
        merged_dependencies[owner] = masks, dependencies
      deferred.add(node.id)
      active = [(owner, known[0][1], _ONCE)]
    elif node.id in deferred:
      # A divergent deferred view stays a remap, but cannot justify further shared work upstream.
      active = (
        [known[0]]
        if known
        and len(known) == len(uses)
        and all(owners[root] == owners[known[0][0]] and at is known[0][1] and kind == known[0][2] for root, at, kind in known)
        else [None]
      )
    elif node.id in roots:
      active = [(node.id, index, _ONCE)]
    else:
      active = [None]
    kinds = {arg.id: kind for arg, kind in _reads(node)}
    if any(use is None for use in active):
      for arg, _ in _reads(node):
        requests.setdefault(arg.id, set()).add(None)
    for use in active:
      if use is None:
        continue
      root, at, kind = use
      for arg, source in reads_at(node, at):
        requests.setdefault(arg.id, set()).add((root, source, max(kind, kinds[arg.id])))
  merged = {owners[root]: tuple(roots[r] for r in roots if r in groups[owners[root]]) for root in roots if len(groups[owners[root]]) > 1}
  return {node.id: group for group in merged.values() for node in group}
