"""Elementwise producer fusion for the Program IR."""

from __future__ import annotations

from ...ad.rules import ELEMENTWISE
from ...ir import program as p
from ...ir.match import Pattern, rewrite
from ...ir.program import ProgramNode, ProgramOp, walk_program
from ._common import (
  _alias_sources,
  buffer_refs,
  inline_producer as _as_inline_producer,
  _map_procs,
  _postorder,
  _private_decls,
  _proc_parts,
  _rebuild_proc,
  _resolve_alias,
  _size_of,
  affine_coefficients,
  fold_index,
  prune_dead_buffers,
  substitute_var as _subst_var,
  trip_count as _trip_count,
  rebuild_program,
)

_EXPENSIVE_OPS = frozenset(row.program for row in ELEMENTWISE.values() if row.expensive)


def _has_expensive(node: ProgramNode) -> bool:
  return any(n.op in _EXPENSIVE_OPS for n in walk_program(node))


def _max_load_executions(node: ProgramNode, buf: str, factor: int) -> int | None:
  """Max number of times any single load of ``buf`` runs, given ``factor`` enclosing iterations.

  This is the recompute metric: a load nested in loops of total trip count ``T`` evaluates the
  inlined producer ``T`` times. A consumer whose ``T`` exceeds the producer's element count
  re-reads elements — fine for the elementwise/reduction case (``T == producer_size``), fatal
  for a matmul/contraction (``T == m*n*k ≫ producer_size``). Returns None if any enclosing loop
  has a non-static bound (treat as unbounded — don't inline)."""
  best = 0
  stack = [(node, factor)]
  while stack:
    n, factor = stack.pop()
    if n.op == ProgramOp.FOR:
      tc = _trip_count(n.args[0])
      if tc is None:
        return None
      stack.extend((sub, factor * tc) for sub in n.args[1:])
      continue
    if n.op == ProgramOp.LOAD and n.args[0].attrs["buffer"] == buf:
      best = max(best, factor)
    stack.extend((sub, factor) for sub in n.args)
  return best


def _count_buf_loads(node: ProgramNode, buf: str) -> int:
  """Textual occurrences of a load of ``buf`` in ``node`` (tree multiplicity: ``buf*buf`` is 2)."""
  count: dict[int, int] = {}
  for n in _postorder(node):
    count[id(n)] = (n.op == ProgramOp.LOAD and n.args[0].attrs["buffer"] == buf) + sum(count[id(a)] for a in n.args)
  return count[id(node)]


def _expand_inlinables(node: ProgramNode, inlinable: dict[str, tuple[tuple[tuple[str, int], ...], ProgramNode]]) -> ProgramNode:
  """Replace each ``LOAD(VIEW(buf,[E]))`` with the producer RHS at ``E``; the driver revisits each
  replacement, so a whole chain of single-use producers collapses in one bottom-up pass.

  A nest producer's view that runs row-major over its loops reads at ``E`` itself, scaled; any
  other reads its coordinates as quotients and remainders of ``E``, as a flat broadcast did."""

  def inlinable_load(n: ProgramNode) -> bool:
    return n.args[0].attrs["buffer"] in inlinable and len(n.args[0].args) == 1

  def expand(n: ProgramNode) -> ProgramNode:
    loops, rhs = inlinable[n.args[0].attrs["buffer"]]
    index = n.args[0].args[0]
    if len(loops) == 1:
      return _subst_var(rhs, loops[0][0], index)
    strides, stride = {}, 1
    for name, trip in reversed(loops):
      strides[name] = stride
      stride *= trip
    names = frozenset(strides)

    def row_major(view: ProgramNode) -> ProgramNode:
      # A view running row-major over the producer's loops reads ``scale * index + rest``.
      coefficients = affine_coefficients(view.args[0], names) or {}
      scale = coefficients.get(loops[-1][0], 0)
      if not scale or any(coefficients.get(name, 0) != scale * s for name, s in strides.items()):
        return view
      rest = view.args[0]
      for name in names:
        rest = _subst_var(rest, name, p.const_int(0))
      term = index if scale == 1 else p.mul(index, p.const_int(scale))
      rest = fold_index(rest)
      at = term if rest.op == ProgramOp.CONST_INT and rest.attrs["value"] == 0 else p.add(rest, term)
      return ProgramNode(ProgramOp.VIEW, (at,), view.attrs, view.dtype)

    rhs = rewrite(rhs, [Pattern(ProgramOp.VIEW, lambda v: len(v.args) == 1, row_major)], rebuild=rebuild_program, fixpoint=False)
    for name, trip in loops:
      coord = index if strides[name] == 1 else p.div(index, p.const_int(strides[name]))
      rhs = _subst_var(rhs, name, coord if name == loops[0][0] else p.mod(coord, p.const_int(trip)))
    return rhs

  return rewrite(node, [Pattern(ProgramOp.LOAD, inlinable_load, expand)], rebuild=rebuild_program, fixpoint=False, revisit=True)


def _in_nest_order(rhs: ProgramNode, loops: tuple[tuple[str, int], ...]) -> bool:
  """Whether each read of a nest producer is a broadcast or slice: its strides shrink inwards.

  Inlining such a read turns its coordinates into quotients and remainders of the consumer's
  index, as a flat broadcast's were. A transposed read is left materialized."""
  if len(loops) == 1:
    return True
  names = [name for name, _ in loops]
  for view in (n for n in walk_program(rhs) if n.op == ProgramOp.VIEW):
    coefficients = affine_coefficients(view.args[0], frozenset(names))
    if coefficients is None:
      return False
    strides = [abs(coefficients[name]) for name in names if coefficients.get(name)]
    if any(outer <= inner for outer, inner in zip(strides, strides[1:])):
      return False
  return True


def fuse_elementwise(prog: ProgramNode) -> ProgramNode:
  return _map_procs(prog, _fuse_proc)


def _fuse_proc(proc: ProgramNode) -> ProgramNode:
  params, body = _proc_parts(proc)
  private = _private_decls(body)
  if not private:
    return proc
  aliases = _alias_sources(body)
  # A buffer that backs an alias (a zero-copy pointer view) must keep its storage — never inline it.
  pinned = set(_alias_sources(body).values())

  # Reference census over the real (non-decl) statements.
  load_in: dict[str, set[int]] = {name: set() for name in private}
  store_in: dict[str, set[int]] = {name: set() for name in private}
  call_in: dict[str, set[int]] = {name: set() for name in private}
  resolved_refs = [buffer_refs(stmt, aliases) for stmt in body]
  for i, stmt in enumerate(body):
    if stmt.op == ProgramOp.BUFFER:
      continue
    refs = buffer_refs(stmt)
    loads, stores, calls = refs.loads, refs.stores, refs.call_args
    for b in loads & private.keys():
      load_in[b].add(i)
    for b in stores & private.keys():
      store_in[b].add(i)
    for b in calls & private.keys():
      call_in[b].add(i)

  # A private buffer is inlinable iff: produced by a single inline-shaped loop, written only
  # there, never passed to a CALL, read by exactly one consumer statement, and that consumer
  # re-reads it at most ``producer_size`` times (no compute blow-up — this is what keeps the
  # producer out of a matmul/contraction operand position, where each element is read m*n*k
  # times). Expensive (libm) producers additionally must appear exactly once textually so the
  # call isn't duplicated (``buf*buf``).
  inlinable: dict[str, tuple[tuple[tuple[str, int], ...], ProgramNode]] = {}
  expanded_expensive: dict[str, bool] = {}
  expanded_reads: dict[str, frozenset[str]] = {}
  for i, stmt in enumerate(body):
    pr = _as_inline_producer(stmt)
    if pr is None or pr[0] not in private:
      continue
    buf, loops, rhs = pr
    if buf in pinned or call_in[buf] or store_in[buf] != {i} or not _in_nest_order(rhs, loops):
      continue
    consumers = load_in[buf] - {i}
    if len(consumers) != 1:
      continue
    (ci,) = consumers
    producer_size = _size_of(private[buf].attrs["shape"])
    execs = _max_load_executions(body[ci], buf, 1)
    if execs is None or execs > producer_size:
      continue
    rhs_loads = buffer_refs(rhs).loads
    is_expensive = _has_expensive(rhs) or any(expanded_expensive.get(name, False) for name in rhs_loads)
    if is_expensive and _count_buf_loads(body[ci], buf) != 1:
      continue
    moved_reads = set(buffer_refs(rhs, aliases).reads)
    for name in rhs_loads & inlinable.keys():
      moved_reads.discard(_resolve_alias(name, aliases))
      moved_reads.update(expanded_reads[name])
    if ci <= i or any(moved_reads & resolved_refs[j].writes for j in range(i + 1, ci + 1)):
      continue
    inlinable[buf] = (loops, rhs)
    expanded_expensive[buf] = is_expensive
    expanded_reads[buf] = frozenset(moved_reads)

  if not inlinable:
    return proc

  # Rewrite every surviving statement (the inlinable producers are dropped) with chains expanded.
  producer_idx = {pr[0]: i for i, stmt in enumerate(body) if (pr := _as_inline_producer(stmt)) is not None}
  drop = {producer_idx[buf] for buf in inlinable}
  new_body: list[ProgramNode] = []
  for i, stmt in enumerate(body):
    if i in drop:
      continue
    if stmt.op == ProgramOp.BUFFER and stmt.attrs.get("name") in inlinable:
      continue  # decl of an inlined buffer
    new_body.append(_expand_inlinables(stmt, inlinable))

  return prune_dead_buffers(_rebuild_proc(proc, params, new_body))
