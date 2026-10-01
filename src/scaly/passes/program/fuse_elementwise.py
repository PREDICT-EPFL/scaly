"""Elementwise producer fusion for the Program IR."""

from __future__ import annotations

import functools
from bisect import bisect_right

from ...ir.expr import op_def, registered_ops, registry_version
from ...ir.match import Pattern, rewrite
from ...ir.program import ProgramNode, ProgramOp
from ._common import (
  _alias_sources,
  buffer_refs,
  inline_producer as _as_inline_producer,
  _map_procs,
  _private_decls,
  _proc_parts,
  _rebuild_proc,
  _resolve_alias,
  _size_of,
  prune_dead_buffers,
  substitute_var as _subst_var,
  trip_count as _trip_count,
  _walk,
  rebuild_program,
)


@functools.cache
def _expensive_ops(version: int) -> frozenset[ProgramOp]:
  """The program ops of the expression ops with both the ``elementwise`` and the ``expensive``
  trait: libm calls, which fusion does not duplicate. Keyed by the registry's version."""
  traits = [op_def(op).traits for op in registered_ops()]
  return frozenset(t["elementwise"] for t in traits if t.get("expensive") and "elementwise" in t)


def _has_expensive(node: ProgramNode) -> bool:
  expensive = _expensive_ops(registry_version())
  return any(n.op in expensive for n in _walk(node))


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


def _reads_through_a_table(node: ProgramNode, buf: str) -> bool:
  """Whether ``node`` reads ``buf`` at an index that itself loads (a gather's table, a run-time
  index): inlining a producer there evaluates its index arithmetic per element at run time."""
  for n in _walk(node):
    if n.op == ProgramOp.LOAD and n.args[0].attrs["buffer"] == buf and any(m.op == ProgramOp.LOAD for a in n.args[0].args for m in _walk(a)):
      return True
  return False


_INDEX_ARITH = frozenset({ProgramOp.ADD, ProgramOp.SUB, ProgramOp.MUL, ProgramOp.DIV, ProgramOp.MOD, ProgramOp.NEG, ProgramOp.CONST_INT})


def _composes(rhs: ProgramNode, var: str, consumer: ProgramNode, buf: str, tables: set[str]) -> bool:
  """Whether a moving producer ``buf[v] = src[g(v)]`` read as ``buf[k[..]]`` leaves an index that
  ``fold_arith`` turns into one table: ``g`` is integer arithmetic on ``v`` and constants, and every
  read of ``buf`` in ``consumer`` is at an entry of a constant index table."""
  index = rhs.args[0].args
  if len(index) != 1 or any(n.op not in _INDEX_ARITH and not (n.op == ProgramOp.VAR and n.attrs["name"] == var) for n in _walk(index[0])):
    return False
  reads = [n.args[0] for n in _walk(consumer) if n.op == ProgramOp.LOAD and n.args[0].attrs["buffer"] == buf]
  return all(len(v.args) == 1 and v.args[0].op == ProgramOp.LOAD and v.args[0].args[0].attrs["buffer"] in tables for v in reads)


def _total_load_executions(node: ProgramNode, buf: str) -> int | None:
  """How many times the loads of ``buf`` in ``node`` run together: each occurrence (tree
  multiplicity, so ``buf*buf`` counts twice) times its enclosing loops' trip counts. None under a
  non-static bound. An expensive producer inlined where this exceeds its element count would be
  computed more than once per element."""
  total = 0
  stack = [(node, 1)]
  while stack:
    n, factor = stack.pop()
    if n.op == ProgramOp.FOR:
      tc = _trip_count(n.args[0])
      if tc is None:
        return None
      stack.extend((sub, factor * tc) for sub in n.args[1:])
      continue
    if n.op == ProgramOp.LOAD and n.args[0].attrs["buffer"] == buf:
      total += factor
    stack.extend((sub, factor) for sub in n.args)
  return total


def _expand_inlinables(node: ProgramNode, inlinable: dict[str, tuple[str, ProgramNode]]) -> ProgramNode:
  """Replace each ``LOAD(VIEW(buf,[E]))`` with the producer RHS at ``E``; the driver revisits each
  replacement, so a whole chain of single-use producers collapses in one bottom-up pass."""

  def inlinable_load(n: ProgramNode) -> bool:
    return n.args[0].attrs["buffer"] in inlinable and len(n.args[0].args) == 1

  def expand(n: ProgramNode) -> ProgramNode:
    v, rhs = inlinable[n.args[0].attrs["buffer"]]
    return _subst_var(rhs, v, n.args[0].args[0])

  return rewrite(node, [Pattern(ProgramOp.LOAD, inlinable_load, expand)], rebuild=rebuild_program, fixpoint=False, revisit=True)


def fuse_elementwise(prog: ProgramNode) -> ProgramNode:
  return _map_procs(prog, _fuse_proc)


def _fuse_proc(proc: ProgramNode) -> ProgramNode:
  params, body = _proc_parts(proc)
  private = _private_decls(body)
  if not private:
    return proc
  aliases = _alias_sources(body)
  tables = {s.attrs["name"] for s in body if s.op == ProgramOp.BUFFER and "values" in s.attrs and s.dtype.is_integer}
  # A buffer that backs an alias (a zero-copy pointer view) must keep its storage — never inline it.
  pinned = set(_alias_sources(body).values())

  # Reference census over the real (non-decl) statements.
  load_in: dict[str, set[int]] = {name: set() for name in private}
  store_in: dict[str, set[int]] = {name: set() for name in private}
  call_in: dict[str, set[int]] = {name: set() for name in private}
  resolved_refs = [buffer_refs(stmt, aliases) for stmt in body]
  # Statement positions writing each buffer, ascending: "is it written between i and ci" becomes a
  # binary search instead of a scan of every statement in between.
  writes_at: dict[str, list[int]] = {}
  for j, refs_j in enumerate(resolved_refs):
    for b in refs_j.writes:
      writes_at.setdefault(b, []).append(j)

  def written_between(names: set[str], lo: int, hi: int) -> bool:
    """Whether a statement in ``(lo, hi]`` writes one of ``names``."""
    for b in names:
      at = writes_at.get(b)
      if at:
        k = bisect_right(at, lo)
        if k < len(at) and at[k] <= hi:
          return True
    return False

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
  # times). Expensive (libm) producers additionally must not be computed more than once per
  # element in all: ``buf*buf`` is refused, a reduction's partial sums reading each element once are not.
  inlinable: dict[str, tuple[str, ProgramNode]] = {}
  expanded_expensive: dict[str, bool] = {}
  expanded_reads: dict[str, frozenset[str]] = {}
  for i, stmt in enumerate(body):
    pr = _as_inline_producer(stmt)
    if pr is None or pr[0] not in private:
      continue
    buf, v, rhs = pr
    if buf in pinned or call_in[buf] or store_in[buf] != {i}:
      continue
    consumers = load_in[buf] - {i}
    if len(consumers) != 1:
      continue
    (ci,) = consumers
    producer_size = _size_of(private[buf].attrs["shape"])
    execs = _max_load_executions(body[ci], buf, 1)
    if execs is None or execs > producer_size:
      continue
    # A producer that only moves data (a transpose, a tile) is cheaper copied than read through a
    # gather's table, which would divide its index per element; one that computes is not. Under a
    # constant table the two indices compose into one table (``fold_arith``), and nothing is copied.
    moved = rhs.op == ProgramOp.LOAD and rhs.args[0].attrs["buffer"] not in inlinable and _reads_through_a_table(body[ci], buf)
    if moved and not _composes(rhs, v, body[ci], buf, tables):
      continue
    rhs_loads = buffer_refs(rhs).loads
    is_expensive = _has_expensive(rhs) or any(expanded_expensive.get(name, False) for name in rhs_loads)
    if is_expensive and ((total := _total_load_executions(body[ci], buf)) is None or total > producer_size):
      continue
    moved_reads = set(buffer_refs(rhs, aliases).reads)
    for name in rhs_loads & inlinable.keys():
      moved_reads.discard(_resolve_alias(name, aliases))
      moved_reads.update(expanded_reads[name])
    if ci <= i or written_between(moved_reads, i, ci):
      continue
    inlinable[buf] = (v, rhs)
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


__all__ = ["fuse_elementwise"]
