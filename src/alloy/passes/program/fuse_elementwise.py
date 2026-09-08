"""Elementwise producer fusion for the Program IR."""

from __future__ import annotations

from ...ir.match import Pattern, rewrite
from ...ir.program import ProgramNode, ProgramOp
from ._common import (
  _alias_sources,
  _call_arg_buffer,
  _map_procs,
  _postorder,
  _private_decls,
  _proc_parts,
  _rebuild_proc,
  _size_of,
  _stmt_refs,
  _walk,
  rebuild_program,
)

_EXPENSIVE_OPS: frozenset[ProgramOp] = frozenset(
  {
    ProgramOp.SIN,
    ProgramOp.COS,
    ProgramOp.TAN,
    ProgramOp.ASIN,
    ProgramOp.ACOS,
    ProgramOp.ATAN,
    ProgramOp.SINH,
    ProgramOp.COSH,
    ProgramOp.TANH,
    ProgramOp.ERF,
    ProgramOp.EXP,
    ProgramOp.LOG,
    ProgramOp.SQRT,
    ProgramOp.POW,
    ProgramOp.ATAN2,
  }
)


# ---------------------------------------------------------------------------
# Pass pipeline.
def _as_inline_producer(stmt: ProgramNode) -> tuple[str, str, ProgramNode] | None:
  """If ``stmt`` is ``for v in [0,N) step 1 { buf[v] = rhs }`` return ``(buf, v, rhs)``, else None.

  This is the shape every elementwise / SLICE / GATHER / copy rule emits: a single contiguous
  loop with one STORE indexed by the loop variable. Because the store index *is* the loop var,
  ``buf[k] == rhs[v:=k]`` for all ``k`` the loop visits, so a consumer reading ``buf[E]`` can
  inline ``rhs[v:=E]`` for any in-range ``E``. Reductions (extra init STORE), MATMUL/TRANSPOSE
  (composite store index), and STACK/CONCAT (multiple loops per buffer) deliberately don't match.
  """
  if stmt.op != ProgramOp.FOR:
    return None
  rng, *body = stmt.args
  if len(body) != 1 or body[0].op != ProgramOp.STORE:
    return None
  store = body[0]
  target = store.args[0]
  if target.op != ProgramOp.VIEW or len(target.args) != 1:
    return None
  v = rng.attrs["name"]
  idx = target.args[0]
  if idx.op != ProgramOp.VAR or idx.attrs["name"] != v:
    return None
  start, _stop, step = rng.args
  if not (start.op == ProgramOp.CONST_INT and start.attrs["value"] == 0):
    return None
  if not (step.op == ProgramOp.CONST_INT and step.attrs["value"] == 1):
    return None
  return target.attrs["buffer"], v, store.args[1]


def _subst_var(node: ProgramNode, vname: str, repl: ProgramNode) -> ProgramNode:
  """Substitute every ``VAR(vname)`` in ``node`` with ``repl`` (the consumer's index expr)."""
  pattern = Pattern(ProgramOp.VAR, lambda n: n.attrs["name"] == vname, lambda n: repl)
  return rewrite(node, [pattern], rebuild=rebuild_program, fixpoint=False)


def _has_expensive(node: ProgramNode) -> bool:
  return any(n.op in _EXPENSIVE_OPS for n in _walk(node))


def _trip_count(rng: ProgramNode) -> int | None:
  """Static iteration count of a ``[0, stop) step 1`` RANGE, or None if not statically known."""
  start, stop, step = rng.args
  if start.op != ProgramOp.CONST_INT or stop.op != ProgramOp.CONST_INT or step.op != ProgramOp.CONST_INT:
    return None
  span = int(stop.attrs["value"]) - int(start.attrs["value"])
  step_v = int(step.attrs["value"])
  return max(0, -(-span // step_v)) if step_v > 0 else None


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
  # A buffer that backs an alias (a zero-copy pointer view) must keep its storage — never inline it.
  pinned = set(_alias_sources(body).values())

  # Reference census over the real (non-decl) statements.
  load_in: dict[str, set[int]] = {name: set() for name in private}
  store_in: dict[str, set[int]] = {name: set() for name in private}
  call_in: dict[str, set[int]] = {name: set() for name in private}
  for i, stmt in enumerate(body):
    if stmt.op == ProgramOp.BUFFER:
      continue
    loads, stores, calls = _stmt_refs(stmt)
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
  inlinable: dict[str, tuple[str, ProgramNode]] = {}
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
    if _has_expensive(rhs) and _count_buf_loads(body[ci], buf) != 1:
      continue
    inlinable[buf] = (v, rhs)

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

  return _prune_dead_buffers(_rebuild_proc(proc, params, new_body))


def _prune_dead_buffers(proc: ProgramNode) -> ProgramNode:
  """Drop BUFFER decls (private or constant) no longer referenced by any VIEW or CALL arg."""
  params, body = _proc_parts(proc)
  referenced: set[str] = set()
  for stmt in body:
    if stmt.op == ProgramOp.BUFFER:
      continue
    for n in _walk(stmt):
      if n.op == ProgramOp.VIEW:
        referenced.add(n.attrs["buffer"])
      elif n.op == ProgramOp.BUFFER:
        referenced.add(n.attrs["name"])
      elif n.op == ProgramOp.CALL:
        for a in n.args:
          name = _call_arg_buffer(a)
          if name is not None:
            referenced.add(name)
  # A *live* alias keeps its source alive (chase chains to a fixpoint).
  alias_src = _alias_sources(body)
  changed = True
  while changed:
    changed = False
    for name, src in alias_src.items():
      if name in referenced and src not in referenced:
        referenced.add(src)
        changed = True
  param_names = {pp.attrs["name"] for pp in params}
  kept = [s for s in body if s.op != ProgramOp.BUFFER or s.attrs["name"] in referenced or s.attrs["name"] in param_names]
  if len(kept) == len(body):
    return proc
  return _rebuild_proc(proc, params, kept)


# ---------------------------------------------------------------------------
# Pass 2: empty-loop removal + unit-loop unrolling.
# ---------------------------------------------------------------------------


__all__ = ["fuse_elementwise"]
