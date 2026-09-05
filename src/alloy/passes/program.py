"""Program IR optimization passes: ``PROGRAM -> PROGRAM`` rewrites run after lowering.

The pipeline (``optimize_program``) is invoked at the tail of ``passes.lowering.lower_function``,
between the naive lowering and the verifier. Each pass is a self-contained transform over
the hash-consed ``ProgramNode`` graph registered in ``PASS_PIPELINE`` — adding an optimization is
a new entry here, mirroring the ``@lowers`` registry in ``passes/lowering.py`` and the per-op maps
in ``codegen/c.py``. This is the *third* leg of the architecture: lower (expression ->
Program IR), optimize (Program IR -> Program IR), render (Program IR -> C). New optimizations
(CSE, peepholes, GPU schedule passes) slot in as additional passes without touching the
lowerer or renderer.

These optimizations originally lived baked into the now-deleted legacy tape renderer; the
migration re-expressed them as explicit, individually-testable Program IR passes:

- ``combine_scatter_sums`` accumulates sums of single-use zero-filled scatters into one
  destination, avoiding full-length pad buffers in slice adjoints.

- ``fuse_elementwise`` — inline single-use ``private`` elementwise / slice / gather producers
  (a single-``STORE`` ``FOR`` whose store index is the loop var) into their one consumer by
  substituting the producer's scalar RHS at the consumer's load site, then dropping the
  producer loop + buffer. This is loop fusion: chains collapse into one loop and the
  intermediate buffer round-trips vanish (the legacy renderer's no-materialize + inline-read
  tables, now a graph rewrite).

- ``unroll_unit_loops`` — erase statically empty loops and inline single-iteration loops by
  substituting the loop variable with its only value. This keeps canonical loop-shaped producers
  available to ``fuse_elementwise`` first, then removes the scalar-loop noise before rendering.

- ``pack_workspace`` — lifetime-pack ``private`` BUFFERs into shared slots and spill slots
  ≥ ``WORKSPACE_SPILL_THRESHOLD`` doubles to the caller-provided ``w[]`` (so the function
  declares a real ``sz_w`` instead of stack-allocating every temporary; the legacy renderer's
  lifetime/slot/spill packing, now a graph rewrite). Without this the largest benchmark cells
  overflow the 8 MB stack — it gates the merge.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from ..ir.program import ProgramNode, ProgramOp, add, for_, load, store
from ..ir.types import DType, DeviceSpec

# Slots this large (in elements) or bigger move off the C stack into the caller's ``w[]``.
WORKSPACE_SPILL_THRESHOLD = 1024

# Scalar ProgramOp that lower to a libm call. Inlining one into a consumer whose iteration domain
# is larger than the producer's (a broadcast) — or at more than one site — replays the call,
# so the fusion pass keeps these materialized unless the read is one-to-one. Cheap arithmetic
# is always safe to duplicate.
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
# ---------------------------------------------------------------------------

PassFn = Callable[[ProgramNode], ProgramNode]
ProgramObserver = Callable[[str, ProgramNode], None]
PASS_PIPELINE: list[tuple[str, PassFn]] = []


def register_pass(name: str) -> Callable[[PassFn], PassFn]:
  """Append ``fn`` to the optimization pipeline under ``name`` (order = registration order)."""

  def deco(fn: PassFn) -> PassFn:
    PASS_PIPELINE.append((name, fn))
    return fn

  return deco


def optimize_program(prog: ProgramNode, observe: ProgramObserver | None = None) -> ProgramNode:
  """Run every registered pass over a lowered ``PROGRAM`` in order, returning the optimized one."""
  for name, fn in PASS_PIPELINE:
    prog = fn(prog)
    if observe is not None:
      observe(f"pass:{name}", prog)
  return prog


# ---------------------------------------------------------------------------
# Shared Program IR plumbing.
# ---------------------------------------------------------------------------


def _walk(root: ProgramNode) -> Iterable[ProgramNode]:
  """Yield every distinct node at/below ``root`` (id-deduped; hash-consing collapses equals)."""
  seen: set[int] = set()
  stack = [root]
  while stack:
    n = stack.pop()
    if id(n) in seen:
      continue
    seen.add(id(n))
    yield n
    stack.extend(n.args)


def _transform(node: ProgramNode, fn: Callable[[ProgramNode], ProgramNode]) -> ProgramNode:
  """Bottom-up rebuild: rebuild ``node``'s args, then apply ``fn`` to the (maybe) new node.

  ``fn`` returns a replacement node (or its argument unchanged). Hash-consing makes the
  identity check cheap and keeps untouched subtrees shared.
  """
  if node.args:
    new_args = tuple(_transform(a, fn) for a in node.args)
    if any(a is not b for a, b in zip(new_args, node.args, strict=True)):
      node = ProgramNode(node.op, new_args, node.attrs, node.dtype)
  return fn(node)


def _procs(prog: ProgramNode) -> tuple[list[ProgramNode], list[ProgramNode]]:
  """Split a PROGRAM into (procs, kernels)."""
  pc = int(prog.attrs.get("proc_count", 0))
  return list(prog.args[:pc]), list(prog.args[pc:])


def _proc_parts(proc: ProgramNode) -> tuple[list[ProgramNode], list[ProgramNode]]:
  """Split a PROC into (params, body statements)."""
  pc = int(proc.attrs["param_count"])
  return list(proc.args[:pc]), list(proc.args[pc:])


def _rebuild_proc(proc: ProgramNode, params: list[ProgramNode], body: list[ProgramNode], **extra_attrs: object) -> ProgramNode:
  attrs = {**proc.attrs, "param_count": len(params), **extra_attrs}
  return ProgramNode(ProgramOp.PROC, (*params, *body), attrs, proc.dtype)


def _map_procs(prog: ProgramNode, fn: Callable[[ProgramNode], ProgramNode]) -> ProgramNode:
  """Apply a per-PROC transform to every proc, preserving kernels and proc/kernel counts."""
  procs, kernels = _procs(prog)
  procs = [fn(pr) for pr in procs]
  return ProgramNode(ProgramOp.PROGRAM, (*procs, *kernels), prog.attrs, prog.dtype)


def _size_of(shape: tuple[int, ...]) -> int:
  n = 1
  for d in shape:
    n *= int(d)
  return n or 1


def _call_arg_buffer(arg: ProgramNode) -> str | None:
  if arg.op == ProgramOp.BUFFER:
    return arg.attrs["name"]
  if arg.op == ProgramOp.VIEW:
    return arg.attrs["buffer"]
  return None


def _stmt_refs(stmt: ProgramNode) -> tuple[set[str], set[str], set[str]]:
  """``(loads, stores, call_args)``: buffer names this statement LOADs, STOREs to, or passes
  (in or out) to a CALL. CALL args are pointer passes, not loads/stores — tracked separately so
  fusion knows a buffer feeding a CALL must stay materialized."""
  loads: set[str] = set()
  stores: set[str] = set()
  call_args: set[str] = set()
  for n in _walk(stmt):
    if n.op == ProgramOp.LOAD:
      loads.add(n.args[0].attrs["buffer"])
    elif n.op == ProgramOp.STORE:
      stores.add(n.args[0].attrs["buffer"])
    elif n.op == ProgramOp.CALL:
      for a in n.args:
        name = _call_arg_buffer(a)
        if name is not None:
          call_args.add(name)
  return loads, stores, call_args


def _private_decls(body: list[ProgramNode]) -> dict[str, ProgramNode]:
  """name -> BUFFER decl for every ``private`` buffer declared in ``body`` (skips constants)."""
  return {s.attrs["name"]: s for s in body if s.op == ProgramOp.BUFFER and s.attrs.get("address_space") == "private"}


def _alias_sources(body: list[ProgramNode]) -> dict[str, str]:
  """name -> aliased-source name for every alias BUFFER (zero-copy pointer) in ``body``."""
  return {s.attrs["name"]: s.attrs["alias_of"] for s in body if s.op == ProgramOp.BUFFER and "alias_of" in s.attrs}


def _resolve_alias(name: str, alias_src: dict[str, str]) -> str:
  """Follow an alias chain to the buffer that actually owns storage."""
  seen: set[str] = set()
  while name in alias_src and name not in seen:
    seen.add(name)
    name = alias_src[name]
  return name


# ---------------------------------------------------------------------------
# Pass 1: elementwise fusion / inlining.
# ---------------------------------------------------------------------------


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

  def fn(n: ProgramNode) -> ProgramNode:
    if n.op == ProgramOp.VAR and n.attrs["name"] == vname:
      return repl
    return n

  return _transform(node, fn)


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
  if node.op == ProgramOp.FOR:
    rng = node.args[0]
    tc = _trip_count(rng)
    if tc is None:
      return None
    best = 0
    for sub in node.args[1:]:
      r = _max_load_executions(sub, buf, factor * tc)
      if r is None:
        return None
      best = max(best, r)
    return best
  best = factor if node.op == ProgramOp.LOAD and node.args[0].attrs["buffer"] == buf else 0
  for sub in node.args:
    r = _max_load_executions(sub, buf, factor)
    if r is None:
      return None
    best = max(best, r)
  return best


def _count_buf_loads(node: ProgramNode, buf: str) -> int:
  """Textual occurrences of a load of ``buf`` in ``node`` (tree multiplicity: ``buf*buf`` is 2)."""
  memo: dict[int, int] = {}

  def cnt(n: ProgramNode) -> int:
    if id(n) in memo:
      return memo[id(n)]
    base = 1 if n.op == ProgramOp.LOAD and n.args[0].attrs["buffer"] == buf else 0
    memo[id(n)] = base + sum(cnt(a) for a in n.args)
    return memo[id(n)]

  return cnt(node)


def _expand_inlinables(node: ProgramNode, inlinable: dict[str, tuple[str, ProgramNode]]) -> ProgramNode:
  """Replace each ``LOAD(VIEW(buf,[E]))`` with the producer RHS at ``E``, recursively, so a whole
  chain of single-use producers collapses into one fused expression in a single bottom-up pass."""

  def fn(n: ProgramNode) -> ProgramNode:
    if n.op == ProgramOp.LOAD:
      view = n.args[0]
      buf = view.attrs["buffer"]
      if buf in inlinable and len(view.args) == 1:
        v, rhs = inlinable[buf]
        return _expand_inlinables(_subst_var(rhs, v, view.args[0]), inlinable)
    return n

  return _transform(node, fn)


@register_pass("combine_scatter_sums")
def combine_scatter_sums(prog: ProgramNode) -> ProgramNode:
  """Accumulate single-use sums of zero-filled scatters into one destination."""
  return _map_procs(prog, _combine_scatter_sums_proc)


def _combine_scatter_sums_proc(proc: ProgramNode) -> ProgramNode:
  params, body = _proc_parts(proc)
  private = _private_decls(body)
  aliases = _alias_sources(body)
  pinned = set(aliases.values())
  decls = {s.attrs["name"]: s for s in (*params, *body) if s.op == ProgramOp.BUFFER}
  refs = [_stmt_refs(s) if s.op != ProgramOp.BUFFER else (set(), set(), set()) for s in body]
  reads: dict[str, set[int]] = {}
  writes: dict[str, set[int]] = {}
  for i, (loads, stores, calls) in enumerate(refs):
    for b in loads | calls:
      reads.setdefault(b, set()).add(i)
    for b in stores | calls:
      writes.setdefault(b, set()).add(i)

  sums: dict[str, tuple[int, tuple[str, ...]]] = {}
  pads: dict[str, tuple[int, int]] = {}
  for i, stmt in enumerate(body):
    pr = _as_inline_producer(stmt)
    if pr is None:
      continue
    buf, v, rhs = pr
    if _trip_count(stmt.args[0]) != _size_of(decls[buf].attrs["shape"]):
      continue
    if rhs.op == ProgramOp.ADD and writes[buf] == {i}:
      if all(
        a.op == ProgramOp.LOAD and len(a.args[0].args) == 1 and a.args[0].args[0].op == ProgramOp.VAR and a.args[0].args[0].attrs["name"] == v
        for a in rhs.args
      ):
        sums[buf] = i, tuple(a.args[0].attrs["buffer"] for a in rhs.args)
    if rhs.op != ProgramOp.CONST_FLOAT or rhs.attrs["value"] != 0 or len(writes[buf]) != 2:
      continue
    j = max(writes[buf])
    scatter = body[j]
    if j <= i or scatter.op != ProgramOp.FOR or len(scatter.args) != 2 or scatter.args[1].op != ProgramOp.STORE:
      continue
    target, value = scatter.args[1].args
    if target.attrs["buffer"] != buf or len(target.args) != 1 or value.op != ProgramOp.LOAD:
      continue
    idx = target.args[0]
    if idx.op != ProgramOp.LOAD:
      continue
    table = decls[idx.args[0].attrs["buffer"]]
    indices = table.attrs.get("values", ())
    if not indices or len(set(indices)) != len(indices) or _trip_count(scatter.args[0]) != len(indices):
      continue
    iv = scatter.args[0].attrs["name"]
    if any(len(a.args) != 1 or a.args[0].op != ProgramOp.VAR or a.args[0].attrs["name"] != iv for a in (idx.args[0], value.args[0])):
      continue
    pads[buf] = i, j

  removed: set[int] = set()
  replacements: dict[int, list[ProgramNode]] = {}
  for root, (end, _) in reversed(sums.items()):
    if end in removed:
      continue
    leaves: list[str] = []
    drop: set[int] = set()
    pending = [(root, end)]
    valid = True
    while pending:
      buf, consumer = pending.pop()
      if buf != root and (
        buf not in private or buf in pinned or reads.get(buf) != {consumer} or decls[buf].attrs["shape"] != decls[root].attrs["shape"]
      ):
        valid = False
        break
      if buf in sums:
        i, operands = sums[buf]
        if operands[1] in sums:
          valid = False
          break
        drop.add(i)
        pending.extend((operand, i) for operand in reversed(operands))
      elif buf in pads:
        start, scatter = pads[buf]
        sources = {_resolve_alias(b, aliases) for b in refs[scatter][0]}
        if _resolve_alias(root, aliases) in sources or any(
          sources & {_resolve_alias(b, aliases) for b in stores | calls} for _, stores, calls in refs[scatter + 1 : end]
        ):
          valid = False
          break
        leaves.append(buf)
        drop.update((start, scatter))
      else:
        valid = False
        break
    if not valid or len(leaves) < 2 or drop & removed:
      continue

    def destination(n: ProgramNode) -> ProgramNode:
      if n.op == ProgramOp.VIEW and n.attrs["buffer"] in leaves:
        return ProgramNode(n.op, n.args, {**n.attrs, "buffer": root}, n.dtype)
      return n

    replacement = [_transform(body[pads[leaves[0]][0]], destination)]
    for leaf in leaves:
      scatter = _transform(body[pads[leaf][1]], destination)
      target, value = scatter.args[1].args
      replacement.append(for_(scatter.args[0], [store(target, add(load(target), value))]))
    replacements[end] = replacement
    removed.update(drop)
  if not replacements:
    return proc
  new_body = [s for i, stmt in enumerate(body) for s in (replacements[i] if i in replacements else [] if i in removed else [stmt])]
  return _prune_dead_buffers(_rebuild_proc(proc, params, new_body))


@register_pass("fuse_elementwise")
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


@register_pass("unroll_unit_loops")
def unroll_unit_loops(prog: ProgramNode) -> ProgramNode:
  """Remove static zero-trip loops and inline static one-trip loops.

  Lowering intentionally emits uniform loops even for scalar buffers (shape ``(1,)``). Keeping
  that shape through ``fuse_elementwise`` preserves its producer matcher; after fusion, this pass
  erases the leftover ``for (... < 1)`` noise by substituting the loop variable with its sole value.
  """
  return _map_procs(prog, _unroll_unit_loops_proc)


def _unroll_unit_loops_proc(proc: ProgramNode) -> ProgramNode:
  params, body = _proc_parts(proc)
  new_body: list[ProgramNode] = []
  changed = False
  for stmt in body:
    repl = _unroll_unit_loop_stmt(stmt)
    changed = changed or len(repl) != 1 or repl[0] is not stmt
    new_body.extend(repl)
  return _prune_dead_buffers(_rebuild_proc(proc, params, new_body)) if changed else proc


def _unroll_unit_loop_stmt(stmt: ProgramNode) -> list[ProgramNode]:
  if stmt.op != ProgramOp.FOR:
    return [stmt]

  rng, *body = stmt.args
  new_body: list[ProgramNode] = []
  changed = False
  for sub in body:
    repl = _unroll_unit_loop_stmt(sub)
    changed = changed or len(repl) != 1 or repl[0] is not sub
    new_body.extend(repl)

  trip_count = _trip_count(rng)
  if trip_count == 0:
    return []
  if trip_count == 1:
    vname = rng.attrs["name"]
    only_value = rng.args[0]
    out: list[ProgramNode] = []
    for sub in new_body:
      out.extend(_unroll_unit_loop_stmt(_subst_var(sub, vname, only_value)))
    return out
  if changed:
    return [ProgramNode(ProgramOp.FOR, (rng, *new_body), {**stmt.attrs, "body_len": len(new_body)}, stmt.dtype)] if new_body else []
  return [stmt]


# ---------------------------------------------------------------------------
# Pass 3: workspace lifetime packing + spilling.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _PackPlan:
  rename: dict[str, str] = field(default_factory=dict)  # private buffer name -> slot name
  slot_dtype: dict[str, DType] = field(default_factory=dict)  # slot name -> dtype
  slot_size: dict[str, int] = field(default_factory=dict)  # slot name -> element count
  spill_offset: dict[str, int] = field(default_factory=dict)  # slot name -> offset into w[]
  own_spill: int = 0  # doubles this proc spills to its own w[] window
  callees: set[str] = field(default_factory=set)  # callee names this proc invokes


def _plan_pack(proc: ProgramNode) -> _PackPlan:
  """Lifetime-pack a PROC's private buffers into shared slots and choose which slots spill.

  Greedy left-edge packing over statement order (the lowerer emits values in topological /
  creation order, so first-write order is a valid schedule). Slots are dtype-homogeneous;
  only ``float64`` slots are spill-eligible since ``w[]`` is ``double*``.
  """
  _params, body = _proc_parts(proc)
  private = _private_decls(body)
  alias_src = _alias_sources(body)
  # Aliases own no storage (they're pointers into another buffer); pack only real buffers, but
  # a read of an alias extends the lifetime of the buffer it points at.
  packable = {name: decl for name, decl in private.items() if name not in alias_src}
  plan = _PackPlan()
  if not packable:
    for stmt in body:
      for n in _walk(stmt):
        if n.op == ProgramOp.CALL:
          plan.callees.add(n.attrs["callee"])
    return plan

  def owner(name: str) -> str:
    return _resolve_alias(name, alias_src)

  # Lifetime of each packable buffer over statement positions: [first write, last read-or-write].
  # ``deps[b]`` is the transitive set of packable buffers that produced the current contents of
  # ``b``. At CALL boundaries, keep every input's producer closure live through the CALL. This
  # prevents a CALL output from reusing a slot that contributed to one of its inputs — a pattern
  # that is semantically valid but triggers an Apple-clang inlining miscompile on macOS 15 arm64.
  first_write: dict[str, int] = {}
  last_use: dict[str, int] = {}
  deps: dict[str, set[str]] = {}
  for i, stmt in enumerate(body):
    if stmt.op == ProgramOp.BUFFER:
      continue
    loads, stores, _calls = _stmt_refs(stmt)
    call_inputs = {owner(b) for b in calls_in(stmt)} & packable.keys()
    call_outputs = {owner(b) for b in calls_out(stmt)} & packable.keys()
    writes = {owner(b) for b in stores} & packable.keys() | call_outputs
    reads = ({owner(b) for b in loads} & packable.keys()) | call_inputs
    for b in call_inputs:
      for dep in deps.get(b, set()):
        last_use[dep] = max(last_use.get(dep, i), i)
    for b in writes:
      first_write.setdefault(b, i)
      deps[b] = set(reads)
      for r in reads:
        deps[b].update(deps.get(r, set()))
    for b in writes | reads:
      last_use[b] = i
    for n in _walk(stmt):
      if n.op == ProgramOp.CALL:
        plan.callees.add(n.attrs["callee"])

  # Pack per dtype, in first-write order (ties: declaration order via the dict insertion order).
  order = sorted((name for name in packable if name in first_write), key=lambda b: (first_write[b], b))
  free_at: dict[str, int] = {}  # slot -> first statement index at which it is reusable
  counter = 0
  for buf in order:
    dt = packable[buf].dtype
    size = _size_of(packable[buf].attrs["shape"])
    chosen: str | None = None
    for slot, fa in free_at.items():
      if plan.slot_dtype[slot] == dt and fa <= first_write[buf]:
        chosen = slot
        break
    if chosen is None:
      chosen = f"s{counter}"
      counter += 1
      plan.slot_dtype[chosen] = dt
      plan.slot_size[chosen] = 0
    plan.rename[buf] = chosen
    plan.slot_size[chosen] = max(plan.slot_size[chosen], size)
    free_at[chosen] = last_use[buf] + 1

  # Spill plan: float64 slots at/above the threshold get a sequential window in w[].
  total = 0
  for slot in plan.slot_size:
    if plan.slot_dtype[slot].is_floating and plan.slot_size[slot] >= WORKSPACE_SPILL_THRESHOLD:
      plan.spill_offset[slot] = total
      total += plan.slot_size[slot]
  plan.own_spill = total
  return plan


def calls_out(stmt: ProgramNode) -> set[str]:
  out: set[str] = set()
  for n in _walk(stmt):
    if n.op == ProgramOp.CALL:
      n_in, n_out = int(n.attrs["n_in"]), int(n.attrs["n_out"])
      for a in n.args[n_in : n_in + n_out]:
        name = _call_arg_buffer(a)
        if name is not None:
          out.add(name)
  return out


def calls_in(stmt: ProgramNode) -> set[str]:
  ins: set[str] = set()
  for n in _walk(stmt):
    if n.op == ProgramOp.CALL:
      n_in = int(n.attrs["n_in"])
      for a in n.args[:n_in]:
        name = _call_arg_buffer(a)
        if name is not None:
          ins.add(name)
  return ins


@register_pass("pack_workspace")
def pack_workspace(prog: ProgramNode) -> ProgramNode:
  procs, kernels = _procs(prog)
  plans = {pr.attrs["name"]: _plan_pack(pr) for pr in procs}

  # A solver wrapper (rendered by codegen/solver, so it has no PROC here) is an opaque callee: it owns
  # no spill of its own but passes its ``w`` straight through to its oracle Functions, so its
  # workspace is the max over those oracle PROCs. The lowerer records solver -> oracle names on
  # the PROGRAM; we seed those names into the sz_w recursion below.
  solver_oracles: dict[str, tuple[str, ...]] = prog.attrs.get("solver_oracles", {})
  solver_external_workspace: dict[str, int] = prog.attrs.get("solver_external_workspace", {})

  # sz_w(proc) = own spill + max callee workspace (callees share the post-own-spill window).
  sz_w: dict[str, int] = {}

  def total(name: str) -> int:
    if name in sz_w:
      return sz_w[name]
    if name in solver_oracles:
      sz_w[name] = 0  # break cycles defensively; a solver owns no spill itself
      sz_w[name] = max((solver_external_workspace.get(name, 0), *(total(o) for o in solver_oracles[name] if o in plans)))
      return sz_w[name]
    sz_w[name] = plans[name].own_spill  # break cycles defensively
    callee_max = max((total(c) for c in plans[name].callees if c in plans or c in solver_oracles), default=0)
    sz_w[name] = plans[name].own_spill + callee_max
    return sz_w[name]

  for name in plans:
    total(name)

  procs = [_apply_pack(pr, plans[pr.attrs["name"]], sz_w) for pr in procs]
  return ProgramNode(ProgramOp.PROGRAM, (*procs, *kernels), prog.attrs, prog.dtype)


def _apply_pack(proc: ProgramNode, plan: _PackPlan, sz_w: dict[str, int]) -> ProgramNode:
  params, body = _proc_parts(proc)
  name = proc.attrs["name"]
  if not plan.rename and not plan.callees:
    return _rebuild_proc(proc, params, body, sz_w=sz_w.get(name, 0), w_self=plan.own_spill)

  # Slot BUFFER nodes (one decl per slot, sized to the slot max, tagged with a spill offset).
  slot_bufs: dict[str, ProgramNode] = {}
  for slot, size in plan.slot_size.items():
    attrs: dict[str, object] = {
      "name": slot,
      "shape": (size,),
      "address_space": "private",
      "device": DeviceSpec.parse(None),
    }
    if slot in plan.spill_offset:
      attrs["workspace_offset"] = plan.spill_offset[slot]
    slot_bufs[slot] = ProgramNode(ProgramOp.BUFFER, (), attrs, plan.slot_dtype[slot])

  def fn(n: ProgramNode) -> ProgramNode:
    if n.op == ProgramOp.BUFFER and "alias_of" in n.attrs and n.attrs["alias_of"] in plan.rename:
      return ProgramNode(ProgramOp.BUFFER, n.args, {**n.attrs, "alias_of": plan.rename[n.attrs["alias_of"]]}, n.dtype)
    if n.op == ProgramOp.BUFFER and n.attrs.get("name") in plan.rename:
      return slot_bufs[plan.rename[n.attrs["name"]]]
    if n.op == ProgramOp.VIEW and n.attrs.get("buffer") in plan.rename:
      return ProgramNode(ProgramOp.VIEW, n.args, {**n.attrs, "buffer": plan.rename[n.attrs["buffer"]]}, n.dtype)
    if n.op == ProgramOp.CALL:
      callee = n.attrs["callee"]
      return ProgramNode(ProgramOp.CALL, n.args, {**n.attrs, "w_self": plan.own_spill, "callee_needs_w": sz_w.get(callee, 0) > 0}, n.dtype)
    return n

  new_body: list[ProgramNode] = []
  seen_slot: set[str] = set()
  for stmt in body:
    rewritten = _transform(stmt, fn)
    if rewritten.op == ProgramOp.BUFFER:
      slot_name = rewritten.attrs["name"]
      if slot_name in seen_slot:
        continue  # one decl per slot (members collapse to the same node)
      seen_slot.add(slot_name)
    new_body.append(rewritten)

  return _rebuild_proc(proc, params, new_body, sz_w=sz_w.get(name, 0), w_self=plan.own_spill)


__all__ = [
  "PASS_PIPELINE",
  "ProgramObserver",
  "WORKSPACE_SPILL_THRESHOLD",
  "combine_scatter_sums",
  "fuse_elementwise",
  "optimize_program",
  "pack_workspace",
  "register_pass",
  "unroll_unit_loops",
]
