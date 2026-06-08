"""Program IR optimization passes: ``PROGRAM -> PROGRAM`` rewrites run after lowering.

The pipeline (``optimize_program``) is invoked at the tail of ``lowering.lower_function``,
between the naive lowering and the verifier. Each pass is a self-contained transform over
the hash-consed ``PNode`` graph registered in ``PASS_PIPELINE`` — adding an optimization is
a new entry here, mirroring the ``@lowers`` registry in ``lowering.py`` and the per-op maps
in ``codegen/program_c.py``. This is the *third* leg of the architecture: lower (semantic ->
Program IR), optimize (Program IR -> Program IR), render (Program IR -> C). New optimizations
(CSE, peepholes, GPU schedule passes) slot in as additional passes without touching the
lowerer or renderer.

Ported optimizations (the legacy tape renderer baked these into ``codegen/c.py``; here they
become explicit, individually-testable Program IR passes):

- ``fuse_elementwise`` — inline single-use ``private`` elementwise / slice / gather producers
  (a single-``STORE`` ``FOR`` whose store index is the loop var) into their one consumer by
  substituting the producer's scalar RHS at the consumer's load site, then dropping the
  producer loop + buffer. This is loop fusion: chains collapse into one loop and the
  intermediate buffer round-trips vanish. Port of ``c.py``'s ``_inline_scalar_table`` /
  ``_skipped_instructions`` (no-materialize + inline-read tables).

- ``pack_workspace`` — lifetime-pack ``private`` BUFFERs into shared slots and spill slots
  ≥ ``WORKSPACE_SPILL_THRESHOLD`` doubles to the caller-provided ``w[]`` (so the function
  declares a real ``sz_w`` instead of stack-allocating every temporary). Port of
  ``c.py``'s ``_compute_lifetimes`` / ``_pack_slots`` / ``_spill_plan``. Without this the
  largest benchmark cells overflow the 8 MB stack — it gates the merge.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from .program import PNode, POps
from .types import DType, DeviceSpec

# Slots this large (in elements) or bigger move off the C stack into the caller's ``w[]``.
# Matches ``codegen/c.py::_WORKSPACE_SPILL_THRESHOLD`` so the two renderers report the same sz_w.
WORKSPACE_SPILL_THRESHOLD = 1024

# Scalar POps that lower to a libm call. Inlining one into a consumer whose iteration domain
# is larger than the producer's (a broadcast) — or at more than one site — replays the call,
# so the fusion pass keeps these materialized unless the read is one-to-one. Cheap arithmetic
# is always safe to duplicate. Mirrors ``c.py::_EXPENSIVE_UNARY`` (extended to libm binaries).
_EXPENSIVE_OPS: frozenset[POps] = frozenset(
  {
    POps.SIN,
    POps.COS,
    POps.TAN,
    POps.ASIN,
    POps.ACOS,
    POps.ATAN,
    POps.SINH,
    POps.COSH,
    POps.TANH,
    POps.EXP,
    POps.LOG,
    POps.SQRT,
    POps.POW,
    POps.ATAN2,
  }
)


# ---------------------------------------------------------------------------
# Pass pipeline.
# ---------------------------------------------------------------------------

PassFn = Callable[[PNode], PNode]
PASS_PIPELINE: list[tuple[str, PassFn]] = []


def register_pass(name: str) -> Callable[[PassFn], PassFn]:
  """Append ``fn`` to the optimization pipeline under ``name`` (order = registration order)."""

  def deco(fn: PassFn) -> PassFn:
    PASS_PIPELINE.append((name, fn))
    return fn

  return deco


def optimize_program(prog: PNode) -> PNode:
  """Run every registered pass over a lowered ``PROGRAM`` in order, returning the optimized one."""
  for _name, fn in PASS_PIPELINE:
    prog = fn(prog)
  return prog


# ---------------------------------------------------------------------------
# Shared Program IR plumbing.
# ---------------------------------------------------------------------------


def _walk(root: PNode) -> Iterable[PNode]:
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


def _transform(node: PNode, fn: Callable[[PNode], PNode]) -> PNode:
  """Bottom-up rebuild: rebuild ``node``'s args, then apply ``fn`` to the (maybe) new node.

  ``fn`` returns a replacement node (or its argument unchanged). Hash-consing makes the
  identity check cheap and keeps untouched subtrees shared.
  """
  if node.args:
    new_args = tuple(_transform(a, fn) for a in node.args)
    if any(a is not b for a, b in zip(new_args, node.args, strict=True)):
      node = PNode(node.op, new_args, node.attrs, node.dtype)
  return fn(node)


def _procs(prog: PNode) -> tuple[list[PNode], list[PNode]]:
  """Split a PROGRAM into (procs, kernels)."""
  pc = int(prog.attrs.get("proc_count", 0))
  return list(prog.args[:pc]), list(prog.args[pc:])


def _proc_parts(proc: PNode) -> tuple[list[PNode], list[PNode]]:
  """Split a PROC into (params, body statements)."""
  pc = int(proc.attrs["param_count"])
  return list(proc.args[:pc]), list(proc.args[pc:])


def _rebuild_proc(proc: PNode, params: list[PNode], body: list[PNode], **extra_attrs: object) -> PNode:
  attrs = {**proc.attrs, "param_count": len(params), **extra_attrs}
  return PNode(POps.PROC, (*params, *body), attrs, proc.dtype)


def _map_procs(prog: PNode, fn: Callable[[PNode], PNode]) -> PNode:
  """Apply a per-PROC transform to every proc, preserving kernels and proc/kernel counts."""
  procs, kernels = _procs(prog)
  procs = [fn(pr) for pr in procs]
  return PNode(POps.PROGRAM, (*procs, *kernels), prog.attrs, prog.dtype)


def _size_of(shape: tuple[int, ...]) -> int:
  n = 1
  for d in shape:
    n *= int(d)
  return n or 1


def _call_arg_buffer(arg: PNode) -> str | None:
  if arg.op == POps.BUFFER:
    return arg.attrs["name"]
  if arg.op == POps.VIEW:
    return arg.attrs["buffer"]
  return None


def _stmt_refs(stmt: PNode) -> tuple[set[str], set[str], set[str]]:
  """``(loads, stores, call_args)``: buffer names this statement LOADs, STOREs to, or passes
  (in or out) to a CALL. CALL args are pointer passes, not loads/stores — tracked separately so
  fusion knows a buffer feeding a CALL must stay materialized."""
  loads: set[str] = set()
  stores: set[str] = set()
  call_args: set[str] = set()
  for n in _walk(stmt):
    if n.op == POps.LOAD:
      loads.add(n.args[0].attrs["buffer"])
    elif n.op == POps.STORE:
      stores.add(n.args[0].attrs["buffer"])
    elif n.op == POps.CALL:
      for a in n.args:
        name = _call_arg_buffer(a)
        if name is not None:
          call_args.add(name)
  return loads, stores, call_args


def _private_decls(body: list[PNode]) -> dict[str, PNode]:
  """name -> BUFFER decl for every ``private`` buffer declared in ``body`` (skips constants)."""
  return {s.attrs["name"]: s for s in body if s.op == POps.BUFFER and s.attrs.get("address_space") == "private"}


def _alias_sources(body: list[PNode]) -> dict[str, str]:
  """name -> aliased-source name for every alias BUFFER (zero-copy pointer) in ``body``."""
  return {s.attrs["name"]: s.attrs["alias_of"] for s in body if s.op == POps.BUFFER and "alias_of" in s.attrs}


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


def _as_inline_producer(stmt: PNode) -> tuple[str, str, PNode] | None:
  """If ``stmt`` is ``for v in [0,N) step 1 { buf[v] = rhs }`` return ``(buf, v, rhs)``, else None.

  This is the shape every elementwise / SLICE / GATHER / copy rule emits: a single contiguous
  loop with one STORE indexed by the loop variable. Because the store index *is* the loop var,
  ``buf[k] == rhs[v:=k]`` for all ``k`` the loop visits, so a consumer reading ``buf[E]`` can
  inline ``rhs[v:=E]`` for any in-range ``E``. Reductions (extra init STORE), MATMUL/TRANSPOSE
  (composite store index), and STACK/CONCAT (multiple loops per buffer) deliberately don't match.
  """
  if stmt.op != POps.FOR:
    return None
  rng, *body = stmt.args
  if len(body) != 1 or body[0].op != POps.STORE:
    return None
  store = body[0]
  target = store.args[0]
  if target.op != POps.VIEW or len(target.args) != 1:
    return None
  v = rng.attrs["name"]
  idx = target.args[0]
  if idx.op != POps.VAR or idx.attrs["name"] != v:
    return None
  start, _stop, step = rng.args
  if not (start.op == POps.CONST_INT and start.attrs["value"] == 0):
    return None
  if not (step.op == POps.CONST_INT and step.attrs["value"] == 1):
    return None
  return target.attrs["buffer"], v, store.args[1]


def _subst_var(node: PNode, vname: str, repl: PNode) -> PNode:
  """Substitute every ``VAR(vname)`` in ``node`` with ``repl`` (the consumer's index expr)."""

  def fn(n: PNode) -> PNode:
    if n.op == POps.VAR and n.attrs["name"] == vname:
      return repl
    return n

  return _transform(node, fn)


def _has_expensive(node: PNode) -> bool:
  return any(n.op in _EXPENSIVE_OPS for n in _walk(node))


def _trip_count(rng: PNode) -> int | None:
  """Static iteration count of a ``[0, stop) step 1`` RANGE, or None if not statically known."""
  start, stop, step = rng.args
  if start.op != POps.CONST_INT or stop.op != POps.CONST_INT or step.op != POps.CONST_INT:
    return None
  span = int(stop.attrs["value"]) - int(start.attrs["value"])
  step_v = int(step.attrs["value"])
  return max(0, -(-span // step_v)) if step_v > 0 else None


def _max_load_executions(node: PNode, buf: str, factor: int) -> int | None:
  """Max number of times any single load of ``buf`` runs, given ``factor`` enclosing iterations.

  This is the recompute metric: a load nested in loops of total trip count ``T`` evaluates the
  inlined producer ``T`` times. A consumer whose ``T`` exceeds the producer's element count
  re-reads elements — fine for the elementwise/reduction case (``T == producer_size``), fatal
  for a matmul/contraction (``T == m*n*k ≫ producer_size``). Returns None if any enclosing loop
  has a non-static bound (treat as unbounded — don't inline)."""
  if node.op == POps.FOR:
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
  best = factor if node.op == POps.LOAD and node.args[0].attrs["buffer"] == buf else 0
  for sub in node.args:
    r = _max_load_executions(sub, buf, factor)
    if r is None:
      return None
    best = max(best, r)
  return best


def _count_buf_loads(node: PNode, buf: str) -> int:
  """Textual occurrences of a load of ``buf`` in ``node`` (tree multiplicity: ``buf*buf`` is 2)."""
  memo: dict[int, int] = {}

  def cnt(n: PNode) -> int:
    if id(n) in memo:
      return memo[id(n)]
    base = 1 if n.op == POps.LOAD and n.args[0].attrs["buffer"] == buf else 0
    memo[id(n)] = base + sum(cnt(a) for a in n.args)
    return memo[id(n)]

  return cnt(node)


def _expand_inlinables(node: PNode, inlinable: dict[str, tuple[str, PNode]]) -> PNode:
  """Replace each ``LOAD(VIEW(buf,[E]))`` with the producer RHS at ``E``, recursively, so a whole
  chain of single-use producers collapses into one fused expression in a single bottom-up pass."""

  def fn(n: PNode) -> PNode:
    if n.op == POps.LOAD:
      view = n.args[0]
      buf = view.attrs["buffer"]
      if buf in inlinable and len(view.args) == 1:
        v, rhs = inlinable[buf]
        return _expand_inlinables(_subst_var(rhs, v, view.args[0]), inlinable)
    return n

  return _transform(node, fn)


@register_pass("fuse_elementwise")
def fuse_elementwise(prog: PNode) -> PNode:
  return _map_procs(prog, _fuse_proc)


def _fuse_proc(proc: PNode) -> PNode:
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
    if stmt.op == POps.BUFFER:
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
  inlinable: dict[str, tuple[str, PNode]] = {}
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
  new_body: list[PNode] = []
  for i, stmt in enumerate(body):
    if i in drop:
      continue
    if stmt.op == POps.BUFFER and stmt.attrs.get("name") in inlinable:
      continue  # decl of an inlined buffer
    new_body.append(_expand_inlinables(stmt, inlinable))

  return _prune_dead_buffers(_rebuild_proc(proc, params, new_body))


def _prune_dead_buffers(proc: PNode) -> PNode:
  """Drop BUFFER decls (private or constant) no longer referenced by any VIEW or CALL arg."""
  params, body = _proc_parts(proc)
  referenced: set[str] = set()
  for stmt in body:
    if stmt.op == POps.BUFFER:
      continue
    for n in _walk(stmt):
      if n.op == POps.VIEW:
        referenced.add(n.attrs["buffer"])
      elif n.op == POps.BUFFER:
        referenced.add(n.attrs["name"])
      elif n.op == POps.CALL:
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
  kept = [s for s in body if s.op != POps.BUFFER or s.attrs["name"] in referenced or s.attrs["name"] in param_names]
  if len(kept) == len(body):
    return proc
  return _rebuild_proc(proc, params, kept)


# ---------------------------------------------------------------------------
# Pass 2: workspace lifetime packing + spilling.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _PackPlan:
  rename: dict[str, str] = field(default_factory=dict)  # private buffer name -> slot name
  slot_dtype: dict[str, DType] = field(default_factory=dict)  # slot name -> dtype
  slot_size: dict[str, int] = field(default_factory=dict)  # slot name -> element count
  spill_offset: dict[str, int] = field(default_factory=dict)  # slot name -> offset into w[]
  own_spill: int = 0  # doubles this proc spills to its own w[] window
  callees: set[str] = field(default_factory=set)  # callee names this proc invokes


def _plan_pack(proc: PNode) -> _PackPlan:
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
        if n.op == POps.CALL:
          plan.callees.add(n.attrs["callee"])
    return plan

  def owner(name: str) -> str:
    return _resolve_alias(name, alias_src)

  # Lifetime of each packable buffer over statement positions: [first write, last read-or-write].
  first_write: dict[str, int] = {}
  last_use: dict[str, int] = {}
  for i, stmt in enumerate(body):
    if stmt.op == POps.BUFFER:
      continue
    loads, stores, calls = _stmt_refs(stmt)
    writes = {owner(b) for b in stores | calls_out(stmt)} & packable.keys()
    reads = {owner(b) for b in loads | calls_in(stmt)} & packable.keys()
    for b in writes:
      first_write.setdefault(b, i)
    for b in writes | reads:
      last_use[b] = i
    for n in _walk(stmt):
      if n.op == POps.CALL:
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


def calls_out(stmt: PNode) -> set[str]:
  out: set[str] = set()
  for n in _walk(stmt):
    if n.op == POps.CALL:
      n_in, n_out = int(n.attrs["n_in"]), int(n.attrs["n_out"])
      for a in n.args[n_in : n_in + n_out]:
        name = _call_arg_buffer(a)
        if name is not None:
          out.add(name)
  return out


def calls_in(stmt: PNode) -> set[str]:
  ins: set[str] = set()
  for n in _walk(stmt):
    if n.op == POps.CALL:
      n_in = int(n.attrs["n_in"])
      for a in n.args[:n_in]:
        name = _call_arg_buffer(a)
        if name is not None:
          ins.add(name)
  return ins


@register_pass("pack_workspace")
def pack_workspace(prog: PNode) -> PNode:
  procs, kernels = _procs(prog)
  plans = {pr.attrs["name"]: _plan_pack(pr) for pr in procs}

  # sz_w(proc) = own spill + max callee workspace (callees share the post-own-spill window).
  sz_w: dict[str, int] = {}

  def total(name: str) -> int:
    if name in sz_w:
      return sz_w[name]
    sz_w[name] = plans[name].own_spill  # break cycles defensively
    callee_max = max((total(c) for c in plans[name].callees if c in plans), default=0)
    sz_w[name] = plans[name].own_spill + callee_max
    return sz_w[name]

  for name in plans:
    total(name)

  procs = [_apply_pack(pr, plans[pr.attrs["name"]], sz_w) for pr in procs]
  return PNode(POps.PROGRAM, (*procs, *kernels), prog.attrs, prog.dtype)


def _apply_pack(proc: PNode, plan: _PackPlan, sz_w: dict[str, int]) -> PNode:
  params, body = _proc_parts(proc)
  name = proc.attrs["name"]
  if not plan.rename and not plan.callees:
    return _rebuild_proc(proc, params, body, sz_w=sz_w.get(name, 0), w_self=plan.own_spill)

  # Slot BUFFER nodes (one decl per slot, sized to the slot max, tagged with a spill offset).
  slot_bufs: dict[str, PNode] = {}
  for slot, size in plan.slot_size.items():
    attrs: dict[str, object] = {
      "name": slot,
      "shape": (size,),
      "address_space": "private",
      "device": DeviceSpec.parse(None),
    }
    if slot in plan.spill_offset:
      attrs["workspace_offset"] = plan.spill_offset[slot]
    slot_bufs[slot] = PNode(POps.BUFFER, (), attrs, plan.slot_dtype[slot])

  def fn(n: PNode) -> PNode:
    if n.op == POps.BUFFER and "alias_of" in n.attrs and n.attrs["alias_of"] in plan.rename:
      return PNode(POps.BUFFER, n.args, {**n.attrs, "alias_of": plan.rename[n.attrs["alias_of"]]}, n.dtype)
    if n.op == POps.BUFFER and n.attrs.get("name") in plan.rename:
      return slot_bufs[plan.rename[n.attrs["name"]]]
    if n.op == POps.VIEW and n.attrs.get("buffer") in plan.rename:
      return PNode(POps.VIEW, n.args, {**n.attrs, "buffer": plan.rename[n.attrs["buffer"]]}, n.dtype)
    if n.op == POps.CALL:
      callee = n.attrs["callee"]
      return PNode(POps.CALL, n.args, {**n.attrs, "w_self": plan.own_spill, "callee_needs_w": sz_w.get(callee, 0) > 0}, n.dtype)
    return n

  new_body: list[PNode] = []
  seen_slot: set[str] = set()
  for stmt in body:
    rewritten = _transform(stmt, fn)
    if rewritten.op == POps.BUFFER:
      slot_name = rewritten.attrs["name"]
      if slot_name in seen_slot:
        continue  # one decl per slot (members collapse to the same node)
      seen_slot.add(slot_name)
    new_body.append(rewritten)

  return _rebuild_proc(proc, params, new_body, sz_w=sz_w.get(name, 0), w_self=plan.own_spill)


__all__ = [
  "PASS_PIPELINE",
  "WORKSPACE_SPILL_THRESHOLD",
  "fuse_elementwise",
  "optimize_program",
  "pack_workspace",
  "register_pass",
]
