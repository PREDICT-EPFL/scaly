"""Lifetime-based private-buffer workspace packing for the Program IR."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter

from ...ir.match import Pattern, rewrite
from ...ir.program import ProgramNode, ProgramOp, walk_program
from ...ir.types import DType, DeviceSpec, dtypes
from ._common import (
  _alias_sources,
  buffer_refs,
  _private_decls,
  _proc_parts,
  _rebuild_proc,
  _resolve_alias,
  _size_of,
  rebuild_program,
)

SCALAR_SPILL_ALLOWANCE = 4096
WORKSPACE_ALIGNMENT = 64  # bytes between the start of the workspace and every spilled slot


@dataclass(slots=True)
class _PackPlan:
  rename: dict[str, str] = field(default_factory=dict)  # private buffer name -> slot name
  slot_dtype: dict[str, DType] = field(default_factory=dict)  # slot name -> dtype
  slot_size: dict[str, int] = field(default_factory=dict)  # slot name -> element count
  spill_offset: dict[str, int] = field(default_factory=dict)  # slot name -> offset into w[]
  own_spill: int = 0  # workspace units this proc spills to its own w[] window
  callees: list[str] = field(default_factory=list)  # one entry per call site


def _plan_pack(proc: ProgramNode) -> _PackPlan:
  """Lifetime-pack a PROC's private buffers into shared slots.

  Greedy left-edge packing over statement order (the lowerer emits values in topological /
  creation order, so first-write order is a valid schedule). Slots are dtype-homogeneous.
  """
  _params, body = _proc_parts(proc)
  private = _private_decls(body)
  alias_src = _alias_sources(body)
  # Aliases own no storage (they're pointers into another buffer); pack only real buffers, but
  # a read of an alias extends the lifetime of the buffer it points at.
  packable = {name: decl for name, decl in private.items() if name not in alias_src}
  plan = _PackPlan()
  pending = list(body)
  while pending:
    stmt = pending.pop()
    if stmt.op == ProgramOp.CALL:
      plan.callees.append(stmt.attrs["callee"])
    elif stmt.op in (ProgramOp.FOR, ProgramOp.BLOCK):
      pending.extend(stmt.args[1:] if stmt.op == ProgramOp.FOR else stmt.args)
  if not packable:
    return plan

  def owner(name: str) -> str:
    return _resolve_alias(name, alias_src)

  # Lifetime of each packable buffer over statement positions: [first write, last read-or-write].
  # ``deps[b]`` is the transitive set of packable buffers that produced the current contents of
  # ``b``. At CALL boundaries, keep every input's producer closure live through the CALL. This
  # prevents a CALL output from reusing a slot that contributed to one of its inputs — a pattern
  # that is semantically valid but triggers an Apple-clang inlining miscompile on macOS 15 arm64.
  first_write: dict[str, int] = {}
  first_read: dict[str, int] = {}
  last_use: dict[str, int] = {}
  deps: dict[str, set[str]] = {}
  statements: list[ProgramNode] = []
  scopes: list[tuple[int, int]] = []

  def append(stmt: ProgramNode) -> None:
    mapped = (
      stmt.op == ProgramOp.FOR and not stmt.args[0].attrs.get("carried") and (stmt.args[0].attrs.get("mapped") or stmt.attrs.get("vector_mapped"))
    )
    if mapped:
      start = len(statements)
      statements.append(stmt.args[0])
      children = stmt.args[1].args[1:] if "vector_helper" in stmt.attrs else stmt.args[1:]
      for child in children:
        append(child)
      scopes.append((start, len(statements)))
    else:
      statements.append(stmt)

  for stmt in body:
    append(stmt)
  references: list[set[str]] = []
  for i, stmt in enumerate(statements):
    references.append(set())
    if stmt.op == ProgramOp.BUFFER:
      continue
    refs = buffer_refs(stmt)
    loads, stores = refs.loads, refs.stores
    call_inputs = {owner(b) for b in refs.call_inputs} & packable.keys()
    call_outputs = {owner(b) for b in refs.call_outputs} & packable.keys()
    writes = {owner(b) for b in stores} & packable.keys() | call_outputs
    reads = ({owner(b) for b in loads} & packable.keys()) | call_inputs
    references[i] = writes | reads
    for b in reads:
      first_read.setdefault(b, i)
    for b in call_inputs:
      for dep in deps.get(b, set()):
        references[i].add(dep)
        last_use[dep] = max(last_use.get(dep, i), i)
    produced_from = reads.union(*(deps.get(r, ()) for r in reads))
    for b in writes:
      first_write.setdefault(b, i)
      deps[b] = produced_from
    for b in writes | reads:
      last_use[b] = i

  # A lowered temporary is defined anew at every call. Its kernels inside an inlined map may
  # share storage within one iteration; inputs and values used after the map stay live across all trips.
  for start, stop in scopes:
    for b in set().union(*references[start:stop]):
      if b not in first_write:
        continue
      if not packable[b].attrs.get("per_call") or first_write[b] < start or first_read.get(b, stop) <= first_write[b] or last_use[b] >= stop:
        first_write[b] = min(first_write.get(b, start), start)
        last_use[b] = max(last_use[b], stop - 1)

  # Pack per dtype, in first-write order (ties: declaration order via the dict insertion order).
  order = sorted((name for name in packable if name in first_write), key=lambda b: first_write[b])
  free_at: dict[str, int] = {}  # slot -> first statement index at which it is reusable
  used_names = {n.attrs["name"] for n in walk_program(proc) if n.op == ProgramOp.BUFFER}
  counter = 0
  for buf in order:
    dt = packable[buf].dtype
    size = _size_of(packable[buf].attrs["shape"])
    chosen: str | None = None
    for slot, fa in free_at.items():
      # A scalar temporary must not pin an array slot until its consumer runs.
      if plan.slot_dtype[slot] == dt and fa <= first_write[buf] and not (size == 1 and 1 < plan.slot_size[slot]):
        chosen = slot
        break
    if chosen is None:
      while (chosen := f"s{counter}") in used_names:
        counter += 1
      used_names.add(chosen)
      counter += 1
      plan.slot_dtype[chosen] = dt
      plan.slot_size[chosen] = 0
    plan.rename[buf] = chosen
    plan.slot_size[chosen] = max(plan.slot_size[chosen], size)
    free_at[chosen] = last_use[buf] + 1

  return plan


def _slot_bytes(plan: _PackPlan, slot: str) -> int:
  return plan.slot_size[slot] * plan.slot_dtype[slot].itemsize


def _aligned(units: int, unit: DType) -> int:
  step = WORKSPACE_ALIGNMENT // unit.itemsize
  return -(-units // step) * step


def pack_workspace(prog: ProgramNode, *, frame_budget: int = 65536) -> ProgramNode:
  """Pack private storage and spill the largest slots until each estimated frame fits.

  The workspace has the program's ``workspace_dtype``: a float64 or float32 workspace holds only
  slots of its dtype, and a uint8 one, addressed in bytes, holds every dtype. Other locals still
  count toward the estimate, at their real size. Each spilled slot starts at a multiple of
  WORKSPACE_ALIGNMENT bytes. Zero spills every eligible slot. Scalar spills still need stack space,
  covered by a fixed allowance.
  """
  procs = prog.args
  unit: DType = prog.attrs.get("workspace_dtype", dtypes.float64)
  plans = {pr.attrs["name"]: _plan_pack(pr) for pr in procs}
  procedures = {pr.attrs["name"]: pr for pr in procs}
  inline_slots: dict[str, Counter[tuple[str, str]]] = {}

  def slots(name: str) -> Counter[tuple[str, str]]:
    if name not in inline_slots:
      counts = Counter({(name, slot): 1 for slot in plans[name].slot_size})
      for callee in plans[name].callees:
        if callee in plans and not procedures[callee].attrs.get("noinline"):
          counts.update(slots(callee))
      inline_slots[name] = counts
    return inline_slots[name]

  def estimate(name: str) -> int:
    return SCALAR_SPILL_ALLOWANCE + sum(
      count * _slot_bytes(plans[owner], slot) for (owner, slot), count in slots(name).items() if slot not in plans[owner].spill_offset
    )

  for name in plans:
    frame = estimate(name)
    for owner, slot in sorted(slots(name), key=lambda key: _slot_bytes(plans[key[0]], key[1]), reverse=True):
      if frame_budget and frame <= frame_budget:
        break
      plan = plans[owner]
      if unit in (plan.slot_dtype[slot], dtypes.uint8) and slot not in plan.spill_offset:
        plan.spill_offset[slot] = plan.own_spill
        plan.own_spill = _aligned(plan.own_spill + _slot_bytes(plan, slot) // unit.itemsize, unit)
        frame -= slots(name)[owner, slot] * _slot_bytes(plan, slot)

  # sz_w(proc) = own spill + max callee workspace: callees share the window after the own spill.
  # A byte workspace instead lays its callees' windows end to end, so every byte of it holds one
  # dtype for the whole call and no access changes the type of memory another one reads.
  # An external function's workspace is counted in doubles.
  sz_w: dict[str, int] = {fn.name: fn.workspace * dtypes.float64.itemsize // unit.itemsize for fn in prog.attrs.get("externs", ())}
  windows: dict[str, dict[str, int]] = {}

  def total(name: str) -> int:
    if name in sz_w:
      return sz_w[name]
    sz_w[name] = plans[name].own_spill  # break cycles defensively
    callees = sorted({c for c in plans[name].callees if c in plans or c in sz_w})
    if unit != dtypes.uint8:
      sz_w[name] = plans[name].own_spill + max((total(c) for c in callees), default=0)
      return sz_w[name]
    offset = plans[name].own_spill
    windows[name] = {}
    for callee in callees:
      windows[name][callee] = offset
      offset = _aligned(offset + total(callee), unit)
    sz_w[name] = offset
    return sz_w[name]

  for name in plans:
    total(name)

  procs = [_apply_pack(pr, plans[pr.attrs["name"]], sz_w, estimate(pr.attrs["name"]), windows.get(pr.attrs["name"], {})) for pr in procs]
  return ProgramNode(ProgramOp.PROGRAM, tuple(procs), prog.attrs, dtype=prog.dtype)


def _apply_pack(proc: ProgramNode, plan: _PackPlan, sz_w: dict[str, int], frame_bytes: int, windows: dict[str, int]) -> ProgramNode:
  params, body = _proc_parts(proc)
  name = proc.attrs["name"]
  if not plan.rename and not plan.callees:
    return _rebuild_proc(proc, params, body, sz_w=sz_w.get(name, 0), w_self=plan.own_spill, frame_bytes=frame_bytes)

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
    slot_bufs[slot] = ProgramNode(ProgramOp.BUFFER, (), attrs, dtype=plan.slot_dtype[slot])

  def fn(n: ProgramNode) -> ProgramNode:
    if n.op == ProgramOp.BUFFER and "alias_of" in n.attrs and n.attrs["alias_of"] in plan.rename:
      return ProgramNode(ProgramOp.BUFFER, n.args, {**n.attrs, "alias_of": plan.rename[n.attrs["alias_of"]]}, dtype=n.dtype)
    if n.op == ProgramOp.BUFFER and n.attrs.get("name") in plan.rename:
      return slot_bufs[plan.rename[n.attrs["name"]]]
    if n.op == ProgramOp.VIEW and n.attrs.get("buffer") in plan.rename:
      return ProgramNode(ProgramOp.VIEW, n.args, {**n.attrs, "buffer": plan.rename[n.attrs["buffer"]]}, dtype=n.dtype)
    if n.op == ProgramOp.CALL:
      callee = n.attrs["callee"]
      w_self = windows.get(callee, plan.own_spill)
      return ProgramNode(ProgramOp.CALL, n.args, {**n.attrs, "w_self": w_self, "callee_needs_w": sz_w.get(callee, 0) > 0}, dtype=n.dtype)
    return n

  new_body: list[ProgramNode] = []
  seen_slot: set[str] = set()
  for stmt in body:
    rewritten = rewrite(stmt, [Pattern(None, lambda n: True, fn)], rebuild=rebuild_program, fixpoint=False)
    if rewritten.op == ProgramOp.BUFFER:
      slot_name = rewritten.attrs["name"]
      if slot_name in seen_slot:
        continue  # one decl per slot (members collapse to the same node)
      seen_slot.add(slot_name)
    new_body.append(rewritten)

  return _rebuild_proc(proc, params, new_body, sz_w=sz_w.get(name, 0), w_self=plan.own_spill, frame_bytes=frame_bytes)
