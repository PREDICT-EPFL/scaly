"""Lifetime-based private-buffer workspace packing for the Program IR."""

from __future__ import annotations

import heapq

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field

from ...ir.match import Pattern, rewrite
from ...ir.program import ProgramNode, ProgramOp
from ...ir.types import DType, DeviceSpec
from ._common import (
  _alias_sources,
  buffer_refs,
  _private_decls,
  _proc_parts,
  _procs,
  _rebuild_proc,
  _resolve_alias,
  _size_of,
  _walk,
  rebuild_program,
)

WORKSPACE_SPILL_THRESHOLD = 1024


@dataclass(slots=True)
class _PackPlan:
  rename: dict[str, str] = field(default_factory=dict)  # private buffer name -> slot name
  slot_dtype: dict[str, DType] = field(default_factory=dict)  # slot name -> dtype
  slot_size: dict[str, int] = field(default_factory=dict)  # slot name -> element count
  spill_offset: dict[str, int] = field(default_factory=dict)  # slot name -> offset into w[]
  slot_lanes: dict[str, int] = field(default_factory=dict)  # slot name -> lanes, for a slot of vectors' lanes
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
  # The closures only matter at a CALL; without one (straight-line code, typically) they would
  # cost time quadratic in the length of a dependency chain for nothing.
  track_deps = any(n.op == ProgramOp.CALL for stmt in body if stmt.op != ProgramOp.BUFFER for n in _walk(stmt))
  for i, stmt in enumerate(body):
    if stmt.op == ProgramOp.BUFFER:
      continue
    refs = buffer_refs(stmt)
    loads, stores = refs.loads, refs.stores
    call_inputs = {owner(b) for b in refs.call_inputs} & packable.keys()
    call_outputs = {owner(b) for b in refs.call_outputs} & packable.keys()
    writes = {owner(b) for b in stores} & packable.keys() | call_outputs
    reads = ({owner(b) for b in loads} & packable.keys()) | call_inputs
    for b in call_inputs:
      for dep in deps.get(b, set()):
        last_use[dep] = max(last_use.get(dep, i), i)
    # Every buffer written here now depends on everything that produced what this statement read.
    # Take the closure from the dependencies *before* the statement: updating ``deps`` while
    # iterating ``writes`` made the result depend on set order, so the generated C varied with
    # ``PYTHONHASHSEED`` whenever a statement wrote one buffer and read another it also wrote.
    if track_deps:
      produced_from = set(reads)
      for r in reads:
        produced_from.update(deps.get(r, set()))
      for b in writes:
        deps[b] = set(produced_from)
    for b in writes:
      first_write.setdefault(b, i)
    for b in writes | reads:
      last_use[b] = i
    for n in _walk(stmt):
      if n.op == ProgramOp.CALL:
        plan.callees.add(n.attrs["callee"])

  # Pack per dtype, in first-write order (ties: declaration order via the dict insertion order).
  order = sorted((name for name in packable if name in first_write), key=lambda b: (first_write[b], b))
  # A buffer of a vector's lanes (a block's running sums) shares slots only with others of its
  # lanes: in a slot that scalar code also uses, the C compiler kept neither the vector nor the
  # scalars in registers.
  ranks = _assign_slots([((packable[buf].dtype, packable[buf].attrs.get("lanes")), first_write[buf], last_use[buf]) for buf in order])
  names: dict[int, str] = {}
  used_names = {n.attrs["name"] for n in _walk(proc) if n.op == ProgramOp.BUFFER}
  counter = 0
  for buf, rank in zip(order, ranks, strict=True):
    if rank not in names:
      while (chosen := f"s{counter}") in used_names:
        counter += 1
      used_names.add(chosen)
      counter += 1
      names[rank] = chosen
      plan.slot_dtype[chosen] = packable[buf].dtype
      plan.slot_size[chosen] = 0
      if (lanes := packable[buf].attrs.get("lanes")) is not None:
        plan.slot_lanes[chosen] = lanes
    chosen = names[rank]
    plan.rename[buf] = chosen
    plan.slot_size[chosen] = max(plan.slot_size[chosen], _size_of(packable[buf].attrs["shape"]))

  # Spill plan: float64 slots at/above the threshold get a sequential window in w[].
  total = 0
  for slot in plan.slot_size:
    if plan.slot_dtype[slot].is_floating and plan.slot_size[slot] >= WORKSPACE_SPILL_THRESHOLD:
      plan.spill_offset[slot] = total
      total += plan.slot_size[slot]
  plan.own_spill = total
  return plan


def _assign_slots(items: Sequence[tuple[Hashable, int, int]]) -> list[int]:
  """Greedy left-edge slot assignment. ``items`` are buffers in first-write order, each ``(kind,
  first write, last use)``; each takes the earliest-created slot of its kind that is free by its
  first write (a slot frees the statement after its buffer's last use), else a new slot. Returns
  each buffer's slot, numbered in creation order.

  First writes only grow along ``items``, so a slot once free stays free until it is taken: per
  kind, slots wait in a heap by the position they free at and move to a heap by creation order
  when that position is reached."""
  waiting: dict[Hashable, list[tuple[int, int]]] = {}
  free: dict[Hashable, list[int]] = {}
  slots: list[int] = []
  created = 0
  for kind, first, last in items:
    pending, ready = waiting.setdefault(kind, []), free.setdefault(kind, [])
    while pending and pending[0][0] <= first:
      heapq.heappush(ready, heapq.heappop(pending)[1])
    if ready:
      slot = heapq.heappop(ready)
    else:
      slot, created = created, created + 1
    slots.append(slot)
    heapq.heappush(pending, (last + 1, slot))
  return slots


def pack_workspace(prog: ProgramNode) -> ProgramNode:
  procs, kernels = _procs(prog)
  plans = {pr.attrs["name"]: _plan_pack(pr) for pr in procs}

  # An extern callee (it renders its own C, so it has no PROC here) is opaque: it owns no spill of
  # its own but passes its ``w`` straight through to its dependencies and hand-written sources, so
  # its workspace is the max over those. The lowerer records extern -> dependency names and the
  # sources' workspace on the PROGRAM; we seed them into the sz_w recursion below.
  extern_deps: dict[str, tuple[str, ...]] = prog.attrs.get("extern_deps", {})
  extern_workspace: dict[str, int] = prog.attrs.get("extern_workspace", {})

  # sz_w(proc) = own spill + max callee workspace (callees share the post-own-spill window).
  sz_w: dict[str, int] = {}

  def total(name: str) -> int:
    if name in sz_w:
      return sz_w[name]
    if name in extern_deps:
      sz_w[name] = 0  # break cycles defensively; an extern callee owns no spill itself
      sz_w[name] = max((extern_workspace.get(name, 0), *(total(d) for d in extern_deps[name] if d in plans)))
      return sz_w[name]
    sz_w[name] = plans[name].own_spill  # break cycles defensively
    callee_max = max((total(c) for c in plans[name].callees if c in plans or c in extern_deps), default=0)
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
    if slot in plan.slot_lanes:
      attrs["lanes"] = plan.slot_lanes[slot]
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
    rewritten = rewrite(stmt, [Pattern(None, lambda n: True, fn)], rebuild=rebuild_program, fixpoint=False)
    if rewritten.op == ProgramOp.BUFFER:
      slot_name = rewritten.attrs["name"]
      if slot_name in seen_slot:
        continue  # one decl per slot (members collapse to the same node)
      seen_slot.add(slot_name)
    new_body.append(rewritten)

  return _rebuild_proc(proc, params, new_body, sz_w=sz_w.get(name, 0), w_self=plan.own_spill)


__all__ = ["WORKSPACE_SPILL_THRESHOLD", "pack_workspace"]
