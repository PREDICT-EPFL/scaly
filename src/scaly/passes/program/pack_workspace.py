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


@dataclass(slots=True)
class _PackPlan:
  rename: dict[str, str] = field(default_factory=dict)  # private buffer name -> slot name
  slot_dtype: dict[str, DType] = field(default_factory=dict)  # slot name -> dtype
  slot_size: dict[str, int] = field(default_factory=dict)  # slot name -> element count
  spill_offset: dict[str, int] = field(default_factory=dict)  # slot name -> offset into w[]
  own_spill: int = 0  # doubles this proc spills to its own w[] window
  callees: list[str] = field(default_factory=list)  # one entry per call site


def _plan_pack(proc: ProgramNode) -> _PackPlan:
  """Lifetime-pack a PROC's private buffers into shared slots.

  Greedy left-edge packing over statement order (the lowerer emits values in topological /
  creation order, so first-write order is a valid schedule). Slots are dtype-homogeneous.
  Storage remains eight bytes per element until the typed workspace layout is introduced.
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
  last_use: dict[str, int] = {}
  deps: dict[str, set[str]] = {}
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
    produced_from = reads.union(*(deps.get(r, ()) for r in reads))
    for b in writes:
      first_write.setdefault(b, i)
      deps[b] = produced_from
    for b in writes | reads:
      last_use[b] = i

  # Pack per dtype, in first-write order (ties: declaration order via the dict insertion order).
  order = sorted((name for name in packable if name in first_write), key=lambda b: (first_write[b], b))
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


def pack_workspace(prog: ProgramNode, *, frame_budget: int = 65536) -> ProgramNode:
  """Pack private storage and spill the largest slots until each estimated frame fits.

  Only float64 slots can use the double workspace. Other locals still count toward the estimate.
  Zero spills every eligible slot. Scalar spills still need stack space, covered by a fixed allowance.
  """
  procs = prog.args
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
      count * plans[owner].slot_size[slot] * 8 for (owner, slot), count in slots(name).items() if slot not in plans[owner].spill_offset
    )

  for name in plans:
    frame = estimate(name)
    for owner, slot in sorted(slots(name), key=lambda key: plans[key[0]].slot_size[key[1]], reverse=True):
      if frame_budget and frame <= frame_budget:
        break
      plan = plans[owner]
      if plan.slot_dtype[slot] == dtypes.float64 and slot not in plan.spill_offset:
        plan.spill_offset[slot] = plan.own_spill
        plan.own_spill += plan.slot_size[slot]
        frame -= slots(name)[owner, slot] * plan.slot_size[slot] * 8

  # A solver wrapper (rendered by codegen/solver, so it has no PROC here) is an opaque callee: it owns
  # no spill of its own but passes its ``w`` straight through to its oracle Functions, so its
  # workspace is the max over those oracle PROCs. The lowerer records solver -> oracle names on
  # the PROGRAM; we seed those names into the sz_w recursion below.
  solver_oracles: dict[str, tuple[str, ...]] = prog.attrs.get("solver_oracles", {})
  solver_external_workspace: dict[str, int] = prog.attrs.get("solver_external_workspace", {})

  # sz_w(proc) = own spill + max callee workspace (callees share the post-own-spill window).
  sz_w: dict[str, int] = {fn.name: fn.workspace for fn in prog.attrs.get("externs", ())}

  def total(name: str) -> int:
    if name in sz_w:
      return sz_w[name]
    if name in solver_oracles:
      sz_w[name] = 0  # break cycles defensively; a solver owns no spill itself
      sz_w[name] = max((solver_external_workspace.get(name, 0), *(total(o) for o in solver_oracles[name] if o in plans)))
      return sz_w[name]
    sz_w[name] = plans[name].own_spill  # break cycles defensively
    callee_max = max((total(c) for c in plans[name].callees if c in plans or c in solver_oracles or c in sz_w), default=0)
    sz_w[name] = plans[name].own_spill + callee_max
    return sz_w[name]

  for name in plans:
    total(name)

  procs = [_apply_pack(pr, plans[pr.attrs["name"]], sz_w, estimate(pr.attrs["name"])) for pr in procs]
  return ProgramNode(ProgramOp.PROGRAM, tuple(procs), prog.attrs, dtype=prog.dtype)


def _apply_pack(proc: ProgramNode, plan: _PackPlan, sz_w: dict[str, int], frame_bytes: int) -> ProgramNode:
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
      return ProgramNode(ProgramOp.CALL, n.args, {**n.attrs, "w_self": plan.own_spill, "callee_needs_w": sz_w.get(callee, 0) > 0}, dtype=n.dtype)
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
