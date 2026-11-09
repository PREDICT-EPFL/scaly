"""Scatter-sum combination for the Program IR."""

from __future__ import annotations

import numpy as np

from ...ir.match import Pattern, rewrite
from ...ir.program import ProgramNode, ProgramOp, add, for_, load, store
from ._common import (
  _alias_sources,
  _index_values,
  _map_procs,
  _private_decls,
  _proc_parts,
  _rebuild_proc,
  _resolve_alias,
  _size_of,
  buffer_refs,
  inline_producer as _as_inline_producer,
  prune_dead_buffers,
  trip_count as _trip_count,
  rebuild_program,
)


def combine_scatter_sums(prog: ProgramNode) -> ProgramNode:
  """Accumulate single-use sums of zero-filled scatters into one destination."""
  return _map_procs(prog, _combine_scatter_sums_proc)


def _combine_scatter_sums_proc(proc: ProgramNode) -> ProgramNode:
  params, body = _proc_parts(proc)
  private = _private_decls(body)
  aliases = _alias_sources(body)
  pinned = set(aliases.values())
  decls = {s.attrs["name"]: s for s in (*params, *body) if s.op == ProgramOp.BUFFER}
  refs = [buffer_refs(s) if s.op != ProgramOp.BUFFER else None for s in body]
  reads: dict[str, set[int]] = {}
  writes: dict[str, set[int]] = {}
  for i, ref in enumerate(refs):
    if ref is None:
      continue
    for b in ref.reads:
      reads.setdefault(b, set()).add(i)
    for b in ref.writes:
      writes.setdefault(b, set()).add(i)

  sums: dict[str, tuple[int, tuple[str, ...]]] = {}
  pads: dict[str, tuple[int, int]] = {}
  destinations: dict[str, np.ndarray] = {}
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
    accumulating = value.op == ProgramOp.ADD and value.args[0] == load(target)
    if accumulating:
      value = value.args[1]
    if target.attrs["buffer"] != buf or len(target.args) != 1 or value.op != ProgramOp.LOAD:
      continue
    iv = scatter.args[0].attrs["name"]
    source = value.args[0]
    if len(source.args) != 1 or source.args[0].op != ProgramOp.VAR or source.args[0].attrs["name"] != iv:
      continue
    if source.attrs["buffer"] == buf:
      continue
    indices = _index_values(target.args[0], scatter.args[0], decls)
    if indices is None or not len(indices) or (not accumulating and len(np.unique(indices)) != len(indices)):
      continue
    pads[buf] = i, j
    destinations[buf] = indices

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
        buf not in private
        or buf in pinned
        or reads.get(buf, set()) - ({pads[buf][1]} if buf in pads else set()) != {consumer}
        or _size_of(decls[buf].attrs["shape"]) != _size_of(decls[root].attrs["shape"])
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
        scatter_refs = refs[scatter]
        assert scatter_refs is not None
        sources = {_resolve_alias(b, aliases) for b in scatter_refs.loads if b != buf}
        if _resolve_alias(root, aliases) in sources or any(
          sources & {_resolve_alias(b, aliases) for b in ref.writes} for ref in refs[scatter + 1 : end] if ref is not None
        ):
          valid = False
          break
        leaves.append(buf)
        drop.update((start, scatter))
      else:
        valid = False
        break
    occupied: set[int] = set()
    for leaf in leaves:
      indices = destinations[leaf]
      # A repeated leaf must finish its own sum before adding to an earlier leaf.
      if len(np.unique(indices)) != len(indices) and occupied.intersection(indices.tolist()):
        valid = False
        break
      occupied.update(indices.tolist())
    if not valid or len(leaves) < 2 or drop & removed:
      continue

    destination = Pattern(
      ProgramOp.VIEW, lambda n: n.attrs["buffer"] in leaves, lambda n: ProgramNode(n.op, n.args, {**n.attrs, "buffer": root}, n.dtype)
    )
    redirect = lambda stmt: rewrite(stmt, [destination], rebuild=rebuild_program, fixpoint=False)

    replacement = [redirect(body[pads[leaves[0]][0]])]
    for leaf in leaves:
      scatter = redirect(body[pads[leaf][1]])
      target, value = scatter.args[1].args
      if value.op == ProgramOp.ADD and value.args[0] == load(target):
        value = value.args[1]
      replacement.append(for_(scatter.args[0], [store(target, add(load(target), value))]))
    replacements[end] = replacement
    removed.update(drop)
  if not replacements:
    return proc
  new_body = [s for i, stmt in enumerate(body) for s in (replacements[i] if i in replacements else [] if i in removed else [stmt])]
  return prune_dead_buffers(_rebuild_proc(proc, params, new_body))
