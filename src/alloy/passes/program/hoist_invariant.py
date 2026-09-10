"""Loop-invariant callee work hoisted out of mapped loops in the Program IR."""

from __future__ import annotations

from ...ir.program import ProgramNode, ProgramOp, buffer, for_
from ._common import _alias_sources, _proc_parts, _procs, _rebuild_proc, _resolve_alias, _walk
from .fuse_elementwise import _prune_dead_buffers

_Split = tuple[ProgramNode, ProgramNode, tuple[int, ...], list[ProgramNode]]


def hoist_invariant(prog: ProgramNode) -> ProgramNode:
  """Split a mapped callee whose inputs are partly the same at every trip into a prologue, run once
  before the loop, and a body that receives the prologue's buffers as extra inputs.

  A callee cannot know which of its arguments a ``VMAP`` broadcasts, so it recomputes everything
  derived from them at every trip. Inside the callee, a private buffer is invariant when every
  statement writing it reads only invariant inputs, constants and other invariant buffers; the
  statements producing the invariant buffers the rest of the body reads move to the prologue.
  """
  procs, kernels = _procs(prog)
  table = {pr.attrs["name"]: pr for pr in procs}
  splits: dict[tuple[str, tuple[int, ...]], _Split | None] = {}
  rewritten: list[ProgramNode] = []
  for pr in procs:  # callees precede callers, so a split always sees the callee's rewritten body
    table[pr.attrs["name"]] = _hoist_proc(pr, table, splits)
    rewritten.append(table[pr.attrs["name"]])
  if all(a is b for a, b in zip(procs, rewritten, strict=True)):
    return prog
  inserted: dict[str, list[ProgramNode]] = {key[0]: [] for key in splits}
  for key, split in splits.items():
    if split is not None:
      inserted[key[0]].extend(split[:2])
  # A split callee stays only while a CALL still needs it; other PROCs may be reached from a
  # solver wrapper without any CALL, so they are never pruned here.
  every = [*rewritten, *(pr for pair in inserted.values() for pr in pair)]
  called = {n.attrs["callee"] for pr in every for n in _walk(pr) if n.op == ProgramOp.CALL}
  called.update(o for oracles in prog.attrs.get("solver_oracles", {}).values() for o in oracles)
  out: list[ProgramNode] = []
  for pr in rewritten:
    out.extend(inserted.get(pr.attrs["name"], ()))
    if pr.attrs["name"] not in inserted or pr.attrs["name"] in called:
      out.append(pr)
  return ProgramNode(ProgramOp.PROGRAM, (*out, *kernels), {**prog.attrs, "proc_count": len(out)}, prog.dtype)


def _hoist_proc(proc: ProgramNode, table: dict[str, ProgramNode], splits: dict[tuple[str, tuple[int, ...]], _Split | None]) -> ProgramNode:
  params, body = _proc_parts(proc)
  aliases = _alias_sources(body)
  new_body: list[ProgramNode] = []
  for stmt in body:
    if stmt.op != ProgramOp.FOR or len(stmt.args) != 2 or stmt.args[1].op != ProgramOp.CALL or stmt.args[1].attrs["callee"] not in table:
      new_body.append(stmt)
      continue
    rng, c = stmt.args
    n_in = int(c.attrs["n_in"])
    invariant = tuple(k for k in range(n_in) if not any(n.op == ProgramOp.VAR for n in _walk(c.args[k])))
    written = {_resolve_alias(a.attrs.get("buffer", a.attrs.get("name")), aliases) for a in c.args[n_in:]}
    if not invariant or any(_resolve_alias(c.args[k].attrs.get("buffer", c.args[k].attrs.get("name")), aliases) in written for k in invariant):
      new_body.append(stmt)
      continue
    key = (c.attrs["callee"], invariant)
    if key not in splits:
      splits[key] = _split(table[key[0]], invariant)
    split = splits[key]
    if split is None:
      new_body.append(stmt)
      continue
    prologue, hoisted, used, exported = split
    bufs = [buffer(f"{rng.attrs['name']}_{b.attrs['name']}", b.dtype, b.attrs["shape"], address_space="private") for b in exported]
    new_body.extend(bufs)
    new_body.append(_call(prologue, [*(c.args[k] for k in used), *bufs]))
    new_body.append(for_(rng, [_call(hoisted, [*c.args[:n_in], *bufs, *c.args[n_in:]])]))
  return proc if len(new_body) == len(body) else _rebuild_proc(proc, params, new_body)


def _call(callee: ProgramNode, args: list[ProgramNode]) -> ProgramNode:
  n_in = int(callee.attrs["input_count"])
  return ProgramNode(ProgramOp.CALL, tuple(args), {"callee": callee.attrs["name"], "n_in": n_in, "n_out": len(args) - n_in, "returns": ()})


def _refs(stmt: ProgramNode, aliases: dict[str, str]) -> tuple[set[str], set[str]]:
  """Alias-resolved ``(reads, writes)`` of one statement, a nested CALL's inputs read and outputs written."""
  reads: set[str] = set()
  writes: set[str] = set()
  for n in _walk(stmt):
    if n.op == ProgramOp.LOAD:
      reads.add(n.args[0].attrs["buffer"])
    elif n.op == ProgramOp.STORE:
      writes.add(n.args[0].attrs["buffer"])
    elif n.op == ProgramOp.CALL:
      for k, a in enumerate(n.args):
        (reads if k < int(n.attrs["n_in"]) else writes).add(a.attrs.get("buffer", a.attrs.get("name")))
  return {_resolve_alias(b, aliases) for b in reads}, {_resolve_alias(b, aliases) for b in writes}


def _split(proc: ProgramNode, invariant: tuple[int, ...]) -> _Split | None:
  params, body = _proc_parts(proc)
  n_in = int(proc.attrs["input_count"])
  aliases = _alias_sources(body)
  fixed = {params[k].attrs["name"] for k in invariant} | {s.attrs["name"] for s in body if s.op == ProgramOp.BUFFER and "values" in s.attrs}
  outputs = {pp.attrs["name"] for pp in params[n_in:]}
  refs = {i: _refs(s, aliases) for i, s in enumerate(body) if s.op != ProgramOp.BUFFER}
  writers: dict[str, set[int]] = {}
  for i, (_, writes) in refs.items():
    for b in writes:
      writers.setdefault(b, set()).add(i)
  hoist = {i for i, (_, writes) in refs.items() if not writes & outputs}
  while True:
    known = fixed | {b for b, ws in writers.items() if ws <= hoist}
    kept = {i for i in hoist if refs[i][0] <= known and refs[i][1] <= known}
    if kept == hoist:
      break
    hoist = kept
  produced = {b for b, ws in writers.items() if ws <= hoist}
  exported_names = {b for i, (reads, _) in refs.items() if i not in hoist for b in reads & produced}
  if not exported_names:
    return None
  # A reader that ran between two hoisted writes saw the earlier value; the prologue only keeps the last.
  if any(i < max(writers[b]) for i, (reads, _) in refs.items() if i not in hoist for b in reads & exported_names):
    return None
  decls = {s.attrs["name"]: s for s in body if s.op == ProgramOp.BUFFER}
  exported = [decls[b] for b in sorted(exported_names)]
  shared = [buffer(b.attrs["name"], b.dtype, b.attrs["shape"]) for b in exported]
  read = {b for i in hoist for b in refs[i][0]}
  used = tuple(k for k in invariant if params[k].attrs["name"] in read)
  keep = lambda s: s.op == ProgramOp.BUFFER and s.attrs["name"] not in exported_names
  tag = "".join(map(str, invariant))  # one split per invariant-position set, so the names stay distinct
  name = proc.attrs["name"]
  # The prologue runs once per call, so expanding it on its own would only grow the source with
  # the invariant argument's size; under ``auto`` it expands only inside an expanding caller.
  scalar = proc.attrs.get("scalarize") and ("callee" if proc.attrs.get("lowering") == "auto" else True)
  prologue = _proc(
    proc, f"{name}_hoist{tag}", [params[k] for k in used], shared, [s for i, s in enumerate(body) if keep(s) or i in hoist], scalarize=scalar
  )
  hoisted = _proc(
    proc,
    f"{name}_hoisted{tag}",
    [*params[:n_in], *shared],
    params[n_in:],
    [s for i, s in enumerate(body) if keep(s) or (i in refs and i not in hoist)],
    hoisted_from=proc.attrs.get("hoisted_from", name),
  )
  return prologue, hoisted, used, exported


def _proc(
  origin: ProgramNode, name: str, inputs: list[ProgramNode], outputs: list[ProgramNode], body: list[ProgramNode], **attrs: object
) -> ProgramNode:
  return _prune_dead_buffers(_rebuild_proc(origin, [*inputs, *outputs], body, name=name, input_count=len(inputs), **attrs))


__all__ = ["hoist_invariant"]
