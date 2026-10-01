"""Loop-invariant callee work hoisted out of mapped loops in the Program IR."""

from __future__ import annotations

from ...ir.program import ProgramNode, ProgramOp, buffer, for_
from ...utils.names import c_ident
from ._common import (
  allocated_name,
  buffer_refs,
  prune_dead_buffers,
  prune_procedures,
  _alias_sources,
  _proc_parts,
  _procs,
  _rebuild_proc,
  _resolve_alias,
  _walk,
  trip_count,
)

_Split = tuple[ProgramNode, ProgramNode, tuple[int, ...], list[ProgramNode]]

# A call inside a mapped callee is split when its prologue would hold at least this share of the
# call's work (``_work``). Measured: the stage derivatives of the race cars and the chain, whose
# ODE calls derive one to twenty-six values from broadcast parameters (under a tenth of each call),
# ran 1.2 to 2.5 times slower split, because the halves are not expanded into one straight-line
# stage as the whole calls were; a custom rule's Riccati recursion on broadcast data is two thirds
# of its call, and hoisting it took a gradient from 59 to 6 ms.
_CALL_SHARE = 0.25


def _work(proc: ProgramNode, table: dict[str, ProgramNode], work: dict[str, int]) -> int:
  """The values ``proc`` computes in one call, kept in ``work`` by name: its stores, each counted
  once for every trip of the loops around it (a loop whose bounds are not constants counts as one
  trip), and the work of the procedures it calls (one for a procedure neither in ``table`` nor
  counted before, an extern body)."""
  name = proc.attrs["name"]
  if name in work:
    return work[name]

  def count(node: ProgramNode, trips: int) -> int:
    if node.op == ProgramOp.FOR:
      steps = trip_count(node.args[0])
      return sum(count(stmt, trips * (1 if steps is None else steps)) for stmt in node.args[1:])
    if node.op in (ProgramOp.STORE, ProgramOp.STORE_PAIR):
      return trips
    if node.op == ProgramOp.CALL:
      callee = node.attrs["callee"]
      if callee not in work and callee not in table:
        return trips
      return trips * (work[callee] if callee in work else _work(table[callee], table, work))
    return sum(count(arg, trips) for arg in node.args)

  work[name] = sum(count(stmt, 1) for stmt in proc.args[int(proc.attrs["param_count"]) :])
  return work[name]


def hoist_invariant(prog: ProgramNode) -> ProgramNode:
  """Split a mapped callee whose inputs are partly the same at every trip into a prologue, run once
  before the loop, and a body that receives the prologue's buffers as extra inputs.

  A callee cannot know which of its arguments a ``VMAP`` broadcasts, so it recomputes everything
  derived from them at every trip. Inside the callee, a private buffer is invariant when every
  statement writing it reads only invariant inputs, constants and other invariant buffers; the
  statements producing the invariant buffers the rest of the body reads move to the prologue.

  The split goes through calls. A call inside the callee whose arguments are partly invariant is
  split the same way, and its prologue, which reads invariant buffers only, moves out with the
  rest: without that, everything a nested Function derives from a broadcast argument was
  recomputed at every trip, however little of the call depended on the trip (the Riccati recursion
  inside a custom derivative rule, mapped over a batch that shares its weights). A call is split
  only when a quarter or more of its work is invariant (``_CALL_SHARE``): a split procedure is no
  longer expanded into its caller as one body, which costs more than a few hoisted scalars save.
  """
  procs, kernels = _procs(prog)
  table = {pr.attrs["name"]: pr for pr in procs}
  used_names = {c_ident(name) for name in table}
  pure: set[str] = set()
  splits: dict[tuple[str, tuple[int, ...]], _Split | None] = {}
  work: dict[str, int] = {}
  rewritten: list[ProgramNode] = []
  for pr in procs:  # callees precede callers, so a split always sees the callee's rewritten body
    table[pr.attrs["name"]] = _hoist_proc(pr, table, splits, used_names, pure, work)
    rewritten.append(table[pr.attrs["name"]])
    if all(n.attrs["callee"] in pure for n in _walk(table[pr.attrs["name"]]) if n.op == ProgramOp.CALL):
      pure.add(pr.attrs["name"])
  if all(a is b for a, b in zip(procs, rewritten, strict=True)):
    return prog
  inserted: dict[str, list[ProgramNode]] = {key[0]: [] for key in splits}
  for key, split in splits.items():
    if split is not None:
      inserted[key[0]].extend(split[:2])
  out: list[ProgramNode] = []
  for pr in rewritten:
    out.extend(inserted.get(pr.attrs["name"], ()))
    out.append(pr)
  return prune_procedures(ProgramNode(ProgramOp.PROGRAM, (*out, *kernels), {**prog.attrs, "proc_count": len(out)}, prog.dtype))


def _hoist_proc(
  proc: ProgramNode,
  table: dict[str, ProgramNode],
  splits: dict[tuple[str, tuple[int, ...]], _Split | None],
  used_names: set[str],
  pure: set[str],
  work: dict[str, int],
) -> ProgramNode:
  params, body = _proc_parts(proc)
  aliases = _alias_sources(body)
  local_names = {c_ident(n.attrs["name"]) for n in _walk(proc) if n.op in (ProgramOp.BUFFER, ProgramOp.RANGE, ProgramOp.VAR)}
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
      splits[key] = _split(table[key[0]], invariant, used_names, pure, table, splits, work)
    split = splits[key]
    if split is None:
      new_body.append(stmt)
      continue
    prologue, hoisted, used, exported = split
    bufs = [
      buffer(allocated_name(f"{rng.attrs['name']}_{b.attrs['name']}", local_names), b.dtype, b.attrs["shape"], address_space="private")
      for b in exported
    ]
    new_body.extend(bufs)
    new_body.append(_call(prologue, [*(c.args[k] for k in used), *bufs]))
    new_body.append(for_(rng, [_call(hoisted, [*c.args[:n_in], *bufs, *c.args[n_in:]])]))
  return proc if len(new_body) == len(body) else _rebuild_proc(proc, params, new_body)


def _call(callee: ProgramNode, args: list[ProgramNode]) -> ProgramNode:
  n_in = int(callee.attrs["input_count"])
  return ProgramNode(ProgramOp.CALL, tuple(args), {"callee": callee.attrs["name"], "n_in": n_in, "n_out": len(args) - n_in, "returns": ()})


def _split(
  proc: ProgramNode,
  invariant: tuple[int, ...],
  used_names: set[str],
  pure: set[str],
  table: dict[str, ProgramNode],
  splits: dict[tuple[str, tuple[int, ...]], _Split | None],
  work: dict[str, int],
) -> _Split | None:
  params, body = _proc_parts(proc)
  n_in = int(proc.attrs["input_count"])
  outputs = {pp.attrs["name"] for pp in params[n_in:]}
  local_names = {c_ident(n.attrs["name"]) for n in _walk(proc) if n.op in (ProgramOp.BUFFER, ProgramOp.RANGE, ProgramOp.VAR)}
  while True:
    aliases = _alias_sources(body)
    fixed = {params[k].attrs["name"] for k in invariant} | {s.attrs["name"] for s in body if s.op == ProgramOp.BUFFER and "values" in s.attrs}
    stmt_refs = {i: buffer_refs(s, aliases) for i, s in enumerate(body) if s.op != ProgramOp.BUFFER}
    refs = {i: (set(r.reads), set(r.writes)) for i, r in stmt_refs.items()}
    writers: dict[str, set[int]] = {}
    for i, (_, writes) in refs.items():
      for b in writes:
        writers.setdefault(b, set()).add(i)
    opaque_calls = {
      i
      for i, stmt in enumerate(body)
      if stmt.op != ProgramOp.BUFFER and any(n.op == ProgramOp.CALL and n.attrs["callee"] not in pure for n in _walk(stmt))
    }
    call_buffers = {name for i in opaque_calls for name in stmt_refs[i].reads | stmt_refs[i].writes}
    # A loop whose variable outlives it (``exit_var``) and a statement reading such a variable belong
    # together, and buffer references do not show that link: neither may move.
    scoped = {i for i, stmt in enumerate(body) if stmt.op != ProgramOp.BUFFER and _binds_or_reads_outer_var(stmt)}
    hoist = {
      i
      for i, (reads, writes) in refs.items()
      if not writes & outputs and i not in opaque_calls and i not in scoped and not (reads | writes) & call_buffers
    }
    while True:
      known = fixed | {b for b, ws in writers.items() if ws <= hoist}
      kept = {i for i in hoist if refs[i][0] <= known and refs[i][1] <= known}
      if kept == hoist:
        break
      hoist = kept
    through = _split_a_call(body, hoist | opaque_calls | scoped, known, aliases, local_names, used_names, pure, table, splits, work)
    if through is None:
      break
    body = through
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
  tag = "_".join(map(str, invariant))
  name = proc.attrs["name"]
  # The prologue runs once per call, so expanding it on its own would only grow the source with
  # the invariant argument's size; under ``auto`` it expands only inside an expanding caller.
  prologue_mode = proc.attrs.get("scalarize_mode", "disabled")
  if prologue_mode != "disabled" and proc.attrs.get("lowering") != "scalar":
    prologue_mode = "inline"
  prologue = _proc(
    proc,
    allocated_name(f"{name}_hoist_{tag}", used_names),
    [params[k] for k in used],
    shared,
    [s for i, s in enumerate(body) if keep(s) or i in hoist],
    scalarize_mode=prologue_mode,
  )
  hoisted = _proc(
    proc,
    allocated_name(f"{name}_hoisted_{tag}", used_names),
    [*params[:n_in], *shared],
    params[n_in:],
    [s for i, s in enumerate(body) if keep(s) or (i in refs and i not in hoist)],
    hoisted_from=proc.attrs.get("hoisted_from", name),
  )
  for generated in (prologue, hoisted):
    if all(n.attrs["callee"] in pure for n in _walk(generated) if n.op == ProgramOp.CALL):
      pure.add(generated.attrs["name"])
    _work(generated, table, work)  # counted while the halves it calls are at hand
  return prologue, hoisted, used, exported


def _split_a_call(
  body: list[ProgramNode],
  settled: set[int],
  known: set[str],
  aliases: dict[str, str],
  local_names: set[str],
  used_names: set[str],
  pure: set[str],
  table: dict[str, ProgramNode],
  splits: dict[tuple[str, tuple[int, ...]], _Split | None],
  work: dict[str, int],
) -> list[ProgramNode] | None:
  """``body`` with the first call that stays in the loop, to a callee some of whose inputs are
  ``known`` invariant, replaced by that callee's two halves: the prologue's buffers, the call of
  the prologue (which reads invariant buffers only, so the caller's split then moves it out), and
  the call of the rest. None when no call splits."""
  for i, stmt in enumerate(body):
    if stmt.op != ProgramOp.CALL or i in settled or stmt.attrs["callee"] not in table or stmt.attrs["callee"] not in pure:
      continue
    n_in = int(stmt.attrs["n_in"])
    names = [_resolve_alias(a.attrs.get("buffer", a.attrs.get("name")), aliases) for a in stmt.args]
    invariant = tuple(k for k in range(n_in) if names[k] in known)
    if not invariant or set(names[n_in:]) & {names[k] for k in invariant}:
      continue
    key = (stmt.attrs["callee"], invariant)
    if key not in splits:
      splits[key] = None  # while it is being split, and if it does not split
      splits[key] = _split(table[key[0]], invariant, used_names, pure, table, splits, work)
    split = splits[key]
    if split is None:
      continue
    prologue, hoisted, used, exported = split
    moved, kept = work[prologue.attrs["name"]], work[hoisted.attrs["name"]]
    if moved < _CALL_SHARE * (moved + kept):
      continue
    bufs = [buffer(allocated_name(f"{key[0]}_{b.attrs['name']}", local_names), b.dtype, b.attrs["shape"], address_space="private") for b in exported]
    calls = [_call(prologue, [*(stmt.args[k] for k in used), *bufs]), _call(hoisted, [*stmt.args[:n_in], *bufs, *stmt.args[n_in:]])]
    return [*body[:i], *bufs, *calls, *body[i + 1 :]]
  return None


def _binds_or_reads_outer_var(stmt: ProgramNode) -> bool:
  """Whether ``stmt`` holds a loop whose variable outlives it, sets a variable at the top level, or
  reads a variable it does not bind."""
  nodes = list(_walk(stmt))
  if stmt.op == ProgramOp.ASSIGN or any(n.op == ProgramOp.FOR and n.attrs.get("exit_var") for n in nodes):
    return True
  bound = {n.attrs["name"] for n in nodes if n.op == ProgramOp.RANGE} | {n.attrs["target"] for n in nodes if n.op == ProgramOp.ASSIGN}
  return any(n.op == ProgramOp.VAR and n.attrs["name"] not in bound for n in nodes)


def _proc(
  origin: ProgramNode, name: str, inputs: list[ProgramNode], outputs: list[ProgramNode], body: list[ProgramNode], **attrs: object
) -> ProgramNode:
  return prune_dead_buffers(_rebuild_proc(origin, [*inputs, *outputs], body, name=name, input_count=len(inputs), **attrs))


__all__ = ["hoist_invariant"]
