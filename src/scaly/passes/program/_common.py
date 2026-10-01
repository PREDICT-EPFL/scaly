"""Shared Program IR traversal and procedure helpers for optimization passes."""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from ...ir.expr import op_def, registered_ops
from ...ir.match import Pattern, rewrite
from ...ir.program import ProgramNode, ProgramOp
from ...utils.names import c_ident


@functools.cache
def expensive_ops(version: int) -> frozenset[ProgramOp]:
  """The program ops of the expression ops with both the ``elementwise`` and the ``expensive``
  trait: libm calls, which fusion does not duplicate and a select does not compute ahead. Keyed by
  the registry's version (``ir.expr.registry_version``)."""
  traits = [op_def(op).traits for op in registered_ops()]
  return frozenset(t["elementwise"] for t in traits if t.get("expensive") and "elementwise" in t)


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


def _postorder(root: ProgramNode) -> Iterable[ProgramNode]:
  """Yield every distinct node at/below ``root`` with each node after all of its args."""
  seen: set[int] = set()
  stack = [(root, False)]
  while stack:
    n, ready = stack.pop()
    if id(n) in seen:
      continue
    if ready:
      seen.add(id(n))
      yield n
      continue
    stack.append((n, True))
    stack.extend((a, False) for a in reversed(n.args) if id(a) not in seen)


def rebuild_program(node: ProgramNode, args: tuple[ProgramNode, ...]) -> ProgramNode:
  """The program-dialect adapter for ``ir.match.rewrite``: ``node`` with new ``args``."""
  return ProgramNode(node.op, args, node.attrs, node.dtype)


def inline_producer(stmt: ProgramNode) -> tuple[str, str, ProgramNode] | None:
  """Match ``for v in [0,N) step 1 { buf[v] = rhs }``."""
  if stmt.op != ProgramOp.FOR:
    return None
  rng, *body = stmt.args
  if len(body) != 1 or body[0].op != ProgramOp.STORE:
    return None
  target, rhs = body[0].args
  if target.op != ProgramOp.VIEW or len(target.args) != 1:
    return None
  name = rng.attrs["name"]
  index = target.args[0]
  start, _stop, step = rng.args
  if index.op != ProgramOp.VAR or index.attrs["name"] != name:
    return None
  if start.op != ProgramOp.CONST_INT or start.attrs["value"] != 0:
    return None
  if step.op != ProgramOp.CONST_INT or step.attrs["value"] != 1:
    return None
  return target.attrs["buffer"], name, rhs


def substitute_var(node: ProgramNode, name: str, replacement: ProgramNode) -> ProgramNode:
  """Replace each variable named ``name`` below ``node``."""
  pattern = Pattern(ProgramOp.VAR, lambda n: n.attrs["name"] == name, lambda _n: replacement)
  return rewrite(node, [pattern], rebuild=rebuild_program, fixpoint=False)


def trip_count(rng: ProgramNode) -> int | None:
  """Return a static range's iteration count, or ``None`` for dynamic or invalid ranges."""
  start, stop, step = rng.args
  if any(node.op != ProgramOp.CONST_INT for node in (start, stop, step)):
    return None
  span = int(stop.attrs["value"]) - int(start.attrs["value"])
  stride = int(step.attrs["value"])
  return max(0, -(-span // stride)) if stride > 0 else None


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


@dataclass(frozen=True)
class BufferRefs:
  """Buffer names read and written directly or through calls by one statement."""

  loads: frozenset[str]
  stores: frozenset[str]
  call_inputs: frozenset[str]
  call_outputs: frozenset[str]

  @property
  def reads(self) -> frozenset[str]:
    return self.loads | self.call_inputs

  @property
  def writes(self) -> frozenset[str]:
    return self.stores | self.call_outputs

  @property
  def call_args(self) -> frozenset[str]:
    return self.call_inputs | self.call_outputs


def buffer_refs(stmt: ProgramNode, aliases: dict[str, str] | None = None) -> BufferRefs:
  """Return direct and call-mediated buffer references, optionally resolved to storage owners."""
  loads: set[str] = set()
  stores: set[str] = set()
  call_inputs: set[str] = set()
  call_outputs: set[str] = set()
  for n in _walk(stmt):
    if n.op == ProgramOp.LOAD:
      loads.add(n.args[0].attrs["buffer"])
    elif n.op in (ProgramOp.STORE, ProgramOp.STORE_PAIR):
      stores.add(n.args[0].attrs["buffer"])
    elif n.op in (ProgramOp.CALL, ProgramOp.LAUNCH):
      args = n.args if n.op == ProgramOp.CALL else n.args[int(n.attrs["grid_dims"]) + int(n.attrs["block_dims"]) :]
      n_in, n_out = n.attrs.get("n_in"), n.attrs.get("n_out")
      for k, a in enumerate(args):
        name = _call_arg_buffer(a)
        if name is not None:
          if n_in is None or k < n_in:
            call_inputs.add(name)
          if n_in is None or n_out is None or n_in <= k < n_in + n_out:
            call_outputs.add(name)
  if aliases is not None:
    alias_map = aliases

    def resolve(names: set[str]) -> frozenset[str]:
      return frozenset(_resolve_alias(name, alias_map) for name in names)

    return BufferRefs(resolve(loads), resolve(stores), resolve(call_inputs), resolve(call_outputs))
  return BufferRefs(frozenset(loads), frozenset(stores), frozenset(call_inputs), frozenset(call_outputs))


def _private_decls(body: list[ProgramNode]) -> dict[str, ProgramNode]:
  """name -> BUFFER decl for every ``private`` buffer declared in ``body`` (skips constants)."""
  return {s.attrs["name"]: s for s in body if s.op == ProgramOp.BUFFER and s.attrs.get("address_space") == "private"}


def _alias_sources(body: Iterable[ProgramNode]) -> dict[str, str]:
  """name -> aliased-source name for every alias BUFFER (zero-copy pointer) in ``body``."""
  return {s.attrs["name"]: s.attrs["alias_of"] for s in body if s.op == ProgramOp.BUFFER and "alias_of" in s.attrs}


_INT_BINOP = {
  ProgramOp.ADD: np.add,
  ProgramOp.SUB: np.subtract,
  ProgramOp.MUL: np.multiply,
  ProgramOp.DIV: np.floor_divide,
  ProgramOp.MOD: np.mod,
}


def _index_values(idx: ProgramNode, rng: ProgramNode, decls: dict[str, ProgramNode]) -> np.ndarray | None:
  """The values index expression ``idx`` takes over the trip of ``rng``, or None if it is not
  statically known non-negative arithmetic over the range variable and constant tables.

  The counterpart of ``LowerCtx.index_at``: a pass that needs the concrete indices of a gather or
  scatter reads them back from the expression, whether or not a table was materialized. A negative
  dividend is refused because C division truncates where numpy floors."""
  start, stop, step = rng.args
  if any(a.op != ProgramOp.CONST_INT for a in rng.args) or int(step.attrs["value"]) <= 0:
    return None
  trip = np.arange(int(start.attrs["value"]), int(stop.attrs["value"]), int(step.attrs["value"]), dtype=np.int64)
  return _values(idx, rng.attrs["name"], trip, decls)


def _values(idx: ProgramNode, var: str, trip: np.ndarray, decls: dict[str, ProgramNode]) -> np.ndarray | None:
  if idx.op == ProgramOp.CONST_INT:
    return np.full(len(trip), int(idx.attrs["value"]), dtype=np.int64)
  if idx.op == ProgramOp.VAR:
    return trip if idx.attrs["name"] == var else None
  if idx.op == ProgramOp.LOAD:
    view = idx.args[0]
    decl = decls.get(view.attrs["buffer"])
    values = decl.attrs.get("values") if decl is not None else None
    if values is None or len(view.args) != 1:
      return None
    inner = _values(view.args[0], var, trip, decls)
    if inner is None or inner.min() < 0 or inner.max() >= len(values):
      return None
    return np.asarray(values, dtype=np.int64)[inner]
  binop = _INT_BINOP.get(idx.op)
  if binop is None or len(idx.args) != 2:
    return None
  left, right = (_values(a, var, trip, decls) for a in idx.args)
  if left is None or right is None:
    return None
  if idx.op in (ProgramOp.DIV, ProgramOp.MOD) and (left.min() < 0 or right.min() < 1):
    return None
  return binop(left, right)


def _resolve_alias(name: str, alias_src: dict[str, str]) -> str:
  """Follow an alias chain to the buffer that actually owns storage."""
  seen: set[str] = set()
  while name in alias_src and name not in seen:
    seen.add(name)
    name = alias_src[name]
  return name


def allocated_name(base: str, spellings: set[str]) -> str:
  """Reserve ``base`` or a numbered suffix against occupied C identifier spellings."""
  name = base
  suffix = 2
  while c_ident(name) in spellings:
    name = f"{base}_{suffix}"
    suffix += 1
  spellings.add(c_ident(name))
  return name


def prune_dead_buffers(proc: ProgramNode) -> ProgramNode:
  """Drop buffer declarations no remaining statement or live alias references."""
  params, body = _proc_parts(proc)
  referenced: set[str] = set()
  for stmt in body:
    if stmt.op != ProgramOp.BUFFER:
      refs = buffer_refs(stmt)
      referenced.update(refs.reads | refs.writes)
  alias_src = _alias_sources(body)
  while True:
    sources = {src for name, src in alias_src.items() if name in referenced and src not in referenced}
    if not sources:
      break
    referenced.update(sources)
  param_names = {param.attrs["name"] for param in params}
  kept = [stmt for stmt in body if stmt.op != ProgramOp.BUFFER or stmt.attrs["name"] in referenced or stmt.attrs["name"] in param_names]
  return proc if len(kept) == len(body) else _rebuild_proc(proc, params, kept)


def prune_procedures(prog: ProgramNode) -> ProgramNode:
  """Keep procedures reachable from the entry procedure and from the dependencies of extern callees."""
  procs, kernels = _procs(prog)
  if not procs:
    return prog
  table = {proc.attrs["name"]: proc for proc in procs}
  roots = {procs[-1].attrs["name"]}
  roots.update(name for names in prog.attrs.get("extern_deps", {}).values() for name in names)
  roots.update(node.attrs["callee"] for kernel in kernels for node in _walk(kernel) if node.op == ProgramOp.CALL)
  reachable: set[str] = set()
  pending = list(roots)
  while pending:
    name = pending.pop()
    if name in reachable or name not in table:
      continue
    reachable.add(name)
    pending.extend(n.attrs["callee"] for n in _walk(table[name]) if n.op == ProgramOp.CALL)
  kept = [proc for proc in procs if proc.attrs["name"] in reachable]
  if len(kept) == len(procs):
    return prog
  return ProgramNode(ProgramOp.PROGRAM, (*kept, *kernels), {**prog.attrs, "proc_count": len(kept)}, prog.dtype)
