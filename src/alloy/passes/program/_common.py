"""Shared Program IR traversal and procedure helpers for optimization passes."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from ...ir.program import ProgramNode, ProgramOp


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
