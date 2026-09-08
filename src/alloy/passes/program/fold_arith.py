"""Arithmetic cleanup of loop bodies: constant-buffer reads become constants, then the shared identities fold."""

from __future__ import annotations

from ...ir import program as p
from ...ir.match import Pattern, rewrite
from ...ir.program import ProgramNode, ProgramOp
from ..arith import CONSTANTS, constant, fold_program
from ._common import _alias_sources, _map_procs, _private_decls, _proc_parts, _rebuild_proc, _size_of, _stmt_refs, rebuild_program
from .fuse_elementwise import _as_inline_producer, _prune_dead_buffers, _trip_count


def fold_arith(prog: ProgramNode) -> ProgramNode:
  """Fold arithmetic inside every procedure body, including ``FOR`` bodies.

  Runs after fusion, whose index substitution is what exposes ``x * 1``, ``x + 0`` and reads of a
  uniform constant buffer (a broadcast scalar constant) inside loop bodies. A loop that fills a
  private buffer with one constant becomes a constant buffer, so its readers fold on the next round.
  """
  return _map_procs(prog, _fold_proc)


def _fold_proc(proc: ProgramNode) -> ProgramNode:
  params, body = _proc_parts(proc)
  changed = False
  while True:
    new_body = _constant_fills(_fold_body(body))
    if all(a is b for a, b in zip(new_body, body, strict=True)):
      break
    body, changed = new_body, True
  return _prune_dead_buffers(_rebuild_proc(proc, params, body)) if changed else proc


def _fold_body(body: list[ProgramNode]) -> list[ProgramNode]:
  constants = {s.attrs["name"]: s.attrs["values"] for s in body if s.op == ProgramOp.BUFFER and "values" in s.attrs}

  def constant_read(n: ProgramNode) -> bool:
    view = n.args[0]
    values = constants.get(view.attrs["buffer"])
    if not values or len(view.args) != 1:
      return False
    index = view.args[0]
    return (index.op == ProgramOp.CONST_INT and 0 <= index.attrs["value"] < len(values)) or all(v == values[0] for v in values)

  def read(n: ProgramNode) -> ProgramNode:
    index = n.args[0].args[0]
    values = constants[n.args[0].attrs["buffer"]]
    return constant(values[index.attrs["value"] if index.op == ProgramOp.CONST_INT else 0], n.dtype)

  patterns = [Pattern(ProgramOp.LOAD, constant_read, read), Pattern(None, lambda n: bool(n.args), fold_program)]
  return [rewrite(stmt, patterns, rebuild=rebuild_program) for stmt in body]


def _constant_fills(body: list[ProgramNode]) -> list[ProgramNode]:
  """Turn ``for i: buf[i] <- c`` over a whole private, otherwise unwritten buffer into a constant buffer."""
  private = _private_decls(body)
  aliased = set(_alias_sources(body).values())
  writers: dict[str, int] = {}
  for stmt in body:
    if stmt.op != ProgramOp.BUFFER:
      _, stores, call_args = _stmt_refs(stmt)
      for name in stores | call_args:
        writers[name] = writers.get(name, 0) + 1
  fills: dict[str, ProgramNode] = {}
  for stmt in body:
    producer = _as_inline_producer(stmt)
    if producer is None:
      continue
    name, _, rhs = producer
    decl = private.get(name)
    if decl is None or name in aliased or writers.get(name) != 1 or rhs.op not in CONSTANTS or rhs.dtype != decl.dtype or "alias_of" in decl.attrs:
      continue
    size = _size_of(decl.attrs["shape"])
    if _trip_count(stmt.args[0]) == size:
      fills[name] = p.const_buffer(name, decl.dtype, decl.attrs["shape"], [rhs.attrs["value"]] * size)
  if not fills:
    return body
  out: list[ProgramNode] = []
  for stmt in body:
    if stmt.op == ProgramOp.BUFFER and stmt.attrs["name"] in fills:
      out.append(fills[stmt.attrs["name"]])
    elif (producer := _as_inline_producer(stmt)) is not None and producer[0] in fills:
      continue
    else:
      out.append(stmt)
  return out
