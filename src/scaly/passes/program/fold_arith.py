"""Arithmetic cleanup of loop bodies: constant-buffer reads become constants, index arithmetic on a constant table becomes a table, then the shared identities fold."""

from __future__ import annotations

import numpy as np

from ...ir import program as p
from ...ir.match import Pattern, rewrite
from ...ir.program import ProgramNode, ProgramOp, _attr_key
from ...ir.types import dtypes
from ...utils.names import c_ident
from ..arith import CONSTANTS, constant, fold_program
from ._common import (
  _alias_sources,
  _walk,
  allocated_name,
  _map_procs,
  _private_decls,
  _proc_parts,
  _rebuild_proc,
  _size_of,
  buffer_refs,
  inline_producer as _as_inline_producer,
  prune_dead_buffers,
  rebuild_program,
  trip_count as _trip_count,
)


def fold_arith(prog: ProgramNode) -> ProgramNode:
  """Fold arithmetic inside every procedure body, including ``FOR`` bodies.

  Runs after fusion, whose index substitution is what exposes ``x * 1``, ``x + 0`` and reads of a
  uniform constant buffer (a broadcast scalar constant) inside loop bodies. A loop that fills a
  private buffer with one constant becomes a constant buffer, so its readers fold on the next round.
  Fusion also leaves index arithmetic on an index table's entry, where a gather reads a moved array
  (``t[k[i] / 4 + (k[i] % 4) * n]``, a transpose under the gather): that is a table too, computed
  here once instead of a division per element at run time (``_fold_tables``).
  """
  return _map_procs(prog, _fold_proc)


def _fold_proc(proc: ProgramNode) -> ProgramNode:
  params, body = _proc_parts(proc)
  changed = False
  while True:
    new_body = _constant_fills(_fold_tables(_fold_body(body), {param.attrs["name"] for param in params}))
    if len(new_body) == len(body) and all(a is b for a, b in zip(new_body, body, strict=True)):
      break
    body, changed = new_body, True
  rebuilt = _rebuild_proc(proc, params, body) if changed else proc
  return prune_dead_buffers(rebuilt)


def _fold_body(body: list[ProgramNode]) -> list[ProgramNode]:
  constants = {s.attrs["name"]: s.attrs["values"] for s in body if s.op == ProgramOp.BUFFER and "values" in s.attrs}

  def constant_read(n: ProgramNode) -> bool:
    view = n.args[0]
    values = constants.get(view.attrs["buffer"])
    if not values or len(view.args) != 1:
      return False
    index = view.args[0]
    if index.op == ProgramOp.CONST_INT:  # a constant index outside the table (in code that never runs) stays a load
      return 0 <= index.attrs["value"] < len(values)
    first = _attr_key(values[0])  # by bits, as interning keys a constant: -0.0 is not 0.0
    return all(_attr_key(v) == first for v in values)

  def read(n: ProgramNode) -> ProgramNode:
    index = n.args[0].args[0]
    values = constants[n.args[0].attrs["buffer"]]
    return constant(values[index.attrs["value"] if index.op == ProgramOp.CONST_INT else 0], n.dtype)

  patterns = [Pattern(ProgramOp.LOAD, constant_read, read), Pattern(None, lambda n: bool(n.args), fold_program)]
  return [rewrite(stmt, patterns, rebuild=rebuild_program) for stmt in body]


_TABLE_ARITH = frozenset({ProgramOp.ADD, ProgramOp.SUB, ProgramOp.MUL, ProgramOp.DIV, ProgramOp.MOD, ProgramOp.NEG})


def _table_function(n: ProgramNode, tables: dict[str, np.ndarray]) -> tuple[ProgramNode | None, np.ndarray] | None:
  """``n`` as a table over an index: ``(index, values)`` when ``n`` is integer arithmetic on
  constants and on entries of constant integer tables all read at one index expression (``index``
  is None for a constant alone), or a read of such a table at such a function, else None. A
  division or remainder is taken only of a non-negative entry by a positive one, where C's and
  NumPy's agree."""
  if n.op == ProgramOp.CONST_INT:
    return None, np.asarray(n.attrs["value"], dtype=np.int64)
  if n.op == ProgramOp.LOAD:
    view = n.args[0]
    values = tables.get(view.attrs["buffer"])
    if values is None or len(view.args) != 1:
      return None
    inner = _table_function(view.args[0], tables)
    if inner is None or inner[0] is None:
      return view.args[0], values
    at = inner[1]  # a table read at a table's entry: the two composed
    return (inner[0], values[at]) if at.size and at.min() >= 0 and at.max() < values.size else None
  if n.op not in _TABLE_ARITH or not n.dtype.is_integer:
    return None
  parts = []
  for arg in n.args:
    part = _table_function(arg, tables)
    if part is None:
      return None
    parts.append(part)
  indices = {id(at): at for at, _ in parts if at is not None}
  if len(indices) > 1 or len({values.size for at, values in parts if at is not None}) > 1:
    return None
  index = next(iter(indices.values()), None)
  values = [values for _, values in parts]
  if n.op == ProgramOp.NEG:
    return index, -values[0]
  x, y = values
  if n.op in (ProgramOp.DIV, ProgramOp.MOD):
    if np.any(x < 0) or np.any(y <= 0):
      return None
    return index, x // y if n.op == ProgramOp.DIV else x % y
  return index, x + y if n.op == ProgramOp.ADD else x - y if n.op == ProgramOp.SUB else x * y


def _fold_tables(body: list[ProgramNode], reserved: set[str]) -> list[ProgramNode]:
  """Replace index arithmetic with a division or remainder of a constant table's entry by a read of
  one table holding its values (``_table_function``), and a constant table read at such an index
  by one holding the values picked. The tables left without a reader are pruned by the caller."""
  tables = {
    s.attrs["name"]: np.asarray(s.attrs["values"], dtype=np.int64)
    for s in body
    if s.op == ProgramOp.BUFFER and "values" in s.attrs and s.dtype.is_integer and len(s.attrs["shape"]) == 1
  }
  if not tables:
    return body
  constants = {s.attrs["name"]: s for s in body if s.op == ProgramOp.BUFFER and "values" in s.attrs}

  def candidate(stmt: ProgramNode) -> bool:
    """Whether ``stmt`` holds anything the rewrite could fire on, in one walk: an integer division
    or remainder, or a constant read at a table's entry. Most statements hold neither."""
    for n in _walk(stmt):
      if n.op in (ProgramOp.DIV, ProgramOp.MOD) and n.dtype.is_integer:
        return True
      if (
        n.op == ProgramOp.LOAD
        and n.args[0].attrs["buffer"] in constants
        and any(a.op == ProgramOp.LOAD and a.args[0].attrs["buffer"] in tables for a in n.args[0].args)
      ):
        return True
    return False

  todo = [stmt.op != ProgramOp.BUFFER and candidate(stmt) for stmt in body]
  if not any(todo):
    return body
  # Every name the procedure spells, so that a new table's cannot meet one: parameters, buffers,
  # loop variables and assigned locals.
  spellings = {c_ident(name) for name in reserved} | {c_ident(s.attrs["name"]) for s in body if s.op == ProgramOp.BUFFER}
  for stmt in body:
    for n in _walk(stmt):
      if n.op == ProgramOp.VAR:
        spellings.add(c_ident(n.attrs["name"]))
      elif n.op == ProgramOp.ASSIGN:
        spellings.add(c_ident(n.attrs["target"]))
      elif n.op == ProgramOp.FOR:
        spellings.add(c_ident(n.args[0].attrs["name"]))
  made: dict[bytes, ProgramNode] = {}
  picks: dict[tuple, ProgramNode] = {}
  derived: set[str] = set()

  def divides(n: ProgramNode) -> bool:
    """Whether ``n`` holds what a table saves at run time: an integer division or remainder, or a
    read of a table this pass made of one (so that the whole expression around it becomes one
    table). Sums and products of a table's entry, and a table read at a table's entry, stay as
    they are: a derived table for each costs more memory than the arithmetic costs time."""
    return any(m.op in (ProgramOp.DIV, ProgramOp.MOD) or (m.op == ProgramOp.VIEW and m.attrs["buffer"] in derived) for m in _walk(n))

  def composed(n: ProgramNode) -> bool:
    if n.op not in _TABLE_ARITH or not n.dtype.is_integer or not divides(n):
      return False
    found = _table_function(n, tables)
    return found is not None and found[0] is not None

  def value_read(n: ProgramNode) -> bool:
    """A constant table of another type (the values a gather picks) read at an index table's entry,
    when the values picked are no more than the table held: the gather is done here."""
    decl = constants.get(n.args[0].attrs["buffer"])
    if decl is None or decl.attrs["name"] in tables or len(n.args[0].args) != 1:
      return False
    at = _table_function(n.args[0].args[0], tables)
    if at is None or at[0] is None or not at[1].size or at[1].size > len(decl.attrs["values"]):
      return False
    return bool(at[1].min() >= 0 and at[1].max() < len(decl.attrs["values"]))

  def picked(n: ProgramNode) -> ProgramNode:
    decl = constants[n.args[0].attrs["buffer"]]
    found = _table_function(n.args[0].args[0], tables)
    assert found is not None and found[0] is not None
    values = [decl.attrs["values"][int(k)] for k in found[1]]
    key = (decl.dtype.name, tuple(_attr_key(v) for v in values))
    if key not in picks:
      picks[key] = p.const_buffer(allocated_name("k", spellings), decl.dtype, (len(values),), values)
      constants[picks[key].attrs["name"]] = picks[key]
    return p.load(p.view(picks[key], [found[0]]))

  def table_read(n: ProgramNode) -> ProgramNode:
    found = _table_function(n, tables)
    assert found is not None and found[0] is not None
    index, values = found[0], np.ascontiguousarray(found[1], dtype=np.int64)
    key = values.tobytes()
    if key not in made:
      made[key] = p.const_buffer(allocated_name("k", spellings), dtypes.int64, (values.size,), [int(v) for v in values])
      tables[made[key].attrs["name"]] = values
      derived.add(made[key].attrs["name"])
    return p.load(p.view(made[key], [index]))

  patterns = [Pattern(None, composed, table_read), Pattern(ProgramOp.LOAD, value_read, picked)]
  out = [rewrite(stmt, patterns, rebuild=rebuild_program) if go else stmt for stmt, go in zip(body, todo, strict=True)]
  if not made and not picks:
    return body
  first = next(i for i, stmt in enumerate(out) if stmt.op != ProgramOp.BUFFER)
  return [*out[:first], *made.values(), *picks.values(), *out[first:]]


def _constant_fills(body: list[ProgramNode]) -> list[ProgramNode]:
  """Turn ``for i: buf[i] <- c`` over a whole private, otherwise unwritten buffer into a constant buffer."""
  private = _private_decls(body)
  aliased = set(_alias_sources(body).values())
  writers: dict[str, int] = {}
  for stmt in body:
    if stmt.op != ProgramOp.BUFFER:
      refs = buffer_refs(stmt)
      for name in refs.writes | refs.call_inputs:
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
