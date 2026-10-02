"""Split independent scalar ranges into bounded chunks and explicit vector lanes."""

from __future__ import annotations

from typing import Literal

from collections import Counter

from ...ir import program as p
from ...ir.match import Pattern, rewrite
from ...ir.program import walk_program, ProgramNode, ProgramOp, RangeKind
from ...utils.names import c_ident
from ._common import (
  _map_procs,
  _proc_parts,
  _rebuild_proc,
  _resolve_alias,
  _index_values,
  allocated_name,
  buffer_refs,
  rebuild_program,
  substitute_var,
  trip_count,
)


def _minimum(left: ProgramNode, right: ProgramNode) -> ProgramNode:
  return ProgramNode(ProgramOp.MINIMUM, (left, right), dtype=left.dtype)


def _peak_live(body: list[ProgramNode]) -> int:
  def flatten(statements: list[ProgramNode]) -> list[ProgramNode]:
    return [node for stmt in statements for node in (flatten(list(stmt.args[1:])) if stmt.op == ProgramOp.FOR else [stmt])]

  body = flatten(body)
  definitions = {n.attrs["target"]: i for i, n in enumerate(body) if n.op == ProgramOp.ASSIGN and n.dtype.is_floating}
  last = dict(definitions)
  for i, stmt in enumerate(body):
    for node in walk_program(stmt):
      if node.op == ProgramOp.VAR and node.attrs["name"] in definitions:
        last[node.attrs["name"]] = i
  events: dict[int, int] = {}
  for name, first in definitions.items():
    events[first] = events.get(first, 0) + 1
    events[last[name] + 1] = events.get(last[name] + 1, 0) - 1
  live = peak = 0
  for i in sorted(events):
    live += events[i]
    peak = max(peak, live)
  return max(1, peak)


def widen_ranges(prog: ProgramNode, *, lanes: Literal["auto"] | Literal[1, 2, 4, 8] = "auto") -> ProgramNode:
  """Widen independent mapped or contiguous elementwise ranges with a fixed or target-selected width."""
  if lanes not in ("auto", 1, 2, 4, 8):
    raise ValueError("lanes must be 'auto', 1, 2, 4 or 8")

  procedures = {n.attrs["name"]: n for n in prog.args if n.op == ProgramOp.PROC}
  global_spellings = {c_ident(n.attrs[k]) for n in walk_program(prog) for k in ("name", "target", "callee") if k in n.attrs}
  global_spellings.update(f"{c_ident(name)}_raw" for name in procedures)

  def widen_proc(proc: ProgramNode) -> ProgramNode:
    params, body = _proc_parts(proc)
    spellings = {c_ident(n.attrs[k]) for n in walk_program(proc) for k in ("name", "target") if k in n.attrs}
    ordinal = 0
    promoted: list[ProgramNode] = []
    declarations = {n.attrs["name"]: n for n in walk_program(proc) if n.op == ProgramOp.BUFFER}

    def transform(stmt: ProgramNode) -> ProgramNode:
      nonlocal ordinal
      if stmt.op != ProgramOp.FOR:
        return stmt
      rng, *inner = stmt.args

      def recurse() -> ProgramNode:
        return ProgramNode(stmt.op, (rng, *(transform(child) for child in stmt.args[1:])), stmt.attrs, stmt.dtype)

      if rng.attrs.get("mapped"):
        inner = _inline_calls(inner, procedures, spellings)
      count = trip_count(rng)
      refs = buffer_refs(stmt)
      if count is None or count < 2 or rng.args[2].attrs.get("value") != 1:
        return recurse()
      statements = [n for n in walk_program(p.block(*inner)) if n.op in (ProgramOp.CALL, ProgramOp.LAUNCH, ProgramOp.FOR)]
      if any(n.op != ProgramOp.FOR or any(a.op != ProgramOp.CONST_INT for a in n.args[0].args) for n in statements):
        return recurse()
      if any(n.dtype.is_floating and n.dtype.bits != 64 for n in walk_program(p.block(*inner))):
        return recurse()
      if any(
        n.op in (ProgramOp.STORE, ProgramOp.STORE_PAIR) and (not n.dtype.is_floating or n.dtype.bits != 64) for n in walk_program(p.block(*inner))
      ):
        return recurse()
      assigned = [n for n in walk_program(p.block(*inner)) if n.op == ProgramOp.ASSIGN]
      declared = {n.attrs["target"] for n in assigned if n.attrs.get("declare")}
      if any(n.attrs["target"] not in declared for n in assigned):
        return recurse()
      local_buffers = {n.attrs["name"]: n for n in walk_program(p.block(*inner)) if n.op == ProgramOp.BUFFER and "values" not in n.attrs}
      local_names = set(local_buffers)
      local_owners: dict[str, tuple[str, int]] = {}
      for name in local_names:
        owner, offset, seen = name, 0, set()
        while owner in local_buffers and "alias_of" in local_buffers[owner].attrs and owner not in seen:
          seen.add(owner)
          attrs = local_buffers[owner].attrs
          offset += attrs["alias_offset"]
          owner = attrs["alias_of"]
        if owner not in local_buffers or owner in seen:
          return recurse()
        local_owners[name] = owner, offset
      aliases = {
        n.attrs["name"]: n.attrs["alias_of"] for n in walk_program(p.block(*body, *inner)) if n.op == ProgramOp.BUFFER and "alias_of" in n.attrs
      }
      refs = buffer_refs(p.block(*inner), aliases)
      index_writes = Counter(n.attrs["target"] for n in assigned if n.dtype.is_integer)
      if any(count != 1 for count in index_writes.values()) or any(n.dtype.is_integer and not n.attrs.get("declare") for n in assigned):
        return recurse()
      index_definitions = {n.attrs["target"]: n.args[0] for n in assigned if n.dtype.is_integer}
      varying_indices = set(index_writes)
      reduction = _ordered_reduction(inner, rng, aliases)
      independent = _contiguous_outputs(inner, rng, index_definitions, varying_indices) and not any(
        n.op == ProgramOp.VIEW and n.attrs["buffer"] in aliases for n in walk_program(p.block(*inner))
      )
      if ((refs.reads & refs.writes) - local_names and reduction is None and not independent) or refs.call_args:
        return recurse()
      if reduction is not None:
        inner = [reduction]
      if not rng.attrs.get("mapped") and reduction is None and not independent:
        return recurse()
      inner = _interleave_stores(_split_pairs(inner), aliases)
      peak = _peak_live(inner)
      trip_cap = 1 << (count - 1).bit_length()
      caps = tuple(max(w for w in (1, 2, 4, 8) if w <= trip_cap and (peak * w <= 4 * slots or w == 1)) for slots in (16, 32, 64, 256))
      cap = caps[-1]
      ordinal += 1
      helper = allocated_name(f"{c_ident(proc.attrs['name'])}_lanes_{ordinal}", global_spellings)
      prefix = helper
      while any(name.startswith(prefix + "_") for name in spellings | global_spellings):
        prefix += "_local"
      width_macro = allocated_name(f"SCALY_WIDTH_{helper}", global_spellings)
      width = p.var(width_macro)
      outer_name = allocated_name(f"{rng.attrs['name']}_chunk", spellings)
      lane_name = allocated_name(f"{rng.attrs['name']}_lane", spellings)
      outer, lane = p.var(outer_name), p.var(lane_name)
      offset = p.mul(outer, width)
      index = p.add(rng.args[0], _minimum(p.add(offset, lane), p.const_int(count - 1)))
      valid = _minimum(width, p.sub(p.const_int(count), offset))
      lane_range = p.range_(lane_name, 0, valid, kind=RangeKind.VECTOR)
      lane_range = ProgramNode(
        lane_range.op,
        lane_range.args,
        {**lane_range.attrs, "lanes": lanes, "lane_cap": cap, "lane_caps": caps, "lane_width": width_macro, "peak_live": peak},
        lane_range.dtype,
      )

      def access(view: ProgramNode) -> ProgramNode:
        expanded = _expand_index(view.args[0], index_definitions) if len(view.args) == 1 else None
        values = _index_values(expanded, rng, declarations) if expanded is not None else None
        stride = _axis_stride(expanded, rng.attrs["name"], varying_indices) if expanded is not None else None
        if values is not None and len(values) > 1 and all(values[j] - values[j - 1] == values[1] - values[0] for j in range(2, len(values))):
          stride = int(values[1] - values[0])
        attrs = {**view.attrs, "lane_layout": "contiguous" if stride == 1 else "strided" if stride is not None else "gather", "lane_storage": 8}
        if stride is not None:
          attrs["lane_stride"] = stride
        return ProgramNode(view.op, view.args, attrs, view.dtype)

      def local_buffer(node: ProgramNode) -> ProgramNode:
        size = 1
        for dim in node.attrs["shape"]:
          size *= dim
        attrs = {**node.attrs, "shape": (size * 8,), "lane_storage": 8}
        return ProgramNode(node.op, node.args, attrs, node.dtype)

      def local_view(node: ProgramNode) -> ProgramNode:
        owner, offset = local_owners[node.attrs["buffer"]]
        scalar_index = p.add(node.args[0], p.const_int(offset)) if offset else node.args[0]
        inner_index = p.add(p.mul(scalar_index, width), lane)
        return ProgramNode(node.op, (inner_index,), {**node.attrs, "buffer": owner, "lane_stride": 1, "lane_local": True}, node.dtype)

      accesses = Pattern(ProgramOp.VIEW, lambda _n: True, access)
      inner = [rewrite(n, [accesses], rebuild=rebuild_program, fixpoint=False) for n in inner]
      storage = [
        Pattern(ProgramOp.BUFFER, lambda n: n.attrs["name"] in local_names, local_buffer),
        Pattern(ProgramOp.VIEW, lambda n: n.attrs["buffer"] in local_names, local_view),
      ]
      inner = [rewrite(n, storage, rebuild=rebuild_program, fixpoint=False) for n in inner]

      def promote(statements: list[ProgramNode]) -> list[ProgramNode]:
        kept: list[ProgramNode] = []
        for node in statements:
          if node.op == ProgramOp.BUFFER and node.attrs["name"] in local_names:
            if "alias_of" not in node.attrs:
              promoted.append(node)
          elif node.op == ProgramOp.FOR:
            children = promote(list(node.args[1:]))
            kept.append(ProgramNode(node.op, (node.args[0], *children), {**node.attrs, "body_len": len(children)}, node.dtype))
          else:
            kept.append(node)
        return kept

      inner = promote(inner)
      vector = p.for_(lane_range, [substitute_var(n, rng.attrs["name"], index) for n in inner])
      chunks = p.div(p.add(p.const_int(count), p.sub(width, p.const_int(1))), width)
      outer_range = p.range_(outer_name, 0, chunks, kind=RangeKind.SERIAL)
      result = p.for_(outer_range, [vector])
      return ProgramNode(
        result.op,
        result.args,
        {**result.attrs, "vector_helper": helper, "vector_prefix": prefix, "vector_count": count, "vector_mapped": bool(rng.attrs.get("mapped"))},
        result.dtype,
      )

    transformed = [transform(stmt) for stmt in body]
    return _rebuild_proc(proc, params, [*promoted, *transformed])

  return _map_procs(prog, widen_proc)


def _inline_calls(body: list[ProgramNode], procedures: dict[str, ProgramNode], spellings: set[str]) -> list[ProgramNode]:
  out: list[ProgramNode] = []
  for stmt in body:
    if stmt.op != ProgramOp.CALL or stmt.attrs["callee"] not in procedures:
      out.append(stmt)
      continue
    callee = procedures[stmt.attrs["callee"]]
    params, statements = _proc_parts(callee)
    if any(n.op in (ProgramOp.CALL, ProgramOp.LAUNCH) for n in walk_program(p.block(*statements))):
      out.append(stmt)
      continue
    arguments = dict(zip((n.attrs["name"] for n in params), stmt.args, strict=True))
    renames: dict[str, str] = {}
    aliases = [n for n in statements if n.op == ProgramOp.BUFFER and "alias_of" in n.attrs]
    for alias in aliases:
      source = arguments.get(alias.attrs["alias_of"])
      if source is None:
        continue
      buffer = source.attrs["buffer"] if source.op == ProgramOp.VIEW else source.attrs["name"]
      offset = source.args[0] if source.op == ProgramOp.VIEW else p.const_int(0)
      arguments[alias.attrs["name"]] = ProgramNode(
        ProgramOp.VIEW, (p.add(offset, p.const_int(alias.attrs["alias_offset"])),), {"buffer": buffer}, alias.dtype
      )
    statements = [n for n in statements if not (n.op == ProgramOp.BUFFER and n.attrs["name"] in arguments)]
    for node in walk_program(p.block(*statements)):
      key = "target" if node.op == ProgramOp.ASSIGN else "name" if node.op in (ProgramOp.BUFFER, ProgramOp.RANGE) else None
      if key is not None and node.attrs[key] not in renames:
        renames[node.attrs[key]] = allocated_name(f"lane_{node.attrs[key]}", spellings)

    def replace(node: ProgramNode) -> ProgramNode:
      attrs = dict(node.attrs)
      args = node.args
      if node.op == ProgramOp.VIEW:
        name = attrs["buffer"]
        if name in arguments:
          arg = arguments[name]
          attrs["buffer"] = arg.attrs["buffer"] if arg.op == ProgramOp.VIEW else arg.attrs["name"]
          if arg.op == ProgramOp.VIEW:
            args = (p.add(arg.args[0], args[0]),)
        elif name in renames:
          attrs["buffer"] = renames[name]
      for key in ("name", "target", "alias_of"):
        if key in attrs and attrs[key] in renames:
          attrs[key] = renames[attrs[key]]
      return ProgramNode(node.op, args, attrs, node.dtype)

    pattern = Pattern(None, lambda _n: True, replace)
    out.extend(rewrite(n, [pattern], rebuild=rebuild_program, fixpoint=False) for n in statements)
  return out


def _ordered_reduction(body: list[ProgramNode], rng: ProgramNode, aliases: dict[str, str]) -> ProgramNode | None:
  if rng.attrs["kind"] != RangeKind.REDUCE or len(body) != 1 or body[0].op != ProgramOp.STORE:
    return None
  stmt = body[0]
  target, value = stmt.args
  if value.op != ProgramOp.ADD or any(n.op == ProgramOp.VAR and n.attrs["name"] == rng.attrs["name"] for n in walk_program(target)):
    return None
  if not any(arg.op == ProgramOp.LOAD and arg.args[0] is target for arg in value.args):
    return None
  owner = _resolve_alias(target.attrs["buffer"], aliases)
  if any(
    n.op == ProgramOp.LOAD and n.args[0] is not target and _resolve_alias(n.args[0].attrs["buffer"], aliases) == owner for n in walk_program(value)
  ):
    return None
  return ProgramNode(stmt.op, stmt.args, {**stmt.attrs, "ordered_reduction": True}, stmt.dtype)


def _axis_stride(index: ProgramNode, axis: str, varying: set[str]) -> int | None:
  if index.op == ProgramOp.CONST_INT:
    return 0
  if index.op == ProgramOp.VAR:
    return 1 if index.attrs["name"] == axis else None if index.attrs["name"] in varying else 0
  if index.op in (ProgramOp.ADD, ProgramOp.SUB):
    left, right = (_axis_stride(arg, axis, varying) for arg in index.args)
    return None if left is None or right is None else left + (right if index.op == ProgramOp.ADD else -right)
  if index.op == ProgramOp.MUL:
    left, right = index.args
    if left.op == ProgramOp.CONST_INT:
      stride = _axis_stride(right, axis, varying)
      return None if stride is None else left.attrs["value"] * stride
    if right.op == ProgramOp.CONST_INT:
      stride = _axis_stride(left, axis, varying)
      return None if stride is None else right.attrs["value"] * stride
  return None


def _contiguous_outputs(body: list[ProgramNode], rng: ProgramNode, definitions: dict[str, ProgramNode], varying: set[str]) -> bool:
  nodes = list(walk_program(p.block(*body)))
  if any(n.op == ProgramOp.STORE_PAIR for n in nodes):
    return False
  stores = [n.args[0] for n in nodes if n.op == ProgramOp.STORE]
  if any(left.attrs["buffer"] == right.attrs["buffer"] and left is not right for left in stores for right in stores):
    return False
  if not stores or any(
    len(view.args) != 1 or _axis_stride(_expand_index(view.args[0], definitions), rng.attrs["name"], varying) != 1 for view in stores
  ):
    return False
  written = {view.attrs["buffer"] for view in stores}
  return all(n.args[0] in stores for n in nodes if n.op == ProgramOp.LOAD and n.args[0].attrs["buffer"] in written)


def _split_pairs(body: list[ProgramNode]) -> list[ProgramNode]:
  out: list[ProgramNode] = []
  for stmt in body:
    if stmt.op == ProgramOp.FOR:
      children = _split_pairs(list(stmt.args[1:]))
      out.append(ProgramNode(stmt.op, (stmt.args[0], *children), {**stmt.attrs, "body_len": len(children)}, stmt.dtype))
    elif stmt.op == ProgramOp.STORE_PAIR:
      view = stmt.args[0]
      second = ProgramNode(view.op, (p.add(view.args[0], p.const_int(1)),), view.attrs, view.dtype)
      out.extend((p.store(view, stmt.args[1]), p.store(second, stmt.args[2])))
    else:
      out.append(stmt)
  return out


def _interleave_stores(body: list[ProgramNode], aliases: dict[str, str]) -> list[ProgramNode]:
  if any(n.op not in (ProgramOp.ASSIGN, ProgramOp.STORE) for n in body):
    return body
  definitions = {n.attrs["target"]: i for i, n in enumerate(body) if n.op == ProgramOp.ASSIGN}
  assignments = [n for n in body if n.op == ProgramOp.ASSIGN]
  if len(definitions) != len(assignments) or any(not n.attrs.get("declare") for n in assignments):
    return body
  refs = buffer_refs(p.block(*body), aliases)
  if refs.reads & refs.writes:
    return body
  first_store = next((i for i, n in enumerate(body) if n.op == ProgramOp.STORE), len(body))
  if any(n.op == ProgramOp.ASSIGN for n in body[first_store:]):
    return body
  dependencies = {
    i: sorted({definitions[n.attrs["name"]] for n in walk_program(stmt) if n.op == ProgramOp.VAR and n.attrs["name"] in definitions})
    for i, stmt in enumerate(body)
  }
  emitted: set[int] = set()
  out: list[ProgramNode] = []

  def emit(index: int) -> None:
    stack = [(index, False)]
    while stack:
      current, ready = stack.pop()
      if current in emitted:
        continue
      if ready:
        out.append(body[current])
        emitted.add(current)
      else:
        stack.append((current, True))
        stack.extend((dependency, False) for dependency in reversed(dependencies[current]) if dependency not in emitted)

  for i, stmt in enumerate(body[:first_store]):
    if any(n.op == ProgramOp.LOAD for n in walk_program(stmt)):
      emit(i)
  for i in range(first_store, len(body)):
    emit(i)
  for i in range(first_store):
    emit(i)
  return out


def _expand_index(index: ProgramNode, definitions: dict[str, ProgramNode]) -> ProgramNode:
  pattern = Pattern(ProgramOp.VAR, lambda n: n.attrs["name"] in definitions, lambda n: definitions[n.attrs["name"]])
  return rewrite(index, [pattern], rebuild=rebuild_program, fixpoint=False, revisit=True)
