"""Propagate mapped scalar ranges through static assembly views and schedule surviving stores."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import prod

import numpy as np

from ...ir import program as p
from ...ir.match import Pattern, rewrite
from ...ir.program import walk_program, ProgramNode, ProgramOp
from ...utils.names import c_ident
from ..affine import affine_index_map
from ..arith import constant, fold_program
from ._common import (
  _alias_sources,
  _map_procs,
  _postorder,
  _proc_parts,
  _rebuild_proc,
  buffer_refs,
  prune_dead_buffers,
  rebuild_program,
  substitute_var,
  trip_count,
)
from .scheduling import ScalarNameAllocator, schedule_values


@dataclass(frozen=True)
class _Cell:
  trip: int | None
  value: ProgramNode


def _rewrite(node: ProgramNode, replacements: dict[ProgramNode, ProgramNode], folded: dict[ProgramNode, ProgramNode]) -> ProgramNode:
  replaced = rewrite(node, [Pattern(None, lambda n: n in replacements, lambda n: replacements[n])], rebuild=rebuild_program, fixpoint=False)
  return rewrite(replaced, [Pattern(None, lambda n: bool(n.args), fold_program)], rebuild=rebuild_program, fixpoint=False, memo=folded)


def _inline(call: ProgramNode, callee: ProgramNode) -> list[ProgramNode]:
  params, body = _proc_parts(callee)
  pointers = {}
  for param, arg in zip(params, call.args, strict=True):
    pointers[param.attrs["name"]] = (arg.attrs["name"], p.const_int(0)) if arg.op == ProgramOp.BUFFER else (arg.attrs["buffer"], arg.args[0])
  variables: dict[str, ProgramNode] = {}

  def view(node: ProgramNode) -> ProgramNode:
    name, offset = pointers[node.attrs["buffer"]]
    index = fold_program(p.add(offset, node.args[0]))
    return ProgramNode(node.op, (index,), {**node.attrs, "buffer": name}, node.dtype)

  patterns = [
    Pattern(ProgramOp.VAR, lambda n: n.attrs["name"] in variables, lambda n: variables[n.attrs["name"]]),
    Pattern(ProgramOp.VIEW, lambda n: n.attrs["buffer"] in pointers, view),
  ]
  stores = []
  for stmt in body:
    value = rewrite(stmt, patterns, rebuild=rebuild_program, fixpoint=False)
    if value.op == ProgramOp.ASSIGN:
      variables[value.attrs["target"]] = value.args[0]
    else:
      stores.append(value)
  return stores


def _loops(stmt: ProgramNode) -> tuple[list[ProgramNode], tuple[ProgramNode, ...]] | None:
  ranges = []
  while stmt.op == ProgramOp.FOR:
    rng = stmt.args[0]
    if trip_count(rng) is None or trip_count(rng) == 0 or rng.attrs["kind"] == p.RangeKind.REDUCE:
      return None
    ranges.append(rng)
    body = stmt.args[1:]
    if all(s.op == ProgramOp.STORE for s in body):
      return ranges, body
    if len(body) != 1:
      return None
    stmt = body[0]
  return None


def _indices(node: ProgramNode, variables: dict[str, np.ndarray], constants: dict[str, np.ndarray], count: int) -> np.ndarray | None:
  values: dict[ProgramNode, np.ndarray] = {}
  operations = {ProgramOp.ADD: np.add, ProgramOp.SUB: np.subtract, ProgramOp.MUL: np.multiply, ProgramOp.DIV: np.floor_divide, ProgramOp.MOD: np.mod}
  for n in _postorder(node):
    if n.op == ProgramOp.CONST_INT:
      value = np.full(count, n.attrs["value"], dtype=np.int64)
    elif n.op == ProgramOp.VAR:
      value = variables.get(n.attrs["name"])
      if value is None:
        return None
    elif n.op == ProgramOp.VIEW:
      if len(n.args) != 1:
        return None
      value = values[n.args[0]]
    elif n.op == ProgramOp.LOAD:
      table = constants.get(n.args[0].attrs["buffer"])
      idx = values[n.args[0]]
      if table is None or np.any(idx < 0) or np.any(idx >= len(table)):
        return None
      value = table[idx]
    elif n.op in operations:
      left, right = (values[a] for a in n.args)
      if n.op in {ProgramOp.DIV, ProgramOp.MOD} and (np.any(left < 0) or np.any(right <= 0)):
        return None
      value = operations[n.op](left, right)
    else:
      return None
    values[n] = value
  return values[node]


def _loads(node: ProgramNode) -> list[ProgramNode]:
  loads = []
  seen = set()
  stack = [node]
  while stack:
    n = stack.pop()
    if n in seen:
      continue
    seen.add(n)
    if n.op == ProgramOp.LOAD:
      loads.append(n)
    else:
      stack.extend(n.args)
  return loads


def _fuse(proc: ProgramNode, procs: dict[str, ProgramNode]) -> ProgramNode:
  params, body = _proc_parts(proc)
  aliases = {
    stmt.attrs["name"]: (stmt.attrs["alias_of"], int(stmt.attrs["alias_offset"]))
    for stmt in body
    if stmt.op == ProgramOp.BUFFER and "alias_of" in stmt.attrs
  }

  def resolve_view(node: ProgramNode) -> ProgramNode:
    name = node.attrs["buffer"]
    offset = 0
    while name in aliases:
      name, delta = aliases[name]
      offset += delta
    index = fold_program(p.add(node.args[0], p.const_int(offset)))
    return ProgramNode(node.op, (index,), {**node.attrs, "buffer": name}, node.dtype)

  def resolve_call(node: ProgramNode) -> ProgramNode:
    args = tuple(
      resolve_view(p.view(arg, [p.const_int(0)])) if arg.op == ProgramOp.BUFFER and arg.attrs["name"] in aliases else arg for arg in node.args
    )
    return rebuild_program(node, args)

  if aliases:
    patterns = [
      Pattern(ProgramOp.VIEW, lambda n: n.attrs["buffer"] in aliases and len(n.args) == 1, resolve_view),
      Pattern(ProgramOp.CALL, lambda n: True, resolve_call),
    ]
    body = [rewrite(stmt, patterns, rebuild=rebuild_program, fixpoint=False) for stmt in body]
  seeds: dict[int, ProgramNode] = {}
  seed_offsets: dict[int, int] = {}
  anchor: tuple[str, int, int] | None = None
  for i, stmt in enumerate(body):
    if stmt.op != ProgramOp.FOR or len(stmt.args) != 2 or stmt.args[1].op != ProgramOp.CALL:
      continue
    call = stmt.args[1]
    callee = procs.get(call.attrs["callee"])
    if callee is not None and callee.attrs.get("scalarized"):
      if buffer_refs(call).reads & buffer_refs(call).writes:
        return proc
      seeds[i] = p.for_(stmt.args[0], _inline(call, callee))
      seed_offsets[i] = 0
      rng = stmt.args[0]
      for arg in call.args[: call.attrs["n_in"]]:
        if arg.op != ProgramOp.VIEW or len(arg.args) != 1:
          continue
        offsets = _indices(arg.args[0], {rng.attrs["name"]: np.array([0, 1], dtype=np.int64)}, {}, 2)
        if offsets is None or offsets[0] == offsets[1]:
          continue
        base, stride = int(offsets[0]), int(offsets[1] - offsets[0])
        name = arg.attrs["buffer"]
        if anchor is None:
          anchor = (name, stride, base % abs(stride))
        if name == anchor[0] and stride == anchor[1] and (base - anchor[2]) % stride == 0:
          seed_offsets[i] = (base - anchor[2]) // stride
          break
  if not seeds:
    return proc
  refs = [buffer_refs(stmt, _alias_sources(body)) for stmt in body]
  region = set(seeds)
  produced = set().union(*(buffer_refs(stmt).writes for stmt in seeds.values()))
  private = {stmt.attrs["name"] for stmt in body if stmt.op == ProgramOp.BUFFER and stmt.attrs.get("address_space") == "private"}
  mapped_trips = max(trip_count(seed.args[0]) or 0 for seed in seeds.values())
  while True:
    consumed = set().union(*(refs[i].reads for i in region)) & private
    upstream = {
      i
      for i, ref in enumerate(refs)
      if ref.writes & consumed and (parsed := _loops(body[i])) is not None and prod(trip_count(rng) or 0 for rng in parsed[0]) >= mapped_trips
    }
    consumers = {i for i, ref in enumerate(refs) if ref.reads & produced and body[i].op != ProgramOp.BUFFER}
    extra = (consumers | upstream) - region
    if not extra:
      break
    if any(_loops(body[i]) is None for i in extra):
      return proc
    region.update(extra)
    produced.update(*(refs[i].writes for i in extra))
  for i, ref in enumerate(refs):
    if i not in region and ref.writes & produced:
      parsed = _loops(body[i])
      if parsed is None or any(store.args[1].op not in {ProgramOp.CONST_INT, ProgramOp.CONST_FLOAT} for store in parsed[1]):
        return proc
      region.add(i)
  first, last = min(region), max(region)
  for i in region:
    reads = refs[i].reads - produced
    if any(ref.writes & reads for j, ref in enumerate(refs) if i < j <= last and j not in region):
      return proc
  if any((ref.reads | ref.writes) & produced for i, ref in enumerate(refs) if first <= i <= last and i not in region):
    return proc
  declarations = {n.attrs["name"]: n for n in (*params, *body) if n.op == ProgramOp.BUFFER}
  constant_values = {name: n.attrs["values"] for name, n in declarations.items() if "values" in n.attrs}
  constants = {
    name: np.asarray(n.attrs["values"], dtype=np.int64) for name, n in declarations.items() if "values" in n.attrs and n.dtype == p.dtypes.int64
  }
  occupied = {c_ident(n.attrs["name"]) for n in declarations.values()}
  occupied.update(c_ident(n.attrs["name"]) for n in walk_program(proc) if n.op == ProgramOp.VAR)
  occupied.update(c_ident(n.attrs["name"]) for n in walk_program(proc) if n.op == ProgramOp.RANGE)
  occupied.update(c_ident(n.attrs["target"]) for n in walk_program(proc) if n.op == ProgramOp.ASSIGN)
  names = ScalarNameAllocator(occupied)
  coordinate = names.fresh()
  stage = p.var(coordinate)
  cells: dict[str, dict[int, _Cell]] = {}
  new_constants: list[ProgramNode] = []
  folded: dict[ProgramNode, ProgramNode] = {}

  def index_expr(indices: list[int], trips: list[int]) -> ProgramNode:
    base = indices[0]
    stride = (indices[-1] - base) // (trips[-1] - trips[0]) if len(trips) > 1 else 0
    if all(value == base + stride * (trip - trips[0]) for value, trip in zip(indices, trips, strict=True)):
      return fold_program(p.add(p.const_int(base - stride * trips[0]), fold_program(p.mul(p.const_int(stride), stage))))
    values = [0] * (trips[-1] - trips[0] + 1)
    for trip, value in zip(trips, indices, strict=True):
      values[trip - trips[0]] = value
    mapping = affine_index_map(np.asarray(values, dtype=np.int64))
    index = fold_program(p.sub(stage, p.const_int(trips[0])))
    period = len(mapping.residual)
    if period == 1:
      result = p.const_int(int(mapping.residual[0]))
    else:
      table = p.const_buffer(names.fresh(), p.dtypes.int64, (period,), mapping.residual.tolist())
      new_constants.append(table)
      result = p.load(p.view(table, [index if period == len(values) else p.mod(index, p.const_int(period))]))
    stride = period
    for dim, coeff in zip(reversed(mapping.dims), reversed(mapping.coeffs), strict=True):
      if coeff:
        quotient = index if stride == 1 else p.div(index, p.const_int(stride))
        term = p.mul(p.mod(quotient, p.const_int(dim)), p.const_int(coeff))
        result = fold_program(p.add(result, term))
      stride *= dim
    return result

  for i in sorted(region):
    stmt = seeds.get(i, body[i])
    parsed = _loops(stmt)
    if parsed is None:
      return proc
    ranges, stores = parsed
    written = set().union(*(buffer_refs(store).writes for store in stores))
    if any(buffer_refs(store).reads & (written - buffer_refs(store).writes) for store in stores):
      return proc
    axes = [np.arange(*(int(a.attrs["value"]) for a in rng.args), dtype=np.int64) for rng in ranges]
    grids = np.meshgrid(*axes, indexing="ij")
    variables = {rng.attrs["name"]: grid.reshape(-1) for rng, grid in zip(ranges, grids, strict=True)}
    count = grids[0].size
    for store in stores:
      target, rhs = store.args
      pending = [rhs]
      scalar_vars = set()
      seen = set()
      while pending:
        node = pending.pop()
        if node in seen:
          continue
        seen.add(node)
        if node.op == ProgramOp.VAR:
          scalar_vars.add(node.attrs["name"])
        elif node.op != ProgramOp.LOAD:
          pending.extend(node.args)
      if scalar_vars & {rng.attrs["name"] for rng in ranges}:
        return proc
      read_vars = {node.attrs["name"] for node in walk_program(rhs) if node.op == ProgramOp.VAR}
      if any(
        node.op == ProgramOp.ASSIGN and node.attrs["target"] in read_vars
        for j, stmt in enumerate(body)
        if i < j <= last and j not in region
        for node in walk_program(stmt)
      ):
        return proc
      dest = target.attrs["buffer"]
      if i in seeds and dest not in {param.attrs["name"] for param in params} and not any(dest in ref.reads for ref in refs):
        continue
      if len(target.args) != 1:
        return proc
      output_indices = _indices(target.args[0], variables, constants, count)
      if output_indices is None or len(set(output_indices.tolist())) != count:
        return proc
      output = cells.setdefault(dest, {})
      if rhs.op in {ProgramOp.CONST_INT, ProgramOp.CONST_FLOAT} and i not in seeds:
        for idx in output_indices:
          output[int(idx)] = _Cell(None, rhs)
        continue
      if i in seeds and not any(load.args[0].attrs["buffer"] in cells for load in _loads(rhs)):
        if len(ranges) != 1:
          return proc
        value = substitute_var(rhs, ranges[0].attrs["name"], fold_program(p.sub(stage, p.const_int(seed_offsets[i]))))
        for trip, idx in zip(axes[0], output_indices, strict=True):
          if int(idx) in output:
            return proc
          output[int(idx)] = _Cell(int(trip) + seed_offsets[i], value)
        continue
      loads = _loads(rhs)
      indexed = []
      for load in loads:
        if len(load.args[0].args) != 1:
          return proc
        indices = _indices(load.args[0].args[0], variables, constants, count)
        if indices is None:
          return proc
        if load.args[0].attrs["buffer"] == dest and not np.array_equal(indices, output_indices):
          return proc
        indexed.append(indices)
      mapped = [j for j, load in enumerate(loads) if load.args[0].attrs["buffer"] in cells or load.args[0].attrs["buffer"] in constant_values]
      groups: dict[tuple[ProgramNode, ...], dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
      for position in range(count):
        inputs = []
        for j in mapped:
          name = loads[j].args[0].attrs["buffer"]
          index = int(indexed[j][position])
          inputs.append(_Cell(None, constant(constant_values[name][index], loads[j].dtype)) if name in constant_values else cells[name].get(index))
        if any(cell is None for cell in inputs):
          return proc
        present = [cell for cell in inputs if cell is not None]
        trips = {cell.trip for cell in present if cell.trip is not None and cell.value.op not in {ProgramOp.CONST_INT, ProgramOp.CONST_FLOAT}}
        if not trips:
          trips = set(sorted(cell.trip for cell in present if cell.trip is not None)[:1])
        if len(trips) > 1:
          return proc
        if not trips:
          replacements = dict(zip((loads[j] for j in mapped), (cell.value for cell in present), strict=True))
          for j, load in enumerate(loads):
            if j not in mapped:
              view = ProgramNode(ProgramOp.VIEW, (p.const_int(int(indexed[j][position])),), load.args[0].attrs, load.args[0].dtype)
              replacements[load] = p.load(view)
          value = _rewrite(rhs, replacements, folded)
          output[int(output_indices[position])] = _Cell(None, value)
          continue
        trip = next(iter(trips))
        groups[tuple(cell.value for cell in present)][trip].append(position)
      for values, positions in groups.items():
        for occurrence in range(max(map(len, positions.values()))):
          trips = sorted(trip for trip, ps in positions.items() if len(ps) > occurrence)
          ps = [positions[trip][occurrence] for trip in trips]
          replacements = dict(zip((loads[j] for j in mapped), values, strict=True))
          for j, load in enumerate(loads):
            if j not in mapped:
              idx = index_expr([int(indexed[j][pos]) for pos in ps], trips)
              view = ProgramNode(ProgramOp.VIEW, (idx,), load.args[0].attrs, load.args[0].dtype)
              replacements[load] = p.load(view)
          value = _rewrite(rhs, replacements, folded)
          for trip, pos in zip(trips, ps, strict=True):
            idx = int(output_indices[pos])
            output[idx] = _Cell(trip, value)

  outputs = {param.attrs["name"] for param in params[proc.attrs["input_count"] :]}
  final: list[tuple[str, ProgramNode, list[int], list[int]]] = []
  boundaries: set[int] = set()
  for name in produced & outputs:
    groups: dict[ProgramNode, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for idx, cell in cells.get(name, {}).items():
      if cell.trip is None:
        return proc
      groups[cell.value][cell.trip].append(idx)
    for value, positions in groups.items():
      for occurrence in range(max(map(len, positions.values()))):
        trips = sorted(trip for trip, ps in positions.items() if len(ps) > occurrence)
        indices = [positions[trip][occurrence] for trip in trips]
        final.append((name, value, trips, indices))
        for start, end in zip(trips, trips[1:] + [trips[-1] + 2], strict=True):
          if end != start + 1:
            boundaries.add(start + 1)
            boundaries.add(end)
        boundaries.add(trips[0])
  if len(final) > sum(n.op == ProgramOp.STORE for i in region for n in walk_program(seeds.get(i, body[i]))):
    return proc
  bounds = sorted(boundaries)
  if len(bounds) > 2 * len(region) + 2:
    return proc
  loops = []
  for start, stop in zip(bounds, bounds[1:]):
    stores = []
    for name, value, trips, indices in final:
      if start not in trips:
        continue
      selected = [(trip, idx) for trip, idx in zip(trips, indices, strict=True) if start <= trip < stop]
      if len(selected) != stop - start:
        return proc
      target = p.view(declarations[name], [index_expr([idx for _, idx in selected], [trip for trip, _ in selected])])
      stores.append(p.store(target, value))
    if stores:
      assignments, values = schedule_values([store.args[1] for store in stores], names)
      scheduled = [*assignments, *(p.store(store.args[0], value) for store, value in zip(stores, values, strict=True))]
      rng = p.range_(coordinate, start, stop, kind=p.RangeKind.GLOBAL)
      rng = ProgramNode(rng.op, rng.args, {**rng.attrs, "mapped": True}, rng.dtype)
      loops.append(p.for_(rng, scheduled))
  if buffer_refs(ProgramNode(ProgramOp.BLOCK, tuple(loops))).reads & {node.attrs["name"] for node in new_constants}:
    return proc
  replacement = loops
  result = [stmt for i, stmt in enumerate(body) if i not in region]
  insertion = sum(i < last and i not in region for i in range(len(body)))
  result[insertion:insertion] = replacement
  return prune_dead_buffers(_rebuild_proc(proc, params, result))


def fuse_ranges(prog: ProgramNode) -> ProgramNode:
  """Fuse compatible static assembly views into mapped scalar producer ranges."""
  procs = {pr.attrs["name"]: pr for pr in prog.args[: prog.attrs["proc_count"]]}
  return _map_procs(prog, lambda proc: _fuse(proc, procs))
