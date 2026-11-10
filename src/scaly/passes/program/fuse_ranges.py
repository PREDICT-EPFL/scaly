"""Propagate mapped scalar ranges through static assembly views and schedule surviving stores."""

from __future__ import annotations

from dataclasses import dataclass

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
  trip_count,
)
from .scheduling import ScalarNameAllocator, schedule_values
from .fuse_elementwise import _has_expensive


@dataclass
class _Cells:
  trips: np.ndarray
  values: np.ndarray
  positions: np.ndarray
  order: np.ndarray


@dataclass(frozen=True)
class _Value:
  node: ProgramNode
  loads: dict[ProgramNode, np.ndarray]


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
  # Bound repeated scalar work through upstream buffers, including chains of small loops.
  costs: dict[str, int] = {}
  eligible = set()
  for i, stmt in enumerate(body):
    if i in seeds:
      continue
    parsed = _loops(stmt)
    if parsed is None or _has_expensive(stmt):
      continue
    scalar_ops: dict[ProgramNode, int] = {}
    for store in parsed[1]:
      for node in _postorder(store.args[1]):
        scalar_ops[node] = (
          costs.get(node.args[0].attrs["buffer"], 0) if node.op == ProgramOp.LOAD else bool(node.args) + sum(scalar_ops[arg] for arg in node.args)
        )
    if max((scalar_ops[store.args[1]] for store in parsed[1]), default=0) > 8:
      continue
    eligible.add(i)
    for store in parsed[1]:
      name = store.args[0].attrs["buffer"]
      costs[name] = max(costs.get(name, 0), scalar_ops[store.args[1]])
  while True:
    consumed = set().union(*(refs[i].reads for i in region)) & private
    upstream = {i for i in eligible if refs[i].writes & consumed}
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
  no_trip = np.iinfo(np.int64).min
  cells: dict[str, _Cells] = {}
  values: list[_Value] = []
  value_ids: dict[ProgramNode, int] = {}
  store_count = sum(n.op == ProgramOp.STORE for i in region for n in walk_program(seeds.get(i, body[i])))
  new_constants: list[ProgramNode] = []
  folded: dict[ProgramNode, ProgramNode] = {}
  rewritten: dict[tuple[ProgramNode, tuple[tuple[ProgramNode, ProgramNode], ...]], ProgramNode] = {}

  def rewrite_value(rhs: ProgramNode, replacements: dict[ProgramNode, ProgramNode]) -> ProgramNode:
    key = rhs, tuple(replacements.items())
    value = rewritten.get(key)
    if value is None:
      value = rewritten[key] = _rewrite(rhs, replacements, folded)
    return value

  def intern(node: ProgramNode, loads: dict[ProgramNode, np.ndarray] | None = None) -> int:
    used = set(_loads(node)) if loads else set()
    pending = {load: indices for load, indices in (loads or {}).items() if load in used}
    if not pending and node in value_ids:
      return value_ids[node]
    result = len(values)
    values.append(_Value(node, pending))
    if not pending:
      value_ids[node] = result
    return result

  def groups(columns: list[np.ndarray], count: int, limit: int | None = None) -> list[np.ndarray] | None:
    inverse = np.zeros(count, dtype=np.int64)
    first = np.array([0])
    for column in columns:
      if np.all(column == column[0]):
        continue
      _, ids = np.unique(column, return_inverse=True)
      _, first, inverse = np.unique(inverse * (int(ids.max()) + 1) + ids, return_index=True, return_inverse=True)
    if limit is not None and len(first) > limit:
      return None
    order = np.argsort(inverse, kind="stable")
    parts = np.split(order, np.flatnonzero(np.diff(inverse[order])) + 1)
    return [parts[k] for k in np.argsort(first)]

  def occurrences(trips: np.ndarray):
    order = np.argsort(trips, kind="stable")
    starts = np.r_[True, np.diff(trips[order]) != 0]
    ranks = np.arange(len(order)) - np.maximum.accumulate(np.where(starts, np.arange(len(order)), 0))
    for occurrence in range(int(ranks.max()) + 1):
      yield order[ranks == occurrence]

  def defer(load: ProgramNode, indices: np.ndarray, pending: dict[ProgramNode, np.ndarray]) -> ProgramNode:
    if np.all(indices == indices[0]):
      return p.load(rebuild_program(load.args[0], (p.const_int(int(indices[0])),)))
    for existing, previous in pending.items():
      if existing.args[0].attrs == load.args[0].attrs and existing.dtype == load.dtype and np.array_equal(previous, indices):
        return existing
    index = p.var(f"__fuse_index_{len(pending)}")
    placeholder = p.load(rebuild_program(load.args[0], (index,)))
    pending[placeholder] = indices
    return placeholder

  def index_expr(indices: np.ndarray, trips: np.ndarray) -> ProgramNode:
    base = int(indices[0])
    stride = int(indices[-1] - base) // int(trips[-1] - trips[0]) if len(trips) > 1 else 0
    if np.all(indices == base + stride * (trips - trips[0])):
      return fold_program(p.add(p.const_int(base - stride * int(trips[0])), fold_program(p.mul(p.const_int(stride), stage))))
    table_values = np.zeros(int(trips[-1] - trips[0]) + 1, dtype=np.int64)
    table_values[trips - trips[0]] = indices
    mapping = affine_index_map(table_values)
    index = fold_program(p.sub(stage, p.const_int(int(trips[0]))))
    period = len(mapping.residual)
    if period == 1:
      result = p.const_int(int(mapping.residual[0]))
    else:
      table = p.const_buffer(names.fresh(), p.dtypes.int64, (period,), mapping.residual.tolist())
      new_constants.append(table)
      result = p.load(p.view(table, [index if period == len(table_values) else p.mod(index, p.const_int(period))]))
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
    store_refs = [buffer_refs(store) for store in stores]
    written = set().union(*(ref.writes for ref in store_refs))
    if any(ref.reads & (written - ref.writes) for ref in store_refs):
      return proc
    axes = [np.arange(*(int(a.attrs["value"]) for a in rng.args), dtype=np.int64) for rng in ranges]
    grids = np.meshgrid(*axes, indexing="ij")
    variables = {rng.attrs["name"]: grid.reshape(-1) for rng, grid in zip(ranges, grids, strict=True)}
    count = grids[0].size
    assigned_later = {
      node.attrs["target"]
      for j, stmt in enumerate(body)
      if i < j <= last and j not in region
      for node in walk_program(stmt)
      if node.op == ProgramOp.ASSIGN
    }
    substituted: dict[ProgramNode, ProgramNode] = {}
    if i in seeds:
      shifted_stage = fold_program(p.sub(stage, p.const_int(seed_offsets[i])))
      pattern = Pattern(ProgramOp.VAR, lambda n: n.attrs["name"] == ranges[0].attrs["name"], lambda _n: shifted_stage)
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
      if assigned_later and any(node.op == ProgramOp.VAR and node.attrs["name"] in assigned_later for node in walk_program(rhs)):
        return proc
      dest = target.attrs["buffer"]
      if i in seeds and dest not in {param.attrs["name"] for param in params} and not any(dest in ref.reads for ref in refs):
        continue
      if len(target.args) != 1:
        return proc
      output_indices = _indices(target.args[0], variables, constants, count)
      if output_indices is None or np.any(np.diff(np.sort(output_indices)) == 0):
        return proc
      if dest not in cells:
        size = prod(declarations[dest].attrs["shape"])
        cells[dest] = _Cells(np.full(size, no_trip), np.full(size, -1), np.zeros(size, dtype=np.int64), np.empty(0, dtype=np.int64))
      output = cells[dest]
      missing = output.values[output_indices] == -1
      output.order = np.concatenate((output.order, output_indices[missing]))

      def save(positions: np.ndarray, trips: np.ndarray | int, value: int) -> None:
        template = values[value]
        if template.loads:
          # Indices that coincide at only some positions can expose scalar identities there.
          bindings = {
            load: template.loads[load] if load in template.loads else np.full(len(positions), load.args[0].args[0].attrs["value"])
            for load in _loads(template.node)
            if load in template.loads or load.args[0].args[0].op == ProgramOp.CONST_INT
          }
          leaves = list(bindings)
          equalities = [
            bindings[left] == bindings[right]
            for j, left in enumerate(leaves)
            for right in leaves[j + 1 :]
            if left.args[0].attrs == right.args[0].attrs and left.dtype == right.dtype
          ]
          parts = groups(equalities, len(positions))
          assert parts is not None
          for part in parts:
            replacements = {}
            pending = {}
            for load in leaves:
              if load not in template.loads:
                continue
              indices = bindings[load][part]
              alias = next(
                (
                  other
                  for other in leaves
                  if other not in template.loads
                  and other.args[0].attrs == load.args[0].attrs
                  and other.dtype == load.dtype
                  and np.array_equal(bindings[other][part], indices)
                ),
                None,
              )
              replacements[load] = alias if alias is not None else defer(load, indices, pending)
            node = rewrite_value(template.node, replacements)
            indices = output_indices[positions[part]]
            output.trips[indices] = trips[part] if isinstance(trips, np.ndarray) else trips
            output.values[indices] = intern(node, pending)
            output.positions[indices] = np.arange(len(part))
          return
        indices = output_indices[positions]
        output.trips[indices] = trips
        output.values[indices] = value
        output.positions[indices] = np.arange(len(positions))

      if rhs.op in {ProgramOp.CONST_INT, ProgramOp.CONST_FLOAT} and i not in seeds:
        save(np.arange(count), no_trip, intern(rhs))
        continue
      if i in seeds and not any(load.args[0].attrs["buffer"] in cells for load in _loads(rhs)):
        if len(ranges) != 1 or not np.all(missing):
          return proc
        value = rewrite(rhs, [pattern], rebuild=rebuild_program, fixpoint=False, memo=substituted)
        save(np.arange(count), axes[0] + seed_offsets[i], intern(value))
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
      mapped = {}
      for j, load in enumerate(loads):
        name = load.args[0].attrs["buffer"]
        indices = indexed[j]
        if name in constant_values:
          table = np.asarray(constant_values[name])[indices]
          distinct, inverse = np.unique(table, return_inverse=True)
          ids = np.array([intern(constant(value.item(), load.dtype)) for value in distinct])
          mapped[j] = (np.full(count, no_trip), ids[inverse], np.zeros(count, dtype=np.int64))
        elif name in cells:
          source = cells[name]
          ids = source.values[indices]
          if np.any(ids == -1):
            return proc
          mapped[j] = (source.trips[indices], ids, source.positions[indices])
      trips = np.full(count, no_trip)
      fallback = np.full(count, np.iinfo(np.int64).max)
      nonconstant = np.array([value.node.op not in {ProgramOp.CONST_INT, ProgramOp.CONST_FLOAT} for value in values])
      for source_trips, ids, _ in mapped.values():
        active = (source_trips != no_trip) & nonconstant[ids]
        if np.any(active & (trips != no_trip) & (trips != source_trips)):
          return proc
        trips = np.where(active, source_trips, trips)
        fallback = np.minimum(fallback, np.where(source_trips == no_trip, np.iinfo(np.int64).max, source_trips))
      trips = np.where((trips == no_trip) & (fallback != np.iinfo(np.int64).max), fallback, trips)
      partitions = groups([ids for _, ids, _ in mapped.values()], count)
      assert partitions is not None
      for positions in partitions:
        without_trip = positions[trips[positions] == no_trip]
        with_trip = positions[trips[positions] != no_trip]
        batches = [without_trip] if len(without_trip) else []
        if len(with_trip):
          batches.extend(with_trip[ps] for ps in occurrences(trips[with_trip]))
        for ps in batches:
          replacements = {}
          pending: dict[ProgramNode, np.ndarray] = {}

          for j, load in enumerate(loads):
            if j in mapped:
              _, ids, source_positions = mapped[j]
              value = values[int(ids[ps[0]])]
              bindings = {leaf: defer(leaf, indices[source_positions[ps]], pending) for leaf, indices in value.loads.items()}
              replacements[load] = rewrite_value(value.node, bindings) if bindings else value.node
            elif trips[ps[0]] == no_trip:
              replacements[load] = defer(load, indexed[j][ps], pending)
            else:
              idx = index_expr(indexed[j][ps], trips[ps])
              replacements[load] = p.load(rebuild_program(load.args[0], (idx,)))
          value = rewrite_value(rhs, replacements)
          save(ps, trips[ps], intern(value, pending))

  final: list[tuple[str, ProgramNode, np.ndarray, np.ndarray]] = []
  boundaries: set[int] = set()
  for param in params[proc.attrs["input_count"] :]:
    name = param.attrs["name"]
    if name not in produced or name not in cells:
      continue
    output = cells[name]
    indices = output.order
    trips = output.trips[indices]
    if np.any(trips == no_trip):
      return proc
    resolved = np.empty(len(indices), dtype=np.int64)
    partitions = groups([output.values[indices]], len(indices))
    assert partitions is not None
    for ps in partitions:
      value = values[int(output.values[indices[ps[0]]])]
      positions = output.positions[indices[ps]]
      bindings = {load: offsets[positions] for load, offsets in value.loads.items()}
      parts = groups(list(bindings.values()), len(ps), store_count)
      if parts is None:
        return proc
      for part in parts:
        replacements = {}
        for load, offsets in bindings.items():
          offset = int(offsets[part[0]])
          replacements[load] = p.load(rebuild_program(load.args[0], (p.const_int(offset),)))
        node = rewrite_value(value.node, replacements) if replacements else value.node
        resolved[ps[part]] = intern(node)
    partitions = groups([resolved], len(indices))
    assert partitions is not None
    for ps in partitions:
      value = values[int(resolved[ps[0]])].node
      for selected in occurrences(trips[ps]):
        selected = ps[selected]
        ts, ix = trips[selected], indices[selected]
        final.append((name, value, ts, ix))
        ends = np.r_[ts[1:], ts[-1] + 2]
        gaps = ends != ts + 1
        boundaries.update((ts[gaps] + 1).tolist())
        boundaries.update(ends[gaps].tolist())
        boundaries.add(int(ts[0]))
  if len(final) > store_count:
    return proc
  bounds = sorted(boundaries)
  if len(bounds) > 2 * len(region) + 2:
    return proc
  loops = []
  for start, stop in zip(bounds, bounds[1:]):
    stores = []
    for name, value, trips, indices in final:
      if not np.any(trips == start):
        continue
      selected = (trips >= start) & (trips < stop)
      if np.count_nonzero(selected) != stop - start:
        return proc
      target = p.view(declarations[name], [index_expr(indices[selected], trips[selected])])
      stores.append(p.store(target, value))
    if stores:
      assignments, scheduled_values = schedule_values([store.args[1] for store in stores], names)
      scheduled = [*assignments, *(p.store(store.args[0], value) for store, value in zip(stores, scheduled_values, strict=True))]
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
