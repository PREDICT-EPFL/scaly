"""Store constant 0/1 multiplier tables of a direction x position loop as bits, and split the loop so it vectorizes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...ir import program as p
from ...ir.program import ProgramNode, ProgramOp, RangeKind, walk_program
from ...ir.types import dtypes
from ._common import _map_procs, _proc_parts, _rebuild_proc, rebuild_program

# Entries per stored word: one uint8 per eight entries of a direction's row, so a lane block of eight
# shares one word and the bit is the lane's position in the block.
_WORD_BITS = 8


@dataclass(frozen=True)
class _Layout:
  """A loop over ``directions * positions`` entries whose other reads depend on the position only.

  ``reads`` maps each position-dependent read to ``(start, stride)``: entry ``d * positions + j``
  reads ``start + stride * j`` of its buffer.
  """

  directions: int
  positions: int
  reads: dict[int, tuple[int, int]]


def pack_masks(prog: ProgramNode) -> ProgramNode:
  """Rewrite each eligible ``out[i] = sum_l mask_l[i] * x_l[f(i)]`` loop over bit-packed masks.

  A forward-over-reverse Hessian over overlapping stage windows assembles its global compressed
  product as such sums: ``mask_l`` holds, for each direction and stage entry, whether local color
  ``l`` feeds that global direction. The masks are ``static const double`` tables of
  ``directions * stages * entries`` zeros and ones, which grow linearly with the horizon, and the
  flat index ``f(i) = i % (stages * entries)`` keeps the loop from vectorizing. This stores each
  mask as one bit per entry, a direction's row padded to whole bytes, and splits the loop into
  directions, blocks of eight positions and the eight lanes of a block, so the reads of ``x_l`` are
  contiguous and each lane keeps or zeroes its value by testing one bit of the block's byte.
  """
  return _map_procs(prog, _pack_proc)


_INDEX_OPS = {
  ProgramOp.ADD: np.add,
  ProgramOp.SUB: np.subtract,
  ProgramOp.MUL: np.multiply,
  ProgramOp.DIV: np.floor_divide,
  ProgramOp.MOD: np.remainder,
  ProgramOp.MINIMUM: np.minimum,
  ProgramOp.MAXIMUM: np.maximum,
}


def _int_values(node: ProgramNode, name: str, i: np.ndarray) -> np.ndarray | None:
  """Evaluate an integer index expression at every ``name = i``, or None when it reads memory or another variable."""
  if node.op == ProgramOp.CONST_INT:
    return np.full_like(i, int(node.attrs["value"]))
  if node.op == ProgramOp.VAR:
    return i if node.attrs["name"] == name else None
  if node.op not in _INDEX_OPS:
    return None
  a, b = (_int_values(arg, name, i) for arg in node.args)
  if a is None or b is None:
    return None
  # C truncates toward zero where NumPy floors; they agree for a nonnegative dividend and positive divisor.
  if node.op in (ProgramOp.DIV, ProgramOp.MOD) and ((a < 0).any() or (b <= 0).any()):
    return None
  return _INDEX_OPS[node.op](a, b)


def _uses_var(node: ProgramNode, name: str) -> bool:
  return any(n.op == ProgramOp.VAR and n.attrs["name"] == name for n in walk_program(node))


def _is_mask(node: ProgramNode, name: str, masks: dict[str, ProgramNode]) -> bool:
  if node.op != ProgramOp.LOAD:
    return False
  view = node.args[0]
  index = view.args[0]
  return view.attrs["buffer"] in masks and index.op == ProgramOp.VAR and index.attrs["name"] == name


def _mask_products(value: ProgramNode, name: str, masks: dict[str, ProgramNode]) -> list[ProgramNode]:
  return [
    n
    for n in walk_program(value)
    if n.op == ProgramOp.MUL and any(_is_mask(a, name, masks) for a in n.args) and not all(_is_mask(a, name, masks) for a in n.args)
  ]


def _layout(stmt: ProgramNode, masks: dict[str, ProgramNode]) -> tuple[_Layout, set[str]] | None:
  """The layout of an eligible loop and the masks it multiplies by, or None."""
  if stmt.op != ProgramOp.FOR or len(stmt.args) != 2:
    return None
  rng, store = stmt.args
  start, stop, step = (a.attrs.get("value") if a.op == ProgramOp.CONST_INT else None for a in rng.args)
  name = rng.attrs["name"]
  if start != 0 or step != 1 or stop is None or store.op != ProgramOp.STORE:
    return None
  target, value = store.args
  if not (target.args[0].op == ProgramOp.VAR and target.args[0].attrs["name"] == name) or not value.dtype.is_floating:
    return None
  products = _mask_products(value, name, masks)
  if not products:
    return None
  # Each use of a mask must be one the rewrite replaces: a mask[i] operand of one of those products.
  product_ids = {id(n) for n in products}
  for parent in [store, *walk_program(value)]:
    for arg in parent.args:
      if arg.op == ProgramOp.LOAD and arg.args[0].attrs["buffer"] in masks and not (id(parent) in product_ids and _is_mask(arg, name, masks)):
        return None
  used = {a.args[0].attrs["buffer"] for prod in products for a in prod.args if _is_mask(a, name, masks)}
  shapes = {tuple(masks[m].attrs["shape"]) for m in used}
  if len(shapes) != 1:
    return None
  shape = next(iter(shapes))
  if len(shape) < 2 or int(np.prod(shape)) != stop:
    return None
  directions = shape[0]
  positions = stop // directions
  # Every other use of the loop variable is a read whose index depends on the position alone,
  # contiguously or with a constant stride, so the split loop can read it as start + stride * j.
  reads: dict[int, tuple[int, int]] = {}
  mask_loads = {id(a) for prod in products for a in prod.args if _is_mask(a, name, masks)}
  for node in walk_program(value):
    if node.op == ProgramOp.LOAD and id(node) not in mask_loads and _uses_var(node.args[0], name):
      index = node.args[0].args[0]
      # Python integers, as object arrays: int64 would wrap where the index arithmetic overflows.
      values = _int_values(index, name, np.arange(stop, dtype=object))
      if values is None:
        return None
      values = values.astype(np.int64)
      per_position = values[:positions]
      stride = int(per_position[1] - per_position[0]) if positions > 1 else 0
      if not (values == np.tile(per_position, directions)).all() or not (np.diff(per_position) == stride).all():
        return None
      reads[id(node)] = (int(per_position[0]), stride)
  # The loop variable must not appear anywhere but in those reads and the mask loads.
  allowed = {id(n) for n in walk_program(value) if n.op == ProgramOp.LOAD and (id(n) in reads or id(n) in mask_loads)}
  remaining = [n for n in walk_program(value) if n.op == ProgramOp.LOAD and id(n) not in allowed]
  if any(_uses_var(n, name) for n in remaining) or any(n.op == ProgramOp.VAR and n.attrs["name"] == name for n in _walk_outside_loads(value)):
    return None
  return _Layout(directions, positions, reads), used


def _walk_outside_loads(node: ProgramNode):
  stack = [node]
  while stack:
    n = stack.pop()
    if n.op == ProgramOp.LOAD:
      continue
    yield n
    stack.extend(n.args)


def _packed(decl: ProgramNode, layout: _Layout) -> ProgramNode:
  """The mask as one bit per entry, each direction's row padded to whole bytes."""
  bits = np.asarray(decl.attrs["values"], dtype=np.float64).reshape(layout.directions, layout.positions) != 0
  row = -(-layout.positions // _WORD_BITS)
  padded = np.zeros((layout.directions, row * _WORD_BITS), dtype=bool)
  padded[:, : layout.positions] = bits
  words = np.packbits(padded.reshape(layout.directions, row, _WORD_BITS), axis=-1, bitorder="little").reshape(-1)
  return p.const_buffer(decl.attrs["name"], dtypes.uint8, (layout.directions * row,), [int(w) for w in words])


def _owner(name: str, buffers: dict[str, ProgramNode]) -> tuple[str, int]:
  """The buffer an alias chain ends at, and the alias's offset into it."""
  offset, seen = 0, set()
  while name in buffers and "alias_of" in buffers[name].attrs and name not in seen:
    seen.add(name)
    offset += int(buffers[name].attrs.get("alias_offset", 0))
    name = buffers[name].attrs["alias_of"]
  return name, offset


def _rewrite(stmt: ProgramNode, layout: _Layout, masks: dict[str, ProgramNode], names: set[str], buffers: dict[str, ProgramNode]) -> ProgramNode:
  rng, store = stmt.args
  target, value = store.args
  name = rng.attrs["name"]
  kind = rng.attrs["kind"]

  def fresh(suffix: str) -> str:
    candidate, k = f"{name}_{suffix}", 0
    while candidate in names:
      k += 1
      candidate = f"{name}_{suffix}{k}"
    names.add(candidate)
    return candidate

  row = -(-layout.positions // _WORD_BITS)
  full = layout.positions // _WORD_BITS
  direction, block, lane = p.var(fresh("dir")), p.var(fresh("blk")), p.var(fresh("bit"))
  tail = p.var(fresh("tail"))

  def body(position: ProgramNode, word_index: ProgramNode, bit: ProgramNode) -> ProgramNode:
    def visit(node: ProgramNode) -> ProgramNode:
      if node.op == ProgramOp.MUL and any(_is_mask(a, name, masks) for a in node.args) and not all(_is_mask(a, name, masks) for a in node.args):
        mask, other = (node.args[0], node.args[1]) if _is_mask(node.args[0], name, masks) else (node.args[1], node.args[0])
        word = p.load(ProgramNode(ProgramOp.VIEW, (word_index,), mask.args[0].attrs, dtypes.uint8))
        return p.bit_select(word, bit, visit(other))
      if node.op == ProgramOp.LOAD and id(node) in layout.reads:
        # Read the alias's owner directly: the vectorizer will not widen a loop that reads through an alias.
        view = node.args[0]
        owner, offset = _owner(view.attrs["buffer"], buffers)
        start, stride = layout.reads[id(node)]
        start += offset
        moving = position if stride == 1 else p.mul(p.const_int(stride), position)
        index = p.const_int(start) if not stride else moving if not start else p.add(p.const_int(start), moving)
        return p.load(ProgramNode(ProgramOp.VIEW, (index,), {**view.attrs, "buffer": owner}, view.dtype))
      if not node.args:
        return node
      return rebuild_program(node, tuple(visit(a) for a in node.args))

    entry = p.add(p.mul(direction, p.const_int(layout.positions)), position)
    out = ProgramNode(ProgramOp.VIEW, (entry,), target.attrs, target.dtype)
    return p.store(out, visit(value))

  loops = []
  if full:
    position = p.add(p.mul(block, p.const_int(_WORD_BITS)), lane)
    word_index = p.add(p.mul(direction, p.const_int(row)), block)
    lanes = p.for_(p.range_(lane.attrs["name"], 0, _WORD_BITS, kind=RangeKind.SERIAL), [body(position, word_index, lane)])
    loops.append(p.for_(p.range_(block.attrs["name"], 0, full, kind=RangeKind.SERIAL), [lanes]))
  if layout.positions % _WORD_BITS:
    position = p.add(p.const_int(full * _WORD_BITS), tail)
    word_index = p.add(p.mul(direction, p.const_int(row)), p.const_int(full))
    loops.append(p.for_(p.range_(tail.attrs["name"], 0, layout.positions % _WORD_BITS, kind=RangeKind.SERIAL), [body(position, word_index, tail)]))
  return p.for_(p.range_(direction.attrs["name"], 0, layout.directions, kind=kind), loops)


def _pack_proc(proc: ProgramNode) -> ProgramNode:
  params, body = _proc_parts(proc)
  masks = {
    s.attrs["name"]: s
    for s in body
    if s.op == ProgramOp.BUFFER
    and "values" in s.attrs
    and "alias_of" not in s.attrs
    and s.dtype.is_floating
    and set(np.unique(np.asarray(s.attrs["values"], dtype=np.float64)).tolist()) <= {0.0, 1.0}
  }
  if not masks:
    return proc
  candidates = {i: found for i, stmt in enumerate(body) if (found := _layout(stmt, masks)) is not None}
  # A mask is packed only when no other statement or alias reads it and its loops share one layout.
  # Statements are compared as a whole: interned views make per-view counts merge identical reads.
  aliased = {b.attrs["alias_of"] for b in walk_program(proc) if b.op == ProgramOp.BUFFER and "alias_of" in b.attrs}
  while True:
    elsewhere = {
      n.attrs["buffer"]
      for i, stmt in enumerate(body)
      if i not in candidates and stmt.op != ProgramOp.BUFFER
      for n in walk_program(stmt)
      if n.op == ProgramOp.VIEW
    }
    layouts: dict[str, set[tuple[int, int]]] = {}
    for layout, used in candidates.values():
      for mask in used:
        layouts.setdefault(mask, set()).add((layout.directions, layout.positions))
    packable = {m for m in layouts if m not in elsewhere and m not in aliased and len(layouts[m]) == 1}
    dropped = [i for i, (_, used) in candidates.items() if not used <= packable]
    if not dropped:
      break
    for i in dropped:
      del candidates[i]
  if not candidates:
    return proc
  names = {n.attrs["name"] for n in walk_program(proc) if n.op in (ProgramOp.RANGE, ProgramOp.VAR) and "name" in n.attrs}
  buffers = {n.attrs["name"]: n for n in walk_program(proc) if n.op == ProgramOp.BUFFER}
  packed_layout = {m: layout for layout, used in candidates.values() for m in used}
  out = []
  for i, stmt in enumerate(body):
    if stmt.op == ProgramOp.BUFFER and stmt.attrs.get("name") in packed_layout:
      out.append(_packed(stmt, packed_layout[stmt.attrs["name"]]))
    elif i in candidates:
      out.append(_rewrite(stmt, candidates[i][0], masks, names, buffers))
    else:
      out.append(stmt)
  return _rebuild_proc(proc, params, out)
