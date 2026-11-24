"""Decide which entries of each node its consumers read, so a stored node is computed only there."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ...ad.rules import ELEMENTWISE
from ...ir.expr import Expr, ExprOp, _index_values, topo
from ...ir.support import project_entries, reduce_entries, slice_indices, transpose_entries
from ..affine import affine_index_map
from .fusion import ALIASES, adds_into_zeros, scatter_targets

Entries = np.ndarray | None
# A node with more entries is demanded whole, so planning never spends memory in proportion to a large
# value, such as a broadcast that is only ever evaluated where it is read.
TRACKED = 1 << 22


def plan_demand(roots: Sequence[Expr], in_place: frozenset[int], narrows: Callable[[Expr, np.ndarray], bool]) -> dict[int, np.ndarray]:
  """The entries each node is computed at, for the nodes computed at fewer than all of them.

  Walking from the outputs, a node's demanded entries are the union of the entries its consumers
  read. A node in ``in_place`` is evaluated where it is read, so it is computed at exactly its
  demanded entries. A stored node is computed there when ``narrows`` accepts them, and at every
  entry otherwise. A node's operands are then demanded at the entries its computation reads, which
  every lowering rule must not exceed."""
  # Each read node's demanded entries as a mask, None when all of them.
  wanted: dict[int, np.ndarray | None] = {root.id: None for root in roots}
  domains: dict[int, np.ndarray] = {}
  for node in reversed(topo(roots)):
    if node.id not in wanted:
      continue
    want = wanted.pop(node.id)
    computed: Entries = None
    if want is not None and node.type.pattern is None:
      entries = np.flatnonzero(want)
      if 0 < len(entries) < node.size and (node.id in in_place or narrows(node, entries)):
        computed = domains[node.id] = entries
    tracked = all(arg.size <= TRACKED for arg in node.args)
    reads = operand_entries(node, computed, placeholder=node.id in in_place) if tracked else (None,) * len(node.args)
    for arg, read in zip(node.args, reads, strict=True):
      if arg.id in wanted and wanted[arg.id] is None:
        continue
      if read is None or arg.type.pattern is not None or not arg.size:
        wanted[arg.id] = None
        continue
      mask = wanted.get(arg.id)
      if mask is None:
        mask = wanted[arg.id] = np.zeros(arg.size, dtype=bool)
      mask[read] = True
  return domains


def operand_entries(node: Expr, entries: Entries, *, placeholder: bool = False) -> tuple[Entries, ...]:
  """The flat entries of each operand that computing ``node`` at ``entries`` reads, ``None`` for all
  of them; ``entries`` is ``None`` for the whole node. A concatenation evaluated in place reads
  every part at each entry, an unselected part at the first entry it computes, so with
  ``placeholder`` a part it reads nowhere still computes its first entry."""
  op = ExprOp(node.op)
  dense = node.type.pattern is None and all(arg.type.pattern is None for arg in node.args)
  everything: tuple[Entries, ...] = (None,) * len(node.args)
  if not dense:
    return everything
  if op == ExprOp.GATHER:
    if node.args[1].value is None:
      return everything
    picked = _index_values(node).reshape(-1)
    picked = picked if entries is None else picked[entries]
    # A filling gather loads a clamped index for an entry out of range, then discards it.
    return (np.clip(picked, 0, node.args[0].size - 1) if node.attrs["mode"] == "fill" else picked, None)
  if op == ExprOp.SLICE:
    return (slice_indices(node.args[0].shape, node.attrs["index"], entries).reshape(-1),)
  if entries is None:
    return everything
  if op in ELEMENTWISE:
    return tuple(project_entries(entries, arg.shape, node.shape) for arg in node.args)
  if op in ALIASES:
    return (entries,)
  if op == ExprOp.TRANSPOSE:
    axes = tuple(int(a) for a in node.attrs["axes"])
    return (transpose_entries(entries, node.shape, tuple(axes.index(dim) for dim in range(len(axes)))),)
  if op in (ExprOp.CONCAT, ExprOp.STACK):
    owner, place = parts(node, entries)
    picked = [place[owner == k] for k in range(len(node.args))]
    return tuple(np.zeros(1, dtype=np.int64) if placeholder and arg.size and not len(p) else p for p, arg in zip(picked, node.args, strict=True))
  if op == ExprOp.SCATTER and adds_into_zeros(node):
    return (None, None, scattered_values(node, entries))
  if op == ExprOp.REDUCE:
    src = node.args[0]
    return (np.flatnonzero(np.isin(reduce_entries(np.arange(src.size), src.shape, node.attrs["axes"]), entries)),)
  if op == ExprOp.MATMUL and len(node.args) == 2:
    return matmul_entries(node, entries)
  return everything


def scattered_values(node: Expr, entries: Entries) -> np.ndarray:
  """The values a scatter into zeros stores when its consumers read ``entries``: those landing there,
  when that needs no index table a loop over all of them does not, else all of them."""
  targets = scatter_targets(node)
  every = np.arange(len(targets), dtype=np.int64)
  if entries is None:
    return every
  lanes = np.flatnonzero(np.isin(targets, entries))
  return lanes if affine(lanes) and (affine(targets[lanes]) or not affine(targets)) else every


def parts(node: Expr, entries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """For each entry of a concatenation or a stack, the operand it comes from and its place there."""
  axis = int(node.attrs.get("axis", 0))
  owners, places = [], []
  for k, arg in enumerate(node.args):
    shape = arg.shape if node.op == ExprOp.CONCAT else (*arg.shape[:axis], 1, *arg.shape[axis:])
    owners.append(np.full(shape, k, dtype=np.int64))
    places.append(np.arange(arg.size, dtype=np.int64).reshape(shape))
  return np.concatenate(owners, axis=axis).reshape(-1)[entries], np.concatenate(places, axis=axis).reshape(-1)[entries]


def matmul_entries(node: Expr, entries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """The rows of the left operand and the columns of the right one that the products at ``entries`` contract."""
  a, b = node.args
  coords = np.unravel_index(entries, node.shape)
  batch = len(node.shape) - (len(a.shape) > 1) - (len(b.shape) > 1)
  inner = a.shape[-1]

  def free(arg: Expr, axis: int) -> tuple[np.ndarray, int]:
    """Each entry's batch and free coordinate in ``arg``, numbered together, and the free extent."""
    lead, extent = arg.shape[:-2], arg.shape[axis] if len(arg.shape) > 1 else 1
    axes = [np.zeros(len(entries), dtype=np.int64) if dim == 1 else coords[batch - len(lead) + k] for k, dim in enumerate(lead)]
    axes.append(coords[batch if axis == -2 else -1] if len(arg.shape) > 1 else np.zeros(len(entries), dtype=np.int64))
    return np.unique(np.ravel_multi_index(tuple(axes), (*lead, extent))), extent

  rows, _ = free(a, -2)
  columns, width = free(b, -1)
  left = (rows[:, None] * inner + np.arange(inner)).reshape(-1)
  right = (columns[:, None] // width * inner * width + np.arange(inner) * width + columns[:, None] % width).reshape(-1)
  return left, right


def cheap(entries: np.ndarray, size: int) -> bool:
  """Whether computing at ``entries`` costs less than computing all ``size``: an affine run needs no
  table, and a table pays when it skips at least a third of the work."""
  return 3 * len(entries) <= 2 * size or affine(entries)


def affine(entries: np.ndarray) -> bool:
  """Whether a loop reaches ``entries`` by arithmetic on its index, without a table."""
  return len(affine_index_map(entries).residual) <= 1
