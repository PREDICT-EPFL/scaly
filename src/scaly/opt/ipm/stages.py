"""A QP's stages, read from its sparsity: the block-tridiagonal partition of the condensed matrix that the stagewise backend factors."""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .structure import QPStructure


@dataclass(frozen=True, eq=False)
class Stages:
  """A partition of a QP's variables into ``K`` blocks of ``B`` slots, in an order in which the
  condensed matrix ``P + A^T A + G^T G`` is block tridiagonal: an entry couples two variables of
  one block or of neighbouring blocks. A block's variables that couple with the next block are among
  its last ``c`` slots, so the block below a diagonal block is zero outside its last ``c`` columns.

  ``order`` holds the variable in each slot, block after block, and -1 in a slot that pads a block
  with fewer than ``B`` variables: such a slot is a row and column of the identity.
  """

  K: int
  B: int
  c: int
  order: np.ndarray

  @property
  def slots(self) -> np.ndarray:
    """The slot of each variable."""
    used = np.flatnonzero(self.order >= 0)
    out = np.empty(used.size, dtype=np.int64)
    out[self.order[used]] = used
    return out

  @property
  def cells(self) -> int:
    """The entries of the block storage: the ``K`` diagonal blocks, then the ``c`` columns of the ``K - 1`` blocks below them."""
    return self.K * self.B * self.B + (self.K - 1) * self.B * self.c

  def cell(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Where the entry that couples slots ``a`` and ``b`` is stored: in the lower triangle of a
    diagonal block, or in the block below one."""
    B, c = self.B, self.c
    row, col = np.maximum(a, b), np.minimum(a, b)
    block = col // B
    same = row // B == block
    assert np.all(same | ((row // B == block + 1) & (col % B >= B - c))), "the stages are a block-tridiagonal partition"
    inside = block * B * B + (row % B) * B + col % B
    under = self.K * B * B + block * B * c + (row % B) * c + (col % B - (B - c))
    return np.where(same, inside, under)

  @property
  def factor_work(self) -> int:
    """The multiply-adds of one factorization: per block a Cholesky, the solve of the block below
    against the trailing triangle, and the product that updates the next block."""
    return _work(self.K, self.B, self.c)

  @property
  def solve_work(self) -> int:
    """The multiply-adds of one pair of triangular solves."""
    return self.K * (self.B * self.B + 2 * self.B * self.c)


def _work(K: int, B: int, c: int) -> int:
  return K * (B**3 // 6) + (K - 1) * (B * c * c // 2 + B * B * c)


def condensed_pattern(s: QPStructure) -> sparse.csr_array:
  """The pattern of ``P + A^T A + G^T G`` with its diagonal, both triangles, as a boolean matrix."""
  n = s.n

  def of(rows: np.ndarray, cols: np.ndarray, count: int) -> sparse.csr_array:
    return sparse.csr_array((np.ones(rows.size, dtype=np.int64), (rows, cols)), shape=(count, n))

  p = of(s.P_rows, s.P_cols, n)
  a, g = of(s.A_rows, s.A_cols, s.p), of(s.G_rows, s.G_cols, s.m)
  total = sparse.csr_array(p + p.T + a.T @ a + g.T @ g + sparse.eye_array(n, dtype=np.int64))
  total.data[:] = 1
  total.sort_indices()
  return total


def _neighbours(adj: sparse.csr_array, front: np.ndarray) -> np.ndarray:
  """The column indices of the rows ``front`` of ``adj``, one after another."""
  starts = adj.indptr[front]
  lengths = adj.indptr[front + 1] - starts
  return adj.indices[np.repeat(starts - np.cumsum(lengths) + lengths, lengths) + np.arange(int(lengths.sum()))]


def _levels(adj: sparse.csr_array, start: np.ndarray, alone: np.ndarray) -> list[np.ndarray]:
  """The level sets of a breadth-first search from ``start``: each level is every unseen neighbour
  of the one before. A part of the graph the search does not reach is searched from its
  lowest-numbered variable, and its levels are laid beside the others (``beside``). The variables
  ``alone`` marks, which nothing couples, are left out: ``_placed`` puts them in. Each level costs
  the entries of its rows, so a search is as long as the matrix has entries."""
  n = adj.shape[0]
  seen = alone.copy()
  parts: list[list[np.ndarray]] = []
  front = np.unique(start)
  front = front[~seen[front]]
  lowest = 0
  while front.size:
    levels: list[np.ndarray] = []
    while front.size:
      seen[front] = True
      levels.append(front)
      reached = np.unique(_neighbours(adj, front))
      front = reached[~seen[reached]]
    parts.append(levels)
    while lowest < n and seen[lowest]:
      lowest += 1
    front = np.array([lowest] if lowest < n else [], dtype=np.int64)
  return beside(parts)


def beside(parts: list[list[np.ndarray]]) -> list[np.ndarray]:
  """Unconnected parts of a graph as one run of levels: the part with the most levels as it is,
  and each other part's levels joined to as many consecutive ones of it, where the largest level
  that results is smallest (the first such place). Nothing couples two parts, so a level may hold
  variables of both."""
  if not parts:
    return []
  parts = sorted(parts, key=len, reverse=True)
  levels = list(parts[0])
  sizes = np.array([level.size for level in levels])
  for part in parts[1:]:
    add = np.array([level.size for level in part])
    at = int(np.argmin((np.lib.stride_tricks.sliding_window_view(sizes, add.size) + add).max(axis=1)))
    for k, level in enumerate(part):
      levels[at + k] = np.concatenate([levels[at + k], level])
    sizes[at : at + add.size] += add
  return levels


def _placed(levels: list[np.ndarray], alone: np.ndarray) -> list[list[np.ndarray]]:
  """``levels`` with the variables ``alone`` added, which nothing couples: first in the room the
  levels leave below the largest, which pads them anyway; what is left either spread over every
  level, each one larger, or in levels of their own as large as the others, in front, where they
  couple with nothing. Both are returned when both are possible, for ``stages`` to keep the one
  of less work: a few more slots in each of many blocks cost more than one more block, and in
  each of one or two blocks less."""
  if not alone.size:
    return [levels]
  if not levels:
    return [[alone]]
  size = max(level.size for level in levels)
  filled, taken = [], 0
  for level in levels:
    more = min(size - level.size, alone.size - taken)
    filled.append(np.concatenate([level, alone[taken : taken + more]]))
    taken += more
  rest = alone[taken:]
  if not rest.size:
    return [filled]
  spread = [np.concatenate([level, extra]) for level, extra in zip(filled, np.array_split(rest, len(filled)), strict=True)]
  own = [rest[k : k + size] for k in range(0, rest.size, size)]
  return [spread, own + filled]


def _shifted(levels: list[np.ndarray]) -> list[np.ndarray]:
  """``levels`` with as many variables of the first level moved to the second as the larger levels
  after it leave room for. A variable of the first level has no level before it to couple with,
  so it may as well sit in the second, and each one that does is one coupling slot fewer in the
  first: the search from every variable that shares a state's rows starts with that stage's
  inputs beside its states, all of them coupled with the next state."""
  if len(levels) < 3:
    return levels
  room = max(level.size for level in levels[2:]) - levels[1].size
  moved = min(levels[0].size - 1, room)
  if moved <= 0:
    return levels
  first = np.sort(levels[0])
  return [first[: first.size - moved], np.concatenate([first[first.size - moved :], levels[1]]), *levels[2:]]


def _far_end(adj: sparse.csr_array, v: int, alone: np.ndarray) -> int:
  """A variable of least degree in the last level of the search from ``v``."""
  last = _levels(adj, np.array([v]), alone)[-1]
  degree = np.diff(adj.indptr)[last]
  return int(last[np.argmin(degree)])


def _twins(adj: sparse.csr_array, v: int, closed: bool) -> np.ndarray:
  """The variables with ``v``'s neighbours: with itself counted (``closed``), those that sit in the
  same rows as ``v`` everywhere; without, those that see the same others and not each other."""
  n = adj.shape[0]
  pattern = adj if closed else sparse.csr_array(adj - sparse.eye_array(n, dtype=adj.dtype))
  if not closed:
    pattern.eliminate_zeros()
  row = np.zeros(n, dtype=np.int64)
  row[pattern.indices[pattern.indptr[v] : pattern.indptr[v + 1]]] = 1
  size = int(row.sum())
  same = (np.diff(pattern.indptr) == size) & (pattern @ row == size)
  same[v] = True
  return np.flatnonzero(same)


def _partition(adj: sparse.csr_array, levels: list[np.ndarray]) -> Stages:
  n = adj.shape[0]
  level = np.empty(n, dtype=np.int64)
  for k, members in enumerate(levels):
    level[members] = k
  # A variable couples with the next block when a neighbour is there.
  coo = adj.tocoo()
  ahead = np.zeros(n, dtype=bool)
  ahead[coo.row[level[coo.col] == level[coo.row] + 1]] = True
  K = len(levels)
  B = max(members.size for members in levels)
  c = max([int(ahead[members].sum()) for members in levels[:-1]], default=0)
  c = max(c, 1) if K > 1 else 0
  # The last block couples with none, but its variables are laid out as the one before it lays
  # out its own where the two are the same size: then a row of a constraint matrix reads the same
  # slots of every block, which is what its dense arrays are made of (``dense_blocks``).
  last = np.zeros(n, dtype=bool)
  if K > 1 and levels[-1].size == levels[-2].size:
    last[np.sort(levels[-1])] = ahead[np.sort(levels[-2])]
  order = np.full(K * B, -1, dtype=np.int64)
  for k, members in enumerate(levels):
    members = np.sort(members)
    behind = (last if k == K - 1 else ahead)[members]
    inside = np.concatenate([members[~behind], members[behind]])
    order[(k + 1) * B - inside.size : (k + 1) * B] = inside
  return Stages(K, B, c, order)


def pairs_in_rows(rows: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
  """Every unordered pair of entries that share a row, an entry with itself included, for entries
  in rows ``rows`` of a matrix of ``count`` rows: the products of ``M^T W M``, each once."""
  order = np.argsort(rows, kind="stable")
  per_row = np.bincount(rows, minlength=count)
  starts = np.concatenate([[0], np.cumsum(per_row)])
  first, second = [np.zeros(0, dtype=np.int64)], [np.zeros(0, dtype=np.int64)]
  for size in np.unique(per_row[per_row > 0]):
    members = order[starts[np.flatnonzero(per_row == size)][:, None] + np.arange(size)[None, :]]
    a, b = np.triu_indices(int(size))
    first.append(members[:, a].reshape(-1))
    second.append(members[:, b].reshape(-1))
  return np.concatenate(first), np.concatenate(second)


@dataclass(frozen=True, eq=False)
class DenseBlocks:
  """The part of a constraint matrix ``M`` whose share of ``M^T W M`` is multiplied as dense arrays,
  one per block.

  A row of ``M`` reads the variables of one block, or of a block and the one before it, where it
  reads coupling slots only. So the rows whose last block is ``k`` fit one array of ``c + B``
  columns: the coupling slots of block ``k - 1``, then block ``k``. The columns of that array which
  are mostly filled, over all blocks, are the ``nd`` dense ones, and the entries in them go into
  ``(K, r, nd)`` arrays, ``r`` the most rows a block has.

  Attributes:
    entries: ``(K, r, nd)``, the entry of ``M`` in each cell, -1 where there is none.
    rows: ``(K, r)``, the row of ``M`` in each row of a block's array, -1 where there is none. A
      row with entries in fewer than half the dense columns is in no array.
    slots: ``(K, nd)``, the slot each column of a block's array stands for; -1 in the first block
      for a column of the block before it.
    source: the entries of the products ``(K, nd, nd)`` that the block storage keeps, one per
      coupled pair of slots.
    cell: where each goes (``Stages.cell``).
    taken: per entry of ``M``, whether it is in an array.
    products: the pairs of entries the dense columns hold in one row, summed over the rows: what
      the same products cost through index tables.
  """

  entries: np.ndarray
  rows: np.ndarray
  slots: np.ndarray
  source: np.ndarray
  cell: np.ndarray
  taken: np.ndarray
  products: int

  @property
  def work(self) -> int:
    """The multiply-adds of the dense products."""
    K, r, nd = self.entries.shape
    return K * r * nd * nd


def dense_blocks(st: Stages, rows: np.ndarray, cols: np.ndarray, count: int) -> DenseBlocks | None:
  """The dense columns of a matrix of ``count`` rows with entries at ``(rows, cols)``, over the
  partition ``st``; None when no column is mostly filled."""
  K, B, c = st.K, st.B, st.c
  if not rows.size:
    return None
  slot = st.slots[cols]
  block = slot // B
  last = np.zeros(count, dtype=np.int64)
  np.maximum.at(last, rows, block)
  mine = last[rows]
  before = block < mine
  assert np.all(block[before] == mine[before] - 1) and np.all(slot[before] % B >= B - c), (
    "a row reads one block and the coupling slots of the one before"
  )
  column = np.where(before, slot % B - (B - c), c + slot % B)
  dense = np.flatnonzero(2 * np.bincount(column, minlength=c + B) >= count)
  if not dense.size:
    return None
  place = np.full(c + B, -1, dtype=np.int64)
  place[dense] = np.arange(dense.size)
  nd = int(dense.size)
  # A row with few entries in those columns stays out: it would be a row of zeros in its block's
  # array, and one block with many of them would lengthen every block's.
  filled = np.bincount(rows[place[column] >= 0], minlength=count)
  among = np.flatnonzero(2 * filled >= nd)
  if not among.size:
    return None
  taken = (place[column] >= 0) & (2 * filled[rows] >= nd)
  # A row's place among the rows of its block that are taken, in row order.
  by_block = among[np.argsort(last[among], kind="stable")]
  starts = np.searchsorted(last[by_block], np.arange(K))
  rank = np.full(count, -1, dtype=np.int64)
  rank[by_block] = np.arange(among.size) - starts[last[by_block]]
  r = int(rank.max()) + 1
  entries = np.full((K, r, nd), -1, dtype=np.int64)
  hit = np.flatnonzero(taken)
  entries[mine[hit], rank[rows[hit]], place[column[hit]]] = hit
  row_table = np.full((K, r), -1, dtype=np.int64)
  row_table[last[among], rank[among]] = among
  # The slot each dense column stands for in each block's array; the first block has none before it.
  slots = np.where(dense[None, :] < c, (np.arange(K)[:, None] - 1) * B + (B - c) + dense[None, :], np.arange(K)[:, None] * B + dense[None, :] - c)
  a, b = np.tril_indices(nd)
  kept = slots[:, b] >= 0
  source = (np.arange(K)[:, None] * nd * nd + a[None, :] * nd + b[None, :])[kept]
  cell = st.cell(slots[:, a][kept], slots[:, b][kept])
  per_row = np.bincount(rows[hit], minlength=count).astype(np.int64)
  return DenseBlocks(entries, row_table, np.where(slots >= 0, slots, -1), source, cell, taken, int(per_row @ per_row))


_STAGES: weakref.WeakKeyDictionary[QPStructure, Stages] = weakref.WeakKeyDictionary()


def stages(s: QPStructure) -> Stages:
  """The partition of ``s`` the stagewise backend factors, found once per structure.

  The blocks are the level sets of a breadth-first search of the condensed matrix's graph, which
  are block tridiagonal for any pattern. How narrow they come out depends on where the search
  starts: from one state of a multistage problem the second level holds the rest of that stage
  and the whole of the next. So the search is tried from both ends of the graph (a variable far
  from a variable far from the first), and at each from that variable alone, from every variable
  that shares its rows, and from every variable that shares its neighbours, each also with its
  first level thinned into the second (``_shifted``); the partition whose factorization is the
  least work is kept. On a multistage problem, in any variable order, that is the stages, with as
  many coupling slots ``c`` as there are states. Variables that nothing couples (no row reads them,
  and ``P`` only on its diagonal) take no part in the search, and are placed after it
  (``_placed``); when there are only such variables, they are one block.
  """
  if s not in _STAGES:
    adj = condensed_pattern(s)
    degree = np.diff(adj.indptr)
    alone = degree == 1
    if alone.all():
      _STAGES[s] = _partition(adj, [np.arange(s.n)])
      return _STAGES[s]
    coupled = np.flatnonzero(~alone)
    first = int(coupled[np.argmin(degree[coupled])])
    far = _far_end(adj, first, alone)
    ends = dict.fromkeys((far, _far_end(adj, far, alone), first))
    if alone.any():
      # Also from the first variable that is coupled, and the far end from it: where the search went
      # when it started from a variable that nothing couples, and on some patterns the better start.
      low = int(coupled[0])
      ends.update(dict.fromkeys((low, _far_end(adj, low, alone))))
    best: Stages | None = None
    for v in ends:
      for start in (np.array([v]), _twins(adj, v, True), _twins(adj, v, False)):
        levels = _levels(adj, start, alone)
        for candidate in (levels, _shifted(levels)):
          for placed in _placed(candidate, np.flatnonzero(alone)):
            found = _partition(adj, placed)
            if best is None or found.factor_work < best.factor_work:
              best = found
    assert best is not None
    _STAGES[s] = best
  return _STAGES[s]


__all__ = ["DenseBlocks", "Stages", "beside", "condensed_pattern", "dense_blocks", "pairs_in_rows", "stages"]
