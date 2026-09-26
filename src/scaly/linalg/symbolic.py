"""Symbolic analysis for a sparse ``LDL^T``: ordering, elimination tree, the pattern of ``L`` and the
tables a left-looking factorization reads, all in NumPy when the graph is built.

Nothing here builds an expression. The factorization (``linalg.ldl``) turns the tables into loops;
this module decides what those loops index. The matrix is symmetric and given by the pattern of
one or both of its triangles; values never enter, so the result is the structural pattern, with no
numerical cancellation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import splu

Ordering = Literal["natural", "rcm", "mmd", "auto"]
ORDERINGS: tuple[Ordering, ...] = ("natural", "rcm", "mmd")


def _symmetric_graph(n: int, rows: np.ndarray, cols: np.ndarray) -> sparse.csr_array:
  """The pattern of ``A + A^T`` with a full diagonal, as a CSR array of ones."""
  r = np.concatenate([rows, cols, np.arange(n)])
  c = np.concatenate([cols, rows, np.arange(n)])
  g = sparse.csr_array((np.ones(r.size), (r, c)), shape=(n, n))
  g.sum_duplicates()
  g.data[:] = 1.0
  return g


def ordering(n: int, rows: np.ndarray, cols: np.ndarray, method: Ordering = "mmd") -> np.ndarray:
  """A fill-reducing symmetric permutation, ``perm[new] = old``.

  ``natural`` keeps the order; ``rcm`` is reverse Cuthill–McKee (small bandwidth, for banded and
  stage-ordered problems); ``mmd`` is SuperLU's multiple minimum degree on ``A + A^T``, taken from
  a factorization of a diagonally dominant matrix with the same pattern (only the column
  permutation is used)."""
  if method not in ORDERINGS:
    raise ValueError(f"ordering must be one of {ORDERINGS}, got {method!r}")
  if method == "natural" or n <= 1:
    return np.arange(n, dtype=np.int64)
  g = _symmetric_graph(n, rows, cols)
  if method == "rcm":
    return np.asarray(reverse_cuthill_mckee(g, symmetric_mode=True), dtype=np.int64)
  m = sparse.csc_array(g)
  m.data[:] = -1.0
  m = sparse.csc_array(m + sparse.diags_array(np.full(n, float(n + 1)), format="csc"))
  lu = splu(m, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options={"SymmetricMode": True})
  # ``perm_c[old] = new``: SuperLU factors column ``perm_c[i]`` of the permuted matrix from column ``i``.
  return np.argsort(np.asarray(lu.perm_c, dtype=np.int64)).astype(np.int64)


@dataclass(frozen=True)
class Segment:
  """Consecutive columns ``[start, stop)`` factored by one loop, with every column's tables padded to
  the widest in the segment: ``a`` entries of the matrix column, ``u`` update lanes, ``c`` entries
  of the column of ``L``."""

  start: int
  stop: int
  a: int
  u: int
  c: int

  @property
  def length(self) -> int:
    return self.stop - self.start


@dataclass(frozen=True)
class CostModel:
  """Padded work, in lane units, that segment selection minimizes: ``step`` per column, ``a`` per
  matrix entry lane, ``u`` per update lane, ``c`` per ``L`` entry lane, and ``segment`` per loop
  (code size and one more procedure)."""

  step: float = 4.0
  a: float = 1.0
  u: float = 1.5
  c: float = 2.0
  segment: float = 256.0


@dataclass(frozen=True, eq=False)
class SymbolicLDL:
  """The structure of ``P K P^T = L D L^T`` for a symmetric ``K`` with a fixed pattern.

  Ragged per-column tables share a pointer array (``x_ptr[j] : x_ptr[j + 1]`` is column ``j``):

  - ``a_*``: the permuted matrix's lower triangle, column ``j`` rows ``>= j`` (diagonal first), and
    ``a_source``, the position of each entry in the input matrix's values.
  - ``l_*``: the strictly lower pattern of ``L``, sorted rows; positions are ``l_ptr[j] + t``.
  - ``r_*``: row ``j`` of ``L``: the columns ``k < j`` with ``L[j, k] != 0`` and the position of
    ``L[j, k]``.
  - ``u_*``: the left-looking updates of column ``j``, one lane per entry ``L[i, k]`` with
    ``i >= j`` of every column ``k`` in row ``j``'s pattern: ``w[i] -= L[i, k] * D[k] * L[j, k]``.
    ``u_rows`` is ``i``, ``u_lik`` the position of ``L[i, k]``, ``u_ljk`` that of ``L[j, k]``,
    ``u_k`` the column ``k``.
  """

  n: int
  method: str
  perm: np.ndarray
  iperm: np.ndarray
  a_ptr: np.ndarray
  a_rows: np.ndarray
  a_source: np.ndarray
  parent: np.ndarray
  postorder: np.ndarray
  l_ptr: np.ndarray
  l_rows: np.ndarray
  r_ptr: np.ndarray
  r_cols: np.ndarray
  r_pos: np.ndarray
  u_ptr: np.ndarray
  u_rows: np.ndarray
  u_lik: np.ndarray
  u_ljk: np.ndarray
  u_k: np.ndarray
  _stats: dict[str, float] = field(default_factory=dict, repr=False)

  @property
  def nnz_l(self) -> int:
    """Stored entries of ``L`` below the diagonal."""
    return int(self.l_rows.size)

  @property
  def col_counts(self) -> np.ndarray:
    return np.diff(self.l_ptr)

  def stats(self) -> dict[str, float]:
    """Sizes that decide the schedule: fill, work, tree shape, supernodes, widest tables."""
    if not self._stats:
      counts = self.col_counts
      depth = np.zeros(self.n, dtype=np.int64)
      for j in self.postorder[::-1]:  # parents before children
        p = self.parent[j]
        depth[j] = depth[p] + 1 if p >= 0 else 0
      kids = np.bincount(self.parent[self.parent >= 0], minlength=self.n)
      j = np.arange(1, self.n)
      merges = (self.parent[j - 1] == j) & (counts[j - 1] == counts[j] + 1) & (kids[j] == 1)
      fundamental = self.n - int(merges.sum())
      self._stats.update(
        n=self.n,
        nnz_a=int(self.a_rows.size),
        nnz_l=self.nnz_l,
        fill=self.nnz_l - (int(self.a_rows.size) - self.n),
        flops=float(np.sum(counts.astype(np.float64) * (counts + 3)) / 2 + self.u_rows.size),
        update_lanes=int(self.u_rows.size),
        height=int(depth.max() + 1) if self.n else 0,
        supernodes=int(fundamental),
        max_col=int(counts.max()) if self.n else 0,
        max_row=int(np.diff(self.r_ptr).max()) if self.n else 0,
        max_update=int(np.diff(self.u_ptr).max()) if self.n else 0,
      )
    return dict(self._stats)

  def widths(self) -> np.ndarray:
    """Per column ``(a, u, c)``: matrix entries, update lanes and ``L`` entries."""
    return np.stack([np.diff(self.a_ptr), np.diff(self.u_ptr), np.diff(self.l_ptr)], axis=1)

  def segments(self, cost: CostModel | None = None, *, max_chunks: int = 2048) -> list[Segment]:
    """Split the columns into consecutive segments minimizing padded work plus a price per segment.

    Dynamic programming over segment boundaries: ``best[b] = min over a < b of best[a] + cost of
    [a, b)``, where the cost of a segment is its length times the per-column price of its widest
    tables. Boundaries fall on chunks of ``ceil(n / max_chunks)`` columns, so the search is at most
    ``max_chunks^2`` vectorized steps whatever ``n``."""
    cost = cost or CostModel()
    n = self.n
    if n == 0:
      return []
    w = self.widths()
    size = -(-n // max_chunks)
    bounds = np.arange(0, n + size, size)
    bounds[-1] = n
    bounds = np.unique(bounds)
    chunks = len(bounds) - 1
    chunk_max = np.stack([w[bounds[c] : bounds[c + 1]].max(axis=0) for c in range(chunks)])
    prices = np.array([cost.a, cost.u, cost.c])
    best = np.full(chunks + 1, np.inf)
    best[0] = 0.0
    choice = np.zeros(chunks + 1, dtype=np.int64)
    for b in range(1, chunks + 1):
      widest = np.maximum.accumulate(chunk_max[b - 1 :: -1], axis=0)  # row t: max over chunks [b-1-t, b)
      starts = np.arange(b - 1, -1, -1)
      lengths = bounds[b] - bounds[starts]
      total = best[starts] + lengths * (cost.step + widest @ prices) + cost.segment
      t = int(np.argmin(total))
      best[b], choice[b] = total[t], starts[t]
    cuts = [chunks]
    while cuts[-1] > 0:
      cuts.append(int(choice[cuts[-1]]))
    cuts.reverse()
    out = []
    for lo, hi in zip(cuts[:-1], cuts[1:], strict=True):
      start, stop = int(bounds[lo]), int(bounds[hi])
      a, u, c = (int(x) for x in w[start:stop].max(axis=0))
      out.append(Segment(start, stop, a, u, c))
    return out

  def padded_work(self, segments: list[Segment], cost: CostModel | None = None) -> float:
    """The cost ``segments`` minimizes, for comparing schedules."""
    cost = cost or CostModel()
    return float(sum(s.length * (cost.step + cost.a * s.a + cost.u * s.u + cost.c * s.c) + cost.segment for s in segments))


# Update lanes beyond which the tables would take gigabytes: the ordering is the problem then.
MAX_UPDATE_LANES = 50_000_000


def analyze(
  shape: tuple[int, int],
  rows: np.ndarray,
  cols: np.ndarray,
  method: Ordering = "mmd",
  *,
  perm: np.ndarray | None = None,
  max_update_lanes: int = MAX_UPDATE_LANES,
) -> SymbolicLDL:
  """Analyze a symmetric matrix given by the coordinates of its stored entries, in the order of its
  values (one triangle, both, or any mix; a mirrored pair is read from its lower entry).

  ``method="auto"`` tries every ordering and keeps the least work. ``perm`` overrides ``method``
  with a given ordering, ``perm[new] = old``. An analysis whose
  left-looking updates exceed ``max_update_lanes`` multiply-adds is refused before its tables are
  built."""
  n = int(shape[0])
  if shape[0] != shape[1]:
    raise ValueError(f"LDL^T needs a square matrix, got {shape}")
  if perm is None and method == "auto":
    # Every ordering, keeping the one with the least factorization work: a stage-ordered MPC
    # matrix often factors best as given, a mesh with minimum degree.
    found = []
    for candidate in ORDERINGS:
      try:
        found.append(analyze(shape, rows, cols, candidate, max_update_lanes=max_update_lanes))
      except ValueError:
        continue
    if not found:
      raise ValueError(f"every ordering leaves more than {max_update_lanes} update multiply-adds")
    best = min(found, key=lambda s: (s.u_rows.size, s.nnz_l))
    object.__setattr__(best, "method", f"auto:{best.method}")
    return best
  rows, cols = np.asarray(rows, dtype=np.int64).reshape(-1), np.asarray(cols, dtype=np.int64).reshape(-1)
  if rows.size and (rows.min() < 0 or rows.max() >= n or cols.min() < 0 or cols.max() >= n):
    raise ValueError(f"coordinates out of bounds for shape {shape}")
  if perm is None:
    perm = ordering(n, rows, cols, method)
    name = method
  else:
    perm = np.asarray(perm, dtype=np.int64).reshape(-1)
    name = "given"
    if perm.size != n or not np.array_equal(np.sort(perm), np.arange(n)):
      raise ValueError("perm must be a permutation of range(n)")
  iperm = np.empty(n, dtype=np.int64)
  iperm[perm] = np.arange(n)

  # The permuted lower triangle, each entry pointing back at an input value.
  pr, pc = iperm[rows], iperm[cols]
  lo, hi = np.minimum(pr, pc), np.maximum(pr, pc)
  from_lower = rows >= cols
  keys = lo * n + hi  # column-major over the lower triangle: column lo, row hi
  order = np.lexsort((~from_lower, keys))  # per key, an entry that was in the input's lower triangle first
  first = np.ones(order.size, dtype=bool)
  first[1:] = keys[order][1:] != keys[order][:-1]
  chosen = order[first]
  a_col, a_row = lo[chosen], hi[chosen]
  missing = np.setdiff1d(np.arange(n), a_col[a_row == a_col])
  if missing.size:
    raise ValueError(f"LDL^T without pivoting needs every diagonal entry stored; columns {missing[:8].tolist()} have none")
  a_ptr = np.concatenate([[0], np.cumsum(np.bincount(a_col, minlength=n))]).astype(np.int64)
  a_rows, a_source = a_row, chosen.astype(np.int64)

  # Row lists of the lower triangle: for each row i, the columns k < i with A[i, k] stored.
  strict = a_row > a_col
  row_order = np.lexsort((a_col[strict], a_row[strict]))
  arow_cols = a_col[strict][row_order]
  arow_ptr = np.concatenate([[0], np.cumsum(np.bincount(a_row[strict], minlength=n))]).astype(np.int64)

  parent = _etree(n, arow_ptr, arow_cols)
  postorder = _postorder(parent)
  r_ptr, r_cols = _row_patterns(n, parent, arow_ptr, arow_cols)

  # Columns of L from its rows; rows come out sorted because rows are visited in order.
  l_counts = np.bincount(r_cols, minlength=n)
  l_ptr = np.concatenate([[0], np.cumsum(l_counts)]).astype(np.int64)
  row_of = np.repeat(np.arange(n), np.diff(r_ptr))
  col_order = np.lexsort((row_of, r_cols))
  l_rows = row_of[col_order]
  pos_of_entry = np.empty(r_cols.size, dtype=np.int64)
  pos_of_entry[col_order] = np.arange(r_cols.size)
  r_pos = pos_of_entry  # position of L[j, k] for the row-ordered entry (j, k)

  # Left-looking update lanes: for column j and each k in row j, the entries of column k from row j down.
  counts = l_ptr[r_cols + 1] - r_pos  # entries of column k at rows >= j
  total = int(counts.sum())
  if total > max_update_lanes:
    raise ValueError(
      f"the {name} ordering leaves nnz(L) = {r_cols.size} and {total} update multiply-adds, over the limit of {max_update_lanes}; "
      "use a fill-reducing ordering or raise max_update_lanes"
    )
  u_ptr = np.concatenate([[0], np.cumsum(np.bincount(row_of, weights=counts, minlength=n))]).astype(np.int64)
  starts = np.repeat(r_pos, counts)
  offsets = np.arange(starts.size) - np.repeat(np.cumsum(counts) - counts, counts)
  u_lik = starts + offsets
  u_rows = l_rows[u_lik]
  u_ljk = np.repeat(r_pos, counts)
  u_k = np.repeat(r_cols, counts)
  return SymbolicLDL(
    n, name, perm, iperm, a_ptr, a_rows, a_source, parent, postorder, l_ptr, l_rows, r_ptr, r_cols, r_pos, u_ptr, u_rows, u_lik, u_ljk, u_k
  )


def _etree(n: int, arow_ptr: np.ndarray, arow_cols: np.ndarray) -> np.ndarray:
  """Liu's elimination tree with path compression: ``parent[j]`` is the smallest ``i > j`` with
  ``L[i, j] != 0``, -1 at a root."""
  parent = np.full(n, -1, dtype=np.int64)
  ancestor = np.full(n, -1, dtype=np.int64)
  ptr, cols = arow_ptr.tolist(), arow_cols.tolist()
  par, anc = parent.tolist(), ancestor.tolist()
  for i in range(n):
    for t in range(ptr[i], ptr[i + 1]):
      j = cols[t]
      while j != -1 and j < i:
        nxt = anc[j]
        anc[j] = i
        if nxt == -1:
          par[j] = i
        j = nxt
  return np.asarray(par, dtype=np.int64)


def _postorder(parent: np.ndarray) -> np.ndarray:
  """A postorder of the forest: every node after its descendants, each subtree contiguous."""
  n = parent.size
  children: list[list[int]] = [[] for _ in range(n)]
  roots = []
  for j in range(n - 1, -1, -1):
    (children[parent[j]] if parent[j] >= 0 else roots).append(j)
  out: list[int] = []
  for root in reversed(roots):
    stack = [(root, False)]
    while stack:
      node, done = stack.pop()
      if done:
        out.append(node)
        continue
      stack.append((node, True))
      stack.extend((c, False) for c in children[node])
  return np.asarray(out, dtype=np.int64)


def _row_patterns(n: int, parent: np.ndarray, arow_ptr: np.ndarray, arow_cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Row ``i`` of ``L``: the union of the tree paths from each ``k`` in row ``i`` of ``A`` up to ``i``
  (the row subtree), sorted. Linear in the entries of ``L``."""
  par = parent.tolist()
  ptr, cols = arow_ptr.tolist(), arow_cols.tolist()
  mark = [-1] * n
  counts = [0] * n
  flat: list[int] = []
  for i in range(n):
    mark[i] = i
    found: list[int] = []
    for t in range(ptr[i], ptr[i + 1]):
      j = cols[t]
      while j != -1 and mark[j] != i:
        found.append(j)
        mark[j] = i
        j = par[j]
    found.sort()
    counts[i] = len(found)
    flat.extend(found)
  r_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
  return r_ptr, np.asarray(flat, dtype=np.int64)
