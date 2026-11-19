"""The symbolic analysis of a sparse symmetric factorization: ordering, elimination tree and the pattern of L, from a pattern alone.

``analyze`` runs when a graph is built, on the pattern of a sparse type, and never sees a value, so
the pattern it finds is structural and ignores numerical cancellation. The factorizations in
``scaly.linalg`` and the solver plugins read its tables to decide what their loops or their
straight-line code index. It imports nothing above ``scaly.ir``, so it sits below ``Function``.

The matrix ``A`` is symmetric, given by the stored coordinates of one triangle, both, or any mix:
the analysis works on the pattern of ``A + A^T`` with a full diagonal. It describes
``P A P^T = L D L^T`` (or ``L L^T``) with ``L`` unit lower triangular.
"""

from __future__ import annotations

import heapq
import math
import threading
import weakref
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import reverse_cuthill_mckee

from scaly.ir.types import CanonicalPattern

Ordering = Literal["natural", "rcm", "amd"]
ORDERINGS: tuple[Ordering, ...] = ("natural", "rcm", "amd")


@dataclass(frozen=True, eq=False)
class SymbolicAnalysis:
  """The structure of ``P A P^T = L D L^T`` for a symmetric ``A`` with a fixed pattern.

  Equal tables give the same object while one is alive, so ``is`` compares two analyses, and
  ``digest`` names them.
  Every table is a tuple of ints. A ragged per-column table pairs a pointer tuple with an index
  tuple: ``x_ptr[j]:x_ptr[j + 1]`` is column ``j``.

  - ``perm`` and ``iperm``: the ordering, ``perm[new] = old`` and ``iperm[old] = new``.
  - ``a_ptr``, ``a_rows``, ``a_source``: the lower triangle of the permuted matrix ``P A P^T``,
    column by column with rows ascending, and for each entry the position in the canonical pattern
    of the stored value it reads. Of a pair stored in both triangles, it reads the lower entry. A
    diagonal entry appears only when it is stored.
  - ``parent``: the elimination tree, ``parent[j]`` the first row below the diagonal in column
    ``j`` of ``L``, or -1 at a root.
  - ``postorder``: the nodes of the tree, each after its descendants, every subtree contiguous.
  - ``l_ptr``, ``l_rows``: the strictly lower pattern of ``L``, column by column with rows ascending.
  """

  n: int
  perm: tuple[int, ...]
  iperm: tuple[int, ...]
  a_ptr: tuple[int, ...]
  a_rows: tuple[int, ...]
  a_source: tuple[int, ...]
  parent: tuple[int, ...]
  postorder: tuple[int, ...]
  l_ptr: tuple[int, ...]
  l_rows: tuple[int, ...]
  digest: str

  @property
  def col_counts(self) -> tuple[int, ...]:
    """Entries of each column of ``L`` below the diagonal."""
    return tuple(self.l_ptr[j + 1] - self.l_ptr[j] for j in range(self.n))

  @property
  def nnz_l(self) -> int:
    """Entries of ``L`` below the diagonal."""
    return self.l_ptr[-1]


# Weak, so an analysis lives only as long as something holds it: the pattern of L alone can run to
# tens of megabytes, and a solver plugin keeps only the tree and the column pointers.
_LOCK = threading.Lock()
_BY_PATTERN: weakref.WeakValueDictionary[tuple[str, str], SymbolicAnalysis] = weakref.WeakValueDictionary()
_BY_DIGEST: weakref.WeakValueDictionary[str, SymbolicAnalysis] = weakref.WeakValueDictionary()


def analyze(pattern: CanonicalPattern, ordering: Ordering = "amd") -> SymbolicAnalysis:
  """Analyze the symmetric matrix whose stored coordinates are ``pattern``, under one ordering.

  ``natural`` keeps the given order. ``rcm`` is reverse Cuthill–McKee, SciPy's, which keeps a
  banded or stage-ordered matrix banded. ``amd`` is approximate minimum degree, which reduces the
  fill of ``L`` on general patterns.
  """
  if ordering not in ORDERINGS:
    raise ValueError(f"ordering must be one of {ORDERINGS}, got {ordering!r}")
  if pattern.shape[0] != pattern.shape[1]:
    raise ValueError(f"a symmetric factorization needs a square pattern, got {pattern.shape}")
  key = (pattern.digest, ordering)
  with _LOCK:
    if (cached := _BY_PATTERN.get(key)) is not None:
      return cached
  analysis = _analyze(pattern, ordering)
  with _LOCK:
    analysis = _BY_DIGEST.setdefault(analysis.digest, analysis)
    return _BY_PATTERN.setdefault(key, analysis)


def _analyze(pattern: CanonicalPattern, ordering: Ordering) -> SymbolicAnalysis:
  n = pattern.shape[0]
  rows, cols = np.asarray(pattern.rows, dtype=np.int64), np.asarray(pattern.cols, dtype=np.int64)
  off = rows != cols
  adjacency = sparse.csr_array(
    (np.ones(2 * int(off.sum())), (np.concatenate([rows[off], cols[off]]), np.concatenate([cols[off], rows[off]]))), shape=(n, n)
  )
  adjacency.sum_duplicates()
  adjacency.sort_indices()
  if ordering == "natural" or n == 0:  # SciPy's RCM refuses an empty graph
    perm = np.arange(n, dtype=np.int64)
  elif ordering == "rcm":
    perm = np.asarray(reverse_cuthill_mckee(adjacency, symmetric_mode=True), dtype=np.int64)
  else:
    perm = np.asarray(_amd(n, adjacency.indptr.tolist(), adjacency.indices.tolist()), dtype=np.int64)
  iperm = np.empty(n, dtype=np.int64)
  iperm[perm] = np.arange(n)

  a_col, a_row, a_source = _permuted_lower(n, rows, cols, iperm)
  a_ptr = np.concatenate([[0], np.cumsum(np.bincount(a_col, minlength=n))])

  # Rows of the strict lower triangle: for each row i, the columns k < i with A[i, k] stored.
  strict = a_row > a_col
  order = np.lexsort((a_col[strict], a_row[strict]))
  arow_cols = a_col[strict][order].tolist()
  arow_ptr = np.concatenate([[0], np.cumsum(np.bincount(a_row[strict], minlength=n))]).tolist()

  parent = _etree(n, arow_ptr, arow_cols)
  r_ptr, r_cols = _row_patterns(n, parent, arow_ptr, arow_cols)
  # Columns of L from its rows; each column's rows come out ascending since rows are visited in order.
  row_of = np.repeat(np.arange(n), np.diff(r_ptr))
  r_cols_array = np.asarray(r_cols, dtype=np.int64)
  l_ptr = np.concatenate([[0], np.cumsum(np.bincount(r_cols_array, minlength=n))])
  l_rows = row_of[np.lexsort((row_of, r_cols_array))]

  as_tuple = lambda v: tuple(int(x) for x in v)  # noqa: E731
  tables = {
    "perm": as_tuple(perm),
    "iperm": as_tuple(iperm),
    "a_ptr": as_tuple(a_ptr),
    "a_rows": as_tuple(a_row),
    "a_source": as_tuple(a_source),
    "parent": tuple(parent),
    "postorder": tuple(_postorder(parent)),
    "l_ptr": as_tuple(l_ptr),
    "l_rows": as_tuple(l_rows),
  }
  digest = sha256(repr((n, sorted(tables.items()))).encode("ascii")).hexdigest()
  return SymbolicAnalysis(n=n, digest=digest, **tables)


def _permuted_lower(n: int, rows: np.ndarray, cols: np.ndarray, iperm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """The permuted lower triangle in column-major order: its columns, its rows, and for each entry the
  position of the stored value it reads, of a pair stored in both triangles the lower one."""
  pr, pc = iperm[rows], iperm[cols]
  lo, hi = np.minimum(pr, pc), np.maximum(pr, pc)
  keys = lo * n + hi  # column-major over the lower triangle: column lo, row hi
  order = np.lexsort((rows < cols, keys))  # per key, the entry from the input's lower triangle first
  first = np.ones(order.size, dtype=bool)
  first[1:] = keys[order][1:] != keys[order][:-1]
  chosen = order[first]
  return lo[chosen], hi[chosen], chosen


def _etree(n: int, arow_ptr: list[int], arow_cols: list[int]) -> list[int]:
  """Liu's elimination tree with path compression, from the rows of the strict lower triangle."""
  parent, ancestor = [-1] * n, [-1] * n
  for i in range(n):
    for t in range(arow_ptr[i], arow_ptr[i + 1]):
      j = arow_cols[t]
      while j != -1 and j < i:
        following = ancestor[j]
        ancestor[j] = i
        if following == -1:
          parent[j] = i
        j = following
  return parent


def _postorder(parent: list[int]) -> list[int]:
  """A depth-first postorder of the forest, children in ascending order."""
  n = len(parent)
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
  return out


def _row_patterns(n: int, parent: list[int], arow_ptr: list[int], arow_cols: list[int]) -> tuple[list[int], list[int]]:
  """Row ``i`` of ``L``: the union of the tree paths from each ``k`` in row ``i`` of ``A`` up to
  ``i`` (the row subtree), sorted. Linear in the entries of ``L``."""
  mark = [-1] * n
  ptr, flat = [0], []
  for i in range(n):
    mark[i] = i
    found: list[int] = []
    for t in range(arow_ptr[i], arow_ptr[i + 1]):
      j = arow_cols[t]
      while j != -1 and mark[j] != i:
        found.append(j)
        mark[j] = i
        j = parent[j]
    found.sort()
    flat.extend(found)
    ptr.append(len(flat))
  return ptr, flat


def _amd(n: int, indptr: list[int], indices: list[int]) -> list[int]:
  """Approximate minimum degree (Amestoy, Davis and Duff, 1996) on a symmetric graph without
  self-loops; returns ``perm[new] = old``.

  The quotient graph holds the uneliminated variables and the elements, the cliques that eliminated
  variables leave behind. Each step eliminates the variable of least approximate external degree
  together with its supervariable, absorbs its adjacent elements into the new one, merges variables
  whose adjacency became identical, absorbs elements the new one covers, and updates the degrees of
  the new element's variables with AMD's bound. Rows denser than ``max(16, 10 sqrt(n))`` are left
  out of the graph and ordered last, as AMD does.
  """
  dense_limit = max(16, int(10 * math.sqrt(n)))
  dense = [indptr[i + 1] - indptr[i] > dense_limit for i in range(n)]
  var_adj: list[set[int]] = [{j for j in indices[indptr[i] : indptr[i + 1]] if not dense[j]} for i in range(n)]
  elem_adj: list[set[int]] = [set() for _ in range(n)]  # the elements adjacent to each variable
  elem_vars: dict[int, set[int]] = {}  # the principal variables of each element
  elem_size: dict[int, int] = {}  # their total weight
  weight = [1] * n  # the size of each supervariable
  members = [[i] for i in range(n)]
  alive = [not d for d in dense]  # an uneliminated principal variable
  degree = [len(a) for a in var_adj]
  remaining = sum(alive)
  heap = [(degree[i], i) for i in range(n) if alive[i]]
  heapq.heapify(heap)
  order: list[int] = []

  while heap:
    d, p = heapq.heappop(heap)
    if not alive[p] or d != degree[p]:
      continue
    # The new element: p's variables and the variables of every element adjacent to p.
    new = set(var_adj[p])
    for e in elem_adj[p]:
      new |= elem_vars[e]
      for v in elem_vars[e]:
        if v != p:
          elem_adj[v].discard(e)
      del elem_vars[e], elem_size[e]
    new.discard(p)
    alive[p] = False
    remaining -= weight[p]
    order.extend(members[p])
    var_adj[p], elem_adj[p] = set(), set()
    for i in new:
      var_adj[i] -= new
      var_adj[i].discard(p)  # an edge between two of the new variables is covered by the element
      elem_adj[i].add(p)

    # Supervariables: variables of the new element with identical adjacency are indistinguishable.
    buckets: dict[int, list[int]] = {}
    for i in sorted(new):
      buckets.setdefault(hash((sum(var_adj[i]), len(var_adj[i]), sum(elem_adj[i]), len(elem_adj[i]))), []).append(i)
    for candidates in buckets.values():
      while len(candidates) > 1:
        i, rest = candidates[0], []
        for j in candidates[1:]:
          if var_adj[j] == var_adj[i] and elem_adj[j] == elem_adj[i]:
            weight[i] += weight[j]
            members[i] += members[j]
            alive[j] = False
            new.discard(j)
            for v in var_adj[j]:
              var_adj[v].discard(j)
            for e in elem_adj[j]:
              if e != p:
                elem_vars[e].discard(j)
            var_adj[j], elem_adj[j] = set(), set()
          else:
            rest.append(j)
        candidates = rest
    elem_vars[p] = new
    elem_size[p] = new_weight = sum(weight[i] for i in new)

    # |e \ new| for every other element next to the new one; an element inside it is absorbed.
    outside: dict[int, int] = {}
    for i in new:
      for e in elem_adj[i]:
        if e != p:
          outside[e] = outside.get(e, elem_size[e]) - weight[i]
    for e, size in outside.items():
      if size == 0:
        for v in elem_vars[e]:
          elem_adj[v].discard(e)
        del elem_vars[e], elem_size[e]
    for i in new:
      external = new_weight - weight[i]
      bound = sum(weight[j] for j in var_adj[i]) + external + sum(outside[e] for e in elem_adj[i] if e != p)
      degree[i] = min(bound, remaining - weight[i])
      heapq.heappush(heap, (degree[i], i))

  return order + [i for i in range(n) if dense[i]]
