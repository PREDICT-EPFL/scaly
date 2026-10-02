"""What an iteration of each IPM backend costs, counted from a problem's structure: the generation-time choice between them."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ...ir.target import PORTABLE
from ...linalg.symbolic import TooMuchWork
from .kkt import Backend, Kernels, dense_rows, kkt_symbolic
from .stages import stages
from .structure import QPStructure


@dataclass(frozen=True)
class Work:
  """The counts one iteration of either backend is made of, all known before any code is written.

  Attributes:
    vectors: ``n + p + m``, the length of the vectors every step updates.
    entries: the stored entries of ``P`` (upper triangle), ``A`` and ``G``, which the residuals read.
    assembly: the products of the dense backend's condensed matrix ``A^T A / delta + G^T W G``. A
      matrix multiplied through its index tables takes one outer product per row, its rows'
      nonzeros squared; one multiplied as a dense array (``kkt.dense_rows``) counts a quarter of
      ``rows * n^2``, a dense multiply-add being a quarter of a product through the tables.
    factor: ``n^3 / 3``, the dense backend's Cholesky factor.
    solve: ``n^2``, one pair of triangular solves with it.
    updates: the sparse backend's update multiply-adds (``SymbolicLDL.update_lanes``).
    nnz_l: the stored entries of its ``L``, what one pair of triangular solves reads.
  """

  vectors: int
  entries: int
  assembly: int
  factor: int
  solve: int
  updates: int
  nnz_l: int


@dataclass(frozen=True)
class StageWork:
  """The counts one iteration of the stagewise backend is made of, from the partition ``stages``
  finds and the dense arrays ``Kernels.stage_dense`` takes.

  Attributes:
    vectors: ``n + p + m``, as for the other backends.
    entries: the stored entries of ``P``, ``A`` and ``G``.
    cells: the block storage, which a factorization zeroes and scatters the condensed matrix into.
    pairs: the products of two entries of one row of ``A`` or ``G`` that are taken one by one.
    arrays: the entries of the dense arrays, gathered at each factorization and multiplied by a
      vector in the residuals and the solves.
    products: the multiply-adds of the arrays' products with themselves.
    factor: the block factorization's multiply-adds (``Stages.factor_work``).
    solve: one pair of block triangular solves' (``Stages.solve_work``).
  """

  vectors: int
  entries: int
  cells: int
  pairs: int
  arrays: int
  products: int
  factor: int
  solve: int


def stage_work(s: QPStructure) -> StageWork:
  """The counts of ``s``'s iteration on the stagewise backend. Nothing is generated for them: the
  partition and the arrays are read from the patterns."""
  st = stages(s)
  kernels = Kernels(s, "stagewise")
  pairs = arrays = products = 0
  for which, rows, count in (("A", s.A_rows, s.p), ("G", s.G_rows, s.m)):
    per_row = np.bincount(rows, minlength=count).astype(np.int64)
    found = kernels.stage_dense(which)
    taken = np.zeros(count, dtype=np.int64) if found is None else np.bincount(rows[found.taken], minlength=count).astype(np.int64)
    pairs += int((per_row * (per_row + 1) // 2 - taken * (taken + 1) // 2).sum())
    if found is not None:
      arrays += int(found.entries.size)
      products += found.work
  return StageWork(
    vectors=s.n + s.p + s.m,
    entries=int(s.P_rows.size + s.A_rows.size + s.G_rows.size),
    cells=st.cells,
    pairs=pairs,
    arrays=arrays,
    products=products,
    factor=st.factor_work,
    solve=st.solve_work,
  )


def stage_terms(s: QPStructure) -> tuple[float, ...]:
  """What ``STAGEWISE_WEIGHTS`` multiply, in their order: a constant, then ``StageWork``'s counts."""
  w = stage_work(s)
  return (1.0, w.vectors, w.entries, w.cells, w.pairs, w.arrays, w.products, w.factor, w.solve)


def _assembly(rows: np.ndarray, count: int, n: int) -> int:
  """The products of ``M^T W M`` for a matrix of ``count`` rows with entries in rows ``rows``."""
  if not count:
    return 0
  if dense_rows(rows, count, n):
    return count * n * n // 4
  per_row = np.bincount(rows, minlength=count).astype(np.int64)
  return int(per_row @ per_row)


def work(s: QPStructure) -> Work:
  """The counts of ``s``'s iteration on the dense and the sparse backend; raises ``TooMuchWork``
  when the sparse backend's factorization is past what ``linalg.symbolic`` will generate."""
  ldl = kkt_symbolic(s)
  return replace(_dense_work(s), updates=int(ldl.update_lanes), nnz_l=int(ldl.nnz_l))


def _dense_work(s: QPStructure) -> Work:
  """The counts the dense backend's iteration is made of, the sparse backend's left at zero."""
  return Work(
    vectors=s.n + s.p + s.m,
    entries=int(s.P_rows.size + s.A_rows.size + s.G_rows.size),
    assembly=_assembly(s.A_rows, s.p, s.n) + _assembly(s.G_rows, s.m, s.n),
    factor=s.n**3 // 3,
    solve=s.n**2,
    updates=0,
    nnz_l=0,
  )


# Microseconds per unit of each count, fitted on the reference machine to the 55 problems of the IPM
# speed study (``internal/notes/perf_2026_09_30_gaps/backend_fit.py``: the faster backend on 53, the
# worst pick 1.17x the better, and the same leaving each problem out of its own fit). Dense: a
# constant, vectors, entries, the assembly, the factor as loops, the factor as straight-line code,
# the solve. Sparse: a constant, vectors, entries, the update multiply-adds, the entries of ``L``.
# A zero is a count the fit found no time in beside the others. Refit when either backend's
# generated code changes speed: the weights before the dense factor went into blocks took the
# sparse backend for four problems the dense one had become 1.1-1.2x faster on.
DENSE_WEIGHTS = (1.041e-1, 2.237e-3, 2.404e-3, 3.711e-4, 2.615e-5, 0.0, 2.485e-3)
SPARSE_WEIGHTS = (6.997e-2, 0.0, 0.0, 5.457e-5, 9.041e-3)
# The stagewise backend's, in ``stage_terms``'s order: a constant, vectors, entries, the cells of the
# block storage, the pairs of entries, the arrays' entries, the arrays' products, the block
# factorization, the solves. The two weights of multiply-adds are the rates the profile shows at
# blocks of 28 to 42 (the arrays' products at 25 G a second, the factorization at 14); on a family
# of multistage problems alone they are not told apart from the counts that grow with the square of
# a block, and a fit without them would call one block of a hundred rows cheap. The others are
# fitted to 24 multistage problems of 3 to 52 slots a block
# (``internal/notes/perf_2026_09_30_gaps/stagewise_fit.py``): the prediction is within 0.70 to 1.36
# of the measured iteration.
STAGEWISE_WEIGHTS = (0.0, 2.920e-2, 0.0, 2.989e-3, 9.718e-4, 1.359e-3, 4.0e-5, 7.0e-5, 0.0)
# The fewest blocks for the stagewise backend to be a candidate. With one or two it is the dense
# backend's factorization behind another assembly, and its weights were not measured there.
STAGEWISE_BLOCKS = 3


def iteration_us(w: Work, backend: Backend) -> float:
  """The model's time of one iteration of the dense or the sparse backend, in microseconds on the
  reference machine, whose straight-line budget decides whether the dense factor is straight-line
  code. The stagewise backend's is ``stagewise_us``, from counts of its own."""
  if backend == "stagewise":
    raise ValueError("the stagewise backend's time is stagewise_us(s): its counts are not a Work's")
  if backend == "sparse":
    return float(np.dot(SPARSE_WEIGHTS, (1.0, w.vectors, w.entries, w.updates, w.nnz_l)))
  straight = w.factor < PORTABLE.straight_line_ops
  return float(np.dot(DENSE_WEIGHTS, (1.0, w.vectors, w.entries, w.assembly, 0.0 if straight else w.factor, w.factor if straight else 0.0, w.solve)))


def stagewise_us(s: QPStructure) -> float:
  """The model's time of one iteration of the stagewise backend on ``s``, in microseconds on the
  reference machine."""
  return float(np.dot(STAGEWISE_WEIGHTS, stage_terms(s)))


def choose_backend(s: QPStructure) -> Backend:
  """The backend whose iteration the model finds cheapest for ``s``: the sparse one, the dense one,
  or, for a problem whose partition has ``STAGEWISE_BLOCKS`` blocks or more, the stagewise one. The
  sparse backend is left out when its factorization is past what can be generated. The choice
  decides the graph, which is built before any target is known, and the weights are the reference
  machine's, the only one they were measured on; so it is the same for every target."""
  try:
    w = work(s)
    times: dict[Backend, float] = {"sparse": iteration_us(w, "sparse"), "dense": iteration_us(w, "dense")}
  except TooMuchWork:
    times = {"dense": iteration_us(_dense_work(s), "dense")}
  if stages(s).K >= STAGEWISE_BLOCKS:
    times["stagewise"] = stagewise_us(s)
  return min(times, key=times.__getitem__)


__all__ = [
  "DENSE_WEIGHTS",
  "SPARSE_WEIGHTS",
  "STAGEWISE_BLOCKS",
  "STAGEWISE_WEIGHTS",
  "StageWork",
  "Work",
  "choose_backend",
  "iteration_us",
  "stage_terms",
  "stage_work",
  "stagewise_us",
  "work",
]
