"""What an iteration of each IPM backend costs, counted from a problem's structure: the generation-time choice between them."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...ir.target import PORTABLE
from ...linalg.symbolic import TooMuchWork
from .kkt import Backend, dense_rows, kkt_symbolic
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


def _assembly(rows: np.ndarray, count: int, n: int) -> int:
  """The products of ``M^T W M`` for a matrix of ``count`` rows with entries in rows ``rows``."""
  if not count:
    return 0
  if dense_rows(rows, count, n):
    return count * n * n // 4
  per_row = np.bincount(rows, minlength=count).astype(np.int64)
  return int(per_row @ per_row)


def work(s: QPStructure) -> Work:
  """The counts of ``s``'s iteration on either backend; raises ``TooMuchWork`` when the sparse
  backend's factorization is past what ``linalg.symbolic`` will generate."""
  ldl = kkt_symbolic(s)
  return Work(
    vectors=s.n + s.p + s.m,
    entries=int(s.P_rows.size + s.A_rows.size + s.G_rows.size),
    assembly=_assembly(s.A_rows, s.p, s.n) + _assembly(s.G_rows, s.m, s.n),
    factor=s.n**3 // 3,
    solve=s.n**2,
    updates=int(ldl.update_lanes),
    nnz_l=int(ldl.nnz_l),
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


def iteration_us(w: Work, backend: Backend) -> float:
  """The model's time of one iteration of ``backend``, in microseconds on the reference machine,
  whose straight-line budget decides whether the dense factor is straight-line code."""
  if backend == "sparse":
    return float(np.dot(SPARSE_WEIGHTS, (1.0, w.vectors, w.entries, w.updates, w.nnz_l)))
  straight = w.factor < PORTABLE.straight_line_ops
  return float(np.dot(DENSE_WEIGHTS, (1.0, w.vectors, w.entries, w.assembly, 0.0 if straight else w.factor, w.factor if straight else 0.0, w.solve)))


def choose_backend(s: QPStructure) -> Backend:
  """The backend whose iteration the model finds cheaper for ``s``, or the dense one when the
  sparse factorization is past what can be generated. The choice decides the graph, which is built
  before any target is known, and the weights are the reference machine's, the only one they were
  measured on; so it is the same for every target."""
  try:
    w = work(s)
  except TooMuchWork:
    return "dense"
  return min(("sparse", "dense"), key=lambda backend: iteration_us(w, backend))


__all__ = ["DENSE_WEIGHTS", "SPARSE_WEIGHTS", "Work", "choose_backend", "iteration_us", "work"]
