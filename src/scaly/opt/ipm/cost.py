"""What an iteration of each IPM backend costs, counted from a problem's structure: the generation-time choice between them."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...ir.target import PORTABLE
from ...linalg.symbolic import TooMuchWork
from .kkt import Backend, kkt_symbolic
from .structure import QPStructure


@dataclass(frozen=True)
class Work:
  """The counts one iteration of either backend is made of, all known before any code is written.

  Attributes:
    vectors: ``n + p + m``, the length of the vectors every step updates.
    entries: the stored entries of ``P`` (upper triangle), ``A`` and ``G``, which the residuals read.
    assembly: the multiply-adds of the dense backend's condensed matrix ``A^T A / delta + G^T W G``,
      one outer product per row of ``A`` and ``G``: the sum of their rows' nonzeros squared.
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


def work(s: QPStructure) -> Work:
  """The counts of ``s``'s iteration on either backend; raises ``TooMuchWork`` when the sparse
  backend's factorization is past what ``linalg.symbolic`` will generate."""
  rows_a = np.bincount(s.A_rows, minlength=s.p) if s.p else np.zeros(0, dtype=np.int64)
  rows_g = np.bincount(s.G_rows, minlength=s.m) if s.m else np.zeros(0, dtype=np.int64)
  ldl = kkt_symbolic(s)
  return Work(
    vectors=s.n + s.p + s.m,
    entries=int(s.P_rows.size + s.A_rows.size + s.G_rows.size),
    assembly=int(np.sum(rows_a.astype(np.int64) ** 2) + np.sum(rows_g.astype(np.int64) ** 2)),
    factor=s.n**3 // 3,
    solve=s.n**2,
    updates=int(ldl.update_lanes),
    nnz_l=int(ldl.nnz_l),
  )


# Microseconds per unit of each count, fitted on the reference machine to the 55 problems of the IPM
# speed study (``internal/notes/perf_2026_09_30_gaps/backend_fit.py``: the faster backend on 53, the
# worst pick 1.07x the better; 52 and 1.20x leaving each problem out of its own fit). Dense: a
# constant, vectors, entries, the assembly, the factor as loops, the factor as straight-line code,
# the solve. Sparse: a constant, vectors, entries, the update multiply-adds, the entries of ``L``.
DENSE_WEIGHTS = (1.29e-1, 6.44e-3, 3.16e-3, 1.79e-4, 4.71e-5, 8.43e-5, 2.72e-3)
SPARSE_WEIGHTS = (7.71e-2, 1.09e-2, 0.0, 9.89e-5, 7.94e-3)


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
