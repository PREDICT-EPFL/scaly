"""The static structure of a QP for the generated interior-point solver, and its run-time values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...ir.expr import Expr, as_expr, gather, where
from ...linalg.sparse import csc_coordinates

INF = 1e30
"""PIQP's infinity: a bound at or beyond it is absent."""


@dataclass(frozen=True, eq=False)
class QPStructure:
  """What a generated solver is specialised to: the problem

      minimize 1/2 x^T P x + c^T x  subject to  A x = b,  h_l <= G x <= h_u,  x_l <= x <= x_u

  with ``P`` given by its upper triangle, the sparsity patterns of ``P``, ``A`` and ``G``, and which
  bounds are finite. The values, and the finite bounds, are run-time inputs. A row of ``G`` with
  both bounds absent is kept as a zero row with bounds [-1, 1], as PIQP keeps it: it still counts
  in the complementarity measure.
  """

  n: int
  p: int
  m: int
  P_rows: np.ndarray  # upper triangle, CSC order
  P_cols: np.ndarray
  A_rows: np.ndarray
  A_cols: np.ndarray
  G_rows: np.ndarray
  G_cols: np.ndarray
  h_l_given: np.ndarray  # bool (m,): the lower bound is finite
  h_u_given: np.ndarray
  x_l_given: np.ndarray  # bool (n,)
  x_u_given: np.ndarray

  @staticmethod
  def from_patterns(P: Any, A: Any, G: Any, *, h_l: Any, h_u: Any, x_l: Any, x_u: Any) -> QPStructure:
    """From the patterns of ``P`` (its upper triangle is used), ``A`` and ``G``, in any form
    ``linalg.sparse.csc_coordinates`` reads (a ``SparsityType``, a ``SparseMatrix``, a SciPy matrix or a
    boolean mask), and the finiteness of the bounds: boolean masks, or bound vectors whose entries at
    or beyond ``INF`` are absent."""
    n = int(P.shape[0])
    p, m = int(A.shape[0]), int(G.shape[0])
    pr, pc = csc_coordinates(P, (n, n))
    upper = pr <= pc
    pr, pc = pr[upper], pc[upper]

    def given(v: Any, size: int, sign: float) -> np.ndarray:
      arr = np.asarray(v)
      if arr.dtype == bool:
        return arr.reshape(size).copy()
      return (sign * arr.reshape(size) < INF) & np.isfinite(arr.reshape(size))

    ar, ac = csc_coordinates(A, (p, n))
    gr, gc = csc_coordinates(G, (m, n))
    return QPStructure(n, p, m, pr, pc, ar, ac, gr, gc, given(h_l, m, -1.0), given(h_u, m, 1.0), given(x_l, n, -1.0), given(x_u, n, 1.0))

  @property
  def free_rows(self) -> np.ndarray:
    """Rows of ``G`` with neither bound: zeroed, with bounds [-1, 1]."""
    return ~self.h_l_given & ~self.h_u_given

  @property
  def h_l_idx(self) -> np.ndarray:
    """Rows with a lower bound, free rows included (their bound is -1)."""
    return np.flatnonzero(self.h_l_given | self.free_rows)

  @property
  def h_u_idx(self) -> np.ndarray:
    """Rows with an upper bound, free rows included (their bound is 1)."""
    return np.flatnonzero(self.h_u_given | self.free_rows)

  @property
  def x_l_idx(self) -> np.ndarray:
    """Variables with a finite lower bound: the order of ``QPValues.x_l``."""
    return np.flatnonzero(self.x_l_given)

  @property
  def x_u_idx(self) -> np.ndarray:
    """Variables with a finite upper bound: the order of ``QPValues.x_u``."""
    return np.flatnonzero(self.x_u_given)

  @property
  def n_bounds(self) -> int:
    """The number of complementarity pairs: the denominator of mu."""
    return int(self.h_l_idx.size + self.h_u_idx.size + self.x_l_idx.size + self.x_u_idx.size)


@dataclass(frozen=True, eq=False)
class QPValues:
  """Run-time values of a ``QPStructure``'s problem, after PIQP's preprocessing: ``P``, ``A`` and ``G``
  values in the structure's entry order (free rows of ``G`` zeroed), dense ``c`` and ``b``, ``h_l`` and
  ``h_u`` of length ``m`` (free rows at -1 and 1, absent bounds at -INF and INF), and the finite box
  bounds packed in index order (``x_l`` holds the ``x_l_idx`` entries)."""

  P: Expr
  c: Expr
  A: Expr
  b: Expr
  G: Expr
  h_l: Expr
  h_u: Expr
  x_l: Expr
  x_u: Expr

  @staticmethod
  def preprocess(s: QPStructure, *, P: Any, c: Any, A: Any, b: Any, G: Any, h_l: Any, h_u: Any, x_l: Any, x_u: Any) -> QPValues:
    """From raw values: bound vectors of full length (entries of absent bounds are ignored)."""
    G = as_expr(G)
    if s.free_rows.any():
      G = G * Expr.const((~s.free_rows[s.G_rows]).astype(np.float64))
    # ``where`` rather than a product with a mask: an absent bound may arrive as an infinity.
    h_l = where(Expr.const(s.h_l_given, dtype="bool"), as_expr(h_l), Expr.const(np.where(s.free_rows, -1.0, -INF)))
    h_u = where(Expr.const(s.h_u_given, dtype="bool"), as_expr(h_u), Expr.const(np.where(s.free_rows, 1.0, INF)))
    return QPValues(
      P=as_expr(P),
      c=as_expr(c),
      A=as_expr(A),
      b=as_expr(b),
      G=G,
      h_l=h_l,
      h_u=h_u,
      x_l=gather(as_expr(x_l), s.x_l_idx),
      x_u=gather(as_expr(x_u), s.x_u_idx),
    )
