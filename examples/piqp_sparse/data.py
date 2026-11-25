"""A sparse QP's data, PIQP 0.7.0's ``sparse/data.hpp``: the patterns fixed at generation time and the values a solve reads.

The QP is

    minimize    1/2 x^T P x + c^T x
    subject to  A x = b,  h_l <= G x <= h_u,  x_l <= x <= x_u,

and a bound at or beyond ``INF`` in magnitude is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from scipy import sparse

import scaly as sc

from .utils import BOOL, F64, INF, index

if TYPE_CHECKING:
  from scaly.ir.types import CanonicalPattern


def _canonical(matrix: Any, shape: tuple[int, int]) -> tuple[CanonicalPattern, np.ndarray]:
  """The row-major pattern of a SciPy matrix's stored entries, explicit zeros included, and its values in that order."""
  csr = sparse.csr_array(matrix, shape=shape, dtype=np.float64)
  csr.sum_duplicates()
  csr.sort_indices()
  rows = np.repeat(np.arange(shape[0]), np.diff(csr.indptr))
  pattern, order = sc.SparsityPattern(shape, rows=tuple(rows.tolist()), cols=tuple(csr.indices.tolist())).canonicalize()
  assert tuple(order) == tuple(range(len(rows))), "CSR order with sorted indices is row-major"
  return pattern, np.array(csr.data)


def _upper(P: Any) -> sparse.coo_array:
  """PIQP's ``P.triangularView<Upper>()``: the stored entries on and above the diagonal."""
  coo = sparse.coo_array(P)
  keep = coo.row <= coo.col
  return sparse.coo_array((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=P.shape)


@dataclass(frozen=True, eq=False)
class Structure:
  """The sizes and stored patterns of a QP, which decide the generated code: values never enter."""

  n: int  # number of variables
  p: int  # number of equality constraints
  m: int  # number of inequality constraints

  P: CanonicalPattern  # pattern of the upper triangular part of P
  A: CanonicalPattern  # pattern of A, row-major (PIQP stores A transpose)
  G: CanonicalPattern  # pattern of G, row-major (PIQP stores G transpose)

  @classmethod
  def from_matrices(cls, P: Any, A: Any, G: Any) -> tuple[Structure, np.ndarray, np.ndarray, np.ndarray]:
    """The structure of ``P``'s upper triangle, ``A`` and ``G``, and their stored values in its order."""
    n = P.shape[0]
    P_pattern, P_values = _canonical(_upper(P), (n, n))
    A_pattern, A_values = _canonical(A, (A.shape[0], n))
    G_pattern, G_values = _canonical(G, (G.shape[0], n))
    return cls(n, A.shape[0], G.shape[0], P_pattern, A_pattern, G_pattern), P_values, A_values, G_values

  def values(self, which: str, matrix: Any) -> np.ndarray:
    """The stored values of a new ``P``, ``A`` or ``G``, which must keep this structure's pattern."""
    pattern = getattr(self, which)
    new, values = _canonical(_upper(matrix) if which == "P" else matrix, pattern.shape)
    if new is not pattern:
      raise ValueError(f"{which} has a different sparsity pattern than at setup")
    return values

  @cached_property
  def P_rows(self) -> np.ndarray:
    return np.asarray(self.P.rows, dtype=np.int64)

  @cached_property
  def P_cols(self) -> np.ndarray:
    return np.asarray(self.P.cols, dtype=np.int64)

  @cached_property
  def A_rows(self) -> np.ndarray:
    return np.asarray(self.A.rows, dtype=np.int64)

  @cached_property
  def A_cols(self) -> np.ndarray:
    return np.asarray(self.A.cols, dtype=np.int64)

  @cached_property
  def G_rows(self) -> np.ndarray:
    return np.asarray(self.G.rows, dtype=np.int64)

  @cached_property
  def G_cols(self) -> np.ndarray:
    return np.asarray(self.G.cols, dtype=np.int64)

  @cached_property
  def P_diagonal(self) -> np.ndarray:
    """For each variable, the position of ``P``'s stored diagonal entry, or ``-1``."""
    positions = np.full(self.n, -1, dtype=np.int64)
    on_diagonal = self.P_rows == self.P_cols
    positions[self.P_rows[on_diagonal]] = np.flatnonzero(on_diagonal)
    return positions

  @cached_property
  def P_strict(self) -> tuple[CanonicalPattern, np.ndarray]:
    """The pattern of ``P``'s entries above the diagonal and their positions among its values."""
    above = np.flatnonzero(self.P_rows < self.P_cols)
    pattern, order = sc.SparsityPattern(
      (self.n, self.n), rows=tuple(self.P_rows[above].tolist()), cols=tuple(self.P_cols[above].tolist())
    ).canonicalize()
    assert tuple(order) == tuple(range(len(above)))
    return pattern, above


class Data(NamedTuple):
  """The values of a QP in a structure's patterns, after PIQP's ``set_h_l``, ``set_h_u``, ``disable_inf_constraints``, ``set_x_l`` and ``set_x_u``.

  The masks name the finite bounds. PIQP lists their indices instead and keeps the finite box
  bounds first; a mask keeps one entry per row or variable, with ``x_l`` and ``x_u`` zero where
  the bound is infinite. ``x_b_scaling`` is the box bounds' row of the equilibrated problem.
  """

  P: sc.Expr  # values of the upper triangular part of P, in the structure's order
  c: sc.Expr
  A: sc.Expr  # values of A, in the structure's order
  b: sc.Expr
  G: sc.Expr  # values of G, in the structure's order, zero on disabled rows
  h_l: sc.Expr
  h_u: sc.Expr
  x_l: sc.Expr  # lower box bounds, zero where infinite (PIQP: finite bounds first)
  x_u: sc.Expr  # upper box bounds, zero where infinite (PIQP: finite bounds first)

  x_b_scaling: sc.Expr  # scaling of x_l and x_u, i.e. x_l <= x_b_scaling .* x <= x_u

  # which bounds are finite (PIQP: h_l_idx, h_u_idx, x_l_idx, x_u_idx, the indexes of the finite bounds)
  h_l_mask: sc.Expr
  h_u_mask: sc.Expr
  x_l_mask: sc.Expr
  x_u_mask: sc.Expr


def data_types(s: Structure) -> tuple[sc.TensorType, ...]:
  """The type of each field of ``Data`` for a structure, in order."""
  vectors = (sc.TensorType((size,), F64) for size in (s.P.nnz, s.n, s.A.nnz, s.p, s.G.nnz, s.m, s.m, s.n, s.n, s.n))
  masks = (sc.TensorType((size,), BOOL) for size in (s.m, s.m, s.n, s.n))
  return (*vectors, *masks)


def make_data(
  s: Structure, P: sc.Expr, c: sc.Expr, A: sc.Expr, b: sc.Expr, G: sc.Expr, h_l: sc.Expr, h_u: sc.Expr, x_l: sc.Expr, x_u: sc.Expr
) -> Data:
  """The data PIQP's ``setup`` stores, before equilibration.

  A row of ``G`` with both bounds infinite becomes ``-1 <= 0 x <= 1``, which keeps the pattern, as
  PIQP's ``disable_inf_constraints`` does.
  """
  h_l_finite, h_u_finite = h_l > -INF, h_u < INF
  disabled = ~h_l_finite & ~h_u_finite
  if s.G.nnz:
    G = sc.where(sc.gather(disabled, index(s.G_rows)), 0.0, G)
  h_l = sc.where(disabled, -1.0, sc.where(h_l_finite, h_l, -INF))
  h_u = sc.where(disabled, 1.0, sc.where(h_u_finite, h_u, INF))
  x_l_mask, x_u_mask = x_l > -INF, x_u < INF
  return Data(
    P=P,
    c=c,
    A=A,
    b=b,
    G=G,
    h_l=h_l,
    h_u=h_u,
    x_l=sc.where(x_l_mask, x_l, 0.0),
    x_u=sc.where(x_u_mask, x_u, 0.0),
    x_b_scaling=sc.const(np.ones(s.n)),
    h_l_mask=h_l_finite | disabled,
    h_u_mask=h_u_finite | disabled,
    x_l_mask=x_l_mask,
    x_u_mask=x_u_mask,
  )
