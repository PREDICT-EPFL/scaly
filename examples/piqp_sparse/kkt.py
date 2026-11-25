"""The KKT system with every constraint eliminated, PIQP 0.7.0's ``sparse::KKT`` in mode ``KKT_ALL_ELIMINATED``.

Eliminating the equality and inequality duals leaves one symmetric positive definite matrix
over the variables,

    Psi = P + diag(x_reg) + A^T A / delta + G^T diag(z_reg)^-1 G,

which an ``LDL^T`` factorization solves. As in PIQP, everything that depends only on the
patterns is decided when the solver is generated: the pattern of ``Psi``, its fill-reducing
ordering, and the tables of the loops that fill ``A^T A`` and ``G^T diag(z_reg)^-1 G`` column by
column. The values are what the generated code computes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, cached_property
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from scipy import sparse

import scaly as sc
from scaly import linalg

from .data import Data, Structure
from .utils import all_true, index, zeros

if TYPE_CHECKING:
  from scaly.ir.types import CanonicalPattern


class Factor(NamedTuple):
  """``Psi``'s factorization ``P Psi P^T = L D L^T`` and whether every pivot is nonzero, PIQP's test of success."""

  L: sc.Expr
  d: sc.Expr
  ok: sc.Expr


I64 = sc.TensorType((), sc.dtypes.int64)


@dataclass(frozen=True)
class _MtM:
  """The pattern of ``M^T M``'s upper triangle and the tables of PIQP's column loop that fills it.

  ``rows`` and ``cols`` list the entries ``(i, j)``, ``i <= j``, row by row. The loop reads ``M`` by
  columns (``c_ptr``, and for each entry its row ``c_row`` and its position ``c_pos`` among ``M``'s
  row-major values) and by rows (``r_ptr``, ``r_col``), and writes ``M^T M`` by columns: ``h_ptr``,
  and for each entry its row ``h_row`` and its position ``h_pos`` in ``rows``/``cols``.
  """

  rows: np.ndarray
  cols: np.ndarray
  tables: tuple[np.ndarray, ...]  # c_ptr, c_row, c_pos, r_ptr, r_col, h_ptr, h_row, h_pos


def _mt_m(m: int, n: int, rows: np.ndarray, cols: np.ndarray) -> _MtM:
  """PIQP's ``(data.AT * A).triangularView<Upper>()`` pattern, and the tables to fill its values."""
  as_int = lambda xs: np.asarray(xs, dtype=np.int64)  # noqa: E731
  if not len(rows):
    empty = as_int([])
    return _MtM(empty, empty, ())
  M = sparse.csr_array((np.ones(len(rows)), (rows, cols)), shape=(m, n))
  upper = sparse.triu(M.T @ M).tocsr()
  upper.sort_indices()
  out_rows, out_cols = as_int(np.repeat(np.arange(n), np.diff(upper.indptr))), as_int(upper.indices)
  by_col = np.lexsort((rows, cols))  # M by columns, rows ascending
  h_order = np.lexsort((out_rows, out_cols))  # M^T M by columns, rows ascending
  tables = (
    as_int(np.searchsorted(cols[by_col], np.arange(n + 1))),
    as_int(rows[by_col]),
    as_int(by_col),
    as_int(np.searchsorted(rows, np.arange(m + 1))),
    as_int(cols),
    as_int(np.searchsorted(out_cols[h_order], np.arange(n + 1))),
    as_int(out_rows[h_order]),
    as_int(h_order),
  )
  return _MtM(out_rows, out_cols, tables)


# PIQP's update_AT_A and update_GT_W_delta_inv_G: for each column j of M^T M, every row k with an
# entry M[k, j] adds M[k, j] * M[k, i] (/ z_reg[k]) to a dense work vector at every i <= j of row k,
# and column j is then read out of the work vector, which is cleared for the next column. The
# entries i <= j of row k are the start of row k up to M[k, j] itself, so the inner loop runs over
# exactly those, where PIQP skips the others with `if (i > j) continue`.


@sc.function(sc.arg("tmp_scatter", dtype=...), sc.arg("t", I64), sc.arg("M", dtype=...), sc.arg("Mkj", dtype=...), sc.arg("r_col", dtype=...), sc.arg("start", I64), name="kkt_mt_m_add")
def _add(tmp_scatter: sc.Expr, t: sc.Expr, M: sc.Expr, Mkj: sc.Expr, r_col: sc.Expr, start: sc.Expr) -> sc.Expr:
  # tmp_scatter(i) += M(k, j) * M(k, i)
  q = start + t
  return sc.scatter(tmp_scatter, sc.gather(r_col, q), Mkj * sc.gather(M, q), combine="add")


@sc.function(
  sc.arg("tmp_scatter", dtype=...), sc.arg("t", I64), sc.arg("M", dtype=...), sc.arg("Mkj", dtype=...), sc.arg("z_reg_k", dtype=...), sc.arg("r_col", dtype=...), sc.arg("start", I64),
  name="kkt_mt_w_m_add",
)  # fmt: skip
def _add_scaled(tmp_scatter: sc.Expr, t: sc.Expr, M: sc.Expr, Mkj: sc.Expr, z_reg_k: sc.Expr, r_col: sc.Expr, start: sc.Expr) -> sc.Expr:
  # tmp_scatter(i) += G(k, j) * G(k, i) / z_reg(k)
  q = start + t
  return sc.scatter(tmp_scatter, sc.gather(r_col, q), Mkj * sc.gather(M, q) / z_reg_k, combine="add")


@cache
def _row(scaled: bool) -> sc.Function:
  """Row ``k``, the ``t``-th row with an entry in column ``j``; ``z_reg`` is unused for ``A^T A``."""

  @sc.function(
    sc.arg("tmp_scatter", dtype=...), sc.arg("t", I64), sc.arg("M", dtype=...), sc.arg("z_reg", dtype=...),
    sc.group(*(sc.arg(name, dtype=...) for name in ("c_row", "c_pos", "r_ptr", "r_col"))), sc.arg("start", I64),
    name="kkt_mt_w_m_row" if scaled else "kkt_mt_m_row",
  )  # fmt: skip
  def row(tmp_scatter: sc.Expr, t: sc.Expr, M: sc.Expr, z_reg: sc.Expr, tables: tuple[sc.Expr, ...], start: sc.Expr) -> sc.Expr:
    c_row, c_pos, r_ptr, r_col = tables
    p = start + t
    k, pos = sc.gather(c_row, p), sc.gather(c_pos, p)
    first = sc.gather(r_ptr, k)
    b = sc.broadcast
    xs = (b(M), b(sc.gather(M, pos)), *((b(sc.gather(z_reg, k)),) if scaled else ()), b(r_col), b(first))
    tmp_scatter, _ = sc.scan(_add_scaled if scaled else _add, tmp_scatter, xs, length=tmp_scatter.shape[0], trip=pos - first + 1, index=True)
    return tmp_scatter

  return row


@sc.function(
  sc.arg("MtM", dtype=...), sc.arg("tmp_scatter", dtype=...), sc.arg("t", I64), sc.arg("h_row", dtype=...), sc.arg("h_pos", dtype=...), sc.arg("start", I64),
  name="kkt_mt_m_store",
)  # fmt: skip
def _store(MtM: sc.Expr, tmp_scatter: sc.Expr, t: sc.Expr, h_row: sc.Expr, h_pos: sc.Expr, start: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
  # MtM(i, j) = tmp_scatter(i); tmp_scatter(i) = 0
  p = start + t
  i = sc.gather(h_row, p)
  return sc.scatter(MtM, sc.gather(h_pos, p), sc.gather(tmp_scatter, i), combine="set"), sc.scatter(tmp_scatter, i, sc.const(0.0), combine="set")


def _ragged(body: sc.Function, carry: Any, ptr: sc.Expr, j: sc.Expr, length: int, *xs: Any) -> Any:
  """``body`` over the entries ``ptr[j] <= p < ptr[j + 1]``, taking ``start = ptr[j]`` last."""
  start, stop = sc.gather(ptr, j), sc.gather(ptr, j + 1)
  out, _ = sc.scan(body, carry, (*xs, sc.broadcast(start)), length=length, trip=stop - start, index=True)
  return out


@cache
def _column(scaled: bool) -> sc.Function:
  """Column ``j`` of ``M^T M``."""

  @sc.function(
    sc.arg("MtM", dtype=...), sc.arg("tmp_scatter", dtype=...), sc.arg("j", I64), sc.arg("M", dtype=...), sc.arg("z_reg", dtype=...),
    sc.group(*(sc.arg(name, dtype=...) for name in ("c_ptr", "c_row", "c_pos", "r_ptr", "r_col", "h_ptr", "h_row", "h_pos"))),
    name="kkt_mt_w_m_column" if scaled else "kkt_mt_m_column",
  )  # fmt: skip
  def column(MtM: sc.Expr, tmp_scatter: sc.Expr, j: sc.Expr, M: sc.Expr, z_reg: sc.Expr, tables: tuple[sc.Expr, ...]) -> tuple[sc.Expr, sc.Expr]:
    c_ptr, c_row, c_pos, r_ptr, r_col, h_ptr, h_row, h_pos = tables
    # column j of M has at most m entries, column j of M^T M at most n
    m, n, b = r_ptr.shape[0] - 1, tmp_scatter.shape[0], sc.broadcast
    tmp_scatter = _ragged(_row(scaled), tmp_scatter, c_ptr, j, m, b(M), b(z_reg), tuple(b(x) for x in (c_row, c_pos, r_ptr, r_col)))
    return _ragged(_store, (MtM, tmp_scatter), h_ptr, j, n, b(h_row), b(h_pos))

  return column


def _update_MT_M(M: sc.Expr, z_reg: sc.Expr | None, t: _MtM, n: int) -> sc.Expr:
  """``M^T M``'s upper triangle, or ``M^T diag(z_reg)^-1 M``'s, in ``t.rows``/``t.cols`` order."""
  scaled = z_reg is not None
  init = (zeros(len(t.rows)), zeros(n))
  xs = (sc.broadcast(M), sc.broadcast(z_reg if scaled else zeros(1)), tuple(sc.broadcast(index(table)) for table in t.tables))
  (MtM, _), _ = sc.scan(_column(scaled), init, xs, length=n, index=True)
  return MtM


@dataclass(frozen=True, eq=False)
class KKT:
  """The all-eliminated KKT system of one QP structure."""

  structure: Structure

  @cached_property
  def _AtA(self) -> _MtM:
    s = self.structure
    return _mt_m(s.p, s.n, s.A_rows, s.A_cols)

  @cached_property
  def _GtG(self) -> _MtM:
    s = self.structure
    return _mt_m(s.m, s.n, s.G_rows, s.G_cols)

  @cached_property
  def _upper(self) -> tuple[np.ndarray, np.ndarray]:
    """The coordinates ``(i, j)``, ``i <= j``, of ``Psi``'s upper triangle: ``P``, the diagonal, ``A^T A`` and ``G^T G``."""
    s = self.structure
    rows = np.concatenate([s.P_rows, np.arange(s.n), self._AtA.rows, self._GtG.rows])
    cols = np.concatenate([s.P_cols, np.arange(s.n), self._AtA.cols, self._GtG.cols])
    pattern = sparse.coo_array((np.ones(len(rows)), (rows, cols)), shape=(s.n, s.n)).tocsr()
    pattern.sum_duplicates()
    coo = pattern.tocoo()
    return coo.row.astype(np.int64), coo.col.astype(np.int64)

  @cached_property
  def pattern(self) -> CanonicalPattern:
    """``Psi``'s upper triangle, which ``scaly.linalg.ldl`` reads as the mirror of the lower one."""
    rows, cols = self._upper
    pattern, order = sc.SparsityPattern((self.structure.n, self.structure.n), rows=tuple(rows.tolist()), cols=tuple(cols.tolist())).canonicalize()
    assert tuple(order) == tuple(range(len(rows)))
    return pattern

  @cached_property
  def perm(self) -> np.ndarray:
    """PIQP's ``AMDOrdering``: ``perm[new] = old``, the ordering ``linalg.ldl`` applies to ``Psi``."""
    return linalg.permutation(self.pattern, ordering="amd")

  @cached_property
  def iperm(self) -> np.ndarray:
    return np.argsort(self.perm)

  def _targets(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """PIQP's ``P_utri_to_Ki`` and friends: where each upper entry lands among ``Psi``'s values."""
    upper_rows, upper_cols = self._upper
    where = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(upper_rows, upper_cols, strict=True))}
    return np.asarray([where[(int(i), int(j))] for i, j in zip(rows, cols, strict=True)], dtype=np.int64)

  @cached_property
  def _P_to_Ki(self) -> np.ndarray:
    return self._targets(self.structure.P_rows, self.structure.P_cols)

  @cached_property
  def _diagonal_to_Ki(self) -> np.ndarray:
    n = np.arange(self.structure.n)
    return self._targets(n, n)

  @cached_property
  def _AtA_to_Ki(self) -> np.ndarray:
    return self._targets(self._AtA.rows, self._AtA.cols)

  @cached_property
  def _GtG_to_Ki(self) -> np.ndarray:
    return self._targets(self._GtG.rows, self._GtG.cols)

  # Values.

  def init_AtA(self, data: Data) -> sc.Expr:
    """The values of ``A^T A``'s upper triangle, PIQP's ``update_AT_A``, which it runs once per setup; each factorization scales them by ``1 / delta``."""
    if not len(self._AtA.rows):
      return zeros(0)
    return _update_MT_M(data.A, None, self._AtA, self.structure.n)

  def update_GT_W_delta_inv_G(self, data: Data, z_reg: sc.Expr) -> sc.Expr:
    """The values of ``G^T diag(z_reg)^-1 G``'s upper triangle, PIQP's ``update_GT_W_delta_inv_G``."""
    return _update_MT_M(data.G, z_reg, self._GtG, self.structure.n)

  def assemble(self, data: Data, AtA: sc.Expr, delta: sc.Expr, x_reg: sc.Expr, z_reg: sc.Expr) -> sc.Expr:
    """``Psi``'s values in PIQP's order of addition: ``P``, then ``x_reg`` on the diagonal, ``A^T A / delta`` and ``G^T diag(z_reg)^-1 G``."""
    nnz = self.pattern.nnz
    values = sc.segment_sum(data.P, index(self._P_to_Ki), nnz) if self.structure.P.nnz else zeros(nnz)
    values = sc.scatter(values, index(self._diagonal_to_Ki), x_reg, combine="add")
    if len(self._AtA_to_Ki):
      values = sc.scatter(values, index(self._AtA_to_Ki), (1.0 / delta) * AtA, combine="add")
    if len(self._GtG_to_Ki):
      values = sc.scatter(values, index(self._GtG_to_Ki), self.update_GT_W_delta_inv_G(data, z_reg), combine="add")
    return values

  def update_scalings_and_factor(self, data: Data, AtA: sc.Expr, delta: sc.Expr, x_reg: sc.Expr, z_reg: sc.Expr) -> Factor:
    """Assemble ``Psi`` and factor it, PIQP's ``update_scalings_and_factor``: ``linalg.ldl`` reorders it by ``perm`` and factors ``P Psi P^T``.

    Success means no zero pivot, as in PIQP: ``scaly.linalg``'s own status also refuses a pivot
    that is tiny next to its diagonal entry, which happens late in a solve as ``rho`` falls to
    ``1e-10``.
    """
    L, d, _ = linalg.ldl(sc.pack(self.assemble(data, AtA, delta, x_reg, z_reg), self.pattern), ordering="amd")
    return Factor(L, d, all_true(sc.not_equal(d, 0.0)))

  def solve_ldlt_in_place(self, factor: Factor, b: sc.Expr) -> sc.Expr:
    """``Psi^-1 b``, PIQP's ``ordering.perm``, ``LDLt::solve_inplace`` (``lsolve``, ``dsolve``, ``ltsolve``) and ``ordering.permt``."""
    y = sc.gather(b, index(self.perm))
    y = linalg.solve_triangular(factor.L, y, lower=True, unit_diagonal=True)
    y = y * (1.0 / factor.d)
    y = linalg.solve_triangular(factor.L, y, lower=True, trans=True, unit_diagonal=True)
    return sc.gather(y, index(self.iperm))

  def solve(
    self, data: Data, factor: Factor, delta: sc.Expr, z_reg: sc.Expr, rhs_x: sc.Expr, rhs_y: sc.Expr, rhs_z: sc.Expr
  ) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    """Solve the regularized KKT system for ``(dx, dy, dz)`` with the duals eliminated, PIQP's ``KKT::solve`` in this mode.

    As in PIQP, the products here are single ones, ``data.GT * work_z`` and ``data.AT.transpose() * lhs_x``,
    not ``KKTSystem``'s paired ``eval_*`` products.
    """
    s = self.structure
    delta_inv = 1.0 / delta
    z_reg_inv = 1.0 / z_reg

    rhs = rhs_x
    if s.m:
      rhs = rhs + product(data.G, s.G, z_reg_inv * rhs_z, transpose=True)
    if s.p:
      rhs = rhs + delta_inv * product(data.A, s.A, rhs_y, transpose=True)

    lhs_x = self.solve_ldlt_in_place(factor, rhs)

    lhs_y = delta_inv * product(data.A, s.A, lhs_x) - delta_inv * rhs_y
    lhs_z = (product(data.G, s.G, lhs_x) - rhs_z) * z_reg_inv
    return lhs_x, lhs_y, lhs_z


def product(values: sc.Expr, pattern: CanonicalPattern, x: sc.Expr, *, transpose: bool = False) -> sc.Expr:
  """``M x``, or ``M^T x`` with ``transpose``, for the sparse matrix storing ``values`` in ``pattern``.

  One loop over the stored entries, reading two index tables of ``nnz`` entries. It adds the terms
  of each output in the order PIQP's CSC products do. ``sc.pack(values, pattern) @ x`` computes
  the same product, but inlined into a larger function such as the KKT solve, scaly currently emits
  one small loop per row, which made the generated C 43% larger over the Maros-Meszaros problems.
  """
  rows, cols = pattern.shape
  if not pattern.nnz:
    return zeros(cols if transpose else rows)
  r, c = np.asarray(pattern.rows, dtype=np.int64), np.asarray(pattern.cols, dtype=np.int64)
  if transpose:
    r, c = c, r
  # out[r[k]] += values[k] * x[c[k]], in the stored order
  return sc.segment_sum(values * sc.gather(x, index(c)), index(r), cols if transpose else rows)
