"""PIQP's KKT system as generated code, behind a dense, a sparse and a stagewise backend."""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np

from ...function import ConcreteFunction
from ...function.sugar import vmap, while_loop
from ...ir.expr import (
  Expr,
  as_expr,
  concat,
  gather,
  isfinite,
  logical_and,
  logical_not,
  logical_or,
  maximum,
  minimum,
  norm_inf,
  scatter,
  segment_sum,
  stack,
  where,
)
from ...linalg import SparseLDL, SparseMatrix, cho_solve, cholesky
from ...linalg.blocks import BlockTridiagonalCholesky
from ...linalg.symbolic import SymbolicLDL, analyze
from .ruiz import ScaledQP
from .stages import DenseBlocks, Stages, dense_blocks, pairs_in_rows, stages
from .structure import QPStructure

Backend = Literal["dense", "sparse", "stagewise"]
EPS = float(np.finfo(np.float64).eps)
# How a factorization went, as the retry loop's header holds it under ``ok``: below one half is a failure.
_FACTORED, _RESIDUE, _SINGULAR = 1.0, 0.25, 0.0
HEADER = ("rho", "delta", "reg_limit", "ir", "retries", "tried", "ok", "changed")
"""The retry loop's scalars: the regularization it ends with, whether refinement is on, the retries
spent, whether an attempt was made and how it went (``_FACTORED``, ``_RESIDUE`` or ``_SINGULAR``),
and whether a retry changed ``rho`` and ``delta``."""


@dataclass(frozen=True)
class Refinement:
  """PIQP's factorization retries and iterative refinement. A factorization that fails turns
  refinement on: the factored matrix gets a static regularization, and every solve is refined
  against the matrix without it. Each later failure scales ``rho`` and ``delta`` by 100, at most
  ``max_factor_retires`` times, and raises the regularization floor towards ``reg_limit_cap``."""

  always: bool = False
  eps_abs: float = 1e-12
  eps_rel: float = 1e-12
  max_iter: int = 10
  min_improvement_rate: float = 5.0
  static_eps: float = 1e-8
  static_rel: float = EPS * EPS
  max_factor_retires: int = 10
  reg_limit_cap: float = 1e-8


@dataclass(frozen=True, eq=False)
class Iterate:
  """Primal-dual variables in PIQP's layout: ``x`` (n), ``y`` (p), ``z_l``, ``z_u``, ``s_l``, ``s_u``
  (m, zero on rows without that bound), and ``z_bl``, ``s_bl`` (the finite lower box bounds, in
  index order) and ``z_bu``, ``s_bu`` (the upper ones). Also a right-hand side of the KKT system."""

  x: Expr
  y: Expr
  z_l: Expr
  z_u: Expr
  z_bl: Expr
  z_bu: Expr
  s_l: Expr
  s_u: Expr
  s_bl: Expr
  s_bu: Expr

  def fields(self) -> tuple[Expr, ...]:
    """The ten vectors in PIQP's order."""
    return (self.x, self.y, self.z_l, self.z_u, self.z_bl, self.z_bu, self.s_l, self.s_u, self.s_bl, self.s_bu)

  @staticmethod
  def sizes(s: QPStructure) -> tuple[int, ...]:
    """The lengths of ``fields`` for structure ``s``."""
    nl, nu = s.x_l_idx.size, s.x_u_idx.size
    return (s.n, s.p, s.m, s.m, nl, nu, s.m, s.m, nl, nu)

  def flat(self) -> Expr:
    """The fields as one vector, in order."""
    return concat([f.reshape((f.size,)) for f in self.fields()])

  @staticmethod
  def unflat(s: QPStructure, v: Expr) -> Iterate:
    """The iterate ``flat`` made, back from its vector."""
    offsets = np.cumsum([0, *Iterate.sizes(s)])
    return Iterate(*(v[int(a) : int(b)] for a, b in zip(offsets[:-1], offsets[1:], strict=True)))


# The fewest columns, and the fewest products of the sparse form, a dense ``M^T W M`` pays from. Both
# are the reference machine's, like ``cost.py``'s weights: the graph is built before any target is known.
DENSE_PRODUCT_COLUMNS = 8
DENSE_PRODUCT_WORK = 4096


def dense_rows(rows: np.ndarray, count: int, n: int) -> bool:
  """Whether ``M^T W M`` for a matrix of ``count`` rows and ``n`` columns, with entries in rows
  ``rows``, is cheaper as a dense product than through its index tables. The sparse form multiplies
  each pair of entries of a row, ``sum(nnz_row^2)`` products, at about four times a dense
  multiply-add's cost (PRIMALC2's seven dense rows of 231: 373 000 products in 164 us against
  43 us), so the dense form wins from a quarter of ``count * n^2``. Two floors, both measured in the
  solver: under eight columns the dense product has no whole register tile to run in (DUALC2,
  seven columns: 1.06x slower), and under 4 096 products the sparse form is too little work for the
  dense one's copies to pay (LOTSCHD, 532 products: 1.2x slower). The measurements are
  ``internal/notes/perf_2026_09_30_gaps/results/assembly_c216.txt`` and ``ipm_dense_c216_rule*.md``."""
  per_row = np.bincount(np.asarray(rows, dtype=np.int64), minlength=count).astype(np.float64)
  products = float(per_row @ per_row)
  return n >= DENSE_PRODUCT_COLUMNS and products >= DENSE_PRODUCT_WORK and 4.0 * products >= float(count) * n * n


def _where_rows(mask: np.ndarray, value: Expr) -> Expr:
  """``value`` on the rows of ``mask``, exactly zero elsewhere (an infinity there is discarded)."""
  return where(Expr.const(mask, dtype="bool"), value, 0.0)


def _split(v: Expr, sizes: tuple[int, ...]) -> list[Expr]:
  offsets = np.cumsum([0, *sizes])
  return [v[int(a) : int(b)] for a, b in zip(offsets[:-1], offsets[1:], strict=True)]


def _call(fn: ConcreteFunction, *args: Expr) -> Expr:
  out = fn.symbolic_call(tuple(args))
  return out[0] if isinstance(out, tuple) else out


class StageArrays:
  """The entries of a constraint matrix that ``Kernels.stage_dense`` takes, as one dense array per
  block (``stages.DenseBlocks``), and the products they make: with a vector, with a vector from
  the left, and each array with itself. The other entries stay a sparse matrix."""

  def __init__(self, name: str, st: Stages, found: DenseBlocks, rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int], weighted: bool):
    K, r, nd = found.entries.shape
    self.found, self.shape = found, shape
    self.held, self.rest = np.flatnonzero(found.taken), np.flatnonzero(~found.taken)
    self._rows, self._cols = rows, cols
    # The variable each column of a block's array stands for, and the cell of the arrays' products each row reads.
    self.variable = np.where(found.slots >= 0, st.order[np.maximum(found.slots, 0)], -1).reshape(-1)
    self.row_cell = np.full(shape[0], K * r, dtype=np.int64)
    in_array = np.flatnonzero(found.rows.reshape(-1) >= 0)
    self.row_cell[found.rows.reshape(-1)[in_array]] = in_array
    self.in_array = in_array
    block, x, y = Expr.sym("J", (r, nd)), Expr.sym("x", (nd,)), Expr.sym("y", (r,))
    self._times = ConcreteFunction.from_exprs(f"{name}_times", [block, x], [block @ x], ["J", "x"], ["Jx"])
    self._t_times = ConcreteFunction.from_exprs(f"{name}_t_times", [block, y], [block.T @ y], ["J", "y"], ["Jty"])
    if weighted:
      self._square = ConcreteFunction.from_exprs(f"{name}_square", [block, y], [(block.T * y) @ block], ["J", "w"], ["JtWJ"])
    else:
      self._square = ConcreteFunction.from_exprs(f"{name}_square", [block], [block.T @ block], ["J"], ["JtJ"])

  def arrays(self, values: Expr) -> Expr:
    """The arrays, flat, over the matrix's values in entry order."""
    entries = self.found.entries
    return gather(concat([values, Expr.const(np.zeros(1))]), np.where(entries >= 0, entries, values.size).reshape(-1))

  def other(self, values: Expr) -> SparseMatrix | None:
    """The entries in no array, as a matrix of the whole shape."""
    if not self.rest.size:
      return None
    return SparseMatrix.from_coo(self._rows[self.rest], self._cols[self.rest], gather(values, self.rest), self.shape)

  def _per_row(self, v: Expr) -> Expr:
    rows = self.found.rows.reshape(-1)
    return gather(concat([v, Expr.const(np.zeros(1))]), np.where(rows >= 0, rows, v.size))

  def times(self, values: Expr, x: Expr) -> Expr:
    """``M x``."""
    K, r, nd = self.found.entries.shape
    xs = gather(concat([x, Expr.const(np.zeros(1))]), np.where(self.variable >= 0, self.variable, x.size))
    out = gather(concat([vmap(self._times, K, [(self.arrays(values), 0, r * nd), (xs, 0, nd)]), Expr.const(np.zeros(1))]), self.row_cell)
    other = self.other(values)
    return out if other is None else out + other @ x

  def t_times(self, values: Expr, y: Expr) -> Expr:
    """``M^T y``. A variable can be in two blocks' arrays, and their shares add."""
    K, r, nd = self.found.entries.shape
    shares = vmap(self._t_times, K, [(self.arrays(values), 0, r * nd), (self._per_row(y), 0, r)])
    there = np.flatnonzero(self.variable >= 0)
    out = scatter(gather(shares, there), self.variable[there], (self.shape[1],))
    other = self.other(values)
    return out if other is None else out + other.T @ y

  def squares(self, values: Expr, weights: Expr | None) -> Expr:
    """Each array's ``J^T W J``, flat ``(K * nd * nd,)``; ``weights`` is per row of the matrix."""
    K, r, nd = self.found.entries.shape
    specs = [(self.arrays(values), 0, r * nd)]
    if weights is not None:
      specs.append((self._per_row(weights), 0, r))
    return vmap(self._square, K, specs)


class Matrices:
  """``P`` (its upper triangle), ``A`` and ``G`` of a structure over values in its entry order.
  ``staged`` holds, for the stagewise backend, the dense arrays of ``A`` and of ``G`` where they
  have them: their products with a vector then run through the arrays."""

  def __init__(self, s: QPStructure, P: Expr, A: Expr, G: Expr, staged: dict[str, StageArrays] | None = None):
    self.s = s
    self.staged = staged or {}
    self.P = SparseMatrix.from_coo(s.P_rows, s.P_cols, P, (s.n, s.n))
    self.A = SparseMatrix.from_coo(s.A_rows, s.A_cols, A, (s.p, s.n))
    self.G = SparseMatrix.from_coo(s.G_rows, s.G_cols, G, (s.m, s.n))
    self.P_values, self.A_values, self.G_values = P, A, G  # in the structure's entry order
    diag = np.flatnonzero(s.P_rows == s.P_cols)
    self.P_diag = scatter(gather(P, diag), s.P_rows[diag], (s.n,)) if diag.size else Expr.const(np.zeros(s.n))
    off = np.flatnonzero(s.P_rows != s.P_cols)
    self.P_lower = SparseMatrix.from_coo(s.P_cols[off], s.P_rows[off], gather(P, off), (s.n, s.n)) if off.size else None

  def P_times(self, x: Expr) -> Expr:
    """``P x`` from the upper triangle."""
    return self.P @ x + self.P_lower @ x if self.P_lower is not None else self.P @ x

  def full_P(self) -> SparseMatrix:
    """``P`` with both triangles stored."""
    return self.P + self.P_lower if self.P_lower is not None else self.P

  def A_times(self, x: Expr) -> Expr:
    """``A x``."""
    return self.staged["A"].times(self.A_values, x) if "A" in self.staged else self.A @ x

  def At_times(self, y: Expr) -> Expr:
    """``A^T y``."""
    return self.staged["A"].t_times(self.A_values, y) if "A" in self.staged else self.A.T @ y

  def G_times(self, x: Expr) -> Expr:
    """``G x``."""
    return self.staged["G"].times(self.G_values, x) if "G" in self.staged else self.G @ x

  def Gt_times(self, z: Expr) -> Expr:
    """``G^T z``."""
    return self.staged["G"].t_times(self.G_values, z) if "G" in self.staged else self.G.T @ z


class Kernels:
  """The generated pieces every KKT system over one structure shares, each a ``Function`` over
  symbols, so that the initial point and the loop call the same code: one factorization attempt
  inside PIQP's retry loop, the solve, and the solve's iterative refinement.

  ``dense`` factors the condensed matrix ``P + diag(x_reg) + A^T A / delta + G^T diag(1/z_reg) G`` by
  Cholesky, as PIQP's dense backend does; ``sparse`` factors the whole KKT matrix by ``SparseLDL``,
  as its sparse backend does; ``stagewise`` factors the condensed matrix block by block in the
  order ``stages`` finds, in which it is block tridiagonal (``linalg.blocks``), as PIQP's multistage
  backend does. The problem travels as one data vector ``[P | A | G | x_b]``, a
  factorization as a record ``[delta, delta_reg | x_reg | z_reg | z_reg_ir | factor]``: ``x_reg``
  with the static regularization, ``z_reg`` without it, as PIQP keeps them."""

  def __init__(self, s: QPStructure, backend: Backend, refinement: Refinement | None = None, *, name: str = "ipm"):
    if backend not in ("dense", "sparse", "stagewise"):
      raise ValueError(f"backend must be 'dense', 'sparse' or 'stagewise', got {backend!r}")
    self.s, self.backend, self.refinement, self.name = s, backend, refinement or Refinement(), name
    has_l, has_u = np.zeros(s.m, dtype=bool), np.zeros(s.m, dtype=bool)
    has_l[s.h_l_idx], has_u[s.h_u_idx] = True, True
    self.has_l, self.has_u = has_l, has_u
    nl, nu = s.x_l_idx.size, s.x_u_idx.size
    self.d_sizes = (s.P_rows.size, s.A_rows.size, s.G_rows.size, s.n)
    self.v_sizes = (s.m, s.m, nl, nu, s.m, s.m, nl, nu, 1)  # never empty: a loop param of its own
    self.size = s.n + s.p + s.m
    self._ldl: SparseLDL | None = None
    self._blocks: BlockTridiagonalCholesky | None = None

  # --- layouts -----------------------------------------------------------------------------------

  def data(self, q: ScaledQP) -> Expr:
    """The problem data a kernel takes: ``[P | A | G | x_b]`` of a scaled problem."""
    return concat([e for e in (q.values.P, q.values.A, q.values.G, q.x_b) if e.size])

  def matrices(self, d: Expr) -> tuple[Matrices, Expr]:
    """``P``, ``A`` and ``G`` over a data vector, and the box scaling ``x_b``."""
    P, A, G, x_b = _split(d, self.d_sizes)
    return self.matrices_of(P, A, G), x_b

  def matrices_of(self, P: Expr, A: Expr, G: Expr) -> Matrices:
    """``P``, ``A`` and ``G`` over values in the structure's entry order, with this backend's products."""
    return Matrices(self.s, P, A, G, self.stage_arrays)

  @cached_property
  def stage_arrays(self) -> dict[str, StageArrays]:
    """For the stagewise backend, the dense arrays of ``A`` and of ``G``, where ``stage_dense`` finds them."""
    s, out = self.s, {}
    if self.backend == "stagewise":
      for which, rows, cols, count in (("A", s.A_rows, s.A_cols, s.p), ("G", s.G_rows, s.G_cols, s.m)):
        found = self.stage_dense(which)
        if found is not None:
          out[which] = StageArrays(f"{self.name}_stage_{which}", stages(s), found, rows, cols, (count, s.n), weighted=which == "G")
    return out

  def bounds(self, it: Iterate) -> Expr:
    """The duals and slacks a factorization reads, as one vector."""
    parts = (it.z_l, it.z_u, it.z_bl, it.z_bu, it.s_l, it.s_u, it.s_bl, it.s_bu, Expr.const(np.zeros(1)))
    return concat([e for e in parts if e.size])

  def _unbounds(self, v: Expr) -> Iterate:
    z_l, z_u, z_bl, z_bu, s_l, s_u, s_bl, s_bu, _ = _split(v, self.v_sizes)
    none = Expr.const(np.zeros(0))
    return Iterate(none, none, z_l, z_u, z_bl, z_bu, s_l, s_u, s_bl, s_bu)

  @property
  def record_sizes(self) -> tuple[int, ...]:
    """The lengths of a factorization record's parts: ``delta``, ``delta_reg``, ``x_reg``, ``z_reg``, ``z_reg_ir``, the factor."""
    s = self.s
    return (1, 1, s.n, s.m, s.m, self._factorization[1])

  def record(self, rec: Expr) -> dict[str, Expr]:
    """A factorization record's parts by name."""
    delta, delta_reg, x_reg, z_reg, z_reg_ir, factor = _split(rec, self.record_sizes)
    return {"delta": delta[0], "delta_reg": delta_reg[0], "x_reg": x_reg, "z_reg": z_reg, "z_reg_ir": z_reg_ir, "factor": factor}

  # --- the factorization -------------------------------------------------------------------------

  @cached_property
  def _factorization(self) -> tuple[ConcreteFunction, int]:
    """The factorization as a ``Function`` of ``(data, x_reg, delta_reg, z_reg_ir)`` returning the
    factor and two tests of it (1 or 0): PIQP's own, and one that also fails a factor that has
    lost every digit of a pivot (``_digits_left``); and the size of the factor."""
    s = self.s
    d, xr, dr, zr = Expr.sym("D", (sum(self.d_sizes),)), Expr.sym("x_reg", (s.n,)), Expr.sym("delta_reg", ()), Expr.sym("z_reg_ir", (s.m,))
    mats, _ = self.matrices(d)
    if self.backend == "dense":
      dense = self._condensed(mats, xr, dr, zr)
      f = cholesky(dense).reshape((s.n * s.n,))
      diag = np.arange(s.n) * (s.n + 1)
      l_diag = gather(f, diag)
      # Eigen's LLT fails on a pivot that is not positive; a NaN one fails here too.
      ok = where(l_diag > 0.0, 0.0, 1.0).sum() < 0.5
      digits = logical_and(ok, self._digits_left(l_diag, gather(dense.reshape((s.n * s.n,)), diag)))
    elif self.backend == "stagewise":
      blocks, below, c_diag = self._stage_blocks(mats, xr, dr, zr)
      self._blocks = BlockTridiagonalCholesky(blocks, below, name=f"{self.name}_kkt")
      f = self._blocks.values
      l_diag = self._blocks.diagonal
      # A pivot that is not positive fails, as BLASFEO's Cholesky does for PIQP's multistage backend.
      ok = where(l_diag > 0.0, 0.0, 1.0).sum() < 0.5
      digits = logical_and(ok, self._digits_left(l_diag, c_diag))
    else:
      kkt = self.kkt_matrix(mats, xr, dr, zr)
      self._ldl = SparseLDL(kkt, symbolic=kkt_symbolic(s), name=f"{self.name}_kkt")
      f = self._ldl.values
      pivots = f[self._ldl.d_offset : self._ldl.d_offset + self.size]
      symbolic = self._ldl.symbolic
      first = symbolic.a_ptr[:-1]  # a column of the permuted matrix starts at its diagonal entry
      assert np.array_equal(symbolic.a_rows[first], np.arange(self.size)), "the KKT matrix stores its whole diagonal"
      # PIQP's sparse LDL^T fails only on a pivot that is exactly zero: an infinite one, from a dual
      # at zero, passes, and so does a NaN one.
      ok = where(pivots.abs() <= 0.0, 1.0, 0.0).sum() < 0.5
      digits = self._pivots_left(pivots, gather(kkt.values, symbolic.a_source[first]))
    names = ["D", "x_reg", "delta_reg", "z_reg_ir"]
    outs = [f, where(ok, 1.0, 0.0), where(digits, 1.0, 0.0)]
    fn = ConcreteFunction.from_exprs(f"{self.name}_kkt_factor", [d, xr, dr, zr], outs, names, ["factor", "ok", "digits"])
    return fn, int(f.size)

  def _condensed(self, mats: Matrices, xr: Expr, dr: Expr, zr: Expr) -> Expr:
    """The condensed matrix ``P + diag(x_reg) + A^T A / delta + G^T diag(1 / z_reg) G`` as a dense
    array; the Cholesky reads its lower triangle. A product whose matrix has dense rows
    (``dense_rows``) is a dense product, which the lowering runs in register tiles, and then
    ``P``'s stored upper triangle goes straight into the lower one, transposed. With no such
    product the matrix is assembled through its index tables, both triangles, as before."""
    s = self.s
    dense_a, dense_g = (bool(count) and dense_rows(rows, count, s.n) for rows, count in ((s.A_rows, s.p), (s.G_rows, s.m)))
    if not (dense_a or dense_g):
      c = mats.full_P().add_diagonal(xr)
      if s.p:
        c = c + (mats.A.T @ mats.A) * (1.0 / dr)
      if s.m:
        c = c + mats.G.T @ mats.G.scale_rows(1.0 / zr)
      return c.to_dense()
    lower = mats.P.T.add_diagonal(xr)
    dense: Expr | None = None
    if dense_a:
      a = mats.A.to_dense()
      dense = (a.T @ a) * (1.0 / dr)
    elif s.p:
      lower = lower + (mats.A.T @ mats.A) * (1.0 / dr)
    if dense_g:
      g = mats.G.to_dense()
      product = (g.T * (1.0 / zr)) @ g
      dense = product if dense is None else dense + product
    elif s.m:
      lower = lower + mats.G.T @ mats.G.scale_rows(1.0 / zr)
    assert dense is not None
    return lower.to_dense() + dense

  def stage_dense(self, which: Literal["A", "G"]) -> DenseBlocks | None:
    """The part of ``A`` or ``G`` the stagewise backend multiplies as dense arrays, one per block,
    or None when its products stay in the index tables: the rule is ``dense_rows``'s, on the
    columns ``dense_blocks`` finds mostly filled."""
    s = self.s
    rows, cols, count = (s.A_rows, s.A_cols, s.p) if which == "A" else (s.G_rows, s.G_cols, s.m)
    found = dense_blocks(stages(s), rows, cols, count)
    if found is None or found.entries.shape[2] < DENSE_PRODUCT_COLUMNS or found.products < DENSE_PRODUCT_WORK or 4 * found.products < found.work:
      return None
    return found

  def _stage_product(self, which: Literal["A", "G"], mats: Matrices, weights: Expr | None) -> list[tuple[Expr, np.ndarray]]:
    """``M^T W M`` for ``M`` one of ``A`` and ``G`` and ``W`` a diagonal (None for the identity), as
    values and the cells of the block storage they add to: each product of two entries of a row
    once, and for the columns ``stage_dense`` takes, the products of their arrays, one per block."""
    s, st = self.s, stages(self.s)
    values = mats.A_values if which == "A" else mats.G_values
    rows, cols, count = (s.A_rows, s.A_cols, s.p) if which == "A" else (s.G_rows, s.G_cols, s.m)
    slot = st.slots[cols]
    found = self.stage_dense(which)
    first, second = pairs_in_rows(rows, count)
    if found is not None:
      apart = ~(found.taken[first] & found.taken[second])
      first, second = first[apart], second[apart]
    out: list[tuple[Expr, np.ndarray]] = []
    if first.size:
      products = gather(values, first) * gather(values, second)
      if weights is not None:
        products = products * gather(weights, rows[first])
      out.append((products, st.cell(slot[first], slot[second])))
    if found is not None:
      out.append((gather(self.stage_arrays[which].squares(values, weights), found.source), found.cell))
    return out

  def _stage_blocks(self, mats: Matrices, xr: Expr, dr: Expr, zr: Expr) -> tuple[Expr, Expr | None, Expr]:
    """The condensed matrix in the slots of ``stages``: the lower triangles of its diagonal blocks
    ``(K, B, B)``, the last ``c`` columns of the blocks below them ``(K - 1, B, c)`` (None with one
    block), and its diagonal. A padding slot is a row of the identity. Every term is a list of
    values with the cell each adds to, and one scatter puts them all in place, so the matrix is
    never formed as a sparse one: what ``stage_dense`` takes of ``A^T A`` and ``G^T W G`` is
    multiplied as dense arrays, one per block, and the rest entry by entry."""
    s, st = self.s, stages(self.s)
    K, B, c = st.K, st.B, st.c
    slot = st.slots
    pads = np.flatnonzero(st.order < 0)
    terms = [(mats.P_values, st.cell(slot[s.P_rows], slot[s.P_cols])), (xr, st.cell(slot, slot))]
    if pads.size:
      terms.append((Expr.const(np.ones(pads.size)), st.cell(pads, pads)))
    if s.p:
      terms += [(values * (1.0 / dr), cell) for values, cell in self._stage_product("A", mats, None)]
    if s.m:
      terms += self._stage_product("G", mats, 1.0 / zr)
    cells = scatter(concat([values for values, _ in terms]), np.concatenate([cell for _, cell in terms]), (st.cells,))
    on_diagonal = (np.arange(K)[:, None] * B * B + np.arange(B)[None, :] * (B + 1)).reshape(-1)
    split = K * B * B
    return cells[:split].reshape((K, B, B)), cells[split:].reshape((K - 1, B, c)) if K > 1 else None, gather(cells, on_diagonal)

  def kkt_matrix(self, mats: Matrices, xr: Expr, dr: Expr, zr: Expr) -> SparseMatrix:
    """The whole KKT matrix the sparse backend factors, its upper triangle:
    ``[[P + diag(x_reg), A^T, G^T], [., -delta_reg I, .], [., ., -diag(z_reg)]]``, without the blocks
    of an empty ``A`` or ``G``."""
    s = self.s
    upper = {
      (0, 0): mats.P.add_diagonal(xr),
      (0, 1): mats.A.T,
      (0, 2): mats.G.T,
      (1, 1): SparseMatrix.diag(-dr * Expr.const(np.ones(s.p))),
      (2, 2): SparseMatrix.diag(-zr),
    }
    kept = [k for k, size in enumerate((s.n, s.p, s.m)) if size or k == 0]
    return SparseMatrix.block([[upper.get((r, c)) for c in kept] for r in kept])

  @staticmethod
  def _pivots_left(pivots: Expr, k_diag: Expr) -> Expr:
    """Whether every pivot of the sparse ``LDL^T`` is larger than one ulp of the diagonal entry of
    the KKT matrix it came from. PIQP's sparse factorization fails only on a pivot that is exactly
    zero. A pivot that cancels need not come out exactly zero, though: on QRECIPE this
    factorization left pivots 12 to 30 orders of magnitude below one ulp of their entries, which
    are rounding residue and not a pivot. They passed the zero test, the steps they gave were of
    length 1e-25 and less, and the solver took 74 iterations where PIQP takes 19.

    A pivot with no digit of its entry left is taken as a failed factorization, with refinement on
    or off, and the retry runs as for a zero pivot, with one difference (``_try``): the floor of
    the regularization, which a singular matrix raises for the rest of the solve, stays where it
    is, since the matrix is not singular and only its factorization lost the pivot. An infinite
    pivot passes, and so does a NaN one, as in PIQP."""
    # As a difference: an infinite pivot's is NaN (its entry is infinite too), and passes like a NaN.
    return where(pivots.abs() - EPS * k_diag.abs() <= 0.0, 1.0, 0.0).sum() < 0.5

  @staticmethod
  def _digits_left(l_diag: Expr, c_diag: Expr) -> Expr:
    """Whether every Cholesky pivot ``L_kk^2`` exceeds ``eps`` times the diagonal entry it came
    from. A pivot below one ulp of that entry has no significant digit left, and its sign is
    rounding noise: on the reference's path through the Maros-Meszaros subset, LAPACK's Cholesky
    fails exactly where such a pivot comes out negative, and our kernel, summing in another order,
    need not. Without refinement such a factor is taken as failed, so refinement turns on
    wherever PIQP's rounding might have turned it on; once refinement is on, which corrects a
    noisy factor, only PIQP's own test counts and retries scale the regularization."""
    return where(l_diag * l_diag > EPS * c_diag, 0.0, 1.0).sum() < 0.5

  def _attempt(self, d: Expr, v: Expr, rho: Expr, delta: Expr, ir: Expr) -> tuple[list[Expr], Expr]:
    """``update_scalings_and_factor``: the regularizations for the current slacks and duals, the
    static one if refinement is on, and the factorization. Returns the parts of the record and
    how it went: ``_FACTORED``, ``_SINGULAR`` (PIQP's own failure) or, from the sparse backend,
    ``_RESIDUE`` (no pivot zero, but one with no digit left: ``_pivots_left``)."""
    s, r = self.s, self.refinement
    fn, _ = self._factorization
    mats, xb = self.matrices(d)
    it = self._unbounds(v)
    lo, up = s.x_l_idx, s.x_u_idx
    z_l_inv = _where_rows(self.has_l, 1.0 / it.z_l)
    z_u_inv = _where_rows(self.has_u, 1.0 / it.z_u)
    x_reg = rho * Expr.const(np.ones(s.n))
    if lo.size:
      x_reg = x_reg + segment_sum(gather(xb, lo) ** 2 / ((1.0 / it.z_bl) * it.s_bl + delta), lo, s.n)
    if up.size:
      x_reg = x_reg + segment_sum(gather(xb, up) ** 2 / ((1.0 / it.z_bu) * it.s_bu + delta), up, s.n)
    w_l_inv = _where_rows(self.has_l, 1.0 / (z_l_inv * it.s_l + delta))
    w_u_inv = _where_rows(self.has_u, 1.0 / (z_u_inv * it.s_u + delta))
    z_reg = 1.0 / (w_l_inv + w_u_inv) if s.m else Expr.const(np.zeros(0))
    max_diag = norm_inf(mats.P_diag + x_reg)
    if s.m:
      max_diag = maximum(max_diag, norm_inf(z_reg))
    reg = where(ir, r.static_eps + r.static_rel * max_diag, 0.0)
    x_reg, delta_reg, z_reg_ir = x_reg + reg, delta + reg, z_reg + reg
    f, ok, digits = fn.symbolic_call((d, x_reg, delta_reg, z_reg_ir))
    if self.backend != "sparse":
      outcome = where(where(ir, ok, digits) > 0.5, _FACTORED, _SINGULAR)
    else:
      outcome = where(digits > 0.5, _FACTORED, where(ok > 0.5, _RESIDUE, _SINGULAR))
    return [stack([delta, delta_reg]), x_reg, z_reg, z_reg_ir, f], outcome

  def _try(self, c: Expr, d: Expr, v: Expr) -> Expr:
    """One pass of the retry loop from the header at the start of ``c``: the next header and record."""
    r = self.refinement
    h = {k: c[i] for i, k in enumerate(HEADER)}
    failed = logical_and(h["tried"] > 0.5, h["ok"] < 0.5)
    retry = logical_and(failed, h["ir"] > 0.5)
    ir = where(failed, 1.0, h["ir"])
    rho = where(retry, h["rho"] * 100.0, h["rho"])
    delta = where(retry, h["delta"] * 100.0, h["delta"])
    # A singular matrix raises the floor of the regularization for the rest of the solve, as in
    # PIQP. Residue does not: over a solve its retries would ratchet the floor to its cap, and a
    # problem that only loses a pivot now and then would end regularized too heavily to converge.
    singular = h["ok"] < 0.5 * _RESIDUE
    reg_limit = where(logical_and(retry, singular), minimum(10.0 * h["reg_limit"], r.reg_limit_cap), h["reg_limit"])
    rec, outcome = self._attempt(d, v, rho, delta, ir > 0.5)
    head = stack([rho, delta, reg_limit, ir, h["retries"] + where(retry, 1.0, 0.0), Expr.const(1.0), outcome, where(retry, 1.0, h["changed"])])
    # One concatenation, so the factor goes from the factorization's result into the carry once.
    return concat([head, *rec])

  @cached_property
  def _retry(self) -> tuple[ConcreteFunction, ConcreteFunction, ConcreteFunction]:
    """PIQP's loop around ``update_scalings_and_factor``: after a failure, turn refinement on; after
    one with refinement on, scale ``rho`` and ``delta`` by 100, up to ``max_factor_retires`` times.
    The first attempt runs before the loop, from the header alone (``first``): most factorizations
    succeed at once, and the loop then takes no step and copies no record."""
    r = self.refinement
    size = len(HEADER) + sum(self.record_sizes)
    c, d, v = Expr.sym("c", (size,)), Expr.sym("D", (sum(self.d_sizes),)), Expr.sym("V", (sum(self.v_sizes),))
    head = Expr.sym("c", (len(HEADER),))
    names = ["c", "D", "V"]
    body = ConcreteFunction.from_exprs(f"{self.name}_factor_try", [c, d, v], [self._try(c, d, v)], names, ["next"])
    first = ConcreteFunction.from_exprs(f"{self.name}_factor_first", [head, d, v], [self._try(head, d, v)], names, ["next"])
    h = {k: c[i] for i, k in enumerate(HEADER)}
    go = logical_and(h["ok"] < 0.5, logical_or(h["ir"] < 0.5, h["retries"] < float(r.max_factor_retires)))
    cond = ConcreteFunction.from_exprs(f"{self.name}_factor_go", [c, d, v], [go], names, ["go"])
    return cond, body, first

  def factor(self, d: Expr, v: Expr, rho: Expr, delta: Expr, reg_limit: Expr, ir: Expr) -> tuple[dict[str, Expr], Expr]:
    """The retry loop from ``rho``, ``delta``, ``reg_limit`` and the refinement flag ``ir`` (0 or 1):
    the header it ends with (``HEADER``) and the last attempt's record."""
    cond, body, first = self._retry
    zero = Expr.const(0.0)
    head = stack([as_expr(rho), as_expr(delta), as_expr(reg_limit), as_expr(ir), zero, zero, zero, zero])
    out, _ = while_loop(cond, body, _call(first, head, d, v), max_iter=self.refinement.max_factor_retires + 1, params=(d, v))
    return {k: out[i] for i, k in enumerate(HEADER)}, out[len(HEADER) :]

  # --- the solve and its refinement -------------------------------------------------------------

  @cached_property
  def _solve(self) -> ConcreteFunction:
    """``K^{-1} r`` for the factor in a record, as ``(record, data, r)``."""
    s = self.s
    rec, d, rhs = Expr.sym("rec", (sum(self.record_sizes),)), Expr.sym("D", (sum(self.d_sizes),)), Expr.sym("r", (self.size,))
    parts = self.record(rec)
    if self.backend != "sparse":
      mats, _ = self.matrices(d)
      n, p = s.n, s.p
      rx, ry, rz = rhs[:n], rhs[n : n + p], rhs[n + p :]
      delta_inv = 1.0 / parts["delta_reg"]
      z_inv = 1.0 / parts["z_reg_ir"]
      x = rx
      if s.m:
        x = x + mats.Gt_times(z_inv * rz)
      if s.p:
        x = x + delta_inv * mats.At_times(ry)
      if self.backend == "dense":
        x = cho_solve(parts["factor"].reshape((n, n)), x)
      else:
        assert self._blocks is not None
        st = stages(s)
        # A padding slot is a row of the identity, coupled with nothing: whatever its right-hand
        # side holds (here the first variable's) reaches no variable.
        slotted = gather(x, np.maximum(st.order, 0))
        x = gather(self._blocks.solve_with(parts["factor"], slotted), st.slots)
      out = [x]
      if s.p:
        out.append(delta_inv * mats.A_times(x) - delta_inv * ry)
      if s.m:
        out.append((mats.G_times(x) - rz) * z_inv)
      sol = concat(out)
    else:
      assert self._ldl is not None
      sol = self._ldl.solve_with(parts["factor"], rhs)
    return ConcreteFunction.from_exprs(f"{self.name}_kkt_solve", [rec, d, rhs], [sol], ["rec", "D", "r"], ["l"])

  def _times(self, rec: Expr, d: Expr, lhs: Expr) -> Expr:
    """PIQP's ``mul_condensed_kkt``: the KKT matrix times ``lhs``, with ``x_reg`` (static
    regularization included), and ``delta`` and ``z_reg`` (not included)."""
    s = self.s
    n, p = s.n, s.p
    parts = self.record(rec)
    mats, _ = self.matrices(d)
    x, y, z = lhs[:n], lhs[n : n + p], lhs[n + p :]
    rx = mats.P_times(x) + parts["x_reg"] * x
    out = []
    if s.p:
      rx = rx + mats.At_times(y)
      out.append(mats.A_times(x) - parts["delta"] * y)
    if s.m:
      rx = rx + mats.Gt_times(z)
      out.append(mats.G_times(x) - parts["z_reg"] * z)
    return concat([rx, *out])

  @cached_property
  def _refinement(self) -> tuple[ConcreteFunction, ConcreteFunction]:
    """PIQP's refinement steps as a loop, each one solve and one product: the condition and the step."""
    r, size = self.refinement, self.size
    rec, d, rhs = Expr.sym("rec", (sum(self.record_sizes),)), Expr.sym("D", (sum(self.d_sizes),)), Expr.sym("rhs", (size,))
    # carry = [stop, error | lhs | residual]
    c, tol = Expr.sym("c", (2 * size + 2,)), Expr.sym("tol", (1,))
    err, lhs, res = c[1], c[2 : 2 + size], c[2 + size :]
    cand = _call(self._solve, rec, d, res) + lhs
    res_c = rhs - self._times(rec, d, cand)
    err_c = norm_inf(res_c)
    finite = isfinite(err_c)
    rate = err / err_c
    slow = rate < r.min_improvement_rate
    accept = logical_and(finite, logical_or(logical_not(slow), rate > 1.0))
    stop = logical_or(logical_or(logical_not(finite), slow), logical_not(err_c > tol[0]))
    nxt = concat([stack([where(stop, 1.0, 0.0), err_c]), where(accept, cand, lhs), res_c])
    names = ["c", "rec", "D", "rhs", "tol"]
    step = ConcreteFunction.from_exprs(f"{self.name}_refine_step", [c, rec, d, rhs, tol], [nxt], names, ["next"])
    step_go = ConcreteFunction.from_exprs(f"{self.name}_refine_go", [c, rec, d, rhs, tol], [c[0] < 0.5], names, ["go"])
    return step_go, step

  @cached_property
  def _refined(self) -> tuple[ConcreteFunction, ConcreteFunction]:
    """The refinement behind one gate, a loop of at most one pass that opens when refinement is on:
    its carry is the solution alone, so a solve without refinement moves nothing else. Inside, the
    first residual and tolerance, then PIQP's refinement steps."""
    r, size = self.refinement, self.size
    step_go, step = self._refinement
    x, rec, d, rhs, on = (
      Expr.sym("x", (size,)),
      Expr.sym("rec", (sum(self.record_sizes),)),
      Expr.sym("D", (sum(self.d_sizes),)),
      Expr.sym("rhs", (size,)),
      Expr.sym("on", (1,)),
    )
    e0 = rhs - self._times(rec, d, x)
    err0, tol = norm_inf(e0), r.eps_abs + r.eps_rel * norm_inf(rhs)
    # A residual that is not finite ends the solve, as PIQP's does.
    stop0 = where(err0 > tol, 0.0, 1.0)
    out, _ = while_loop(step_go, step, concat([stack([stop0, err0]), x, e0]), max_iter=r.max_iter, params=(rec, d, rhs, tol.reshape((1,))))
    names = ["x", "rec", "D", "rhs", "on"]
    gate = ConcreteFunction.from_exprs(f"{self.name}_refine", [x, rec, d, rhs, on], [out[2 : 2 + size]], names, ["x_next"])
    opens = ConcreteFunction.from_exprs(f"{self.name}_refine_on", [x, rec, d, rhs, on], [on[0] > 0.5], names, ["go"])
    return opens, gate

  def solve(self, rec: Expr, d: Expr, rhs: Expr, ir: Expr) -> Expr:
    """``kkt_solver->solve`` for ``rhs`` and, with refinement on (``ir``, a bool), PIQP's refinement:
    while the residual exceeds ``eps_abs + eps_rel * |rhs|``, one more solve of it, kept if it
    improved the residual and stopping once the improvement falls below ``min_improvement_rate``."""
    lhs = _call(self._solve, rec, d, rhs)
    opens, gate = self._refined
    out, _ = while_loop(opens, gate, lhs, max_iter=1, params=(rec, d, rhs, where(ir, 1.0, 0.0).reshape((1,))))
    return out


class KKT:
  """PIQP's ``KKTSystem`` over one scaled problem: the barrier regularizations of the box and
  inequality blocks, the reduced right-hand sides, the solve of

      [[P + diag(x_reg), A^T,     G^T         ],   [dx]   [rx]
       [A,               -delta I, 0          ], * [dy] = [ry]
       [G,               0,       -diag(z_reg)]]   [dz]   [rz]

  and the recovery of the inequality and box duals and slacks, through the shared ``Kernels``."""

  def __init__(self, kernels: Kernels, q: ScaledQP, *, data: Expr | None = None):
    """``data``, when given, is ``kernels.data(q)`` computed elsewhere (a loop's param)."""
    self.kernels, self.s, self.q = kernels, kernels.s, q
    self.mats = kernels.matrices_of(q.values.P, q.values.A, q.values.G)
    self.data = kernels.data(q) if data is None else data

  @property
  def backend(self) -> Backend:
    """Which KKT backend the kernels generate."""
    return self.kernels.backend

  @property
  def A(self) -> SparseMatrix:
    """The equality constraint matrix."""
    return self.mats.A

  @property
  def G(self) -> SparseMatrix:
    """The inequality constraint matrix."""
    return self.mats.G

  def P_times(self, x: Expr) -> Expr:
    """``P x`` from the upper triangle."""
    return self.mats.P_times(x)

  def factor(self, rho: Expr | float, delta: Expr | float, it: Iterate, *, reg_limit: Expr | float = 0.0, ir: Expr | float | None = None) -> Factor:
    """``update_scalings_and_factor`` inside PIQP's retry loop. ``ir`` (1 or 0) says whether
    refinement is already on; by default, whether ``Refinement.always`` is set."""
    k = self.kernels
    if ir is None:
      ir = 1.0 if k.refinement.always else 0.0
    head, rec = k.factor(self.data, k.bounds(it), as_expr(rho), as_expr(delta), as_expr(reg_limit), as_expr(ir))
    return Factor(self, it, head, rec)


@dataclass(frozen=True, eq=False)
class Factor:
  """A factored KKT system at one iterate, ready to solve right-hand sides; ``head`` holds what the
  retry loop ended with (``HEADER``), ``record`` the factorization."""

  kkt: KKT
  it: Iterate
  head: dict[str, Expr]
  record: Expr

  @property
  def ok(self) -> Expr:
    """Whether a factorization succeeded within the retries."""
    return self.head["ok"] > 0.5

  @property
  def rho(self) -> Expr:
    """``rho`` after the retries."""
    return self.head["rho"]

  @property
  def delta(self) -> Expr:
    """``delta`` after the retries."""
    return self.head["delta"]

  @property
  def ir(self) -> Expr:
    """Whether refinement is on (1 or 0)."""
    return self.head["ir"]

  @property
  def changed(self) -> Expr:
    """Whether a retry changed ``rho`` and ``delta``."""
    return self.head["changed"] > 0.5

  def solve(self, rhs: Iterate) -> Iterate:
    """``KKTSystem::solve``: the step for right-hand side ``rhs``."""
    k, s, it = self.kkt, self.kkt.s, self.it
    kern = k.kernels
    xb, d = k.q.x_b, self.delta
    lo, up = s.x_l_idx, s.x_u_idx
    z_reg = kern.record(self.record)["z_reg"]
    z_l_inv = _where_rows(kern.has_l, 1.0 / it.z_l)
    z_u_inv = _where_rows(kern.has_u, 1.0 / it.z_u)
    z_bl_inv, z_bu_inv = 1.0 / it.z_bl, 1.0 / it.z_bu
    w_l_inv = _where_rows(kern.has_l, 1.0 / (z_l_inv * it.s_l + d))
    w_u_inv = _where_rows(kern.has_u, 1.0 / (z_u_inv * it.s_u + d))
    rz_l_bar = _where_rows(kern.has_l, rhs.z_l - z_l_inv * rhs.s_l)
    rz_u_bar = _where_rows(kern.has_u, rhs.z_u - z_u_inv * rhs.s_u)
    rhs_z_bar = (w_u_inv * rz_u_bar - w_l_inv * rz_l_bar) * z_reg
    bl = (rhs.z_bl - z_bl_inv * rhs.s_bl) / (it.s_bl * z_bl_inv + d)
    bu = (rhs.z_bu - z_bu_inv * rhs.s_bu) / (it.s_bu * z_bu_inv + d)
    rhs_x_bar = rhs.x
    if lo.size:
      rhs_x_bar = rhs_x_bar - segment_sum(gather(xb, lo) * bl, lo, s.n)
    if up.size:
      rhs_x_bar = rhs_x_bar + segment_sum(gather(xb, up) * bu, up, s.n)
    sol = kern.solve(self.record, k.data, concat([rhs_x_bar, rhs.y, rhs_z_bar]), self.ir > 0.5)
    lx, ly, lz = sol[: s.n], sol[s.n : s.n + s.p], sol[s.n + s.p :]
    both = kern.has_l & kern.has_u
    r_sum = w_l_inv * w_u_inv * (rz_l_bar + rz_u_bar)
    z_l = where(Expr.const(both, dtype="bool"), -z_reg * (r_sum + w_l_inv * lz), _where_rows(kern.has_l, -lz))
    z_u = where(Expr.const(both, dtype="bool"), -z_reg * (r_sum - w_u_inv * lz), _where_rows(kern.has_u, lz))
    s_l = _where_rows(kern.has_l, z_l_inv * (rhs.s_l - it.s_l * z_l))
    s_u = _where_rows(kern.has_u, z_u_inv * (rhs.s_u - it.s_u * z_u))
    z_bl = (-gather(xb, lo) * gather(lx, lo) - rhs.z_bl + z_bl_inv * rhs.s_bl) / (it.s_bl * z_bl_inv + d)
    z_bu = (gather(xb, up) * gather(lx, up) - rhs.z_bu + z_bu_inv * rhs.s_bu) / (it.s_bu * z_bu_inv + d)
    s_bl = z_bl_inv * (rhs.s_bl - it.s_bl * z_bl)
    s_bu = z_bu_inv * (rhs.s_bu - it.s_bu * z_bu)
    return Iterate(lx, ly, z_l, z_u, z_bl, z_bu, s_l, s_u, s_bl, s_bu)


_SYMBOLIC: weakref.WeakKeyDictionary[QPStructure, SymbolicLDL] = weakref.WeakKeyDictionary()


def kkt_symbolic(s: QPStructure) -> SymbolicLDL:
  """The symbolic factorization of the sparse backend's KKT matrix over ``s``, analysed once per
  structure: the backend's ``SparseLDL`` and the backend choice (``opt.ipm.cost``) share it."""
  if s not in _SYMBOLIC:
    kernels = Kernels(s, "sparse")
    mats, _ = kernels.matrices(Expr.sym("D", (sum(kernels.d_sizes),)))
    matrix = kernels.kkt_matrix(mats, Expr.sym("x_reg", (s.n,)), Expr.sym("delta_reg", ()), Expr.sym("z_reg_ir", (s.m,)))
    rows, cols = matrix.coordinates()
    _SYMBOLIC[s] = analyze(matrix.shape, rows, cols, "auto")
  return _SYMBOLIC[s]


__all__ = ["HEADER", "KKT", "Backend", "Factor", "Iterate", "Kernels", "Matrices", "Refinement", "dense_rows", "kkt_symbolic"]
