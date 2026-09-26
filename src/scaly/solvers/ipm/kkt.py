"""PIQP's KKT system as generated code, behind a dense and a sparse backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ...ir.expr import Expr, concat, gather, isfinite, norm_inf, reduce_min, scatter, segment_sum, where
from ...linalg import SparseLDL, SparseMatrix, cho_solve, cholesky
from .ruiz import ScaledQP
from .structure import QPStructure

Backend = Literal["dense", "sparse"]


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
    return (self.x, self.y, self.z_l, self.z_u, self.z_bl, self.z_bu, self.s_l, self.s_u, self.s_bl, self.s_bu)

  @staticmethod
  def sizes(s: QPStructure) -> tuple[int, ...]:
    nl, nu = s.x_l_idx.size, s.x_u_idx.size
    return (s.n, s.p, s.m, s.m, nl, nu, s.m, s.m, nl, nu)

  def flat(self) -> Expr:
    return concat([f.reshape((f.size,)) for f in self.fields()])

  @staticmethod
  def unflat(s: QPStructure, v: Expr) -> Iterate:
    offsets = np.cumsum([0, *Iterate.sizes(s)])
    return Iterate(*(v[int(a) : int(b)] for a, b in zip(offsets[:-1], offsets[1:], strict=True)))


def _where_rows(mask: np.ndarray, value: Expr) -> Expr:
  """``value`` on the rows of ``mask``, exactly zero elsewhere (an infinity there is discarded)."""
  return where(Expr.const(mask, dtype="bool"), value, 0.0)


class KKT:
  """PIQP's ``KKTSystem`` over one scaled problem: the barrier regularizations of the box and
  inequality blocks, the reduced right-hand sides, the solve of

      [[P + diag(x_reg), A^T,     G^T         ],   [dx]   [rx]
       [A,               -delta I, 0          ], * [dy] = [ry]
       [G,               0,       -diag(z_reg)]]   [dz]   [rz]

  and the recovery of the inequality and box duals and slacks. ``dense`` factors the condensed
  matrix ``P + diag(x_reg) + A^T A / delta + G^T diag(1/z_reg) G`` by Cholesky, as PIQP's dense
  backend does; ``sparse`` factors the whole matrix by ``SparseLDL``, as its sparse backend does."""

  def __init__(self, s: QPStructure, q: ScaledQP, backend: Backend, *, name: str = "ipm"):
    if backend not in ("dense", "sparse"):
      raise ValueError(f"backend must be 'dense' or 'sparse', got {backend!r}")
    self.s, self.q, self.backend, self.name = s, q, backend, name
    has_l, has_u = np.zeros(s.m, dtype=bool), np.zeros(s.m, dtype=bool)
    has_l[s.h_l_idx], has_u[s.h_u_idx] = True, True
    self.has_l, self.has_u = has_l, has_u
    v = q.values
    self.P = SparseMatrix.from_coo(s.P_rows, s.P_cols, v.P, (s.n, s.n))  # upper triangle
    self.A = SparseMatrix.from_coo(s.A_rows, s.A_cols, v.A, (s.p, s.n))
    self.G = SparseMatrix.from_coo(s.G_rows, s.G_cols, v.G, (s.m, s.n))
    diag = np.flatnonzero(s.P_rows == s.P_cols)
    self.P_diag = scatter(gather(v.P, diag), s.P_rows[diag], (s.n,)) if diag.size else Expr.const(np.zeros(s.n))

  # --- the full P and the matrix products the residuals need ------------------------------------

  def P_times(self, x: Expr) -> Expr:
    """``P x`` from the upper triangle."""
    off = np.flatnonzero(self.s.P_rows != self.s.P_cols)
    return self.P @ x + (
      SparseMatrix.from_coo(self.s.P_cols[off], self.s.P_rows[off], gather(self.q.values.P, off), (self.s.n, self.s.n)) @ x if off.size else 0.0
    )

  # --- factor and solve --------------------------------------------------------------------------

  def factor(self, rho: Expr, delta: Expr, it: Iterate) -> Factor:
    """``update_scalings_and_factor`` without iterative refinement: the regularizations for the
    current slacks and duals, and the factorization."""
    s, xb = self.s, self.q.x_b
    lo, up = s.x_l_idx, s.x_u_idx
    z_l_inv = _where_rows(self.has_l, 1.0 / it.z_l)
    z_u_inv = _where_rows(self.has_u, 1.0 / it.z_u)
    z_bl_inv, z_bu_inv = 1.0 / it.z_bl, 1.0 / it.z_bu
    x_reg = rho * Expr.const(np.ones(s.n)) + (segment_sum(gather(xb, lo) ** 2 / (z_bl_inv * it.s_bl + delta), lo, s.n) if lo.size else 0.0)
    x_reg = x_reg + (segment_sum(gather(xb, up) ** 2 / (z_bu_inv * it.s_bu + delta), up, s.n) if up.size else 0.0)
    w_l_inv = _where_rows(self.has_l, 1.0 / (z_l_inv * it.s_l + delta))
    w_u_inv = _where_rows(self.has_u, 1.0 / (z_u_inv * it.s_u + delta))
    z_reg = 1.0 / (w_l_inv + w_u_inv) if s.m else Expr.const(np.zeros(0))
    if self.backend == "dense":
      handle = self._dense_factor(x_reg, delta, z_reg)
    else:
      handle = self._sparse_factor(x_reg, delta, z_reg)
    return Factor(self, it, rho, delta, x_reg, z_reg, z_l_inv, z_u_inv, z_bl_inv, z_bu_inv, w_l_inv, w_u_inv, handle)

  def _dense_factor(self, x_reg: Expr, delta: Expr, z_reg: Expr) -> Expr:
    """The condensed matrix, assembled sparse (the products' patterns are known when the graph is
    built) and densified once, then its Cholesky factor."""
    s = self.s
    upper = self.P
    off = np.flatnonzero(s.P_rows != s.P_cols)
    full = upper + SparseMatrix.from_coo(s.P_cols[off], s.P_rows[off], gather(self.q.values.P, off), (s.n, s.n)) if off.size else upper
    c = full.add_diagonal(x_reg)
    if s.p:
      c = c + (self.A.T @ self.A) * (1.0 / delta)
    if s.m:
      c = c + self.G.T @ self.G.scale_rows(1.0 / z_reg)
    return cholesky(c.to_dense())

  def _sparse_factor(self, x_reg: Expr, delta: Expr, z_reg: Expr) -> SparseLDL:
    """The upper triangle of the whole KKT matrix, factored by ``SparseLDL``; the blocks of an
    absent constraint kind are left out."""
    s = self.s
    upper = {
      (0, 0): self.P.add_diagonal(x_reg),
      (0, 1): self.A.T,
      (0, 2): self.G.T,
      (1, 1): SparseMatrix.diag(-delta * Expr.const(np.ones(s.p))),
      (2, 2): SparseMatrix.diag(-z_reg),
    }
    kept = [k for k, size in enumerate((s.n, s.p, s.m)) if size or k == 0]
    blocks = [[upper.get((r, c)) for c in kept] for r in kept]
    return SparseLDL(SparseMatrix.block(blocks), name=f"{self.name}_kkt")


@dataclass(frozen=True, eq=False)
class Factor:
  """A factored KKT system at one iterate, ready to solve right-hand sides."""

  kkt: KKT
  it: Iterate
  rho: Expr
  delta: Expr
  x_reg: Expr
  z_reg: Expr
  z_l_inv: Expr
  z_u_inv: Expr
  z_bl_inv: Expr
  z_bu_inv: Expr
  w_l_inv: Expr
  w_u_inv: Expr
  handle: object

  @property
  def ok(self) -> Expr:
    """Whether the factorization succeeded: every pivot finite (Cholesky of a matrix that is not
    positive definite gives NaN; LDL^T gives inf or NaN on a zero pivot)."""
    k = self.kkt
    if k.backend == "dense":
      handle = self.handle
      assert isinstance(handle, Expr)
      return isfinite(norm_inf(gather(handle.reshape((handle.size,)), np.arange(k.s.n) * (k.s.n + 1))))
    handle = self.handle
    assert isinstance(handle, SparseLDL)
    return isfinite(reduce_min(handle.d.abs())) if k.s.n + k.s.p + k.s.m else Expr.const(True, dtype="bool")

  def _kkt_solve(self, rx: Expr, ry: Expr, rz: Expr) -> tuple[Expr, Expr, Expr]:
    k, s = self.kkt, self.kkt.s
    if k.backend == "dense":
      handle = self.handle
      assert isinstance(handle, Expr)
      r = rx
      if s.p:
        r = r + (k.A.T @ ry) / self.delta
      if s.m:
        r = r + k.G.T @ (rz / self.z_reg)
      x = cho_solve(handle, r)
      y = (k.A @ x - ry) / self.delta if s.p else Expr.const(np.zeros(0))
      z = (k.G @ x - rz) / self.z_reg if s.m else Expr.const(np.zeros(0))
      return x, y, z
    handle = self.handle
    assert isinstance(handle, SparseLDL)
    sol = handle.solve(concat([rx, ry, rz]))
    return sol[: s.n], sol[s.n : s.n + s.p], sol[s.n + s.p :]

  def solve(self, rhs: Iterate) -> Iterate:
    """``KKTSystem::solve`` without refinement: the step for right-hand side ``rhs``."""
    k, s, it = self.kkt, self.kkt.s, self.it
    xb, d = k.q.x_b, self.delta
    lo, up = s.x_l_idx, s.x_u_idx
    rz_l_bar = _where_rows(k.has_l, rhs.z_l - self.z_l_inv * rhs.s_l)
    rz_u_bar = _where_rows(k.has_u, rhs.z_u - self.z_u_inv * rhs.s_u)
    rhs_z_bar = (self.w_u_inv * rz_u_bar - self.w_l_inv * rz_l_bar) * self.z_reg
    bl = (rhs.z_bl - self.z_bl_inv * rhs.s_bl) / (it.s_bl * self.z_bl_inv + d)
    bu = (rhs.z_bu - self.z_bu_inv * rhs.s_bu) / (it.s_bu * self.z_bu_inv + d)
    rhs_x_bar = rhs.x
    if lo.size:
      rhs_x_bar = rhs_x_bar - segment_sum(gather(xb, lo) * bl, lo, s.n)
    if up.size:
      rhs_x_bar = rhs_x_bar + segment_sum(gather(xb, up) * bu, up, s.n)
    lx, ly, lz = self._kkt_solve(rhs_x_bar, rhs.y, rhs_z_bar)
    both = k.has_l & k.has_u
    r_sum = self.w_l_inv * self.w_u_inv * (rz_l_bar + rz_u_bar)
    z_l = where(Expr.const(both, dtype="bool"), -self.z_reg * (r_sum + self.w_l_inv * lz), _where_rows(k.has_l, -lz))
    z_u = where(Expr.const(both, dtype="bool"), -self.z_reg * (r_sum - self.w_u_inv * lz), _where_rows(k.has_u, lz))
    s_l = _where_rows(k.has_l, self.z_l_inv * (rhs.s_l - it.s_l * z_l))
    s_u = _where_rows(k.has_u, self.z_u_inv * (rhs.s_u - it.s_u * z_u))
    z_bl = (-gather(xb, lo) * gather(lx, lo) - rhs.z_bl + self.z_bl_inv * rhs.s_bl) / (it.s_bl * self.z_bl_inv + d)
    z_bu = (gather(xb, up) * gather(lx, up) - rhs.z_bu + self.z_bu_inv * rhs.s_bu) / (it.s_bu * self.z_bu_inv + d)
    s_bl = self.z_bl_inv * (rhs.s_bl - it.s_bl * z_bl)
    s_bu = self.z_bu_inv * (rhs.s_bu - it.s_bu * z_bu)
    return Iterate(lx, ly, z_l, z_u, z_bl, z_bu, s_l, s_u, s_bl, s_bu)


__all__ = ["KKT", "Backend", "Factor", "Iterate"]
