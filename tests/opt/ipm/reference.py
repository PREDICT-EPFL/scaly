"""A NumPy reference of PIQP 0.6.2, the oracle for the Scaly interior-point solver.

A line-by-line port of ``piqp::SolverBase`` (``solver.hpp``), the Ruiz equilibration
(``sparse/preconditioner.hpp``), the problem preprocessing (``sparse/data.hpp``) and the KKT
system with iterative refinement (``kkt_system.hpp``). Every quantity is computed as PIQP computes
it, in the same order. The one deliberate difference is the linear solve: PIQP factors the
regularized KKT matrix with an AMD-ordered LDL^T without pivoting (sparse backend) or a condensed
Cholesky (dense backend); this solves the same matrix with SuperLU, so results agree to rounding
and decision traces agree wherever the iteration is not sensitive to rounding.

``solve(qp)`` returns the unscaled solution, PIQP's status code and one trace row per iteration in
the layout of PIQP's verbose table (plus sigma): the row printed at the top of iteration ``k``
holds the state after ``k`` iterations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

from tests.opt.ipm.problems import QP

PIQP_INF = 1e30
EPS = np.finfo(np.float64).eps

SOLVED, MAX_ITER_REACHED, PRIMAL_INFEASIBLE, DUAL_INFEASIBLE, NUMERICS, UNSOLVED = 1, -1, -2, -3, -8, -9
TRACE_COLUMNS = ("iter", "primal_obj", "dual_obj", "duality_gap", "primal_res", "dual_res", "rho", "delta", "mu", "primal_step", "dual_step", "sigma")


@dataclass
class Settings:
  """PIQP 0.6.2's defaults (``settings.hpp``)."""

  rho_init: float = 1e-6
  delta_init: float = 1e-4
  eps_abs: float = 1e-8
  eps_rel: float = 1e-9
  check_duality_gap: bool = True
  eps_duality_gap_abs: float = 1e-8
  eps_duality_gap_rel: float = 1e-9
  infeasibility_threshold: float = 0.9
  reg_lower_limit: float = 1e-10
  reg_finetune_lower_limit: float = 1e-13
  reg_finetune_primal_update_threshold: int = 7
  reg_finetune_dual_update_threshold: int = 7
  max_iter: int = 250
  max_factor_retires: int = 10
  preconditioner_scale_cost: bool = False
  preconditioner_iter: int = 10
  tau: float = 0.99
  iterative_refinement_always_enabled: bool = False
  iterative_refinement_eps_abs: float = 1e-12
  iterative_refinement_eps_rel: float = 1e-12
  iterative_refinement_max_iter: int = 10
  iterative_refinement_min_improvement_rate: float = 5.0
  iterative_refinement_static_regularization_eps: float = 1e-8
  iterative_refinement_static_regularization_rel: float = EPS * EPS


# --- data (sparse/data.hpp) -------------------------------------------------------------------


@dataclass
class Data:
  n: int
  p: int
  m: int
  P: sparse.csc_array  # full symmetric, rebuilt from the upper triangle PIQP stores
  A: sparse.csr_array
  G: sparse.csr_array
  c: np.ndarray
  b: np.ndarray
  h_l: np.ndarray  # length m; -PIQP_INF where absent
  h_u: np.ndarray
  x_l: np.ndarray  # finite lower bounds packed in the head (n_x_l of them), as PIQP stores them
  x_u: np.ndarray
  h_l_idx: np.ndarray
  h_u_idx: np.ndarray
  x_l_idx: np.ndarray
  x_u_idx: np.ndarray
  x_b_scaling: np.ndarray

  @property
  def n_h_l(self) -> int:
    return self.h_l_idx.size

  @property
  def n_h_u(self) -> int:
    return self.h_u_idx.size

  @property
  def n_x_l(self) -> int:
    return self.x_l_idx.size

  @property
  def n_x_u(self) -> int:
    return self.x_u_idx.size

  @property
  def P_diag(self) -> np.ndarray:
    return self.P.diagonal()


def setup_data(qp: QP) -> Data:
  n, p, m = qp.n, qp.A.shape[0], qp.G.shape[0]
  upper = sparse.triu(qp.P, format="csc")
  P = sparse.csc_array(upper + sparse.triu(upper, 1).T)
  G = sparse.csr_array(qp.G, shape=(m, n), copy=True)
  h_l = np.where(qp.h_l > -PIQP_INF, qp.h_l, -PIQP_INF).astype(np.float64)
  h_u = np.where(qp.h_u < PIQP_INF, qp.h_u, PIQP_INF).astype(np.float64)
  # disable_inf_constraints: a row infinite on both sides keeps its place with G's row zeroed
  # and the bounds [-1, 1], so it still counts in mu.
  both = (h_l <= -PIQP_INF) & (h_u >= PIQP_INF)
  if np.any(both):
    for i in np.flatnonzero(both):
      G.data[G.indptr[i] : G.indptr[i + 1]] = 0.0
    h_l[both], h_u[both] = -1.0, 1.0
  x_l_idx = np.flatnonzero(qp.x_l > -PIQP_INF)
  x_u_idx = np.flatnonzero(qp.x_u < PIQP_INF)
  x_l, x_u = np.zeros(n), np.zeros(n)
  x_l[: x_l_idx.size] = qp.x_l[x_l_idx]
  x_u[: x_u_idx.size] = qp.x_u[x_u_idx]
  return Data(
    n=n,
    p=p,
    m=m,
    P=P,
    A=sparse.csr_array(qp.A, shape=(p, n), copy=True),
    G=G,
    c=np.array(qp.c, dtype=np.float64),
    b=np.array(qp.b, dtype=np.float64),
    h_l=h_l,
    h_u=h_u,
    x_l=x_l,
    x_u=x_u,
    h_l_idx=np.flatnonzero(h_l > -PIQP_INF),
    h_u_idx=np.flatnonzero(h_u < PIQP_INF),
    x_l_idx=x_l_idx,
    x_u_idx=x_u_idx,
    x_b_scaling=np.ones(n),
  )


# --- Ruiz equilibration (sparse/preconditioner.hpp) -------------------------------------------


def _limit_scaling(d: np.ndarray) -> np.ndarray:
  return np.where(d < 1e-4, 1.0, np.minimum(d, 1e4))


def _row_max(a: sparse.sparray, rows: int) -> np.ndarray:
  out = np.zeros(rows)
  coo = sparse.coo_array(a)
  np.maximum.at(out, coo.row, np.abs(coo.data))
  return out


def _col_max(a: sparse.sparray, cols: int) -> np.ndarray:
  out = np.zeros(cols)
  coo = sparse.coo_array(a)
  np.maximum.at(out, coo.col, np.abs(coo.data))
  return out


@dataclass
class Ruiz:
  n: int
  p: int
  m: int
  c: float = 1.0
  delta: np.ndarray = field(default_factory=lambda: np.ones(0))
  delta_b: np.ndarray = field(default_factory=lambda: np.ones(0))

  def scale_data(self, data: Data, scale_cost: bool, max_iter: int, epsilon: float = 1e-3) -> None:
    n, p, m = self.n, self.p, self.m
    self.c = 1.0
    self.delta, self.delta_b = np.ones(n + p + m), np.ones(n)
    d_iter, d_iter_b = np.zeros(n + p + m), np.zeros(n)
    for _ in range(max_iter):
      if max(np.abs(1 - d_iter).max(initial=0.0), np.abs(1 - d_iter_b).max(initial=0.0)) <= epsilon:
        break
      d_iter = np.zeros(n + p + m)
      d_iter[:n] = np.maximum(_col_max(data.P, n), data.x_b_scaling)
      d_iter[:n] = np.maximum(d_iter[:n], _col_max(data.A, n))
      d_iter[n : n + p] = _row_max(data.A, p)
      d_iter[:n] = np.maximum(d_iter[:n], _col_max(data.G, n))
      d_iter[n + p :] = _row_max(data.G, m)
      d_iter_b = data.x_b_scaling.copy()
      d_iter = 1.0 / np.sqrt(_limit_scaling(d_iter))
      d_iter_b = 1.0 / np.sqrt(_limit_scaling(d_iter_b))
      dx = sparse.diags_array(d_iter[:n])
      data.P = sparse.csc_array(dx @ data.P @ dx)
      data.c = data.c * d_iter[:n]
      data.A = sparse.csr_array(sparse.diags_array(d_iter[n : n + p]) @ data.A @ dx)
      data.G = sparse.csr_array(sparse.diags_array(d_iter[n + p :]) @ data.G @ dx)
      data.x_b_scaling = data.x_b_scaling * (d_iter_b * d_iter[:n])
      self.delta = self.delta * d_iter
      self.delta_b = self.delta_b * d_iter_b
      if scale_cost:
        cost_col_max = _col_max(data.P, n)
        # PIQP computes these maxima in the memory of the box-scaling iterate, which the loop's
        # stopping test reads next: with cost scaling on, that test sees them instead.
        d_iter_b = cost_col_max
        gamma = float(_limit_scaling(np.array([cost_col_max.sum() / n]))[0])
        gamma = float(_limit_scaling(np.array([max(gamma, np.abs(data.c).max(initial=0.0))]))[0])
        gamma = 1.0 / gamma
        data.P = data.P * gamma
        data.c = data.c * gamma
        self.c *= gamma
    data.b = data.b * self.delta[n : n + p]
    data.h_l = data.h_l * self.delta[n + p :]
    data.h_u = data.h_u * self.delta[n + p :]
    data.x_l[: data.n_x_l] *= self.delta_b[data.x_l_idx]
    data.x_u[: data.n_x_u] *= self.delta_b[data.x_u_idx]

  @property
  def c_inv(self) -> float:
    return 1.0 / self.c

  def unscale_cost(self, v):
    return self.c_inv * v

  def unscale_primal(self, x):
    return x * self.delta[: self.n]

  def unscale_dual_eq(self, y):
    return y * self.c_inv * self.delta[self.n : self.n + self.p]

  def unscale_dual_ineq(self, z):
    return z * self.c_inv * self.delta[self.n + self.p :]

  def unscale_dual_b(self, z, idx):
    return z * self.c_inv * self.delta_b[idx]

  def unscale_slack_ineq(self, s):
    return s / self.delta[self.n + self.p :]

  def unscale_slack_b(self, s, idx):
    return s / self.delta_b[idx]

  def unscale_primal_res_eq(self, r):
    return r / self.delta[self.n : self.n + self.p]

  def unscale_primal_res_ineq(self, r):
    return r / self.delta[self.n + self.p :]

  def unscale_primal_res_b(self, r, idx):
    return r / self.delta_b[idx]

  def unscale_dual_res(self, r):
    return r * self.c_inv / self.delta[: self.n]


# --- variables --------------------------------------------------------------------------------


@dataclass
class Variables:
  x: np.ndarray
  y: np.ndarray
  z_l: np.ndarray
  z_u: np.ndarray
  z_bl: np.ndarray
  z_bu: np.ndarray
  s_l: np.ndarray
  s_u: np.ndarray
  s_bl: np.ndarray
  s_bu: np.ndarray

  @staticmethod
  def zeros(n: int, p: int, m: int) -> Variables:
    return Variables(*(np.zeros(k) for k in (n, p, m, m, n, n, m, m, n, n)))

  def copy(self) -> Variables:
    return Variables(*(v.copy() for v in (self.x, self.y, self.z_l, self.z_u, self.z_bl, self.z_bu, self.s_l, self.s_u, self.s_bl, self.s_bu)))


def _inf_norm(*vs: np.ndarray) -> float:
  return max((float(np.abs(v).max()) for v in vs if v.size), default=0.0)


def _signed_max(start: float, *vs: np.ndarray) -> float:
  """``max(start, v_i)`` without absolute values. PIQP folds the box-bound terms of its primal
  residuals and of the primal proximal infeasibility in this way (``(std::max)(inf, value)``), so a
  negative box residual never counts; the equality and inequality parts use the infinity norm."""
  return max([start, *(float(v.max()) for v in vs if v.size)])


# --- the KKT system (kkt_system.hpp) ----------------------------------------------------------


class KKTSystem:
  def __init__(self, data: Data):
    self.P_diag = data.P_diag.copy()
    self.P_off = sparse.csc_array(data.P - sparse.diags_array(self.P_diag))
    self.lu = None

  def update_scalings_and_factor(self, data: Data, settings: Settings, iterative_refinement: bool, rho: float, delta: float, v: Variables) -> bool:
    with np.errstate(divide="ignore"):
      self.rho, self.delta = rho, delta
      self.s_l, self.s_u = v.s_l.copy(), v.s_u.copy()
      self.s_bl, self.s_bu = v.s_bl[: data.n_x_l].copy(), v.s_bu[: data.n_x_u].copy()
      self.z_l_inv, self.z_u_inv = 1.0 / v.z_l, 1.0 / v.z_u
      self.z_bl_inv, self.z_bu_inv = 1.0 / v.z_bl[: data.n_x_l], 1.0 / v.z_bu[: data.n_x_u]
    xbs = data.x_b_scaling
    x_reg = np.full(data.n, rho)
    lo, up = data.x_l_idx, data.x_u_idx
    np.add.at(x_reg, lo, xbs[lo] * xbs[lo] / (self.z_bl_inv * self.s_bl + delta))
    np.add.at(x_reg, up, xbs[up] * xbs[up] / (self.z_bu_inv * self.s_bu + delta))
    z_reg = np.zeros(data.m)
    hl, hu = data.h_l_idx, data.h_u_idx
    z_reg[hl] += 1.0 / (self.z_l_inv[hl] * self.s_l[hl] + delta)
    z_reg[hu] += 1.0 / (self.z_u_inv[hu] * self.s_u[hu] + delta)
    with np.errstate(divide="ignore"):
      z_reg = 1.0 / z_reg
    z_reg_ir = z_reg.copy()
    delta_reg = delta
    if iterative_refinement:
      max_diag = max(_inf_norm(self.P_diag + x_reg), _inf_norm(z_reg_ir))
      reg = settings.iterative_refinement_static_regularization_eps + settings.iterative_refinement_static_regularization_rel * max_diag
      delta_reg += reg
      x_reg = x_reg + reg
      z_reg_ir = z_reg_ir + reg
    self.x_reg, self.z_reg = x_reg, z_reg
    self.use_iterative_refinement = iterative_refinement
    kkt = sparse.bmat(
      [
        [self.P_off + sparse.diags_array(self.P_diag + x_reg), data.A.T, data.G.T],
        [data.A, sparse.diags_array(np.full(data.p, -delta_reg)) if data.p else None, None],
        [data.G, None, sparse.diags_array(-z_reg_ir) if data.m else None],
      ],
      format="csc",
    )
    if not np.all(np.isfinite(kkt.data)):
      return False
    try:
      self.lu = splinalg.splu(kkt)
    except RuntimeError:  # exactly singular: PIQP's LDL^T fails on a zero pivot
      return False
    return True

  def _kkt_solve(self, data: Data, rx, ry, rz):
    assert self.lu is not None, "solve before a successful factorization"
    sol = self.lu.solve(np.concatenate([rx, ry, rz]))
    return sol[: data.n], sol[data.n : data.n + data.p], sol[data.n + data.p :]

  def _mul_condensed(self, data: Data, lx, ly, lz):
    rx = self.P_off @ lx + (self.P_diag + self.x_reg) * lx + data.A.T @ ly + data.G.T @ lz
    ry = data.A @ lx - self.delta * ly
    rz = data.G @ lx - self.z_reg * lz
    return rx, ry, rz

  def solve(self, data: Data, settings: Settings, rhs: Variables, lhs: Variables) -> bool:
    hl, hu, lo, up, xbs, d = data.h_l_idx, data.h_u_idx, data.x_l_idx, data.x_u_idx, data.x_b_scaling, self.delta
    rhs_z_bar = np.zeros(data.m)
    rhs_z_bar[hl] -= 1.0 / (self.z_l_inv[hl] * self.s_l[hl] + d) * (rhs.z_l[hl] - self.z_l_inv[hl] * rhs.s_l[hl])
    rhs_z_bar[hu] += 1.0 / (self.z_u_inv[hu] * self.s_u[hu] + d) * (rhs.z_u[hu] - self.z_u_inv[hu] * rhs.s_u[hu])
    rhs_z_bar *= self.z_reg
    rhs_x_bar = rhs.x.copy()
    nl, nu = data.n_x_l, data.n_x_u
    np.add.at(rhs_x_bar, lo, -xbs[lo] * (rhs.z_bl[:nl] - self.z_bl_inv * rhs.s_bl[:nl]) / (self.s_bl * self.z_bl_inv + d))
    np.add.at(rhs_x_bar, up, xbs[up] * (rhs.z_bu[:nu] - self.z_bu_inv * rhs.s_bu[:nu]) / (self.s_bu * self.z_bu_inv + d))
    lx, ly, lz = self._kkt_solve(data, rhs_x_bar, rhs.y, rhs_z_bar)
    ok = True
    if self.use_iterative_refinement:
      rhs_norm = _inf_norm(rhs_x_bar, rhs.y, rhs_z_bar)

      def error(x, y, z):
        kx, ky, kz = self._mul_condensed(data, x, y, z)
        e = (rhs_x_bar - kx, rhs.y - ky, rhs_z_bar - kz)
        return e, _inf_norm(*e)

      err, refine_error = error(lx, ly, lz)
      if not np.isfinite(refine_error):
        return False
      for _ in range(settings.iterative_refinement_max_iter):
        if refine_error <= settings.iterative_refinement_eps_abs + settings.iterative_refinement_eps_rel * rhs_norm:
          break
        prev = refine_error
        dx, dy, dz = self._kkt_solve(data, *err)
        cx, cy, cz = dx + lx, dy + ly, dz + lz
        err, refine_error = error(cx, cy, cz)
        if not np.isfinite(refine_error):
          return False
        with np.errstate(divide="ignore", invalid="ignore"):  # IEEE, as in C++: an exact solve improves infinitely
          rate = float(np.float64(prev) / np.float64(refine_error))
        if rate < settings.iterative_refinement_min_improvement_rate:
          if rate > 1.0:
            lx, ly, lz = cx, cy, cz
          break
        lx, ly, lz = cx, cy, cz
    else:
      ok = bool(np.all(np.isfinite(lx)) and np.all(np.isfinite(ly)) and np.all(np.isfinite(lz)))
    lhs.x, lhs.y = lx, ly
    # dual recovery, row by row as PIQP distinguishes rows with a lower bound, an upper bound or both
    has_l = np.zeros(data.m, dtype=bool)
    has_u = np.zeros(data.m, dtype=bool)
    has_l[hl], has_u[hu] = True, True
    both, only_l, only_u = has_l & has_u, has_l & ~has_u, has_u & ~has_l
    with np.errstate(divide="ignore", invalid="ignore"):
      rz_l_bar = rhs.z_l - self.z_l_inv * rhs.s_l
      w_l_inv = 1.0 / (self.z_l_inv * self.s_l + d)
      rz_u_bar = rhs.z_u - self.z_u_inv * rhs.s_u
      w_u_inv = 1.0 / (self.z_u_inv * self.s_u + d)
      r_sum = w_l_inv * w_u_inv * (rz_l_bar + rz_u_bar)
      z_l = np.zeros(data.m)
      z_u = np.zeros(data.m)
      z_l[both] = (-self.z_reg * (r_sum + w_l_inv * lz))[both]
      z_u[both] = (-self.z_reg * (r_sum - w_u_inv * lz))[both]
      z_l[only_l] = -lz[only_l]
      z_u[only_u] = lz[only_u]
      s_l = np.zeros(data.m)
      s_u = np.zeros(data.m)
      s_l[has_l] = (self.z_l_inv * (rhs.s_l - self.s_l * z_l))[has_l]
      s_u[has_u] = (self.z_u_inv * (rhs.s_u - self.s_u * z_u))[has_u]
    lhs.z_l, lhs.z_u, lhs.s_l, lhs.s_u = z_l, z_u, s_l, s_u
    z_bl = np.zeros(data.n)
    z_bu = np.zeros(data.n)
    z_bl[:nl] = (-xbs[lo] * lx[lo] - rhs.z_bl[:nl] + self.z_bl_inv * rhs.s_bl[:nl]) / (self.s_bl * self.z_bl_inv + d)
    z_bu[:nu] = (xbs[up] * lx[up] - rhs.z_bu[:nu] + self.z_bu_inv * rhs.s_bu[:nu]) / (self.s_bu * self.z_bu_inv + d)
    s_bl = np.zeros(data.n)
    s_bu = np.zeros(data.n)
    s_bl[:nl] = self.z_bl_inv * (rhs.s_bl[:nl] - self.s_bl * z_bl[:nl])
    s_bu[:nu] = self.z_bu_inv * (rhs.s_bu[:nu] - self.s_bu * z_bu[:nu])
    lhs.z_bl, lhs.z_bu, lhs.s_bl, lhs.s_bu = z_bl, z_bu, s_bl, s_bu
    return ok


# --- the solver (solver.hpp) ------------------------------------------------------------------


@dataclass
class Info:
  status: int = UNSOLVED
  iter: int = 0
  rho: float = 0.0
  delta: float = 0.0
  mu: float = 0.0
  sigma: float = 0.0
  primal_step: float = 0.0
  dual_step: float = 0.0
  primal_res: float = 0.0
  primal_res_rel: float = 0.0
  dual_res: float = 0.0
  dual_res_rel: float = 0.0
  primal_res_reg: float = 0.0
  primal_res_reg_rel: float = 0.0
  dual_res_reg: float = 0.0
  dual_res_reg_rel: float = 0.0
  primal_prox_inf: float = 0.0
  dual_prox_inf: float = 0.0
  prev_primal_res: float = 0.0
  prev_dual_res: float = 0.0
  primal_obj: float = 0.0
  dual_obj: float = 0.0
  duality_gap: float = 0.0
  duality_gap_rel: float = 0.0
  factor_retires: int = 0
  reg_limit: float = 0.0
  no_primal_update: int = 0
  no_dual_update: int = 0


@dataclass
class Result:
  status: int
  x: np.ndarray
  y: np.ndarray
  z_l: np.ndarray
  z_u: np.ndarray
  z_bl: np.ndarray
  z_bu: np.ndarray
  s_l: np.ndarray
  s_u: np.ndarray
  s_bl: np.ndarray
  s_bu: np.ndarray
  info: Info
  trace: np.ndarray  # one row per printed iteration, columns TRACE_COLUMNS


class Solver:
  def __init__(self, qp: QP, settings: Settings | None = None):
    self.settings = settings or Settings()
    self.data = setup_data(qp)
    d = self.data
    self.pre = Ruiz(d.n, d.p, d.m)
    self.pre.scale_data(d, self.settings.preconditioner_scale_cost, self.settings.preconditioner_iter)
    self.kkt = KKTSystem(d)

  # residuals ---------------------------------------------------------------------------------

  def _calculate_mu(self) -> float:
    d, r = self.data, self.r
    total = r.s_l @ r.z_l + r.s_u @ r.z_u + r.s_bl[: d.n_x_l] @ r.z_bl[: d.n_x_l] + r.s_bu[: d.n_x_u] @ r.z_bu[: d.n_x_u]
    return float(total) / float(d.n_h_l + d.n_h_u + d.n_x_l + d.n_x_u)

  def _calculate_step(self, step: Variables) -> tuple[float, float]:
    d, r = self.data, self.r
    alpha_s, alpha_z = 1.0, 1.0
    for s, ds in ((r.s_l, step.s_l), (r.s_u, step.s_u), (r.s_bl[: d.n_x_l], step.s_bl[: d.n_x_l]), (r.s_bu[: d.n_x_u], step.s_bu[: d.n_x_u])):
      neg = ds < 0
      if np.any(neg):
        alpha_s = min(alpha_s, float(np.min(-s[neg] / ds[neg])))
    for z, dz in ((r.z_l, step.z_l), (r.z_u, step.z_u), (r.z_bl[: d.n_x_l], step.z_bl[: d.n_x_l]), (r.z_bu[: d.n_x_u], step.z_bu[: d.n_x_u])):
      neg = dz < 0
      if np.any(neg):
        alpha_z = min(alpha_z, float(np.min(-z[neg] / dz[neg])))
    return alpha_s, alpha_z

  def _update_residuals_nr(self) -> None:
    d, r, pre, info, nr = self.data, self.r, self.pre, self.info, self.res_nr
    lo, up = d.x_l_idx, d.x_u_idx
    nr.y = -(d.A @ r.x)
    work_x = d.A.T @ r.y
    nr.z_l = d.G @ r.x
    work_x = work_x + d.G.T @ (r.z_u - r.z_l)
    nr.z_u = -nr.z_l
    nr.x = -(d.P @ r.x)
    dual_rel_norm = _inf_norm(pre.unscale_dual_res(nr.x))
    tmp = -(r.x @ nr.x)
    info.primal_obj = 0.5 * tmp
    info.dual_obj = -0.5 * tmp
    gap_rel_norm = pre.unscale_cost(abs(tmp))
    for value, sign in (
      (d.c @ r.x, "p"),
      (d.b @ r.y, "d"),
      (-(d.h_l[d.h_l_idx] @ r.z_l[d.h_l_idx]), "d"),
      (d.h_u[d.h_u_idx] @ r.z_u[d.h_u_idx], "d"),
    ):
      if sign == "p":
        info.primal_obj += value
      else:
        info.dual_obj -= value
      gap_rel_norm = max(gap_rel_norm, pre.unscale_cost(abs(value)))
    for value in (-(d.x_l[: d.n_x_l] @ r.z_bl[: d.n_x_l]), d.x_u[: d.n_x_u] @ r.z_bu[: d.n_x_u]):
      info.dual_obj -= value
      gap_rel_norm = max(gap_rel_norm, pre.unscale_cost(abs(value)))
    info.duality_gap = abs(info.primal_obj - info.dual_obj)
    info.primal_obj = pre.unscale_cost(info.primal_obj)
    info.dual_obj = pre.unscale_cost(info.dual_obj)
    info.duality_gap = pre.unscale_cost(info.duality_gap)
    info.duality_gap_rel = info.duality_gap / max(1.0, gap_rel_norm)

    nr.x = nr.x - d.c
    dual_rel_norm = max(dual_rel_norm, _inf_norm(pre.unscale_dual_res(d.c)))
    np.add.at(work_x, lo, -d.x_b_scaling[lo] * r.z_bl[: d.n_x_l])
    np.add.at(work_x, up, d.x_b_scaling[up] * r.z_bu[: d.n_x_u])
    dual_rel_norm = max(dual_rel_norm, _inf_norm(pre.unscale_dual_res(work_x)))
    nr.x = nr.x - work_x

    primal_rel_norm = _inf_norm(pre.unscale_primal_res_eq(nr.y))
    nr.y = nr.y + d.b
    primal_rel_norm = max(primal_rel_norm, _inf_norm(pre.unscale_primal_res_eq(d.b)))
    hl, hu = d.h_l_idx, d.h_u_idx
    z_l = np.zeros(d.m)
    z_u = np.zeros(d.m)
    if hl.size:
      primal_rel_norm = max(primal_rel_norm, float(np.max(pre.unscale_primal_res_ineq(nr.z_l)[hl])))
      z_l[hl] = nr.z_l[hl] - d.h_l[hl] - r.s_l[hl]
      primal_rel_norm = max(
        primal_rel_norm, float(np.max(pre.unscale_primal_res_ineq(d.h_l)[hl])), float(np.max(pre.unscale_primal_res_ineq(r.s_l)[hl]))
      )
    if hu.size:
      primal_rel_norm = max(primal_rel_norm, float(np.max(pre.unscale_primal_res_ineq(nr.z_u)[hu])))
      z_u[hu] = nr.z_u[hu] + d.h_u[hu] - r.s_u[hu]
      primal_rel_norm = max(
        primal_rel_norm, float(np.max(pre.unscale_primal_res_ineq(d.h_u)[hu])), float(np.max(pre.unscale_primal_res_ineq(r.s_u)[hu]))
      )
    nr.z_l, nr.z_u = z_l, z_u
    z_bl = np.zeros(d.n)
    z_bu = np.zeros(d.n)
    nl, nu = d.n_x_l, d.n_x_u
    if nl:
      z_bl[:nl] = d.x_b_scaling[lo] * r.x[lo]
      for vec in (z_bl[:nl], d.x_l[:nl], r.s_bl[:nl]):
        primal_rel_norm = max(primal_rel_norm, float(np.max(pre.unscale_primal_res_b(vec, lo))))
      z_bl[:nl] += -d.x_l[:nl] - r.s_bl[:nl]
    if nu:
      z_bu[:nu] = -d.x_b_scaling[up] * r.x[up]
      for vec in (z_bu[:nu], d.x_u[:nu], r.s_bu[:nu]):
        primal_rel_norm = max(primal_rel_norm, float(np.max(pre.unscale_primal_res_b(vec, up))))
      z_bu[:nu] += d.x_u[:nu] - r.s_bu[:nu]
    nr.z_bl, nr.z_bu = z_bl, z_bu

    info.prev_primal_res = info.primal_res
    info.prev_dual_res = info.dual_res
    info.primal_res = self._primal_res(nr)
    info.primal_res_rel = info.primal_res / max(1.0, primal_rel_norm)
    info.dual_res = _inf_norm(pre.unscale_dual_res(nr.x))
    info.dual_res_rel = info.dual_res / max(1.0, dual_rel_norm)

  def _primal_res(self, v: Variables) -> float:
    d, pre = self.data, self.pre
    lo, up = d.x_l_idx, d.x_u_idx
    inf = _inf_norm(pre.unscale_primal_res_eq(v.y), pre.unscale_primal_res_ineq(v.z_l), pre.unscale_primal_res_ineq(v.z_u))
    return _signed_max(inf, pre.unscale_primal_res_b(v.z_bl[: d.n_x_l], lo), pre.unscale_primal_res_b(v.z_bu[: d.n_x_u], up))

  def _update_residuals_r(self) -> None:
    d, r, pre, info, nr, res, prox = self.data, self.r, self.pre, self.info, self.res_nr, self.res, self.prox
    nl, nu = d.n_x_l, d.n_x_u
    res.x = nr.x - info.rho * (r.x - prox.x)
    res.y = nr.y - info.delta * (prox.y - r.y)
    res.z_l = nr.z_l - info.delta * (prox.z_l - r.z_l)
    res.z_u = nr.z_u - info.delta * (prox.z_u - r.z_u)
    res.z_bl = res.z_bl.copy()
    res.z_bu = res.z_bu.copy()
    res.z_bl[:nl] = nr.z_bl[:nl] - info.delta * (prox.z_bl[:nl] - r.z_bl[:nl])
    res.z_bu[:nu] = nr.z_bu[:nu] - info.delta * (prox.z_bu[:nu] - r.z_bu[:nu])
    primal_rel_scaling = info.primal_res / info.primal_res_rel if info.primal_res_rel > 0 else 1.0
    dual_rel_scaling = info.dual_res / info.dual_res_rel if info.dual_res_rel > 0 else 1.0
    info.primal_res_reg = self._primal_res(res)
    info.primal_res_reg_rel = info.primal_res_reg / primal_rel_scaling
    info.dual_res_reg = _inf_norm(pre.unscale_dual_res(res.x))
    info.dual_res_reg_rel = info.dual_res_reg / dual_rel_scaling
    lo, up = d.x_l_idx, d.x_u_idx
    primal_prox_inf = _signed_max(
      _inf_norm(pre.unscale_dual_eq(prox.y - r.y), pre.unscale_dual_ineq(prox.z_l - r.z_l), pre.unscale_dual_ineq(prox.z_u - r.z_u)),
      pre.unscale_dual_b(prox.z_bl[:nl] - r.z_bl[:nl], lo),
      pre.unscale_dual_b(prox.z_bu[:nu] - r.z_bu[:nu], up),
    )
    info.primal_prox_inf = primal_prox_inf * info.delta
    info.dual_prox_inf = _inf_norm(pre.unscale_primal(r.x - prox.x)) * info.rho

  # the main loop -----------------------------------------------------------------------------

  def _factor(self, retry_changes_regularization: bool) -> bool | None:
    """Factor with PIQP's retry logic. Returns whether the regularization changed, or None on failure."""
    info, s = self.info, self.settings
    changed = False
    while not self.kkt.update_scalings_and_factor(self.data, s, self.enable_ir, info.rho, info.delta, self.r):
      if not self.enable_ir:
        self.enable_ir = True
        continue
      if info.factor_retires < s.max_factor_retires:
        info.delta *= 100
        info.rho *= 100
        info.factor_retires += 1
        info.reg_limit = min(10 * info.reg_limit, s.eps_abs)
        changed = retry_changes_regularization
        continue
      return None
    info.factor_retires = 0
    return changed

  def _record(self) -> None:
    i = self.info
    self.rows.append(
      [i.iter, i.primal_obj, i.dual_obj, i.duality_gap, i.primal_res, i.dual_res, i.rho, i.delta, i.mu, i.primal_step, i.dual_step, i.sigma]
    )

  def solve(self) -> Result:
    status = self._solve_impl()
    return self._result(status)

  def _solve_impl(self) -> int:
    d, s = self.data, self.settings
    n, p, m = d.n, d.p, d.m
    nl, nu = d.n_x_l, d.n_x_u
    hl, hu = d.h_l_idx, d.h_u_idx
    self.rows: list[list[float]] = []
    info = self.info = Info(reg_limit=s.reg_lower_limit, rho=s.rho_init, delta=s.delta_init)
    r = self.r = Variables.zeros(n, p, m)
    self.res_nr = Variables.zeros(n, p, m)
    self.res = Variables.zeros(n, p, m)
    step = Variables.zeros(n, p, m)
    r.s_l[hl] = r.z_l[hl] = 1.0
    r.s_u[hu] = r.z_u[hu] = 1.0
    r.s_bl[:nl] = r.z_bl[:nl] = 1.0
    r.s_bu[:nu] = r.z_bu[:nu] = 1.0
    self.enable_ir = s.iterative_refinement_always_enabled

    if self._factor(False) is None:
      return NUMERICS

    res = self.res
    res.x, res.y = -d.c, d.b.copy()
    res.z_l, res.z_u = -d.h_l, d.h_u.copy()
    res.z_bl, res.z_bu = -d.x_l, d.x_u.copy()
    res.s_l, res.s_u, res.s_bl, res.s_bu = np.zeros(m), np.zeros(m), np.zeros(n), np.zeros(n)
    self.kkt.solve(d, s, res, r)

    if m + nl + nu > 0:
      delta_s = 0.0
      delta_z = 0.0
      if m > 0:
        delta_s = max(delta_s, -r.s_l.min(), -r.s_u.min())
        delta_z = max(delta_z, -r.z_l.min(), -r.z_u.min())
      if nl > 0:
        delta_s = max(delta_s, -r.s_bl[:nl].min())
        delta_z = max(delta_z, -r.z_bl[:nl].min())
      if nu > 0:
        delta_s = max(delta_s, -r.s_bu[:nu].min())
        delta_z = max(delta_z, -r.z_bu[:nu].min())
      r.s_l[hl] += delta_s
      r.z_l[hl] += delta_z
      r.s_u[hu] += delta_s
      r.z_u[hu] += delta_z
      r.s_bl[:nl] += delta_s
      r.s_bu[:nu] += delta_s
      r.z_bl[:nl] += delta_z
      r.z_bu[:nu] += delta_z
      info.mu = max(self._calculate_mu(), 1e-10)
      for z, sl, idx in ((r.z_l, r.s_l, hl), (r.z_u, r.s_u, hu), (r.z_bl, r.s_bl, np.arange(nl)), (r.z_bu, r.s_bu, np.arange(nu))):
        c = z[idx] - delta_z
        z[idx] = (c + np.sqrt(c * c + 4 * info.mu)) / 2
        sl[idx] = z[idx] - c
      info.mu = self._calculate_mu()

    prox = self.prox = Variables.zeros(n, p, m)
    prox.x, prox.y, prox.z_l, prox.z_u = r.x.copy(), r.y.copy(), r.z_l.copy(), r.z_u.copy()
    prox.z_bl[:nl] = r.z_bl[:nl]
    prox.z_bu[:nu] = r.z_bu[:nu]

    while info.iter < s.max_iter:
      if info.iter == 0:
        self._update_residuals_nr()
        info.prev_primal_res = info.primal_res
        info.prev_dual_res = info.dual_res
      self._record()

      if (
        (info.primal_res < s.eps_abs or info.primal_res_rel < s.eps_rel)
        and (info.dual_res < s.eps_abs or info.dual_res_rel < s.eps_rel)
        and (not s.check_duality_gap or info.duality_gap < s.eps_duality_gap_abs or info.duality_gap_rel < s.eps_duality_gap_rel)
      ):
        return SOLVED

      self._update_residuals_r()

      if (
        info.no_dual_update > min(5, s.reg_finetune_dual_update_threshold)
        and info.primal_prox_inf > s.infeasibility_threshold
        and (info.primal_res_reg < s.eps_abs or info.primal_res_reg_rel < s.eps_rel)
      ):
        return PRIMAL_INFEASIBLE
      if (
        info.no_primal_update > min(5, s.reg_finetune_primal_update_threshold)
        and info.dual_prox_inf > s.infeasibility_threshold
        and (info.dual_res_reg < s.eps_abs or info.dual_res_reg_rel < s.eps_rel)
      ):
        return DUAL_INFEASIBLE

      info.iter += 1

      shifted = False
      for z, idx in ((r.z_l, hl), (r.z_u, hu)):
        small = idx[z[idx] < EPS]
        if small.size:
          z[small] += EPS
          shifted = True
      for z, k in ((r.z_bl, nl), (r.z_bu, nu)):
        if k > 0 and z[:k].min() < EPS:
          z[:k] += EPS
          shifted = True
      if shifted:
        info.mu = self._calculate_mu()

      if (
        info.no_primal_update > s.reg_finetune_primal_update_threshold and info.rho == info.reg_limit and info.reg_limit != s.reg_finetune_lower_limit
      ) or (
        info.no_dual_update > s.reg_finetune_dual_update_threshold and info.delta == info.reg_limit and info.reg_limit != s.reg_finetune_lower_limit
      ):
        if info.dual_prox_inf < s.infeasibility_threshold and info.primal_prox_inf < s.infeasibility_threshold:
          info.reg_limit = s.reg_finetune_lower_limit
          info.no_primal_update = 0
          info.no_dual_update = 0

      changed = self._factor(True)
      if changed is None:
        return NUMERICS
      if changed:
        self._update_residuals_r()

      res = self.res
      if m + nl + nu > 0:
        # predictor
        res.s_l = -r.s_l * r.z_l
        res.s_u = -r.s_u * r.z_u
        res.s_bl = res.s_bl.copy()
        res.s_bu = res.s_bu.copy()
        res.s_bl[:nl] = -r.s_bl[:nl] * r.z_bl[:nl]
        res.s_bu[:nu] = -r.s_bu[:nu] * r.z_bu[:nu]
        self.kkt.solve(d, s, res, step)
        alpha_s, alpha_z = self._calculate_step(step)
        alpha_s *= s.tau
        alpha_z *= s.tau
        sigma = (r.s_l + alpha_s * step.s_l) @ (r.z_l + alpha_z * step.z_l)
        sigma += (r.s_u + alpha_s * step.s_u) @ (r.z_u + alpha_z * step.z_u)
        sigma += (r.s_bl[:nl] + alpha_s * step.s_bl[:nl]) @ (r.z_bl[:nl] + alpha_z * step.z_bl[:nl])
        sigma += (r.s_bu[:nu] + alpha_s * step.s_bu[:nu]) @ (r.z_bu[:nu] + alpha_z * step.z_bu[:nu])
        with np.errstate(divide="ignore", invalid="ignore"):  # mu = 0 gives NaN, which the clip below maps to 1, as std::min does
          sigma /= info.mu * float(d.n_h_l + d.n_h_u + nl + nu)
        sigma = max(0.0, min(1.0, float(sigma)))
        info.sigma = sigma * sigma * sigma
        # corrector
        res.s_l = res.s_l + (-step.s_l * step.z_l + info.sigma * info.mu)
        res.s_u = res.s_u + (-step.s_u * step.z_u + info.sigma * info.mu)
        res.s_bl[:nl] += -step.s_bl[:nl] * step.z_bl[:nl] + info.sigma * info.mu
        res.s_bu[:nu] += -step.s_bu[:nu] * step.z_bu[:nu] + info.sigma * info.mu
        self.kkt.solve(d, s, res, step)
        alpha_s, alpha_z = self._calculate_step(step)
        info.primal_step = alpha_s * s.tau
        info.dual_step = alpha_z * s.tau
        r.x = r.x + info.primal_step * step.x
        r.y = r.y + info.dual_step * step.y
        r.z_l = r.z_l + info.dual_step * step.z_l
        r.z_u = r.z_u + info.dual_step * step.z_u
        r.z_bl[:nl] += info.dual_step * step.z_bl[:nl]
        r.z_bu[:nu] += info.dual_step * step.z_bu[:nu]
        r.s_l = r.s_l + info.primal_step * step.s_l
        r.s_u = r.s_u + info.primal_step * step.s_u
        r.s_bl[:nl] += info.primal_step * step.s_bl[:nl]
        r.s_bu[:nu] += info.primal_step * step.s_bu[:nu]
        mu_prev = info.mu
        info.mu = self._calculate_mu()
        with np.errstate(divide="ignore", invalid="ignore"):  # IEEE, as in C++: 0/0 is NaN and max(0, NaN) is 0
          mu_rate = max(0.0, float(np.float64(mu_prev - info.mu) / np.float64(mu_prev)))
        self._update_residuals_nr()
        if (
          info.dual_res < 0.95 * info.prev_dual_res
          or (info.dual_res < s.eps_abs or info.dual_res_rel < s.eps_rel)
          or (info.rho == s.reg_finetune_lower_limit and info.dual_prox_inf < s.infeasibility_threshold)
        ):
          prox.x = r.x.copy()
          info.rho = max(info.reg_limit, (1.0 - mu_rate) * info.rho)
        else:
          info.no_primal_update += 1
          if info.iter < 5 or info.dual_prox_inf < s.infeasibility_threshold:
            info.rho = max(info.reg_limit, (1.0 - 0.666 * mu_rate) * info.rho)
        if (
          info.primal_res < 0.95 * info.prev_primal_res
          or (info.primal_res < s.eps_abs or info.primal_res_rel < s.eps_rel)
          or (info.delta == s.reg_finetune_lower_limit and info.primal_prox_inf < s.infeasibility_threshold)
        ):
          prox.y, prox.z_l, prox.z_u = r.y.copy(), r.z_l.copy(), r.z_u.copy()
          prox.z_bl[:nl] = r.z_bl[:nl]
          prox.z_bu[:nu] = r.z_bu[:nu]
          info.delta = max(info.reg_limit, (1.0 - mu_rate) * info.delta)
        else:
          info.no_dual_update += 1
          if info.iter < 5 or info.primal_prox_inf < s.infeasibility_threshold:
            info.delta = max(info.reg_limit, (1.0 - 0.666 * mu_rate) * info.delta)
      else:
        self.kkt.solve(d, s, res, step)
        info.primal_step = info.dual_step = 1.0
        r.x = r.x + step.x
        r.y = r.y + step.y
        self._update_residuals_nr()
        if info.dual_res < 0.95 * info.prev_dual_res or (info.dual_res < s.eps_abs or info.dual_res_rel < s.eps_rel):
          prox.x = r.x.copy()
          info.rho = max(info.reg_limit, 0.1 * info.rho)
        else:
          info.no_primal_update += 1
          if info.iter < 5 or info.dual_prox_inf < s.infeasibility_threshold:
            info.rho = max(info.reg_limit, 0.5 * info.rho)
        if info.primal_res < 0.95 * info.prev_primal_res or (info.primal_res < s.eps_abs or info.primal_res_rel < s.eps_rel):
          prox.y = r.y.copy()
          info.delta = max(info.reg_limit, 0.1 * info.delta)
        else:
          info.no_dual_update += 1
          if info.iter < 5 or info.primal_prox_inf < s.infeasibility_threshold:
            info.delta = max(info.reg_limit, 0.5 * info.delta)
    return MAX_ITER_REACHED

  def _result(self, status: int) -> Result:
    d, pre, r = self.data, self.pre, self.r
    self.info.status = status
    nl, nu = d.n_x_l, d.n_x_u
    lo, up = d.x_l_idx, d.x_u_idx
    x = pre.unscale_primal(r.x)
    y = pre.unscale_dual_eq(r.y)
    z_l, z_u = pre.unscale_dual_ineq(r.z_l), pre.unscale_dual_ineq(r.z_u)
    s_l, s_u = pre.unscale_slack_ineq(r.s_l), pre.unscale_slack_ineq(r.s_u)
    # restore_dual: absent bounds get z = 0 and s = PIQP_INF, box values move to their variables
    s_l = np.where(z_l == 0, PIQP_INF, s_l)
    s_u = np.where(z_u == 0, PIQP_INF, s_u)
    z_bl, z_bu = np.zeros(d.n), np.zeros(d.n)
    s_bl, s_bu = np.full(d.n, PIQP_INF), np.full(d.n, PIQP_INF)
    z_bl[lo] = pre.unscale_dual_b(r.z_bl[:nl], lo)
    s_bl[lo] = pre.unscale_slack_b(r.s_bl[:nl], lo)
    z_bu[up] = pre.unscale_dual_b(r.z_bu[:nu], up)
    s_bu[up] = pre.unscale_slack_b(r.s_bu[:nu], up)
    trace = np.array(self.rows, dtype=np.float64).reshape(-1, len(TRACE_COLUMNS))
    return Result(status, x, y, z_l, z_u, z_bl, z_bu, s_l, s_u, s_bl, s_bu, self.info, trace)


def solve(qp: QP, settings: Settings | None = None) -> Result:
  return Solver(qp, settings).solve()
