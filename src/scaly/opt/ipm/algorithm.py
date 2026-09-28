"""PIQP 0.6.2's proximal interior-point iteration as generated code: the loop and its pieces."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...function import ConcreteFunction
from ...function.sugar import _while_node, while_loop
from ...ir.expr import (
  Expr,
  cast,
  concat,
  equal,
  gather,
  isfinite,
  logical_and,
  logical_not,
  logical_or,
  maximum,
  minimum,
  norm_inf,
  not_equal,
  reduce_max,
  reduce_min,
  scatter,
  stack,
  where,
)
from .kkt import EPS, KKT, Backend, Iterate, Kernels, Refinement
from .ruiz import ScaledQP, Scaling, ruiz, scale
from .structure import INF, QPStructure, QPValues

RUNNING, SOLVED, MAX_ITER_REACHED, PRIMAL_INFEASIBLE, DUAL_INFEASIBLE, NUMERICS = 0, 1, -1, -2, -3, -8
"""PIQP's status codes (``RUNNING`` only inside the loop; a solve that ends there reports ``SOLVED``)."""
INVALID_BOUNDS = -11
"""Not PIQP's: a bound the ``QPStructure`` declares finite arrived infinite, NaN or at or beyond
``INF``. The generated solver is specialised to which bounds exist, so it does not start; PIQP,
which reads the bounds at run time, would drop that one."""
TRACE_FIELDS = ("iter", "primal_obj", "dual_obj", "duality_gap", "primal_res", "dual_res", "rho", "delta", "mu", "primal_step", "dual_step", "sigma")
"""The columns of a trace row: PIQP's verbose table, and sigma."""
SCALARS = (
  "iter",
  "status",
  "rho",
  "delta",
  "mu",
  "sigma",
  "primal_step",
  "dual_step",
  "primal_res",
  "primal_res_rel",
  "dual_res",
  "dual_res_rel",
  "prev_primal_res",
  "prev_dual_res",
  "primal_obj",
  "dual_obj",
  "duality_gap",
  "duality_gap_rel",
  "primal_res_reg",
  "primal_res_reg_rel",
  "dual_res_reg",
  "dual_res_reg_rel",
  "primal_prox_inf",
  "dual_prox_inf",
  "reg_limit",
  "no_primal_update",
  "no_dual_update",
  "ir",
)
INFO_FIELDS = SCALARS
"""The entries of a solve's ``info`` vector, in order: PIQP's ``info`` fields the loop keeps."""


@dataclass(frozen=True)
class Settings:
  """PIQP 0.6.2's settings that the generated solver folds in as constants (defaults as PIQP's)."""

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
  max_factor_retires: int = 10

  def __post_init__(self) -> None:
    if self.max_iter < 1:
      raise ValueError(f"max_iter must be at least 1, as PIQP requires, got {self.max_iter}")

  def refinement(self) -> Refinement:
    """The retry and refinement settings, as the KKT system takes them."""
    return Refinement(
      always=self.iterative_refinement_always_enabled,
      eps_abs=self.iterative_refinement_eps_abs,
      eps_rel=self.iterative_refinement_eps_rel,
      max_iter=self.iterative_refinement_max_iter,
      min_improvement_rate=self.iterative_refinement_min_improvement_rate,
      static_eps=self.iterative_refinement_static_regularization_eps,
      static_rel=self.iterative_refinement_static_regularization_rel,
      max_factor_retires=self.max_factor_retires,
      reg_limit_cap=self.eps_abs,
    )


@dataclass
class Layout:
  """Named segments of a flat vector."""

  sizes: dict[str, int] = field(default_factory=dict)

  def offsets(self) -> dict[str, tuple[int, int]]:
    """Each segment's ``(start, stop)`` in the vector."""
    out, at = {}, 0
    for name, size in self.sizes.items():
      out[name] = (at, at + size)
      at += size
    return out

  @property
  def size(self) -> int:
    """The vector's length."""
    return sum(self.sizes.values())

  def unpack(self, v: Expr) -> dict[str, Expr]:
    """The segments of ``v`` by name, scalars as scalars."""
    out = {}
    for name, (a, b) in self.offsets().items():
      out[name] = v[a] if name in SCALARS else v[a:b]
    return out

  def pack(self, values: dict[str, Expr]) -> Expr:
    """The vector holding ``values``, segment by segment."""
    return concat([values[name].reshape((size,)) for name, size in self.sizes.items()])


def state_layout(s: QPStructure) -> Layout:
  """The loop's carry: the iterate, the proximal centres (``xi``, ``lam``, ``nu_*``), the
  unregularized residuals of the iterate (``r_*``, as PIQP keeps ``res_nr`` from one iteration to
  the next) and PIQP's info scalars (``SCALARS``)."""
  nl, nu = s.x_l_idx.size, s.x_u_idx.size
  sizes = dict(zip(("x", "y", "z_l", "z_u", "z_bl", "z_bu", "s_l", "s_u", "s_bl", "s_bu"), Iterate.sizes(s), strict=True))
  sizes.update({"xi": s.n, "lam": s.p, "nu_l": s.m, "nu_u": s.m, "nu_bl": nl, "nu_bu": nu})
  sizes.update({"r_x": s.n, "r_y": s.p, "r_z_l": s.m, "r_z_u": s.m, "r_z_bl": nl, "r_z_bu": nu})
  sizes.update({name: 1 for name in SCALARS})
  return Layout(sizes)


def _smax(acc: Expr, *vs: Expr) -> Expr:
  """``max(acc, v_i)`` without absolute values, over non-empty ``vs`` (PIQP's box terms)."""
  for v in vs:
    if v.size:
      acc = maximum(acc, reduce_max(v))
  return acc


def _norm(*vs: Expr) -> Expr:
  """The infinity norm over several vectors (zero when all are empty)."""
  parts = [norm_inf(v) for v in vs if v.size]
  out = Expr.const(0.0)
  for part in parts:
    out = maximum(out, part)
  return out


class Iteration:
  """The pieces of PIQP's loop over one scaled problem, as expressions."""

  def __init__(self, s: QPStructure, q: ScaledQP, kernels: Kernels, settings: Settings, *, data: Expr | None = None):
    self.s, self.q, self.settings = s, q, settings
    self.kkt = KKT(kernels, q, data=data)
    sc_ = q.scaling
    n, p = s.n, s.p
    self.c_inv = 1.0 / sc_.c
    self.d_x, self.d_eq, self.d_in = sc_.delta[:n], sc_.delta[n : n + p], sc_.delta[n + p :]
    self.d_bl, self.d_bu = gather(sc_.delta_b, s.x_l_idx), gather(sc_.delta_b, s.x_u_idx)
    self.n_bounds = s.n_bounds

  # --- PIQP's scale maps ------------------------------------------------------------------------

  def unscale_cost(self, v: Expr) -> Expr:
    """An objective value in the problem's own units."""
    return v * self.c_inv

  def unscale_dual_res(self, r: Expr) -> Expr:
    """A dual residual in the problem's own units."""
    return r * self.c_inv / self.d_x

  def mu(self, v: Iterate) -> Expr:
    """The complementarity measure: the mean of ``s z`` over every bound (zero with none)."""
    if not self.n_bounds:
      return Expr.const(0.0)
    total = (v.s_l * v.z_l).sum() + (v.s_u * v.z_u).sum() + (v.s_bl * v.z_bl).sum() + (v.s_bu * v.z_bu).sum()
    return total / float(self.n_bounds)

  # --- residuals (update_residuals_nr, update_residuals_r) --------------------------------------

  def residuals(self, v: Iterate) -> tuple[Iterate, dict[str, Expr]]:
    """``update_residuals_nr``: the unregularized residuals and the info they give (without the
    ``prev_*`` bookkeeping, which the caller does)."""
    s, k, d = self.s, self.kkt, self.q.values
    xb = self.q.x_b
    lo, up, hl, hu = s.x_l_idx, s.x_u_idx, s.h_l_idx, s.h_u_idx
    zero_n = Expr.const(np.zeros(s.n))
    nr_y = -(k.A @ v.x) if s.p else Expr.const(np.zeros(0))
    work_x = (k.A.T @ v.y if s.p else zero_n) + (k.G.T @ (v.z_u - v.z_l) if s.m else zero_n)
    g_x = k.G @ v.x if s.m else Expr.const(np.zeros(0))
    px = k.P_times(v.x)
    nr_x = -px
    dual_rel = _norm(self.unscale_dual_res(nr_x))
    tmp = (v.x * px).sum()
    primal_obj, dual_obj = 0.5 * tmp, -0.5 * tmp
    gap_rel = self.unscale_cost(tmp.abs())
    terms = [((d.c * v.x).sum(), True)]
    if s.p:
      terms.append(((d.b * v.y).sum(), False))
    if hl.size:
      terms.append((-(gather(d.h_l, hl) * gather(v.z_l, hl)).sum(), False))
    if hu.size:
      terms.append(((gather(d.h_u, hu) * gather(v.z_u, hu)).sum(), False))
    if lo.size:
      terms.append((-(d.x_l * v.z_bl).sum(), False))
    if up.size:
      terms.append(((d.x_u * v.z_bu).sum(), False))
    for value, primal in terms:
      if primal:
        primal_obj = primal_obj + value
      else:
        dual_obj = dual_obj - value
      gap_rel = maximum(gap_rel, self.unscale_cost(value.abs()))
    gap = (primal_obj - dual_obj).abs()
    info = {
      "primal_obj": self.unscale_cost(primal_obj),
      "dual_obj": self.unscale_cost(dual_obj),
      "duality_gap": self.unscale_cost(gap),
    }
    info["duality_gap_rel"] = info["duality_gap"] / maximum(1.0, gap_rel)
    nr_x = nr_x - d.c
    dual_rel = maximum(dual_rel, _norm(self.unscale_dual_res(d.c)))
    if lo.size:
      work_x = work_x - _scatter(s.n, lo, gather(xb, lo) * v.z_bl)
    if up.size:
      work_x = work_x + _scatter(s.n, up, gather(xb, up) * v.z_bu)
    dual_rel = maximum(dual_rel, _norm(self.unscale_dual_res(work_x)))
    nr_x = nr_x - work_x
    primal_rel = _norm(nr_y / self.d_eq) if s.p else Expr.const(0.0)
    if s.p:
      nr_y = nr_y + d.b
      primal_rel = maximum(primal_rel, _norm(d.b / self.d_eq))
    z_l = Expr.const(np.zeros(s.m))
    z_u = Expr.const(np.zeros(s.m))
    if hl.size:
      d_hl = gather(self.d_in, hl)
      primal_rel = _smax(primal_rel, gather(g_x, hl) / d_hl, gather(d.h_l, hl) / d_hl, gather(v.s_l, hl) / d_hl)
      z_l = _scatter(s.m, hl, gather(g_x, hl) - gather(d.h_l, hl) - gather(v.s_l, hl))
    if hu.size:
      d_hu = gather(self.d_in, hu)
      primal_rel = _smax(primal_rel, -gather(g_x, hu) / d_hu, gather(d.h_u, hu) / d_hu, gather(v.s_u, hu) / d_hu)
      z_u = _scatter(s.m, hu, -gather(g_x, hu) + gather(d.h_u, hu) - gather(v.s_u, hu))
    z_bl = gather(xb, lo) * gather(v.x, lo)
    z_bu = -gather(xb, up) * gather(v.x, up)
    if lo.size:
      primal_rel = _smax(primal_rel, z_bl / self.d_bl, d.x_l / self.d_bl, v.s_bl / self.d_bl)
      z_bl = z_bl - d.x_l - v.s_bl
    if up.size:
      primal_rel = _smax(primal_rel, z_bu / self.d_bu, d.x_u / self.d_bu, v.s_bu / self.d_bu)
      z_bu = z_bu + d.x_u - v.s_bu
    nr = Iterate(nr_x, nr_y, z_l, z_u, z_bl, z_bu, v.s_l * 0.0, v.s_u * 0.0, v.s_bl * 0.0, v.s_bu * 0.0)
    info["primal_res"] = self.primal_res(nr)
    info["primal_res_rel"] = info["primal_res"] / maximum(1.0, primal_rel)
    info["dual_res"] = _norm(self.unscale_dual_res(nr_x))
    info["dual_res_rel"] = info["dual_res"] / maximum(1.0, dual_rel)
    return nr, info

  def primal_res(self, r: Iterate) -> Expr:
    """The unscaled primal residual's infinity norm, with PIQP's signed maximum over the box terms."""
    inf = _norm(*(v for v in (r.y / self.d_eq if self.s.p else None, r.z_l / self.d_in, r.z_u / self.d_in) if v is not None))
    return _smax(inf, r.z_bl / self.d_bl, r.z_bu / self.d_bu)

  def regularized(self, nr: Iterate, v: Iterate, prox: Iterate, rho: Expr, delta: Expr, info: dict[str, Expr]) -> tuple[Iterate, dict[str, Expr]]:
    """``update_residuals_r``: the residuals of the regularized problem, and the proximal
    infeasibility measures."""
    res = Iterate(
      nr.x - rho * (v.x - prox.x),
      nr.y - delta * (prox.y - v.y),
      nr.z_l - delta * (prox.z_l - v.z_l),
      nr.z_u - delta * (prox.z_u - v.z_u),
      nr.z_bl - delta * (prox.z_bl - v.z_bl),
      nr.z_bu - delta * (prox.z_bu - v.z_bu),
      nr.s_l,
      nr.s_u,
      nr.s_bl,
      nr.s_bu,
    )
    primal_scale = where(info["primal_res_rel"] > 0.0, info["primal_res"] / info["primal_res_rel"], 1.0)
    dual_scale = where(info["dual_res_rel"] > 0.0, info["dual_res"] / info["dual_res_rel"], 1.0)
    out = {"primal_res_reg": self.primal_res(res), "dual_res_reg": _norm(self.unscale_dual_res(res.x))}
    out["primal_res_reg_rel"] = out["primal_res_reg"] / primal_scale
    out["dual_res_reg_rel"] = out["dual_res_reg"] / dual_scale
    c_inv = self.c_inv
    prox_inf = _norm(
      *(
        e
        for e in (
          (prox.y - v.y) * c_inv * self.d_eq if self.s.p else None,
          (prox.z_l - v.z_l) * c_inv * self.d_in,
          (prox.z_u - v.z_u) * c_inv * self.d_in,
        )
        if e is not None
      )
    )
    prox_inf = _smax(prox_inf, (prox.z_bl - v.z_bl) * c_inv * self.d_bl, (prox.z_bu - v.z_bu) * c_inv * self.d_bu)
    out["primal_prox_inf"] = prox_inf * delta
    out["dual_prox_inf"] = _norm((v.x - prox.x) * self.d_x) * rho
    return res, out

  # --- step lengths -------------------------------------------------------------------------

  def step_lengths(self, v: Iterate, dv: Iterate) -> tuple[Expr, Expr]:
    """``calculate_step``: the largest steps in ``[0, 1]`` keeping slacks and duals non-negative."""

    def longest(pairs: list[tuple[Expr, Expr]]) -> Expr:
      alpha = Expr.const(1.0)
      for x, dx in pairs:
        if x.size:
          ratio = where(dx < 0.0, -x / dx, INF)
          alpha = minimum(alpha, reduce_min(ratio))
      return alpha

    alpha_s = longest([(v.s_l, dv.s_l), (v.s_u, dv.s_u), (v.s_bl, dv.s_bl), (v.s_bu, dv.s_bu)])
    alpha_z = longest([(v.z_l, dv.z_l), (v.z_u, dv.z_u), (v.z_bl, dv.z_bl), (v.z_bu, dv.z_bu)])
    return alpha_s, alpha_z


def _scatter(size: int, idx: np.ndarray, values: Expr) -> Expr:
  return scatter(values, idx, (size,))


def _iterate_of(st: dict[str, Expr]) -> Iterate:
  return Iterate(st["x"], st["y"], st["z_l"], st["z_u"], st["z_bl"], st["z_bu"], st["s_l"], st["s_u"], st["s_bl"], st["s_bu"])


def _prox_of(st: dict[str, Expr]) -> Iterate:
  none = Expr.const(np.zeros(0))
  return Iterate(st["xi"], st["lam"], st["nu_l"], st["nu_u"], st["nu_bl"], st["nu_bu"], none, none, none, none)


RESIDUALS = ("r_x", "r_y", "r_z_l", "r_z_u", "r_z_bl", "r_z_bu")


def _residuals_of(st: dict[str, Expr]) -> Iterate:
  """The carried residuals, with the zero complementarity parts ``residuals`` gives."""
  x, y, z_l, z_u, z_bl, z_bu = (st[k] for k in RESIDUALS)
  return Iterate(x, y, z_l, z_u, z_bl, z_bu, z_l * 0.0, z_u * 0.0, z_bl * 0.0, z_bu * 0.0)


def _set_residuals(st: dict[str, Expr], nr: Iterate) -> None:
  for name, e in zip(RESIDUALS, (nr.x, nr.y, nr.z_l, nr.z_u, nr.z_bl, nr.z_bu), strict=True):
    st[name] = e


def _set_iterate(st: dict[str, Expr], v: Iterate) -> None:
  for name, e in zip(("x", "y", "z_l", "z_u", "z_bl", "z_bu", "s_l", "s_u", "s_bl", "s_bu"), v.fields(), strict=True):
    st[name] = e


class Solver:
  """PIQP 0.6.2 over one ``QPStructure``, as expressions: ``solve(values)`` builds the whole solve
  (equilibration, initial point, the loop, unscaling) for run-time ``values``."""

  def __init__(self, s: QPStructure, backend: Backend = "sparse", settings: Settings | None = None, *, name: str = "ipm"):
    self.s, self.backend, self.settings, self.name = s, backend, settings or Settings(), name
    self.layout = state_layout(s)

  # The loop's params: the scaled data and the scalings.
  def _param_symbols(self) -> tuple[list[Expr], ScaledQP]:
    s = self.s
    shapes = {
      "P": s.P_rows.size,
      "c": s.n,
      "A": s.A_rows.size,
      "b": s.p,
      "G": s.G_rows.size,
      "h_l": s.m,
      "h_u": s.m,
      "x_l": s.x_l_idx.size,
      "x_u": s.x_u_idx.size,
    }
    syms = {k: Expr.sym(f"q_{k}", (size,)) for k, size in shapes.items()}
    x_b = Expr.sym("q_x_b", (s.n,))
    delta, delta_b, cost = Expr.sym("q_delta", (s.n + s.p + s.m,)), Expr.sym("q_delta_b", (s.n,)), Expr.sym("q_cost", ())
    q = ScaledQP(QPValues(**syms), x_b, Scaling(delta, delta_b, cost))
    return [*syms.values(), x_b, delta, delta_b, cost], q

  @staticmethod
  def _param_values(q: ScaledQP) -> list[Expr]:
    v = q.values
    return [v.P, v.c, v.A, v.b, v.G, v.h_l, v.h_u, v.x_l, v.x_u, q.x_b, q.scaling.delta, q.scaling.delta_b, q.scaling.c]

  # --- the pieces of the loop ---------------------------------------------------------------

  def initial_state(self, it: Iteration, invalid: Expr | None = None) -> dict[str, Expr]:
    """PIQP's initial point: one solve at ``rho_init`` and ``delta_init``, shifted into the interior
    and projected onto the central path, with its residuals. When the first factorization fails
    (after every retry), or ``invalid`` says the bounds cannot be used, the state is the start PIQP
    returns then: the ones point before any solve, with no residuals."""
    s, st = self.s, self.settings
    hl, hu = s.h_l_idx, s.h_u_idx
    nl, nu = s.x_l_idx.size, s.x_u_idx.size
    ones_l = _scatter(s.m, hl, Expr.const(np.ones(hl.size))) if hl.size else Expr.const(np.zeros(s.m))
    ones_u = _scatter(s.m, hu, Expr.const(np.ones(hu.size))) if hu.size else Expr.const(np.zeros(s.m))
    one = Iterate(
      Expr.const(np.zeros(s.n)),
      Expr.const(np.zeros(s.p)),
      ones_l,
      ones_u,
      Expr.const(np.ones(nl)),
      Expr.const(np.ones(nu)),
      ones_l,
      ones_u,
      Expr.const(np.ones(nl)),
      Expr.const(np.ones(nu)),
    )
    ir0 = 1.0 if st.iterative_refinement_always_enabled else 0.0
    factor = it.kkt.factor(st.rho_init, st.delta_init, one, reg_limit=st.reg_lower_limit, ir=ir0)
    d = it.q.values
    rhs = Iterate(-d.c, d.b, -d.h_l, d.h_u, -d.x_l, d.x_u, one.s_l * 0.0, one.s_u * 0.0, one.s_bl * 0.0, one.s_bu * 0.0)
    v = factor.solve(rhs)
    mu = Expr.const(0.0)
    if s.m + nl + nu > 0:
      delta_s, delta_z = Expr.const(0.0), Expr.const(0.0)
      for sv, zv in ((v.s_l, v.z_l), (v.s_u, v.z_u), (v.s_bl, v.z_bl), (v.s_bu, v.z_bu)):
        if sv.size:
          delta_s = maximum(delta_s, -reduce_min(sv))
          delta_z = maximum(delta_z, -reduce_min(zv))
      mask_l = _mask(s.m, hl)
      mask_u = _mask(s.m, hu)
      shifted = Iterate(
        v.x,
        v.y,
        v.z_l + delta_z * mask_l,
        v.z_u + delta_z * mask_u,
        v.z_bl + delta_z,
        v.z_bu + delta_z,
        v.s_l + delta_s * mask_l,
        v.s_u + delta_s * mask_u,
        v.s_bl + delta_s,
        v.s_bu + delta_s,
      )
      mu = maximum(it.mu(shifted), 1e-10)

      def project(z: Expr, rows: np.ndarray | None) -> tuple[Expr, Expr]:
        """Onto the central path ``s z = mu``: rows without the bound keep their zeros."""
        c = z - delta_z
        z_new = (c + (c * c + 4.0 * mu).sqrt()) / 2.0
        s_new = z_new - c
        if rows is None:
          return z_new, s_new
        on = Expr.const(np.isin(np.arange(s.m), rows), dtype="bool")
        return where(on, z_new, z), where(on, s_new, 0.0)

      z_l, s_l = project(shifted.z_l, hl)
      z_u, s_u = project(shifted.z_u, hu)
      z_bl, s_bl = project(shifted.z_bl, None)
      z_bu, s_bu = project(shifted.z_bu, None)
      v = Iterate(v.x, v.y, z_l, z_u, z_bl, z_bu, s_l, s_u, s_bl, s_bu)
      mu = it.mu(v)
    nr, info = it.residuals(v)
    state: dict[str, Expr] = {}
    _set_iterate(state, v)
    state.update({"xi": v.x, "lam": v.y, "nu_l": v.z_l, "nu_u": v.z_u, "nu_bl": v.z_bl, "nu_bu": v.z_bu})
    state.update({k: Expr.const(0.0) for k in SCALARS})
    state.update(info)
    _set_residuals(state, nr)
    header = {"rho": factor.rho, "delta": factor.delta, "reg_limit": factor.head["reg_limit"], "ir": factor.ir}
    state.update({**header, "mu": mu, "prev_primal_res": info["primal_res"], "prev_dual_res": info["dual_res"]})
    start: dict[str, Expr] = {}
    _set_iterate(start, one)
    start.update({"xi": one.x, "lam": one.y, "nu_l": one.z_l, "nu_u": one.z_u, "nu_bl": one.z_bl, "nu_bu": one.z_bu})
    start.update({k: state[k] * 0.0 for k in (*RESIDUALS, *SCALARS)})
    start.update(header)
    abort = logical_not(factor.ok) if invalid is None else logical_or(logical_not(factor.ok), invalid)
    state = {k: where(abort, start[k], state[k]) for k in state}
    status = where(factor.ok, float(RUNNING), float(NUMERICS))
    state["status"] = status if invalid is None else where(invalid, float(INVALID_BOUNDS), status)
    return state

  def converged(self, st: dict[str, Expr]) -> Expr:
    """PIQP's convergence test on a state: primal and dual residuals and, if checked, the duality gap, each absolute or relative."""
    t = self.settings
    primal = logical_or(st["primal_res"] < t.eps_abs, st["primal_res_rel"] < t.eps_rel)
    dual = logical_or(st["dual_res"] < t.eps_abs, st["dual_res_rel"] < t.eps_rel)
    ok = logical_and(primal, dual)
    if t.check_duality_gap:
      ok = logical_and(ok, logical_or(st["duality_gap"] < t.eps_duality_gap_abs, st["duality_gap_rel"] < t.eps_duality_gap_rel))
    return ok

  def step(self, it: Iteration, st: dict[str, Expr]) -> dict[str, Expr]:
    """One pass of PIQP's loop after its convergence test: the infeasibility tests, then the step."""
    s, t = self.s, self.settings
    v, prox = _iterate_of(st), _prox_of(st)
    # The residuals of this iterate, as the previous pass (or the initial point) left them.
    nr = _residuals_of(st)
    info = {k: st[k] for k in ("primal_res", "primal_res_rel", "dual_res", "dual_res_rel")}
    res, reg = it.regularized(nr, v, prox, st["rho"], st["delta"], {**info})
    reg_top = reg  # what an exit at this pass reports, as PIQP returns right after computing it
    primal_inf = logical_and(
      logical_and(st["no_dual_update"] > float(min(5, t.reg_finetune_dual_update_threshold)), reg["primal_prox_inf"] > t.infeasibility_threshold),
      logical_or(reg["primal_res_reg"] < t.eps_abs, reg["primal_res_reg_rel"] < t.eps_rel),
    )
    dual_inf = logical_and(
      logical_and(st["no_primal_update"] > float(min(5, t.reg_finetune_primal_update_threshold)), reg["dual_prox_inf"] > t.infeasibility_threshold),
      logical_or(reg["dual_res_reg"] < t.eps_abs, reg["dual_res_reg_rel"] < t.eps_rel),
    )
    stop = where(primal_inf, float(PRIMAL_INFEASIBLE), where(dual_inf, float(DUAL_INFEASIBLE), float(RUNNING)))

    new = dict(st)
    new.update(reg)
    new["iter"] = st["iter"] + 1.0
    # The boundary shift: duals below machine epsilon move off the boundary.
    shifted = False
    z_l, z_u, z_bl, z_bu = v.z_l, v.z_u, v.z_bl, v.z_bu
    any_shift = Expr.const(False, dtype="bool")
    for idx, name in ((s.h_l_idx, "z_l"), (s.h_u_idx, "z_u")):
      if idx.size:
        z = z_l if name == "z_l" else z_u
        small = logical_and(Expr.const(np.isin(np.arange(s.m), idx), dtype="bool"), z < EPS)
        any_shift = logical_or(any_shift, reduce_max(cast(small, "float64")) > 0.0)
        z = where(small, z + EPS, z)
        if name == "z_l":
          z_l = z
        else:
          z_u = z
        shifted = True
    for name, k in (("z_bl", s.x_l_idx.size), ("z_bu", s.x_u_idx.size)):
      if k:
        z = z_bl if name == "z_bl" else z_bu
        low = reduce_min(z) < EPS
        any_shift = logical_or(any_shift, low)
        z = where(low, z + EPS, z)
        if name == "z_bl":
          z_bl = z
        else:
          z_bu = z
        shifted = True
    v = Iterate(v.x, v.y, z_l, z_u, z_bl, z_bu, v.s_l, v.s_u, v.s_bl, v.s_bu)
    mu = where(any_shift, it.mu(v), st["mu"]) if shifted else st["mu"]
    # The fine-tuning switch: lower the regularization floor when an update keeps failing.
    finetune = logical_and(
      logical_or(
        logical_and(
          logical_and(st["no_primal_update"] > float(t.reg_finetune_primal_update_threshold), equal(st["rho"], st["reg_limit"])),
          not_equal(st["reg_limit"], t.reg_finetune_lower_limit),
        ),
        logical_and(
          logical_and(st["no_dual_update"] > float(t.reg_finetune_dual_update_threshold), equal(st["delta"], st["reg_limit"])),
          not_equal(st["reg_limit"], t.reg_finetune_lower_limit),
        ),
      ),
      logical_and(reg["dual_prox_inf"] < t.infeasibility_threshold, reg["primal_prox_inf"] < t.infeasibility_threshold),
    )
    reg_limit = where(finetune, t.reg_finetune_lower_limit, st["reg_limit"])
    no_primal = where(finetune, 0.0, st["no_primal_update"])
    no_dual = where(finetune, 0.0, st["no_dual_update"])
    factor = it.kkt.factor(st["rho"], st["delta"], v, reg_limit=reg_limit, ir=st["ir"])
    rho, delta, reg_limit = factor.rho, factor.delta, factor.head["reg_limit"]
    # A retry that changed the regularization updates the regularized residuals, at the shifted duals.
    res_c, reg_c = it.regularized(nr, v, prox, rho, delta, {**info})
    changed = factor.changed
    res = Iterate(*(where(changed, a, b) for a, b in zip(res_c.fields(), res.fields(), strict=True)))
    reg = {key: where(changed, reg_c[key], reg[key]) for key in reg}
    new.update(reg)

    if self.s.n_bounds:
      rhs = Iterate(res.x, res.y, res.z_l, res.z_u, res.z_bl, res.z_bu, -v.s_l * v.z_l, -v.s_u * v.z_u, -v.s_bl * v.z_bl, -v.s_bu * v.z_bu)
      dv = factor.solve(rhs)
      alpha_s, alpha_z = it.step_lengths(v, dv)
      alpha_s, alpha_z = alpha_s * t.tau, alpha_z * t.tau
      sig = ((v.s_l + alpha_s * dv.s_l) * (v.z_l + alpha_z * dv.z_l)).sum()
      sig = sig + ((v.s_u + alpha_s * dv.s_u) * (v.z_u + alpha_z * dv.z_u)).sum()
      sig = sig + ((v.s_bl + alpha_s * dv.s_bl) * (v.z_bl + alpha_z * dv.z_bl)).sum()
      sig = sig + ((v.s_bu + alpha_s * dv.s_bu) * (v.z_bu + alpha_z * dv.z_bu)).sum()
      sig = sig / (mu * float(self.s.n_bounds))
      # std::max(0, std::min(1, sigma)): NaN (mu = 0) becomes 1.
      sig = where(sig < 1.0, where(0.0 < sig, sig, 0.0), 1.0)
      sigma = sig * sig * sig
      sm = sigma * mu
      rhs2 = Iterate(
        rhs.x,
        rhs.y,
        rhs.z_l,
        rhs.z_u,
        rhs.z_bl,
        rhs.z_bu,
        rhs.s_l + (-dv.s_l * dv.z_l + sm) * _mask(s.m, s.h_l_idx),
        rhs.s_u + (-dv.s_u * dv.z_u + sm) * _mask(s.m, s.h_u_idx),
        rhs.s_bl + (-dv.s_bl * dv.z_bl + sm),
        rhs.s_bu + (-dv.s_bu * dv.z_bu + sm),
      )
      dv = factor.solve(rhs2)
      alpha_s, alpha_z = it.step_lengths(v, dv)
      p_step, d_step = alpha_s * t.tau, alpha_z * t.tau
      v_new = Iterate(
        v.x + p_step * dv.x,
        v.y + d_step * dv.y,
        v.z_l + d_step * dv.z_l,
        v.z_u + d_step * dv.z_u,
        v.z_bl + d_step * dv.z_bl,
        v.z_bu + d_step * dv.z_bu,
        v.s_l + p_step * dv.s_l,
        v.s_u + p_step * dv.s_u,
        v.s_bl + p_step * dv.s_bl,
        v.s_bu + p_step * dv.s_bu,
      )
      mu_new = it.mu(v_new)
      mu_rate = (mu - mu_new) / mu
      mu_rate = where(0.0 < mu_rate, mu_rate, 0.0)  # std::max(0, NaN) is 0
      primal_factor, primal_damped = 1.0 - mu_rate, 1.0 - 0.666 * mu_rate
      dual_factor, dual_damped = primal_factor, primal_damped
    else:
      dv = factor.solve(Iterate(res.x, res.y, res.z_l, res.z_u, res.z_bl, res.z_bu, v.s_l, v.s_u, v.s_bl, v.s_bu))
      p_step, d_step = Expr.const(1.0), Expr.const(1.0)
      v_new = Iterate(v.x + dv.x, v.y + dv.y, v.z_l, v.z_u, v.z_bl, v.z_bu, v.s_l, v.s_u, v.s_bl, v.s_bu)
      sigma, mu_new = st["sigma"], mu
      primal_factor = dual_factor = Expr.const(0.1)
      primal_damped = dual_damped = Expr.const(0.5)

    nr_new, info_new = it.residuals(v_new)
    prev_primal, prev_dual = st["primal_res"], st["dual_res"]
    # Proximal updates: the centres move when a residual falls by at least 5%.
    dual_ok = logical_or(info_new["dual_res"] < 0.95 * prev_dual, logical_or(info_new["dual_res"] < t.eps_abs, info_new["dual_res_rel"] < t.eps_rel))
    primal_ok = logical_or(
      info_new["primal_res"] < 0.95 * prev_primal, logical_or(info_new["primal_res"] < t.eps_abs, info_new["primal_res_rel"] < t.eps_rel)
    )
    if self.s.n_bounds:
      dual_ok = logical_or(dual_ok, logical_and(equal(rho, t.reg_finetune_lower_limit), reg["dual_prox_inf"] < t.infeasibility_threshold))
      primal_ok = logical_or(primal_ok, logical_and(equal(delta, t.reg_finetune_lower_limit), reg["primal_prox_inf"] < t.infeasibility_threshold))
    early = new["iter"] < 5.0
    rho_new = where(
      dual_ok,
      maximum(reg_limit, primal_factor * rho),
      where(logical_or(early, reg["dual_prox_inf"] < t.infeasibility_threshold), maximum(reg_limit, primal_damped * rho), rho),
    )
    delta_new = where(
      primal_ok,
      maximum(reg_limit, dual_factor * delta),
      where(logical_or(early, reg["primal_prox_inf"] < t.infeasibility_threshold), maximum(reg_limit, dual_damped * delta), delta),
    )
    new_prox = {
      "xi": where(dual_ok, v_new.x, prox.x),
      "lam": where(primal_ok, v_new.y, prox.y),
      "nu_l": where(primal_ok, v_new.z_l, prox.z_l),
      "nu_u": where(primal_ok, v_new.z_u, prox.z_u),
      "nu_bl": where(primal_ok, v_new.z_bl, prox.z_bl),
      "nu_bu": where(primal_ok, v_new.z_bu, prox.z_bu),
    }
    if not self.s.n_bounds:
      # Only x and y centre moves in PIQP's branch without inequalities.
      new_prox.update({k: getattr(prox, attr) for k, attr in (("nu_l", "z_l"), ("nu_u", "z_u"), ("nu_bl", "z_bl"), ("nu_bu", "z_bu"))})
    _set_iterate(new, v_new)
    _set_residuals(new, nr_new)
    new.update(new_prox)
    new.update(info_new)
    new.update(
      {
        "mu": mu_new,
        "sigma": sigma,
        "primal_step": p_step,
        "dual_step": d_step,
        "prev_primal_res": prev_primal,
        "prev_dual_res": prev_dual,
        "rho": rho_new,
        "delta": delta_new,
        "reg_limit": reg_limit,
        "ir": factor.ir,
        "no_primal_update": no_primal + where(dual_ok, 0.0, 1.0),
        "no_dual_update": no_dual + where(primal_ok, 0.0, 1.0),
      }
    )
    at_limit = new["iter"] >= float(t.max_iter)
    new["status"] = where(at_limit, float(MAX_ITER_REACHED), float(RUNNING))
    # An infeasibility found at the top of the pass ends it there, with the regularized residuals
    # that found it. A factorization that fails every retry ends it later, after the count, the
    # boundary shift, the fine-tuning switch and the retries' regularization: those stay too.
    infeasible = stop < 0.0
    failed = logical_and(logical_not(infeasible), logical_not(factor.ok))
    halted = logical_or(infeasible, failed)
    out = {key: where(halted, st[key], new[key]) for key in self.layout.sizes}
    for key, value in reg_top.items():
      out[key] = where(halted, value, new[key])
    done = {"iter": new["iter"], "z_l": v.z_l, "z_u": v.z_u, "z_bl": v.z_bl, "z_bu": v.z_bu, "mu": mu}
    done.update({"no_primal_update": no_primal, "no_dual_update": no_dual, "rho": rho, "delta": delta, "reg_limit": reg_limit, "ir": factor.ir})
    for key, value in done.items():
      out[key] = where(failed, value, out[key])
    out["status"] = where(infeasible, stop, where(failed, float(NUMERICS), new["status"]))
    return out

  # --- the whole solve --------------------------------------------------------------------------

  def solve(self, values: QPValues, *, trace: bool = False) -> dict[str, Expr]:
    """The solve for run-time ``values`` (as ``QPValues.preprocess`` gives them): the unscaled
    solution in PIQP's result layout, the status, the iteration count, and with ``trace`` one row
    per iteration in PIQP's table layout (``TRACE_FIELDS``): the first ``trace_rows`` rows are the
    ones PIQP prints, a row for every pass that reached the convergence test (all ``iter + 1``, but
    for a solve that ran out of iterations or failed to factor, whose last pass does not)."""
    s, t = self.s, self.settings
    scaling = ruiz(s, values, scale_cost=t.preconditioner_scale_cost, max_iter=t.preconditioner_iter, alias_cost=self.backend == "sparse")
    q = scale(s, values, scaling)
    kernels = Kernels(s, self.backend, t.refinement(), name=self.name)
    outer = Iteration(s, q, kernels, t)
    state0 = self.initial_state(outer, self._invalid_bounds(values))
    params, q_sym = self._param_symbols()
    # The KKT data vector is loop-invariant: built once, a param, not concatenated every step.
    data = Expr.sym("q_data", (outer.kkt.data.size,))
    params = [*params, data]
    carry = Expr.sym("ipm_state", (self.layout.size,))
    inner = Iteration(s, q_sym, kernels, t, data=data)
    st = self.layout.unpack(carry)
    body = ConcreteFunction.from_exprs(
      f"{self.name}_step", [carry, *params], [self.layout.pack(self.step(inner, st))], ["state", *(str(e.name) for e in params)], ["next"]
    )
    go = logical_and(equal(st["status"], float(RUNNING)), logical_not(self.converged(st)))
    cond = ConcreteFunction.from_exprs(f"{self.name}_go", [carry, *params], [go], ["state", *(str(e.name) for e in params)], ["go"])
    init = self.layout.pack(state0)
    loop_params = [*self._param_values(q), outer.kkt.data]
    final, _ = while_loop(cond, body, init, max_iter=t.max_iter + 1, params=loop_params)
    fin = self.layout.unpack(final)
    status = where(equal(fin["status"], float(RUNNING)), float(SOLVED), fin["status"])
    out = self._unscaled(fin, q)
    out["status"] = status
    out["iter"] = fin["iter"]
    out["info"] = stack([fin[k] for k in SCALARS])
    if trace:
      carries = _while_node(cond, body, init, t.max_iter + 1, -1, loop_params, False)
      rows = carries.reshape((t.max_iter + 1, self.layout.size))
      offsets = self.layout.offsets()
      cols = [offsets[name][0] for name in TRACE_FIELDS]
      out["trace"] = gather(
        rows.reshape((rows.size,)), (np.arange(t.max_iter + 1)[:, None] * self.layout.size + np.array(cols)[None, :]).reshape(-1)
      ).reshape((t.max_iter + 1, len(TRACE_FIELDS)))
      out["final_row"] = stack([fin[name] for name in TRACE_FIELDS])
      short = logical_or(equal(status, float(MAX_ITER_REACHED)), equal(status, float(NUMERICS)))
      out["trace_rows"] = fin["iter"] + where(short, 0.0, 1.0)
    return out

  def _invalid_bounds(self, values: QPValues) -> Expr | None:
    """Whether a bound the structure declares finite is not (NaN, infinite, or at or beyond
    ``INF``); None when the structure declares none."""
    s = self.s
    parts = [gather(vec, np.flatnonzero(given)) for vec, given in ((values.h_l, s.h_l_given), (values.h_u, s.h_u_given)) if given.any()]
    parts += [vec for vec in (values.x_l, values.x_u) if vec.size]
    if not parts:
      return None
    bounds = concat(parts)
    usable = logical_and(isfinite(bounds), bounds.abs() < INF)
    return where(usable, 0.0, 1.0).sum() > 0.5

  def _unscaled(self, fin: dict[str, Expr], q: ScaledQP) -> dict[str, Expr]:
    """``unscale_results`` and ``restore_dual``: box values move to their variables, absent bounds
    get zero duals and slacks of PIQP_INF."""
    s = self.s
    sc_ = q.scaling
    n, p = s.n, s.p
    c_inv = 1.0 / sc_.c
    dx, deq, din = sc_.delta[:n], sc_.delta[n : n + p], sc_.delta[n + p :]
    lo, up = s.x_l_idx, s.x_u_idx
    z_l, z_u = fin["z_l"] * c_inv * din, fin["z_u"] * c_inv * din
    s_l, s_u = fin["s_l"] / din, fin["s_u"] / din
    out = {
      "x": fin["x"] * dx,
      "y": fin["y"] * c_inv * deq,
      "z_l": z_l,
      "z_u": z_u,
      "s_l": where(equal(z_l, 0.0), INF, s_l),
      "s_u": where(equal(z_u, 0.0), INF, s_u),
    }
    for idx, zname, sname in ((lo, "z_bl", "s_bl"), (up, "z_bu", "s_bu")):
      db = gather(sc_.delta_b, idx)
      z = _scatter(n, idx, fin[zname] * c_inv * db) if idx.size else Expr.const(np.zeros(n))
      sl = fin[sname] / db
      mask = _mask(n, idx)
      out[zname] = z
      out[sname] = (_scatter(n, idx, sl) if idx.size else Expr.const(np.zeros(n))) + INF * (1.0 - mask)
    return out


def _mask(size: int, idx: np.ndarray) -> Expr:
  m = np.zeros(size)
  m[idx] = 1.0
  return Expr.const(m)
