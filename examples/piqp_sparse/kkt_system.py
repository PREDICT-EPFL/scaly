"""The Newton system of one interior-point iteration, PIQP 0.7.0's ``KKTSystem``: slacks and bounds folded into two diagonals, refinement, and recovery of every step.

``KKTSystem`` reduces the system over all of PIQP's variables to the one ``KKT`` factors: the
slacks and box bounds become ``x_reg`` on the variables and ``z_reg`` on the inequality rows,
two-sided rows combine their lower and upper parts, and after the solve the eliminated
duals and slacks are recovered.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import scaly as sc

from .data import Data
from .kkt import KKT, Factor, product
from .settings import Settings
from .utils import Layout, branch, const_bool, const_int, index, inf_norm, loop, std_max, unique_name, zeros
from .variables import Variables


class Scalings(NamedTuple):
  """What ``update_scalings_and_factor`` stores for the solves that follow it.

  ``delta`` is the proximal parameter and ``delta_reg`` the one ``KKT`` factors with, larger by
  the static regularization while refinement is on. ``x_reg`` includes that regularization,
  ``z_reg`` does not, and ``z_reg_kkt`` does.
  """

  rho: sc.Expr
  delta: sc.Expr
  delta_reg: sc.Expr
  refine: sc.Expr
  s_l: sc.Expr
  s_u: sc.Expr
  s_bl: sc.Expr
  s_bu: sc.Expr
  z_l_inv: sc.Expr
  z_u_inv: sc.Expr
  z_bl_inv: sc.Expr
  z_bu_inv: sc.Expr
  x_reg: sc.Expr
  z_reg: sc.Expr
  z_reg_kkt: sc.Expr


def _masked(mask: sc.Expr, value: sc.Expr) -> sc.Expr:
  return sc.where(mask, value, 0.0)


@dataclass(frozen=True, eq=False)
class KKTSystem:
  """PIQP's ``KKTSystem`` over the all-eliminated ``KKT``."""

  kkt: KKT
  settings: Settings

  def P_diag(self, data: Data) -> sc.Expr:
    """``P``'s diagonal, zero where it stores none."""
    s = self.kkt.structure
    return sc.gather(data.P, sc.const(s.P_diagonal, dtype=sc.dtypes.int64), mode="fill", fill_value=0.0) if s.P.nnz else zeros(s.n)

  # eval_P_x
  @cached_property
  def _eval_P_x(self) -> sc.Function:
    return _eval_P_x_function(self.kkt)

  def eval_P_x(self, data: Data, x: sc.Expr) -> sc.Expr:
    """``P x`` from its upper triangle: ``P_utri x`` plus the strictly lower part of ``P_utri^T x``."""
    s = self.kkt.structure
    return self._eval_P_x(data.P, x) if s.P.nnz else zeros(s.n)

  # eval_A_xn_and_AT_xt
  @cached_property
  def _eval_A_xn_and_AT_xt(self) -> sc.Function:
    return _eval_xn_and_t_xt_function(self.kkt, "A")

  def eval_A_xn_and_AT_xt(self, data: Data, xn: sc.Expr, xt: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    """``(A xn, A^T xt)``."""
    s = self.kkt.structure
    return self._eval_A_xn_and_AT_xt(data.A, xn, xt) if s.A.nnz else (zeros(s.p), zeros(s.n))

  # eval_G_xn_and_GT_xt
  @cached_property
  def _eval_G_xn_and_GT_xt(self) -> sc.Function:
    return _eval_xn_and_t_xt_function(self.kkt, "G")

  def eval_G_xn_and_GT_xt(self, data: Data, xn: sc.Expr, xt: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    """``(G xn, G^T xt)``."""
    s = self.kkt.structure
    return self._eval_G_xn_and_GT_xt(data.G, xn, xt) if s.G.nnz else (zeros(s.m), zeros(s.n))

  def update_scalings(self, data: Data, refine: sc.Expr, rho: sc.Expr, delta: sc.Expr, v: Variables) -> Scalings:
    """The first half of PIQP's ``update_scalings_and_factor``: the diagonals ``x_reg`` and ``z_reg``."""
    st = self.settings
    z_l_inv, z_u_inv = 1.0 / v.z_l, 1.0 / v.z_u
    z_bl_inv, z_bu_inv = 1.0 / v.z_bl, 1.0 / v.z_bu
    x_b = data.x_b_scaling * data.x_b_scaling
    x_reg = rho + _masked(data.x_l_mask, x_b / (z_bl_inv * v.s_bl + delta))
    x_reg = x_reg + _masked(data.x_u_mask, x_b / (z_bu_inv * v.s_bu + delta))
    z_reg = _masked(data.h_l_mask, 1.0 / (z_l_inv * v.s_l + delta))
    z_reg = 1.0 / (z_reg + _masked(data.h_u_mask, 1.0 / (z_u_inv * v.s_u + delta)))
    max_diag = std_max(inf_norm(self.P_diag(data) + x_reg), inf_norm(z_reg))
    reg = sc.where(refine, st.iterative_refinement_static_regularization_eps + st.iterative_refinement_static_regularization_rel * max_diag, 0.0)
    return Scalings(
      rho=rho,
      delta=delta,
      delta_reg=sc.where(refine, delta + reg, delta),
      refine=refine,
      s_l=v.s_l,
      s_u=v.s_u,
      s_bl=v.s_bl,
      s_bu=v.s_bu,
      z_l_inv=z_l_inv,
      z_u_inv=z_u_inv,
      z_bl_inv=z_bl_inv,
      z_bu_inv=z_bu_inv,
      x_reg=sc.where(refine, x_reg + reg, x_reg),
      z_reg=z_reg,
      z_reg_kkt=sc.where(refine, z_reg + reg, z_reg),
    )

  def update_scalings_and_factor(
    self, data: Data, AtA: sc.Expr, refine: sc.Expr, rho: sc.Expr, delta: sc.Expr, v: Variables
  ) -> tuple[Scalings, Factor]:
    scalings = self.update_scalings(data, refine, rho, delta, v)
    return scalings, self.kkt.update_scalings_and_factor(data, AtA, scalings.delta_reg, scalings.x_reg, scalings.z_reg_kkt)

  def mul_condensed_kkt(
    self, data: Data, sc_: Scalings, x: sc.Expr, y: sc.Expr, z: sc.Expr
  ) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    """The reduced system's product, with the unregularized ``delta`` and ``z_reg`` that refinement corrects towards."""
    rx = self.eval_P_x(data, x)
    rx = rx + sc_.x_reg * x
    ry, work_x = self.eval_A_xn_and_AT_xt(data, x, y)
    rx = rx + work_x
    ry = ry - sc_.delta * y
    rz, work_x = self.eval_G_xn_and_GT_xt(data, x, z)
    rx = rx + work_x
    rz = rz - sc_.z_reg * z
    return rx, ry, rz

  def get_refine_error(
    self, data: Data, sc_: Scalings, lhs: tuple[sc.Expr, sc.Expr, sc.Expr], rhs: tuple[sc.Expr, sc.Expr, sc.Expr]
  ) -> tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr]:
    product = self.mul_condensed_kkt(data, sc_, *lhs)
    err_x, err_y, err_z = rhs[0] - product[0], rhs[1] - product[1], rhs[2] - product[2]
    return err_x, err_y, err_z, _inf_norm3(err_x, err_y, err_z)

  def iterative_refinement(
    self, data: Data, sc_: Scalings, factor: Factor, lhs: tuple[sc.Expr, sc.Expr, sc.Expr], rhs: tuple[sc.Expr, sc.Expr, sc.Expr]
  ) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    """PIQP's iterative refinement: correct the solution while each correction shrinks the residual at least ``min_improvement_rate`` times.

    PIQP gives up on a non-finite residual without recovering the duals, a result its caller
    ignores. Here the last finite solution is kept and the duals are recovered from it.
    """
    st = self.settings
    tolerance = st.iterative_refinement_eps_abs + st.iterative_refinement_eps_rel * _inf_norm3(*rhs)
    err_x, err_y, err_z, error = self.get_refine_error(data, sc_, lhs, rhs)
    state = {"x": lhs[0], "y": lhs[1], "z": lhs[2], "err_x": err_x, "err_y": err_y, "err_z": err_z, "error": error}
    state |= {"i": const_int(0), "done": ~error.isfinite()}
    # A loop's step sees only its carry and its parameters, so the system travels as parameters.
    params = {"rx": rhs[0], "ry": rhs[1], "rz": rhs[2], "tolerance": tolerance, "L": factor.L, "d": factor.d}
    params |= {f"data_{k}": v for k, v in data._asdict().items()} | {f"sc_{k}": v for k, v in sc_._asdict().items()}

    def system(q: dict[str, sc.Expr]) -> tuple[Data, Scalings, Factor, tuple[sc.Expr, sc.Expr, sc.Expr]]:
      d = Data(**{k[5:]: v for k, v in q.items() if k.startswith("data_")})
      s = Scalings(**{k[3:]: v for k, v in q.items() if k.startswith("sc_")})
      return d, s, Factor(q["L"], q["d"], const_bool(True)), (q["rx"], q["ry"], q["rz"])

    def keep_going(s: dict[str, sc.Expr], q: dict[str, sc.Expr]) -> sc.Expr:
      return ~s["done"] & (s["i"] < st.iterative_refinement_max_iter) & ~(s["error"] <= q["tolerance"])

    def step(s: dict[str, sc.Expr], q: dict[str, sc.Expr]) -> dict[str, sc.Expr]:
      data, sc_, factor, rhs = system(q)
      correction = self.kkt.solve(data, factor, sc_.delta_reg, sc_.z_reg_kkt, s["err_x"], s["err_y"], s["err_z"])
      refined = (correction[0] + s["x"], correction[1] + s["y"], correction[2] + s["z"])
      err_x, err_y, err_z, error = self.get_refine_error(data, sc_, refined, rhs)
      rate = s["error"] / error
      finite = error.isfinite()
      accept = finite & ((rate >= st.iterative_refinement_min_improvement_rate) | (rate > 1.0))
      out = dict(s)
      for name, new in zip(("x", "y", "z"), refined, strict=True):
        out[name] = sc.where(accept, new, s[name])
      out |= {"err_x": err_x, "err_y": err_y, "err_z": err_z, "error": error, "i": s["i"] + 1}
      out["done"] = ~finite | (rate < st.iterative_refinement_min_improvement_rate)
      return out

    if st.iterative_refinement_max_iter == 0:
      return lhs
    final = loop(keep_going, step, state, params, st.iterative_refinement_max_iter, "refine")
    return final["x"], final["y"], final["z"]

  def solve(self, data: Data, AtA: sc.Expr, sc_: Scalings, factor: Factor, rhs: Variables) -> Variables:
    """Solve the Newton system for the step of every variable, PIQP's ``KKTSystem::solve``."""
    # rhs_z_bar
    w_l = 1.0 / (sc_.z_l_inv * sc_.s_l + sc_.delta)
    w_u = 1.0 / (sc_.z_u_inv * sc_.s_u + sc_.delta)
    rz_l_bar = rhs.z_l - sc_.z_l_inv * rhs.s_l
    rz_u_bar = rhs.z_u - sc_.z_u_inv * rhs.s_u
    rhs_z_bar = (-_masked(data.h_l_mask, w_l * rz_l_bar) + _masked(data.h_u_mask, w_u * rz_u_bar)) * sc_.z_reg

    # rhs_x_bar
    w_bl = sc_.s_bl * sc_.z_bl_inv + sc_.delta
    w_bu = sc_.s_bu * sc_.z_bu_inv + sc_.delta
    rhs_x_bar = rhs.x - _masked(data.x_l_mask, data.x_b_scaling * (rhs.z_bl - sc_.z_bl_inv * rhs.s_bl) / w_bl)
    rhs_x_bar = rhs_x_bar + _masked(data.x_u_mask, data.x_b_scaling * (rhs.z_bu - sc_.z_bu_inv * rhs.s_bu) / w_bu)

    # kkt_solver->solve
    reduced = (rhs_x_bar, rhs.y, rhs_z_bar)
    lhs = self.kkt.solve(data, factor, sc_.delta_reg, sc_.z_reg_kkt, *reduced)

    # iterative_refinement
    lhs = self.refine_if_enabled(data, AtA, sc_, factor, lhs, reduced)
    dx, dy, dz = lhs

    # dual_recovery
    both = data.h_l_mask & data.h_u_mask
    r_sum = w_l * w_u * (rz_l_bar + rz_u_bar)
    dz_l = sc.where(both, -sc_.z_reg * (r_sum + w_l * dz), sc.where(data.h_l_mask, -dz, 0.0))
    dz_u = sc.where(both, -sc_.z_reg * (r_sum - w_u * dz), sc.where(data.h_u_mask, dz, 0.0))
    ds_l = _masked(data.h_l_mask, sc_.z_l_inv * (rhs.s_l - sc_.s_l * dz_l))
    ds_u = _masked(data.h_u_mask, sc_.z_u_inv * (rhs.s_u - sc_.s_u * dz_u))

    # box_dual_recovery
    dz_bl = _masked(data.x_l_mask, (-data.x_b_scaling * dx - rhs.z_bl + sc_.z_bl_inv * rhs.s_bl) / w_bl)
    dz_bu = _masked(data.x_u_mask, (data.x_b_scaling * dx - rhs.z_bu + sc_.z_bu_inv * rhs.s_bu) / w_bu)
    ds_bl = _masked(data.x_l_mask, sc_.z_bl_inv * (rhs.s_bl - sc_.s_bl * dz_bl))
    ds_bu = _masked(data.x_u_mask, sc_.z_bu_inv * (rhs.s_bu - sc_.s_bu * dz_bu))
    return Variables(dx, dy, dz_l, dz_u, dz_bl, dz_bu, ds_l, ds_u, ds_bl, ds_bu)

  def refine_if_enabled(
    self,
    data: Data,
    AtA: sc.Expr,
    sc_: Scalings,
    factor: Factor,
    lhs: tuple[sc.Expr, sc.Expr, sc.Expr],
    rhs: tuple[sc.Expr, sc.Expr, sc.Expr],
  ) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    """Refinement runs only while it is enabled, after a failed factorization or with ``iterative_refinement_always_enabled``."""
    if self.settings.iterative_refinement_max_iter == 0:
      return lhs
    operands = {"lx": lhs[0], "ly": lhs[1], "lz": lhs[2], "rx": rhs[0], "ry": rhs[1], "rz": rhs[2], "AtA": AtA, "L": factor.L, "d": factor.d}
    operands |= {f"data_{k}": v for k, v in data._asdict().items()} | {f"sc_{k}": v for k, v in sc_._asdict().items()}
    outputs = Layout.of({"lx": lhs[0], "ly": lhs[1], "lz": lhs[2]})

    def unpack(o: dict[str, sc.Expr]) -> tuple[Data, Scalings, Factor]:
      d = Data(**{k[5:]: v for k, v in o.items() if k.startswith("data_")})
      s = Scalings(**{k[3:]: v for k, v in o.items() if k.startswith("sc_")})
      return d, s, Factor(o["L"], o["d"], const_bool(True))

    def keep(o: dict[str, sc.Expr]) -> dict[str, sc.Expr]:
      return {"lx": o["lx"], "ly": o["ly"], "lz": o["lz"]}

    def refined(o: dict[str, sc.Expr]) -> dict[str, sc.Expr]:
      d, s, f = unpack(o)
      x, y, z = self.iterative_refinement(d, s, f, (o["lx"], o["ly"], o["lz"]), (o["rx"], o["ry"], o["rz"]))
      return {"lx": x, "ly": y, "lz": z}

    out = branch(sc_.refine, operands, keep, refined, outputs, "refinement")
    return out["lx"], out["ly"], out["lz"]


def _inf_norm3(x: sc.Expr, y: sc.Expr, z: sc.Expr) -> sc.Expr:
  return std_max(std_max(inf_norm(x), inf_norm(y)), inf_norm(z))


# Products with the data: PIQP's eval_P_x, eval_A_xn_and_AT_xt and eval_G_xn_and_GT_xt.
#
# Each is one sc.function, so the generated C holds one loop per product and one copy of its
# index tables, called from update_residuals_nr, check_infeasibility_certificates and
# mul_condensed_kkt. PIQP's alpha arguments are left out: the callers negate a result where
# PIQP passes alpha = -1, which gives the same values.


def _eval_P_x_function(kkt: KKT) -> sc.Function:
  s = kkt.structure
  strict, above = s.P_strict

  @sc.function(sc.arg("P", s.P.nnz), sc.arg("x", s.n), outputs=sc.arg("z", s.n), name=unique_name("eval_P_x"))
  def eval_P_x(P: sc.Expr, x: sc.Expr) -> sc.Expr:
    z = product(P, s.P, x)
    if len(above):
      # the strictly lower part of P_utri^T
      z = z + product(sc.gather(P, index(above)), strict, x, transpose=True)
    return z

  return eval_P_x


def _eval_xn_and_t_xt_function(kkt: KKT, which: str) -> sc.Function:
  s = kkt.structure
  pattern = getattr(s, which)
  rows, cols = pattern.shape

  @sc.function(
    sc.arg(which, pattern.nnz),
    sc.arg("xn", cols),
    sc.arg("xt", rows),
    outputs=sc.group(sc.arg("zn", rows), sc.arg("zt", cols)),
    name=unique_name(f"eval_{which}_xn_and_{which}T_xt"),
  )
  def eval_xn_and_t_xt(M: sc.Expr, xn: sc.Expr, xt: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    zn = product(M, pattern, xn)
    zt = product(M, pattern, xt, transpose=True)
    return zn, zt

  return eval_xn_and_t_xt
