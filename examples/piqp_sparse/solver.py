"""The solver, PIQP 0.7.0's ``SparseSolver``: ``setup``, ``update`` and ``solve`` in Python, and the proximal interior-point method of ``solver.tpp`` as generated code.

``build_solver`` returns the three Scaly Functions generated for each structure and settings.
``setup`` is PIQP's setup: the data's infinite bounds, its Ruiz equilibration and the product
``A^T A`` the factorizations reuse. ``solve`` is PIQP's ``solve``: the whole iteration, from the
initial point to the unscaled result, as one call. ``update`` is PIQP's update when the scaling is
reused: the setup without the equilibration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from scipy import sparse

import scaly as sc

from .data import Data, Structure, data_types, make_data
from .kkt import KKT, Factor
from .kkt_system import KKTSystem, Scalings
from .preconditioner import Scaling, reuse_scaling, scale_data, scaling_types
from .results import FROM_SCALY_STATUS, INFO_FIELDS, INTEGER_INFO, TO_SCALY_STATUS, Info, Result, Status
from .settings import Settings
from .utils import (
  Kernel,
  F64,
  I32,
  INF,
  Layout,
  any_true,
  branch,
  const_bool,
  const_int,
  inf_norm,
  loop,
  masked_inf_norm,
  masked_max,
  masked_min,
  std_max,
  std_min,
  to_float,
  zeros,
)
from .variables import Variables

_EPS = float(np.finfo(np.float64).eps)
RAW_INPUTS = ("P", "c", "A", "b", "G", "h_l", "h_u", "x_l", "x_u")
_RESIDUALS = ("x", "y", "z_l", "z_u", "z_bl", "z_bu")
_BOUNDS = ("l", "u", "bl", "bu")
_COUNTERS = ("iter", "factor_retires", "no_primal_update", "no_dual_update")
_CERTIFICATES = ("cert_c_x", "cert_dual_lin", "prox_c_x", "prox_dual_lin", "delta_c_x", "delta_dual_lin")


def _scalar(value: float) -> sc.Expr:
  return sc.const(float(value))


# The state of the iteration, PIQP's m_result, res_nr, prox_vars, delta_vars and m_result.info, as
# named Scaly values. Every function below reads a state and returns the next one.
State = dict[str, sc.Expr]


@dataclass(frozen=True, eq=False)
class _Method:
  """One generated solver: the iteration of ``solver.tpp`` for one structure and settings."""

  structure: Structure
  settings: Settings
  name: str = "piqp_sparse"

  @cached_property
  def kkt(self) -> KKT:
    return KKT(self.structure)

  @cached_property
  def kkt_system(self) -> KKTSystem:
    return KKTSystem(self.kkt, self.settings)

  @cached_property
  def has_inequalities(self) -> bool | None:
    """PIQP's ``m + n_x_l + n_x_u > 0``: true when ``G`` has rows, otherwise decided by the bounds at run time."""
    return True if self.structure.m else None

  # Inputs and their names.

  def params(self, data: Data, scaling: Scaling, AtA: sc.Expr) -> State:
    return {f"data_{k}": v for k, v in data._asdict().items()} | {f"scaling_{k}": v for k, v in scaling._asdict().items()} | {"AtA": AtA}

  @staticmethod
  def data(p: State) -> Data:
    return Data(**{k[5:]: v for k, v in p.items() if k.startswith("data_")})

  @staticmethod
  def scaling(p: State) -> Scaling:
    return Scaling(**{k[8:]: v for k, v in p.items() if k.startswith("scaling_")})

  @staticmethod
  def variables(st: State) -> Variables:
    return Variables(*(st[f] for f in Variables._fields))

  # PIQP's calculate_mu and calculate_step.

  def inequality_count(self, data: Data) -> sc.Expr:
    return sum((to_float(mask).sum() for mask in (data.h_l_mask, data.h_u_mask, data.x_l_mask, data.x_u_mask) if mask.size), _scalar(0.0))

  def calculate_mu(self, st: State, data: Data) -> sc.Expr:
    complementarity = sc.dot(st["s_l"], st["z_l"]) + sc.dot(st["s_u"], st["z_u"]) + sc.dot(st["s_bl"], st["z_bl"]) + sc.dot(st["s_bu"], st["z_bu"])
    return complementarity / self.inequality_count(data)

  def calculate_step(self, st: State, step: Variables) -> tuple[sc.Expr, sc.Expr]:
    """The largest steps in ``[0, 1]`` that keep the slacks and the duals nonnegative."""

    def largest(value: sc.Expr, direction: sc.Expr) -> sc.Expr:
      if not value.size:
        return _scalar(1.0)
      return sc.minimum(1.0, sc.where(direction < 0.0, -value / direction, np.inf).min())

    alpha_s = sc.minimum(sc.minimum(largest(st["s_l"], step.s_l), largest(st["s_u"], step.s_u)), sc.minimum(largest(st["s_bl"], step.s_bl), largest(st["s_bu"], step.s_bu)))
    alpha_z = sc.minimum(sc.minimum(largest(st["z_l"], step.z_l), largest(st["z_u"], step.z_u)), sc.minimum(largest(st["z_bl"], step.z_bl), largest(st["z_bu"], step.z_bu)))
    return alpha_s, alpha_z

  # PIQP's residuals: update_residuals_nr, update_residuals_r and the norms they report.

  def primal_res(self, r: State, data: Data, sc_: Scaling) -> sc.Expr:
    """PIQP's ``primal_res_nr`` and ``primal_res_r``. The box terms enter signed, as PIQP's do, so a negative one never counts."""
    norm = std_max(std_max(inf_norm(sc_.unscale_primal_res_eq(r["y"])), inf_norm(sc_.unscale_primal_res_ineq(r["z_l"]))), inf_norm(sc_.unscale_primal_res_ineq(r["z_u"])))
    norm = std_max(norm, masked_max(sc_.unscale_primal_res_b(r["z_bl"]), data.x_l_mask))
    return std_max(norm, masked_max(sc_.unscale_primal_res_b(r["z_bu"]), data.x_u_mask))

  def update_residuals_nr(self, st: State, data: Data, sc_: Scaling) -> State:
    """The unregularized residuals, the objectives and the duality gap, all reported unscaled."""
    kkt, st = self.kkt_system, dict(st)
    x, y = st["x"], st["y"]
    # we calculate these term here first to be able to reuse temporary vectors
    # res_nr.y = -A * x
    # work_x = A^T * y
    Ax, work_x = kkt.eval_A_xn_and_AT_xt(data, x, y)
    res_y = -Ax
    # res_nr.z_u = -G * x
    # res_nr.z_l = G * x
    # work_x += G^T * (z_u - z_l)
    Gx, work_x_2 = kkt.eval_G_xn_and_GT_xt(data, x, st["z_u"] - st["z_l"])
    work_x = work_x + work_x_2

    # first part of dual residual and infeasibility calculation (used in cost calculation)
    res_x = -kkt.eval_P_x(data, x)
    dual_rel_norm = inf_norm(sc_.unscale_dual_res(res_x))

    xPx = -sc.dot(x, res_x)
    primal_obj, dual_obj = 0.5 * xPx, -0.5 * xPx
    gap_norm = sc_.unscale_cost(xPx.abs())
    c_x = sc.dot(data.c, x)
    primal_obj = primal_obj + c_x
    gap_norm = std_max(gap_norm, sc_.unscale_cost(c_x.abs()))
    dual_lin = _scalar(0.0)
    terms = (
      sc.dot(data.b, y),
      -sc.dot(sc.where(data.h_l_mask, data.h_l, 0.0), st["z_l"]),
      sc.dot(sc.where(data.h_u_mask, data.h_u, 0.0), st["z_u"]),
      -sc.dot(data.x_l, st["z_bl"]),
      sc.dot(data.x_u, st["z_bu"]),
    )
    for i, term in enumerate(terms):
      dual_obj = dual_obj - term
      dual_lin = term if i == 0 else dual_lin + term
      gap_norm = std_max(gap_norm, sc_.unscale_cost(term.abs()))
    gap = (primal_obj - dual_obj).abs()
    st["primal_obj"] = sc_.unscale_cost(primal_obj)
    st["dual_obj"] = sc_.unscale_cost(dual_obj)
    st["duality_gap"] = sc_.unscale_cost(gap)
    st["duality_gap_rel"] = st["duality_gap"] / std_max(1.0, gap_norm)
    st["cert_c_x"], st["cert_dual_lin"] = c_x, dual_lin

    res_x = res_x - data.c
    dual_rel_norm = std_max(dual_rel_norm, inf_norm(sc_.unscale_dual_res(data.c)))
    work_x = work_x - sc.where(data.x_l_mask, data.x_b_scaling * st["z_bl"], 0.0)
    work_x = work_x + sc.where(data.x_u_mask, data.x_b_scaling * st["z_bu"], 0.0)
    dual_rel_norm = std_max(dual_rel_norm, inf_norm(sc_.unscale_dual_res(work_x)))
    res_x = res_x - work_x

    primal_rel_norm = inf_norm(sc_.unscale_primal_res_eq(res_y))
    res_y = res_y + data.b
    primal_rel_norm = std_max(primal_rel_norm, inf_norm(sc_.unscale_primal_res_eq(data.b)))

    # The inequality and box rows enter the relative norm signed, as in PIQP.
    def signed(value: sc.Expr, mask: sc.Expr, scale: Callable[[sc.Expr], sc.Expr]) -> sc.Expr:
      return masked_max(scale(value), mask)

    ineq, box = sc_.unscale_primal_res_ineq, sc_.unscale_primal_res_b
    for value in (Gx, data.h_l, st["s_l"]):
      primal_rel_norm = std_max(primal_rel_norm, signed(value, data.h_l_mask, ineq))
    for value in (-Gx, data.h_u, st["s_u"]):
      primal_rel_norm = std_max(primal_rel_norm, signed(value, data.h_u_mask, ineq))
    res_z_l = sc.where(data.h_l_mask, Gx + (-data.h_l - st["s_l"]), 0.0)
    res_z_u = sc.where(data.h_u_mask, -Gx + (data.h_u - st["s_u"]), 0.0)
    x_b = data.x_b_scaling * x
    for value in (x_b, data.x_l, st["s_bl"]):
      primal_rel_norm = std_max(primal_rel_norm, signed(value, data.x_l_mask, box))
    for value in (-x_b, data.x_u, st["s_bu"]):
      primal_rel_norm = std_max(primal_rel_norm, signed(value, data.x_u_mask, box))
    res_z_bl = sc.where(data.x_l_mask, x_b + (-data.x_l - st["s_bl"]), 0.0)
    res_z_bu = sc.where(data.x_u_mask, -x_b + (data.x_u - st["s_bu"]), 0.0)

    res = {"x": res_x, "y": res_y, "z_l": res_z_l, "z_u": res_z_u, "z_bl": res_z_bl, "z_bu": res_z_bu}
    st |= {f"res_nr_{k}": v for k, v in res.items()}
    st["prev_primal_res"], st["prev_dual_res"] = st["primal_res"], st["dual_res"]
    st["primal_res"] = self.primal_res(res, data, sc_)
    st["primal_res_rel"] = st["primal_res"] / std_max(1.0, primal_rel_norm)
    st["dual_res"] = inf_norm(sc_.unscale_dual_res(res_x))
    st["dual_res_rel"] = st["dual_res"] / std_max(1.0, dual_rel_norm)
    return st

  def update_residuals_r(self, st: State, data: Data, sc_: Scaling) -> tuple[State, State]:
    """The residuals of the proximal subproblem, and the proximal infeasibility measures; returns the state and the residuals."""
    st = dict(st)
    rho, delta = st["rho"], st["delta"]
    res = {"x": st["res_nr_x"] - rho * (st["x"] - st["prox_x"])}
    for k in ("y", "z_l", "z_u", "z_bl", "z_bu"):
      res[k] = st[f"res_nr_{k}"] - delta * (st[f"prox_{k}"] - st[k])
    primal_scaling = sc.where(st["primal_res_rel"] > 0.0, st["primal_res"] / st["primal_res_rel"], 1.0)
    dual_scaling = sc.where(st["dual_res_rel"] > 0.0, st["dual_res"] / st["dual_res_rel"], 1.0)
    st["primal_res_reg"] = self.primal_res(res, data, sc_)
    st["primal_res_reg_rel"] = st["primal_res_reg"] / primal_scaling
    st["dual_res_reg"] = inf_norm(sc_.unscale_dual_res(res["x"]))
    st["dual_res_reg_rel"] = st["dual_res_reg"] / dual_scaling

    prox_inf = std_max(inf_norm(sc_.unscale_dual_eq(st["prox_y"] - st["y"])), inf_norm(sc_.unscale_dual_ineq(st["prox_z_l"] - st["z_l"])))
    prox_inf = std_max(prox_inf, inf_norm(sc_.unscale_dual_ineq(st["prox_z_u"] - st["z_u"])))
    prox_inf = std_max(prox_inf, masked_max(sc_.unscale_dual_b(st["prox_z_bl"] - st["z_bl"]), data.x_l_mask))
    prox_inf = std_max(prox_inf, masked_max(sc_.unscale_dual_b(st["prox_z_bu"] - st["z_bu"]), data.x_u_mask))
    st["primal_prox_inf"] = prox_inf * delta
    st["dual_prox_inf"] = inf_norm(sc_.unscale_primal(st["x"] - st["prox_x"])) * rho
    return st, res

  @staticmethod
  def update_primal_prox_center(st: State, move: sc.Expr) -> State:
    st = dict(st)
    st["prox_x"] = sc.where(move, st["x"], st["prox_x"])
    st["prox_c_x"] = sc.where(move, st["cert_c_x"], st["prox_c_x"])
    return st

  @staticmethod
  def update_dual_prox_center(st: State, move: sc.Expr) -> State:
    st = dict(st)
    for k in ("y", "z_l", "z_u", "z_bl", "z_bu"):
      st[f"prox_{k}"] = sc.where(move, st[k], st[f"prox_{k}"])
    st["prox_dual_lin"] = sc.where(move, st["cert_dual_lin"], st["prox_dual_lin"])
    return st

  # PIQP's check_infeasibility_certificates, store_primal_infeasibility_certificate and store_dual_infeasibility_certificate.

  def check_infeasibility_certificates(self, st: State, data: Data, sc_: Scaling) -> State:
    """Test the proximal step as a Farkas certificate of primal infeasibility or as an unbounded direction, and store the one found."""
    kkt, settings, st = self.kkt_system, self.settings, dict(st)
    d = {k: st[f"delta_{k}"] for k in _RESIDUALS}
    dz_l, dz_u = sc_.unscale_dual_ineq(d["z_l"]), sc_.unscale_dual_ineq(d["z_u"])
    dz_bl, dz_bu = sc_.unscale_dual_b(d["z_bl"]), sc_.unscale_dual_b(d["z_bu"])
    norm_dual = std_max(inf_norm(sc_.unscale_dual_eq(d["y"])), masked_inf_norm(dz_l, data.h_l_mask))
    norm_dual = std_max(std_max(norm_dual, masked_inf_norm(dz_u, data.h_u_mask)), masked_inf_norm(dz_bl, data.x_l_mask))
    norm_dual = std_max(norm_dual, masked_inf_norm(dz_bu, data.x_u_mask))
    eps_p = settings.eps_primal_inf * norm_dual
    primal = (st["primal_res"] >= 0.95 * st["prev_primal_res"]) & (norm_dual > 0.0) & (sc_.unscale_cost(st["delta_dual_lin"]) < -eps_p)
    for value, mask in ((dz_l, data.h_l_mask), (dz_u, data.h_u_mask), (dz_bl, data.x_l_mask), (dz_bu, data.x_u_mask)):
      primal = primal & ~(masked_min(value, mask) < -eps_p)

    dx = sc_.unscale_primal(d["x"])
    norm_dx = inf_norm(dx)
    eps_d = settings.eps_dual_inf * norm_dx
    dual = (st["dual_res"] >= 0.95 * st["prev_dual_res"]) & (norm_dx > 0.0) & (sc_.unscale_cost(st["delta_c_x"]) < -eps_d)
    dual = dual & ~(masked_min(dx, data.x_l_mask) < -eps_d) & ~(masked_max(dx, data.x_u_mask) > eps_d)

    dz = sc.where(data.h_u_mask, d["z_u"], 0.0) - sc.where(data.h_l_mask, d["z_l"], 0.0)
    # cert_work_p = A dx, cert_work_n_1 = A^T dy
    Adx, ATdy = kkt.eval_A_xn_and_AT_xt(data, d["x"], d["y"])
    # cert_work_m_2 = G dx, cert_work_n_2 = G^T (dz_u - dz_l)
    Gdx, GTdz = kkt.eval_G_xn_and_GT_xt(data, d["x"], dz)
    farkas = ATdy + GTdz
    farkas = farkas - sc.where(data.x_l_mask, data.x_b_scaling * d["z_bl"], 0.0) + sc.where(data.x_u_mask, data.x_b_scaling * d["z_bu"], 0.0)
    primal = primal & (inf_norm(sc_.unscale_dual_res(farkas)) <= eps_p)

    Gdx = sc_.unscale_primal_res_ineq(Gdx)
    dual = dual & ~(inf_norm(sc_.unscale_primal_res_eq(Adx)) > eps_d)
    dual = dual & ~(masked_min(Gdx, data.h_l_mask) < -eps_d) & ~(masked_max(Gdx, data.h_u_mask) > eps_d)
    dual = dual & ~primal & (inf_norm(sc_.unscale_dual_res(kkt.eval_P_x(data, d["x"]))) <= eps_d)

    for k in ("y", "z_bl", "z_bu"):
      st[k] = sc.where(primal, d[k] / norm_dual, st[k])
    st["z_l"] = sc.where(primal, sc.where(data.h_l_mask, d["z_l"] / norm_dual, 0.0), st["z_l"])
    st["z_u"] = sc.where(primal, sc.where(data.h_u_mask, d["z_u"] / norm_dual, 0.0), st["z_u"])
    st["x"] = sc.where(dual, d["x"] / norm_dx, st["x"])
    status = sc.where(primal, int(Status.PRIMAL_INFEASIBLE), sc.where(dual, int(Status.DUAL_INFEASIBLE), int(Status.UNSOLVED)))
    st["status"] = sc.cast(status, I32)
    return st

  # Factorization with PIQP's retries: refinement first, then larger regularization.

  # The factorization and the solve run at several places of an iteration: each is one Kernel,
  # one function of the generated C that every place calls.

  _FACTOR_STATE = ("refine", "rho", "delta", "factor_retires", "reg_limit", *Variables._fields)

  @cached_property
  def _factor_kernel(self) -> Kernel:
    def run(inputs: State) -> State:
      st = {k: inputs[f"st_{k}"] for k in self._FACTOR_STATE}
      p = {k: v for k, v in inputs.items() if not k.startswith("st_")}
      st, scalings, factor, failed, changed = self._factor_kkt(st, p)
      out = {f"st_{k}": st[k] for k in self._FACTOR_STATE[:5]}
      out |= {f"scalings_{k}": v for k, v in scalings._asdict().items()} | {f"factor_{k}": v for k, v in factor._asdict().items()}
      return out | {"failed": failed, "changed": changed}

    return Kernel(run, "factor_kkt")

  def factor_kkt(self, st: State, p: State) -> tuple[State, Scalings, Factor, sc.Expr, sc.Expr]:
    """PIQP's loop around ``update_scalings_and_factor``; returns the state, the scalings and factor that succeeded, whether all attempts failed, and whether the regularization grew."""
    out = self._factor_kernel({f"st_{k}": st[k] for k in self._FACTOR_STATE} | dict(p))
    st = dict(st) | {k: out[f"st_{k}"] for k in self._FACTOR_STATE[:5]}
    scalings = Scalings(**{k: out[f"scalings_{k}"] for k in Scalings._fields})
    factor = Factor(**{k: out[f"factor_{k}"] for k in Factor._fields})
    return st, scalings, factor, out["failed"], out["changed"]

  @cached_property
  def _solve_kernel(self) -> Kernel:
    def run(inputs: State) -> State:
      scalings = Scalings(**{k: inputs[f"scalings_{k}"] for k in Scalings._fields})
      factor = Factor(**{k: inputs[f"factor_{k}"] for k in Factor._fields})
      rhs = Variables(*(inputs[f"rhs_{k}"] for k in Variables._fields))
      return self.kkt_system.solve(self.data(inputs), inputs["AtA"], scalings, factor, rhs)._asdict()

    return Kernel(run, "kkt_solve")

  def kkt_solve(self, p: State, scalings: Scalings, factor: Factor, rhs: Variables) -> Variables:
    """PIQP's ``KKTSystem::solve``, with refinement when it is on, as one Kernel."""
    inputs = {k: v for k, v in p.items() if k.startswith("data_")} | {"AtA": p["AtA"]}
    inputs |= {f"scalings_{k}": v for k, v in scalings._asdict().items()} | {f"factor_{k}": v for k, v in factor._asdict().items()}
    inputs |= {f"rhs_{k}": v for k, v in rhs._asdict().items()}
    return Variables(**self._solve_kernel(inputs))

  def _factor_kkt(self, st: State, p: State) -> tuple[State, Scalings, Factor, sc.Expr, sc.Expr]:
    settings, data = self.settings, self.data(p)
    v = self.variables(st)
    # The factor's pattern, for the loop's initial carry; the traced factorization is unused.
    L_pattern = self.kkt.update_scalings_and_factor(data, p["AtA"], _scalar(1.0), zeros(self.structure.n), _ones_like(v.s_l)).L.type.pattern
    assert L_pattern is not None
    attempt = {
      "refine": st["refine"],
      "rho": st["rho"],
      "delta": st["delta"],
      "factor_retires": st["factor_retires"],
      "reg_limit": st["reg_limit"],
      "changed": const_bool(False),
      "ok": const_bool(False),
      "failed": const_bool(False),
      "L": sc.pack(zeros(L_pattern.nnz), L_pattern),
      "d": zeros(self.structure.n),
    }
    params = dict(p) | {f"var_{k}": st[k] for k in ("z_l", "z_u", "z_bl", "z_bu", "s_l", "s_u", "s_bl", "s_bu")}

    def keep_going(a: State, _: State) -> sc.Expr:
      return ~a["ok"] & ~a["failed"]

    def attempt_factor(a: State, q: State) -> State:
      variables = self._factor_variables(q)
      _, factor = self.kkt_system.update_scalings_and_factor(self.data(q), q["AtA"], a["refine"], a["rho"], a["delta"], variables)
      failed_now = ~factor.ok
      retry = failed_now & a["refine"] & (a["factor_retires"] < settings.max_factor_retires)
      out = dict(a)
      out["refine"] = a["refine"] | failed_now
      out["rho"] = sc.where(retry, a["rho"] * 100.0, a["rho"])
      out["delta"] = sc.where(retry, a["delta"] * 100.0, a["delta"])
      out["factor_retires"] = a["factor_retires"] + sc.cast(retry, I32)
      out["reg_limit"] = sc.where(retry, std_min(10.0 * a["reg_limit"], settings.eps_abs), a["reg_limit"])
      out["changed"] = a["changed"] | retry
      out["failed"] = failed_now & a["refine"] & ~retry
      out["ok"], out["L"], out["d"] = factor.ok, factor.L, factor.d
      return out

    final = loop(keep_going, attempt_factor, attempt, params, settings.max_factor_retires + 2, "factor_kkt")
    st = dict(st)
    for k in ("refine", "rho", "delta", "factor_retires", "reg_limit"):
      st[k] = final[k]
    st["factor_retires"] = sc.where(final["failed"], st["factor_retires"], const_int(0))
    scalings = self.kkt_system.update_scalings(data, st["refine"], st["rho"], st["delta"], v)
    return st, scalings, Factor(final["L"], final["d"], final["ok"]), final["failed"], final["changed"]

  def _factor_variables(self, q: State) -> Variables:
    n, p = self.structure.n, self.structure.p
    return Variables(zeros(n), zeros(p), *(q[f"var_{k}"] for k in ("z_l", "z_u", "z_bl", "z_bu", "s_l", "s_u", "s_bl", "s_bu")))

  # The iteration.

  def initial_state(self) -> State:
    """``solve_impl`` up to its loop: every slack and dual of a finite bound at one, ``rho`` and ``delta`` at their initial values."""
    s, settings = self.structure, self.settings
    st: State = {}
    for k, size in zip(Variables._fields, (s.n, s.p, s.m, s.m, s.n, s.n, s.m, s.m, s.n, s.n), strict=True):
      st[k] = zeros(size)
      st[f"prox_{k}"] = zeros(size)
      st[f"delta_{k}"] = zeros(size)
      st[f"res_nr_{k}"] = zeros(size)
      st[f"res_{k}"] = zeros(size)
    for k in Variables._fields[6:]:
      del st[f"prox_{k}"], st[f"delta_{k}"], st[f"res_nr_{k}"], st[f"res_{k}"]
    for field in INFO_FIELDS:
      st[field] = const_int(0) if field in INTEGER_INFO else _scalar(0.0)
    for field in _CERTIFICATES:
      st[field] = _scalar(0.0)
    st["rho"], st["delta"] = _scalar(settings.rho_init), _scalar(settings.delta_init)
    st["reg_limit"] = _scalar(settings.reg_lower_limit)
    st["refine"] = const_bool(settings.iterative_refinement_always_enabled)
    st["status"] = const_int(int(Status.UNSOLVED))
    return st

  def start(self, st: State, p: State) -> State:
    """The initial point: one solve with ``rho`` and ``delta`` at their initial values, shifted into the interior and onto the central path."""
    data = self.data(p)
    one = lambda mask: sc.where(mask, 1.0, 0.0)
    st = dict(st)
    st["s_l"] = st["z_l"] = one(data.h_l_mask)
    st["s_u"] = st["z_u"] = one(data.h_u_mask)
    st["s_bl"] = st["z_bl"] = one(data.x_l_mask)
    st["s_bu"] = st["z_bu"] = one(data.x_u_mask)
    st, scalings, factor, failed, _ = self.factor_kkt(st, p)
    st["status"] = sc.where(failed, const_int(int(Status.NUMERICS)), st["status"])
    # A failed factorization ends the solve with these values; the solve below is then discarded.
    rhs = Variables(-data.c, data.b, -data.h_l, data.h_u, -data.x_l, data.x_u, *(zeros(st[k].size) for k in ("s_l", "s_u", "s_bl", "s_bu")))
    started = dict(st) | self.kkt_solve(p, scalings, factor, rhs)._asdict()
    started = self._center(started, data)
    for k in _RESIDUALS:
      started[f"prox_{k}"] = started[k]
    started = self.update_residuals_nr(started, data, self.scaling(p))
    started["prox_c_x"], started["prox_dual_lin"] = started["cert_c_x"], started["cert_dual_lin"]
    started["prev_primal_res"], started["prev_dual_res"] = started["primal_res"], started["dual_res"]
    return {k: sc.where(failed, v, started[k]) for k, v in st.items()}

  def _center(self, st: State, data: Data) -> State:
    """Shift the first solve's slacks and duals into the interior, then onto the central path at ``mu``."""
    s = self.structure

    def shift(prefix: str) -> sc.Expr:
      worst = _scalar(0.0)
      if s.m:
        worst = std_max(std_max(worst, -st[f"{prefix}_l"].min()), -st[f"{prefix}_u"].min())
      worst = std_max(worst, -masked_min(st[f"{prefix}_bl"], data.x_l_mask))
      return std_max(worst, -masked_min(st[f"{prefix}_bu"], data.x_u_mask))

    def centered(st: State) -> State:
      st = dict(st)
      delta_s, delta_z = shift("s"), shift("z")
      masks = {"l": data.h_l_mask, "u": data.h_u_mask, "bl": data.x_l_mask, "bu": data.x_u_mask}
      for k, mask in masks.items():
        st[f"s_{k}"] = sc.where(mask, st[f"s_{k}"] + delta_s, st[f"s_{k}"])
        st[f"z_{k}"] = sc.where(mask, st[f"z_{k}"] + delta_z, st[f"z_{k}"])
      mu = std_max(self.calculate_mu(st, data), 1e-10)
      for k, mask in masks.items():
        c = st[f"z_{k}"] - delta_z
        z = (c + (c * c + 4.0 * mu).sqrt()) / 2.0
        st[f"z_{k}"] = sc.where(mask, z, st[f"z_{k}"])
        st[f"s_{k}"] = sc.where(mask, z - c, st[f"s_{k}"])
      st["mu"] = self.calculate_mu(st, data)
      return st

    if self.has_inequalities:
      return centered(st)
    shifted = centered(st)
    present = self.inequality_count(data) > 0.0
    return {k: sc.where(present, shifted[k], v) for k, v in st.items()}

  # The loop of solve_impl:
  #
  #   while (iter < max_iter) {
  #     termination test, update_residuals_r, infeasibility test   (check_exit_conditions)
  #     iter++, factorization, predictor and corrector steps        (newton_step)
  #   }
  #
  # PIQP leaves the loop with a return inside its head. A Scaly loop stops only at its
  # condition, so the head runs once before the loop and again at the end of every pass, and
  # the condition, continue_iterating, is PIQP's test before the next pass.

  def continue_iterating(self, st: State) -> sc.Expr:
    """PIQP's ``while (iter < max_iter)``, with the returns of the loop head: no status yet and iterations left."""
    return sc.equal(st["status"], int(Status.UNSOLVED)) & (st["iter"] < self.settings.max_iter)

  def check_exit_conditions(self, st: State, p: State) -> State:
    """The head of the loop, from the verbose line to the proximal centers; ``res_*`` keeps ``update_residuals_r``'s residuals for the step."""
    data, sc_, settings = self.data(p), self.scaling(p), self.settings
    st = dict(st)

    if settings.verbose:
      st["primal_obj"] = sc.print(
        # sc.print returns its first value, which takes primal_obj's place in the state.
        "piqp_sparse prim_obj={} iter={} dual_obj={} duality_gap={} prim_res={} dual_res={} rho={} delta={} mu={} p_step={} d_step={}",
        st["primal_obj"],
        st["iter"],
        st["dual_obj"],
        st["duality_gap"],
        st["primal_res"],
        st["dual_res"],
        st["rho"],
        st["delta"],
        st["mu"],
        st["primal_step"],
        st["dual_step"],
      )

    solved = ((st["primal_res"] < settings.eps_abs) | (st["primal_res_rel"] < settings.eps_rel)) & (
      (st["dual_res"] < settings.eps_abs) | (st["dual_res_rel"] < settings.eps_rel)
    )
    if settings.check_duality_gap:
      solved = solved & ((st["duality_gap"] < settings.eps_duality_gap_abs) | (st["duality_gap_rel"] < settings.eps_duality_gap_rel))

    checked, res = self.update_residuals_r(st, data, sc_)

    started = checked["iter"] > 0
    certified = self.check_infeasibility_certificates(checked, data, sc_)
    checked = {k: sc.where(started, certified[k], v) for k, v in checked.items()}

    # If the regularized subproblem is solved, but the non-regularized residuals
    # stall, we still move the proximal centers. This keeps the outer proximal
    # point iteration going, such that the proximal steps converge to an
    # infeasibility certificate.
    unsolved = sc.equal(checked["status"], int(Status.UNSOLVED))
    move_dual = started & unsolved & (checked["primal_res_reg"] < 0.1 * checked["primal_res"])
    move_primal = started & unsolved & (checked["dual_res_reg"] < 0.1 * checked["dual_res"])
    checked = self.update_dual_prox_center(checked, move_dual)
    checked = self.update_primal_prox_center(checked, move_primal)
    prox_moved = move_dual | move_primal
    moved, moved_res = self.update_residuals_r(checked, data, sc_)
    checked = {k: sc.where(prox_moved, moved[k], v) for k, v in checked.items()}
    checked |= {f"res_{k}": sc.where(prox_moved, moved_res[k], v) for k, v in res.items()}

    # A solved problem returns before update_residuals_r, with the state as it was.
    out = {k: sc.where(solved, st[k], checked[k]) for k in checked}
    out["status"] = sc.where(solved, const_int(int(Status.SOLVED)), checked["status"])
    return out

  def newton_step(self, st: State, p: State) -> State:
    """The rest of a pass of the loop: from ``iter++`` to the update of the regularization."""
    data, settings = self.data(p), self.settings
    res = {k: st[f"res_{k}"] for k in _RESIDUALS}
    st = dict(st)

    st["iter"] = st["iter"] + 1

    # avoid getting to close to boundary which can result in a division by zero
    epsilon = _EPS
    shift_l = data.h_l_mask & (st["z_l"] < epsilon)
    shift_u = data.h_u_mask & (st["z_u"] < epsilon)
    shift_bl = masked_min(st["z_bl"], data.x_l_mask) < epsilon
    shift_bu = masked_min(st["z_bu"], data.x_u_mask) < epsilon
    st["z_l"] = sc.where(shift_l, st["z_l"] + epsilon, st["z_l"])
    st["z_u"] = sc.where(shift_u, st["z_u"] + epsilon, st["z_u"])
    st["z_bl"] = sc.where(data.x_l_mask & shift_bl, st["z_bl"] + epsilon, st["z_bl"])
    st["z_bu"] = sc.where(data.x_u_mask & shift_bu, st["z_bu"] + epsilon, st["z_bu"])
    boundary_shifted = any_true(shift_l) | any_true(shift_u) | shift_bl | shift_bu
    st["mu"] = sc.where(boundary_shifted, self.calculate_mu(st, data), st["mu"])

    # avoid possibility of converging to a local minimum -> decrease the minimum regularization value
    finetune = (
      (st["no_primal_update"] > settings.reg_finetune_primal_update_threshold)
      & sc.equal(st["rho"], st["reg_limit"])
      & sc.not_equal(st["reg_limit"], settings.reg_finetune_lower_limit)
    ) | (
      (st["no_dual_update"] > settings.reg_finetune_dual_update_threshold)
      & sc.equal(st["delta"], st["reg_limit"])
      & sc.not_equal(st["reg_limit"], settings.reg_finetune_lower_limit)
    )
    finetune = finetune & (st["dual_prox_inf"] < settings.infeasibility_threshold) & (st["primal_prox_inf"] < settings.infeasibility_threshold)
    st["reg_limit"] = sc.where(finetune, settings.reg_finetune_lower_limit, st["reg_limit"])
    st["no_primal_update"] = sc.where(finetune, const_int(0), st["no_primal_update"])
    st["no_dual_update"] = sc.where(finetune, const_int(0), st["no_dual_update"])

    st, scalings, factor, failed, regularization_changed = self.factor_kkt(st, p)
    st["status"] = sc.where(failed, const_int(int(Status.NUMERICS)), st["status"])

    # if (regularization_changed) update_residuals_r();
    recomputed, recomputed_res = self.update_residuals_r(st, data, self.scaling(p))
    st = {k: sc.where(regularization_changed, recomputed[k], v) for k, v in st.items()}
    res = {k: sc.where(regularization_changed, recomputed_res[k], v) for k, v in res.items()}

    # A failed factorization returns PIQP_NUMERICS here; the step below is then discarded.
    stepped = self.step(st, p, res, scalings, factor)
    return {k: sc.where(failed, v, stepped[k]) for k, v in st.items()}

  def step(self, st: State, p: State, res: State, scalings: Scalings, factor: Factor) -> State:
    """``if (m + n_x_l + n_x_u > 0)``: the predictor-corrector step, otherwise a full step."""
    if self.has_inequalities:
      return self.predictor_corrector_step(st, p, res, scalings, factor)
    data = self.data(p)
    present = self.inequality_count(data) > 0.0
    operands = st | p | {f"res_{k}": v for k, v in res.items()} | {f"kkt_{f}": v for f, v in scalings._asdict().items()}
    operands |= {"kkt_L": factor.L, "kkt_d": factor.d}

    def unpack(o: State) -> tuple[State, State, State, Scalings, Factor]:
      q = {k: o[k] for k in p}
      return (
        {k: o[k] for k in st},
        q,
        {k: o[f"res_{k}"] for k in _RESIDUALS},
        Scalings(*(o[f"kkt_{f}"] for f in Scalings._fields)),
        Factor(o["kkt_L"], o["kkt_d"], const_bool(True)),
      )

    return branch(present, operands, lambda o: self.full_step(*unpack(o)), lambda o: self.predictor_corrector_step(*unpack(o)), Layout.of(st), "inequalities")

  def predictor_corrector_step(self, st: State, p: State, res: State, scalings: Scalings, factor: Factor) -> State:
    """One step with inequalities: two solves with one factorization, differing only in the right-hand side ``res.s``."""
    data, settings, sc_ = self.data(p), self.settings, self.scaling(p)
    kkt_solve = lambda rhs: self.kkt_solve(p, scalings, factor, rhs)  # noqa: E731
    s = {k: st[f"s_{k}"] for k in _BOUNDS}
    z = {k: st[f"z_{k}"] for k in _BOUNDS}
    mask = {"l": data.h_l_mask, "u": data.h_u_mask, "bl": data.x_l_mask, "bu": data.x_u_mask}

    # ------------------ predictor step ------------------
    res_s = {k: sc.where(mask[k], -s[k] * z[k], 0.0) for k in _BOUNDS}

    step = kkt_solve(self.rhs(res, res_s))

    # step in the non-negative orthant
    alpha_s, alpha_z = self.calculate_step(st, step)

    # avoid getting to close to the boundary
    alpha_s = alpha_s * settings.tau
    alpha_z = alpha_z * settings.tau

    sigma = _scalar(0.0)
    for k in _BOUNDS:
      sigma = sigma + sc.dot(s[k] + alpha_s * getattr(step, f"s_{k}"), z[k] + alpha_z * getattr(step, f"z_{k}"))
    sigma = sigma / (st["mu"] * self.inequality_count(data))
    sigma = std_max(0.0, std_min(1.0, sigma))
    sigma = sigma * sigma * sigma

    # ------------------ corrector step ------------------
    res_s = {k: sc.where(mask[k], res_s[k] + (-getattr(step, f"s_{k}") * getattr(step, f"z_{k}") + sigma * st["mu"]), 0.0) for k in _BOUNDS}

    step = kkt_solve(self.rhs(res, res_s))

    # step in the non-negative orthant
    alpha_s, alpha_z = self.calculate_step(st, step)

    st = dict(st)
    st["sigma"] = sigma

    # avoid getting to close to the boundary
    st["primal_step"] = alpha_s * settings.tau
    st["dual_step"] = alpha_z * settings.tau

    # ------------------ update ------------------
    for k in ("x", "s_l", "s_u", "s_bl", "s_bu"):
      st[k] = st[k] + st["primal_step"] * getattr(step, k)
    for k in ("y", "z_l", "z_u", "z_bl", "z_bu"):
      st[k] = st[k] + st["dual_step"] * getattr(step, k)

    # displacement from the proximal centers, i.e., the proximal step,
    # used as candidate for the infeasibility certificates
    for k in _RESIDUALS:
      st[f"delta_{k}"] = st[k] - st[f"prox_{k}"]

    mu_prev = st["mu"]
    st["mu"] = self.calculate_mu(st, data)
    mu_rate = std_max(0.0, (mu_prev - st["mu"]) / mu_prev)

    # ------------------ update regularization ------------------
    return self.update_regularization(st, data, sc_, 1.0 - mu_rate, 1.0 - 0.666 * mu_rate, finetuned=True)

  def full_step(self, st: State, p: State, res: State, scalings: Scalings, factor: Factor) -> State:
    """One step without inequalities."""
    data, sc_ = self.data(p), self.scaling(p)
    s = self.structure

    # since there are no inequalities we can take full steps
    step = self.kkt_solve(p, scalings, factor, self.rhs(res, {"l": zeros(s.m), "u": zeros(s.m), "bl": zeros(s.n), "bu": zeros(s.n)}))

    st = dict(st)
    st["primal_step"], st["dual_step"] = _scalar(1.0), _scalar(1.0)
    st["x"] = st["x"] + st["primal_step"] * step.x
    st["y"] = st["y"] + st["dual_step"] * step.y

    st["delta_x"] = st["x"] - st["prox_x"]
    st["delta_y"] = st["y"] - st["prox_y"]

    # ------------------ update regularization ------------------
    return self.update_regularization(st, data, sc_, _scalar(0.1), _scalar(0.5), finetuned=False)

  @staticmethod
  def rhs(res: State, res_s: State) -> Variables:
    """PIQP's ``res``: the regularized residuals of ``update_residuals_r`` and the complementarity rows ``res.s``."""
    return Variables(res["x"], res["y"], res["z_l"], res["z_u"], res["z_bl"], res["z_bu"], res_s["l"], res_s["u"], res_s["bl"], res_s["bu"])

  def update_regularization(self, st: State, data: Data, sc_: Scaling, moved: sc.Expr, held: sc.Expr, finetuned: bool) -> State:
    """``update_residuals_nr``, then each proximal center moves if its residual fell, and ``rho`` and ``delta`` shrink."""
    settings = self.settings
    st = self.update_residuals_nr(st, data, sc_)
    st["delta_c_x"] = st["cert_c_x"] - st["prox_c_x"]
    st["delta_dual_lin"] = st["cert_dual_lin"] - st["prox_dual_lin"]
    early = st["iter"] < 5

    dual_ok = (st["dual_res"] < 0.95 * st["prev_dual_res"]) | (st["dual_res"] < settings.eps_abs) | (st["dual_res_rel"] < settings.eps_rel)
    if finetuned:
      dual_ok = dual_ok | sc.equal(st["rho"], settings.reg_finetune_lower_limit)
    st = self.update_primal_prox_center(st, dual_ok)
    shrink = ~dual_ok & (early | (st["dual_prox_inf"] < settings.infeasibility_threshold))
    st["rho"] = sc.where(dual_ok, std_max(st["reg_limit"], moved * st["rho"]), sc.where(shrink, std_max(st["reg_limit"], held * st["rho"]), st["rho"]))
    st["no_primal_update"] = st["no_primal_update"] + sc.cast(~dual_ok, I32)

    primal_ok = (st["primal_res"] < 0.95 * st["prev_primal_res"]) | (st["primal_res"] < settings.eps_abs) | (st["primal_res_rel"] < settings.eps_rel)
    if finetuned:
      primal_ok = primal_ok | sc.equal(st["delta"], settings.reg_finetune_lower_limit)
    st = self.update_dual_prox_center(st, primal_ok)
    shrink = ~primal_ok & (early | (st["primal_prox_inf"] < settings.infeasibility_threshold))
    st["delta"] = sc.where(primal_ok, std_max(st["reg_limit"], moved * st["delta"]), sc.where(shrink, std_max(st["reg_limit"], held * st["delta"]), st["delta"]))
    st["no_dual_update"] = st["no_dual_update"] + sc.cast(~primal_ok, I32)
    return st

  def unscale_results(self, st: State, data: Data, sc_: Scaling) -> Variables:
    """PIQP's ``unscale_results`` and ``restore_dual``: an infinite bound reports a zero dual and an infinite slack."""
    s_l, s_u = sc_.unscale_slack_ineq(st["s_l"]), sc_.unscale_slack_ineq(st["s_u"])
    z_l, z_u = sc_.unscale_dual_ineq(st["z_l"]), sc_.unscale_dual_ineq(st["z_u"])
    box = lambda mask, value, absent: sc.where(mask, value, absent)
    return Variables(
      x=sc_.unscale_primal(st["x"]),
      y=sc_.unscale_dual_eq(st["y"]),
      z_l=z_l,
      z_u=z_u,
      z_bl=box(data.x_l_mask, sc_.unscale_dual_b(st["z_bl"]), 0.0),
      z_bu=box(data.x_u_mask, sc_.unscale_dual_b(st["z_bu"]), 0.0),
      s_l=sc.where(sc.equal(z_l, 0.0), INF, s_l),
      s_u=sc.where(sc.equal(z_u, 0.0), INF, s_u),
      s_bl=box(data.x_l_mask, sc_.unscale_slack_b(st["s_bl"]), INF),
      s_bu=box(data.x_u_mask, sc_.unscale_slack_b(st["s_bu"]), INF),
    )

  def solve_impl(self, data: Data, scaling: Scaling, AtA: sc.Expr) -> tuple[Variables, sc.Expr, State]:
    """``solve_impl``, ``unscale_results`` and ``restore_dual``."""
    p = self.params(data, scaling, AtA)
    st = self.start(self.initial_state(), p)

    def check_exit_conditions(st_: State, p_: State) -> State:
      head = self.check_exit_conditions(st_, p_)
      go = self.continue_iterating(st_)
      return {k: sc.where(go, head[k], v) for k, v in st_.items()}

    def iteration(st_: State, p_: State) -> State:
      return check_exit_conditions(self.newton_step(st_, p_), p_)

    st = check_exit_conditions(st, p)
    st = loop(lambda st_, _: self.continue_iterating(st_), iteration, st, p, max(self.settings.max_iter, 1), "piqp_iteration")

    status = sc.where(sc.equal(st["status"], int(Status.UNSOLVED)), const_int(int(Status.MAX_ITER_REACHED)), st["status"])
    return self.unscale_results(st, data, scaling), _scaly_status(status), st

  # The generated Functions.

  def _raw_inputs(self) -> Any:
    s = self.structure
    sizes = dict(zip(RAW_INPUTS, (s.P.nnz, s.n, s.A.nnz, s.p, s.G.nnz, s.m, s.m, s.n, s.n), strict=True))
    return Layout(RAW_INPUTS, tuple(sc.TensorType((size,), F64) for size in sizes.values()), "raw_")

  @cached_property
  def _data_layout(self) -> Layout:
    return Layout(Data._fields, data_types(self.structure), "data_")

  @cached_property
  def _scaling_layout(self) -> Layout:
    return Layout(Scaling._fields, scaling_types(self.structure), "scaling_")

  @cached_property
  def _AtA_type(self) -> sc.TensorType:
    return sc.TensorType((len(self.kkt._AtA.rows),), F64)

  @cached_property
  def setup_function(self) -> sc.Function:
    """PIQP's ``setup``: infinite bounds, Ruiz equilibration and ``A^T A``. Returns the scaled data, the scaling and ``A^T A``."""
    raw, data_layout, scaling_layout = self._raw_inputs(), self._data_layout, self._scaling_layout
    outputs = sc.group(data_layout.tree(), scaling_layout.tree(), sc.arg("AtA", self._AtA_type))

    @sc.function(raw.tree(), outputs=outputs, name=f"{self.name}_setup")
    def setup(tree: Any) -> Any:
      data = make_data(self.structure, **raw.unpack(tree))
      data, scaling = scale_data(self.structure, data, self.settings)
      return data_layout.pack(data._asdict()), scaling_layout.pack(scaling._asdict()), self.kkt.init_AtA(data)

    return setup

  @cached_property
  def update_function(self) -> sc.Function:
    """PIQP's ``update`` when the scaling is reused: new data with the previous scaling."""
    raw, data_layout, scaling_layout = self._raw_inputs(), self._data_layout, self._scaling_layout
    outputs = sc.group(data_layout.tree(), sc.arg("AtA", self._AtA_type))

    @sc.function(raw.tree(), scaling_layout.tree(), outputs=outputs, name=f"{self.name}_update")
    def update(tree: Any, scaling_tree: Any) -> Any:
      data = make_data(self.structure, **raw.unpack(tree))
      data = reuse_scaling(self.structure, data, Scaling(**scaling_layout.unpack(scaling_tree)))
      return data_layout.pack(data._asdict()), self.kkt.init_AtA(data)

    return update

  @cached_property
  def solve_function(self) -> sc.Function:
    """PIQP's ``solve``: the variables, the ``ScalySolveStatus`` code and the statistics of ``INFO_FIELDS``."""
    data_layout, scaling_layout = self._data_layout, self._scaling_layout
    s = self.structure
    variables = Layout(Variables._fields, tuple(sc.TensorType((size,), F64) for size in (s.n, s.p, s.m, s.m, s.n, s.n, s.m, s.m, s.n, s.n)), "result_")
    info = Layout(INFO_FIELDS, tuple(sc.TensorType((), I32 if f in INTEGER_INFO else F64) for f in INFO_FIELDS), "info_")
    outputs = sc.group(variables.tree(), sc.arg("status", sc.TensorType((), I32)), info.tree())

    @sc.function(data_layout.tree(), scaling_layout.tree(), sc.arg("AtA", self._AtA_type), outputs=outputs, name=f"{self.name}_solve")
    def solve(data_tree: Any, scaling_tree: Any, AtA: sc.Expr) -> Any:
      data, scaling = Data(**data_layout.unpack(data_tree)), Scaling(**scaling_layout.unpack(scaling_tree))
      result, status, st = self.solve_impl(data, scaling, AtA)
      return variables.pack(result._asdict()), status, info.pack(st)

    return solve


def _scaly_status(status: sc.Expr) -> sc.Expr:
  """PIQP's final status as its ``ScalySolveStatus`` code, ``ERROR`` for any other."""
  out = const_int(int(sc.ScalySolveStatus.ERROR))
  for piqp, scaly in TO_SCALY_STATUS.items():
    out = sc.where(sc.equal(status, int(piqp)), const_int(int(scaly)), out)
  return out


def _ones_like(x: sc.Expr) -> sc.Expr:
  return sc.const(np.ones(x.shape))


_METHODS: dict[tuple[Any, ...], _Method] = {}


def _method(structure: Structure, settings: Settings, name: str = "piqp_sparse") -> _Method:
  key = (structure.n, structure.P.digest, structure.A.digest, structure.G.digest, settings, name)
  if (cached := _METHODS.get(key)) is None:
    cached = _METHODS[key] = _Method(structure, settings, name)
  return cached


def build_solver(P: Any, A: Any = None, G: Any = None, settings: Settings | None = None, name: str = "piqp_sparse") -> tuple[sc.Function, sc.Function, sc.Function]:
  """The generated ``setup``, ``solve`` and ``update`` for the patterns of ``P`` (its upper triangle), ``A`` and ``G``.

  Only the patterns are read; the values come in at run time, as the stored entries of each matrix
  in the order of ``Structure.from_matrices``. With ``raw`` the values of ``P, c, A, b, G, h_l, h_u,
  x_l, x_u``, nested by ``pack_raw``::

    raw = pack_raw(P_values, c, A_values, b, G_values, h_l, h_u, x_l, x_u)
    data, scaling, AtA = setup(raw)              # infinite bounds, equilibration, A^T A
    variables, status, info = solve(data, scaling, AtA)
    data, AtA = update(raw, scaling)             # new values, the same scaling

  ``status`` is an int32 ``ScalySolveStatus`` code and ``info`` holds ``INFO_FIELDS``, whose
  counters are int32. The C entry points are ``{name}_setup``, ``{name}_solve`` and ``{name}_update``.
  """
  P = sparse.csc_array(P)
  n = P.shape[0]
  A = sparse.csc_array((0, n)) if A is None else sparse.csc_array(A)
  G = sparse.csc_array((0, n)) if G is None else sparse.csc_array(G)
  method = _method(Structure.from_matrices(P, A, G)[0], settings or Settings(), name)
  return method.setup_function, method.solve_function, method.update_function


def pack_raw(*values: np.ndarray) -> Any:
  """The values of ``P, c, A, b, G, h_l, h_u, x_l, x_u``, in order, as the nested input of ``setup`` and ``update``: ``sc.group`` takes at most eight parts."""
  return Layout(RAW_INPUTS, (), "").pack(dict(zip(RAW_INPUTS, values, strict=True)))


def _vector(value: Any, size: int, fill: float, name: str) -> np.ndarray:
  if value is None:
    return np.full(size, fill)
  array = np.asarray(value, dtype=np.float64).reshape(-1)
  if array.shape != (size,):
    raise ValueError(f"{name} must have {size} entries, got {array.shape[0]}")
  return array


class SparseSolver:
  """PIQP's ``SparseSolver`` with the KKT system ``sparse_ldlt_cond``, generated with Scaly.

  ``setup`` fixes the sparsity patterns and generates the solver for them, ``update`` changes the
  values within those patterns, and ``solve`` runs the generated code and returns PIQP's status.
  ``settings`` is read at ``setup``; changing it afterwards generates another solver on the next
  ``setup`` or ``update``.
  """

  def __init__(self, settings: Settings | None = None) -> None:
    self.settings = settings or Settings()
    self._structure: Structure | None = None
    self._raw: dict[str, np.ndarray] = {}
    self._scaled: Any = None
    self._result: Result | None = None

  @property
  def result(self) -> Result:
    if self._result is None:
      raise RuntimeError("solve has not run yet")
    return self._result

  def setup(
    self,
    P: Any,
    c: Any,
    A: Any = None,
    b: Any = None,
    G: Any = None,
    h_l: Any = None,
    h_u: Any = None,
    x_l: Any = None,
    x_u: Any = None,
  ) -> None:
    """Generate the solver for the patterns of ``P`` (its upper triangle), ``A`` and ``G``, and equilibrate the data."""
    P = sparse.csc_array(P)
    n = P.shape[0]
    if P.shape != (n, n):
      raise ValueError("P must be square")
    A = sparse.csc_array((0, n)) if A is None else sparse.csc_array(A)
    G = sparse.csc_array((0, n)) if G is None else sparse.csc_array(G)
    if A.shape[1] != n or G.shape[1] != n:
      raise ValueError("A and G must have one column per variable")
    if G.shape[0] and h_l is None and h_u is None:
      raise ValueError("h_l or h_u should be provided")
    if A.shape[0] and b is None:
      raise ValueError("b must be provided with A")
    structure, P_values, A_values, G_values = Structure.from_matrices(P, A, G)
    self._structure = structure
    self._raw = {
      "P": P_values,
      "c": _vector(c, n, 0.0, "c"),
      "A": A_values,
      "b": _vector(b, structure.p, 0.0, "b"),
      "G": G_values,
      "h_l": _vector(h_l, structure.m, -np.inf, "h_l"),
      "h_u": _vector(h_u, structure.m, np.inf, "h_u"),
      "x_l": _vector(x_l, n, -np.inf, "x_l"),
      "x_u": _vector(x_u, n, np.inf, "x_u"),
    }
    self._scale(reuse=False)

  def update(
    self,
    P: Any = None,
    c: Any = None,
    A: Any = None,
    b: Any = None,
    G: Any = None,
    h_l: Any = None,
    h_u: Any = None,
    x_l: Any = None,
    x_u: Any = None,
  ) -> None:
    """Change values within the patterns given to ``setup``. The scaling is recomputed when a matrix changes, unless ``preconditioner_reuse_on_update``."""
    s = self._structure
    if s is None:
      raise RuntimeError("Solver not setup yet")
    matrices = {"P": P, "A": A, "G": G}
    for name, matrix in matrices.items():
      if matrix is not None:
        self._raw[name] = s.values(name, sparse.csc_array(matrix))
    vectors = {"c": (c, s.n), "b": (b, s.p), "h_l": (h_l, s.m), "h_u": (h_u, s.m), "x_l": (x_l, s.n), "x_u": (x_u, s.n)}
    for name, (value, size) in vectors.items():
      if value is not None:
        self._raw[name] = _vector(value, size, 0.0, name)
    changed_matrix = any(matrix is not None for matrix in matrices.values())
    self._scale(reuse=self.settings.preconditioner_reuse_on_update or not changed_matrix)

  def _scale(self, reuse: bool) -> None:
    assert self._structure is not None
    method = _method(self._structure, self.settings)
    nested = pack_raw(*(self._raw[name] for name in RAW_INPUTS))
    if reuse and self._scaled is not None:
      data, AtA = method.update_function(nested, self._scaled[1])
      self._scaled = (data, self._scaled[1], AtA)
    else:
      self._scaled = method.setup_function(nested)

  def solve(self) -> Status:
    """Run the generated solver and return its status. The solution and statistics are in ``result``."""
    if self._structure is None:
      raise RuntimeError("Solver not setup yet")
    if not self.settings.verify_settings():
      return Status.INVALID_SETTINGS
    method = _method(self._structure, self.settings)
    variables, status, info = method.solve_function(*self._scaled)
    values = dict(zip(Variables._fields, _leaves(variables), strict=True))
    stats = dict(zip(INFO_FIELDS, _leaves(info), strict=True))
    status = FROM_SCALY_STATUS[sc.ScalySolveStatus(int(status))]
    fields: dict[str, Any] = {k: (int(v) if k in INTEGER_INFO else float(v)) for k, v in stats.items()}
    self._result = Result(**{k: np.asarray(v) for k, v in values.items()}, info=Info(status=status, **fields))
    return status


def _leaves(tree: Any) -> list[Any]:
  if isinstance(tree, tuple):
    return [leaf for part in tree for leaf in _leaves(part)]
  return [tree]

