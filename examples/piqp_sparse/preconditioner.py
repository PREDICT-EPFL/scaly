"""Ruiz equilibration, PIQP 0.7.0's ``sparse::RuizEquilibration``: the scaling of a QP's data and the unscaling of what a solve reports."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

import scaly as sc

from .data import Data, Structure
from .settings import Settings
from .utils import F64, index, inf_norm, loop

# PIQP's bounds on a scaling factor, MIN_SCALING and MAX_SCALING.
_MIN_SCALING = 1e-4
_MAX_SCALING = 1e4


class Scaling(NamedTuple):
  """The scaling of the cost (``c``), the variables, the equality and inequality rows, and the box bounds.

  PIQP keeps the three middle ones as one vector ``delta`` of length ``n + p + m``.
  """

  c: sc.Expr
  delta_x: sc.Expr
  delta_y: sc.Expr
  delta_z: sc.Expr
  delta_b: sc.Expr

  def unscale_cost(self, cost: sc.Expr) -> sc.Expr:
    return (1.0 / self.c) * cost

  def unscale_primal(self, x: sc.Expr) -> sc.Expr:
    return x * self.delta_x

  def unscale_dual_eq(self, y: sc.Expr) -> sc.Expr:
    return y * (1.0 / self.c) * self.delta_y

  def unscale_dual_ineq(self, z: sc.Expr) -> sc.Expr:
    return z * (1.0 / self.c) * self.delta_z

  def unscale_dual_b(self, z_b: sc.Expr) -> sc.Expr:
    return z_b * (1.0 / self.c) * self.delta_b

  def unscale_slack_ineq(self, s: sc.Expr) -> sc.Expr:
    return s * (1.0 / self.delta_z)

  def unscale_slack_b(self, s_b: sc.Expr) -> sc.Expr:
    return s_b * (1.0 / self.delta_b)

  def unscale_primal_res_eq(self, r: sc.Expr) -> sc.Expr:
    return r * (1.0 / self.delta_y)

  def unscale_primal_res_ineq(self, r: sc.Expr) -> sc.Expr:
    return r * (1.0 / self.delta_z)

  def unscale_primal_res_b(self, r: sc.Expr) -> sc.Expr:
    return r * (1.0 / self.delta_b)

  def unscale_dual_res(self, r: sc.Expr) -> sc.Expr:
    return r * (1.0 / self.c) * (1.0 / self.delta_x)


def scaling_types(s: Structure) -> tuple[sc.TensorType, ...]:
  """The type of each field of ``Scaling`` for a structure, in order."""
  return (sc.TensorType((), F64), *(sc.TensorType((size,), F64) for size in (s.n, s.p, s.m, s.n)))


def limit_scaling(d: sc.Expr) -> sc.Expr:
  """PIQP's ``limit_scaling``: a scaling below ``MIN_SCALING`` becomes one and one above ``MAX_SCALING`` is capped."""
  return sc.where(d < _MIN_SCALING, 1.0, sc.where(d > _MAX_SCALING, _MAX_SCALING, d))


def pre_post_mult_diagonal(values: sc.Expr, pre: sc.Expr, pre_idx: np.ndarray, post: sc.Expr, post_idx: np.ndarray) -> sc.Expr:
  """PIQP's ``pre_mult_diagonal`` then ``post_mult_diagonal`` on the stored ``values``: ``values[k] * pre[pre_idx[k]] * post[post_idx[k]]``."""
  return values * sc.gather(pre, index(pre_idx)) * sc.gather(post, index(post_idx))


def kkt_scaling(s: Structure, data: dict[str, sc.Expr]) -> sc.Expr:
  """The largest magnitude in each column of the full KKT matrix, PIQP's ``kkt_scaling`` block.

  [ P AT GT D ]
  [ A 0  0  0 ]
  [ G 0  0  0 ]
  [ D 0  0  0 ]
  where D is the diagonal of the bounds scaling.
  """
  n, p = s.n, s.p
  values: list[sc.Expr] = [data["x_b_scaling"]]
  ids: list[np.ndarray] = [np.arange(n)]
  if s.P.nnz:
    # P_utri and its mirror below the diagonal
    above = s.P_strict[1]
    P = data["P"].abs()
    values += [P, sc.gather(P, index(above))]
    ids += [s.P_cols, s.P_rows[above]]
  if s.A.nnz:
    A = data["A"].abs()
    values += [A, A]
    ids += [s.A_cols, n + s.A_rows]
  if s.G.nnz:
    G = data["G"].abs()
    values += [G, G]
    ids += [s.G_cols, n + p + s.G_rows]
  return sc.segment_max(sc.concat(values), index(np.concatenate(ids)), n + p + s.m, fill_value=0.0)


def cost_scaling(s: Structure, data: dict[str, sc.Expr]) -> sc.Expr:
  """``gamma``, PIQP's ``cost_scaling`` block: one over the mean column norm of ``P``, at least ``|c|_inf``."""
  if s.P.nnz:
    above = s.P_strict[1]
    P = data["P"].abs()
    delta_iter_cost = sc.segment_max(sc.concat([P, sc.gather(P, index(above))]), index(np.concatenate([s.P_cols, s.P_rows[above]])), s.n, fill_value=0.0)

    # average over the columns of P which are actually populated, since empty
    # columns carry no cost and would only drag the average towards zero
    n_populated = sc.cast(delta_iter_cost > 0.0, F64).sum()
    gamma_sum = sc.where(delta_iter_cost > 0.0, delta_iter_cost, 0.0).sum()
    gamma = sc.where(n_populated > 0.0, gamma_sum / n_populated, 0.0)
  else:
    gamma = sc.const(0.0)
  gamma = limit_scaling(gamma)
  gamma = sc.maximum(gamma, inf_norm(data["c"]))
  gamma = limit_scaling(gamma)
  return 1.0 / gamma


def scale_cost_A_G(s: Structure, data: dict[str, sc.Expr], delta_x: sc.Expr, delta_y: sc.Expr, delta_z: sc.Expr) -> None:
  """The ``scale cost``, ``scale AT`` and ``scale GT`` blocks of ``scale_data``, without ``c``."""
  # scale cost
  if s.P.nnz:
    data["P"] = pre_post_mult_diagonal(data["P"], delta_x, s.P_rows, delta_x, s.P_cols)
  # scale AT
  if s.A.nnz:
    data["A"] = pre_post_mult_diagonal(data["A"], delta_x, s.A_cols, delta_y, s.A_rows)
  # scale GT
  if s.G.nnz:
    data["G"] = pre_post_mult_diagonal(data["G"], delta_x, s.G_cols, delta_z, s.G_rows)


def scale_bounds(data: dict[str, sc.Expr], scaling: Scaling) -> None:
  """The ``scale bounds`` block of ``scale_data``."""
  data["b"] = data["b"] * scaling.delta_y
  data["h_l"] = data["h_l"] * scaling.delta_z
  data["h_u"] = data["h_u"] * scaling.delta_z
  data["x_l"] = data["x_l"] * scaling.delta_b
  data["x_u"] = data["x_u"] * scaling.delta_b


def scale_data(s: Structure, data: Data, settings: Settings) -> tuple[Data, Scaling]:
  """PIQP's ``scale_data`` without ``reuse_prev_scaling``: at most ``preconditioner_iter`` Ruiz iterations."""
  n, p = s.n, s.p
  epsilon = 1e-3

  # init scaling in case max_iter is 0
  state = {
    "P": data.P,
    "c": data.c,
    "A": data.A,
    "G": data.G,
    "x_b_scaling": data.x_b_scaling,
    "cost": sc.const(1.0),
    "delta": sc.const(np.ones(n + p + s.m)),
    "delta_b": sc.const(np.ones(n)),
    "delta_iter": sc.const(np.zeros(n + p + s.m)),
    "delta_iter_b": sc.const(np.zeros(n)),
  }

  def check_exit_cond(st: dict[str, sc.Expr], _: dict[str, sc.Expr]) -> sc.Expr:
    """PIQP's loop condition: some scaling of the last iteration is more than ``epsilon`` away from one."""
    return sc.maximum(inf_norm(1.0 - st["delta_iter"]), inf_norm(1.0 - st["delta_iter_b"])) > epsilon

  def ruiz_iteration(st: dict[str, sc.Expr], _: dict[str, sc.Expr]) -> dict[str, sc.Expr]:
    st = dict(st)

    # calculate scaling of full KKT matrix
    delta_iter = kkt_scaling(s, st)
    delta_iter_b = st["x_b_scaling"]

    delta_iter = limit_scaling(delta_iter)
    delta_iter_b = limit_scaling(delta_iter_b)

    delta_iter = 1.0 / delta_iter.sqrt()
    delta_iter_b = 1.0 / delta_iter_b.sqrt()
    delta_x, delta_y, delta_z = delta_iter[:n], delta_iter[n : n + p], delta_iter[n + p :]

    # scale cost, AT and GT
    scale_cost_A_G(s, st, delta_x, delta_y, delta_z)
    st["c"] = st["c"] * delta_x

    # scale box scalings
    st["x_b_scaling"] = st["x_b_scaling"] * (delta_iter_b * delta_x)

    st["delta"] = st["delta"] * delta_iter
    st["delta_b"] = st["delta_b"] * delta_iter_b

    if settings.preconditioner_scale_cost:
      # scaling for the cost
      gamma = cost_scaling(s, st)

      # scale cost
      st["P"] = st["P"] * gamma
      st["c"] = st["c"] * gamma

      st["cost"] = st["cost"] * gamma

    st["delta_iter"], st["delta_iter_b"] = delta_iter, delta_iter_b
    return st

  final = loop(check_exit_cond, ruiz_iteration, state, {}, settings.preconditioner_iter, "ruiz") if settings.preconditioner_iter else state

  delta = final["delta"]
  scaling = Scaling(final["cost"], delta[:n], delta[n : n + p], delta[n + p :], final["delta_b"])
  scaled = data._asdict()
  scaled.update({name: final[name] for name in ("P", "c", "A", "G", "x_b_scaling")})

  # scale bounds
  scale_bounds(scaled, scaling)
  return Data(**scaled), scaling


def reuse_scaling(s: Structure, data: Data, scaling: Scaling) -> Data:
  """PIQP's ``scale_data`` with ``reuse_prev_scaling``: the previous scaling applied to new data."""
  scaled = data._asdict()

  # scale cost
  if s.P.nnz:
    scaled["P"] = scaled["P"] * scaling.c
  scale_cost_A_G(s, scaled, scaling.delta_x, scaling.delta_y, scaling.delta_z)
  scaled["c"] = scaled["c"] * (scaling.c * scaling.delta_x)

  # scale box scalings
  scaled["x_b_scaling"] = scaled["x_b_scaling"] * (scaling.delta_b * scaling.delta_x)

  # scale bounds
  scale_bounds(scaled, scaling)
  return Data(**scaled)
