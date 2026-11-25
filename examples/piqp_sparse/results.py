"""A solve's status, statistics and solution, as PIQP 0.7.0's ``piqp/results.hpp`` defines them."""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np


class Status(enum.IntEnum):
  """PIQP's ``Status`` codes."""

  SOLVED = 1
  MAX_ITER_REACHED = -1
  PRIMAL_INFEASIBLE = -2
  DUAL_INFEASIBLE = -3
  NUMERICS = -8
  UNSOLVED = -9
  INVALID_SETTINGS = -10


# The statistics the generated solve returns, in this order, after its status. PIQP's timings are
# left out: a Python-level clock around a compiled call would mostly measure the call.
INFO_FIELDS = (
  "iter",
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
  "primal_res_reg",
  "primal_res_reg_rel",
  "dual_res_reg",
  "dual_res_reg_rel",
  "primal_prox_inf",
  "dual_prox_inf",
  "prev_primal_res",
  "prev_dual_res",
  "primal_obj",
  "dual_obj",
  "duality_gap",
  "duality_gap_rel",
  "factor_retires",
  "reg_limit",
  "no_primal_update",
  "no_dual_update",
)

INTEGER_INFO = frozenset({"iter", "factor_retires", "no_primal_update", "no_dual_update"})


@dataclass(frozen=True)
class Info:
  """PIQP's ``Info``: how the solve ended and the last iteration's quantities."""

  status: Status
  iter: int
  rho: float
  delta: float
  mu: float
  sigma: float
  primal_step: float
  dual_step: float
  primal_res: float
  primal_res_rel: float
  dual_res: float
  dual_res_rel: float
  primal_res_reg: float
  primal_res_reg_rel: float
  dual_res_reg: float
  dual_res_reg_rel: float
  primal_prox_inf: float
  dual_prox_inf: float
  prev_primal_res: float
  prev_dual_res: float
  primal_obj: float
  dual_obj: float
  duality_gap: float
  duality_gap_rel: float
  factor_retires: int
  reg_limit: float
  no_primal_update: int
  no_dual_update: int


@dataclass(frozen=True, eq=False)
class Result:
  """PIQP's ``Result``: the primal and dual solution, the slacks and the statistics.

  An infinite bound has a zero dual and an infinite slack, as PIQP's ``restore_dual`` leaves them.
  """

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
