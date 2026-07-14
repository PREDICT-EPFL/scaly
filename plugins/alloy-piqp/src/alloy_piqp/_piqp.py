"""Low-level nanobind bindings for PIQP's dense C interface."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from alloy.solvers import SolverDescriptor, SolverStatus

from ._piqp_ext import PIQPDenseSolver as _PIQPDenseSolver

PIQP_STATUS_NAMES = {
  1: "solved",
  -1: "max_iter_reached",
  -2: "primal_infeasible",
  -3: "dual_infeasible",
  -8: "numerics",
  -9: "unsolved",
  -10: "invalid_settings",
}

PIQP_INF = 1e30


@dataclass(frozen=True, slots=True)
class QPSolution:
  status: int
  status_name: str
  iter: int
  primal_obj: float
  x: np.ndarray
  lam_eq: np.ndarray
  lam_ineq_l: np.ndarray
  lam_ineq_u: np.ndarray
  lam_box_l: np.ndarray
  lam_box_u: np.ndarray


def _array(value: np.ndarray | None) -> np.ndarray | None:
  return None if value is None else np.ascontiguousarray(value, dtype=np.float64)


class PIQPDenseSolver:
  """Owning handle around a PIQP dense workspace."""

  __slots__ = ("_solver",)

  def __init__(self, n: int, p: int, m: int, *, settings: dict[str, float | int] | None = None) -> None:
    self._solver = _PIQPDenseSolver(n, p, m, settings or {})

  def update(
    self,
    *,
    P: np.ndarray,
    c: np.ndarray,
    A_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    G_ineq: np.ndarray | None = None,
    l_ineq: np.ndarray | None = None,
    u_ineq: np.ndarray | None = None,
    x_lb: np.ndarray | None = None,
    x_ub: np.ndarray | None = None,
  ) -> None:
    self._solver.update(_array(P), _array(c), _array(A_eq), _array(b_eq), _array(G_ineq), _array(l_ineq), _array(u_ineq), _array(x_lb), _array(x_ub))

  def solve(self) -> QPSolution:
    status, iterations, primal_obj, x, lam_eq, z_l, z_u, z_bl, z_bu = self._solver.solve()
    return QPSolution(status, PIQP_STATUS_NAMES.get(status, f"unknown({status})"), iterations, primal_obj, x, lam_eq, z_l, z_u, z_bl, z_bu)

  def cleanup(self) -> None:
    self._solver.cleanup()


def _qp_backend(descriptor: SolverDescriptor, inputs: Sequence[np.ndarray]) -> tuple[list[np.ndarray], SolverStatus]:
  """Pure-numeric PIQP runner used by direct ``SolverFunction`` calls."""
  n, p_dim, m_dim = descriptor.n, descriptor.n_eq, descriptor.n_ineq
  oracle = descriptor.oracle
  assert oracle is not None

  param_args = list(inputs[3:])
  data = oracle.eval_list(*param_args)
  out_by_name = dict(zip(descriptor.oracle_output_names, data, strict=True))

  P_arr = out_by_name["P"].reshape(n, n)
  c_arr = out_by_name["c"]
  A_arr = out_by_name["A_eq"].reshape(p_dim, n) if p_dim else None
  b_arr = out_by_name["b_eq"] if p_dim else None
  G_arr = out_by_name["G_ineq"].reshape(m_dim, n) if m_dim else None
  l_arr = out_by_name["l_ineq"] if m_dim else None
  u_arr = out_by_name["u_ineq"] if m_dim else None
  xl_arr = out_by_name["x_lb"]
  xu_arr = out_by_name["x_ub"]

  workspace: PIQPDenseSolver | None = descriptor.runtime.get("piqp_workspace")
  if workspace is None:
    workspace = PIQPDenseSolver(n, p_dim, m_dim, settings=dict(descriptor.options))
    descriptor.runtime["piqp_workspace"] = workspace
  workspace.update(
    P=P_arr,
    c=c_arr,
    A_eq=A_arr,
    b_eq=b_arr,
    G_ineq=G_arr,
    l_ineq=l_arr,
    u_ineq=u_arr,
    x_lb=xl_arr,
    x_ub=xu_arr,
  )
  sol = workspace.solve()
  outs = [
    sol.x,
    np.asarray(sol.primal_obj, dtype=np.float64),
    sol.lam_eq,
    sol.lam_ineq_u - sol.lam_ineq_l,
    sol.lam_box_u - sol.lam_box_l,
  ]
  return outs, SolverStatus(code=sol.status, name=sol.status_name, iter=sol.iter)
