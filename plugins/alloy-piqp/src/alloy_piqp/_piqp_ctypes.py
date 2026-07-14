"""Low-level ctypes bindings retained for parity testing while the nanobind path soaks.

Numeric-only: this module accepts ``numpy`` arrays and yields a ``QPSolution``.
Symbolic plumbing (oracle building, parameter binding) lives in ``qp.py``.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from alloy.solvers import SolverDescriptor, SolverStatus
from alloy.toolchain import load_solver_library

_piqp_lib: ctypes.CDLL | None = None


def piqp_lib() -> ctypes.CDLL:
  global _piqp_lib
  if _piqp_lib is None:
    _piqp_lib = load_solver_library("piqpc")
  return _piqp_lib


piqp_float = ctypes.c_double
piqp_int = ctypes.c_int

_FLOAT_P = ctypes.POINTER(piqp_float)
_INT_P = ctypes.POINTER(piqp_int)


class _piqp_data_dense(ctypes.Structure):  # noqa: N801 - mirror C name
  _fields_ = [
    ("n", piqp_int),
    ("p", piqp_int),
    ("m", piqp_int),
    ("P", _FLOAT_P),
    ("c", _FLOAT_P),
    ("A", _FLOAT_P),
    ("b", _FLOAT_P),
    ("G", _FLOAT_P),
    ("h_l", _FLOAT_P),
    ("h_u", _FLOAT_P),
    ("x_l", _FLOAT_P),
    ("x_u", _FLOAT_P),
  ]


class _piqp_settings(ctypes.Structure):  # noqa: N801
  _fields_ = [
    ("rho_init", piqp_float),
    ("delta_init", piqp_float),
    ("eps_abs", piqp_float),
    ("eps_rel", piqp_float),
    ("check_duality_gap", piqp_int),
    ("eps_duality_gap_abs", piqp_float),
    ("eps_duality_gap_rel", piqp_float),
    ("infeasibility_threshold", piqp_float),
    ("reg_lower_limit", piqp_float),
    ("reg_finetune_lower_limit", piqp_float),
    ("reg_finetune_primal_update_threshold", piqp_int),
    ("reg_finetune_dual_update_threshold", piqp_int),
    ("max_iter", piqp_int),
    ("max_factor_retires", piqp_int),
    ("preconditioner_scale_cost", piqp_int),
    ("preconditioner_reuse_on_update", piqp_int),
    ("preconditioner_iter", piqp_int),
    ("tau", piqp_float),
    ("kkt_solver", piqp_int),
    ("iterative_refinement_always_enabled", piqp_int),
    ("iterative_refinement_eps_abs", piqp_float),
    ("iterative_refinement_eps_rel", piqp_float),
    ("iterative_refinement_max_iter", piqp_int),
    ("iterative_refinement_min_improvement_rate", piqp_float),
    ("iterative_refinement_static_regularization_eps", piqp_float),
    ("iterative_refinement_static_regularization_rel", piqp_float),
    ("verbose", piqp_int),
    ("compute_timings", piqp_int),
  ]


class _piqp_info(ctypes.Structure):  # noqa: N801
  _fields_ = [
    ("status", piqp_int),
    ("iter", piqp_int),
    ("rho", piqp_float),
    ("delta", piqp_float),
    ("mu", piqp_float),
    ("sigma", piqp_float),
    ("primal_step", piqp_float),
    ("dual_step", piqp_float),
    ("primal_res", piqp_float),
    ("primal_res_rel", piqp_float),
    ("dual_res", piqp_float),
    ("dual_res_rel", piqp_float),
    ("primal_res_reg", piqp_float),
    ("primal_res_reg_rel", piqp_float),
    ("dual_res_reg", piqp_float),
    ("dual_res_reg_rel", piqp_float),
    ("primal_prox_inf", piqp_float),
    ("dual_prox_inf", piqp_float),
    ("prev_primal_res", piqp_float),
    ("prev_dual_res", piqp_float),
    ("primal_obj", piqp_float),
    ("dual_obj", piqp_float),
    ("duality_gap", piqp_float),
    ("duality_gap_rel", piqp_float),
    ("factor_retires", piqp_int),
    ("reg_limit", piqp_float),
    ("no_primal_update", piqp_int),
    ("no_dual_update", piqp_int),
    ("setup_time", piqp_float),
    ("update_time", piqp_float),
    ("solve_time", piqp_float),
    ("kkt_factor_time", piqp_float),
    ("kkt_solve_time", piqp_float),
    ("run_time", piqp_float),
  ]


class _piqp_result(ctypes.Structure):  # noqa: N801
  _fields_ = [
    ("x", _FLOAT_P),
    ("y", _FLOAT_P),
    ("z_l", _FLOAT_P),
    ("z_u", _FLOAT_P),
    ("z_bl", _FLOAT_P),
    ("z_bu", _FLOAT_P),
    ("s_l", _FLOAT_P),
    ("s_u", _FLOAT_P),
    ("s_bl", _FLOAT_P),
    ("s_bu", _FLOAT_P),
    ("info", _piqp_info),
  ]


class _piqp_solver_info(ctypes.Structure):  # noqa: N801
  _fields_ = [
    ("is_dense", piqp_int),
    ("n", piqp_int),
    ("p", piqp_int),
    ("m", piqp_int),
  ]


class _piqp_workspace(ctypes.Structure):  # noqa: N801
  _fields_ = [
    ("solver_handle", ctypes.c_void_p),
    ("solver_info", _piqp_solver_info),
    ("result", ctypes.POINTER(_piqp_result)),
  ]


_WORKSPACE_P = ctypes.POINTER(_piqp_workspace)
_WORKSPACE_PP = ctypes.POINTER(_WORKSPACE_P)


def _bind_lib() -> None:
  lib = piqp_lib()
  lib.piqp_set_default_settings_dense.argtypes = [ctypes.POINTER(_piqp_settings)]
  lib.piqp_set_default_settings_dense.restype = None
  lib.piqp_setup_dense.argtypes = [_WORKSPACE_PP, ctypes.POINTER(_piqp_data_dense), ctypes.POINTER(_piqp_settings)]
  lib.piqp_setup_dense.restype = None
  lib.piqp_update_dense.argtypes = [
    _WORKSPACE_P,
    _FLOAT_P,
    _FLOAT_P,
    _FLOAT_P,
    _FLOAT_P,
    _FLOAT_P,
    _FLOAT_P,
    _FLOAT_P,
    _FLOAT_P,
    _FLOAT_P,
  ]
  lib.piqp_update_dense.restype = None
  lib.piqp_solve.argtypes = [_WORKSPACE_P]
  lib.piqp_solve.restype = piqp_int
  lib.piqp_cleanup.argtypes = [_WORKSPACE_P]
  lib.piqp_cleanup.restype = None


_bound = False


def _ensure_bound() -> None:
  global _bound
  if not _bound:
    _bind_lib()
    _bound = True


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


def _ptr(arr: np.ndarray | None) -> ctypes._Pointer | None:
  if arr is None or arr.size == 0:
    return None
  return arr.ctypes.data_as(_FLOAT_P)


def _row_major_to_col_major(M: np.ndarray, rows: int, cols: int) -> np.ndarray:
  """PIQP's dense interface reads ``P``/``A``/``G`` in column-major order."""
  return np.ascontiguousarray(M.reshape(rows, cols).T)


def _read_result_array(ptr: ctypes._Pointer, length: int) -> np.ndarray:
  if length == 0 or not ptr:
    return np.zeros(length, dtype=np.float64)
  buf = (ctypes.c_double * length).from_address(ctypes.addressof(ptr.contents))
  return np.array(buf, dtype=np.float64, copy=True)


class PIQPDenseSolver:
  """Owning handle around a PIQP dense workspace.

  Setup is deferred until the first :meth:`update` call so that PIQP sees the
  real problem data instead of placeholder ``±PIQP_INF`` bounds (which would
  trigger spurious "free constraint" warnings during setup).
  """

  __slots__ = (
    "_n",
    "_p",
    "_m",
    "_ws",
    "_settings_struct",
    "_P_col",
    "_A_col",
    "_G_col",
    "_c",
    "_b",
    "_h_l",
    "_h_u",
    "_x_l",
    "_x_u",
  )

  def __init__(self, n: int, p: int, m: int, *, settings: dict[str, float | int] | None = None) -> None:
    _ensure_bound()
    lib = piqp_lib()
    self._n = int(n)
    self._p = int(p)
    self._m = int(m)

    self._P_col = np.zeros(n * n, dtype=np.float64)
    self._c = np.zeros(n, dtype=np.float64)
    self._A_col = np.zeros(p * n, dtype=np.float64) if p else np.zeros(0, dtype=np.float64)
    self._b = np.zeros(p, dtype=np.float64)
    self._G_col = np.zeros(m * n, dtype=np.float64) if m else np.zeros(0, dtype=np.float64)
    self._h_l = np.full(m, -PIQP_INF, dtype=np.float64)
    self._h_u = np.full(m, PIQP_INF, dtype=np.float64)
    self._x_l = np.full(n, -PIQP_INF, dtype=np.float64)
    self._x_u = np.full(n, PIQP_INF, dtype=np.float64)

    self._settings_struct = _piqp_settings()
    lib.piqp_set_default_settings_dense(ctypes.byref(self._settings_struct))
    for k, v in (settings or {}).items():
      if not hasattr(self._settings_struct, k):
        raise ValueError(f"unknown PIQP setting {k!r}")
      setattr(self._settings_struct, k, v)
    self._ws = _WORKSPACE_P()  # NULL until first update

  def _copy_in(
    self,
    *,
    P: np.ndarray,
    c: np.ndarray,
    A_eq: np.ndarray | None,
    b_eq: np.ndarray | None,
    G_ineq: np.ndarray | None,
    l_ineq: np.ndarray | None,
    u_ineq: np.ndarray | None,
    x_lb: np.ndarray | None,
    x_ub: np.ndarray | None,
  ) -> None:
    n, p, m = self._n, self._p, self._m
    np.copyto(self._P_col, _row_major_to_col_major(P, n, n).reshape(-1))
    np.copyto(self._c, np.asarray(c, dtype=np.float64).reshape(-1))
    if p:
      assert A_eq is not None and b_eq is not None
      np.copyto(self._A_col, _row_major_to_col_major(A_eq, p, n).reshape(-1))
      np.copyto(self._b, np.asarray(b_eq, dtype=np.float64).reshape(-1))
    if m:
      assert G_ineq is not None and l_ineq is not None and u_ineq is not None
      l_arr = np.asarray(l_ineq, dtype=np.float64).reshape(-1)
      u_arr = np.asarray(u_ineq, dtype=np.float64).reshape(-1)
      np.copyto(self._G_col, _row_major_to_col_major(G_ineq, m, n).reshape(-1))
      np.copyto(self._h_l, np.where(np.isneginf(l_arr), -PIQP_INF, l_arr))
      np.copyto(self._h_u, np.where(np.isposinf(u_arr), PIQP_INF, u_arr))
    if x_lb is None:
      np.copyto(self._x_l, np.full(n, -PIQP_INF))
    else:
      xl_arr = np.asarray(x_lb, dtype=np.float64).reshape(-1)
      np.copyto(self._x_l, np.where(np.isneginf(xl_arr), -PIQP_INF, xl_arr))
    if x_ub is None:
      np.copyto(self._x_u, np.full(n, PIQP_INF))
    else:
      xu_arr = np.asarray(x_ub, dtype=np.float64).reshape(-1)
      np.copyto(self._x_u, np.where(np.isposinf(xu_arr), PIQP_INF, xu_arr))

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
    self._copy_in(P=P, c=c, A_eq=A_eq, b_eq=b_eq, G_ineq=G_ineq, l_ineq=l_ineq, u_ineq=u_ineq, x_lb=x_lb, x_ub=x_ub)
    n, p, m = self._n, self._p, self._m
    lib = piqp_lib()
    if not self._ws:
      data = _piqp_data_dense(
        n=n,
        p=p,
        m=m,
        P=_ptr(self._P_col),
        c=_ptr(self._c),
        A=_ptr(self._A_col) if p else None,
        b=_ptr(self._b) if p else None,
        G=_ptr(self._G_col) if m else None,
        h_l=_ptr(self._h_l) if m else None,
        h_u=_ptr(self._h_u) if m else None,
        x_l=_ptr(self._x_l),
        x_u=_ptr(self._x_u),
      )
      lib.piqp_setup_dense(ctypes.byref(self._ws), ctypes.byref(data), ctypes.byref(self._settings_struct))
      return
    lib.piqp_update_dense(
      self._ws,
      _ptr(self._P_col),
      _ptr(self._c),
      _ptr(self._A_col) if p else None,
      _ptr(self._b) if p else None,
      _ptr(self._G_col) if m else None,
      _ptr(self._h_l) if m else None,
      _ptr(self._h_u) if m else None,
      _ptr(self._x_l),
      _ptr(self._x_u),
    )

  def solve(self) -> QPSolution:
    status = piqp_lib().piqp_solve(self._ws)
    res = self._ws.contents.result.contents
    n, p, m = self._n, self._p, self._m
    return QPSolution(
      status=int(status),
      status_name=PIQP_STATUS_NAMES.get(int(status), f"unknown({status})"),
      iter=int(res.info.iter),
      primal_obj=float(res.info.primal_obj),
      x=_read_result_array(res.x, n),
      lam_eq=_read_result_array(res.y, p),
      lam_ineq_l=_read_result_array(res.z_l, m),
      lam_ineq_u=_read_result_array(res.z_u, m),
      lam_box_l=_read_result_array(res.z_bl, n),
      lam_box_u=_read_result_array(res.z_bu, n),
    )

  def cleanup(self) -> None:
    if self._ws:
      piqp_lib().piqp_cleanup(self._ws)
      self._ws = _WORKSPACE_P()

  def __del__(self) -> None:
    try:
      self.cleanup()
    except Exception:  # noqa: BLE001 - destructor must not raise
      pass


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
