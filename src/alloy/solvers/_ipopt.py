"""Low-level ctypes bindings for IPOPT's C standard interface.

Numeric-only: accepts plain Python callables for ``eval_f``/``eval_g``/
``eval_grad_f``/``eval_jac_g``/``eval_h``, plus the sparse Jacobian and Hessian
patterns, and runs a solve. Higher-level oracle assembly lives in ``nlp.py``.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from ._lib import ipopt_lib

ipnumber = ctypes.c_double
ipindex = ctypes.c_int

_NUM_P = ctypes.POINTER(ipnumber)
_IDX_P = ctypes.POINTER(ipindex)

IPOPT_INF = 2e19  # nlp_lower_bound_inf / nlp_upper_bound_inf default


class _IpoptProblemInfo(ctypes.Structure):
  pass


_IpoptProblem = ctypes.POINTER(_IpoptProblemInfo)
_UserData = ctypes.c_void_p

Eval_F_CB = ctypes.CFUNCTYPE(ctypes.c_bool, ipindex, _NUM_P, ctypes.c_bool, _NUM_P, _UserData)
Eval_Grad_F_CB = ctypes.CFUNCTYPE(ctypes.c_bool, ipindex, _NUM_P, ctypes.c_bool, _NUM_P, _UserData)
Eval_G_CB = ctypes.CFUNCTYPE(ctypes.c_bool, ipindex, _NUM_P, ctypes.c_bool, ipindex, _NUM_P, _UserData)
Eval_Jac_G_CB = ctypes.CFUNCTYPE(ctypes.c_bool, ipindex, _NUM_P, ctypes.c_bool, ipindex, ipindex, _IDX_P, _IDX_P, _NUM_P, _UserData)
Eval_H_CB = ctypes.CFUNCTYPE(
  ctypes.c_bool,
  ipindex,
  _NUM_P,
  ctypes.c_bool,
  ipnumber,
  ipindex,
  _NUM_P,
  ctypes.c_bool,
  ipindex,
  _IDX_P,
  _IDX_P,
  _NUM_P,
  _UserData,
)


_bound = False


def _ensure_bound() -> None:
  global _bound
  if _bound:
    return
  lib = ipopt_lib()
  lib.CreateIpoptProblem.argtypes = [
    ipindex,
    _NUM_P,
    _NUM_P,
    ipindex,
    _NUM_P,
    _NUM_P,
    ipindex,
    ipindex,
    ipindex,
    Eval_F_CB,
    Eval_G_CB,
    Eval_Grad_F_CB,
    Eval_Jac_G_CB,
    Eval_H_CB,
  ]
  lib.CreateIpoptProblem.restype = _IpoptProblem
  lib.FreeIpoptProblem.argtypes = [_IpoptProblem]
  lib.FreeIpoptProblem.restype = None
  lib.AddIpoptStrOption.argtypes = [_IpoptProblem, ctypes.c_char_p, ctypes.c_char_p]
  lib.AddIpoptStrOption.restype = ctypes.c_bool
  lib.AddIpoptNumOption.argtypes = [_IpoptProblem, ctypes.c_char_p, ipnumber]
  lib.AddIpoptNumOption.restype = ctypes.c_bool
  lib.AddIpoptIntOption.argtypes = [_IpoptProblem, ctypes.c_char_p, ipindex]
  lib.AddIpoptIntOption.restype = ctypes.c_bool
  lib.IpoptSolve.argtypes = [_IpoptProblem, _NUM_P, _NUM_P, _NUM_P, _NUM_P, _NUM_P, _NUM_P, _UserData]
  lib.IpoptSolve.restype = ctypes.c_int
  _bound = True


IPOPT_STATUS_NAMES = {
  0: "solve_succeeded",
  1: "solved_to_acceptable_level",
  2: "infeasible_problem_detected",
  3: "search_direction_becomes_too_small",
  4: "diverging_iterates",
  5: "user_requested_stop",
  6: "feasible_point_found",
  -1: "maximum_iterations_exceeded",
  -2: "restoration_failed",
  -3: "error_in_step_computation",
  -4: "maximum_cputime_exceeded",
  -5: "maximum_walltime_exceeded",
  -10: "not_enough_degrees_of_freedom",
  -11: "invalid_problem_definition",
  -12: "invalid_option",
  -13: "invalid_number_detected",
  -100: "unrecoverable_exception",
  -101: "nonipopt_exception_thrown",
  -102: "insufficient_memory",
  -199: "internal_error",
}


@dataclass(frozen=True, slots=True)
class NLPSolution:
  status: int
  status_name: str
  x: np.ndarray
  obj: float
  g: np.ndarray
  mult_g: np.ndarray
  mult_x_L: np.ndarray
  mult_x_U: np.ndarray


def _copy_ptr_in(ptr, length: int) -> np.ndarray:
  if length == 0 or not ptr:
    return np.zeros(length, dtype=np.float64)
  buf = (ctypes.c_double * length).from_address(ctypes.addressof(ptr.contents))
  return np.frombuffer(buf, dtype=np.float64, count=length).copy()


def _write_array(ptr, source: np.ndarray) -> None:
  if source.size == 0:
    return
  ctypes.memmove(ptr, source.ctypes.data, source.nbytes)


def _write_index_array(ptr, source: np.ndarray) -> None:
  if source.size == 0:
    return
  arr = np.ascontiguousarray(source, dtype=np.int32)
  ctypes.memmove(ptr, arr.ctypes.data, arr.nbytes)


def _read_array(ptr, length: int) -> np.ndarray:
  if length == 0 or not ptr:
    return np.zeros(length, dtype=np.float64)
  return np.frombuffer((ctypes.c_double * length).from_address(ctypes.addressof(ptr.contents)), dtype=np.float64, count=length).copy()


def _replace_nonfinite(arr: np.ndarray, sign: int) -> np.ndarray:
  out = np.asarray(arr, dtype=np.float64).copy()
  if sign < 0:
    out = np.where(np.isneginf(out) | (out <= -IPOPT_INF), -IPOPT_INF, out)
  else:
    out = np.where(np.isposinf(out) | (out >= IPOPT_INF), IPOPT_INF, out)
  return out


def solve_ipopt(
  *,
  n: int,
  m: int,
  x0: np.ndarray,
  x_L: np.ndarray,
  x_U: np.ndarray,
  g_L: np.ndarray,
  g_U: np.ndarray,
  jac_rows: np.ndarray,
  jac_cols: np.ndarray,
  hess_rows: np.ndarray,
  hess_cols: np.ndarray,
  eval_f,
  eval_grad_f,
  eval_g,
  eval_jac_g,
  eval_h,
  options: dict[str, str | int | float] | None = None,
) -> NLPSolution:
  """Wrap CreateIpoptProblem + IpoptSolve.

  ``eval_*`` are Python callables that operate on numpy arrays:

  - ``eval_f(x) -> float``
  - ``eval_grad_f(x) -> ndarray(n,)``
  - ``eval_g(x) -> ndarray(m,)``
  - ``eval_jac_g(x) -> ndarray(nnz_jac,)``  matching the (jac_rows, jac_cols) order
  - ``eval_h(x, obj_factor, lam) -> ndarray(nnz_hess,)``  matching (hess_rows, hess_cols)
  """
  _ensure_bound()
  lib = ipopt_lib()
  x_L_buf = np.ascontiguousarray(_replace_nonfinite(x_L, -1))
  x_U_buf = np.ascontiguousarray(_replace_nonfinite(x_U, +1))
  g_L_buf = np.ascontiguousarray(_replace_nonfinite(g_L, -1))
  g_U_buf = np.ascontiguousarray(_replace_nonfinite(g_U, +1))
  jac_rows_arr = np.ascontiguousarray(jac_rows, dtype=np.int32)
  jac_cols_arr = np.ascontiguousarray(jac_cols, dtype=np.int32)
  hess_rows_arr = np.ascontiguousarray(hess_rows, dtype=np.int32)
  hess_cols_arr = np.ascontiguousarray(hess_cols, dtype=np.int32)
  nele_jac = int(jac_rows_arr.size)
  nele_hess = int(hess_rows_arr.size)

  error_holder: list[BaseException] = []

  def _wrap(fn):
    """Wrap a Python callable so that any raised exception is captured and signaled
    back through the C bool return — IPOPT will treat ``false`` as an evaluation
    failure and (depending on the situation) restart or fail gracefully.
    """

    def safe(*args, **kwargs):
      try:
        return fn(*args, **kwargs)
      except BaseException as exc:  # noqa: BLE001
        if not error_holder:
          error_holder.append(exc)
        return False

    return safe

  @_wrap
  def _eval_f(n_, x_ptr, _new_x, obj_ptr, _user_data):
    x = _copy_ptr_in(x_ptr, n_)
    obj_ptr[0] = float(eval_f(x))
    return True

  @_wrap
  def _eval_grad_f(n_, x_ptr, _new_x, grad_ptr, _user_data):
    x = _copy_ptr_in(x_ptr, n_)
    out = np.ascontiguousarray(eval_grad_f(x), dtype=np.float64)
    if out.size != n_:
      raise ValueError(f"eval_grad_f returned size {out.size}, expected {n_}")
    _write_array(grad_ptr, out)
    return True

  @_wrap
  def _eval_g(n_, x_ptr, _new_x, m_, g_ptr, _user_data):
    if m_ == 0:
      return True
    x = _copy_ptr_in(x_ptr, n_)
    out = np.ascontiguousarray(eval_g(x), dtype=np.float64)
    if out.size != m_:
      raise ValueError(f"eval_g returned size {out.size}, expected {m_}")
    _write_array(g_ptr, out)
    return True

  @_wrap
  def _eval_jac_g(n_, x_ptr, _new_x, m_, nele_jac_, iRow, jCol, values, _user_data):
    if nele_jac_ == 0:
      return True
    if values:
      x = _copy_ptr_in(x_ptr, n_)
      out = np.ascontiguousarray(eval_jac_g(x), dtype=np.float64)
      if out.size != nele_jac_:
        raise ValueError(f"eval_jac_g returned size {out.size}, expected {nele_jac_}")
      _write_array(values, out)
    else:
      _write_index_array(iRow, jac_rows_arr)
      _write_index_array(jCol, jac_cols_arr)
    return True

  @_wrap
  def _eval_h(n_, x_ptr, _new_x, obj_factor, m_, lam_ptr, _new_lam, nele_hess_, iRow, jCol, values, _user_data):
    if nele_hess_ == 0:
      return True
    if values:
      x = _copy_ptr_in(x_ptr, n_)
      lam = _copy_ptr_in(lam_ptr, m_) if m_ else np.zeros(0, dtype=np.float64)
      out = np.ascontiguousarray(eval_h(x, float(obj_factor), lam), dtype=np.float64)
      if out.size != nele_hess_:
        raise ValueError(f"eval_h returned size {out.size}, expected {nele_hess_}")
      _write_array(values, out)
    else:
      _write_index_array(iRow, hess_rows_arr)
      _write_index_array(jCol, hess_cols_arr)
    return True

  cb_f = Eval_F_CB(_eval_f)
  cb_grad_f = Eval_Grad_F_CB(_eval_grad_f)
  cb_g = Eval_G_CB(_eval_g)
  cb_jac_g = Eval_Jac_G_CB(_eval_jac_g)
  cb_h = Eval_H_CB(_eval_h)

  problem = lib.CreateIpoptProblem(
    ipindex(n),
    x_L_buf.ctypes.data_as(_NUM_P),
    x_U_buf.ctypes.data_as(_NUM_P),
    ipindex(m),
    g_L_buf.ctypes.data_as(_NUM_P) if m else None,
    g_U_buf.ctypes.data_as(_NUM_P) if m else None,
    ipindex(nele_jac),
    ipindex(nele_hess),
    ipindex(0),  # index_style = 0 (C style)
    cb_f,
    cb_g,
    cb_grad_f,
    cb_jac_g,
    cb_h,
  )
  if not problem:
    raise RuntimeError("CreateIpoptProblem returned NULL")
  try:
    for key, value in (options or {}).items():
      kb = key.encode()
      if isinstance(value, bool):
        ok = lib.AddIpoptIntOption(problem, kb, ipindex(int(value)))
      elif isinstance(value, int):
        ok = lib.AddIpoptIntOption(problem, kb, ipindex(value))
      elif isinstance(value, float):
        ok = lib.AddIpoptNumOption(problem, kb, ipnumber(value))
      elif isinstance(value, str):
        ok = lib.AddIpoptStrOption(problem, kb, value.encode())
      else:
        raise TypeError(f"unsupported IPOPT option value type for {key}: {type(value).__name__}")
      if not ok:
        raise ValueError(f"IPOPT rejected option {key}={value!r}")

    x_buf = np.ascontiguousarray(x0, dtype=np.float64).copy()
    g_buf = np.zeros(m, dtype=np.float64)
    obj_buf = (ipnumber * 1)(0.0)
    mult_g = np.zeros(m, dtype=np.float64)
    mult_x_L = np.zeros(n, dtype=np.float64)
    mult_x_U = np.zeros(n, dtype=np.float64)

    status = lib.IpoptSolve(
      problem,
      x_buf.ctypes.data_as(_NUM_P),
      g_buf.ctypes.data_as(_NUM_P) if m else None,
      ctypes.cast(obj_buf, _NUM_P),
      mult_g.ctypes.data_as(_NUM_P) if m else None,
      mult_x_L.ctypes.data_as(_NUM_P),
      mult_x_U.ctypes.data_as(_NUM_P),
      None,
    )
  finally:
    lib.FreeIpoptProblem(problem)
    # Keep callbacks alive until here.
    del cb_f, cb_grad_f, cb_g, cb_jac_g, cb_h

  if error_holder:
    raise error_holder[0]

  return NLPSolution(
    status=int(status),
    status_name=IPOPT_STATUS_NAMES.get(int(status), f"unknown({status})"),
    x=x_buf,
    obj=float(obj_buf[0]),
    g=g_buf,
    mult_g=mult_g,
    mult_x_L=mult_x_L,
    mult_x_U=mult_x_U,
  )
