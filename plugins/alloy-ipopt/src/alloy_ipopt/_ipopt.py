"""Low-level ctypes bindings for IPOPT's C standard interface.

Numeric-only: accepts plain Python callables for ``eval_f``/``eval_g``/
``eval_grad_f``/``eval_jac_g``/``eval_h``, plus the sparse Jacobian and Hessian
patterns, and runs a solve. Higher-level oracle assembly lives in ``nlp.py``.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from alloy.solvers import SolverDescriptor, SolverStatus
from alloy.toolchain import load_solver_library

_ipopt_lib: ctypes.CDLL | None = None


def ipopt_lib() -> ctypes.CDLL:
  global _ipopt_lib
  if _ipopt_lib is None:
    _ipopt_lib = load_solver_library("ipopt")
  return _ipopt_lib


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
Intermediate_CB = ctypes.CFUNCTYPE(
  ctypes.c_bool,
  ipindex,
  ipindex,
  ipnumber,
  ipnumber,
  ipnumber,
  ipnumber,
  ipnumber,
  ipnumber,
  ipnumber,
  ipnumber,
  ipindex,
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
  lib.SetIntermediateCallback.argtypes = [_IpoptProblem, Intermediate_CB]
  lib.SetIntermediateCallback.restype = ctypes.c_bool
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
  iters: int
  eval_counts: dict[str, int]


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
  lam_g0: np.ndarray | None = None,
  z_L0: np.ndarray | None = None,
  z_U0: np.ndarray | None = None,
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

  def seed(value: np.ndarray | None, shape: tuple[int, ...], name: str) -> np.ndarray:
    if value is None:
      return np.zeros(shape, dtype=np.float64)
    out = np.asarray(value, dtype=np.float64)
    if out.shape != shape:
      raise ValueError(f"{name} has shape {out.shape}, expected {shape}")
    return np.ascontiguousarray(out).copy()

  mult_g = seed(lam_g0, (m,), "lam_g0")
  mult_x_L = seed(z_L0, (n,), "z_L0")
  mult_x_U = seed(z_U0, (n,), "z_U0")

  error_holder: list[BaseException] = []
  eval_counts = {name: 0 for name in ("eval_f", "eval_grad_f", "eval_g", "eval_jac_g", "eval_h")}
  iters = 0

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
    eval_counts["eval_f"] += 1
    x = _copy_ptr_in(x_ptr, n_)
    obj_ptr[0] = float(eval_f(x))
    return True

  @_wrap
  def _eval_grad_f(n_, x_ptr, _new_x, grad_ptr, _user_data):
    eval_counts["eval_grad_f"] += 1
    x = _copy_ptr_in(x_ptr, n_)
    out = np.ascontiguousarray(eval_grad_f(x), dtype=np.float64)
    if out.size != n_:
      raise ValueError(f"eval_grad_f returned size {out.size}, expected {n_}")
    _write_array(grad_ptr, out)
    return True

  @_wrap
  def _eval_g(n_, x_ptr, _new_x, m_, g_ptr, _user_data):
    eval_counts["eval_g"] += 1
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
      eval_counts["eval_jac_g"] += 1
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
      eval_counts["eval_h"] += 1
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

  def _intermediate(_alg_mod, iter_count, *_args):
    nonlocal iters
    iters = int(iter_count)
    return True

  cb_intermediate = Intermediate_CB(_intermediate)

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
    if not lib.SetIntermediateCallback(problem, cb_intermediate):
      raise RuntimeError("SetIntermediateCallback failed")
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
    del cb_f, cb_grad_f, cb_g, cb_jac_g, cb_h, cb_intermediate

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
    iters=iters,
    eval_counts=eval_counts,
  )


def _nlp_backend(descriptor: SolverDescriptor, inputs: Sequence[np.ndarray]) -> tuple[list[np.ndarray], SolverStatus]:
  """Pure-numeric IPOPT runner used by direct ``SolverFunction`` calls."""
  n, n_h, n_g = descriptor.n, descriptor.n_eq, descriptor.n_ineq
  m = n_h + n_g

  base_fn = descriptor.base
  grad_fn = descriptor.grad
  jac_fn = descriptor.jac
  hess_fn = descriptor.hess
  bound_fn = descriptor.bounds
  assert base_fn is not None and grad_fn is not None and hess_fn is not None and bound_fn is not None

  x0 = inputs[0]
  lam_g0 = np.concatenate([inputs[1].reshape(-1), inputs[2].reshape(-1)])
  param_args = list(inputs[3:])

  bounds = dict(zip(bound_fn.output_names, bound_fn.eval_list(*param_args), strict=True))
  x_lb = bounds["x_lb"]
  x_ub = bounds["x_ub"]
  if m:
    l_in = bounds["l_ineq"] if "l_ineq" in bounds else np.full(n_g, -IPOPT_INF)
    u_in = bounds["u_ineq"] if "u_ineq" in bounds else np.full(n_g, IPOPT_INF)
  else:
    l_in = np.zeros(0)
    u_in = np.zeros(0)
  g_L = np.concatenate([np.zeros(n_h), l_in])
  g_U = np.concatenate([np.zeros(n_h), u_in])

  jac_sp = descriptor.jac_sparsity
  hess_sp = descriptor.hess_sparsity
  assert hess_sp is not None
  hess_rows_full = np.asarray(hess_sp.rows, dtype=np.int32)
  hess_cols_full = np.asarray(hess_sp.cols, dtype=np.int32)
  lower_mask = np.asarray(descriptor.hess_lower_mask, dtype=bool)
  hess_rows = hess_rows_full[lower_mask]
  hess_cols = hess_cols_full[lower_mask]
  jac_rows = np.asarray(jac_sp.rows, dtype=np.int32) if jac_sp is not None else np.zeros(0, dtype=np.int32)
  jac_cols = np.asarray(jac_sp.cols, dtype=np.int32) if jac_sp is not None else np.zeros(0, dtype=np.int32)

  def eval_f(x_val: np.ndarray) -> float:
    out = base_fn.eval_list(x_val, *param_args)
    return float(out[0])

  def eval_g(x_val: np.ndarray) -> np.ndarray:
    out = base_fn.eval_list(x_val, *param_args)
    return np.asarray(out[1], dtype=np.float64).reshape(-1)

  def eval_grad_f(x_val: np.ndarray) -> np.ndarray:
    out = grad_fn.eval_list(x_val, *param_args)
    return np.asarray(out[0], dtype=np.float64).reshape(-1)

  def eval_jac_g(x_val: np.ndarray) -> np.ndarray:
    assert jac_fn is not None
    out = jac_fn.eval_list(x_val, *param_args)
    return np.asarray(out[0], dtype=np.float64).reshape(-1)

  def eval_h(x_val: np.ndarray, obj_factor: float, lam_all: np.ndarray) -> np.ndarray:
    if m:
      out = hess_fn.eval_list(x_val, np.asarray(obj_factor, dtype=np.float64), lam_all, *param_args)
    else:
      out = hess_fn.eval_list(x_val, np.asarray(obj_factor, dtype=np.float64), *param_args)
    vals = np.asarray(out[0], dtype=np.float64).reshape(-1)
    return vals[lower_mask]

  sol = solve_ipopt(
    n=n,
    m=m,
    x0=np.asarray(x0, dtype=np.float64).reshape(-1),
    x_L=x_lb,
    x_U=x_ub,
    g_L=g_L,
    g_U=g_U,
    jac_rows=jac_rows,
    jac_cols=jac_cols,
    hess_rows=hess_rows,
    hess_cols=hess_cols,
    eval_f=eval_f,
    eval_grad_f=eval_grad_f,
    eval_g=eval_g if m else (lambda _x: np.zeros(0)),
    eval_jac_g=eval_jac_g if m else (lambda _x: np.zeros(0)),
    eval_h=eval_h,
    options={k: v for k, v in descriptor.options},
    lam_g0=lam_g0,
  )

  h_out = sol.g[:n_h] if n_h else np.zeros(0)
  g_out = sol.g[n_h:] if n_g else np.zeros(0)
  lam_h = sol.mult_g[:n_h] if n_h else np.zeros(0)
  lam_g = sol.mult_g[n_h:] if n_g else np.zeros(0)
  lam_box = sol.mult_x_U - sol.mult_x_L
  outs = [sol.x, np.asarray(sol.obj, dtype=np.float64), h_out, g_out, lam_h, lam_g, lam_box]
  return outs, SolverStatus(code=sol.status, name=sol.status_name, iter=sol.iters, stats={"iters": sol.iters, **sol.eval_counts})
