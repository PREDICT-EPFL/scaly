"""PIQP 0.7.0 itself, through the C interface of the library the ``scaly-piqp`` plugin vendors: the reference the generated solver is compared with."""

from __future__ import annotations

import ctypes
import os
import re
import sys
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from .results import Status
from .settings import Settings

_int = ctypes.c_int
_float = ctypes.c_double
_ptr = ctypes.POINTER(_float)
# piqp_kkt_solver in piqp_typedef.h: the full KKT system, and the one with every constraint eliminated.
SPARSE_LDLT = 1
SPARSE_LDLT_COND = 4


class _CSC(ctypes.Structure):
  _fields_ = [("m", _int), ("n", _int), ("nnz", _int), ("p", ctypes.POINTER(_int)), ("i", ctypes.POINTER(_int)), ("x", _ptr)]


class _Data(ctypes.Structure):
  _fields_ = [
    ("n", _int),
    ("p", _int),
    ("m", _int),
    ("P", ctypes.POINTER(_CSC)),
    ("c", _ptr),
    ("A", ctypes.POINTER(_CSC)),
    ("b", _ptr),
    ("G", ctypes.POINTER(_CSC)),
    ("h_l", _ptr),
    ("h_u", _ptr),
    ("x_l", _ptr),
    ("x_u", _ptr),
  ]


_SETTINGS = [
  ("rho_init", _float),
  ("delta_init", _float),
  ("eps_abs", _float),
  ("eps_rel", _float),
  ("check_duality_gap", _int),
  ("eps_duality_gap_abs", _float),
  ("eps_duality_gap_rel", _float),
  ("infeasibility_threshold", _float),
  ("eps_primal_inf", _float),
  ("eps_dual_inf", _float),
  ("reg_lower_limit", _float),
  ("reg_finetune_lower_limit", _float),
  ("reg_finetune_primal_update_threshold", _int),
  ("reg_finetune_dual_update_threshold", _int),
  ("max_iter", _int),
  ("max_factor_retires", _int),
  ("preconditioner_scale_cost", _int),
  ("preconditioner_reuse_on_update", _int),
  ("preconditioner_iter", _int),
  ("tau", _float),
  ("kkt_solver", _int),
  ("num_threads", _int),
  ("iterative_refinement_always_enabled", _int),
  ("iterative_refinement_eps_abs", _float),
  ("iterative_refinement_eps_rel", _float),
  ("iterative_refinement_max_iter", _int),
  ("iterative_refinement_min_improvement_rate", _float),
  ("iterative_refinement_static_regularization_eps", _float),
  ("iterative_refinement_static_regularization_rel", _float),
  ("verbose", _int),
  ("compute_timings", _int),
]


class _Settings(ctypes.Structure):
  _fields_ = _SETTINGS


_INFO = [
  ("status", _int),
  ("iter", _int),
  *((name, _float) for name in ("rho", "delta", "mu", "sigma", "primal_step", "dual_step", "primal_res", "primal_res_rel", "dual_res")),
  *((name, _float) for name in ("dual_res_rel", "primal_res_reg", "primal_res_reg_rel", "dual_res_reg", "dual_res_reg_rel")),
  *((name, _float) for name in ("primal_prox_inf", "dual_prox_inf", "prev_primal_res", "prev_dual_res", "primal_obj", "dual_obj")),
  ("duality_gap", _float),
  ("duality_gap_rel", _float),
  ("factor_retires", _int),
  ("reg_limit", _float),
  ("no_primal_update", _int),
  ("no_dual_update", _int),
  *((name, _float) for name in ("setup_time", "update_time", "solve_time", "kkt_factor_time", "kkt_solve_time", "run_time")),
]


class _Info(ctypes.Structure):
  _fields_ = _INFO


_VARIABLES = ("x", "y", "z_l", "z_u", "z_bl", "z_bu", "s_l", "s_u", "s_bl", "s_bu")


class _Result(ctypes.Structure):
  _fields_ = [*((name, _ptr) for name in _VARIABLES), ("info", _Info)]


class _Workspace(ctypes.Structure):
  _fields_ = [("solver_handle", ctypes.c_void_p), ("is_dense", _int), ("n", _int), ("p", _int), ("m", _int), ("result", ctypes.POINTER(_Result))]


def _library() -> ctypes.CDLL:
  import scaly_piqp

  lib_dir = Path(scaly_piqp.__file__).parent / "lib"
  names = {"darwin": "libpiqpc.dylib", "win32": "piqpc.dll"}
  lib = ctypes.CDLL(str(lib_dir / names.get(sys.platform, "libpiqpc.so")))
  lib.piqp_set_default_settings_sparse.argtypes = [ctypes.POINTER(_Settings)]
  lib.piqp_setup_sparse.argtypes = [ctypes.POINTER(ctypes.POINTER(_Workspace)), ctypes.POINTER(_Data), ctypes.POINTER(_Settings)]
  lib.piqp_solve.argtypes = [ctypes.POINTER(_Workspace)]
  lib.piqp_solve.restype = _int
  lib.piqp_cleanup.argtypes = [ctypes.POINTER(_Workspace)]
  return lib


@dataclass(frozen=True, eq=False)
class ReferenceResult:
  """PIQP's solution, status and statistics, and its iteration table when it ran verbose."""

  status: Status
  variables: dict[str, np.ndarray]
  info: dict[str, float]
  log: list[dict[str, float]]


def _csc(matrix: Any, keep: list[Any]) -> Any:
  csc = sparse.csc_array(matrix)
  csc.sort_indices()
  p, i, x = (np.ascontiguousarray(csc.indptr, dtype=np.int32), np.ascontiguousarray(csc.indices, dtype=np.int32), np.ascontiguousarray(csc.data, dtype=np.float64))
  keep += [p, i, x]
  value = _CSC(csc.shape[0], csc.shape[1], csc.nnz, p.ctypes.data_as(ctypes.POINTER(_int)), i.ctypes.data_as(ctypes.POINTER(_int)), x.ctypes.data_as(_ptr))
  keep.append(value)
  return ctypes.pointer(value)


def _vector(value: Any, keep: list[Any]) -> Any:
  if value is None:
    return None
  array = np.ascontiguousarray(value, dtype=np.float64)
  keep.append(array)
  return array.ctypes.data_as(_ptr)


_ROW = re.compile(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$")
LOG_COLUMNS = ("iter", "primal_obj", "dual_obj", "duality_gap", "primal_res", "dual_res", "rho", "delta", "mu", "primal_step", "dual_step")


def parse_log(text: str) -> list[dict[str, float]]:
  """The rows of PIQP's iteration table."""
  rows = []
  for line in text.splitlines():
    if (match := _ROW.match(line)) is not None:
      rows.append(dict(zip(LOG_COLUMNS, map(float, match.groups()), strict=True)))
  return rows


def solve(
  P: Any,
  c: Any,
  A: Any = None,
  b: Any = None,
  G: Any = None,
  h_l: Any = None,
  h_u: Any = None,
  x_l: Any = None,
  x_u: Any = None,
  settings: Settings | None = None,
  kkt_solver: int = SPARSE_LDLT_COND,
  compute_timings: bool = False,
) -> ReferenceResult:
  """Set up and solve one QP with PIQP's sparse interface, by default with ``kkt_solver = sparse_ldlt_cond``.

  With ``compute_timings``, PIQP's own timers fill ``info``'s ``setup_time``, ``solve_time`` and
  ``run_time``, in seconds.
  """
  settings = settings or Settings()
  lib = _library()
  keep: list[Any] = []
  n = sparse.csc_array(P).shape[0]
  data = _Data(
    n,
    0 if A is None else sparse.csc_array(A).shape[0],
    0 if G is None else sparse.csc_array(G).shape[0],
    _csc(P, keep),
    _vector(c, keep),
    None if A is None else _csc(A, keep),
    _vector(b, keep),
    None if G is None else _csc(G, keep),
    _vector(h_l, keep),
    _vector(h_u, keep),
    _vector(x_l, keep),
    _vector(x_u, keep),
  )
  native = _Settings()
  lib.piqp_set_default_settings_sparse(ctypes.byref(native))
  for field in fields(settings):
    if field.name != "verbose":
      setattr(native, field.name, getattr(settings, field.name))
  native.kkt_solver = kkt_solver
  native.compute_timings = int(compute_timings)
  native.verbose = int(settings.verbose)
  workspace = ctypes.POINTER(_Workspace)()
  log = ""
  with _captured_stdout() as captured:
    lib.piqp_setup_sparse(ctypes.byref(workspace), ctypes.byref(data), ctypes.byref(native))
    status = Status(lib.piqp_solve(workspace))
  log = captured.read()
  result = workspace.contents.result.contents
  sizes = {"x": n, "y": data.p, "z_l": data.m, "z_u": data.m, "z_bl": n, "z_bu": n, "s_l": data.m, "s_u": data.m, "s_bl": n, "s_bu": n}
  variables = {name: np.ctypeslib.as_array(getattr(result, name), shape=(size,)).copy() if size else np.zeros(0) for name, size in sizes.items()}
  info = {name: getattr(result.info, name) for name, _ in _INFO}
  lib.piqp_cleanup(workspace)
  return ReferenceResult(status, variables, info, parse_log(log))


class _captured_stdout:
  """Redirect the C library's standard output to a temporary file, flushing around it."""

  def __enter__(self) -> _captured_stdout:
    sys.stdout.flush()
    self._libc = ctypes.CDLL(None)
    self._file = tempfile.TemporaryFile(mode="w+b")
    self._saved = os.dup(1)
    os.dup2(self._file.fileno(), 1)
    return self

  def __exit__(self, *exc: object) -> None:
    self._libc.fflush(None)
    os.dup2(self._saved, 1)
    os.close(self._saved)

  def read(self) -> str:
    self._file.seek(0)
    text = self._file.read().decode()
    self._file.close()
    return text
