"""Load PIQP and IPOPT shared libraries through ``ctypes``.

The actual discovery logic is centralized in :mod:`alloy.toolchain`; this module
keeps the small solver-facing API stable without importing toolchain at module
import time (so ``python -m alloy.toolchain`` can run without a runpy warning).
"""

from __future__ import annotations

import ctypes
from typing import Any


def _toolchain() -> Any:
  from alloy import toolchain

  return toolchain


class SolverLibraryError(RuntimeError):
  """Compatibility alias; actual errors are raised by ``alloy.toolchain``."""


def has_solver_library(stem: str) -> bool:
  if stem not in {"piqpc", "ipopt"}:
    return False
  return _toolchain().solver_library_loadable(stem)


_piqp_lib: ctypes.CDLL | None = None
_ipopt_lib: ctypes.CDLL | None = None


def piqp_lib() -> ctypes.CDLL:
  global _piqp_lib
  if _piqp_lib is None:
    _piqp_lib = _toolchain().load_solver_library("piqpc")
  return _piqp_lib


def ipopt_lib() -> ctypes.CDLL:
  global _ipopt_lib
  if _ipopt_lib is None:
    _ipopt_lib = _toolchain().load_solver_library("ipopt")
  return _ipopt_lib
