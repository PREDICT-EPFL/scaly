"""Load the vendored PIQP and IPOPT shared libraries through ``ctypes``."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
_LIB_DIR = _PKG_ROOT / "lib"


def _lib_ext() -> str:
  return ".dylib" if sys.platform == "darwin" else ".so"


class SolverLibraryError(RuntimeError):
  """Raised when a vendored solver library cannot be located or loaded."""


def _load(stem: str) -> ctypes.CDLL:
  path = _LIB_DIR / f"lib{stem}{_lib_ext()}"
  if not path.exists():
    raise SolverLibraryError(f"vendored library {path.name!r} not found under {_LIB_DIR}. Run `uv sync` to trigger the build hook.")
  return ctypes.CDLL(str(path))


_piqp_lib: ctypes.CDLL | None = None
_ipopt_lib: ctypes.CDLL | None = None


def piqp_lib() -> ctypes.CDLL:
  global _piqp_lib
  if _piqp_lib is None:
    _piqp_lib = _load("piqpc")
  return _piqp_lib


def ipopt_lib() -> ctypes.CDLL:
  global _ipopt_lib
  if _ipopt_lib is None:
    _ipopt_lib = _load("ipopt")
  return _ipopt_lib
