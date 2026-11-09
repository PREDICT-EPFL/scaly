"""Fatrop backend metadata, options and paths to its vendored native library."""

import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from scaly.codegen.solver import SolverWrapperCtx
  from scaly.function.concrete import ConcreteFunction


def include_dir() -> Path:
  """Return the directory containing Fatrop's C interface and BLASFEO headers."""
  return Path(__file__).resolve().parent / "include"


def lib_dir() -> Path:
  """Return the directory containing the Fatrop shared library."""
  return Path(__file__).resolve().parent / "lib"


class _Backend:
  name = "fatrop"
  kind = "nlp"
  hess_triangle = "lower"
  stage_oracles = True
  protocol_version = 8
  lib_stem = "fatrop"
  link_flags = ("-lfatrop",) + (("-DFATROP_SYMBOL_EXPORT=__declspec(dllimport)",) if sys.platform == "win32" else ())
  header = "fatrop/ocp/OCPCInterface.h"
  include_dir = staticmethod(include_dir)
  lib_dir = staticmethod(lib_dir)

  def prepare_options(self, options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = {"print_level": 0, "tol": 1e-8, **options}
    for key, value in resolved.items():
      if key in {"print_level", "max_iter", "acceptable_iter"}:
        if type(value) is not int:
          raise TypeError(f"Fatrop option {key!r} must be an integer")
        if not 0 <= value <= 2**31 - 1:
          raise ValueError(f"Fatrop option {key!r} must be a nonnegative C integer")
      elif key in {"tol", "constr_viol_tol", "compl_inf_tol", "tol_acceptable", "mu_init"}:
        if type(value) not in (float, int):
          raise TypeError(f"Fatrop option {key!r} must be a number")
        if not math.isfinite(value) or value <= 0:
          raise ValueError(f"Fatrop option {key!r} must be finite and positive")
        resolved[key] = float(value)
      elif key == "linsol_it_ref":
        if type(value) is not bool:
          raise TypeError(f"Fatrop option {key!r} must be a boolean")
      else:
        raise ValueError(f"Unknown Fatrop option {key!r}")
    return {}, resolved

  def render_wrapper(self, fun: ConcreteFunction, ctx: SolverWrapperCtx) -> list[str]:
    from .codegen import render_wrapper

    return render_wrapper(fun, ctx)


BACKEND = _Backend()
