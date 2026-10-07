from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from scaly_piqp import include_dir, lib_dir

from .external import external_nlp

if TYPE_CHECKING:
  from scaly.codegen.solver import SolverWrapperCtx
  from scaly.function.concrete import ConcreteFunction


def _option_map(options: dict[str, Any]) -> dict[str, Any]:
  out = dict(options)
  for ignored in ("print_level", "sb", "warm_start_init_point"):
    out.pop(ignored, None)
  allowed = {
    "dual_tol",
    "globalization",
    "hessian",
    "line_search_beta",
    "max_iter",
    "merit",
    "qp",
    "qp_max_iter",
    "qp_tol",
    "regularization",
    "tol",
    "trace",
    "watchdog",
  }
  unknown = sorted(set(out) - allowed)
  if unknown:
    raise NotImplementedError(f"scaly-sqp options are not supported: {unknown}")
  return out


def _positive_int(options: dict[str, Any], name: str, default: int) -> int:
  value = options.get(name, default)
  if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
    raise ValueError(f"scaly-sqp {name} must be a positive integer, got {value!r}")
  return value


def _nonnegative_int(options: dict[str, Any], name: str, default: int) -> int:
  value = options.get(name, default)
  if isinstance(value, bool) or not isinstance(value, int) or value < 0:
    raise ValueError(f"scaly-sqp {name} must be a non-negative integer, got {value!r}")
  return value


def _positive_float(options: dict[str, Any], name: str, default: float) -> float:
  value = options.get(name, default)
  if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
    raise ValueError(f"scaly-sqp {name} must be finite and positive, got {value!r}")
  return float(value)


def _prepare_options(options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
  opts = _option_map(options)
  max_iter = _positive_int(opts, "max_iter", 50)
  tol = _positive_float(opts, "tol", 1e-6)
  dual_tol = _positive_float(opts, "dual_tol", 1e-4)
  beta = _positive_float(opts, "line_search_beta", 0.7)
  merit_offset = _positive_float(opts, "merit", 10.0)
  regularization = _positive_float(opts, "regularization", 1e-6)
  qp_max_iter = _positive_int(opts, "qp_max_iter", 50)
  qp_tol = _positive_float(opts, "qp_tol", 1e-6)
  watchdog = _nonnegative_int(opts, "watchdog", 0)
  if beta >= 1.0:
    raise ValueError(f"scaly-sqp line_search_beta must be below 1, got {beta!r}")
  globalization = opts.get("globalization", "filter")
  if globalization not in ("filter", "l1"):
    raise ValueError(f"scaly-sqp globalization must be 'filter' or 'l1', got {globalization!r}")
  if globalization == "filter" and "watchdog" in opts:
    raise ValueError("scaly-sqp watchdog applies only to globalization='l1'")
  hessian_mode = opts.get("hessian", "exact")
  if hessian_mode not in ("exact", "objective"):
    raise ValueError(f"scaly-sqp hessian must be 'exact' or 'objective', got {hessian_mode!r}")
  qp_mode = opts.get("qp", "sparse")
  if qp_mode not in ("sparse", "dense"):
    raise ValueError(f"scaly-sqp qp must be 'sparse' or 'dense', got {qp_mode!r}")
  trace = opts.get("trace", False)
  if not isinstance(trace, bool):
    raise ValueError(f"scaly-sqp trace must be a bool, got {trace!r}")
  return {"qp": qp_mode}, dict(
    max_iter=max_iter,
    tol=tol,
    dual_tol=dual_tol,
    line_search_beta=beta,
    merit=merit_offset,
    regularization=regularization,
    qp_max_iter=qp_max_iter,
    qp_tol=qp_tol,
    watchdog=watchdog,
    globalization=int(globalization == "l1"),
    hessian=int(hessian_mode == "exact"),
    trace=trace,
  )


class _Backend:
  name = "sqp"
  kind = "nlp"
  hess_triangle = "upper"
  protocol_version = 8
  lib_stem = "piqpc"
  link_flags = ("-lpiqpc",)
  header = "piqp/piqp.h"

  include_dir = staticmethod(include_dir)
  lib_dir = staticmethod(lib_dir)

  prepare_options = staticmethod(_prepare_options)

  def render_wrapper(self, fun: ConcreteFunction, ctx: SolverWrapperCtx) -> list[str]:
    from .codegen import render_wrapper

    return render_wrapper(fun, ctx)


BACKEND = _Backend()

__all__ = ["BACKEND", "external_nlp"]
