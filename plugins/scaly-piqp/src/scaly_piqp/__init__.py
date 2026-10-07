from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from scaly.codegen.solver import SolverWrapperCtx
  from scaly.function.concrete import ConcreteFunction


_SETTINGS = frozenset(
  {
    "rho_init",
    "delta_init",
    "eps_abs",
    "eps_rel",
    "check_duality_gap",
    "eps_duality_gap_abs",
    "eps_duality_gap_rel",
    "infeasibility_threshold",
    "reg_lower_limit",
    "reg_finetune_lower_limit",
    "reg_finetune_primal_update_threshold",
    "reg_finetune_dual_update_threshold",
    "max_iter",
    "max_factor_retires",
    "preconditioner_scale_cost",
    "preconditioner_reuse_on_update",
    "preconditioner_iter",
    "tau",
    "kkt_solver",
    "iterative_refinement_always_enabled",
    "iterative_refinement_eps_abs",
    "iterative_refinement_eps_rel",
    "iterative_refinement_max_iter",
    "iterative_refinement_min_improvement_rate",
    "iterative_refinement_static_regularization_eps",
    "iterative_refinement_static_regularization_rel",
    "verbose",
    "compute_timings",
  }
)


_INTEGER_SETTINGS = frozenset(
  {
    "check_duality_gap",
    "reg_finetune_primal_update_threshold",
    "reg_finetune_dual_update_threshold",
    "max_iter",
    "max_factor_retires",
    "preconditioner_scale_cost",
    "preconditioner_reuse_on_update",
    "preconditioner_iter",
    "kkt_solver",
    "iterative_refinement_always_enabled",
    "iterative_refinement_max_iter",
    "verbose",
    "compute_timings",
  }
)


def include_dir() -> Path:
  return Path(__file__).resolve().parent / "include"


def lib_dir() -> Path:
  return Path(__file__).resolve().parent / "lib"


class _Backend:
  """PIQP solver plugin: vendored lib/header metadata + the C wrapper
  template (``codegen.py``). Solves run through scaly's generated C wrapper."""

  name = "piqp"
  kind = "qp"
  protocol_version = 8
  lib_stem = "piqpc"
  link_flags = ("-lpiqpc",)
  header = "piqp/piqp.h"

  include_dir = staticmethod(include_dir)
  lib_dir = staticmethod(lib_dir)

  def validate_options(self, options: Mapping[str, object]) -> None:
    for key in options:
      if key != "sparse" and key not in _SETTINGS:
        raise ValueError(f"Unknown PIQP option {key!r}. Supported settings: {', '.join(sorted(_SETTINGS))}")

  def prepare_options(self, options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    self.validate_options(options)
    resolved = {"verbose": 0, "sparse": False, **options}
    for key, value in resolved.items():
      if key == "sparse":
        if not isinstance(value, bool):
          raise TypeError("PIQP option 'sparse' must be a bool")
      elif key in _INTEGER_SETTINGS:
        if not isinstance(value, int):
          raise TypeError(f"PIQP option {key!r} must be an integer")
      elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"PIQP option {key!r} must be a number")
    return {"sparse": resolved.pop("sparse")}, resolved

  def render_wrapper(self, fun: ConcreteFunction, ctx: SolverWrapperCtx) -> list[str]:
    from .codegen import render_wrapper

    return render_wrapper(fun, ctx)


BACKEND = _Backend()
