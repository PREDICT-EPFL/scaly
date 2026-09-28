from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from scaly.ext import require_ext_api
from scaly.opt.external import External

require_ext_api(1, "scaly-piqp")

if TYPE_CHECKING:
  from scaly.function import ConcreteFunction
  from scaly.opt.external import SolverWrapperCtx


def include_dir() -> Path:
  return Path(__file__).resolve().parent / "include"


def lib_dir() -> Path:
  return Path(__file__).resolve().parent / "lib"


@dataclass(frozen=True)
class PIQP(External):
  """PIQP, the proximal interior-point QP solver, vendored and called from the generated C wrapper
  (``codegen.py``). ``sparse`` passes the matrices in their structural patterns to PIQP's sparse
  interface instead of dense to its dense one; ``options`` are PIQP's settings by name
  (``eps_abs``, ``max_iter``, ``verbose``, ...), numbers or booleans."""

  name = "opt.piqp"
  kind = "qp"
  lib_stem = "piqpc"
  link_flags = ("-lpiqpc",)
  header = "piqp/piqp.h"

  sparse: bool = False

  include_dir = staticmethod(include_dir)
  lib_dir = staticmethod(lib_dir)

  def check_options(self) -> None:
    for key, val in self.options.items():
      if not isinstance(val, (bool, int, float)):
        raise TypeError(f"PIQP setting {key}={val!r} must be a number or a bool")

  def render_wrapper(self, fun: ConcreteFunction, ctx: SolverWrapperCtx) -> list[str]:
    from .codegen import render_wrapper

    return render_wrapper(fun, ctx)


__all__ = ["PIQP", "include_dir", "lib_dir"]
