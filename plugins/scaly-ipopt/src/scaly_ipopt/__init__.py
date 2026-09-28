from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import platform
from typing import TYPE_CHECKING

from scaly.ext import require_ext_api
from scaly.opt.external import External

require_ext_api(1, "scaly-ipopt")

if TYPE_CHECKING:
  from scaly.opt.external import SolverWrapperCtx
  from scaly.function import ConcreteFunction

_RAW_BUILD_CONFIG = json.loads((Path(__file__).resolve().parent / "build_config.json").read_text())
BUILD_CONFIG = {name: config for name, config in _RAW_BUILD_CONFIG.items() if name != "blas"}
BUILD_CONFIG["blas"] = _RAW_BUILD_CONFIG["blas"]["darwin" if platform.system() == "Darwin" else "linux"]


def include_dir() -> Path:
  return Path(__file__).resolve().parent / "include"


def lib_dir() -> Path:
  return Path(__file__).resolve().parent / "lib"


@dataclass(frozen=True)
class IPOPT(External):
  """IPOPT, the interior-point NLP solver, vendored (with MUMPS) and called from the generated C
  wrapper (``codegen.py``). ``options`` are IPOPT's options by name, as its C interface takes them."""

  name = "opt.ipopt"
  kind = "nlp"
  hess_triangle = "lower"
  lib_stem = "ipopt"
  link_flags = ("-lipopt",)
  header = "coin-or/IpStdCInterface.h"
  build_config = BUILD_CONFIG

  include_dir = staticmethod(include_dir)
  lib_dir = staticmethod(lib_dir)

  def render_wrapper(self, fun: ConcreteFunction, ctx: SolverWrapperCtx) -> list[str]:
    from .codegen import render_wrapper

    return render_wrapper(fun, ctx)


__all__ = ["BUILD_CONFIG", "IPOPT", "include_dir", "lib_dir"]
