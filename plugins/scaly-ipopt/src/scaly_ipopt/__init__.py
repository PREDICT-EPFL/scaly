from __future__ import annotations

import json
import re
from pathlib import Path
import platform
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from scaly.codegen.solver import SolverWrapperCtx
  from scaly.function.concrete import ConcreteFunction

_RAW_BUILD_CONFIG = json.loads((Path(__file__).resolve().parent / "build_config.json").read_text())
BUILD_CONFIG = {name: config for name, config in _RAW_BUILD_CONFIG.items() if name != "blas"}
BUILD_CONFIG["blas"] = _RAW_BUILD_CONFIG["blas"]["darwin" if platform.system() == "Darwin" else "linux"]


def include_dir() -> Path:
  return Path(__file__).resolve().parent / "include"


def lib_dir() -> Path:
  return Path(__file__).resolve().parent / "lib"


class _Backend:
  """IPOPT solver plugin: vendored lib/header metadata + the C wrapper
  template (``codegen.py``). Solves run through scaly's generated C wrapper."""

  name = "ipopt"
  kind = "nlp"
  hess_triangle = "lower"
  protocol_version = 8
  lib_stem = "ipopt"
  link_flags = ("-lipopt",)
  header = "coin-or/IpStdCInterface.h"
  build_config = BUILD_CONFIG

  include_dir = staticmethod(include_dir)
  lib_dir = staticmethod(lib_dir)

  def prepare_options(self, options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = {"print_level": 0, **options}
    token = re.compile(r"[A-Za-z0-9_./+-]+\Z")
    for key, value in resolved.items():
      if not isinstance(key, str) or not token.fullmatch(key):
        raise ValueError(f"Invalid IPOPT option name {key!r}")
      if not isinstance(value, (str, int, float)):
        raise TypeError(f"IPOPT option {key!r} must be a string, integer or float")
      if isinstance(value, str) and not token.fullmatch(value):
        raise ValueError(f"Invalid IPOPT option {key}={value!r}")
    return {}, resolved

  def render_wrapper(self, fun: ConcreteFunction, ctx: SolverWrapperCtx) -> list[str]:
    from .codegen import render_wrapper

    return render_wrapper(fun, ctx)


BACKEND = _Backend()
