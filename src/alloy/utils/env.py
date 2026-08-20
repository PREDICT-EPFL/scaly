"""Process environment and platform facts alloy reads.

A leaf: it imports nothing from alloy, so the pieces that need an environment variable or a
shared-library suffix — AD, solver-library discovery, the JIT — can share one definition without
depending on each other. Keep the variable names and defaults registered here so the JIT, solver
loading, tests, and diagnostics agree on one source of truth.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


class ToolchainError(RuntimeError):
  """Raised when native compiler/header/library discovery fails."""


@dataclass(frozen=True, slots=True)
class EnvVar:
  name: str
  default: str | None
  help: str


ENV_VARS: tuple[EnvVar, ...] = (
  EnvVar("ALLOY_CACHE_DIR", None, "Override the JIT cache root."),
  EnvVar("ALLOY_CC", None, "Override the C compiler used by the JIT."),
  EnvVar("ALLOY_SOLVER_INCLUDE_DIR", None, "Override the vendored solver C header directory."),
  EnvVar("ALLOY_SOLVER_LIB_DIR", None, "Override the vendored solver shared-library directory."),
  EnvVar("ALLOY_<NAME>_LIB", None, "Exact path to an installed solver plugin's shared library (e.g. ALLOY_PIQP_LIB, ALLOY_IPOPT_LIB)."),
  EnvVar("ALLOY_SOLVER_SYSTEM_FALLBACK", "0", "Experimental: allow ctypes/pkg-config/default-linker system solver fallback."),
  EnvVar("ALLOY_BUILD_SOLVERS", "auto", "Build-hook solver mode: auto, skip, or required/1/true."),
  EnvVar("ALLOY_STRICT_JVP_MANY", "0", "Raise instead of using the unrolled multi-seed JVP fallback."),
  EnvVar("ALLOY_VIZ_DIR", None, "Visualization recording directory."),
  EnvVar("ALLOY_TRACKING_SWEEP", "0", "Run the opt-in tracking sparse-Jacobian sweep."),
)


def env(name: str, default: str | None = None) -> str | None:
  return os.environ.get(name, default)


def env_bool(name: str, default: bool = False) -> bool:
  raw = os.environ.get(name)
  if raw is None or raw == "":
    return default
  val = raw.strip().lower()
  if val in {"1", "true", "yes", "on"}:
    return True
  if val in {"0", "false", "no", "off"}:
    return False
  raise ToolchainError(f"{name} must be a boolean (1/0, true/false, yes/no, on/off), got {raw!r}")


def env_path(name: str) -> Path | None:
  raw = os.environ.get(name)
  return Path(raw).expanduser() if raw else None


def alloy_env_vars() -> tuple[EnvVar, ...]:
  return ENV_VARS


def shared_lib_ext() -> str:
  if sys.platform == "darwin":
    return ".dylib"
  if sys.platform == "win32":
    return ".dll"
  return ".so"


def shared_lib_flag() -> str:
  return "-dynamiclib" if sys.platform == "darwin" else "-shared"
