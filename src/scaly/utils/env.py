"""Process environment and platform facts scaly reads.

A leaf: it imports nothing from scaly, so the pieces that need an environment variable or a
shared-library suffix — AD, solver-library discovery, the JIT — can share one definition without
depending on each other. The compiler's variables are registered here; a package that reads its
own (the solvers) lists them through the ``scaly.env_vars`` entry points, and
``scaly_env_vars`` joins them, so the JIT, tests and diagnostics agree on one list.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path


class ToolchainError(RuntimeError):
  """Raised when native compiler/header/library discovery fails."""


@dataclass(frozen=True, slots=True)
class EnvVar:
  name: str
  default: str | None
  help: str


ENV_VARS: tuple[EnvVar, ...] = (
  EnvVar("SCALY_CACHE_DIR", None, "Override the JIT cache root."),
  EnvVar("SCALY_CC", None, "Override the C compiler used by the JIT."),
  EnvVar("SCALY_CC_OPT", "-O2", "Optimization flag the JIT passes to the C compiler."),
  EnvVar("SCALY_STRICT_JVP_MANY", "0", "Raise instead of using the unrolled multi-seed JVP fallback."),
  EnvVar("SCALY_VIZ_DIR", None, "Visualization recording directory."),
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


ENV_VAR_ENTRY_POINTS = "scaly.env_vars"
"""The entry-point group through which a package lists the environment variables it reads."""


def scaly_env_vars() -> tuple[EnvVar, ...]:
  """Every environment variable the installed packages read: the compiler's, then each package's."""
  extra = [var for ep in sorted(entry_points(group=ENV_VAR_ENTRY_POINTS), key=lambda ep: ep.name) for var in ep.load()]
  return (*ENV_VARS, *extra)


def shared_lib_ext() -> str:
  if sys.platform == "darwin":
    return ".dylib"
  if sys.platform == "win32":
    return ".dll"
  return ".so"


def shared_lib_flag() -> str:
  return "-dynamiclib" if sys.platform == "darwin" else "-shared"
