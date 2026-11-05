"""Process environment and platform facts scaly reads.

A leaf: it imports nothing from scaly, so the pieces that need an environment variable or a
shared-library suffix — AD, solver-library discovery, the JIT — can share one definition without
depending on each other.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


class ToolchainError(RuntimeError):
  """Raised when native compiler/header/library discovery fails."""


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


def shared_lib_ext() -> str:
  if sys.platform == "darwin":
    return ".dylib"
  if sys.platform == "win32":
    return ".dll"
  return ".so"


def shared_lib_flag() -> str:
  return "-dynamiclib" if sys.platform == "darwin" else "-shared"
