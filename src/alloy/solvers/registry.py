"""Solver plugin discovery and protocol validation."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from functools import cache
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Protocol

ORACLE_PROTOCOL_VERSION = 1
ENTRY_POINT_GROUP = "alloy.solvers"


class SolverPluginError(RuntimeError):
  pass


class SolverBackend(Protocol):
  """Solver plugin metadata: which problem family it solves and where its
  vendored native library/headers live. Solves themselves always run through
  the generated C wrapper (``codegen/solver_c``); plugins ship no Python
  solve path."""

  name: str
  kind: str  # "qp" | "nlp" - which descriptor family the plugin solves
  protocol_version: int
  lib_stem: str
  link_flags: tuple[str, ...]
  header: str

  def lib_dir(self) -> Path: ...

  def include_dir(self) -> Path: ...


def available_backends() -> dict[str, EntryPoint]:
  return {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUP)}


def _missing_backend(name: str, available: Sequence[str]) -> SolverPluginError:
  return SolverPluginError(f"no solver plugin {name!r} installed; available: {list(available)}. Install alloy-{name} (uv add alloy-{name})")


@cache
def get_backend(name: str) -> SolverBackend:
  eps = available_backends()
  if name not in eps:
    raise _missing_backend(name, sorted(eps))
  try:
    backend: SolverBackend = eps[name].load()
    version = backend.protocol_version
  except SolverPluginError:
    raise
  except Exception as exc:
    raise SolverPluginError(f"solver plugin {name!r} failed to load: {exc}") from exc
  if version != ORACLE_PROTOCOL_VERSION:
    raise SolverPluginError(f"solver plugin {name!r} protocol version {version} does not match Alloy protocol version {ORACLE_PROTOCOL_VERSION}")
  return backend


def require_backend(name: str, kind: str) -> SolverBackend:
  backend = get_backend(name)
  if backend.kind != kind:
    raise SolverPluginError(f"solver plugin {name!r} solves {backend.kind} problems, not {kind}")
  return backend


def installed_backend_paths() -> list[tuple[Path, Path]]:
  out: list[tuple[Path, Path]] = []
  for name in sorted(available_backends()):
    try:
      backend = get_backend(name)
      out.append((backend.include_dir(), backend.lib_dir()))
    except Exception as exc:  # noqa: BLE001 - one broken plugin must not hide the others
      warnings.warn(f"could not inspect solver plugin {name!r}: {exc}", RuntimeWarning, stacklevel=2)
  return out
