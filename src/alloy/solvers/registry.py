"""Solver plugin discovery and protocol validation.

The plugin contract (see ``docs/dev/solver_plugins.md``) has two halves:

- **packaging metadata** — where the vendored native library and C headers
  live, and how to link them (``lib_stem``, ``link_flags``, ``header``,
  ``lib_dir()``, ``include_dir()``);
- **codegen** — ``render_wrapper(fun, ctx)``, the C template that drives the
  solver's C API from the generated translation unit. Core owns the ABI
  around it (the ``_raw`` calling convention, the ``alloy_solver_stats``
  struct, the oracle kernels); the plugin owns everything solver-specific.

``SOLVER_PLUGIN_PROTOCOL_VERSION`` covers both halves: it is bumped whenever
the oracle conventions, the ``SolverWrapperCtx`` surface, or the stats ABI
handed to plugins change, and every plugin must declare the version it was
written against.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from functools import cache
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast, overload

if TYPE_CHECKING:
  from alloy.codegen.solver import SolverWrapperCtx
  from alloy.function import Function

SOLVER_PLUGIN_PROTOCOL_VERSION = 6
ENTRY_POINT_GROUP = "alloy.solvers"


class SolverPluginError(RuntimeError):
  pass


class SolverBackend(Protocol):
  """The full solver plugin protocol: packaging metadata plus the C wrapper
  template. Solves always run through the generated C wrapper — plugins ship
  no Python solve path."""

  name: str
  kind: str  # "qp" | "nlp" - which descriptor family the plugin solves
  protocol_version: int
  lib_stem: str  # shared library stem: lib<stem>.{dylib,so}
  link_flags: tuple[str, ...]
  header: str  # C header path relative to include_dir(), e.g. "piqp/piqp.h"

  def lib_dir(self) -> Path: ...

  def include_dir(self) -> Path: ...

  def render_wrapper(self, fun: Function, ctx: SolverWrapperCtx) -> list[str]:
    """Emit the C wrapper for one solver ``Function`` (see docs/dev/solver_plugins.md).

    Must define ``static void <ctx.raw_symbol>(...)`` with the descriptor's
    ``in*``/``out*`` signature plus a trailing ``double* w``, drive the solver's
    C API with the oracle kernels (``ctx.raw_symbol_of``), and fill
    ``ctx.stats_symbol`` on every call.
    """
    ...


class NlpSolverBackend(SolverBackend, Protocol):
  """Solver backend protocol for NLP plugins, including Hessian layout."""

  hess_triangle: Literal["lower", "upper"]


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
  if version != SOLVER_PLUGIN_PROTOCOL_VERSION:
    raise SolverPluginError(
      f"solver plugin {name!r} protocol version {version} does not match Alloy protocol version {SOLVER_PLUGIN_PROTOCOL_VERSION}"
    )
  if getattr(backend, "name", None) != name:
    raise SolverPluginError(f"solver plugin {name!r} declares name {getattr(backend, 'name', None)!r}; it must equal the entry-point name")
  if not callable(getattr(backend, "render_wrapper", None)):
    raise SolverPluginError(f"solver plugin {name!r} does not provide a callable render_wrapper(fun, ctx) hook")
  return backend


@overload
def require_backend(name: str, kind: Literal["nlp"]) -> NlpSolverBackend: ...


@overload
def require_backend(name: str, kind: Literal["qp"]) -> SolverBackend: ...


def require_backend(name: str, kind: str) -> SolverBackend:
  backend = get_backend(name)
  if backend.kind != kind:
    raise SolverPluginError(f"solver plugin {name!r} solves {backend.kind} problems, not {kind}")
  if kind == "nlp":
    if getattr(backend, "hess_triangle", None) not in {"lower", "upper"}:
      raise SolverPluginError(f"solver plugin {name!r} must declare hess_triangle as 'lower' or 'upper'")
    return cast(NlpSolverBackend, backend)
  return backend


def loaded_backends() -> dict[str, SolverBackend]:
  """Every installed backend that loads cleanly, by name. Broken plugins warn
  and are skipped so one bad install cannot hide the others."""
  out: dict[str, SolverBackend] = {}
  for name in sorted(available_backends()):
    try:
      out[name] = get_backend(name)
    except Exception as exc:  # noqa: BLE001 - one broken plugin must not hide the others
      warnings.warn(f"could not load solver plugin {name!r}: {exc}", RuntimeWarning, stacklevel=2)
  return out


def installed_backend_paths() -> list[tuple[Path, Path]]:
  out: list[tuple[Path, Path]] = []
  for name, backend in loaded_backends().items():
    try:
      out.append((backend.include_dir(), backend.lib_dir()))
    except Exception as exc:  # noqa: BLE001 - one broken plugin must not hide the others
      warnings.warn(f"could not inspect solver plugin {name!r}: {exc}", RuntimeWarning, stacklevel=2)
  return out
