"""The extern callee of a solver: the plugin-rendered wrapper, framed with the stats storage and its
accessor, and what a translation unit holding one needs to compile and load.

A solver Function's C body is a template per backend, parameterized by the ``SolverDescriptor``.
The templates live in the solver plugins (``scaly_piqp.codegen``, ``scaly_ipopt.codegen``, ...):
the plugin's ``SolverBackend.render_wrapper`` hook gets a :class:`SolverWrapperCtx`, and the body it
returns is framed here with the stats static and the exported ``<symbol>_stats`` accessor. The
oracle Functions the template drives lower through Program IR like any other Function and are
rendered as ``<oracle>_raw``. The compiler reaches all of this through the extern-callee protocol
(``scaly.function.extern``); see ``docs/dev/solver_plugins.md`` for the plugin contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..function.extern import BuildRequirements, ExternRenderCtx, ExternState
from .paths import backend_compile_flags
from . import registry
from .stats import SCALY_SOLVER_STATS_VERSION, CSolverStats, SolverStats, stats_c_defs, stats_c_timing_defs

if TYPE_CHECKING:
  from ..function import ConcreteFunction, Function
  from .model import ExternalOracle, SolverDescriptor


@dataclass(frozen=True, slots=True)
class SolverWrapperCtx:
  """Codegen kit handed to a plugin's ``render_wrapper`` hook.

  ``symbol`` is the solver's mangled C identifier (prefix for any static the template declares),
  ``raw_symbol`` the function the template must define, ``stats_symbol`` the
  ``scaly_solver_stats`` static it must fill (declared here, one per solver). ``raw_symbol_of``
  resolves the C symbol of an oracle or derivative Function, or of an ``ExternalOracle``.
  """

  symbol: str
  raw_symbol: str
  stats_symbol: str

  def raw_symbol_of(self, fun: ConcreteFunction | ExternalOracle) -> str:
    return ExternRenderCtx.raw_symbol_of(fun)


def render_solver(fun: ConcreteFunction, desc: SolverDescriptor, ctx: ExternRenderCtx) -> list[str]:
  """The stats static, the plugin's wrapper body, and the exported ``<symbol>_stats`` accessor."""
  wrapper = SolverWrapperCtx(symbol=ctx.symbol, raw_symbol=ctx.raw_symbol, stats_symbol=f"{ctx.symbol}_stats_data")
  body = registry.get_backend(desc.backend).render_wrapper(fun, wrapper)
  return [
    f"static scaly_solver_stats {wrapper.stats_symbol};",
    *body,
    "",
    f"int {ctx.symbol}_stats(scaly_solver_stats* out) {{",
    "  if (!out) return 1;",
    f"  *out = {wrapper.stats_symbol};",
    "  return 0;",
    "}",
  ]


def solver_requirements(symbol: str, desc: SolverDescriptor) -> BuildRequirements:
  """The backend's header, the stats ABI in header and source, the accessor's prototype, and the
  vendored library, isolated in its own linker namespace on Linux."""
  return BuildRequirements(
    includes=("#include <time.h>", f'#include "{registry.get_backend(desc.backend).header}"'),
    header_types=(tuple(stats_c_defs()),),
    source_blocks=(tuple(stats_c_defs()), tuple(stats_c_timing_defs())),
    declarations=(f"int {symbol}_stats(scaly_solver_stats* out);",),
    libraries=(desc.backend,),
    link_flags=backend_compile_flags,
    isolated=True,
  )


def _decode_stats(name: str, raw: Any) -> SolverStats:
  if raw.version == 0:
    raise ValueError(f"solver {name!r} has not run yet (stats version is 0)")
  if raw.version != SCALY_SOLVER_STATS_VERSION:
    raise ValueError(f"solver stats ABI mismatch for {name!r}: artifact version {raw.version}, expected {SCALY_SOLVER_STATS_VERSION}")
  return SolverStats.from_c(raw)


def solver_state(symbol: str, name: str) -> ExternState:
  """The ``<symbol>_stats`` accessor, read as a ``SolverStats``."""
  return ExternState(accessor=f"{symbol}_stats", ctype=CSolverStats, decode=lambda raw: _decode_stats(name, raw))


def solver_stats(fun: Function, name: str | None = None) -> SolverStats:
  """The statistics of the latest solve by the solver ``name`` that ``fun`` reaches (``fun`` itself
  when it is the solver); ``name`` may be left out when there is one. ``fun`` must have run."""
  stats = fun.callee_state(name)
  if not isinstance(stats, SolverStats):
    raise TypeError(f"the extern callee {name or fun.name!r} is not a solver: its state is a {type(stats).__name__}")
  return stats
