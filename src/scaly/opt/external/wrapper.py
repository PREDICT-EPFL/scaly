"""The extern callee of a solver: the plugin-rendered wrapper, framed with the stats storage and its
accessor, and what a translation unit holding one needs to compile and load.

A solver Function's C body is a template per backend, parameterized by the ``SolverDescriptor``.
The templates live in the solver plugins (``scaly_piqp.codegen``, ``scaly_ipopt.codegen``, ...):
the plugin's ``External.render_wrapper`` hook gets a :class:`SolverWrapperCtx`, and the body it
returns is framed here with the stats static and the exported ``<symbol>_stats`` accessor. The
oracle Functions the template drives lower through Program IR like any other Function and are
rendered as ``<oracle>_raw``. The compiler reaches all of this through the extern-callee protocol
(``scaly.function.extern``); see ``docs/dev/solver_plugins.md`` for the plugin contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...function.extern import BuildRequirements, ExternRenderCtx, ExternState
from ..method import REGISTRY, external_method
from .paths import backend_compile_flags
from .stats import SCALY_SOLVER_STATS_VERSION, CSolverStats, SolverStats, stats_c_defs, stats_c_timing_defs

if TYPE_CHECKING:
  from ...function import ConcreteFunction, Function
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


# The ``opt.Info`` outputs after the solution, in order, and the statistics field each copies.
INFO_FIELDS = ("status", "iter", "obj", "primal_viol")


def render_solver(fun: ConcreteFunction, desc: SolverDescriptor, ctx: ExternRenderCtx) -> list[str]:
  """The stats static, the plugin's wrapper body, the frame that calls it and writes the ``Info``
  outputs from the statistics it filled, and the exported ``<symbol>_stats`` accessor.

  The plugin defines ``<raw>_solve`` over the descriptor's inputs and solution outputs; ``<raw>``,
  what the generated code calls, takes the ``Info`` outputs after them. A QP's objective constant,
  which the solver never sees, is added to the objective here, so both report the problem's own."""
  wrapper = SolverWrapperCtx(symbol=ctx.symbol, raw_symbol=f"{ctx.raw_symbol}_solve", stats_symbol=f"{ctx.symbol}_stats_data")
  body = _method(desc).render_wrapper(fun, wrapper)
  n_in, n_out = len(desc.input_signature), len(desc.output_signature)
  params = [*(f"const double* in{i}" for i in range(n_in)), *(f"double* out{i}" for i in range(n_out + len(INFO_FIELDS))), "double* w"]
  args = [*(f"in{i}" for i in range(n_in)), *(f"out{i}" for i in range(n_out)), "w"]
  constant = []
  if desc.objective_constant is not None:
    param_args = [f"in{n_in - len(desc.param_names) + i}" for i in range(len(desc.param_names))]
    constant = [
      "  double objective_constant;",
      f"  {ctx.raw_symbol_of(desc.objective_constant)}({', '.join([*param_args, '&objective_constant', 'w'])});",
      f"  {wrapper.stats_symbol}.obj += objective_constant;",
    ]
  return [
    f"static scaly_solver_stats {wrapper.stats_symbol};",
    *body,
    "",
    f"static void {ctx.raw_symbol}({', '.join(params)}) {{",
    f"  {wrapper.raw_symbol}({', '.join(args)});",
    *constant,
    *(f"  out{n_out + k}[0] = (double){wrapper.stats_symbol}.{field};" for k, field in enumerate(INFO_FIELDS)),
    "}",
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
    includes=("#include <time.h>", f'#include "{_method(desc).header}"'),
    header_types=(tuple(stats_c_defs()),),
    source_blocks=(tuple(stats_c_defs()), tuple(stats_c_timing_defs())),
    declarations=(f"int {symbol}_stats(scaly_solver_stats* out);",),
    libraries=(desc.backend,),
    link_flags=backend_compile_flags,
    isolated=True,
    versions=_distribution(desc.backend),
  )


def _method(desc: SolverDescriptor) -> Any:
  return desc.method if desc.method is not None else external_method(desc.backend)


def _distribution(backend: str) -> tuple[tuple[str, str], ...]:
  """The plugin distribution providing ``backend`` and its version, when it is installed as one."""
  dist = getattr(REGISTRY.installed().get(backend), "dist", None)
  return ((dist.name, dist.version),) if dist is not None else ()


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
