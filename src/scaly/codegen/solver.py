"""C codegen orchestration for ``ExprOp.SOLVER_CALL`` — plugin-rendered wrappers.

A solver ConcreteFunction's C body is a small hand-written template per backend,
parameterised by the ``SolverDescriptor`` — the one sanctioned non-Program-IR
render path (see ``docs/how_it_works/solvers.md``). The templates
themselves live in the solver plugins (``scaly_piqp.codegen``,
``scaly_ipopt.codegen``, ...): core hands the plugin's
``SolverBackend.render_wrapper`` hook a :class:`SolverWrapperCtx` and frames
the returned body with the scaly-owned stats storage and accessor. The oracle
Functions the template drives are *not* hand-written — they lower through
Program IR like any other host ConcreteFunction and are rendered as ``<oracle>_raw``
by ``codegen/c``. See ``docs/dev/solver_plugins.md`` for the contract.

Outer functions that contain a solver as a callee lower through Program IR with
the ``solver ConcreteFunction`` callee treated as opaque (``passes.lowering.lower_function``):
the solver renders to a ``static void qp_xxx_raw(...)`` body here, and the
caller's lowered ``CALL`` emits a ``qp_xxx_raw(...)`` invocation.
``codegen.aot.render_c_source`` orchestrates the whole translation unit, ordering
the solver wrapper after its (Program-IR) oracle PROCs. The graph queries behind
both — which Functions are solvers, what they reach — are ``solvers/graph.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from scaly.codegen.abi import c_ident
from scaly.ir.expr import ExprOp, topo
from scaly.function.concrete import ConcreteFunction
from scaly.solvers.graph import external_oracles, is_solver_function, solver_backends_used, solver_callees, solver_descriptor

if TYPE_CHECKING:
  from scaly.solvers.model import ExternalOracle


def _raw_symbol(fun: ConcreteFunction) -> str:
  return f"{c_ident(fun.name)}_raw"


@dataclass(frozen=True, slots=True)
class SolverWrapperCtx:
  """Codegen kit handed to a plugin's ``render_wrapper`` hook.

  ``symbol`` is the solver's mangled C identifier (prefix for any static the
  template declares), ``raw_symbol`` the function the template must define,
  ``stats_symbol`` the ``scaly_solver_stats`` static it must fill (declared by
  core, one per solver). ``raw_symbol_of`` resolves the C symbol of an oracle
  / derivative ConcreteFunction from the descriptor.
  """

  symbol: str
  raw_symbol: str
  stats_symbol: str
  options_index: int

  def oracle_options(self, fun: ConcreteFunction | ExternalOracle | None) -> tuple[str, ...]:
    """The runtime context argument for a generated oracle, empty for a foreign oracle."""
    return ("solver_options",) if isinstance(fun, ConcreteFunction) else ()

  def raw_symbol_of(self, fun: ConcreteFunction | ExternalOracle) -> str:
    from scaly.solvers.model import ExternalOracle

    return fun.raw_symbol if isinstance(fun, ExternalOracle) else _raw_symbol(fun)


def solver_includes(fun: ConcreteFunction) -> list[str]:
  from scaly.solvers.registry import get_backend

  return [f'#include "{get_backend(name).header}"' for name in solver_backends_used(fun)]


def solver_functions(fun: ConcreteFunction) -> tuple[ConcreteFunction, ...]:
  """Every reachable solver, in the call-time options order."""
  found: dict[str, ConcreteFunction] = {}
  seen: set[int] = set()

  def visit(fn: ConcreteFunction) -> None:
    if id(fn) in seen:
      return
    seen.add(id(fn))
    if is_solver_function(fn):
      symbol = c_ident(fn.name)
      if symbol in found and found[symbol] is not fn:
        raise ValueError(f"duplicate solver symbol {symbol!r} in one generated translation unit")
      found[symbol] = fn
      for callee in solver_callees(fn):
        visit(callee)
      return
    for node in topo(fn.outputs):
      if node.op in {ExprOp.CALL, ExprOp.VMAP}:
        visit(node.attrs["callee"])

  visit(fun)
  return tuple(found.values())


def solver_stats_symbols(fun: ConcreteFunction) -> tuple[str, ...]:
  """C identifiers for every solver wrapper reachable from ``fun``."""
  return tuple(c_ident(fn.name) for fn in solver_functions(fun))


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def render_solver_raw(fun: ConcreteFunction, *, options_index: int, include_external_sources: bool = True) -> list[str]:
  """Frame a plugin-rendered wrapper body with the scaly-owned stats storage
  and the exported ``<symbol>_stats`` accessor. The body itself comes from the
  backend's ``render_wrapper`` hook."""
  from scaly.solvers.registry import get_backend

  desc = solver_descriptor(fun)
  external_sources: list[str] = []
  for oracle in external_oracles(fun):
    if include_external_sources and oracle.source and oracle.source not in external_sources:
      external_sources.append(oracle.source)
  symbol = c_ident(fun.name)
  ctx = SolverWrapperCtx(symbol=symbol, raw_symbol=_raw_symbol(fun), stats_symbol=f"{symbol}_stats_data", options_index=options_index)
  body = get_backend(desc.backend).render_wrapper(fun, ctx)
  return [
    *(line for source in external_sources for line in (*source.splitlines(), "")),
    f"static scaly_solver_stats {ctx.stats_symbol};",
    *body,
    "",
    f"int {symbol}_stats(scaly_solver_stats* out) {{",
    "  if (!out) return 1;",
    f"  *out = {ctx.stats_symbol};",
    "  return 0;",
    "}",
  ]


def solver_options_c_defs() -> list[str]:
  """The call-time option layout shared by Python and generated C/C++ consumers."""
  return [
    "#ifndef SCALY_SOLVER_OPTION_DEFINED",
    "#define SCALY_SOLVER_OPTION_DEFINED",
    "typedef struct { const char* name; int kind; int64_t integer; double number; const char* text; } scaly_solver_option;",
    "#endif",
  ]


def render_solver_defaults(fun: ConcreteFunction) -> list[str]:
  """Render backend defaults from their Python owner for the universal entry and C callers."""
  from scaly.solvers.registry import get_backend

  lines = []
  for fn in solver_functions(fun):
    symbol = c_ident(fn.name)
    lines.append(f"const scaly_solver_option* {symbol}_default_options(void) {{")
    lines.append("  static const scaly_solver_option options[] = {")
    _, defaults = get_backend(fn.descriptor.backend).prepare_options({})
    for name, value in defaults.items():
      if isinstance(value, (bool, int)):
        literal = f"0, {int(value)}, 0.0, NULL"
      elif isinstance(value, float):
        literal = f"1, 0, {value!r}, NULL"
      else:
        literal = f'2, 0, 0.0, "{value}"'
      lines.append(f'    {{ "{name}", {literal} }},')
    lines += ["    { NULL, 0, 0, 0.0, NULL }", "  };", "  return options;", "}", ""]
  return lines


def solver_options_declarations(fun: ConcreteFunction) -> list[str]:
  """Declare the runtime entry, default option accessors and option-array indices."""
  from .abi import c_api_signature

  solvers = solver_functions(fun)
  if not solvers:
    return []
  symbol = c_ident(fun.name)
  return [
    c_api_signature(f"{symbol}_with_options", solver_options=True) + ";",
    f"#define {symbol}_N_SOLVERS {len(solvers)}",
    *(f"#define {symbol}_OPTIONS_{c_ident(fn.name)} {i}" for i, fn in enumerate(solvers)),
    *(f"const scaly_solver_option* {c_ident(fn.name)}_default_options(void);" for fn in solvers),
  ]
