"""C codegen orchestration for ``ExprOp.SOLVER_CALL`` — plugin-rendered wrappers.

A solver Function's C body is a small hand-written template per backend,
parameterised by the ``SolverDescriptor`` — the one sanctioned non-Program-IR
render path (see ``docs/how_it_works/solvers.md``). The templates
themselves live in the solver plugins (``alloy_piqp.codegen``,
``alloy_ipopt.codegen``, ...): core hands the plugin's
``SolverBackend.render_wrapper`` hook a :class:`SolverWrapperCtx` and frames
the returned body with the alloy-owned stats storage and accessor. The oracle
Functions the template drives are *not* hand-written — they lower through
Program IR like any other host Function and are rendered as ``<oracle>_raw``
by ``codegen/c``. See ``docs/dev/solver_plugins.md`` for the contract.

Outer functions that contain a solver as a callee lower through Program IR with
the ``SolverFunction`` callee treated as opaque (``passes.lowering.lower_function``):
the solver renders to a ``static void qp_xxx_raw(...)`` body here, and the
caller's lowered ``CALL`` emits a ``qp_xxx_raw(...)`` invocation.
``codegen.aot.render_c_source`` orchestrates the whole translation unit, ordering
the solver wrapper after its (Program-IR) oracle PROCs. The graph queries behind
both — which Functions are solvers, what they reach — are ``solvers/graph.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from alloy.codegen.abi import c_ident
from alloy.ir.expr import ExprOp, topo
from alloy.function import Function
from alloy.solvers.graph import external_oracles, is_solver_function, solver_backends_used, solver_callees, solver_descriptor

if TYPE_CHECKING:
  from alloy.solvers.solver_function import ExternalOracle


def _raw_symbol(fun: Function) -> str:
  return f"{c_ident(fun.name)}_raw"


@dataclass(frozen=True, slots=True)
class SolverWrapperCtx:
  """Codegen kit handed to a plugin's ``render_wrapper`` hook.

  ``symbol`` is the solver's mangled C identifier (prefix for any static the
  template declares), ``raw_symbol`` the function the template must define,
  ``stats_symbol`` the ``alloy_solver_stats`` static it must fill (declared by
  core, one per solver). ``raw_symbol_of`` resolves the C symbol of an oracle
  / derivative Function from the descriptor.
  """

  symbol: str
  raw_symbol: str
  stats_symbol: str

  def raw_symbol_of(self, fun: Function | ExternalOracle) -> str:
    from alloy.solvers.solver_function import ExternalOracle

    return fun.raw_symbol if isinstance(fun, ExternalOracle) else _raw_symbol(fun)


def solver_includes(fun: Function) -> list[str]:
  from alloy.solvers.registry import get_backend

  return [f'#include "{get_backend(name).header}"' for name in solver_backends_used(fun)]


def solver_stats_symbols(fun: Function) -> tuple[str, ...]:
  """C identifiers for every solver wrapper reachable from ``fun``."""
  found: dict[str, Function] = {}
  seen: set[int] = set()

  def visit(fn: Function) -> None:
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
      if node.op in {ExprOp.CALL, ExprOp.MAP}:
        visit(node.attrs["callee"])

  visit(fun)
  return tuple(found)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def render_solver_raw(fun: Function, *, include_external_sources: bool = True) -> list[str]:
  """Frame a plugin-rendered wrapper body with the alloy-owned stats storage
  and the exported ``<symbol>_stats`` accessor. The body itself comes from the
  backend's ``render_wrapper`` hook."""
  from alloy.solvers.registry import get_backend

  desc = solver_descriptor(fun)
  external_sources: list[str] = []
  for oracle in external_oracles(fun):
    if include_external_sources and oracle.source and oracle.source not in external_sources:
      external_sources.append(oracle.source)
  symbol = c_ident(fun.name)
  ctx = SolverWrapperCtx(symbol=symbol, raw_symbol=_raw_symbol(fun), stats_symbol=f"{symbol}_stats_data")
  body = get_backend(desc.backend).render_wrapper(fun, ctx)
  return [
    *(line for source in external_sources for line in (*source.splitlines(), "")),
    f"static alloy_solver_stats {ctx.stats_symbol};",
    *body,
    "",
    f"int {symbol}_stats(alloy_solver_stats* out) {{",
    "  if (!out) return 1;",
    f"  *out = {ctx.stats_symbol};",
    "  return 0;",
    "}",
  ]
