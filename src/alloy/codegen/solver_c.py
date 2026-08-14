"""C codegen orchestration for ``Ops.SOLVER_CALL`` — plugin-rendered wrappers.

A solver Function's C body is a small hand-written template per backend,
parameterised by the ``SolverDescriptor`` — the one sanctioned non-Program-IR
render path (rule 6 in ``docs/program_ir_migration.md``). The templates
themselves live in the solver plugins (``alloy_piqp.codegen``,
``alloy_ipopt.codegen``, ...): core hands the plugin's
``SolverBackend.render_wrapper`` hook a :class:`SolverWrapperCtx` and frames
the returned body with the alloy-owned stats storage and accessor. The oracle
Functions the template drives are *not* hand-written — they lower through
Program IR like any other host Function and are rendered as ``<oracle>_raw``
by ``codegen/program_c``. See ``docs/solver_plugins.md`` for the contract.

Outer functions that contain a solver as a callee lower through Program IR with
the ``SolverFunction`` callee treated as opaque (``lowering.lower_function``):
the solver renders to a ``static void qp_xxx_raw(...)`` body here, and the
caller's lowered ``CALL`` emits a ``qp_xxx_raw(...)`` invocation.
``codegen/c.render_c_source`` orchestrates the whole translation unit, ordering
the solver wrapper after its (Program-IR) oracle PROCs. The JIT detects native
linkage needs through ``solver_backends_used(...)``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from alloy.expr import topo
from alloy.function import Function
from alloy.ops import Ops
from alloy.toolchain import solver_compile_flags as _toolchain_solver_compile_flags

if TYPE_CHECKING:
  from alloy.solvers.solver_function import ExternalOracle, SolverDescriptor


def _c_ident(name: str) -> str:
  """Must match ``codegen.c._c_ident`` and ``codegen.program_c._c_ident``."""
  ident = re.sub(r"\W", "_", name)
  if ident in ("w", "arg", "res", "iw", "mem"):
    ident += "_"
  return f"_{ident}" if ident[:1].isdigit() else ident


def _raw_symbol(fun: Function) -> str:
  return f"{_c_ident(fun.name)}_raw"


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


def is_solver_function(fun: Function) -> bool:
  desc = getattr(fun, "descriptor", None)
  return isinstance(getattr(desc, "backend", None), str)


def _descriptor(fun: Function) -> SolverDescriptor:
  return fun.descriptor  # ty: ignore[unresolved-attribute]


# ---------------------------------------------------------------------------
# Predicates / dependencies the outer renderer consumes.
# ---------------------------------------------------------------------------


def solver_callees(fun: Function) -> list[Function]:
  """Return the inner Functions a solver Function depends on at codegen time."""
  if not is_solver_function(fun):
    return []
  desc = _descriptor(fun)
  out: list[Function] = []
  for cand in (desc.oracle, desc.base, desc.grad, desc.jac, desc.hess, desc.bounds):
    if isinstance(cand, Function) and cand not in out:
      out.append(cand)
  return out


def solver_backends_used(fun: Function) -> tuple[str, ...]:
  """Sorted names of every solver backend reachable from ``fun`` (through
  CALL/MAP callees, SOLVER_CALL nodes, and solver oracle Functions)."""
  found: set[str] = set()
  seen: set[int] = set()

  def visit(fn: Function) -> None:
    if id(fn) in seen:
      return
    seen.add(id(fn))
    if is_solver_function(fn):
      found.add(_descriptor(fn).backend)
      for callee in solver_callees(fn):
        visit(callee)
      return
    for node in topo(fn.outputs):
      if node.op in {Ops.CALL, Ops.MAP}:
        visit(node.attrs["callee"])
      elif node.op == Ops.SOLVER_CALL:
        found.add(node.attrs["solver"].backend)

  visit(fun)
  return tuple(sorted(found))


def solver_includes(fun: Function) -> list[str]:
  from alloy.solvers.registry import get_backend

  return [f'#include "{get_backend(name).header}"' for name in solver_backends_used(fun)]


def solver_compile_flags(fun: Function, *, rpath: bool = True) -> list[str]:
  """Compiler/linker flags an AOT consumer needs for ``fun``.

  Returns ``[]`` when ``fun`` does not transitively reach any solver.
  Otherwise: plugin package ``-I`` / ``-L`` paths, each reached backend's
  ``link_flags``, and an ``-Wl,-rpath`` pointing at the vendored lib
  directories so the resulting binary finds the shared libs at load time
  without ``LD_LIBRARY_PATH`` / ``DYLD_LIBRARY_PATH`` overrides.

  Set ``rpath=False`` if the consumer plans to bundle the libs elsewhere and
  will set the rpath / install_name themselves.
  """
  names = solver_backends_used(fun)
  if not names:
    return []
  return _toolchain_solver_compile_flags(names, rpath=rpath)


def solver_stats_symbols(fun: Function) -> tuple[str, ...]:
  """C identifiers for every solver wrapper reachable from ``fun``."""
  found: dict[str, Function] = {}
  seen: set[int] = set()

  def visit(fn: Function) -> None:
    if id(fn) in seen:
      return
    seen.add(id(fn))
    if is_solver_function(fn):
      symbol = _c_ident(fn.name)
      if symbol in found and found[symbol] is not fn:
        raise ValueError(f"duplicate solver symbol {symbol!r} in one generated translation unit")
      found[symbol] = fn
      for callee in solver_callees(fn):
        visit(callee)
      return
    for node in topo(fn.outputs):
      if node.op in {Ops.CALL, Ops.MAP}:
        visit(node.attrs["callee"])

  visit(fun)
  return tuple(found)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def external_oracles(fun: Function) -> tuple[ExternalOracle, ...]:
  """External descriptor oracles in call order, deduplicated by identity."""
  from alloy.solvers.solver_function import ExternalOracle

  if not is_solver_function(fun):
    return ()
  desc = _descriptor(fun)
  out: list[ExternalOracle] = []
  for oracle in (desc.base, desc.grad, desc.jac, desc.hess, desc.bounds):
    if isinstance(oracle, ExternalOracle) and oracle not in out:
      out.append(oracle)
  return tuple(out)


def render_solver_raw(fun: Function, *, include_external_sources: bool = True) -> list[str]:
  """Frame a plugin-rendered wrapper body with the alloy-owned stats storage
  and the exported ``<symbol>_stats`` accessor. The body itself comes from the
  backend's ``render_wrapper`` hook."""
  from alloy.solvers.registry import get_backend

  desc = _descriptor(fun)
  external_sources: list[str] = []
  for oracle in external_oracles(fun):
    if include_external_sources and oracle.source and oracle.source not in external_sources:
      external_sources.append(oracle.source)
  symbol = _c_ident(fun.name)
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
