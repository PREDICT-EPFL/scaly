"""AOT: lower a ``Function`` once and package the result as the :class:`CModule` that both the
file-writing driver here and ``codegen/jit.py`` consume.

``_lower`` is the single render context. It lowers ``fun`` exactly once and holds everything the
artifacts read off that lowering, so the header's ``SZ_W``, the source's ``_sz_w`` and a consumer's
workspace allocation cannot disagree. A function with no solver in its call graph renders entirely
through ``codegen/c``. A **solver-bearing** graph is orchestrated here: every non-solver Function
(oracle, host caller, intermediate) is a Program-IR ``_raw`` and each ``solver Function`` is the
``codegen/solver`` wrapper template that drives its (Program-IR) oracles — the one sanctioned
non-Program-IR path (see ``docs/how_it_works/solvers.md``).
"""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from alloy.codegen.abi import abi_status_defines, c_api_signature, c_ident
from alloy.codegen.c import _includes, _render_entry, _render_raw_callee, render_program_c
from alloy.codegen.solver import render_solver_raw, solver_includes, solver_stats_symbols
from alloy.ir.expr import ExprOp, topo
from alloy.function import Function
from alloy.passes.lowering import lower_function, main_proc
from alloy.passes.program import ProgramObserver
from alloy.solvers.graph import external_oracles, is_solver_function, solver_backends_used, solver_callees
from alloy.solvers.paths import backend_compile_flags
from alloy.solvers.stats import stats_c_defs, stats_c_timing_defs

if TYPE_CHECKING:
  from collections.abc import Callable

  from alloy.ir.program import ProgramNode


@dataclass(frozen=True)
class CModule:
  """One rendered function: the ``.h`` and ``.c`` to write, plus what a consumer needs to compile
  and call them. ``body`` is the translation unit. ``program`` is the optimized Program IR that
  produced it. ``workspace_size`` is the entry's ``w[]`` length, and ``backends`` lists the solver
  plugins that the function calls.

  ``header``, ``source`` and ``link_flags`` are rendered on first access. The JIT compiles ``body``
  and asks for none of them; for a big sparse function the header alone is larger than the source.
  """

  fun: Function
  header_name: str
  source_name: str
  body: str
  program: ProgramNode
  workspace_size: int
  backends: tuple[str, ...]
  typed_buffers: bool

  @cached_property
  def header(self) -> str:
    return _render_header(self.fun, self.backends, self.workspace_size, typed_buffers=self.typed_buffers)

  @cached_property
  def source(self) -> str:
    """The ``.c`` as written: ``body`` behind an include of the paired header."""
    include = self.header_name.replace("\\", "\\\\").replace('"', '\\"')
    return f'#include "{include}"\n\n{self.body}'

  @cached_property
  def link_flags(self) -> tuple[str, ...]:
    """Compiler/linker flags for ``backends`` — include, lib, rpath and ``-l`` flags. Empty without
    a solver. Resolved on demand because ``solvers.paths`` raises ``SolverLibraryError`` when a
    backend's library or header is missing: rendering has to stay possible on a machine without the
    vendored solver stack, and against a backend that has no library at all (the fake backends in
    ``tests/solvers/test_registry.py``)."""
    return tuple(backend_compile_flags(self.backends))


class RenderObserver(Protocol):
  """One render's worth of callbacks: every program the pipeline produces, then the C, then the
  outcome. ``alloy.viz.VisualizationRecording`` is the implementation in this repository."""

  def add_program(self, name: str, root: ProgramNode) -> None: ...
  def add_code(self, source: str) -> None: ...
  def finish(self, *, error: str | None = None) -> None: ...


_RENDER_OBSERVERS: list[Callable[[Function], RenderObserver | None]] = []


def register_render_observer(begin: Callable[[Function], RenderObserver | None]) -> None:
  """Watch every source render. ``begin`` is called with the function about to be rendered and
  returns an observer, or ``None`` to sit that render out.

  Codegen owns this hook so that visualization depends on codegen and not the other way round.
  """
  _RENDER_OBSERVERS.append(begin)


@dataclass(frozen=True, slots=True)
class _RenderCtx:
  """One lowering of ``fun`` and the facts every artifact reads off it."""

  fun: Function
  prog: ProgramNode
  backends: tuple[str, ...]
  workspace_size: int


def _lower(fun: Function, observe: ProgramObserver | None = None) -> _RenderCtx:
  """Lower ``fun`` once. ``workspace_size`` is the doubles of scratch it needs in ``w[]`` — the
  packed ``sz_w`` from ``passes.pack_workspace``, which also accounts for a solver wrapper passing
  its ``w`` straight to the oracle."""
  backends = solver_backends_used(fun)
  if backends:
    solver_stats_symbols(fun)  # validate duplicate solver symbols before lowering or compilation
  prog = lower_function(fun, observe=observe)
  sz_w = _solver_root_workspace(prog, fun.name) if is_solver_function(fun) else int(main_proc(prog).attrs.get("sz_w", 0))
  return _RenderCtx(fun, prog, backends, sz_w)


def _solver_root_workspace(prog: ProgramNode, name: str) -> int:
  pc = int(prog.attrs.get("proc_count", 0))
  oracle_names = set(prog.attrs.get("solver_oracles", {}).get(name, ()))
  return max(
    (
      int(prog.attrs.get("solver_external_workspace", {}).get(name, 0)),
      *(int(pr.attrs.get("sz_w", 0)) for pr in prog.args[:pc] if pr.attrs["name"] in oracle_names),
    )
  )


def _c_array(values: tuple[int, ...]) -> str:
  return "{" + ", ".join(str(v) for v in values) + "}"


def _typed_cpp_wrapper(fun: Function, symbol: str) -> list[str]:
  params: list[str] = []
  arg_values: list[str] = []
  res_values: list[str] = []
  checks: list[str] = []
  for name, expr in zip(fun.input_names, fun.inputs, strict=True):
    assert isinstance(name, str)
    ident = c_ident(name)
    type_name = f"{symbol}_{ident}_in"
    params.append(f"const {type_name}& in_{ident}")
    arg_values.append(f"in_{ident}.data")
    checks.append(f'static_assert(sizeof({type_name}) == sizeof(double) * {expr.size}, "{type_name} size mismatch");')
  for name, expr in zip(fun.output_names, fun.outputs, strict=True):
    assert isinstance(name, str)
    ident = c_ident(name)
    type_name = f"{symbol}_{ident}_out"
    params.append(f"{type_name}& out_{ident}")
    res_values.append(f"out_{ident}.data")
    checks.append(f'static_assert(sizeof({type_name}) == sizeof(double) * {expr.size}, "{type_name} size mismatch");')

  arg_init = ", ".join(arg_values) or "nullptr"
  res_init = ", ".join(res_values) or "nullptr"
  return [
    "#ifdef __cplusplus",
    *checks,
    f"static inline int {symbol}_call({', '.join(params)}) {{",
    f"  double w[{symbol}_SZ_W > 0 ? {symbol}_SZ_W : 1];",
    f"  const double* arg[{symbol}_SZ_ARG > 0 ? {symbol}_SZ_ARG : 1] = {{{arg_init}}};",
    f"  double* res[{symbol}_SZ_RES > 0 ? {symbol}_SZ_RES : 1] = {{{res_init}}};",
    f"  return {symbol}(arg, res, nullptr, {symbol}_SZ_W ? w : nullptr, nullptr);",
    "}",
    "#endif",
  ]


def _render_header(fun: Function, backends: tuple[str, ...], sz_w: int, *, typed_buffers: bool) -> str:
  symbol = c_ident(fun.name)
  lines = [
    "#pragma once",
    "",
    *(["#include <stdint.h>", "", *stats_c_defs(), ""] if backends else []),
    *abi_status_defines(guarded=True),
    "",
    f"#define {symbol}_SZ_ARG {len(fun.inputs)}",
    f"#define {symbol}_SZ_RES {len(fun.outputs)}",
    f"#define {symbol}_SZ_IW 0",
    f"#define {symbol}_SZ_W {sz_w}",
    "",
    f"// Universal CasADi-style ABI for {fun.name}.",
    "#ifdef __cplusplus",
    'extern "C" {',
    "#endif",
    c_api_signature(symbol) + ";",
  ]
  lines += [
    f"int {symbol}_sz_arg(void);",
    f"int {symbol}_sz_res(void);",
    f"int {symbol}_sz_iw(void);",
    f"int {symbol}_sz_w(void);",
    f"void* {symbol}_alloc_mem(void);",
    f"int {symbol}_init_mem(void* mem);",
    f"void {symbol}_free_mem(void* mem);",
    *(f"int {solver_symbol}_stats(alloy_solver_stats* out);" for solver_symbol in solver_stats_symbols(fun)),
    "#ifdef __cplusplus",
    "}",
    "#endif",
  ]
  if typed_buffers:
    lines += ["", "// Optional typed buffer wrappers for statically known shapes."]
    for name, expr in zip(fun.input_names, fun.inputs, strict=True):
      lines.append(f"typedef struct {{ double data[{expr.size}]; }} {symbol}_{c_ident(name)}_in;")
    for name, expr in zip(fun.output_names, fun.outputs, strict=True):
      lines.append(f"typedef struct {{ double data[{expr.size}]; }} {symbol}_{c_ident(name)}_out;")
    lines += _typed_cpp_wrapper(fun, symbol)
  sparse_outputs = [(name, sp) for name, sp in zip(fun.output_names, fun.output_sparsities, strict=True) if sp is not None]
  if sparse_outputs:
    lines += ["", "// Sparse output metadata for compact derivative buffers."]
    for name, sp in sparse_outputs:
      assert sp is not None
      prefix = f"{symbol}_{c_ident(name)}"
      lines.append(f"#define {prefix}_NNZ {sp.nnz}")
      lines.append(f"#define {prefix}_NROW {sp.shape[0]}")
      lines.append(f"#define {prefix}_NCOL {sp.shape[1]}")
      row_ptr, col_ind, csr_perm = sp.to_csr()
      col_ptr, row_ind, csc_perm = sp.to_csc()
      lines.append(f"static const int {prefix}_rows[{sp.nnz}] = {_c_array(sp.rows)};")
      lines.append(f"static const int {prefix}_cols[{sp.nnz}] = {_c_array(sp.cols)};")
      lines.append(f"static const int {prefix}_csr_row_ptr[{sp.shape[0] + 1}] = {_c_array(row_ptr)};")
      lines.append(f"static const int {prefix}_csr_col_ind[{sp.nnz}] = {_c_array(col_ind)};")
      # The compact value buffer stays in (rows, cols) COO order, which is not necessarily sorted;
      # values_csr[k] = values[csr_val_perm[k]] (and likewise for CSC) pairs it with the indices.
      lines.append(f"static const int {prefix}_csr_val_perm[{sp.nnz}] = {_c_array(csr_perm)};")
      lines.append(f"static const int {prefix}_csc_col_ptr[{sp.shape[1] + 1}] = {_c_array(col_ptr)};")
      lines.append(f"static const int {prefix}_csc_row_ind[{sp.nnz}] = {_c_array(row_ind)};")
      lines.append(f"static const int {prefix}_csc_val_perm[{sp.nnz}] = {_c_array(csc_perm)};")
  return "\n".join(lines) + "\n"


def _render_source(ctx: _RenderCtx) -> str:
  return _render_solver_bearing_source(ctx) if ctx.backends else render_program_c(ctx.prog, ctx.fun)


def _render_solver_bearing_source(ctx: _RenderCtx) -> str:
  """One translation unit for a solver-bearing graph. The non-solver Functions (oracles, the host
  caller, any intermediates) are Program-IR ``_raw`` callees; each ``solver Function`` is the
  ``solver`` wrapper driving them. ``_function_order`` is topological — a solver sits after its
  oracle PROCs and before the function that calls it — so emitting definitions in that order never
  forward-references a ``_raw``."""
  fun, prog = ctx.fun, ctx.prog  # solver callees opaque; oracles + host fns are PROCs (see passes/lowering.py)
  pc = int(prog.attrs.get("proc_count", 1))
  procs = {pr.attrs["name"]: pr for pr in prog.args[:pc]}
  lines: list[str] = [
    *_includes(("#include <time.h>", *solver_includes(fun))),
    "",
    *abi_status_defines(),
    "",
    *stats_c_defs(),
    "",
    *stats_c_timing_defs(),
    "",
    "#ifdef __cplusplus",
    'extern "C" {',
    "#endif",
    "",
  ]
  order = _function_order(fun)
  external_sources: list[str] = []
  raw_definitions: dict[str, str] = {}
  for fn in order:
    for oracle in external_oracles(fn):
      previous = raw_definitions.setdefault(oracle.raw_symbol, oracle.source)
      if previous != oracle.source:
        raise ValueError(f"external oracle symbol {oracle.raw_symbol!r} has conflicting source definitions")
      if oracle.source and oracle.source not in external_sources:
        external_sources.append(oracle.source)
  for source in external_sources:
    lines += [*source.splitlines(), ""]
  for fn in order if is_solver_function(fun) else order[:-1]:
    lines += render_solver_raw(fn, include_external_sources=False) if is_solver_function(fn) else _render_raw_callee(procs[fn.name])
    lines.append("")
  if is_solver_function(fun):
    lines += _render_solver_entry(fun, ctx.workspace_size)
  else:
    lines += _render_entry(procs[fun.name], fun)
  lines += ["", "#ifdef __cplusplus", "}", "#endif"]
  return "\n".join(lines).rstrip() + "\n"


def _render_solver_entry(fun: Function, sz_w: int) -> list[str]:
  symbol = c_ident(fun.name)
  raw_symbol = f"{symbol}_raw"
  args = [*(f"arg[{i}]" for i in range(len(fun.inputs))), *(f"res[{i}]" for i in range(len(fun.outputs))), "w"]
  lines = [
    f"int {symbol}_sz_arg(void) {{ return {len(fun.inputs)}; }}",
    f"int {symbol}_sz_res(void) {{ return {len(fun.outputs)}; }}",
    f"int {symbol}_sz_iw(void) {{ return 0; }}",
    f"int {symbol}_sz_w(void) {{ return {sz_w}; }}",
    f"void* {symbol}_alloc_mem(void) {{ return NULL; }}",
    f"int {symbol}_init_mem(void* mem) {{ (void)mem; return ALLOY_SUCCESS; }}",
    f"void {symbol}_free_mem(void* mem) {{ (void)mem; }}",
    "",
    c_api_signature(symbol) + " {",
    "  (void)iw;",
    "  (void)mem;",
    "  if (!arg || !res) return ALLOY_ERR_NULL_ABI;",
  ]
  if sz_w:
    lines.append("  if (!w) return ALLOY_ERR_NULL_WORK;")
  else:
    lines.append("  (void)w;")
  lines += [f"  if (!arg[{i}]) return ALLOY_ERR_NULL_INPUT;" for i in range(len(fun.inputs))]
  lines += [f"  if (!res[{i}]) return ALLOY_ERR_NULL_RESULT;" for i in range(len(fun.outputs))]
  lines += [f"  {raw_symbol}({', '.join(args)});", "  return ALLOY_SUCCESS;", "}"]
  return lines


def _render_observed(fun: Function) -> tuple[_RenderCtx, str]:
  """Lower and render ``fun`` under the registered observers: one lowering, one source, and the
  ``add_program`` / ``add_code`` / ``finish`` sequence ``alloy.viz`` records."""
  observers = [obs for begin in _RENDER_OBSERVERS if (obs := begin(fun)) is not None]

  def observe(name: str, root: ProgramNode) -> None:
    for obs in observers:
      obs.add_program(name, root)

  try:
    ctx = _lower(fun, observe if observers else None)
    source = _render_source(ctx)
  except Exception as exc:
    for obs in observers:
      obs.finish(error=repr(exc))
    raise
  for obs in observers:
    obs.add_code(source)
    obs.finish()
  return ctx, source


def render_c_source(fun: Function) -> str:
  """Render a standalone universal-ABI C implementation of ``fun`` and its callees.

  A ``LoweringError`` (e.g. a still-deferred mixed-device CALL) propagates — there is no fallback.
  """
  return _render_observed(fun)[1]


def render_c_api_header(fun: Function, *, typed_buffers: bool = True) -> str:
  """Render the public ``.h`` for ``fun``: the ABI declarations, ``SZ_*`` constants, optional typed
  buffer wrappers, and sparse-output metadata."""
  ctx = _lower(fun)
  return _render_header(ctx.fun, ctx.backends, ctx.workspace_size, typed_buffers=typed_buffers)


def render_c_module(fun: Function, *, header_name: str | None = None, source_name: str | None = None, typed_buffers: bool = True) -> CModule:
  """Render ``fun`` into its ``.h`` / ``.c`` pair from a single lowering."""
  ctx, body = _render_observed(fun)
  symbol = c_ident(fun.name)
  return CModule(
    fun=fun,
    header_name=f"{symbol}.h" if header_name is None else header_name,
    source_name=f"{symbol}.c" if source_name is None else source_name,
    body=body,
    program=ctx.prog,
    workspace_size=ctx.workspace_size,
    backends=ctx.backends,
    typed_buffers=typed_buffers,
  )


def workspace_size(fun: Function) -> int:
  """Doubles of scratch ``fun`` needs in ``w[]`` — the value its header's ``SZ_W`` and its rendered
  ``<symbol>_sz_w`` both quote. ``CModule.workspace_size`` is the same number without a second
  lowering, so prefer it when the module is already in hand."""
  return _lower(fun).workspace_size


def write_module(fun: Function, out_dir: Path, *, typed_buffers: bool = True) -> CModule:
  """Write ``fun``'s ``.h`` / ``.c`` into ``out_dir`` and return the module."""
  module = render_c_module(fun, typed_buffers=typed_buffers)
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / module.header_name).write_text(module.header)
  (out_dir / module.source_name).write_text(module.source)
  return module


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(prog="python -m alloy.codegen", description="Render a Function to a C header/source pair.")
  parser.add_argument("target", help="module:attribute naming a Function or a zero-argument factory returning one")
  parser.add_argument("-o", "--out-dir", type=Path, default=Path(), help="directory to write into (default: cwd)")
  parser.add_argument("--no-typed-buffers", action="store_true", help="omit the typed buffer structs and the C++ call wrapper")
  args = parser.parse_args(argv)
  module_name, _, attr = args.target.partition(":")
  if not attr:
    parser.error(f"target {args.target!r} is not module:attribute")
  fun = getattr(importlib.import_module(module_name), attr)
  if not isinstance(fun, Function):
    fun = fun()
  module = write_module(fun, args.out_dir, typed_buffers=not args.no_typed_buffers)
  print(args.out_dir / module.header_name)
  print(args.out_dir / module.source_name)
  print(f"sz_w: {module.workspace_size}")
  if module.backends:
    print("link flags: " + " ".join(module.link_flags))


def _function_order(fun: Function) -> list[Function]:
  seen: set[int] = set()
  ordered: list[Function] = []

  def visit(fn: Function) -> None:
    if id(fn) in seen:
      return
    seen.add(id(fn))
    for callee in _callees(fn):
      visit(callee)
    ordered.append(fn)

  visit(fun)
  return ordered


def _callees(fun: Function) -> list[Function]:
  ret: list[Function] = []
  seen: set[int] = set()
  if is_solver_function(fun):
    # solver Functions render via a custom template that calls the oracle (and for NLP, the
    # derivative Functions) — these aren't reachable through the solver's own output graph (it
    # only contains SOLVER_CALL nodes), so surface them explicitly here.
    for callee in solver_callees(fun):
      if id(callee) not in seen:
        seen.add(id(callee))
        ret.append(callee)
    return ret
  for node in topo(fun.outputs):
    if node.op not in {ExprOp.CALL, ExprOp.VMAP}:
      continue
    callee = node.attrs["callee"]
    if id(callee) not in seen:
      seen.add(id(callee))
      ret.append(callee)
  return ret


if __name__ == "__main__":
  main()
