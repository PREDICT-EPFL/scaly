"""AOT: lower a ``ConcreteFunction`` once and package the result as the :class:`CModule` that both the
file-writing driver here and ``codegen/jit.py`` consume.

``_lower`` is the single render context. It lowers ``fun`` exactly once and holds everything the
artifacts read off that lowering, so the header's ``SZ_W``, the entry's null check and a consumer's
workspace allocation cannot disagree. The header comes in two languages (``lang="c"`` here,
``lang="cpp"`` in ``codegen/cpp.py``) and either can carry the CasADi layer (``codegen/casadi.py``). A function with no solver in its call graph renders entirely
through ``codegen/c``. A **solver-bearing** graph is orchestrated here: every non-solver ConcreteFunction
(oracle, host caller, intermediate) is a Program-IR ``_raw`` and each ``solver ConcreteFunction`` is the
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

from scaly.codegen.toolchain import BuildRecipe, CPU_LEVELS, CDialect, CpuLevel, LaneCount, VectorLibm
from scaly.codegen.abi import abi_status_defines, buffer_idents, c_api_signature, c_ident
from scaly.codegen.c import _c_reserved_names, _includes, _render_entry, _render_raw_callee, entry_prologue, entry_workspace, render_program_c
from scaly.codegen.casadi import (
  casadi_declarations,
  casadi_defines,
  casadi_gather,
  casadi_output_sparsities,
  check_casadi_layout,
  render_casadi_queries,
)
from scaly.codegen.cpp import render_cpp_header
from scaly.codegen.solver import (
  render_solver_raw,
  solver_includes,
  solver_stats_symbols,
  solver_functions,
  solver_options_c_defs,
  solver_options_c_helpers,
  render_solver_defaults,
  solver_options_declarations,
)
from scaly.ir.expr import ExprOp, topo
from scaly.function.concrete import ConcreteFunction
from scaly.function.model import Function, as_concrete
from scaly.passes.lowering import lower_function, main_proc
from scaly.passes.program import ProgramObserver
from scaly.solvers.graph import external_oracles, is_solver_function, solver_backends_used, solver_callees
from scaly.solvers.paths import backend_compile_flags
from scaly.solvers.solver import Solver
from scaly.solvers.stats import stats_c_defs, stats_c_timing_defs

if TYPE_CHECKING:
  from collections.abc import Callable

  from scaly.ir.program import ProgramNode
  from scaly.ir.types import SparsityPattern


@dataclass(frozen=True)
class CModule:
  """One rendered function: the header and the ``.c`` to write, plus what a consumer needs to
  compile and call them. ``body`` is the translation unit. ``program`` is the optimized Program IR
  that produced it. ``workspace_size`` is the entry's ``w[]`` length, and ``backends`` lists the
  solver plugins that the function calls. ``lang`` picks the header language (``"c"`` or
  ``"cpp"``), and ``casadi`` adds the CasADi-compatible symbols.

  ``header``, ``source`` and ``link_flags`` are rendered on first access. The JIT compiles ``body``
  and asks for none of them. For a big sparse function the header alone is larger than the source.
  """

  fun: ConcreteFunction
  header_name: str
  source_name: str
  body: str
  program: ProgramNode
  workspace_size: int
  backends: tuple[str, ...]
  lang: str = "c"
  casadi: bool = False
  recipe: BuildRecipe = BuildRecipe()

  @cached_property
  def header(self) -> str:
    return self.recipe.comment(self.source_name) + _render_header(self.fun, self.backends, self.workspace_size, lang=self.lang, casadi=self.casadi)

  @cached_property
  def source(self) -> str:
    """The ``.c`` as written: ``body`` behind an include of the paired C header. A C++ header
    cannot be included from C, so under ``lang="cpp"`` the kernel is ``body`` alone."""
    if self.lang == "cpp":
      return self.body
    include = self.header_name.replace("\\", "\\\\").replace('"', '\\"')
    recipe = self.recipe.comment(self.source_name)
    return recipe + f'#include "{include}"\n\n' + self.body.removeprefix(recipe)

  @cached_property
  def link_flags(self) -> tuple[str, ...]:
    """Compiler and linker flags for ``backends``: include, lib, rpath and ``-l`` flags, plus libraries required by the math recipe. Resolved on demand because ``solvers.paths`` raises ``SolverLibraryError`` when a
    backend's library or header is missing: rendering has to stay possible on a machine without the
    vendored solver stack, and against a backend that has no library at all (the fake backends in
    ``tests/solvers/test_registry.py``)."""
    return (*backend_compile_flags(self.backends), *self.recipe.link_flags)


class RenderObserver(Protocol):
  """One render's callbacks: normalized expressions, each Program stage, the C, and the outcome. ``scaly.viz.VisualizationRecording`` is the implementation in this repository."""

  def add_normalized_expr(self, name: str, fun: ConcreteFunction) -> None: ...
  def add_program(self, name: str, root: ProgramNode) -> None: ...
  def add_code(self, source: str) -> None: ...
  def finish(self, *, error: str | None = None) -> None: ...


_RENDER_OBSERVERS: list[Callable[[ConcreteFunction], RenderObserver | None]] = []


def register_render_observer(begin: Callable[[ConcreteFunction], RenderObserver | None]) -> None:
  """Watch every source render. ``begin`` is called with the function about to be rendered and
  returns an observer, or ``None`` to sit that render out.

  Codegen owns this hook so that visualization depends on codegen and not the other way round.
  """
  _RENDER_OBSERVERS.append(begin)


@dataclass(frozen=True, slots=True)
class _RenderCtx:
  """One lowering of ``fun`` and the facts every artifact reads off it."""

  fun: ConcreteFunction
  prog: ProgramNode
  backends: tuple[str, ...]
  workspace_size: int
  recipe: BuildRecipe


def _lower(
  fun: ConcreteFunction,
  observe: ProgramObserver | None = None,
  observe_expr: Callable[[str, ConcreteFunction], None] | None = None,
  *,
  recipe: BuildRecipe = BuildRecipe(),
) -> _RenderCtx:
  """Lower ``fun`` once. ``workspace_size`` is the doubles of scratch it needs in ``w[]`` — the
  packed ``sz_w`` from ``passes.pack_workspace``, which also accounts for a solver wrapper passing
  its ``w`` straight to the oracle."""
  backends = solver_backends_used(fun)
  if backends:
    solver_stats_symbols(fun)  # validate duplicate solver symbols before lowering or compilation
  prog = lower_function(fun, observe=observe, observe_expr=observe_expr, reciprocal=recipe.reciprocal, lanes=recipe.lanes)
  sz_w = _solver_root_workspace(prog, fun.name) if is_solver_function(fun) else int(main_proc(prog).attrs.get("sz_w", 0))
  return _RenderCtx(fun, prog, backends, sz_w, recipe)


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


_ALIGNAS = [
  "#ifndef SCALY_ALIGNAS",
  "#ifdef __cplusplus",
  "#define SCALY_ALIGNAS(n) alignas(n)",
  "#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L",
  "#define SCALY_ALIGNAS(n) _Alignas(n)",
  "#else",
  "#define SCALY_ALIGNAS(n)",
  "#endif",
  "#endif",
]


def _typed_buffers(fun: ConcreteFunction, symbol: str) -> list[str]:
  """One struct per input and output, the caller-owned workspace struct, and a
  ``_call`` wrapper that builds the pointer arrays. C11 and C++ align to 16 bytes; C99 uses natural alignment."""
  params: list[str] = []
  arg_values: list[str] = []
  res_values: list[str] = []
  lines = ["", "// Typed buffers: one struct per input and output, and the caller-owned workspace."]
  inputs, outputs = buffer_idents(fun)
  for ident, expr in zip(inputs, fun.inputs, strict=True):
    lines.append(f"typedef struct {{ SCALY_ALIGNAS(16) double data[{max(expr.size, 1)}]; }} {symbol}_{ident}_t;")
    params.append(f"const {symbol}_{ident}_t* {ident}")
    arg_values.append(f"{ident}->data")
  for ident, expr in zip(outputs, fun.outputs, strict=True):
    lines.append(f"typedef struct {{ SCALY_ALIGNAS(16) double data[{max(expr.size, 1)}]; }} {symbol}_{ident}_t;")
    params.append(f"{symbol}_{ident}_t* {ident}")
    res_values.append(f"{ident}->data")
  params.append(f"{symbol}_workspace_t* workspace")
  lines.append(f"typedef struct {{ SCALY_ALIGNAS(16) double data[{symbol}_SZ_W > 0 ? {symbol}_SZ_W : 1]; }} {symbol}_workspace_t;")
  for runtime in (False, True) if solver_backends_used(fun) else (False,):
    suffix = "_with_options" if runtime else ""
    call_params = [*params, *(("const scaly_solver_option* const* solver_options",) if runtime else ())]
    lines += [
      f"static inline int {symbol}_call{suffix}({', '.join(call_params)}) {{",
      f"  const double* arg[{symbol}_SZ_ARG > 0 ? {symbol}_SZ_ARG : 1] = {{{', '.join(arg_values) or 'NULL'}}};",
      f"  double* res[{symbol}_SZ_RES > 0 ? {symbol}_SZ_RES : 1] = {{{', '.join(res_values) or 'NULL'}}};",
      f"  return {symbol}{suffix}(arg, res, NULL, workspace ? workspace->data : NULL, 0{', solver_options' if runtime else ''});",
      "}",
    ]
  return lines


def _sparse_tables(fun: ConcreteFunction, symbol: str, sparsities: tuple[SparsityPattern | None, ...]) -> list[str]:
  lines: list[str] = []
  for name, sp in zip(fun.output_names, sparsities, strict=True):
    if sp is None:
      continue
    prefix = f"{symbol}_{c_ident(name)}"
    row_ptr, col_ind, csr_perm = sp.to_csr()
    col_ptr, row_ind, csc_perm = sp.to_csc()
    lines += [
      f"#define {prefix}_NNZ {sp.nnz}",
      f"#define {prefix}_NROW {sp.shape[0]}",
      f"#define {prefix}_NCOL {sp.shape[1]}",
      f"static const int {prefix}_rows[{sp.nnz}] = {_c_array(sp.rows)};",
      f"static const int {prefix}_cols[{sp.nnz}] = {_c_array(sp.cols)};",
      f"static const int {prefix}_csr_row_ptr[{sp.shape[0] + 1}] = {_c_array(row_ptr)};",
      f"static const int {prefix}_csr_col_ind[{sp.nnz}] = {_c_array(col_ind)};",
      # The compact value buffer stays in (rows, cols) COO order, which is not necessarily sorted;
      # values_csr[k] = values[csr_val_perm[k]] (and likewise for CSC) pairs it with the indices.
      f"static const int {prefix}_csr_val_perm[{sp.nnz}] = {_c_array(csr_perm)};",
      f"static const int {prefix}_csc_col_ptr[{sp.shape[1] + 1}] = {_c_array(col_ptr)};",
      f"static const int {prefix}_csc_row_ind[{sp.nnz}] = {_c_array(row_ind)};",
      f"static const int {prefix}_csc_val_perm[{sp.nnz}] = {_c_array(csc_perm)};",
    ]
  return ["", "// Sparse output metadata for compact derivative buffers.", *lines] if lines else []


def header_sparsities(fun: ConcreteFunction, *, casadi: bool) -> tuple[SparsityPattern | None, ...]:
  """The patterns a header describes: the native order, or under ``casadi`` the compressed-column
  order the entry gathers into."""
  return casadi_output_sparsities(fun) if casadi else tuple(fun.output_sparsities)


def _render_header(fun: ConcreteFunction, backends: tuple[str, ...], sz_w: int, *, lang: str, casadi: bool) -> str:
  if lang == "cpp":
    return render_cpp_header(fun, backends, sz_w, casadi=casadi, sparsities=header_sparsities(fun, casadi=casadi))
  symbol = c_ident(fun.name)
  lines = [
    "#pragma once",
    "",
    "#include <stddef.h>",
    *(["#include <stdint.h>", "", *stats_c_defs(), *solver_options_c_defs()] if backends else []),
    "",
    *abi_status_defines(guarded=True),
    *(["", *casadi_defines()] if casadi else []),
    "",
    *_ALIGNAS,
    "",
    f"#define {symbol}_SZ_ARG {len(fun.inputs)}",
    f"#define {symbol}_SZ_RES {len(fun.outputs)}",
    f"#define {symbol}_SZ_IW 0",
    f"#define {symbol}_SZ_W {sz_w}",
    "",
    f"// The pointer ABI for {fun.name}.",
    "#ifdef __cplusplus",
    'extern "C" {',
    "#endif",
    c_api_signature(symbol) + ";",
    *solver_options_declarations(fun),
    *(f"int {solver_symbol}_stats(scaly_solver_stats* out);" for solver_symbol in solver_stats_symbols(fun)),
    *(casadi_declarations(symbol) if casadi else []),
    "#ifdef __cplusplus",
    "}",
    "#endif",
  ]
  lines += _typed_buffers(fun, symbol)
  lines += _sparse_tables(fun, symbol, header_sparsities(fun, casadi=casadi))
  return "\n".join(lines) + "\n"


def _render_source(ctx: _RenderCtx, *, casadi: bool) -> str:
  return (
    _render_solver_bearing_source(ctx, casadi=casadi)
    if ctx.backends
    else render_program_c(ctx.prog, ctx.fun, casadi=casadi, dialect=ctx.recipe.dialect, vector_libm=ctx.recipe.vector_libm)
  )


def _render_solver_bearing_source(ctx: _RenderCtx, *, casadi: bool) -> str:
  """One translation unit for a solver-bearing graph. The non-solver Functions (oracles, the host
  caller, any intermediates) are Program-IR ``_raw`` callees; each ``solver ConcreteFunction`` is the
  ``solver`` wrapper driving them. ``_function_order`` is topological — a solver sits after its
  oracle PROCs and before the function that calls it — so emitting each wrapper after the PROCs up
  to its oracles never forward-references a ``_raw``."""
  fun, prog = ctx.fun, ctx.prog  # solver callees opaque; oracles + host fns are PROCs (see passes/lowering/)
  pc = int(prog.attrs.get("proc_count", 1))
  procs = {pr.attrs["name"]: pr for pr in prog.args[:pc]}
  lines: list[str] = [
    *_includes(
      ("#include <time.h>", "#include <string.h>", *solver_includes(fun)),
      dialect=ctx.recipe.dialect,
      prog=prog,
      vector_libm=ctx.recipe.vector_libm,
    ),
    "",
    *abi_status_defines(),
    "",
    *stats_c_defs(),
    "",
    *stats_c_timing_defs(),
    *solver_options_c_defs(),
    *solver_options_c_helpers(),
    "",
    *(casadi_defines() + [""] if casadi else []),
    "#ifdef __cplusplus",
    'extern "C" {',
    "#endif",
    "",
  ]
  order = _function_order(fun)
  solvers = solver_functions(fun)
  options_indices = {fn.name: i for i, fn in enumerate(solvers)}
  runtime_callees = frozenset((*procs, *options_indices))
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
  # Program order also puts a callee before its callers, and it is the only order that knows the
  # PROCs the program passes split off a ConcreteFunction's PROC (``passes/program/hoist_invariant.py``).
  pending = [pr for pr in prog.args[:pc] if pr.attrs["name"] != fun.name]
  reserved_names = _c_reserved_names(prog)

  def flush(until: str | None) -> None:
    names = [pr.attrs["name"] for pr in pending]
    count = names.index(until) + 1 if until in names else len(pending) if until is None else 0
    for pr in pending[:count]:
      lines.extend(
        (
          *_render_raw_callee(
            pr, dialect=ctx.recipe.dialect, vector_libm=ctx.recipe.vector_libm, reserved_names=reserved_names, solver_callees=runtime_callees
          ),
          "",
        )
      )
    del pending[:count]

  for fn in order if is_solver_function(fun) else order[:-1]:
    if is_solver_function(fn):
      lines.extend((*render_solver_raw(fn, options_index=options_indices[fn.name], include_external_sources=False), ""))
    else:
      flush(fn.name)
  flush(None)
  if is_solver_function(fun):
    lines += _render_solver_entry(fun, ctx.workspace_size, casadi=casadi, options_count=len(solvers))
  else:
    lines += _render_entry(
      procs[fun.name],
      fun,
      casadi=casadi,
      dialect=ctx.recipe.dialect,
      vector_libm=ctx.recipe.vector_libm,
      solver_callees=runtime_callees,
      options_count=len(solvers),
    )
  symbol = c_ident(fun.name)
  lines += ["", *render_solver_defaults(fun)]
  lines += [
    c_api_signature(symbol) + " {",
    f"  const scaly_solver_option* options[] = {{ {', '.join(f'{c_ident(fn.name)}_default_options()' for fn in solvers)} }};",
    f"  return {symbol}_with_options(arg, res, iw, w, mem, options);",
    "}",
  ]
  if casadi:
    lines += ["", *render_casadi_queries(fun, entry_workspace(fun, ctx.workspace_size, casadi=True))]
  lines += ["", "#ifdef __cplusplus", "}", "#endif"]
  return "\n".join(lines).rstrip() + "\n"


def _render_solver_entry(fun: ConcreteFunction, sz_w: int, *, casadi: bool, options_count: int) -> list[str]:
  """The entry of a root ``solver ConcreteFunction``: the null checks, then one call into its wrapper."""
  symbol = c_ident(fun.name)
  res = {name: f"res[{i}]" for i, name in enumerate(fun.output_names)}
  lines = entry_prologue(fun, entry_workspace(fun, sz_w, casadi=casadi), options_count=options_count)
  epilogue: list[str] = []
  if casadi:
    gather = casadi_gather(fun, sz_w)
    res.update(gather.ptr)
    lines += gather.setup
    epilogue = gather.epilogue
  args = [*(f"arg[{i}]" for i in range(len(fun.inputs))), *res.values(), "w", "solver_options"]
  return [*lines, f"  {symbol}_raw({', '.join(args)});", *epilogue, "  return SCALY_SUCCESS;", "}"]


def _render_observed(
  fun: ConcreteFunction, *, casadi: bool, recipe: BuildRecipe = BuildRecipe(), source_name: str | None = None
) -> tuple[_RenderCtx, str]:
  """Lower and render ``fun`` under the registered observers: one lowering, one source, and the
  expression, Program, code, and outcome sequence ``scaly.viz`` records."""
  observers = [obs for begin in _RENDER_OBSERVERS if (obs := begin(fun)) is not None]

  def observe(name: str, root: ProgramNode) -> None:
    for obs in observers:
      obs.add_program(name, root)

  def observe_expr(name: str, normalized: ConcreteFunction) -> None:
    for obs in observers:
      obs.add_normalized_expr(name, normalized)

  try:
    if casadi:
      check_casadi_layout(fun)
    ctx = _lower(fun, observe if observers else None, observe_expr if observers else None, recipe=recipe)
    source = recipe.comment(source_name or f"{c_ident(fun.name)}.c") + _render_source(ctx, casadi=casadi)
  except Exception as exc:
    for obs in observers:
      obs.finish(error=repr(exc))
    raise
  for obs in observers:
    obs.add_code(source)
    obs.finish()
  return ctx, source


def render_c_source(
  fun: Function | ConcreteFunction,
  *,
  casadi: bool = False,
  lanes: LaneCount = "auto",
  dialect: CDialect = "gnu",
  vector_libm: VectorLibm = "none",
  reciprocal: bool = False,
  cpu: CpuLevel = "generic",
) -> str:
  """Render a standalone pointer-ABI C implementation of ``fun`` and its callees. ``casadi`` adds
  the CasADi 3.8 compatible symbols.

  A ``LoweringError`` for a function that cannot be lowered propagates. There is no fallback.
  """
  fun = as_concrete(fun)
  recipe = BuildRecipe(cpu=cpu, lanes=lanes, dialect=dialect, vector_libm=vector_libm, reciprocal=reciprocal)
  return _render_observed(fun, casadi=casadi, recipe=recipe)[1]


def _check_lang(lang: str) -> None:
  if lang not in ("c", "cpp"):
    raise ValueError(f"lang must be 'c' or 'cpp', got {lang!r}")


def render_c_api_header(
  fun: Function | ConcreteFunction,
  *,
  lang: str = "c",
  casadi: bool = False,
  lanes: LaneCount = "auto",
  dialect: CDialect = "gnu",
  vector_libm: VectorLibm = "none",
  reciprocal: bool = False,
  cpu: CpuLevel = "generic",
) -> str:
  """Render the public header for ``fun``: the ABI declarations, ``SZ_*`` constants, the typed
  buffers of the chosen ``lang``, the
  sparse-output tables, and with ``casadi`` the CasADi query prototypes."""
  fun = as_concrete(fun)
  _check_lang(lang)
  if casadi:
    check_casadi_layout(fun)
  ctx = _lower(fun, recipe=BuildRecipe(cpu=cpu, lanes=lanes, dialect=dialect, vector_libm=vector_libm, reciprocal=reciprocal))
  return ctx.recipe.comment(f"{c_ident(fun.name)}.c") + _render_header(
    ctx.fun, ctx.backends, entry_workspace(fun, ctx.workspace_size, casadi=casadi), lang=lang, casadi=casadi
  )


def render_c_module(
  fun: Function | ConcreteFunction,
  *,
  header_name: str | None = None,
  source_name: str | None = None,
  lang: str = "c",
  casadi: bool = False,
  lanes: LaneCount = "auto",
  dialect: CDialect = "gnu",
  vector_libm: VectorLibm = "none",
  reciprocal: bool = False,
  cpu: CpuLevel = "generic",
) -> CModule:
  """Render ``fun`` into its header / ``.c`` pair from a single lowering. The kernel is always C.
  ``lang`` picks the header a caller includes (``f.h`` or ``f.hpp``) and ``casadi`` adds the
  CasADi 3.8 compatible symbols to both."""
  fun = as_concrete(fun)
  _check_lang(lang)
  ctx, body = _render_observed(
    fun,
    casadi=casadi,
    recipe=BuildRecipe(cpu=cpu, lanes=lanes, dialect=dialect, vector_libm=vector_libm, reciprocal=reciprocal),
    source_name=source_name,
  )
  symbol = c_ident(fun.name)
  return CModule(
    fun=fun,
    header_name=f"{symbol}.{'hpp' if lang == 'cpp' else 'h'}" if header_name is None else header_name,
    source_name=f"{symbol}.c" if source_name is None else source_name,
    body=body,
    program=ctx.prog,
    workspace_size=entry_workspace(fun, ctx.workspace_size, casadi=casadi),
    backends=ctx.backends,
    lang=lang,
    casadi=casadi,
    recipe=ctx.recipe,
  )


def workspace_size(
  fun: Function | ConcreteFunction,
  *,
  casadi: bool = False,
  lanes: LaneCount = "auto",
  dialect: CDialect = "gnu",
  vector_libm: VectorLibm = "none",
  reciprocal: bool = False,
  cpu: CpuLevel = "generic",
) -> int:
  """Doubles of scratch ``fun`` needs in ``w[]``, the value its header's ``SZ_W`` quotes.
  ``CModule.workspace_size`` is the same number without a second lowering, so prefer it when the
  module is already in hand."""
  fun = as_concrete(fun)
  return entry_workspace(
    fun,
    _lower(fun, recipe=BuildRecipe(cpu=cpu, lanes=lanes, dialect=dialect, vector_libm=vector_libm, reciprocal=reciprocal)).workspace_size,
    casadi=casadi,
  )


def write_module(
  fun: Function | ConcreteFunction | Solver,
  out_dir: Path,
  *,
  lang: str = "c",
  casadi: bool = False,
  lanes: LaneCount = "auto",
  dialect: CDialect = "gnu",
  vector_libm: VectorLibm = "none",
  reciprocal: bool = False,
  cpu: CpuLevel = "generic",
) -> CModule:
  """Write ``fun``'s header / ``.c`` into ``out_dir`` and return the module. A ``Solver`` renders its ``function``."""
  if isinstance(fun, Solver):
    fun = fun.function
  module = render_c_module(fun, lang=lang, casadi=casadi, lanes=lanes, dialect=dialect, vector_libm=vector_libm, reciprocal=reciprocal, cpu=cpu)
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / module.header_name).write_text(module.header)
  (out_dir / module.source_name).write_text(module.source)
  return module


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(
    prog="scaly_codegen", description="Render a ConcreteFunction to a header/source pair: a C kernel and a C or C++ header."
  )
  parser.add_argument("target", help="module:attribute naming a Function, concrete instance or Solver, or a zero-argument factory returning one")
  parser.add_argument("-o", "--out-dir", type=Path, default=Path(), help="directory to write into (default: cwd)")
  parser.add_argument(
    "--lang", choices=("c", "cpp"), default="c", help="header language: C structs and f_call (f.h), or C++ Buffer types in a namespace (f.hpp)"
  )
  parser.add_argument(
    "--casadi",
    action="store_true",
    help="also export the CasADi 3.8 compatible symbols (f_n_in, f_sparsity_out, f_work, ...) and hand sparse outputs over in compressed-column order",
  )
  parser.add_argument("--cpu", choices=CPU_LEVELS, default="generic", help="CPU baseline; native is host-local")
  parser.add_argument("--lanes", choices=("auto", "1", "2", "4", "8"), default="auto")
  parser.add_argument("--dialect", choices=("gnu", "c"), default="gnu")
  parser.add_argument("--vector-libm", choices=("none", "glibc"), default="none")
  parser.add_argument("--reciprocal", action="store_true", help="allow reciprocal multiplication, including changed rounding and overflow")
  args = parser.parse_args(argv)
  module_name, _, attr = args.target.partition(":")
  if not attr:
    parser.error(f"target {args.target!r} is not module:attribute")
  fun = getattr(importlib.import_module(module_name), attr)
  if not isinstance(fun, Function | ConcreteFunction | Solver):
    fun = fun()
  module = write_module(
    fun,
    args.out_dir,
    lang=args.lang,
    casadi=args.casadi,
    cpu=args.cpu,
    lanes="auto" if args.lanes == "auto" else int(args.lanes),  # ty: ignore[invalid-argument-type]
    dialect=args.dialect,
    vector_libm=args.vector_libm,
    reciprocal=args.reciprocal,
  )
  print(args.out_dir / module.header_name)
  print(args.out_dir / module.source_name)
  print(f"sz_w: {module.workspace_size}")
  print(module.recipe.comment(module.source_name).rstrip())
  if module.link_flags:
    print("link flags: " + " ".join(module.link_flags))


def _function_order(fun: ConcreteFunction) -> list[ConcreteFunction]:
  seen: set[int] = set()
  ordered: list[ConcreteFunction] = []

  def visit(fn: ConcreteFunction) -> None:
    if id(fn) in seen:
      return
    seen.add(id(fn))
    for callee in _callees(fn):
      visit(callee)
    ordered.append(fn)

  visit(fun)
  return ordered


def _callees(fun: ConcreteFunction) -> list[ConcreteFunction]:
  ret: list[ConcreteFunction] = []
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
