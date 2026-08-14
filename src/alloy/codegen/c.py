"""Universal-ABI C codegen orchestrator.

The CPU renderer is Program IR (``codegen/program_c``): ``lowering.lower_function`` lowers a
host ``Function`` to a verified ``PROGRAM`` and ``program_c`` renders it. This module is the thin
entry point the JIT / AOT consumers call:

- ``render_c_api_header`` / ``render_c_module`` emit the public ``.h`` / ``.c`` pair.
- ``render_c_source`` produces the translation unit. For a function with no solver in its call
  graph it delegates straight to ``program_c``. For a **solver-bearing** graph it orchestrates:
  every non-solver Function (oracle, host caller, intermediate) is a Program-IR ``_raw`` and each
  ``SolverFunction`` is the ``codegen/solver_c`` wrapper template that drives its (Program-IR)
  oracle Functions. The solver wrapper is the one sanctioned non-Program-IR path (rule 6 in
  ``docs/program_ir_migration.md``); there is no legacy tape-based scalar renderer anymore.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from alloy.abi import c_api_signature
from alloy.codegen.solver_c import (
  is_solver_function,
  render_solver_raw,
  solver_backends_used,
  solver_callees,
  solver_includes,
  solver_stats_symbols,
)
from alloy.expr import topo
from alloy.function import Function
from alloy.ops import Ops
from alloy.passes import ProgramObserver


@dataclass(frozen=True, slots=True)
class CModule:
  header_name: str
  source_name: str
  header: str
  source: str


def _uses_solver(fun: Function) -> bool:
  """True if ``fun`` is a solver or reaches a ``SOLVER_CALL`` anywhere in its graph (so a plain
  function that *calls* a solver is rendered by the solver-bearing orchestrator, not the pure
  Program-IR path — Program IR deliberately does not lower ``SOLVER_CALL``).
  ``solver_backends_used`` traverses callees and ``SOLVER_CALL`` nodes, which ``_function_order``
  does not."""
  return bool(solver_backends_used(fun))


def _workspace_size(fun: Function) -> int:
  """Doubles of scratch ``fun`` needs in ``w[]`` — its Program-IR packed ``sz_w`` (set by
  ``passes.pack_workspace``, which also accounts for a solver wrapper passing its ``w`` straight to
  the oracle). Identical to the value baked into the rendered source; kept for AOT / benchmark
  consumers that import it."""
  from alloy.codegen.program_c import program_ir_sz_w
  from alloy.lowering import lower_function

  if is_solver_function(fun):
    return _solver_root_workspace(lower_function(fun), fun.name)

  return program_ir_sz_w(fun)


def _solver_root_workspace(prog: object, name: str) -> int:
  from alloy.program import PNode

  assert isinstance(prog, PNode)
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


def _c_ident(name: str) -> str:
  ident = re.sub(r"\W", "_", name)
  if ident in ("w", "arg", "res", "iw", "mem"):  # must match program_c._c_ident's reserved-name mangling
    ident += "_"
  return f"_{ident}" if ident[:1].isdigit() else ident


def _c_float(value: float) -> str:
  if math.isnan(value):
    return "NAN"
  if math.isinf(value):
    return "INFINITY" if value > 0 else "-INFINITY"
  return repr(float(value))


def _abi_status_defines() -> list[str]:
  return [
    "#ifndef ALLOY_SUCCESS",
    "#define ALLOY_SUCCESS 0",
    "#endif",
    "#ifndef ALLOY_ERR_NULL_ABI",
    "#define ALLOY_ERR_NULL_ABI 1",
    "#endif",
    "#ifndef ALLOY_ERR_NULL_WORK",
    "#define ALLOY_ERR_NULL_WORK 2",
    "#endif",
    "#ifndef ALLOY_ERR_NULL_RESULT",
    "#define ALLOY_ERR_NULL_RESULT 3",
    "#endif",
    "#ifndef ALLOY_ERR_NULL_INPUT",
    "#define ALLOY_ERR_NULL_INPUT 4",
    "#endif",
  ]


def _typed_cpp_wrapper(fun: Function, symbol: str) -> list[str]:
  params: list[str] = []
  arg_values: list[str] = []
  res_values: list[str] = []
  checks: list[str] = []
  for name, expr in zip(fun.input_names, fun.inputs, strict=True):
    assert isinstance(name, str)
    ident = _c_ident(name)
    type_name = f"{symbol}_{ident}_in"
    params.append(f"const {type_name}& in_{ident}")
    arg_values.append(f"in_{ident}.data")
    checks.append(f'static_assert(sizeof({type_name}) == sizeof(double) * {expr.size}, "{type_name} size mismatch");')
  for name, expr in zip(fun.output_names, fun.outputs, strict=True):
    assert isinstance(name, str)
    ident = _c_ident(name)
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


def render_c_api_header(fun: Function, *, typed_buffers: bool = True) -> str:
  from alloy.solvers.stats import stats_c_defs

  symbol = _c_ident(fun.name)
  lines = [
    "#pragma once",
    "",
    *(["#include <stdint.h>", "", *stats_c_defs(), ""] if _uses_solver(fun) else []),
    *_abi_status_defines(),
    "",
    f"#define {symbol}_SZ_ARG {len(fun.inputs)}",
    f"#define {symbol}_SZ_RES {len(fun.outputs)}",
    f"#define {symbol}_SZ_IW 0",
    f"#define {symbol}_SZ_W {_workspace_size(fun)}",
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
      lines.append(f"typedef struct {{ double data[{expr.size}]; }} {symbol}_{_c_ident(name)}_in;")
    for name, expr in zip(fun.output_names, fun.outputs, strict=True):
      lines.append(f"typedef struct {{ double data[{expr.size}]; }} {symbol}_{_c_ident(name)}_out;")
    lines += _typed_cpp_wrapper(fun, symbol)
  sparse_outputs = [(name, sp) for name, sp in zip(fun.output_names, fun.output_sparsities, strict=True) if sp is not None]
  if sparse_outputs:
    lines += ["", "// Sparse output metadata for compact derivative buffers."]
    for name, sp in sparse_outputs:
      assert sp is not None
      prefix = f"{symbol}_{_c_ident(name)}"
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


def render_c_source(fun: Function) -> str:
  """Render a standalone universal-ABI C implementation of ``fun`` and its callees.

  A function with no solver in its call graph lowers entirely through Program IR. A solver-bearing
  graph is orchestrated by ``_render_solver_bearing_source``: Program-IR ``_raw`` functions for
  every non-solver Function plus the ``solver_c`` wrapper for each ``SolverFunction``. A
  ``LoweringError`` (e.g. a still-deferred mixed-device CALL) propagates — there is no fallback.
  See docs/program_ir_migration.md.
  """
  from alloy.viz._recording import begin_recording

  recording = begin_recording(fun)
  observe = None if recording is None else recording.add_program
  try:
    source = _render_c_source(fun, observe=observe)
  except Exception as exc:
    if recording is not None:
      recording.finish(error=repr(exc))
    raise
  if recording is not None:
    recording.add_code(source)
    recording.finish()
  return source


def _render_c_source(fun: Function, observe: ProgramObserver | None = None) -> str:
  from alloy.codegen.program_c import render_program_c_source

  if not _uses_solver(fun):
    return render_program_c_source(fun, observe=observe)
  solver_stats_symbols(fun)  # validate duplicate solver symbols before lowering or compilation
  return _render_solver_bearing_source(fun, observe=observe)


def _render_solver_bearing_source(fun: Function, observe: ProgramObserver | None = None) -> str:
  """One translation unit for a solver-bearing graph. The non-solver Functions (oracles, the host
  caller, any intermediates) are Program-IR ``_raw`` callees; each ``SolverFunction`` is the
  ``solver_c`` wrapper driving them. ``_function_order`` is topological — a solver sits after its
  oracle PROCs and before the function that calls it — so emitting definitions in that order never
  forward-references a ``_raw``."""
  from alloy.codegen.program_c import _ABI_DEFINES, _includes, _render_entry, _render_raw_callee
  from alloy.lowering import lower_function
  from alloy.solvers.stats import stats_c_defs, stats_c_timing_defs

  prog = lower_function(fun, observe=observe)  # solver callees opaque; oracles + host fns are PROCs (see lowering.py)
  pc = int(prog.attrs.get("proc_count", 1))
  procs = {pr.attrs["name"]: pr for pr in prog.args[:pc]}
  lines: list[str] = [
    *_includes(("#include <time.h>", *solver_includes(fun))),
    "",
    *_ABI_DEFINES,
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
  from alloy.codegen.solver_c import external_oracles

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
    lines += _render_solver_entry(fun, _solver_root_workspace(prog, fun.name))
  else:
    lines += _render_entry(procs[fun.name], fun)
  lines += ["", "#ifdef __cplusplus", "}", "#endif"]
  return "\n".join(lines).rstrip() + "\n"


def _render_solver_entry(fun: Function, sz_w: int) -> list[str]:
  symbol = _c_ident(fun.name)
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


def render_c_module(fun: Function, *, header_name: str | None = None, source_name: str | None = None, typed_buffers: bool = True) -> CModule:
  symbol = _c_ident(fun.name)
  header_name = f"{symbol}.h" if header_name is None else header_name
  source_name = f"{symbol}.c" if source_name is None else source_name
  include_name = header_name.replace("\\", "\\\\").replace('"', '\\"')
  source = f'#include "{include_name}"\n\n' + render_c_source(fun)
  return CModule(header_name, source_name, render_c_api_header(fun, typed_buffers=typed_buffers), source)


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
    # SolverFunctions render via a custom template that calls the oracle (and for NLP, the
    # derivative Functions) — these aren't reachable through the solver's own output graph (it
    # only contains SOLVER_CALL nodes), so surface them explicitly here.
    for callee in solver_callees(fun):
      if id(callee) not in seen:
        seen.add(id(callee))
        ret.append(callee)
    return ret
  for node in topo(fun.outputs):
    if node.op not in {Ops.CALL, Ops.MAP}:
      continue
    callee = node.attrs["callee"]
    if id(callee) not in seen:
      seen.add(id(callee))
      ret.append(callee)
  return ret
