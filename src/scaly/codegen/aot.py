"""Lower a Function once and package its generated C, header and external build requirements."""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from scaly.codegen.external import LibraryBuild, export_libraries, libraries_of, prepare_library
from scaly.function.extern import CLibrary
from scaly.codegen.toolchain import find_c_compiler, BuildRecipe, CPU_LEVELS, CDialect, CpuLevel, LaneCount, VectorLibm
from scaly.codegen.abi import (
  ENTRY_C_TYPES,
  abi_status_defines,
  buffer_idents,
  c_api_signature,
  c_ident,
  leaf_c_type,
  sparse_layout_tables,
  workspace_c_type,
  wrapper_param_idents,
)
from scaly.codegen.c import (
  extern_declarations,
  _c_reserved_names,
  _includes,
  _render_entry,
  _render_raw_callee,
  render_program_c,
)
from scaly.codegen.casadi import (
  casadi_declarations,
  casadi_defines,
  casadi_layout,
  render_casadi_queries,
)
from scaly.codegen.cpp import render_cpp_header
from scaly.codegen.solver import (
  prepare_solver_calls,
  solver_stats_symbols,
  solver_functions,
  solver_options_c_defs,
  render_solver_defaults,
  solver_options_declarations,
)
from scaly.ir.layout import DenseLayout, SparseLayout
from scaly.function.concrete import ConcreteFunction
from scaly.function.model import Function, as_concrete
from scaly.passes.lowering import lower_function, main_proc, workspace_dtype
from scaly.passes.program import ProgramObserver
from scaly.solvers.graph import solver_backends_used
from scaly.solvers.solver import Solver
from scaly.solvers.stats import stats_c_defs

if TYPE_CHECKING:
  from collections.abc import Callable

  from scaly.ir.program import ProgramNode


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

  _libraries: tuple[CLibrary, ...] = ()

  @cached_property
  def _library_builds(self) -> tuple[LibraryBuild, ...]:
    if not self._libraries:
      return ()
    compiler = find_c_compiler()
    if compiler is None:
      raise RuntimeError("external libraries need a C compiler")
    return tuple(prepare_library(library, compiler.command) for library in self._libraries)

  @cached_property
  def frame_bytes(self) -> int:
    """Largest estimated procedure frame in bytes, including inlined locals and scalar spills.

    External C functions and solver implementations are outside this estimate.
    """
    return _frame_bytes(self.program)

  @cached_property
  def header(self) -> str:
    return self.recipe.comment(self.source_name) + _render_header(
      self.fun, self.backends, self.workspace_size, self.frame_bytes, lang=self.lang, casadi=self.casadi
    )

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
    """Resolved external-library and math linker flags for the current toolchain."""
    return (*self.recipe.link_flags, *(flag for build in self._library_builds for flag in build.link_flags))


class RenderObserver(Protocol):
  """One render's callbacks: normalized expressions, each Program stage, the C, and the outcome. ``scaly.viz.VisualizationRecording`` is the implementation in this repository."""

  def add_normalized_expr(self, name: str, fun: ConcreteFunction) -> None: ...
  def add_program(self, name: str, root: ProgramNode) -> None: ...
  def add_code(self, source: str) -> None: ...
  def finish(self, *, error: str | None = None) -> None: ...


_RENDER_OBSERVERS: list[Callable[[ConcreteFunction], RenderObserver | None]] = []


def register_render_observer(begin: Callable[[ConcreteFunction], RenderObserver | None]) -> None:
  """Watch every source render. ``begin`` is called with the function about to be rendered and
  returns an observer, or ``None`` to sit that render out. The JIT also calls it to ask whether a
  render would be watched, and renders a watched function rather than load its library from the
  index, so an observer records nothing before its first callback.

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
  libraries: tuple[CLibrary, ...]


def _lower(
  fun: ConcreteFunction,
  observe: ProgramObserver | None = None,
  observe_expr: Callable[[str, ConcreteFunction], None] | None = None,
  *,
  recipe: BuildRecipe = BuildRecipe(),
) -> _RenderCtx:
  """Lower ``fun`` once. ``workspace_size`` is the scratch it needs in ``w[]``, in ``w``'s units — the
  packed ``sz_w`` from ``passes.pack_workspace``, which also accounts for a solver wrapper passing
  its ``w`` straight to the oracle."""
  backends = solver_backends_used(fun)
  reserved = tuple(f"{c_ident(fn.name)}_{suffix}" for fn in solver_functions(fun) for suffix in ("solve", "stats", "set_options", "default_options"))
  prepared, libraries = prepare_solver_calls(fun, recipe) if backends else (fun, libraries_of(fun))
  prog = lower_function(
    prepared,
    reserved_names=reserved,
    observe=observe,
    observe_expr=observe_expr,
    reciprocal=recipe.reciprocal,
    lanes=recipe.lanes,
    frame_budget=recipe.frame_budget,
  )
  return _RenderCtx(fun, prog, backends, int(main_proc(prog).attrs.get("sz_w", 0)), recipe, libraries)


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
  input_params, output_params, workspace = wrapper_param_idents(symbol, inputs, outputs, type_prefix=f"{symbol}_")
  for ident, param, expr in zip(inputs, input_params, fun.inputs, strict=True):
    lines.append(f"typedef struct {{ SCALY_ALIGNAS(16) {leaf_c_type(expr.type.dtype)} data[{max(expr.type.storage_size, 1)}]; }} {symbol}_{ident}_t;")
    params.append(f"const {symbol}_{ident}_t* {param}")
    arg_values.append(f"{param}->data")
  for ident, param, expr in zip(outputs, output_params, fun.outputs, strict=True):
    lines.append(f"typedef struct {{ SCALY_ALIGNAS(16) {leaf_c_type(expr.type.dtype)} data[{max(expr.type.storage_size, 1)}]; }} {symbol}_{ident}_t;")
    params.append(f"{symbol}_{ident}_t* {param}")
    res_values.append(f"{param}->data")
  params.append(f"{symbol}_workspace_t* {workspace}")
  lines.append(f"typedef struct {{ SCALY_ALIGNAS(16) {workspace_c_type(fun)} data[{symbol}_SZ_W > 0 ? {symbol}_SZ_W : 1]; }} {symbol}_workspace_t;")
  pointee = ENTRY_C_TYPES[workspace_dtype(fun)]
  for runtime in (False, True) if solver_backends_used(fun) else (False,):
    suffix = "_with_options" if runtime else ""
    call_params = [*params, *(("const scaly_solver_option* const* solver_options",) if runtime else ())]
    lines += [
      f"static inline int {symbol}_call{suffix}({', '.join(call_params)}) {{",
      f"  const {pointee}* arg[{symbol}_SZ_ARG > 0 ? {symbol}_SZ_ARG : 1] = {{{', '.join(arg_values) or 'NULL'}}};",
      f"  {pointee}* res[{symbol}_SZ_RES > 0 ? {symbol}_SZ_RES : 1] = {{{', '.join(res_values) or 'NULL'}}};",
      f"  return {symbol}{suffix}(arg, res, NULL, {workspace} ? {workspace}->data : NULL, 0{', solver_options' if runtime else ''});",
      "}",
    ]
  return lines


def _sparse_leaf_tables(fun: ConcreteFunction, symbol: str) -> list[str]:
  lines: list[str] = []
  inputs, outputs = buffer_idents(fun)
  for ident, type_, layout in zip(
    (*inputs, *outputs), (*fun.input_tree.types, *fun.output_tree.types), (*fun.input_tree.layouts, *fun.output_tree.layouts), strict=True
  ):
    if isinstance(layout, DenseLayout):
      lines.append(f"#define {symbol}_{ident}_COLUMN_MAJOR 1")
      continue
    if not isinstance(layout, SparseLayout):
      continue
    prefix = f"{symbol}_{ident}"
    lines += [
      f"#define {prefix}_NROW {layout.pattern.shape[0]}",
      f"#define {prefix}_NCOL {layout.pattern.shape[1]}",
      f"#define {prefix}_NNZ {layout.pattern.nnz}",
      f"#define {prefix}_NDIM {type_.ndim}",
      f"#define {prefix}_STORAGE_SIZE {type_.storage_size}",
      f"static const int {prefix}_shape[{type_.ndim}] = {_c_array(type_.shape)};",
      *(f"static const int {prefix}_{name}[{len(values)}] = {_c_array(values)};" for name, values in sparse_layout_tables(layout) if values),
    ]
  comment = (
    "// Leaf layout metadata."
    if any(isinstance(layout, DenseLayout) for layout in (*fun.input_tree.layouts, *fun.output_tree.layouts))
    else "// Sparse leaf metadata: one local pattern shared by all batch entries."
  )
  return ["", comment, *lines] if lines else []


def _frame_bytes(prog: ProgramNode) -> int:
  return max((int(pr.attrs["frame_bytes"]) for pr in prog.args), default=0)


def _render_header(fun: ConcreteFunction, backends: tuple[str, ...], sz_w: int, frame_bytes: int, *, lang: str, casadi: bool) -> str:
  if lang == "cpp":
    return render_cpp_header(fun, backends, sz_w, frame_bytes, casadi=casadi)
  symbol = c_ident(fun.name)
  leaf_dtypes = {leaf.type.dtype for leaf in (*fun.inputs, *fun.outputs)}
  lines = [
    "#pragma once",
    "",
    *(["#include <stdbool.h>"] if any(dtype.is_bool for dtype in leaf_dtypes) else []),
    "#include <stddef.h>",
    *(["#include <stdint.h>"] if any(dtype.is_integer for dtype in leaf_dtypes) else []),
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
    f"#define {symbol}_FRAME_BYTES {frame_bytes}",
    "",
    f"// The pointer ABI for {fun.name}.",
    "#ifdef __cplusplus",
    'extern "C" {',
    "#endif",
    c_api_signature(fun) + ";",
    *solver_options_declarations(fun),
    *(f"int {solver_symbol}_stats(scaly_solver_stats* out);" for solver_symbol in solver_stats_symbols(fun)),
    *(casadi_declarations(symbol) if casadi else []),
    "#ifdef __cplusplus",
    "}",
    "#endif",
  ]
  lines += _typed_buffers(fun, symbol)
  lines += _sparse_leaf_tables(fun, symbol)
  return "\n".join(lines) + "\n"


def _render_source(ctx: _RenderCtx, *, casadi: bool) -> str:
  return (
    _render_solver_bearing_source(ctx, casadi=casadi)
    if ctx.backends
    else render_program_c(ctx.prog, ctx.fun, casadi=casadi, dialect=ctx.recipe.dialect, vector_libm=ctx.recipe.vector_libm)
  )


def _render_solver_bearing_source(ctx: _RenderCtx, *, casadi: bool) -> str:
  fun, prog = ctx.fun, ctx.prog
  count = int(prog.attrs["proc_count"])
  solvers = solver_functions(fun)
  lines = [
    *_includes(prog=prog, dialect=ctx.recipe.dialect, vector_libm=ctx.recipe.vector_libm),
    "",
    *abi_status_defines(),
    "",
    *stats_c_defs(),
    *solver_options_c_defs(),
    "",
    *(casadi_defines() + [""] if casadi else []),
    "#ifdef __cplusplus",
    'extern "C" {',
    "#endif",
    "",
    *extern_declarations(prog),
  ]
  for fn in solvers:
    symbol = c_ident(fn.name)
    lines.extend(
      [
        f"void {symbol}_set_options(const scaly_solver_option* options);",
        f"int {symbol}_stats(scaly_solver_stats* out);",
      ]
    )
  for proc in prog.args[: count - 1]:
    lines.extend(
      [*_render_raw_callee(proc, dialect=ctx.recipe.dialect, vector_libm=ctx.recipe.vector_libm, reserved_names=_c_reserved_names(prog)), ""]
    )
  entry = _render_entry(
    main_proc(prog),
    fun,
    casadi=casadi,
    dialect=ctx.recipe.dialect,
    vector_libm=ctx.recipe.vector_libm,
    options_count=len(solvers),
    entry_setup=[f"  {c_ident(fn.name)}_set_options(solver_options[{i}]);" for i, fn in enumerate(solvers)],
  )
  lines.extend(entry)
  symbol = c_ident(fun.name)
  lines += ["", *render_solver_defaults(fun)]
  lines += [
    c_api_signature(fun) + " {",
    f"  const scaly_solver_option* options[] = {{ {', '.join(f'{c_ident(fn.name)}_default_options()' for fn in solvers)} }};",
    f"  return {symbol}_with_options(arg, res, iw, w, mem, options);",
    "}",
  ]
  if casadi:
    lines.extend(["", *render_casadi_queries(fun, ctx.workspace_size)])
  lines.extend(["", "#ifdef __cplusplus", "}", "#endif"])
  return "\n".join(lines).rstrip() + "\n"


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
      fun = casadi_layout(fun)
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
  frame_budget: int = 65536,
) -> str:
  """Render a standalone pointer-ABI C implementation of ``fun`` and its callees. ``casadi`` adds
  the CasADi 3.8 compatible symbols.

  A ``LoweringError`` for a function that cannot be lowered propagates. There is no fallback.
  """
  fun = as_concrete(fun)
  recipe = BuildRecipe(cpu=cpu, lanes=lanes, dialect=dialect, vector_libm=vector_libm, reciprocal=reciprocal, frame_budget=frame_budget)
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
  frame_budget: int = 65536,
) -> str:
  """Render the public header for ``fun``: the ABI declarations, ``SZ_*`` constants, the typed
  buffers of the chosen ``lang``, the
  sparse-output tables, and with ``casadi`` the CasADi query prototypes."""
  fun = as_concrete(fun)
  _check_lang(lang)
  if casadi:
    fun = casadi_layout(fun)
  ctx = _lower(
    fun, recipe=BuildRecipe(cpu=cpu, lanes=lanes, dialect=dialect, vector_libm=vector_libm, reciprocal=reciprocal, frame_budget=frame_budget)
  )
  return ctx.recipe.comment(f"{c_ident(fun.name)}.c") + _render_header(
    ctx.fun, ctx.backends, ctx.workspace_size, _frame_bytes(ctx.prog), lang=lang, casadi=casadi
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
  frame_budget: int = 65536,
) -> CModule:
  """Render ``fun`` into its header / ``.c`` pair from a single lowering. The kernel is always C.
  ``lang`` picks the header a caller includes (``f.h`` or ``f.hpp``) and ``casadi`` adds the
  CasADi 3.8 compatible symbols to both."""
  fun = as_concrete(fun)
  _check_lang(lang)
  ctx, body = _render_observed(
    fun,
    casadi=casadi,
    recipe=BuildRecipe(cpu=cpu, lanes=lanes, dialect=dialect, vector_libm=vector_libm, reciprocal=reciprocal, frame_budget=frame_budget),
    source_name=source_name,
  )
  symbol = c_ident(fun.name)
  return CModule(
    fun=ctx.fun,
    header_name=f"{symbol}.{'hpp' if lang == 'cpp' else 'h'}" if header_name is None else header_name,
    source_name=f"{symbol}.c" if source_name is None else source_name,
    body=body,
    program=ctx.prog,
    workspace_size=ctx.workspace_size,
    backends=ctx.backends,
    lang=lang,
    casadi=casadi,
    recipe=ctx.recipe,
    _libraries=ctx.libraries,
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
  frame_budget: int = 65536,
) -> int:
  """Entries of scratch ``fun`` needs in ``w[]``, the value its header's ``SZ_W`` quotes: doubles
  when every leaf is float64, floats when every leaf is float32, and bytes otherwise.
  ``CModule.workspace_size`` is the same number without a second lowering, so prefer it when the
  module is already in hand."""
  fun = as_concrete(fun)
  if casadi:
    fun = casadi_layout(fun)
  return _lower(
    fun, recipe=BuildRecipe(cpu=cpu, lanes=lanes, dialect=dialect, vector_libm=vector_libm, reciprocal=reciprocal, frame_budget=frame_budget)
  ).workspace_size


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
  frame_budget: int = 65536,
) -> CModule:
  """Write ``fun``'s header / ``.c`` into ``out_dir`` and return the module. A ``Solver`` renders its ``function``."""
  if isinstance(fun, Solver):
    fun = fun.function
  module = render_c_module(
    fun, lang=lang, casadi=casadi, lanes=lanes, dialect=dialect, vector_libm=vector_libm, reciprocal=reciprocal, cpu=cpu, frame_budget=frame_budget
  )
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / module.header_name).write_text(module.header)
  (out_dir / module.source_name).write_text(module.source)
  if module._libraries:
    manifest = {
      "source": module.source_name,
      "compile_flags": ["-O2", *module.recipe.cpu_flags, "-fno-math-errno"],
      "link": ["m", *(["mvec"] if module.recipe.vector_libm == "glibc" else [])],
      "libraries": export_libraries(module._libraries, out_dir),
    }
    (out_dir / f"{c_ident(module.fun.name)}.build.json").write_text(json.dumps(manifest, indent=2) + "\n")
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
  parser.add_argument("--frame-budget", type=int, default=65536, help="estimated frame budget in bytes; zero spills all buffers")
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
    frame_budget=args.frame_budget,
  )
  print(args.out_dir / module.header_name)
  print(args.out_dir / module.source_name)
  print(f"sz_w: {module.workspace_size}")
  print(f"frame_bytes: {module.frame_bytes}")
  print(module.recipe.comment(module.source_name).rstrip())
  if module._libraries:
    print("build manifest: " + str(args.out_dir / f"{c_ident(module.fun.name)}.build.json"))
  flags = module.recipe.link_flags
  if flags:
    print("link flags: " + " ".join(flags))


if __name__ == "__main__":
  main()
