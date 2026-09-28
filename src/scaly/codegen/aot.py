"""AOT: lower a ``Function`` once and package the result as the :class:`CModule` that both the
file-writing driver here and ``codegen/jit.py`` consume.

``_lower`` is the single render context. It lowers ``fun`` exactly once and holds everything the
artifacts read off that lowering, so the header's ``SZ_W``, the entry's null check and a consumer's
workspace allocation cannot disagree. The C header is rendered here; output adapters
(``codegen/adapter.py``) may replace it (the C++ one, ``codegen/cpp.py``) or add a layer to either
(CasADi's, ``codegen/casadi.py``), and are found by name. A function with no extern callee in its call graph renders entirely
through ``codegen/c``. An **extern-bearing** graph is orchestrated here: every other Function
(dependency, host caller, intermediate) is a Program-IR ``_raw``, and each Function with an extern
body (``function/extern.py``; a solver, say) is the C its callee renders, calling those ``_raw``s,
the one sanctioned non-Program-IR path. What the callees need beyond that (includes, type
definitions, prototypes, link flags) comes off their ``build_requirements``, merged here once.
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from scaly.codegen.abi import abi_status_defines, buffer_idents, c_api_signature, c_ident
from scaly.codegen.adapter import Adapter, HeaderSpec, available_adapters, entry_workspace, resolve_adapters
from scaly.codegen.c import (
  _includes,
  _render_entry,
  _render_raw_callee,
  adapter_defines,
  adapter_sources,
  entry_prologue,
  render_program_c,
  wrap_entry,
)
from scaly.function import ConcreteFunction, Function
from scaly.function.extern import BuildRequirements, ExternRenderCtx, LinkResolver, extern_functions
from scaly.ir.expr import callees_of, topo
from scaly.passes.lowering import lower_function, main_proc
from scaly.passes.program import ProgramObserver

if TYPE_CHECKING:
  from collections.abc import Callable

  from scaly.ir.program import ProgramNode
  from scaly.ir.types import SparsityType


@dataclass(frozen=True, slots=True)
class Requirements:
  """The ``build_requirements`` of every extern callee in one translation unit, merged.

  Includes are deduplicated and sorted, system headers first. A block of type definitions or of
  source definitions appears once however many callees ask for it, in the order first asked, and
  prototypes keep the callees' order. Callees sharing a link resolver are resolved together, once,
  with the sorted union of their library names; ``link_flags`` does that on demand, because a
  resolver raises when a library is missing and rendering has to work without one."""

  includes: tuple[str, ...] = ()
  header_types: tuple[str, ...] = ()
  source_blocks: tuple[tuple[str, ...], ...] = ()
  declarations: tuple[str, ...] = ()
  links: tuple[tuple[LinkResolver, tuple[str, ...]], ...] = ()
  isolated: bool = False
  versions: tuple[tuple[str, str], ...] = ()

  @staticmethod
  def merge(requirements: tuple[BuildRequirements, ...]) -> Requirements:
    includes = sorted({line for req in requirements for line in req.includes}, key=lambda line: (not line.startswith("#include <"), line))
    header_types = tuple(dict.fromkeys(block for req in requirements for block in req.header_types))
    links: dict[LinkResolver, set[str]] = {}
    for req in requirements:
      if req.link_flags is not None:
        links.setdefault(req.link_flags, set()).update(req.libraries)
    return Requirements(
      includes=tuple(includes),
      header_types=tuple(line for block in header_types for line in block),
      source_blocks=tuple(dict.fromkeys(block for req in requirements for block in req.source_blocks)),
      declarations=tuple(line for req in requirements for line in req.declarations),
      links=tuple((resolver, tuple(sorted(libraries))) for resolver, libraries in links.items()),
      isolated=any(req.isolated for req in requirements),
      versions=tuple(sorted({v for req in requirements for v in req.versions})),
    )

  def link_flags(self) -> tuple[str, ...]:
    return tuple(flag for resolver, libraries in self.links for flag in resolver(libraries))


def requirements(externs: tuple[ConcreteFunction, ...]) -> Requirements:
  """The merged requirements of the Functions with extern bodies in one translation unit."""
  return Requirements.merge(tuple(fn.extern.build_requirements(fn) for fn in externs if fn.extern is not None))


@dataclass(frozen=True)
class CModule:
  """One rendered function: the header and the ``.c`` to write, plus what a consumer needs to
  compile and call them. ``body`` is the translation unit. ``program`` is the optimized Program IR
  that produced it. ``workspace_size`` is the entry's ``w[]`` length, and ``externs`` lists the
  Functions with extern bodies (solvers, say) that the function reaches. ``adapters`` names the
  output adapters applied (``"cpp"`` for the C++ header, ``"casadi"`` for the CasADi-compatible
  symbols); ``typed_buffers`` toggles the C header's structs and ``_call`` wrapper.

  ``header``, ``source`` and ``link_flags`` are rendered on first access. The JIT compiles ``body``
  and asks for none of them; for a big sparse function the header alone is larger than the source.
  """

  fun: ConcreteFunction
  header_name: str
  source_name: str
  body: str
  program: ProgramNode
  workspace_size: int
  externs: tuple[ConcreteFunction, ...]
  typed_buffers: bool
  adapters: tuple[str, ...] = ()

  @cached_property
  def requirements(self) -> Requirements:
    return requirements(self.externs)

  @cached_property
  def header(self) -> str:
    return _render_header(
      self.fun, self.requirements, self.workspace_size, typed_buffers=self.typed_buffers, adapters=resolve_adapters(self.adapters)
    )

  @cached_property
  def source(self) -> str:
    """The ``.c`` as written: ``body`` behind an include of the paired header, unless an adapter's
    header cannot be included from C (the C++ one), and then ``body`` alone."""
    if not all(a.source_includes_header for a in resolve_adapters(self.adapters)):
      return self.body
    include = self.header_name.replace("\\", "\\\\").replace('"', '\\"')
    return f'#include "{include}"\n\n{self.body}'

  @cached_property
  def link_flags(self) -> tuple[str, ...]:
    """Compiler/linker flags the extern callees need: include, lib, rpath and ``-l`` flags. Empty
    without one. Resolved on demand because a resolver raises when a library or header is missing
    (a solver's ``SolverLibraryError``): rendering has to stay possible on a machine without the
    vendored solver stack, and against a backend that has no library at all (the fake backends in
    ``tests/solvers/test_registry.py``)."""
    return self.requirements.link_flags()


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
  externs: tuple[ConcreteFunction, ...]
  workspace_size: int


def _lower(
  fun: ConcreteFunction, observe: ProgramObserver | None = None, observe_expr: Callable[[str, ConcreteFunction], None] | None = None
) -> _RenderCtx:
  """Lower ``fun`` once. ``workspace_size`` is the doubles of scratch it needs in ``w[]`` — the
  packed ``sz_w`` from ``passes.pack_workspace``, which also accounts for an extern callee passing
  its ``w`` straight to its dependencies."""
  externs = extern_functions(fun)  # refuses two extern Functions with one C symbol, before lowering
  prog = lower_function(fun, observe=observe, observe_expr=observe_expr)
  sz_w = _extern_root_workspace(prog, fun.name) if fun.extern is not None else int(main_proc(prog).attrs.get("sz_w", 0))
  return _RenderCtx(fun, prog, externs, sz_w)


def _extern_root_workspace(prog: ProgramNode, name: str) -> int:
  pc = int(prog.attrs.get("proc_count", 0))
  dependencies = set(prog.attrs.get("extern_deps", {}).get(name, ()))
  return max(
    (
      int(prog.attrs.get("extern_workspace", {}).get(name, 0)),
      *(int(pr.attrs.get("sz_w", 0)) for pr in prog.args[:pc] if pr.attrs["name"] in dependencies),
    )
  )


def _c_array(values: tuple[int, ...]) -> str:
  return "{" + ", ".join(str(v) for v in values) + "}"


_ALIGNAS = [
  "#ifndef SCALY_ALIGNAS",
  "#ifdef __cplusplus",
  "#define SCALY_ALIGNAS(n) alignas(n)",
  "#else",
  "#define SCALY_ALIGNAS(n) _Alignas(n)",
  "#endif",
  "#endif",
]


def _typed_buffers(fun: ConcreteFunction, symbol: str) -> list[str]:
  """One 16-byte aligned struct per input and output, the caller-owned workspace struct, and a
  ``_call`` wrapper that builds the pointer arrays. Plain C; the same text compiles as C++."""
  params: list[str] = []
  arg_values: list[str] = []
  res_values: list[str] = []
  lines = ["", "// Typed buffers: one struct per input and output, and the caller-owned workspace."]
  inputs, outputs = buffer_idents(fun)
  # An array of more than one axis is flat in the buffer; its shape and order are stated beside it.
  layout = lambda expr: f"  // {' x '.join(map(str, expr.shape))}, row-major (C order)" if len(expr.shape) > 1 else ""
  for ident, expr in zip(inputs, fun.inputs, strict=True):
    lines.append(f"typedef struct {{ SCALY_ALIGNAS(16) double data[{max(expr.size, 1)}]; }} {symbol}_{ident}_t;{layout(expr)}")
    params.append(f"const {symbol}_{ident}_t* {ident}")
    arg_values.append(f"{ident}->data")
  for ident, expr in zip(outputs, fun.outputs, strict=True):
    lines.append(f"typedef struct {{ SCALY_ALIGNAS(16) double data[{max(expr.size, 1)}]; }} {symbol}_{ident}_t;{layout(expr)}")
    params.append(f"{symbol}_{ident}_t* {ident}")
    res_values.append(f"{ident}->data")
  params.append(f"{symbol}_workspace_t* workspace")
  return [
    *lines,
    f"typedef struct {{ SCALY_ALIGNAS(16) double data[{symbol}_SZ_W > 0 ? {symbol}_SZ_W : 1]; }} {symbol}_workspace_t;",
    f"static inline int {symbol}_call({', '.join(params)}) {{",
    f"  const double* arg[{symbol}_SZ_ARG > 0 ? {symbol}_SZ_ARG : 1] = {{{', '.join(arg_values) or 'NULL'}}};",
    f"  double* res[{symbol}_SZ_RES > 0 ? {symbol}_SZ_RES : 1] = {{{', '.join(res_values) or 'NULL'}}};",
    f"  return {symbol}(arg, res, NULL, workspace ? workspace->data : NULL, 0);",
    "}",
  ]


def _sparse_tables(fun: ConcreteFunction, symbol: str, sparsities: tuple[SparsityType | None, ...]) -> list[str]:
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


def header_sparsities(fun: ConcreteFunction, adapters: Sequence[Adapter]) -> tuple[SparsityType | None, ...]:
  """The patterns a header describes: the native order, or the one an adapter hands the outputs
  over in (CasADi's compressed-column order)."""
  for a in adapters:
    if a.sparsities is not None:
      return a.sparsities(fun)
  return tuple(fun.output_sparsities)


def _header_spec(fun: ConcreteFunction, req: Requirements, sz_w: int, *, typed_buffers: bool, adapters: Sequence[Adapter]) -> HeaderSpec:
  symbol = c_ident(fun.name)
  return HeaderSpec(
    fun=fun,
    sz_w=sz_w,
    typed_buffers=typed_buffers,
    types=req.header_types,
    defines=tuple(line for a in adapters for line in a.defines),
    declarations=(*req.declarations, *(line for a in adapters if a.declarations is not None for line in a.declarations(symbol))),
    sparsities=header_sparsities(fun, adapters),
  )


def _render_header(fun: ConcreteFunction, req: Requirements, sz_w: int, *, typed_buffers: bool, adapters: Sequence[Adapter]) -> str:
  spec = _header_spec(fun, req, sz_w, typed_buffers=typed_buffers, adapters=adapters)
  for a in adapters:
    if a.header is not None:
      return a.header(spec)
  return render_c_header(spec)


def render_c_header(spec: HeaderSpec) -> str:
  """The C header: ABI declarations, ``SZ_*`` constants, the typed buffers unless
  ``spec.typed_buffers`` is off, and the sparse-output tables."""
  fun, sz_w, typed_buffers = spec.fun, spec.sz_w, spec.typed_buffers
  symbol = c_ident(fun.name)
  lines = [
    "#pragma once",
    "",
    "#include <stddef.h>",
    *(["#include <stdint.h>", "", *spec.types] if spec.types else []),
    "",
    *abi_status_defines(guarded=True),
    *(["", *spec.defines] if spec.defines else []),
    *(["", *_ALIGNAS] if typed_buffers else []),
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
    *spec.declarations,
    "#ifdef __cplusplus",
    "}",
    "#endif",
  ]
  if typed_buffers:
    lines += _typed_buffers(fun, symbol)
  lines += _sparse_tables(fun, symbol, spec.sparsities)
  return "\n".join(lines) + "\n"


def _render_source(ctx: _RenderCtx, adapters: tuple[Adapter, ...]) -> str:
  return _render_extern_bearing_source(ctx, adapters) if ctx.externs else render_program_c(ctx.prog, ctx.fun, adapters)


def _render_extern(fn: ConcreteFunction) -> list[str]:
  """The C its callee renders for a Function with an extern body."""
  assert fn.extern is not None
  symbol = c_ident(fn.name)
  return fn.extern.render(fn, ExternRenderCtx(symbol=symbol, raw_symbol=f"{symbol}_raw"))


def _render_extern_bearing_source(ctx: _RenderCtx, adapters: tuple[Adapter, ...]) -> str:
  """One translation unit for a graph that reaches extern callees. The other Functions (their
  dependencies, the host caller, any intermediates) are Program-IR ``_raw`` callees; each Function
  with an extern body is the C its callee renders, calling them. ``_function_order`` is
  topological — an extern Function sits after its dependencies' PROCs and before the function that
  calls it — so emitting each one after the PROCs up to its dependencies never forward-references
  a ``_raw``."""
  fun, prog = ctx.fun, ctx.prog  # extern Functions opaque; their dependencies and host fns are PROCs (see passes/lowering.py)
  pc = int(prog.attrs.get("proc_count", 1))
  procs = {pr.attrs["name"]: pr for pr in prog.args[:pc]}
  req = requirements(ctx.externs)
  lines: list[str] = [
    *_includes(req.includes),
    "",
    *abi_status_defines(),
    "",
    *(line for block in req.source_blocks for line in (*block, "")),
    *adapter_defines(adapters),
    "#ifdef __cplusplus",
    'extern "C" {',
    "#endif",
    "",
  ]
  order = _function_order(fun)
  external_sources: list[str] = []
  raw_definitions: dict[str, str] = {}
  for fn in order:
    for source in fn.extern.extern_sources() if fn.extern is not None else ():
      previous = raw_definitions.setdefault(source.raw_symbol, source.source)
      if previous != source.source:
        raise ValueError(f"extern source symbol {source.raw_symbol!r} has conflicting source definitions")
      if source.source and source.source not in external_sources:
        external_sources.append(source.source)
  for source in external_sources:
    lines += [*source.splitlines(), ""]
  # Program order also puts a callee before its callers, and it is the only order that knows the
  # PROCs the program passes split off a Function's PROC (``passes/program/hoist_invariant.py``).
  pending = [pr for pr in prog.args[:pc] if pr.attrs["name"] != fun.name]

  def flush(until: str | None) -> None:
    names = [pr.attrs["name"] for pr in pending]
    count = names.index(until) + 1 if until in names else len(pending) if until is None else 0
    for pr in pending[:count]:
      lines.extend((*_render_raw_callee(pr), ""))
    del pending[:count]

  for fn in order if fun.extern is not None else order[:-1]:
    if fn.extern is not None:
      lines.extend((*_render_extern(fn), ""))
    else:
      flush(fn.name)
  flush(None)
  if fun.extern is not None:
    lines += _render_extern_entry(fun, ctx.workspace_size, adapters)
  else:
    lines += _render_entry(procs[fun.name], fun, adapters)
  lines += adapter_sources(fun, entry_workspace(fun, ctx.workspace_size, adapters), adapters)
  lines += ["", "#ifdef __cplusplus", "}", "#endif"]
  return "\n".join(lines).rstrip() + "\n"


def _render_extern_entry(fun: ConcreteFunction, sz_w: int, adapters: tuple[Adapter, ...]) -> list[str]:
  """The entry of a root Function with an extern body: the null checks, then one call into it."""
  symbol = c_ident(fun.name)
  res = {name: f"res[{i}]" for i, name in enumerate(fun.output_names)}
  lines = entry_prologue(fun, entry_workspace(fun, sz_w, adapters))
  setup, epilogue = wrap_entry(fun, sz_w, adapters, res)
  lines += setup
  args = [*(f"arg[{i}]" for i in range(len(fun.inputs))), *res.values(), "w"]
  return [*lines, f"  {symbol}_raw({', '.join(args)});", *epilogue, "  return SCALY_SUCCESS;", "}"]


def _render_observed(fun: ConcreteFunction, adapters: tuple[Adapter, ...]) -> tuple[_RenderCtx, str]:
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
    _check_layout(fun, adapters)
    ctx = _lower(fun, observe if observers else None, observe_expr if observers else None)
    source = _render_source(ctx, adapters)
  except Exception as exc:
    for obs in observers:
      obs.finish(error=repr(exc))
    raise
  for obs in observers:
    obs.add_code(source)
    obs.finish()
  return ctx, source


def _check_layout(fun: ConcreteFunction, adapters: tuple[Adapter, ...]) -> None:
  for a in adapters:
    if a.check is not None:
      a.check(fun)


def render_c_source(fun: Function, *, adapters: Sequence[str] = ()) -> str:
  """Render a standalone pointer-ABI C implementation of ``fun`` and its callees, with what the
  named output ``adapters`` add to the source (``"casadi"``: the CasADi 3.8 compatible symbols).

  A ``LoweringError`` (e.g. a still-deferred mixed-device CALL) propagates — there is no fallback.
  """
  fun = fun.concrete
  return _render_observed(fun, resolve_adapters(adapters))[1]


def render_c_api_header(fun: Function, *, typed_buffers: bool = True, adapters: Sequence[str] = ()) -> str:
  """Render the public header for ``fun``: the ABI declarations, ``SZ_*`` constants, the typed
  buffers (``typed_buffers=False`` omits them from the C header), the sparse-output tables, and
  what the named output ``adapters`` add (``"cpp"`` renders the C++ header instead, ``"casadi"``
  adds the CasADi query prototypes)."""
  fun = fun.concrete
  resolved = resolve_adapters(adapters)
  _check_layout(fun, resolved)
  ctx = _lower(fun)
  return _render_header(
    ctx.fun, requirements(ctx.externs), entry_workspace(fun, ctx.workspace_size, resolved), typed_buffers=typed_buffers, adapters=resolved
  )


def render_c_module(
  fun: Function,
  *,
  header_name: str | None = None,
  source_name: str | None = None,
  typed_buffers: bool = True,
  adapters: Sequence[str] = (),
) -> CModule:
  """Render ``fun`` into its header / ``.c`` pair from a single lowering. The kernel is always C.
  The named output ``adapters`` may replace the header (``"cpp"``: ``f.hpp``) or add a layer to
  both files (``"casadi"``: the CasADi 3.8 compatible symbols)."""
  fun = fun.concrete
  resolved = resolve_adapters(adapters)
  ctx, body = _render_observed(fun, resolved)
  symbol = c_ident(fun.name)
  suffix = next((a.header_suffix for a in resolved if a.header is not None), "h")
  return CModule(
    fun=fun,
    header_name=f"{symbol}.{suffix}" if header_name is None else header_name,
    source_name=f"{symbol}.c" if source_name is None else source_name,
    body=body,
    program=ctx.prog,
    workspace_size=entry_workspace(fun, ctx.workspace_size, resolved),
    externs=ctx.externs,
    typed_buffers=typed_buffers,
    adapters=tuple(a.name for a in resolved),
  )


def workspace_size(fun: Function, *, adapters: Sequence[str] = ()) -> int:
  """Doubles of scratch ``fun`` needs in ``w[]`` — the value its header's ``SZ_W`` quotes.
  ``CModule.workspace_size`` is the same number without a second lowering, so prefer it when the
  module is already in hand."""
  fun = fun.concrete
  return entry_workspace(fun, _lower(fun).workspace_size, resolve_adapters(adapters))


def write_module(fun: Function, out_dir: Path, *, typed_buffers: bool = True, adapters: Sequence[str] = ()) -> CModule:
  """Write ``fun``'s header / ``.c`` into ``out_dir`` and return the module."""
  fun = fun.concrete
  module = render_c_module(fun, typed_buffers=typed_buffers, adapters=adapters)
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / module.header_name).write_text(module.header)
  (out_dir / module.source_name).write_text(module.source)
  return module


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(prog="scaly_codegen", description="Render a Function to a header/source pair: a C kernel and a C or C++ header.")
  parser.add_argument("target", help="module:attribute naming a Function with every shape declared, or a zero-argument factory returning one")
  parser.add_argument("-o", "--out-dir", type=Path, default=Path(), help="directory to write into (default: cwd)")
  parser.add_argument(
    "--adapter",
    action="append",
    default=[],
    metavar="NAME",
    help=(
      "apply an output adapter; repeat for several. cpp: a C++ header with Buffer types in a namespace (f.hpp) instead of the C one; "
      "casadi: the CasADi 3.8 compatible symbols (f_n_in, f_sparsity_out, f_work, ...), sparse outputs in compressed-column order. "
      "Installed: " + ", ".join(available_adapters())
    ),
  )
  parser.add_argument("--no-typed-buffers", action="store_true", help="C header only: omit the typed buffer structs and the f_call wrapper")
  args = parser.parse_args(argv)
  module_name, _, attr = args.target.partition(":")
  if not attr:
    parser.error(f"target {args.target!r} is not module:attribute")
  fun = getattr(importlib.import_module(module_name), attr)
  if not isinstance(fun, Function):
    fun = fun()
  if not fun.is_concrete:
    parser.error(
      f"{args.target} has shape holes; export a concrete instance instead, such as `{attr}_3 = {attr}.instantiate(...)`, "
      f"or a zero-argument factory returning one ({', '.join(fun.instances) or 'no instances are built at import'})"
    )
  module = write_module(fun, args.out_dir, typed_buffers=not args.no_typed_buffers, adapters=args.adapter)
  print(args.out_dir / module.header_name)
  print(args.out_dir / module.source_name)
  print(f"sz_w: {module.workspace_size}")
  if module.externs:
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
  if fun.extern is not None:
    # An extern body calls Functions its own output graph does not reach (it holds only EXTERN_CALL
    # nodes), so surface them explicitly here.
    for callee in fun.extern.dependencies():
      if id(callee) not in seen:
        seen.add(id(callee))
        ret.append(callee)
    return ret
  for node in topo(fun.outputs):
    for callee in callees_of(node):
      if id(callee) not in seen:
        seen.add(id(callee))
        ret.append(callee)
  return ret


if __name__ == "__main__":
  main()
