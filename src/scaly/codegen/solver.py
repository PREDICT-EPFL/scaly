"""Generate solver extern libraries and the options/statistics API around their calls."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING
from collections.abc import Callable, Mapping

from scaly.codegen.abi import c_ident
from scaly.utils.names import NameScope
from scaly.codegen.c import _c_reserved_names
from scaly.ir.expr import Expr, callees_of, topo
from scaly.function.concrete import ConcreteFunction
from scaly.function.extern import CLibrary, _Extern
from scaly.solvers.model import _SolverLibrary
from scaly.codegen.toolchain import BuildRecipe
from scaly.utils.env import env
from scaly.solvers.graph import external_oracles, is_solver_function, solver_callees, solver_descriptor

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
  function_symbols: Mapping[ConcreteFunction, str] = field(default_factory=dict)

  def raw_symbol_of(self, fun: ConcreteFunction | ExternalOracle) -> str:
    from scaly.solvers.model import ExternalOracle

    return fun.raw_symbol if isinstance(fun, ExternalOracle) else f"{self.function_symbols.get(fun, c_ident(fun.name))}_raw"


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
    for node in fn.nodes:
      for callee in callees_of(node):
        visit(callee)

  visit(fun)
  return tuple(found.values())


def solver_stats_symbols(fun: ConcreteFunction) -> tuple[str, ...]:
  """C identifiers for every solver wrapper reachable from ``fun``."""
  return tuple(c_ident(fn.name) for fn in solver_functions(fun))


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def render_solver_raw(fun: ConcreteFunction, *, function_symbols: Mapping[ConcreteFunction, str] | None = None) -> list[str]:
  """Frame a plugin-rendered wrapper body with the scaly-owned stats storage
  and the exported ``<symbol>_stats`` accessor. The body itself comes from the
  backend's ``render_wrapper`` hook."""
  from scaly.solvers.registry import get_backend

  desc = solver_descriptor(fun)
  external_sources: list[str] = []
  for oracle in external_oracles(fun):
    if oracle.source and oracle.source not in external_sources:
      external_sources.append(oracle.source)
  symbol = c_ident(fun.name)
  ctx = SolverWrapperCtx(
    symbol=symbol,
    raw_symbol=_raw_symbol(fun),
    stats_symbol=f"{symbol}_stats_data",
    function_symbols=function_symbols or {},
  )
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


def solver_extern(library: _SolverLibrary, recipe: BuildRecipe, prepare: Callable[[ConcreteFunction], ConcreteFunction]) -> _Extern:
  """Declare an external bridge with its wrapper and oracle C source."""
  from scaly.function.extern import extern
  from scaly.function.tree import arg
  from scaly.passes.lowering import lower_function
  from scaly.codegen.c import _includes, _render_raw_callee, extern_declarations
  from scaly.solvers.registry import get_backend
  from scaly.solvers.stats import stats_c_defs, stats_c_timing_defs

  fun = library.function
  oracles = solver_callees(fun)
  inputs, outputs, entries = [], [], {}
  for index, oracle in enumerate(oracles):
    actuals = tuple(arg(f"oracle_{index}_{i}", formal.type).symbols() for i, formal in enumerate(oracle.inputs))
    entry = ConcreteFunction._from_exprs(
      f"{oracle.name}_entry",
      actuals,
      oracle._call(actuals)[: len(oracle.outputs)],
      oracle.input_names,
      oracle.output_names,
      input_layouts=oracle.input_tree.layouts,
      output_layouts=oracle.output_tree.layouts,
    )
    entry = prepare(entry)
    entries[oracle] = entry
    inputs.extend(actuals)
    outputs.extend(entry._call(actuals))
  bundle = ConcreteFunction._from_exprs(
    NameScope().allocate(f"{c_ident(fun.name)}_oracles", generated=True),
    inputs,
    outputs,
    [f"input_{i}" for i in range(len(inputs))],
    [f"output_{i}" for i in range(len(outputs))],
  )
  prog = lower_function(
    bundle,
    reserved_names=(f"{c_ident(fun.name)}_raw", *(oracle.raw_symbol for oracle in external_oracles(fun))),
    lanes=recipe.lanes,
    reciprocal=recipe.reciprocal,
    frame_budget=recipe.frame_budget,
  )
  count = int(prog.attrs["proc_count"])
  procs = prog.args[: count - 1]
  symbols = dict(prog.attrs["function_symbols"])
  external_workspace = max((oracle.workspace_size for oracle in external_oracles(fun)), default=0)
  workspace = max((external_workspace, *(int(proc.attrs.get("sz_w", 0)) for proc in procs)))
  backend = get_backend(fun.descriptor.backend)
  symbol = c_ident(fun.name)
  lines = [
    *_includes(
      ("#include <time.h>", "#include <string.h>", f'#include "{backend.header}"'), prog=prog, dialect=recipe.dialect, vector_libm=recipe.vector_libm
    ),
    *stats_c_defs(),
    *stats_c_timing_defs(),
    *solver_options_c_defs(),
    *extern_declarations(prog),
  ]
  for proc in procs:
    lines.extend(_render_raw_callee(proc, reserved_names=_c_reserved_names(prog), dialect=recipe.dialect, vector_libm=recipe.vector_libm))
  lines.extend(render_solver_raw(fun, function_symbols={oracle: symbols[entry] for oracle, entry in entries.items()}))
  ins = [f"const double* in{i}" for i in range(len(fun.inputs))]
  outs = [f"double* out{i}" for i in range(len(fun.outputs))]
  args = [*(f"in{i}" for i in range(len(ins))), *(f"out{i}" for i in range(len(outs))), "w", f"{symbol}_options"]
  lines.extend(
    [
      f"static const scaly_solver_option* {symbol}_options;",
      f"void {symbol}_set_options(const scaly_solver_option* options) {{ {symbol}_options = options; }}",
      f"void {symbol}_solve({', '.join([*ins, *outs, 'double* w'])}) {{",
      f"  {symbol}_raw({', '.join(args)});",
      "}",
    ]
  )
  library = replace(
    library,
    recipe=recipe,
    sources={f"{symbol}_extern.c": "\n".join(lines) + "\n"},
    compile_flags=(*library.compile_flags, env("SCALY_CC_OPT") or "-O2", *recipe.cpu_flags, "-fno-math-errno"),
    link=(*library.link, *(flag[2:] for flag in recipe.link_flags)),
  )
  return extern(
    library,
    f"{symbol}_solve",
    inputs=tuple(arg(name, value.type) for name, value in zip(fun.input_names, fun.inputs)),
    outputs=fun.output_tree,
    workspace=workspace,
  )


def prepare_solver_calls(fun: ConcreteFunction, recipe: BuildRecipe) -> tuple[ConcreteFunction, tuple[CLibrary, ...]]:
  """Bind solver bridge declarations to the selected recipe before ordinary lowering."""
  from scaly.passes.lowering.ctx import _check_float64_leaves

  functions: dict[ConcreteFunction, ConcreteFunction] = {}
  externs: dict[_Extern, _Extern] = {}
  libraries: dict[CLibrary, None] = {}

  def bind(function: ConcreteFunction) -> ConcreteFunction:
    if function in functions:
      return functions[function]
    _check_float64_leaves(function, "a graph that calls a solver")
    rebuilt: dict[Expr, Expr] = {}
    for node in topo(function.results):
      attrs = dict(node.attrs)
      for name, callee in attrs.items():
        if isinstance(callee, ConcreteFunction):
          attrs[name] = bind(callee)
        elif isinstance(callee, _Extern):
          if callee not in externs:
            library = callee.library
            externs[callee] = solver_extern(library, recipe, bind) if isinstance(library, _SolverLibrary) and library.recipe != recipe else callee
            libraries[externs[callee].library] = None
          attrs[name] = externs[callee]
      if "branches" in attrs:
        branches = tuple(bind(branch) for branch in attrs["branches"])
        if any(a is not b for a, b in zip(branches, attrs["branches"])):
          attrs["branches"] = branches
      args = tuple(rebuilt[arg] for arg in node.args)
      rebuilt[node] = (
        node
        if all(value is node.attrs[name] for name, value in attrs.items()) and all(a is b for a, b in zip(args, node.args))
        else Expr(
          node.op,
          args,
          node.type,
          node.name,
          node.value,
          attrs,
          node.lowering,
        )
      )
    outputs = tuple(rebuilt[output] for output in function.outputs)
    rules = function.rules
    if rules is not None:
      residuals = tuple(rebuilt[output] for output in rules.residuals)
      if any(a is not b for a, b in zip(residuals, rules.residuals)):
        rules = replace(rules, residuals=residuals)
    functions[function] = (
      function
      if all(a is b for a, b in zip(outputs, function.outputs)) and rules is function.rules
      else function._replace(outputs=outputs, rules=rules)
    )
    return functions[function]

  bound = bind(fun)
  return bound, tuple(libraries)


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
    c_api_signature(fun, f"{symbol}_with_options", solver_options=True) + ";",
    f"#define {symbol}_N_SOLVERS {len(solvers)}",
    *(f"#define {symbol}_OPTIONS_{c_ident(fn.name)} {i}" for i, fn in enumerate(solvers)),
    *(f"const scaly_solver_option* {c_ident(fn.name)}_default_options(void);" for fn in solvers),
  ]
