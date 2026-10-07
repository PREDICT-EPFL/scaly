"""Render a lowered Program IR ``PROGRAM`` to standalone scalar C — rendering only, no lowering
decisions of its own.

This is the renderer half of the Program-IR architecture (``docs/how_it_works/architecture.md``):
``passes.lowering.lower_function`` produces the Program IR; this module turns it into a
translation unit exposing the pointer ABI (``<symbol>(arg,res,iw,w,mem)``) so
``codegen.jit.CompiledFunction`` dispatches it unchanged. It is the sole CPU renderer for host functions with no solver in
their call graph; ``codegen/aot.py`` orchestrates the solver-bearing case, reusing
``_render_raw_callee`` / ``_render_entry`` here and splicing in the solver wrappers.

There is **no silent fallback**: a function outside the lowered subset raises
``LoweringError`` loudly. Workspace lifetime/spill packing is a Program-IR pass
(``passes/program/pack_workspace.py``); the renderer just honors the ``sz_w`` / ``workspace_offset`` it sets.
Scalar/statement emission uses compact per-op maps — the lowerer (``passes/lowering/``)
holds the extensible ``ExprOp``-keyed registry.
"""

from __future__ import annotations

import math

from .abi import abi_status_defines, c_api_signature, c_ident
from .casadi import casadi_defines, casadi_gather, casadi_scratch, render_casadi_queries
from ..function.concrete import ConcreteFunction
from ..function.model import Function, as_concrete
from ..passes.lowering import LoweringError, lower_function, main_proc
from ..passes.program import ProgramObserver
from ..ir.program import walk_program, ProgramNode, ProgramOp
from ..passes.program._common import allocated_name, buffer_refs
from .toolchain import CDialect, VectorLibm

# Scalar ProgramOp -> C spelling. Operators render inline; libm ops render as calls.
_BIN_SYM = {ProgramOp.ADD: "+", ProgramOp.SUB: "-", ProgramOp.MUL: "*", ProgramOp.DIV: "/", ProgramOp.MOD: "%"}
_UNARY_C = {
  ProgramOp.SIN: "sin",
  ProgramOp.COS: "cos",
  ProgramOp.TAN: "tan",
  ProgramOp.ASIN: "asin",
  ProgramOp.ACOS: "acos",
  ProgramOp.ATAN: "atan",
  ProgramOp.SINH: "sinh",
  ProgramOp.COSH: "cosh",
  ProgramOp.TANH: "tanh",
  ProgramOp.ERF: "erf",
  ProgramOp.EXP: "exp",
  ProgramOp.LOG: "log",
  ProgramOp.SQRT: "sqrt",
  ProgramOp.ABS: "fabs",
  ProgramOp.FLOOR: "floor",
  ProgramOp.CEIL: "ceil",
}
_BINARY_C = {ProgramOp.POW: "pow", ProgramOp.ATAN2: "atan2", ProgramOp.MINIMUM: "fmin", ProgramOp.MAXIMUM: "fmax"}


def can_render_program_c(fun: Function | ConcreteFunction) -> bool:
  """True iff ``fun`` lowers and renders through the Program IR path. Diagnostic helper
  (e.g. for coverage probes); the hot path is ``aot._render_source`` calling ``render_program_c``."""
  fun = as_concrete(fun)
  try:
    render_program_c_source(fun)
  except LoweringError:
    return False
  return True


def _includes(
  extra: tuple[str, ...] = (), *, dialect: CDialect = "gnu", prog: ProgramNode | None = None, vector_libm: VectorLibm = "none"
) -> list[str]:
  # Vector types for coalesced stores (``_emit_body``): ``aligned(8)`` because the ABI only
  # promises double alignment, ``may_alias`` because they access plain double storage.
  return [
    "#include <math.h>",
    "#include <stddef.h>",
    "#include <stdint.h>",
    *extra,
    *([_VECTOR_TYPEDEF] if dialect == "gnu" else []),
    *_lane_defines(prog),
  ]


# Width 4 measured slower than scalar stores under GCC on the chain M=5 Hessian.
_VECTOR_TYPEDEF = "typedef double double2 __attribute__((vector_size(16), aligned(8), may_alias));"


def render_program_c_source(fun: Function | ConcreteFunction, observe: ProgramObserver | None = None) -> str:
  """Lower a non-solver host ``fun`` and render it. ``codegen/aot.py`` lowers once for the whole
  module and calls ``render_program_c`` directly; this is the standalone convenience."""
  fun = as_concrete(fun)
  return render_program_c(lower_function(fun, observe=observe), fun)


def render_program_c(
  prog: ProgramNode, fun: ConcreteFunction, *, casadi: bool = False, dialect: CDialect = "gnu", vector_libm: VectorLibm = "none"
) -> str:
  """Render ``fun``'s lowered PROGRAM to a standalone pointer-ABI translation unit. ``casadi`` adds
  the CasADi query functions and the compressed-column gather (``codegen/casadi.py``)."""
  proc = main_proc(prog)
  pc = int(prog.attrs.get("proc_count", 1))
  callees = list(prog.args[: pc - 1])
  lines: list[str] = [
    *_includes(dialect=dialect, prog=prog, vector_libm=vector_libm),
    "",
    *abi_status_defines(),
    "",
    *(casadi_defines() + [""] if casadi else []),
    "#ifdef __cplusplus",
    'extern "C" {',
    "#endif",
    "",
  ]
  reserved_names = _c_reserved_names(prog)
  for callee in callees:
    lines += _render_raw_callee(callee, dialect=dialect, vector_libm=vector_libm, reserved_names=reserved_names)
    lines.append("")
  lines += _render_entry(proc, fun, casadi=casadi, dialect=dialect, vector_libm=vector_libm)
  if casadi:
    lines += ["", *render_casadi_queries(fun, entry_workspace(fun, int(proc.attrs.get("sz_w", 0)), casadi=True))]
  lines += ["", "#ifdef __cplusplus", "}", "#endif"]
  return "\n".join(lines).rstrip() + "\n"


def entry_workspace(fun: ConcreteFunction, sz_w: int, *, casadi: bool) -> int:
  """The ``SZ_W`` an entry needs: the packed spill size plus, under ``casadi``, the gather scratch."""
  return sz_w + (casadi_scratch(fun) if casadi else 0)


def entry_prologue(fun: ConcreteFunction, sz_w: int, *, options_count: int = 0) -> list[str]:
  """The opening of a pointer-ABI entry: the signature and the null checks behind the status codes.
  ``iw`` and ``mem`` are accepted and ignored; ``sz_w`` is the total workspace the entry reads."""
  symbol = c_ident(fun.name)
  lines = [
    c_api_signature(f"{symbol}_with_options" if options_count else symbol, solver_options=bool(options_count)) + " {",
    "  (void)iw;",
    "  (void)mem;",
    "  if (!arg || !res) return SCALY_ERR_NULL_ABI;",
    "  if (!w) return SCALY_ERR_NULL_WORK;" if sz_w else "  (void)w;",
  ]
  if options_count:
    lines += [
      "  if (!solver_options) return SCALY_ERR_NULL_INPUT;",
      *(f"  if (!solver_options[{i}]) return SCALY_ERR_NULL_INPUT;" for i in range(options_count)),
    ]
  lines += [f"  if (!arg[{i}]) return SCALY_ERR_NULL_INPUT;" for i in range(len(fun.inputs))]
  lines += [f"  if (!res[{i}]) return SCALY_ERR_NULL_RESULT;" for i in range(len(fun.outputs))]
  return lines


def _render_entry(
  proc: ProgramNode,
  fun: ConcreteFunction,
  *,
  casadi: bool = False,
  dialect: CDialect = "gnu",
  vector_libm: VectorLibm = "none",
  solver_callees: frozenset[str] = frozenset(),
  options_count: int = 0,
) -> list[str]:
  """Emit the pointer-ABI entry ``<symbol>(arg,res,iw,w,mem)`` with ``fun``'s main PROC body
  inlined. ``codegen/aot.py`` reuses this for solver-bearing functions, so the top function's body
  lowers through Program IR exactly like any other host function."""
  param_count = int(proc.attrs["param_count"])
  body = list(proc.args[param_count:])
  sz_w = int(proc.attrs.get("sz_w", 0))  # set by passes/program/pack_workspace.py

  # Buffer name -> C pointer expression for the ABI entry (inputs are arg[i], outputs res[i]).
  ptr_expr: dict[str, str] = {}
  for i, name in enumerate(fun.input_names):
    ptr_expr[name] = f"arg[{i}]"
  for i, name in enumerate(fun.output_names):
    ptr_expr[name] = f"res[{i}]"

  lines = [
    *_render_vector_helpers(proc, dialect=dialect, vector_libm=vector_libm),
    *entry_prologue(fun, entry_workspace(fun, sz_w, casadi=casadi), options_count=options_count),
  ]
  epilogue: list[str] = []
  if casadi:
    gather = casadi_gather(fun, sz_w)
    ptr_expr.update(gather.ptr)
    lines += gather.setup
    epilogue = gather.epilogue
  _emit_local_buffers(body, lines, ptr_expr, indent=2)
  _emit_body(body, ptr_expr, lines, indent=2, dialect=dialect, solver_callees=solver_callees)
  lines += [*epilogue, "  return SCALY_SUCCESS;", "}"]
  return lines


def _force_noinline_raw(proc_name: str) -> bool:
  # Apple clang 17 (Xcode 16.4 / macOS 15 arm64 CI) miscompiles inlined forward-AD helper callees
  # for CALL-node Jacobians. Keep normal user callees inline, but make generated forward helpers
  # real call frames until the compiler issue disappears. See internal/notes/macos_clang_call_miscompile.md.
  return "_fwd" in proc_name


def _c_reserved_names(prog: ProgramNode) -> set[str]:
  names = {
    c_ident(n.attrs[key]) for n in walk_program(prog) for key in ("name", "target", "vector_helper", "vector_prefix", "lane_width") if key in n.attrs
  }
  names.update(f"{c_ident(n.attrs['name'])}_raw" for n in walk_program(prog) if n.op == ProgramOp.PROC)
  return names


def _render_raw_callee(
  proc: ProgramNode,
  *,
  dialect: CDialect = "gnu",
  vector_libm: VectorLibm = "none",
  reserved_names: set[str] | None = None,
  solver_callees: frozenset[str] = frozenset(),
) -> list[str]:
  """A callee renders as ``static inline void <name>_raw(const <dtype>* p0, ..., double* w)`` — a
  pointer per param plus the workspace tail (spilled slots index into ``w``; ``call`` sites pass
  the caller's ``w`` advanced past its own spill window). The leading ``input_count`` params are
  inputs and are ``const``-qualified (read-only by construction), so a solver wrapper can pass
  its ``const double*`` arguments without discarding qualifiers. No ABI wrapper.

  Normal user callees stay inline. Forward-AD helper callees are selectively noinline on purpose;
  see ``_force_noinline_raw`` and internal/notes/macos_clang_call_miscompile.md.
  """
  param_count = int(proc.attrs["param_count"])
  input_count = int(proc.attrs.get("input_count", 0))
  params = list(proc.args[:param_count])
  body = list(proc.args[param_count:])
  sz_w = int(proc.attrs.get("sz_w", 0))
  options_name = allocated_name("solver_options", _c_reserved_names(proc))
  ptr_expr = {pp.attrs["name"]: c_ident(pp.attrs["name"]) for pp in params}
  param_decls = ", ".join(
    [
      *(f"{'const ' if i < input_count else ''}{pp.dtype.c_type}* {c_ident(pp.attrs['name'])}" for i, pp in enumerate(params)),
      "double* w",
      *((f"const scaly_solver_option* const* {options_name}",) if solver_callees else ()),
    ]
  )
  proc_name = proc.attrs["name"]
  noinline = _force_noinline_raw(proc_name)
  qualifier = ("static __attribute__((noinline))" if dialect == "gnu" else "static") if noinline else "static inline"
  raw_name = f"{c_ident(proc_name)}_raw"
  implementation = (
    allocated_name(f"{raw_name}_impl", reserved_names if reserved_names is not None else _c_reserved_names(proc))
    if noinline and dialect == "c"
    else raw_name
  )
  out = [*_render_vector_helpers(proc, dialect=dialect, vector_libm=vector_libm), f"{qualifier} void {implementation}({param_decls}) {{"]
  if not sz_w:
    out.append("  (void)w;")
  _emit_local_buffers(body, out, ptr_expr, indent=2)
  _emit_body(body, ptr_expr, out, indent=2, dialect=dialect, solver_callees=solver_callees, options_name=options_name)
  out.append("}")
  if implementation != raw_name:
    out.append(f"static void (*volatile {raw_name})({param_decls}) = {implementation};")
  return out


def _emit_local_buffers(body: list[ProgramNode], lines: list[str], ptr_expr: dict[str, str], indent: int) -> None:
  pad = " " * indent
  seen: set[str] = set()
  for stmt in body:
    if stmt.op != ProgramOp.BUFFER or stmt.attrs["name"] in seen:
      continue
    seen.add(stmt.attrs["name"])
    name = c_ident(stmt.attrs["name"])
    size = 1
    for d in stmt.attrs["shape"]:
      size *= int(d)
    size = size or 1
    if stmt.attrs.get("address_space") == "constant" and "values" in stmt.attrs:
      fmt = (lambda v: str(int(v))) if stmt.dtype.is_integer else _c_float
      values = ", ".join(fmt(v) for v in stmt.attrs["values"])
      lines.append(f"{pad}static const {stmt.dtype.c_type} {name}[{size}] = {{{values}}};")
    elif "alias_of" in stmt.attrs:
      # Zero-copy alias: a pointer into another buffer (contiguous slice / reshape).
      src = ptr_expr.get(stmt.attrs["alias_of"], c_ident(stmt.attrs["alias_of"]))
      offset = int(stmt.attrs["alias_offset"])
      rhs = src if offset == 0 else f"{src} + {offset}"
      lines.append(f"{pad}const {stmt.dtype.c_type}* {name} = {rhs};")
    elif "workspace_offset" in stmt.attrs:
      # A spilled slot: a window into the caller-provided w[] instead of a stack array.
      lines.append(f"{pad}{stmt.dtype.c_type}* {name} = w + {stmt.attrs['workspace_offset']};")
    else:
      lines.append(f"{pad}{stmt.dtype.c_type} {name}[{size}];")


def _emit_body(
  body: list[ProgramNode],
  ptr_expr: dict[str, str],
  lines: list[str],
  indent: int,
  *,
  dialect: CDialect = "gnu",
  solver_callees: frozenset[str] = frozenset(),
  options_name: str = "solver_options",
) -> None:
  """Render Program statements in order."""
  for stmt in body:
    if stmt.op == ProgramOp.BUFFER:
      continue
    _emit_statement(stmt, ptr_expr, lines, indent, dialect=dialect, solver_callees=solver_callees, options_name=options_name)


def _emit_statement(
  stmt: ProgramNode,
  ptr_expr: dict[str, str],
  lines: list[str],
  indent: int,
  *,
  dialect: CDialect = "gnu",
  solver_callees: frozenset[str] = frozenset(),
  options_name: str = "solver_options",
) -> None:
  pad = " " * indent
  if stmt.op == ProgramOp.FOR and "vector_helper" in stmt.attrs:
    _emit_vector_calls(stmt, ptr_expr, lines, indent)
  elif stmt.op == ProgramOp.FOR:
    rng = stmt.args[0]
    name = c_ident(rng.attrs["name"])
    start = _emit_scalar(rng.args[0], ptr_expr)
    stop = _emit_scalar(rng.args[1], ptr_expr)
    step = _emit_scalar(rng.args[2], ptr_expr)
    incr = f"++{name}" if step == "1" else f"{name} += {step}"
    lines.append(f"{pad}for (long long {name} = {start}; {name} < {stop}; {incr}) {{")
    _emit_body(list(stmt.args[1:]), ptr_expr, lines, indent + 2, dialect=dialect, solver_callees=solver_callees, options_name=options_name)
    lines.append(f"{pad}}}")
  elif stmt.op == ProgramOp.STORE:
    _emit_assignment(_emit_view(stmt.args[0], ptr_expr), [stmt.args[1]], ptr_expr, lines, indent)
  elif stmt.op == ProgramOp.STORE_PAIR:
    view = stmt.args[0]
    ptr = ptr_expr.get(view.attrs["buffer"], c_ident(view.attrs["buffer"]))
    index = _emit_scalar(view.args[0], ptr_expr)
    if dialect == "c":
      for offset, value in enumerate(stmt.args[1:]):
        _emit_assignment(f"{ptr}[({index}) + {offset}]", [value], ptr_expr, lines, indent)
    else:
      target = f"*(double2*)({ptr}{f' + {index}' if index != '0' else ''})"
      _emit_assignment(target, list(stmt.args[1:]), ptr_expr, lines, indent)
  elif stmt.op == ProgramOp.ASSIGN:
    declaration = f"{stmt.dtype.c_type} " if stmt.attrs.get("declare") else ""
    _emit_assignment(c_ident(stmt.attrs["target"]), [stmt.args[0]], ptr_expr, lines, indent, declaration)
  elif stmt.op == ProgramOp.CALL:
    n_in, n_out = int(stmt.attrs["n_in"]), int(stmt.attrs["n_out"])
    ptrs = [_emit_call_arg(a, ptr_expr) for a in stmt.args[: n_in + n_out]]
    # Workspace tail: a callee needing its own w[] gets this proc's w advanced past its spill
    # window; one needing none gets NULL. Set by the workspace pass; absent => no spill anywhere.
    if stmt.attrs.get("callee_needs_w"):
      offset = int(stmt.attrs.get("w_self", 0))
      ptrs.append("w" if offset == 0 else f"w + {offset}")
    else:
      ptrs.append("NULL")
    if stmt.attrs["callee"] in solver_callees:
      ptrs.append(options_name)
    lines.append(f"{pad}{c_ident(stmt.attrs['callee'])}_raw({', '.join(ptrs)});")
  else:
    raise LoweringError(f"Program IR C renderer: statement op {stmt.op} not yet handled")


def _emit_assignment(target: str, values: list[ProgramNode], ptr_expr: dict[str, str], lines: list[str], indent: int, declaration: str = "") -> None:
  """Render one scalar or paired assignment."""
  pad = " " * indent
  parts = [_emit_scalar(value, ptr_expr) for value in values]
  rhs = parts[0] if len(parts) == 1 else f"(double2){{{', '.join(parts)}}}"
  lines.append(f"{pad}{declaration}{target} = {rhs};")


def _emit_call_arg(node: ProgramNode, ptr_expr: dict[str, str]) -> str:
  """A CALL argument is a whole BUFFER (its pointer) or a VIEW (pointer + offset)."""
  if node.op == ProgramOp.BUFFER:
    return ptr_expr.get(node.attrs["name"], c_ident(node.attrs["name"]))
  if node.op == ProgramOp.VIEW:
    ptr = ptr_expr.get(node.attrs["buffer"], c_ident(node.attrs["buffer"]))
    idx = _emit_scalar(node.args[0], ptr_expr)
    return ptr if idx == "0" else f"({ptr} + {idx})"
  raise LoweringError(f"unsupported CALL arg op {node.op}")


def _emit_view(view: ProgramNode, ptr_expr: dict[str, str], var_expr: dict[str, str] | None = None) -> str:
  if view.op != ProgramOp.VIEW:
    raise LoweringError(f"expected a VIEW, got {view.op}")
  ptr = ptr_expr.get(view.attrs["buffer"], c_ident(view.attrs["buffer"]))
  idx = _emit_scalar(view.args[0], ptr_expr, var_expr)
  return f"{ptr}[{idx}]"


def _c_float(value: float) -> str:
  if math.isnan(value):
    return "((double)NAN)"
  if math.isinf(value):
    return "((double)(-INFINITY))" if value < 0 else "((double)INFINITY)"
  return "-0.0" if value == 0 and math.copysign(1, value) < 0 else f"{value:.17g}"


def _emit_scalar(n: ProgramNode, ptr_expr: dict[str, str], var_expr: dict[str, str] | None = None) -> str:
  """Render a prepared scalar tree bottom up."""
  text: dict[int, str] = {}
  stack = [(n, False)]
  while stack:
    node, ready = stack.pop()
    if id(node) in text:
      continue
    op = node.op
    if not ready and op not in {ProgramOp.LOAD, ProgramOp.CONST_INT, ProgramOp.CONST_FLOAT, ProgramOp.VAR}:
      stack.append((node, True))
      stack.extend((a, False) for a in reversed(node.args) if id(a) not in text)
      continue
    args = [text[id(a)] for a in node.args] if ready else []
    if op == ProgramOp.CONST_INT:
      s = str(node.attrs["value"])
    elif op == ProgramOp.CONST_FLOAT:
      value = node.attrs["value"]
      s = _c_float(value)
      if math.isfinite(value) and "." not in s and "e" not in s:
        s += ".0"
    elif op == ProgramOp.VAR:
      s = (var_expr or {}).get(node.attrs["name"], c_ident(node.attrs["name"]))
    elif op == ProgramOp.LOAD:
      s = (var_expr or {}).get(f"@load{id(node)}") or _emit_view(node.args[0], ptr_expr, var_expr)
    elif op == ProgramOp.NEG:
      s = f"(-{args[0]})"
    elif op in _BIN_SYM:
      s = f"({args[0]} {_BIN_SYM[op]} {args[1]})"
    elif op in _UNARY_C:
      s = f"{_UNARY_C[op]}({args[0]})"
    elif op in (ProgramOp.MINIMUM, ProgramOp.MAXIMUM) and node.dtype.is_integer:
      comparison = "<" if op == ProgramOp.MINIMUM else ">"
      s = f"({args[0]} {comparison} {args[1]} ? {args[0]} : {args[1]})"
    elif op in _BINARY_C:
      s = f"{_BINARY_C[op]}({args[0]}, {args[1]})"
    else:
      raise LoweringError(f"Program IR C renderer: scalar op {op} not yet handled")
    text[id(node)] = s
  return text[id(n)]


def _lane_defines(prog: ProgramNode | None) -> list[str]:
  if prog is None:
    return []
  ranges = [n for n in walk_program(prog) if n.op == ProgramOp.RANGE and "lanes" in n.attrs]
  if not ranges:
    return []
  lanes = ranges[0].attrs["lanes"]
  lines = (
    [f"#define SCALY_LANES {lanes}"]
    if lanes != "auto"
    else [
      "#ifndef SCALY_LANES",
      "#if defined(__AVX512F__)",
      "#define SCALY_LANES 8",
      "#elif defined(__AVX__) || (defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS >= 256)",
      "#define SCALY_LANES 4",
      "#elif defined(__SSE2__) || defined(__aarch64__)",
      "#define SCALY_LANES 2",
      "#else",
      "#define SCALY_LANES 1",
      "#endif",
      "#endif",
      "#if SCALY_LANES != 1 && SCALY_LANES != 2 && SCALY_LANES != 4 && SCALY_LANES != 8",
      '#error "SCALY_LANES must be 1, 2, 4 or 8"',
      "#endif",
    ]
  )
  lines += [
    "#if defined(__AVX512F__)",
    "#define SCALY_REGISTER_SLOTS 256",
    "#elif defined(__AVX__) || defined(__aarch64__)",
    "#define SCALY_REGISTER_SLOTS 64",
    "#elif defined(__SSE2__)",
    "#define SCALY_REGISTER_SLOTS 32",
    "#else",
    "#define SCALY_REGISTER_SLOTS 16",
    "#endif",
  ]
  for rng in ranges:
    caps = rng.attrs["lane_caps"]
    cap = f"(SCALY_REGISTER_SLOTS == 256 ? {caps[3]} : SCALY_REGISTER_SLOTS == 64 ? {caps[2]} : SCALY_REGISTER_SLOTS == 32 ? {caps[1]} : {caps[0]})"
    lines.append(f"#define {rng.attrs['lane_width']} (SCALY_LANES < {cap} ? SCALY_LANES : {cap})")
  return lines


def _vector_captures(stmt: ProgramNode) -> tuple[list[ProgramNode], list[ProgramNode]]:
  nodes = list(walk_program(stmt))
  declared = {n.attrs["name"] for n in nodes if n.op == ProgramOp.BUFFER}
  buffers = {n.attrs["buffer"]: n for n in nodes if n.op == ProgramOp.VIEW and n.attrs["buffer"] not in declared}
  local = {n.attrs["target"] for n in nodes if n.op == ProgramOp.ASSIGN}
  local.update(n.attrs["name"] for n in nodes if n.op == ProgramOp.RANGE)
  local.add("SCALY_LANES")
  local.update(n.attrs["lane_width"] for n in nodes if n.op == ProgramOp.RANGE and "lane_width" in n.attrs)
  variables = {n.attrs["name"]: n for n in nodes if n.op == ProgramOp.VAR and n.attrs["name"] not in local}
  return [buffers[name] for name in sorted(buffers)], [variables[name] for name in sorted(variables)]


def _vector_width(stmt: ProgramNode) -> str:
  return stmt.args[1].args[0].attrs["lane_width"]


def _emit_vector_calls(stmt: ProgramNode, ptr_expr: dict[str, str], lines: list[str], indent: int) -> None:
  pad = " " * indent
  helper = stmt.attrs["vector_helper"]
  outer = c_ident(stmt.args[0].attrs["name"])
  count, width = stmt.attrs["vector_count"], _vector_width(stmt)
  buffers, variables = _vector_captures(stmt)
  captures = [ptr_expr.get(n.attrs["buffer"], c_ident(n.attrs["buffer"])) for n in buffers]
  captures += [c_ident(n.attrs["name"]) for n in variables]
  suffix = ", " + ", ".join(captures) if captures else ""
  lines += [
    f"{pad}for (long long {outer} = 0; {outer} < {count} / {width}; ++{outer}) {{",
    f"{pad}  {helper}({outer}, {width}{suffix});",
    f"{pad}}}",
    f"#if ({count} % {width}) != 0",
    f"{pad}{helper}({count} / {width}, {count} % {width}{suffix});",
    "#endif",
  ]


def _render_vector_helpers(proc: ProgramNode, *, dialect: CDialect, vector_libm: VectorLibm) -> list[str]:
  lines: list[str] = []
  for stmt in reversed(list(walk_program(proc))):
    if stmt.op != ProgramOp.FOR or "vector_helper" not in stmt.attrs:
      continue
    helper = stmt.attrs["vector_helper"]
    vector = stmt.args[1]
    lane = c_ident(vector.args[0].attrs["name"])
    outer = c_ident(stmt.args[0].attrs["name"])
    prefix = stmt.attrs["vector_prefix"]
    valid = f"{prefix}_valid"
    width, vec = _vector_width(stmt), f"{prefix}_vec"
    buffers, variables = _vector_captures(stmt)
    writes = buffer_refs(stmt).writes
    params = [f"long long {outer}", f"long long {valid}"]
    params += [f"{'const ' if n.attrs['buffer'] not in writes else ''}{n.dtype.c_type}* {c_ident(n.attrs['buffer'])}" for n in buffers]
    params += [f"{n.dtype.c_type} {c_ident(n.attrs['name'])}" for n in variables]
    operations = {n.op for n in walk_program(vector) if n.op in _UNARY_C or n.op in _BINARY_C and n.dtype.is_floating}
    if dialect == "gnu":
      # Values and parameters use the natural type: aarch64 GCC 13.1 and 14 to 16.1 crash on a by-value
      # parameter whose vector type carries both ``aligned`` and ``may_alias``. ``_mem`` is the
      # under-aligned aliasing view for loads and stores through double buffers.
      lines.append(f"typedef double {vec} __attribute__((vector_size(8 * {width})));")
      lines.append(f"typedef {vec} {vec}_mem __attribute__((aligned(8), may_alias));")
      for op in sorted(operations):
        lines += _vector_math(helper, vec, width, op, vector_libm)
    qualifier = "static inline __attribute__((always_inline))" if dialect == "gnu" else "static inline"
    lines.append(f"{qualifier} void {helper}({', '.join(params)}) {{")
    local_buffers = [n for n in reversed(list(walk_program(vector))) if n.op == ProgramOp.BUFFER]
    _emit_local_buffers(local_buffers, lines, {}, 2)
    if dialect == "c":
      lines.append(f"  for (long long {lane} = 0; {lane} < {valid}; ++{lane}) {{")
      _emit_body(list(vector.args[1:]), {}, lines, 4, dialect="c")
      lines.append("  }")
    else:
      _emit_vector_body(list(vector.args[1:]), vec, helper, vector.args[0].attrs["name"], width, lines, prefix, valid)
    lines.append("}")
  return lines


def _vector_math(helper: str, vec: str, width: str, op: ProgramOp, vector_libm: VectorLibm) -> list[str]:
  function = _UNARY_C.get(op, _BINARY_C.get(op, ""))
  binary = op in _BINARY_C
  params = f"{vec} x, {vec} y" if binary else f"{vec} x"
  args = "x[i], y[i]" if binary else "x[i]"
  lines = [f"static inline __attribute__((always_inline)) {vec} {helper}_{function}({params}) {{"]
  glibc = vector_libm == "glibc" and op in {ProgramOp.SIN, ProgramOp.COS, ProgramOp.TAN, ProgramOp.EXP, ProgramOp.LOG, ProgramOp.POW, ProgramOp.TANH}
  if glibc:
    lines += [
      f"#if {width} > 1",
      "#if !defined(__x86_64__) || !defined(__GLIBC__)",
      '#error "vector_libm=glibc requires x86-64 glibc and -lmvec"',
      "#endif",
    ]
    if op == ProgramOp.TANH:
      lines += ["#if !__GLIBC_PREREQ(2, 35)", '#error "vector_libm=glibc tanh requires glibc >= 2.35 and -lmvec"', "#endif"]
    for index, (w, abi, feature) in enumerate(((8, "e", "__AVX512F__"), (4, "d", "__AVX__"), (2, "b", "__SSE2__"))):
      symbol = f"_ZGV{abi}N{w}{'vv' if binary else 'v'}_{function}"
      lines += [
        f"#{'if' if index == 0 else 'elif'} {width} == {w}",
        f"#ifndef {feature}",
        f'#error "vector_libm=glibc width {w} requires {feature} and -lmvec"',
        "#endif",
        f"  extern {vec} {symbol}({vec}{', ' + vec if binary else ''});",
        f"  return {symbol}(x{', y' if binary else ''});",
      ]
    lines += ["#endif", "#else"]
  lines += [f"  {vec} result;", f"  for (int i = 0; i < {width}; ++i) result[i] = {function}({args});", "  return result;"]
  if glibc:
    lines.append("#endif")
  lines.append("}")
  return lines


def _emit_vector_body(body: list[ProgramNode], vec: str, helper: str, lane: str, width: str, lines: list[str], prefix: str, valid_count: str) -> None:
  lane_name, lane = lane, c_ident(lane)
  vector_vars: set[str] = set()
  lane_vars: dict[str, str] = {}
  serial = 0

  def expression(root: ProgramNode) -> str:
    nonlocal serial
    text: dict[ProgramNode, str] = {}
    stack = [(root, False)]
    while stack:
      node, ready = stack.pop()
      if node in text:
        continue
      if not ready and node.op not in (ProgramOp.LOAD, ProgramOp.VAR, ProgramOp.CONST_FLOAT, ProgramOp.CONST_INT):
        stack.append((node, True))
        stack.extend((a, False) for a in reversed(node.args))
        continue
      args = [text[a] for a in node.args] if ready else []
      if node.op == ProgramOp.LOAD:
        serial += 1
        name = f"{prefix}_load_{serial}"
        lines.append(f"  {vec} {name};")
        view = node.args[0]
        source = _emit_view(view, {}, lane_vars)
        if view.attrs.get("lane_stride") == 1:
          first = _emit_view(view, {}, {**{key: value.replace(f"[{lane}]", "[0]") for key, value in lane_vars.items()}, lane_name: "0"})
          lines.append(f"  if ({valid_count} == {width}) {name} = *(const {vec}_mem*)(&{first});")
          lines.append(f"  else for (long long {lane} = 0; {lane} < {width}; ++{lane}) {name}[{lane}] = {source};")
        else:
          lines.append(f"  double {name}_stage[8];")
          lines.append(f"  for (long long {lane} = 0; {lane} < {width}; ++{lane}) {name}_stage[{lane}] = {source};")
          lines.append(f"  {name} = *(const {vec}_mem*){name}_stage;")
        value = name
      elif node.op == ProgramOp.VAR and node.attrs["name"] in vector_vars:
        value = c_ident(node.attrs["name"])
      elif node.op in (ProgramOp.VAR, ProgramOp.CONST_FLOAT, ProgramOp.CONST_INT):
        serial += 1
        name = f"{prefix}_broadcast_{serial}"
        lines.append(f"  {vec} {name};")
        lines.append(f"  for (int {lane} = 0; {lane} < {width}; ++{lane}) {name}[{lane}] = {_emit_scalar(node, {})};")
        value = name
      elif node.op == ProgramOp.NEG:
        value = f"(-{args[0]})"
      elif node.op in _BIN_SYM:
        value = f"({args[0]} {_BIN_SYM[node.op]} {args[1]})"
      elif node.op in _UNARY_C or node.op in _BINARY_C:
        value = f"{helper}_{_UNARY_C.get(node.op, _BINARY_C.get(node.op))}({', '.join(args)})"
      else:
        raise LoweringError(f"unsupported vector scalar op {node.op}")
      text[node] = value
    return text[root]

  def emit(statements: list[ProgramNode]) -> None:
    nonlocal serial
    for stmt in statements:
      if stmt.op == ProgramOp.BUFFER:
        continue
      if stmt.op == ProgramOp.FOR:
        rng = stmt.args[0]
        name = c_ident(rng.attrs["name"])
        start, stop, step = (_emit_scalar(n, {}) for n in rng.args)
        lines.append(f"  for (long long {name} = {start}; {name} < {stop}; {name} += {step}) {{")
        first = len(lines)
        emit(list(stmt.args[1:]))
        for i in range(first, len(lines)):
          lines[i] = "  " + lines[i]
        lines.append("  }")
      elif stmt.op == ProgramOp.ASSIGN:
        name = c_ident(stmt.attrs["target"])
        if stmt.dtype.is_floating:
          value = expression(stmt.args[0])
          lines.append(f"  {vec + ' ' if stmt.attrs.get('declare') else ''}{name} = {value};")
          vector_vars.add(stmt.attrs["target"])
        else:
          lines.append(f"  {stmt.dtype.c_type} {name}[8];")
          value = _emit_scalar(stmt.args[0], {}, lane_vars)
          lines.append(f"  for (long long {lane} = 0; {lane} < {width}; ++{lane}) {name}[{lane}] = {value};")
          lane_vars[stmt.attrs["target"]] = f"{name}[{lane}]"
      elif stmt.op == ProgramOp.STORE and stmt.attrs.get("ordered_reduction"):
        target, value = stmt.args
        staged = dict(lane_vars)
        for node in walk_program(value):
          if node.op == ProgramOp.LOAD and node.args[0] is not target:
            loaded = expression(node)
            staged[f"@load{id(node)}"] = f"{loaded}[{lane}]"
        destination = _emit_view(target, {}, staged)
        rhs = _emit_scalar(value, {}, staged)
        lines.append(f"  for (long long {lane} = 0; {lane} < {valid_count}; ++{lane}) {destination} = {rhs};")
      elif stmt.op in (ProgramOp.STORE, ProgramOp.STORE_PAIR):
        for offset, root in enumerate(stmt.args[1:]):
          value = expression(root)
          serial += 1
          name = f"{prefix}_store_{serial}"
          lines.append(f"  {vec} {name} = {value};")
          view = stmt.args[0]
          pointer = c_ident(view.attrs["buffer"])
          index = _emit_scalar(view.args[0], {}, lane_vars)
          valid = width if view.attrs.get("lane_local") else valid_count
          if view.attrs.get("lane_stride") == 1:
            first = _emit_scalar(view.args[0], {}, {**{key: value.replace(f"[{lane}]", "[0]") for key, value in lane_vars.items()}, lane_name: "0"})
            lines.append(f"  if ({valid} == {width}) *({vec}_mem*)(&{pointer}[({first}) + {offset}]) = {name};")
            lines.append(f"  else for (long long {lane} = 0; {lane} < {valid}; ++{lane}) {pointer}[({index}) + {offset}] = {name}[{lane}];")
          else:
            lines.append(f"  for (long long {lane} = 0; {lane} < {valid}; ++{lane}) {pointer}[({index}) + {offset}] = {name}[{lane}];")
      else:
        raise LoweringError(f"unsupported vector statement {stmt.op}")

  emit(body)
