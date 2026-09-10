"""Render a lowered Program IR ``PROGRAM`` to standalone scalar C — rendering only, no lowering
decisions of its own.

This is the renderer half of the Program-IR architecture (``docs/how_it_works/architecture.md``):
``passes.lowering.lower_function`` produces the Program IR; this module turns it into a
translation unit exposing the universal ABI (``<symbol>(arg,res,iw,w,mem)`` plus
the ``<symbol>_sz_*`` / ``*_mem`` helpers) so ``codegen.jit.CompiledFunction`` dispatches
it unchanged. It is the sole CPU renderer for host functions with no solver in
their call graph; ``codegen/aot.py`` orchestrates the solver-bearing case, reusing
``_render_raw_callee`` / ``_render_entry`` here and splicing in the solver wrappers.

There is **no silent fallback**: a function outside the lowered subset raises
``LoweringError`` loudly. Workspace lifetime/spill packing is a Program-IR pass
(``passes/program/pack_workspace.py``); the renderer just honors the ``sz_w`` / ``workspace_offset`` it sets.
Scalar/statement emission uses compact per-op maps — the lowerer (``passes/lowering.py``)
holds the extensible ``ExprOp``-keyed registry.
"""

from __future__ import annotations

import math

from .abi import abi_status_defines, c_api_signature, c_ident
from ..function import Function
from ..passes.lowering import LoweringError, lower_function, main_proc
from ..passes.program import ProgramObserver
from ..ir.program import ProgramNode, ProgramOp

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


def can_render_program_c(fun: Function) -> bool:
  """True iff ``fun`` lowers and renders through the Program IR path. Diagnostic helper
  (e.g. for coverage probes); the hot path is ``aot._render_source`` calling ``render_program_c``."""
  if fun.device.kind != "host":
    return False
  try:
    render_program_c_source(fun)
  except LoweringError:
    return False
  return True


def _includes(extra: tuple[str, ...] = ()) -> list[str]:
  # Vector types for coalesced stores (``_emit_body``): ``aligned(8)`` because the ABI only
  # promises double alignment, ``may_alias`` because they access plain double storage.
  return ["#include <math.h>", "#include <stddef.h>", "#include <stdint.h>", *extra, _VECTOR_TYPEDEF]


# Width 4 measured slower than scalar stores under GCC on the chain M=5 Hessian.
_VECTOR_TYPEDEF = "typedef double double2 __attribute__((vector_size(16), aligned(8), may_alias));"


def render_program_c_source(fun: Function, observe: ProgramObserver | None = None) -> str:
  """Lower a non-solver host ``fun`` and render it. ``codegen/aot.py`` lowers once for the whole
  module and calls ``render_program_c`` directly; this is the standalone convenience."""
  return render_program_c(lower_function(fun, observe=observe), fun)


def render_program_c(prog: ProgramNode, fun: Function) -> str:
  """Render ``fun``'s lowered PROGRAM to a standalone universal-ABI translation unit."""
  proc = main_proc(prog)
  pc = int(prog.attrs.get("proc_count", 1))
  callees = list(prog.args[: pc - 1])
  lines: list[str] = [
    *_includes(),
    "",
    *abi_status_defines(),
    "",
    "#ifdef __cplusplus",
    'extern "C" {',
    "#endif",
    "",
  ]
  for callee in callees:
    lines += _render_raw_callee(callee)
    lines.append("")
  lines += _render_entry(proc, fun)
  lines += ["", "#ifdef __cplusplus", "}", "#endif"]
  return "\n".join(lines).rstrip() + "\n"


def _render_entry(proc: ProgramNode, fun: Function) -> list[str]:
  """Emit the universal-ABI entry (``<symbol>_sz_*`` helpers + ``<symbol>(arg,res,iw,w,mem)``) with
  ``fun``'s main PROC body inlined. ``codegen/aot.py`` reuses this for solver-bearing functions, so
  the top function's body lowers through Program IR exactly like any other host function."""
  symbol = c_ident(fun.name)
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
  for i in range(len(fun.inputs)):
    lines.append(f"  if (!arg[{i}]) return ALLOY_ERR_NULL_INPUT;")
  for i in range(len(fun.outputs)):
    lines.append(f"  if (!res[{i}]) return ALLOY_ERR_NULL_RESULT;")
  _emit_local_buffers(body, lines, ptr_expr, indent=2)
  _emit_body(body, ptr_expr, lines, indent=2)
  lines += ["  return ALLOY_SUCCESS;", "}"]
  return lines


def _force_noinline_raw(proc_name: str) -> bool:
  # Apple clang 17 (Xcode 16.4 / macOS 15 arm64 CI) miscompiles inlined forward-AD helper callees
  # for CALL-node Jacobians. Keep normal user callees inline, but make generated forward helpers
  # real call frames until the compiler issue disappears. See docs/macos_clang_call_miscompile.md.
  return "_fwd" in proc_name


def _render_raw_callee(proc: ProgramNode) -> list[str]:
  """A callee renders as ``static inline void <name>_raw(const <dtype>* p0, ..., double* w)`` — a
  pointer per param plus the workspace tail (spilled slots index into ``w``; ``call`` sites pass
  the caller's ``w`` advanced past its own spill window). The leading ``input_count`` params are
  inputs and are ``const``-qualified (read-only by construction), so a solver wrapper can pass
  its ``const double*`` arguments without discarding qualifiers. No ABI wrapper.

  Normal user callees stay inline. Forward-AD helper callees are selectively noinline on purpose;
  see ``_force_noinline_raw`` and docs/macos_clang_call_miscompile.md.
  """
  param_count = int(proc.attrs["param_count"])
  input_count = int(proc.attrs.get("input_count", 0))
  params = list(proc.args[:param_count])
  body = list(proc.args[param_count:])
  sz_w = int(proc.attrs.get("sz_w", 0))
  ptr_expr = {pp.attrs["name"]: c_ident(pp.attrs["name"]) for pp in params}
  param_decls = ", ".join(
    [
      *(f"{'const ' if i < input_count else ''}{pp.dtype.c_type}* {c_ident(pp.attrs['name'])}" for i, pp in enumerate(params)),
      "double* w",
    ]
  )
  proc_name = proc.attrs["name"]
  qualifier = "static __attribute__((noinline))" if _force_noinline_raw(proc_name) else "static inline"
  out = [f"{qualifier} void {c_ident(proc_name)}_raw({param_decls}) {{"]
  if not sz_w:
    out.append("  (void)w;")
  _emit_local_buffers(body, out, ptr_expr, indent=2)
  _emit_body(body, ptr_expr, out, indent=2)
  out.append("}")
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


def _emit_body(body: list[ProgramNode], ptr_expr: dict[str, str], lines: list[str], indent: int) -> None:
  """Render Program statements in order."""
  for stmt in body:
    if stmt.op == ProgramOp.BUFFER:
      continue
    _emit_statement(stmt, ptr_expr, lines, indent)


def _emit_statement(stmt: ProgramNode, ptr_expr: dict[str, str], lines: list[str], indent: int) -> None:
  pad = " " * indent
  if stmt.op == ProgramOp.FOR:
    rng = stmt.args[0]
    name = c_ident(rng.attrs["name"])
    start = _emit_scalar(rng.args[0], ptr_expr)
    stop = _emit_scalar(rng.args[1], ptr_expr)
    step = _emit_scalar(rng.args[2], ptr_expr)
    incr = f"++{name}" if step == "1" else f"{name} += {step}"
    lines.append(f"{pad}for (long long {name} = {start}; {name} < {stop}; {incr}) {{")
    _emit_body(list(stmt.args[1:]), ptr_expr, lines, indent + 2)
    lines.append(f"{pad}}}")
  elif stmt.op == ProgramOp.STORE:
    _emit_assignment(_emit_view(stmt.args[0], ptr_expr), [stmt.args[1]], ptr_expr, lines, indent)
  elif stmt.op == ProgramOp.STORE_PAIR:
    view = stmt.args[0]
    ptr = ptr_expr.get(view.attrs["buffer"], c_ident(view.attrs["buffer"]))
    index = _emit_scalar(view.args[0], ptr_expr) if view.args else "0"
    target = f"*(double2*)({ptr}{f' + {index}' if index != '0' else ''})"
    _emit_assignment(target, list(stmt.args[1:]), ptr_expr, lines, indent)
  elif stmt.op == ProgramOp.ASSIGN:
    declaration = f"{stmt.dtype.c_type} " if stmt.attrs.get("declare") else ""
    _emit_assignment(c_ident(stmt.attrs["target"]), [stmt.args[0]], ptr_expr, lines, indent, declaration)
  elif stmt.op == ProgramOp.CALL:
    if stmt.attrs.get("external"):
      raise LoweringError("external (mixed-device) CALL rendering is deferred to a later migration step")
    n_in, n_out = int(stmt.attrs["n_in"]), int(stmt.attrs["n_out"])
    ptrs = [_emit_call_arg(a, ptr_expr) for a in stmt.args[: n_in + n_out]]
    # Workspace tail: a callee needing its own w[] gets this proc's w advanced past its spill
    # window; one needing none gets NULL. Set by the workspace pass; absent => no spill anywhere.
    if stmt.attrs.get("callee_needs_w"):
      offset = int(stmt.attrs.get("w_self", 0))
      ptrs.append("w" if offset == 0 else f"w + {offset}")
    else:
      ptrs.append("NULL")
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
    idx = _emit_scalar(node.args[0], ptr_expr) if node.args else "0"
    return ptr if idx == "0" else f"({ptr} + {idx})"
  raise LoweringError(f"unsupported CALL arg op {node.op}")


def _emit_view(view: ProgramNode, ptr_expr: dict[str, str]) -> str:
  if view.op != ProgramOp.VIEW:
    raise LoweringError(f"expected a VIEW, got {view.op}")
  ptr = ptr_expr.get(view.attrs["buffer"], c_ident(view.attrs["buffer"]))
  if len(view.args) > 1:
    raise LoweringError("multi-index VIEW rendering is not implemented yet (lands with SLICE/MATMUL)")
  idx = _emit_scalar(view.args[0], ptr_expr) if view.args else "0"
  return f"{ptr}[{idx}]"


def _c_float(value: float) -> str:
  if math.isnan(value):
    return "((double)NAN)"
  if math.isinf(value):
    return "((double)(-INFINITY))" if value < 0 else "((double)INFINITY)"
  return f"{value:.17g}"


def _emit_scalar(n: ProgramNode, ptr_expr: dict[str, str]) -> str:
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
      s = c_ident(node.attrs["name"])
    elif op == ProgramOp.LOAD:
      s = _emit_view(node.args[0], ptr_expr)
    elif op == ProgramOp.NEG:
      s = f"(-{args[0]})"
    elif op in _BIN_SYM:
      s = f"({args[0]} {_BIN_SYM[op]} {args[1]})"
    elif op in _UNARY_C:
      s = f"{_UNARY_C[op]}({args[0]})"
    elif op in _BINARY_C:
      s = f"{_BINARY_C[op]}({args[0]}, {args[1]})"
    else:
      raise LoweringError(f"Program IR C renderer: scalar op {op} not yet handled")
    text[id(node)] = s
  return text[id(n)]


__all__ = [
  "can_render_program_c",
  "render_program_c",
  "render_program_c_source",
]
