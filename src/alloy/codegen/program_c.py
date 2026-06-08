"""Render a lowered Program IR ``PROGRAM`` to standalone scalar C.

This is the renderer half of the Program-IR migration (``docs/program_ir_migration.md``).
``lowering.lower_function`` produces the Program IR; this module turns it into a
translation unit exposing the universal ABI (``<symbol>(arg,res,iw,w,mem)`` plus
the ``<symbol>_sz_*`` / ``*_mem`` helpers) so ``jit.CompiledFunction`` dispatches
it unchanged.

Selection is opt-in during migration via ``ALLOY_USE_PROGRAM_IR_C=1``. There is
**no silent fallback**: if a function is outside the lowered subset, rendering
raises ``LoweringError`` loudly, unless ``ALLOY_PROGRAM_IR_FALLBACK=1`` is set as
an explicit escape hatch (used to run the whole suite under the flag and measure
remaining coverage). The end state drops the flag and the legacy renderer.

Internal temporaries are stack-local arrays (``sz_w`` is 0); lifetime/workspace
packing is a later step. Scalar/statement emission uses compact per-op maps —
the lowerer (``lowering.py``) is where the extensible ``Ops``-keyed registry lives.
"""

from __future__ import annotations

import math
import os
import re

from ..abi import c_api_signature
from ..function import Function
from ..lowering import LoweringError, lower_function, main_proc
from ..program import PNode, POps


def _c_ident(name: str) -> str:
  """Sanitize a Program-IR name into a valid C identifier.

  Buffer/var/callee names may contain ``:`` (e.g. derivative names like ``fwd:eq:z``)
  or other non-identifier characters. This must match ``alloy.codegen.c._c_ident`` so
  the rendered entry symbol agrees with what ``jit.CompiledFunction`` looks up.
  """
  ident = re.sub(r"\W", "_", name)
  return f"_{ident}" if ident[:1].isdigit() else ident


_ABI_DEFINES = (
  "#define ALLOY_SUCCESS 0",
  "#define ALLOY_ERR_NULL_ABI 1",
  "#define ALLOY_ERR_NULL_WORK 2",
  "#define ALLOY_ERR_NULL_RESULT 3",
  "#define ALLOY_ERR_NULL_INPUT 4",
)

# Scalar POps -> C spelling. Operators render inline; libm ops render as calls.
_BIN_SYM = {POps.ADD: "+", POps.SUB: "-", POps.MUL: "*", POps.DIV: "/", POps.MOD: "%"}
_UNARY_C = {
  POps.SIN: "sin",
  POps.COS: "cos",
  POps.TAN: "tan",
  POps.ASIN: "asin",
  POps.ACOS: "acos",
  POps.ATAN: "atan",
  POps.SINH: "sinh",
  POps.COSH: "cosh",
  POps.TANH: "tanh",
  POps.EXP: "exp",
  POps.LOG: "log",
  POps.SQRT: "sqrt",
  POps.ABS: "fabs",
  POps.FLOOR: "floor",
  POps.CEIL: "ceil",
}
_BINARY_C = {POps.POW: "pow", POps.ATAN2: "atan2", POps.MINIMUM: "fmin", POps.MAXIMUM: "fmax"}


def use_program_ir_renderer() -> bool:
  return os.environ.get("ALLOY_USE_PROGRAM_IR_C") == "1"


def program_ir_allow_fallback() -> bool:
  """When the Program IR renderer is selected, allow silent fallback to the legacy
  renderer on ``LoweringError``. Off by default — selection is otherwise strict."""
  return os.environ.get("ALLOY_PROGRAM_IR_FALLBACK") == "1"


def can_render_program_c(fun: Function) -> bool:
  """True iff ``fun`` lowers and renders through the Program IR path. Diagnostic helper
  (e.g. for coverage probes); the hot path just calls ``render_program_c_source``."""
  if fun.device.kind != "host":
    return False
  try:
    render_program_c_source(fun)
  except LoweringError:
    return False
  return True


def program_ir_sz_w(fun: Function) -> int:
  """The scalar workspace size (doubles) the Program IR renderer needs for ``fun`` — i.e. what its
  ``<symbol>_sz_w`` would return. Used so the ABI header agrees with the rendered source under the
  flag (the JIT reads the runtime helper directly; AOT consumers read the header macro)."""
  return int(main_proc(lower_function(fun)).attrs.get("sz_w", 0))


def render_program_c_source(fun: Function) -> str:
  prog = lower_function(fun)
  proc = main_proc(prog)
  pc = int(prog.attrs.get("proc_count", 1))
  callees = list(prog.args[: pc - 1])
  symbol = _c_ident(fun.name)  # must match jit.CompiledFunction's _c_ident(fun.name)
  param_count = int(proc.attrs["param_count"])
  body = list(proc.args[param_count:])
  sz_w = int(proc.attrs.get("sz_w", 0))  # set by the workspace-packing pass (passes.py)

  # Buffer name -> C pointer expression for the ABI entry (inputs are arg[i], outputs res[i]).
  ptr_expr: dict[str, str] = {}
  for i, name in enumerate(fun.input_names):
    ptr_expr[name] = f"arg[{i}]"
  for i, name in enumerate(fun.output_names):
    ptr_expr[name] = f"res[{i}]"

  lines: list[str] = [
    "#include <math.h>",
    "#include <stddef.h>",
    "#include <stdint.h>",
    "",
    *_ABI_DEFINES,
    "",
    "#ifdef __cplusplus",
    'extern "C" {',
    "#endif",
    "",
  ]
  for callee in callees:
    lines += _render_raw_callee(callee)
    lines.append("")
  lines += [
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
  for stmt in body:
    if stmt.op == POps.BUFFER:
      continue
    _emit_statement(stmt, ptr_expr, lines, indent=2)
  lines += ["  return ALLOY_SUCCESS;", "}", "", "#ifdef __cplusplus", "}", "#endif"]
  return "\n".join(lines).rstrip() + "\n"


def _render_raw_callee(proc: PNode) -> list[str]:
  """A callee renders as ``static inline void <name>_raw(<dtype>* p0, ..., double* w)`` — a
  pointer per param plus the workspace tail (spilled slots index into ``w``; ``call`` sites pass
  the caller's ``w`` advanced past its own spill window). No ABI wrapper."""
  param_count = int(proc.attrs["param_count"])
  params = list(proc.args[:param_count])
  body = list(proc.args[param_count:])
  sz_w = int(proc.attrs.get("sz_w", 0))
  ptr_expr = {pp.attrs["name"]: _c_ident(pp.attrs["name"]) for pp in params}
  param_decls = ", ".join([*(f"{pp.dtype.c_type}* {_c_ident(pp.attrs['name'])}" for pp in params), "double* w"])
  out = [f"static inline void {_c_ident(proc.attrs['name'])}_raw({param_decls}) {{"]
  if not sz_w:
    out.append("  (void)w;")
  _emit_local_buffers(body, out, ptr_expr, indent=2)
  for stmt in body:
    if stmt.op == POps.BUFFER:
      continue
    _emit_statement(stmt, ptr_expr, out, indent=2)
  out.append("}")
  return out


def _emit_local_buffers(body: list[PNode], lines: list[str], ptr_expr: dict[str, str], indent: int) -> None:
  pad = " " * indent
  seen: set[str] = set()
  for stmt in body:
    if stmt.op != POps.BUFFER or stmt.attrs["name"] in seen:
      continue
    seen.add(stmt.attrs["name"])
    name = _c_ident(stmt.attrs["name"])
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
      src = ptr_expr.get(stmt.attrs["alias_of"], _c_ident(stmt.attrs["alias_of"]))
      offset = int(stmt.attrs["alias_offset"])
      rhs = src if offset == 0 else f"{src} + {offset}"
      lines.append(f"{pad}const {stmt.dtype.c_type}* {name} = {rhs};")
    elif "workspace_offset" in stmt.attrs:
      # A spilled slot: a window into the caller-provided w[] instead of a stack array.
      lines.append(f"{pad}{stmt.dtype.c_type}* {name} = w + {stmt.attrs['workspace_offset']};")
    else:
      lines.append(f"{pad}{stmt.dtype.c_type} {name}[{size}];")


def _emit_statement(stmt: PNode, ptr_expr: dict[str, str], lines: list[str], indent: int) -> None:
  pad = " " * indent
  if stmt.op == POps.FOR:
    rng = stmt.args[0]
    name = _c_ident(rng.attrs["name"])
    start = _emit_scalar(rng.args[0], ptr_expr)
    stop = _emit_scalar(rng.args[1], ptr_expr)
    step = _emit_scalar(rng.args[2], ptr_expr)
    incr = f"++{name}" if step == "1" else f"{name} += {step}"
    lines.append(f"{pad}for (long long {name} = {start}; {name} < {stop}; {incr}) {{")
    for sub in stmt.args[1:]:
      _emit_statement(sub, ptr_expr, lines, indent + 2)
    lines.append(f"{pad}}}")
  elif stmt.op == POps.STORE:
    lines.append(f"{pad}{_emit_view(stmt.args[0], ptr_expr)} = {_emit_scalar(stmt.args[1], ptr_expr)};")
  elif stmt.op == POps.ASSIGN:
    lines.append(f"{pad}{stmt.attrs['target']} = {_emit_scalar(stmt.args[0], ptr_expr)};")
  elif stmt.op == POps.CALL:
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
    lines.append(f"{pad}{_c_ident(stmt.attrs['callee'])}_raw({', '.join(ptrs)});")
  else:
    raise LoweringError(f"Program IR C renderer: statement op {stmt.op} not yet handled")


def _emit_call_arg(node: PNode, ptr_expr: dict[str, str]) -> str:
  """A CALL argument is a whole BUFFER (its pointer) or a VIEW (pointer + offset)."""
  if node.op == POps.BUFFER:
    return ptr_expr.get(node.attrs["name"], _c_ident(node.attrs["name"]))
  if node.op == POps.VIEW:
    ptr = ptr_expr.get(node.attrs["buffer"], _c_ident(node.attrs["buffer"]))
    idx = _emit_scalar(node.args[0], ptr_expr) if node.args else "0"
    return ptr if idx == "0" else f"({ptr} + {idx})"
  raise LoweringError(f"unsupported CALL arg op {node.op}")


def _emit_view(view: PNode, ptr_expr: dict[str, str]) -> str:
  if view.op != POps.VIEW:
    raise LoweringError(f"expected a VIEW, got {view.op}")
  ptr = ptr_expr.get(view.attrs["buffer"], _c_ident(view.attrs["buffer"]))
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


def _emit_scalar(n: PNode, ptr_expr: dict[str, str]) -> str:
  op = n.op
  if op == POps.CONST_INT:
    return str(n.attrs["value"])
  if op == POps.CONST_FLOAT:
    return _c_float(n.attrs["value"])
  if op == POps.VAR:
    return _c_ident(n.attrs["name"])
  if op == POps.LOAD:
    return _emit_view(n.args[0], ptr_expr)
  if op == POps.NEG:
    return f"(-{_emit_scalar(n.args[0], ptr_expr)})"
  if op in _BIN_SYM:
    return f"({_emit_scalar(n.args[0], ptr_expr)} {_BIN_SYM[op]} {_emit_scalar(n.args[1], ptr_expr)})"
  if op in _UNARY_C:
    return f"{_UNARY_C[op]}({_emit_scalar(n.args[0], ptr_expr)})"
  if op in _BINARY_C:
    return f"{_BINARY_C[op]}({_emit_scalar(n.args[0], ptr_expr)}, {_emit_scalar(n.args[1], ptr_expr)})"
  raise LoweringError(f"Program IR C renderer: scalar op {op} not yet handled")


__all__ = [
  "can_render_program_c",
  "program_ir_allow_fallback",
  "program_ir_sz_w",
  "render_program_c_source",
  "use_program_ir_renderer",
]
