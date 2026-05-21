"""Render a Program IR ``PROC`` to standalone scalar C.

Phase 5 (feature-flagged): this is the future code path for the existing
scalar C renderer. Today it only handles host PROCs containing elementwise
``FOR`` loops over ``GLOBAL`` ranges with ``LOAD``/``STORE``/``ASSIGN``
statements — i.e. the slice that ``alloy.lowering.lower_function`` can
produce.

Enable it via ``ALLOY_USE_PROGRAM_IR_C=1`` for ``Function`` instances whose
semantic IR is in the supported subset; everything else falls back to the
existing scalar C renderer in ``alloy.codegen.c``. Output is independently
compiled and exercised through the same JIT path, so numerical equivalence
is the lock.
"""

from __future__ import annotations

import os

from ..abi import c_api_signature
from ..function import Function
from ..lowering import LoweringError, lower_function, main_proc
from ..program import PNode, POps


_ABI_DEFINES = (
  "#define ALLOY_SUCCESS 0",
  "#define ALLOY_ERR_NULL_ABI 1",
  "#define ALLOY_ERR_NULL_WORK 2",
  "#define ALLOY_ERR_NULL_RESULT 3",
  "#define ALLOY_ERR_NULL_INPUT 4",
)


def use_program_ir_renderer() -> bool:
  return os.environ.get("ALLOY_USE_PROGRAM_IR_C") == "1"


def can_render_program_c(fun: Function) -> bool:
  """Return True iff the Phase 5 lowerer + renderer covers ``fun``'s semantic ops.

  Today this is a cheap probe: try to lower; if it raises ``LoweringError`` we
  fall back to the legacy renderer. The lowering is fast for the small graphs
  it accepts and the result is cached on the Function instance.
  """
  if fun.device.kind != "host":
    return False
  try:
    lower_function(fun)
  except LoweringError:
    return False
  return True


def render_program_c_source(fun: Function) -> str:
  prog = lower_function(fun)
  proc = main_proc(prog)
  pc = int(prog.attrs.get("proc_count", 1))
  callees = list(prog.args[: pc - 1])
  symbol = fun.name
  param_count = int(proc.attrs["param_count"])
  buffer_params = list(proc.args[:param_count])
  body = list(proc.args[param_count:])

  # Map buffer name -> index into arg/res, and the C pointer expression.
  ptr_expr: dict[str, str] = {}
  workspace_buffers: list[PNode] = []
  for i, name in enumerate(fun.input_names):
    ptr_expr[name] = f"arg[{i}]"
  for i, name in enumerate(fun.output_names):
    ptr_expr[name] = f"res[{i}]"
  for buf in buffer_params:
    n = buf.attrs["name"]
    if n not in ptr_expr:
      workspace_buffers.append(buf)
  # Internal "tN" buffers materialize as local arrays. The lowerer emits a
  # BUFFER PNode as a statement at the point of declaration; we collect those.
  ts: list[PNode] = []
  for stmt in body:
    if stmt.op == POps.BUFFER and stmt.attrs["name"] not in ptr_expr:
      if stmt not in ts:
        ts.append(stmt)

  lines: list[str] = [
    "#include <math.h>",
    "#include <stddef.h>",
    "",
    *_ABI_DEFINES,
    "",
    "#ifdef __cplusplus",
    'extern "C" {',
    "#endif",
    "",
  ]
  for callee_proc in callees:
    lines += _render_raw_callee(callee_proc)
    lines.append("")
  lines += [
    f"int {symbol}_sz_arg(void) {{ return {len(fun.inputs)}; }}",
    f"int {symbol}_sz_res(void) {{ return {len(fun.outputs)}; }}",
    f"int {symbol}_sz_iw(void) {{ return 0; }}",
    f"int {symbol}_sz_w(void) {{ return 0; }}",
    f"void* {symbol}_alloc_mem(void) {{ return NULL; }}",
    f"int {symbol}_init_mem(void* mem) {{ (void)mem; return ALLOY_SUCCESS; }}",
    f"void {symbol}_free_mem(void* mem) {{ (void)mem; }}",
    "",
    c_api_signature(symbol) + " {",
    "  (void)iw;",
    "  (void)w;",
    "  (void)mem;",
    "  if (!arg || !res) return ALLOY_ERR_NULL_ABI;",
  ]
  for i in range(len(fun.inputs)):
    lines.append(f"  if (!arg[{i}]) return ALLOY_ERR_NULL_INPUT;")
  for i in range(len(fun.outputs)):
    lines.append(f"  if (!res[{i}]) return ALLOY_ERR_NULL_RESULT;")
  # workspace t-buffers declared as local arrays
  for tb in ts:
    size = 1
    for d in tb.attrs["shape"]:
      size *= int(d)
    size = size or 1
    lines.append(f"  {tb.dtype.c_type} {tb.attrs['name']}[{size}];")
  # render body (skip BUFFER decls already lifted to local arrays)
  for stmt in body:
    if stmt.op == POps.BUFFER:
      continue
    _emit_statement(stmt, ptr_expr, lines, indent=2)
  lines.append("  return ALLOY_SUCCESS;")
  lines.append("}")
  lines.append("")
  lines.append("#ifdef __cplusplus")
  lines.append("}")
  lines.append("#endif")
  return "\n".join(lines).rstrip() + "\n"


def _render_raw_callee(proc: PNode) -> list[str]:
  """Render a callee as ``static inline void``: takes pointers per param, no ABI wrapper."""
  param_count = int(proc.attrs["param_count"])
  params = list(proc.args[:param_count])
  body = list(proc.args[param_count:])
  ptr_expr: dict[str, str] = {p.attrs["name"]: p.attrs["name"] for p in params}
  ts: list[PNode] = []
  for stmt in body:
    if stmt.op == POps.BUFFER and stmt.attrs["name"] not in ptr_expr and stmt not in ts:
      ts.append(stmt)
  param_decls = ", ".join(f"{p.dtype.c_type}* {p.attrs['name']}" for p in params)
  out: list[str] = [f"static inline void {proc.attrs['name']}_raw({param_decls}) {{"]
  for tb in ts:
    size = 1
    for d in tb.attrs["shape"]:
      size *= int(d)
    size = size or 1
    out.append(f"  {tb.dtype.c_type} {tb.attrs['name']}[{size}];")
  for stmt in body:
    if stmt.op == POps.BUFFER:
      continue
    _emit_statement(stmt, ptr_expr, out, indent=2)
  out.append("}")
  return out


def _emit_call_arg(node: PNode, ptr_expr: dict[str, str]) -> str:
  """Render a CALL argument: a BUFFER (whole pointer) or a VIEW (buffer + offset)."""
  if node.op == POps.BUFFER:
    return ptr_expr.get(node.attrs["name"], node.attrs["name"])
  if node.op == POps.VIEW:
    buf = node.attrs["buffer"]
    ptr = ptr_expr.get(buf, buf)
    idx = _emit_scalar(node.args[0], ptr_expr) if node.args else "0"
    return f"({ptr} + {idx})"
  raise NotImplementedError(f"unsupported CALL arg op {node.op}")


def _emit_statement(stmt: PNode, ptr_expr: dict[str, str], lines: list[str], indent: int) -> None:
  pad = " " * indent
  if stmt.op == POps.FOR:
    rng = stmt.args[0]
    # FOR with a body that's currently a single STORE; broadcast to N
    name = rng.attrs["name"]
    start = _emit_scalar(rng.args[0], ptr_expr)
    stop = _emit_scalar(rng.args[1], ptr_expr)
    step = _emit_scalar(rng.args[2], ptr_expr)
    kind = rng.attrs["kind"]
    _ = kind  # for the host-only renderer we treat all kinds as serial loops; GPU rendering will use it
    if step == "1":
      lines.append(f"{pad}for (long long {name} = {start}; {name} < {stop}; ++{name}) {{")
    else:
      lines.append(f"{pad}for (long long {name} = {start}; {name} < {stop}; {name} += {step}) {{")
    for sub in stmt.args[1:]:
      _emit_statement(sub, ptr_expr, lines, indent + 2)
    lines.append(f"{pad}}}")
  elif stmt.op == POps.STORE:
    view = stmt.args[0]
    value = stmt.args[1]
    buf = view.attrs["buffer"]
    ptr = ptr_expr.get(buf, buf)
    idx = _emit_scalar(view.args[0], ptr_expr) if view.args else "0"
    rhs = _emit_scalar(value, ptr_expr)
    lines.append(f"{pad}{ptr}[{idx}] = {rhs};")
  elif stmt.op == POps.ASSIGN:
    lines.append(f"{pad}{stmt.attrs['target']} = {_emit_scalar(stmt.args[0], ptr_expr)};")
  elif stmt.op == POps.CALL:
    n_in = int(stmt.attrs["n_in"])
    n_out = int(stmt.attrs["n_out"])
    in_args = stmt.args[:n_in]
    out_args = stmt.args[n_in : n_in + n_out]
    arg_ptrs = ", ".join(_emit_call_arg(a, ptr_expr) for a in in_args)
    out_ptrs = ", ".join(_emit_call_arg(a, ptr_expr) for a in out_args)
    lines.append(f"{pad}{stmt.attrs['callee']}_raw({arg_ptrs}{', ' if arg_ptrs and out_ptrs else ''}{out_ptrs});")
  else:
    raise NotImplementedError(f"Program IR C renderer: statement op {stmt.op} not yet handled")


def _emit_scalar(n: PNode, ptr_expr: dict[str, str]) -> str:
  if n.op == POps.CONST_INT:
    return str(n.attrs["value"])
  if n.op == POps.CONST_FLOAT:
    v = n.attrs["value"]
    import math

    if math.isnan(v):
      return "((double)NAN)"
    if math.isinf(v):
      return "((double)(-INFINITY))" if v < 0 else "((double)INFINITY)"
    return f"{v:.17g}"
  if n.op == POps.VAR:
    return str(n.attrs["name"])
  if n.op == POps.LOAD:
    view = n.args[0]
    buf = view.attrs["buffer"]
    ptr = ptr_expr.get(buf, buf)
    idx = _emit_scalar(view.args[0], ptr_expr) if view.args else "0"
    return f"{ptr}[{idx}]"
  if n.op == POps.ADD:
    return f"({_emit_scalar(n.args[0], ptr_expr)} + {_emit_scalar(n.args[1], ptr_expr)})"
  if n.op == POps.SUB:
    return f"({_emit_scalar(n.args[0], ptr_expr)} - {_emit_scalar(n.args[1], ptr_expr)})"
  if n.op == POps.MUL:
    return f"({_emit_scalar(n.args[0], ptr_expr)} * {_emit_scalar(n.args[1], ptr_expr)})"
  if n.op == POps.DIV:
    return f"({_emit_scalar(n.args[0], ptr_expr)} / {_emit_scalar(n.args[1], ptr_expr)})"
  if n.op == POps.NEG:
    return f"(-{_emit_scalar(n.args[0], ptr_expr)})"
  if n.op in _UNARY_FN_NAMES:
    return f"{_UNARY_FN_NAMES[n.op]}({_emit_scalar(n.args[0], ptr_expr)})"
  if n.op in _BINARY_FN_NAMES:
    return f"{_BINARY_FN_NAMES[n.op]}({_emit_scalar(n.args[0], ptr_expr)}, {_emit_scalar(n.args[1], ptr_expr)})"
  if n.op == POps.MOD:
    return f"({_emit_scalar(n.args[0], ptr_expr)} % {_emit_scalar(n.args[1], ptr_expr)})"
  raise NotImplementedError(f"Program IR C renderer: scalar op {n.op} not yet handled")


_UNARY_FN_NAMES = {
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


_BINARY_FN_NAMES = {
  POps.POW: "pow",
  POps.ATAN2: "atan2",
  POps.MINIMUM: "fmin",
  POps.MAXIMUM: "fmax",
}


__all__ = ["can_render_program_c", "render_program_c_source", "use_program_ir_renderer"]
