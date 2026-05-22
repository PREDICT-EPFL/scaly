"""Phase 8 follow-up: Metal Shading Language (MSL) kernel source rendering.

Generates MSL kernel source for ``device='metal:N'`` Functions. The kernel uses:

- ``device float*`` buffers (Metal does not support float64, which the
  ``BackendSupport`` table for ``metal`` already enforces at construction).
- ``[[buffer(N)]]`` attributes for each kernel argument in order.
- ``uint tid [[thread_position_in_grid]]`` as the index into the bound
  top-level ``GLOBAL`` FOR.

Host-side dispatch (``MTLDevice``/``MTLCommandQueue``/``MTLComputePipelineState``)
is not generated here — it requires Objective-C, Swift, or metal-cpp bindings.
This first slice covers the kernel source; the host driver lands when there's
a runtime path (PyObjC or metal-cpp) to actually dispatch on this machine.

To verify the generated MSL is syntactically valid:

    xcrun metal -x metal -std=metal2.4 -o /tmp/kernel.air f.metal

This requires the Xcode Metal Toolchain (``xcodebuild -downloadComponent
MetalToolchain``); the optional ``test_metal_msl_compiles_with_xcrun`` test
skips when it's not installed.
"""

from __future__ import annotations

from ..function import Function
from ..lowering import lower_function, main_proc
from ..program import PNode, POps


def can_render_metal(fun: Function) -> bool:
  return fun.device.kind == "metal"


def render_metal_source(fun: Function) -> str:
  if fun.device.kind != "metal":
    raise ValueError(f"render_metal_source needs device='metal:N', got {fun.device}")
  prog = lower_function(fun)
  pc = int(prog.attrs["proc_count"])
  if int(prog.attrs.get("kernel_count", 0)) < 1:
    raise ValueError(f"lowered program for {fun.name!r} has no KERNEL — schedule pass failed")
  kernel = prog.args[pc]  # first kernel after the procs
  # Callees: every PROC before the main one is rendered as an MSL inline
  # function the kernel can call. The main driver PROC is the last proc and
  # we don't emit it (the kernel itself replaces it on the device side).
  main = main_proc(prog)
  callees = [p for p in prog.args[:pc] if p is not main]
  lines: list[str] = [
    "#include <metal_stdlib>",
    "using namespace metal;",
    "",
  ]
  for callee in callees:
    lines += _render_metal_callee(callee)
    lines.append("")
  lines += _render_metal_kernel(kernel)
  return "\n".join(lines).rstrip() + "\n"


def _render_metal_callee(proc: PNode) -> list[str]:
  """Emit a callee PROC as an MSL ``static inline void`` function.

  Arguments mirror the PROC's BUFFER params (every arg is ``device <type>*``).
  The body is rendered with the same statement emitter as the kernel, minus
  the thread-binding step (a callee runs entirely in the calling thread's
  context).
  """
  param_count = int(proc.attrs["param_count"])
  params = list(proc.args[:param_count])
  body = list(proc.args[param_count:])
  decls = ", ".join(f"device {_msl_type(pp.dtype.c_type)}* {pp.attrs['name']}" for pp in params)
  out: list[str] = [f"static inline void {proc.attrs['name']}({decls}) {{"]
  ptr_expr = {pp.attrs["name"]: pp.attrs["name"] for pp in params}
  for stmt in body:
    if stmt.op == POps.BUFFER and stmt.attrs["name"] not in ptr_expr:
      size = 1
      for d in stmt.attrs["shape"]:
        size *= int(d)
      size = size or 1
      out.append(f"  thread {_msl_type(stmt.dtype.c_type)} {stmt.attrs['name']}[{size}];")
  for stmt in body:
    if stmt.op == POps.BUFFER:
      continue
    _emit_stmt_metal(stmt, ptr_expr, out, indent=2)
  out.append("}")
  return out


def _render_metal_kernel(kernel: PNode) -> list[str]:
  param_count = int(kernel.attrs["param_count"])
  params = list(kernel.args[:param_count])
  body = list(kernel.args[param_count:])
  # Build MSL kernel signature with [[buffer(N)]] attributes.
  param_decls: list[str] = []
  for i, pp in enumerate(params):
    # Metal's analogues for our dtypes. ``double`` does not exist on Metal GPUs;
    # construction-time BackendSupport already rejects it for metal devices.
    msl_type = _msl_type(pp.dtype.c_type)
    param_decls.append(f"device {msl_type}* {pp.attrs['name']} [[buffer({i})]]")
  param_decls.append("uint tid [[thread_position_in_grid]]")
  out: list[str] = [
    f"kernel void {kernel.attrs['name']}(",
    "    " + ",\n    ".join(param_decls),
    ") {",
  ]
  ptr_expr = {pp.attrs["name"]: pp.attrs["name"] for pp in params}
  ts: list[PNode] = []
  for stmt in body:
    if stmt.op == POps.BUFFER and stmt.attrs["name"] not in ptr_expr and stmt not in ts:
      ts.append(stmt)
  for tb in ts:
    size = 1
    for d in tb.attrs["shape"]:
      size *= int(d)
    size = size or 1
    out.append(f"  {_msl_type(tb.dtype.c_type)} {tb.attrs['name']}[{size}];")
  bound = False
  for stmt in body:
    if stmt.op == POps.BUFFER:
      continue
    kind = stmt.args[0].attrs.get("kind") if stmt.op == POps.FOR else None
    if not bound and kind is not None and getattr(kind, "value", None) == "global":
      _emit_thread_bound_for_metal(stmt, ptr_expr, out, indent=2)
      bound = True
    else:
      _emit_stmt_metal(stmt, ptr_expr, out, indent=2)
  out.append("}")
  return out


def _msl_type(c_type: str) -> str:
  # Map our ``DType.c_type`` strings to MSL spellings.
  return {
    "double": "float",  # rejected by BackendSupport, but defensive
    "float": "float",
    "int32_t": "int",
    "int64_t": "long",
    "uint8_t": "uchar",
  }.get(c_type, c_type)


def _emit_thread_bound_for_metal(stmt: PNode, ptr_expr: dict[str, str], lines: list[str], indent: int) -> None:
  pad = " " * indent
  rng = stmt.args[0]
  name = rng.attrs["name"]
  start = _emit_scalar(rng.args[0])
  stop = _emit_scalar(rng.args[1])
  lines.append(f"{pad}long {name} = (long)tid + {start};")
  lines.append(f"{pad}if ({name} >= {stop}) return;")
  for sub in stmt.args[1:]:
    _emit_stmt_metal(sub, ptr_expr, lines, indent)


def _emit_stmt_metal(stmt: PNode, ptr_expr: dict[str, str], lines: list[str], indent: int) -> None:
  pad = " " * indent
  if stmt.op == POps.FOR:
    rng = stmt.args[0]
    name = rng.attrs["name"]
    start = _emit_scalar(rng.args[0])
    stop = _emit_scalar(rng.args[1])
    step = _emit_scalar(rng.args[2])
    incr = "++" if step == "1" else f" += {step}"
    lines.append(f"{pad}for (long {name} = {start}; {name} < {stop}; {name}{incr}) {{")
    for sub in stmt.args[1:]:
      _emit_stmt_metal(sub, ptr_expr, lines, indent + 2)
    lines.append(f"{pad}}}")
  elif stmt.op == POps.STORE:
    view = stmt.args[0]
    value = stmt.args[1]
    buf = view.attrs["buffer"]
    ptr = ptr_expr.get(buf, buf)
    idx = _emit_scalar(view.args[0]) if view.args else "0"
    rhs = _emit_scalar(value)
    lines.append(f"{pad}{ptr}[{idx}] = {rhs};")
  elif stmt.op == POps.ASSIGN:
    lines.append(f"{pad}{stmt.attrs['target']} = {_emit_scalar(stmt.args[0])};")
  elif stmt.op == POps.CALL:
    n_in = int(stmt.attrs["n_in"])
    n_out = int(stmt.attrs["n_out"])
    in_args = stmt.args[:n_in]
    out_args = stmt.args[n_in : n_in + n_out]
    parts = [_emit_call_arg_metal(a, ptr_expr) for a in (*in_args, *out_args)]
    lines.append(f"{pad}{stmt.attrs['callee']}({', '.join(parts)});")
  else:
    raise NotImplementedError(f"Metal renderer: statement op {stmt.op} not handled")


def _emit_call_arg_metal(node: PNode, ptr_expr: dict[str, str]) -> str:
  """Render a CALL argument: BUFFER (bare pointer) or VIEW (pointer + offset)."""
  if node.op == POps.BUFFER:
    return ptr_expr.get(node.attrs["name"], node.attrs["name"])
  if node.op == POps.VIEW:
    buf = node.attrs["buffer"]
    ptr = ptr_expr.get(buf, buf)
    idx = _emit_scalar(node.args[0]) if node.args else "0"
    return f"({ptr} + {idx})"
  raise NotImplementedError(f"Metal CALL arg op {node.op}")


def _emit_scalar(n: PNode) -> str:
  if n.op == POps.CONST_INT:
    return str(n.attrs["value"])
  if n.op == POps.CONST_FLOAT:
    v = n.attrs["value"]
    import math

    if math.isnan(v):
      return "NAN"
    if math.isinf(v):
      return "(-INFINITY)" if v < 0 else "INFINITY"
    # MSL rejects ``0f`` / ``1f`` (parsed as an octal/decimal int with a stray
    # 'f') — the float literal needs a decimal point or exponent. ``g`` skips
    # the dot for whole numbers, so force a fractional form.
    s = f"{v:.9g}"
    if "." not in s and "e" not in s and "E" not in s:
      s = s + ".0"
    return s + "f"
  if n.op == POps.VAR:
    return str(n.attrs["name"])
  if n.op == POps.LOAD:
    view = n.args[0]
    buf = view.attrs["buffer"]
    idx = _emit_scalar(view.args[0]) if view.args else "0"
    return f"{buf}[{idx}]"
  binop = {POps.ADD: "+", POps.SUB: "-", POps.MUL: "*", POps.DIV: "/"}.get(n.op)
  if binop is not None:
    return f"({_emit_scalar(n.args[0])} {binop} {_emit_scalar(n.args[1])})"
  if n.op == POps.NEG:
    return f"(-{_emit_scalar(n.args[0])})"
  unary = {
    POps.SIN: "metal::sin",
    POps.COS: "metal::cos",
    POps.TAN: "metal::tan",
    POps.EXP: "metal::exp",
    POps.LOG: "metal::log",
    POps.SQRT: "metal::sqrt",
    POps.TANH: "metal::tanh",
    POps.ABS: "metal::abs",
    POps.FLOOR: "metal::floor",
    POps.CEIL: "metal::ceil",
  }.get(n.op)
  if unary is not None:
    return f"{unary}({_emit_scalar(n.args[0])})"
  raise NotImplementedError(f"Metal renderer scalar op {n.op} not handled")


__all__ = ["can_render_metal", "render_metal_source"]
