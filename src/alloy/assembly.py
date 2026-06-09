"""Textual assembly and DAG serialization for Alloy IR dialects.

This is deliberately a printer, not a parser.  The goal is an MLIR/LLVM-like
stable textual form that can be diffed, copied out of the visualizer, and used
in tests/debugging without coupling it to C syntax.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Set
from typing import Any

import numpy as np

from .expr import Expr, topo
from .function import Function
from .ops import Ops
from .program import BINARY_FN_OPS, SCALAR_OPS, UNARY_FN_OPS, PNode, POps
from .types import TensorType

_EXPR_COLORS = {
  Ops.INPUT: "#c0c0ff",
  Ops.CONST: "#e0e0e0",
  Ops.CALL: "#00b7c8",
  Ops.MAP: "#f6ccff",
  Ops.ADD: "#ffffc0",
  Ops.SUB: "#ffffc0",
  Ops.MUL: "#ffffc0",
  Ops.DIV: "#ffffc0",
  Ops.NEG: "#ffffc0",
  Ops.SUM: "#ffb0b0",
  Ops.MATMUL: "#ffb0b0",
  Ops.RESHAPE: "#d8f9e4",
  Ops.TRANSPOSE: "#d8f9e4",
  Ops.SLICE: "#e5eaff",
  Ops.GATHER: "#e5eaff",
  Ops.SCATTER: "#e5eaff",
  Ops.STACK: "#ffc14d",
  Ops.CONCAT: "#ffc14d",
}

_PNODE_COLORS = {
  POps.PROGRAM: "#c07788",
  POps.PROC: "#c07788",
  POps.KERNEL: "#c07788",
  POps.BUFFER: "#b0bdff",
  POps.VIEW: "#e5eaff",
  POps.RANGE: "#c8a0e0",
  POps.FOR: "#c8a0e0",
  POps.STORE: "#87ceeb",
  POps.LOAD: "#ffc0c0",
  POps.CALL: "#00b7c8",
  POps.LAUNCH: "#00b7c8",
  POps.CONST_INT: "#e0e0e0",
  POps.CONST_FLOAT: "#e0e0e0",
  POps.VAR: "#cef263",
}


def type_asm(t: TensorType) -> str:
  shape = "x".join(str(d) for d in t.shape)
  dims = f"{shape}x" if shape else ""
  diff = " diff" if t.diff else ""
  sparse = " sparse" if t.sparsity is not None else ""
  return f"tensor<{dims}{t.dtype.name}{diff}{sparse}>"


def _memref_asm(n: PNode) -> str:
  shape = "x".join(str(d) for d in n.attrs.get("shape", ()))
  dims = f"{shape}x" if shape else ""
  space = n.attrs.get("address_space")
  space_suffix = f", {space}" if space and space != "global" else ""
  return f"memref<{dims}{n.dtype.name}{space_suffix}>"


def _value_asm(v: Any) -> str:
  if isinstance(v, np.ndarray):
    return np.array2string(v, threshold=8, separator=", ")
  if isinstance(v, str):
    return json.dumps(v)
  if isinstance(v, tuple):
    return "[" + ", ".join(_value_asm(x) for x in v) + "]"
  if isinstance(v, list):
    return "[" + ", ".join(_value_asm(x) for x in v) + "]"
  if isinstance(v, dict):
    return "{" + ", ".join(f"{k}={_value_asm(val)}" for k, val in sorted(v.items())) + "}"
  if hasattr(v, "name") and v.__class__.__name__ == "Function":
    return "@" + v.name
  return str(v)


def _attrs_asm(attrs: dict[str, Any], *, skip: Set[str] = frozenset()) -> str:
  items = [(k, v) for k, v in sorted(attrs.items()) if k not in skip]
  if not items:
    return ""
  return " {" + ", ".join(f"{k}={_value_asm(v)}" for k, v in items) + "}"


# ---------------------------------------------------------------------------
# Semantic dialect assembly.
# ---------------------------------------------------------------------------


def render_expr_assembly(obj: Function | Expr | Iterable[Expr], *, name: str | None = None) -> str:
  """Render the semantic ``Expr`` dialect as a compact SSA assembly listing.

  When given a ``Function``, the listing is a ``sem.module`` containing every transitive semantic
  callee body before the requested function. This mirrors what C rendering eventually needs.
  """
  if isinstance(obj, Function):
    return _render_function_module(obj)
  outs = (obj,) if isinstance(obj, Expr) else tuple(obj)
  return _render_expr_region(outs, name=name)


def _semantic_callees(fun: Function) -> list[Function]:
  seen: set[int] = set()
  ordered: list[Function] = []

  def visit(fn: Function) -> None:
    if id(fn) in seen:
      return
    seen.add(id(fn))
    for node in topo(fn.outputs):
      if node.op in {Ops.CALL, Ops.MAP}:
        callee = node.attrs.get("callee")
        if isinstance(callee, Function):
          visit(callee)
    ordered.append(fn)

  visit(fun)
  return ordered


def _render_function_module(fun: Function) -> str:
  lines = ["sem.module {"]
  for fn in _semantic_callees(fun):
    body = _render_function_expr_assembly(fn).splitlines()
    lines += ["  " + line for line in body]
  lines.append("}")
  return "\n".join(lines)


def _render_function_expr_assembly(fun: Function) -> str:
  ins = ", ".join(f"%{n}: {type_asm(e.type)}" for n, e in zip(fun.input_names, fun.inputs, strict=True))
  outs = ", ".join(f"%{n}: {type_asm(e.type)}" for n, e in zip(fun.output_names, fun.outputs, strict=True))
  lines = [f"sem.func @{fun.name}({ins}) -> ({outs}) {{"]
  body = _render_expr_region(fun.outputs).splitlines()
  lines += ["  " + line for line in body]
  lines.append("}")
  return "\n".join(lines)


def _render_expr_region(outputs: Iterable[Expr], *, name: str | None = None) -> str:
  outs = tuple(outputs)
  nodes = topo(outs)
  loc = {e.id: i for i, e in enumerate(nodes)}
  lines: list[str] = [f"sem.region @{name} {{"] if name else []
  pad = "  " if name else ""
  for i, e in enumerate(nodes):
    args = ", ".join(f"%{loc[a.id]}" for a in e.args)
    op = Ops(e.op)
    attrs = dict(e.attrs)
    if op == Ops.INPUT:
      attrs = {**attrs, "name": e.name, "lowering": e.lowering}
    elif op == Ops.CONST:
      attrs = {**attrs, "value": e.value, "lowering": e.lowering}
    elif op in {Ops.CALL, Ops.MAP}:
      callee = attrs.get("callee")
      attrs = {**attrs, "callee": getattr(callee, "name", callee)}
    text_args = f"({args})" if args else ""
    lines.append(f"{pad}%{i} = sem.{op.value}{text_args}{_attrs_asm(attrs)} : {type_asm(e.type)}")
  lines.append(f"{pad}sem.return " + ", ".join(f"%{loc[e.id]}" for e in outs))
  if name:
    lines.append("}")
  return "\n".join(lines)


# ---------------------------------------------------------------------------
# Program dialect assembly.
# ---------------------------------------------------------------------------


def render_program_assembly(root: PNode) -> str:
  """Render Program IR as structured assembly with explicit control-flow constructs."""
  lines: list[str] = []
  _render_program_node(root, 0, lines)
  return "\n".join(lines)


def _render_program_node(n: PNode, indent: int, lines: list[str]) -> None:
  pad = "  " * indent
  if n.op == POps.PROGRAM:
    lines.append(f"{pad}prog.module{_attrs_asm(n.attrs, skip={'proc_count', 'kernel_count'})} {{")
    pc = int(n.attrs.get("proc_count", 0))
    for sub in n.args[:pc]:
      _render_program_node(sub, indent + 1, lines)
    for sub in n.args[pc:]:
      _render_program_node(sub, indent + 1, lines)
    lines.append(f"{pad}}}")
  elif n.op in (POps.PROC, POps.KERNEL):
    label = "proc" if n.op == POps.PROC else "kernel"
    pc = int(n.attrs["param_count"])
    params = ", ".join(f"%{p.attrs['name']}: {_memref_asm(p)}" for p in n.args[:pc])
    attrs = _attrs_asm(n.attrs, skip={"name", "param_count"})
    lines.append(f"{pad}prog.{label} @{n.attrs['name']}({params}){attrs} {{")
    for stmt in n.args[pc:]:
      _render_stmt(stmt, indent + 1, lines)
    lines.append(f"{pad}}}")
  else:
    lines.append(f"{pad}// expected prog.module/prog.proc, got prog.{n.op.value}")


def _render_stmt(n: PNode, indent: int, lines: list[str]) -> None:
  pad = "  " * indent
  if n.op == POps.BUFFER:
    name = n.attrs["name"]
    if n.attrs.get("address_space") == "constant" and "values" in n.attrs:
      rhs = f" = dense<{_value_asm(n.attrs['values'])}>"
    elif "alias_of" in n.attrs:
      rhs = f" = prog.alias %{n.attrs['alias_of']} offset {n.attrs['alias_offset']}"
    elif "workspace_offset" in n.attrs:
      rhs = f" = prog.workspace offset {n.attrs['workspace_offset']}"
    else:
      rhs = ""
    lines.append(f"{pad}%{name} = prog.buffer : {_memref_asm(n)}{rhs}")
  elif n.op == POps.BLOCK:
    for stmt in n.args:
      _render_stmt(stmt, indent, lines)
  elif n.op == POps.FOR:
    rng = n.args[0]
    var = rng.attrs["name"]
    start, stop, step = (_render_scalar(a) for a in rng.args)
    lines.append(f"{pad}prog.for %{var} = {start} to {stop} step {step} {{kind={rng.attrs['kind'].value}}} {{")
    for stmt in n.args[1:]:
      _render_stmt(stmt, indent + 1, lines)
    lines.append(f"{pad}}}")
  elif n.op == POps.STORE:
    lines.append(f"{pad}prog.store {_render_scalar(n.args[1])}, {_render_view(n.args[0])} : {n.dtype.name}")
  elif n.op == POps.ASSIGN:
    lines.append(f"{pad}%{n.attrs['target']} = prog.assign {_render_scalar(n.args[0])} : {n.dtype.name}")
  elif n.op == POps.CALL:
    args = ", ".join(_render_call_arg(a) for a in n.args)
    rets = n.attrs.get("returns", ())
    ret_prefix = f"{', '.join('%' + r for r in rets)} = " if rets else ""
    lines.append(f"{pad}{ret_prefix}prog.call @{n.attrs['callee']}({args}){_attrs_asm(n.attrs, skip={'callee', 'returns'})}")
  elif n.op == POps.LAUNCH:
    gd = int(n.attrs["grid_dims"])
    bd = int(n.attrs["block_dims"])
    grid = ", ".join(_render_scalar(a) for a in n.args[:gd])
    block = ", ".join(_render_scalar(a) for a in n.args[gd : gd + bd])
    args = ", ".join(_render_call_arg(a) for a in n.args[gd + bd :])
    lines.append(f"{pad}prog.launch @{n.attrs['kernel']} grid({grid}) block({block}) ({args})")
  elif n.op == POps.BARRIER:
    lines.append(f"{pad}prog.barrier {n.attrs['kind']}")
  else:
    lines.append(f"{pad}// stmt {n.op.value}: {_render_scalar(n) if n.op in SCALAR_OPS else _attrs_asm(n.attrs)}")


def _render_call_arg(n: PNode) -> str:
  if n.op == POps.BUFFER:
    return f"%{n.attrs['name']}"
  if n.op == POps.VIEW:
    return _render_view(n)
  return _render_scalar(n)


def _render_view(v: PNode) -> str:
  if v.op != POps.VIEW:
    return f"<not-view:{v.op.value}>"
  idx = ", ".join(_render_scalar(a) for a in v.args)
  return f"%{v.attrs['buffer']}[{idx}]"


def _render_scalar(n: PNode) -> str:
  if n.op == POps.CONST_INT:
    return str(n.attrs["value"])
  if n.op == POps.CONST_FLOAT:
    return f"{n.attrs['value']:g}"
  if n.op == POps.VAR:
    return f"%{n.attrs['name']}"
  if n.op == POps.VIEW:
    return _render_view(n)
  if n.op == POps.LOAD:
    return f"prog.load {_render_view(n.args[0])}"
  if n.op in (POps.ADD, POps.SUB, POps.MUL, POps.DIV, POps.MOD):
    return f"prog.{n.op.value}({_render_scalar(n.args[0])}, {_render_scalar(n.args[1])})"
  if n.op == POps.NEG:
    return f"prog.neg({_render_scalar(n.args[0])})"
  if n.op in UNARY_FN_OPS:
    return f"prog.{n.op.value}({_render_scalar(n.args[0])})"
  if n.op in BINARY_FN_OPS:
    return f"prog.{n.op.value}({_render_scalar(n.args[0])}, {_render_scalar(n.args[1])})"
  return f"<prog.{n.op.value}>"


# ---------------------------------------------------------------------------
# DAG serialization.
# ---------------------------------------------------------------------------


def _pnode_topo(root: PNode) -> list[PNode]:
  seen: set[int] = set()
  out: list[PNode] = []

  def visit(n: PNode) -> None:
    if id(n) in seen:
      return
    seen.add(id(n))
    for arg in n.args:
      visit(arg)
    out.append(n)

  visit(root)
  return out


def expr_graph(obj: Function | Expr | Iterable[Expr]) -> dict[str, Any]:
  outs = tuple(obj.outputs) if isinstance(obj, Function) else ((obj,) if isinstance(obj, Expr) else tuple(obj))
  nodes = topo(outs)
  loc = {e.id: f"e{i}" for i, e in enumerate(nodes)}
  graph_nodes = []
  edges = []
  for i, e in enumerate(nodes):
    op = Ops(e.op)
    label = f"{op.value.upper()}\n%{i}\n{e.type.dtype.name}{e.shape}"
    if e.name:
      label += f"\n{e.name}"
    if op == Ops.CONST and e.value is not None and e.value.size <= 4:
      label += "\n" + np.array2string(e.value, separator=", ")
    graph_nodes.append({"id": loc[e.id], "label": label, "color": _EXPR_COLORS.get(op, "#ffffff")})
    for pos, arg in enumerate(e.args):
      edges.append({"from": loc[arg.id], "to": loc[e.id], "label": str(pos)})
  return {"nodes": graph_nodes, "edges": edges, "outputs": [loc[e.id] for e in outs]}


def program_graph(root: PNode) -> dict[str, Any]:
  nodes = _pnode_topo(root)
  loc = {id(n): f"p{i}" for i, n in enumerate(nodes)}
  graph_nodes = []
  edges = []
  for i, n in enumerate(nodes):
    label = f"{n.op.value.upper()}\n%{i}\n{n.dtype.name}"
    if "name" in n.attrs:
      label += f"\n{n.attrs['name']}"
    elif "target" in n.attrs:
      label += f"\n{n.attrs['target']}"
    elif "callee" in n.attrs:
      label += f"\n{n.attrs['callee']}"
    graph_nodes.append({"id": loc[id(n)], "label": label, "color": _PNODE_COLORS.get(n.op, "#ffffff")})
    for pos, arg in enumerate(n.args):
      edges.append({"from": loc[id(arg)], "to": loc[id(n)], "label": str(pos)})
  return {"nodes": graph_nodes, "edges": edges, "outputs": [loc[id(root)]]}


__all__ = ["expr_graph", "program_graph", "render_expr_assembly", "render_program_assembly", "type_asm"]
