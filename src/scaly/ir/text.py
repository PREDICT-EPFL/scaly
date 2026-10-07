"""Stable textual forms of the two IR dialects: an assembly listing for each, plus the program
debug dump. The expression dialect's debug dump — ``format_expr`` — stays in ``ir/expr.py``,
where ``Expr.debug`` can reach it without this module's imports.

This is deliberately a printer, not a parser.  The goal is an MLIR/LLVM-like
stable textual form that can be diffed, copied out of the visualizer, and used
in tests/debugging without coupling it to C syntax. Presentation — graph JSON, colors,
labels — is ``viz/graph.py``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Set
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .expr import Expr, ExprOp, topo
from .program import BINARY_FN_OPS, SCALAR_OPS, UNARY_FN_OPS, ProgramNode, ProgramOp
from .types import TensorType

if TYPE_CHECKING:
  from ..function.concrete import ConcreteFunction
  from ..function.model import Function


def type_asm(t: TensorType) -> str:
  shape = "x".join(str(d) for d in t.shape)
  dims = f"{shape}x" if shape else ""
  diff = " diff" if t.diff else ""
  return f"tensor<{dims}{t.dtype.name}{diff}>"


def _memref_asm(n: ProgramNode) -> str:
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
  if hasattr(v, "name") and v.__class__.__name__ == "ConcreteFunction":
    return "@" + v.name
  return str(v)


def _attrs_asm(attrs: Mapping[str, Any], *, skip: Set[str] = frozenset()) -> str:
  items = [(k, v) for k, v in sorted(attrs.items()) if k not in skip]
  if not items:
    return ""
  return " {" + ", ".join(f"{k}={_value_asm(v)}" for k, v in items) + "}"


# ---------------------------------------------------------------------------
# Expression dialect assembly.
# ---------------------------------------------------------------------------


def render_expr_assembly(obj: Function | ConcreteFunction | Expr | Iterable[Expr], *, name: str | None = None) -> str:
  """Render the expression ``Expr`` dialect as a compact SSA assembly listing.

  When given a ``ConcreteFunction``, the listing is a ``expr.module`` containing every transitive expression
  callee body before the requested function. This mirrors what C rendering eventually needs.
  """
  if callable(instantiate := getattr(obj, "instantiate", None)):
    obj = instantiate()
  # The ConcreteFunction case is duck-typed on the surface ``_render_function_module`` actually uses:
  # ``ir`` is below ``function`` in the import-layer order, so it cannot import ``ConcreteFunction`` at runtime.
  # It is tested first so an object that is both a ConcreteFunction and iterable still renders as an
  # ``expr.module`` rather than a bare region.
  if all(hasattr(obj, attr) for attr in ("outputs", "name", "input_names", "output_names")):
    return _render_function_module(cast("ConcreteFunction", obj))
  if isinstance(obj, Expr):
    return _render_expr_region((obj,), name=name)
  if isinstance(obj, Iterable):
    return _render_expr_region(tuple(cast("Iterable[Expr]", obj)), name=name)
  raise TypeError(f"render_expr_assembly expects a ConcreteFunction, an Expr, or an iterable of Expr, got {type(obj).__name__}")


def _expr_callees(fun: ConcreteFunction) -> list[ConcreteFunction]:
  seen: set[int] = set()
  ordered: list[ConcreteFunction] = []

  def visit(fn: ConcreteFunction) -> None:
    if id(fn) in seen:
      return
    seen.add(id(fn))
    for node in topo(fn.outputs):
      if node.op in {ExprOp.CALL, ExprOp.VMAP}:
        callee = node.attrs.get("callee")
        if callee is not None:
          visit(callee)
    ordered.append(fn)

  visit(fun)
  return ordered


def _render_function_module(fun: ConcreteFunction) -> str:
  lines = ["expr.module {"]
  for fn in _expr_callees(fun):
    body = _render_function_expr_assembly(fn).splitlines()
    lines += ["  " + line for line in body]
  lines.append("}")
  return "\n".join(lines)


def _render_function_expr_assembly(fun: ConcreteFunction) -> str:
  ins = ", ".join(f"%{n}: {type_asm(e.type)}" for n, e in zip(fun.input_names, fun.inputs, strict=True))
  outs = ", ".join(f"%{n}: {type_asm(e.type)}" for n, e in zip(fun.output_names, fun.outputs, strict=True))
  lines = [f"expr.func @{fun.name}({ins}) -> ({outs}) {{"]
  body = _render_expr_region(fun.outputs).splitlines()
  lines += ["  " + line for line in body]
  lines.append("}")
  return "\n".join(lines)


def _render_expr_region(outputs: Iterable[Expr], *, name: str | None = None) -> str:
  outs = tuple(outputs)
  nodes = topo(outs)
  loc = {e.id: i for i, e in enumerate(nodes)}
  lines: list[str] = [f"expr.region @{name} {{"] if name else []
  pad = "  " if name else ""
  for i, e in enumerate(nodes):
    args = ", ".join(f"%{loc[a.id]}" for a in e.args)
    op = ExprOp(e.op)
    attrs = dict(e.attrs)
    if op == ExprOp.INPUT:
      attrs = {**attrs, "name": e.name, "lowering": e.lowering}
    elif op == ExprOp.CONST:
      attrs = {**attrs, "value": e.value, "lowering": e.lowering}
    elif op in {ExprOp.CALL, ExprOp.VMAP}:
      callee = attrs.get("callee")
      attrs = {**attrs, "callee": getattr(callee, "name", callee)}
    text_args = f"({args})" if args else ""
    lines.append(f"{pad}%{i} = expr.{op.value}{text_args}{_attrs_asm(attrs)} : {type_asm(e.type)}")
  lines.append(f"{pad}expr.return " + ", ".join(f"%{loc[e.id]}" for e in outs))
  if name:
    lines.append("}")
  return "\n".join(lines)


# ---------------------------------------------------------------------------
# Program dialect assembly.
# ---------------------------------------------------------------------------


def render_program_assembly(root: ProgramNode) -> str:
  """Render Program IR as structured assembly with explicit control-flow constructs."""
  lines: list[str] = []
  _render_program_node(root, 0, lines)
  return "\n".join(lines)


def _render_program_node(n: ProgramNode, indent: int, lines: list[str]) -> None:
  pad = "  " * indent
  if n.op == ProgramOp.PROGRAM:
    lines.append(f"{pad}prog.module{_attrs_asm(n.attrs, skip={'proc_count'})} {{")
    for sub in n.args:
      _render_program_node(sub, indent + 1, lines)
    lines.append(f"{pad}}}")
  elif n.op == ProgramOp.PROC:
    pc = int(n.attrs["param_count"])
    params = ", ".join(f"%{p.attrs['name']}: {_memref_asm(p)}" for p in n.args[:pc])
    attrs = _attrs_asm(n.attrs, skip={"name", "param_count"})
    lines.append(f"{pad}prog.proc @{n.attrs['name']}({params}){attrs} {{")
    for stmt in n.args[pc:]:
      _render_stmt(stmt, indent + 1, lines)
    lines.append(f"{pad}}}")
  else:
    lines.append(f"{pad}// expected prog.module/prog.proc, got prog.{n.op.value}")


def _render_stmt(n: ProgramNode, indent: int, lines: list[str]) -> None:
  pad = "  " * indent
  if n.op == ProgramOp.BUFFER:
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
  elif n.op == ProgramOp.BLOCK:
    for stmt in n.args:
      _render_stmt(stmt, indent, lines)
  elif n.op == ProgramOp.FOR:
    rng = n.args[0]
    var = rng.attrs["name"]
    start, stop, step = (_render_scalar(a) for a in rng.args)
    lines.append(f"{pad}prog.for %{var} = {start} to {stop} step {step} {{kind={rng.attrs['kind'].value}}} {{")
    for stmt in n.args[1:]:
      _render_stmt(stmt, indent + 1, lines)
    lines.append(f"{pad}}}")
  elif n.op == ProgramOp.STORE:
    lines.append(f"{pad}prog.store {_render_scalar(n.args[1])}, {_render_view(n.args[0])} : {n.dtype.name}")
  elif n.op == ProgramOp.STORE_PAIR:
    lines.append(f"{pad}prog.store_pair {_render_scalar(n.args[1])}, {_render_scalar(n.args[2])}, {_render_view(n.args[0])} : {n.dtype.name}")
  elif n.op == ProgramOp.ASSIGN:
    lines.append(f"{pad}%{n.attrs['target']} = prog.assign {_render_scalar(n.args[0])} : {n.dtype.name}{_attrs_asm(n.attrs, skip={'target'})}")
  elif n.op == ProgramOp.CALL:
    args = ", ".join(_render_call_arg(a) for a in n.args)
    rets = n.attrs.get("returns", ())
    ret_prefix = f"{', '.join('%' + r for r in rets)} = " if rets else ""
    lines.append(f"{pad}{ret_prefix}prog.call @{n.attrs['callee']}({args}){_attrs_asm(n.attrs, skip={'callee', 'returns'})}")
  else:
    lines.append(f"{pad}// stmt {n.op.value}: {_render_scalar(n) if n.op in SCALAR_OPS else _attrs_asm(n.attrs)}")


def _render_call_arg(n: ProgramNode) -> str:
  if n.op == ProgramOp.BUFFER:
    return f"%{n.attrs['name']}"
  if n.op == ProgramOp.VIEW:
    return _render_view(n)
  return _render_scalar(n)


def _render_view(v: ProgramNode) -> str:
  if v.op != ProgramOp.VIEW:
    return f"<not-view:{v.op.value}>"
  idx = ", ".join(_render_scalar(a) for a in v.args)
  return f"%{v.attrs['buffer']}[{idx}]"


def _render_scalar(n: ProgramNode) -> str:
  if n.op == ProgramOp.CONST_INT:
    return str(n.attrs["value"])
  if n.op == ProgramOp.CONST_FLOAT:
    return f"{n.attrs['value']:g}"
  if n.op == ProgramOp.VAR:
    return f"%{n.attrs['name']}"
  if n.op == ProgramOp.VIEW:
    return _render_view(n)
  if n.op == ProgramOp.LOAD:
    return f"prog.load {_render_view(n.args[0])}"
  if n.op in (ProgramOp.ADD, ProgramOp.SUB, ProgramOp.MUL, ProgramOp.DIV, ProgramOp.MOD):
    return f"prog.{n.op.value}({_render_scalar(n.args[0])}, {_render_scalar(n.args[1])})"
  if n.op == ProgramOp.NEG:
    return f"prog.neg({_render_scalar(n.args[0])})"
  if n.op in UNARY_FN_OPS:
    return f"prog.{n.op.value}({_render_scalar(n.args[0])})"
  if n.op in BINARY_FN_OPS:
    return f"prog.{n.op.value}({_render_scalar(n.args[0])}, {_render_scalar(n.args[1])})"
  return f"<prog.{n.op.value}>"


# ---------------------------------------------------------------------------
# Program dialect pretty printer: backend-neutral debug dump, decoupled from C syntax.
# ---------------------------------------------------------------------------


def format_program(root: ProgramNode, indent: int = 0) -> str:
  lines: list[str] = []
  _format_node(root, indent, lines)
  return "\n".join(lines)


def _format_node(n: ProgramNode, indent: int, lines: list[str]) -> None:
  pad = "  " * indent
  if n.op == ProgramOp.PROGRAM:
    lines.append(f"{pad}program")
    for sub in n.args:
      _format_node(sub, indent + 1, lines)
  elif n.op == ProgramOp.PROC:
    params = ", ".join(f"{p.attrs['name']}:{p.dtype.name}{p.attrs['shape']}" for p in n.args[: n.attrs["param_count"]])
    lines.append(f"{pad}proc {n.attrs['name']}({params}):")
    for sub in n.args[n.attrs["param_count"] :]:
      _format_node(sub, indent + 1, lines)
  elif n.op == ProgramOp.BLOCK:
    for sub in n.args:
      _format_node(sub, indent, lines)
  elif n.op == ProgramOp.FOR:
    rng = n.args[0]
    kind = rng.attrs["kind"].value
    start = _format_scalar(rng.args[0])
    stop = _format_scalar(rng.args[1])
    step = _format_scalar(rng.args[2])
    lines.append(f"{pad}for {rng.attrs['name']} in [{start}, {stop}) step {step} kind={kind}:")
    for sub in n.args[1:]:
      _format_node(sub, indent + 1, lines)
  elif n.op == ProgramOp.STORE:
    target = _format_view(n.args[0])
    value = _format_scalar(n.args[1])
    lines.append(f"{pad}{target} <- {value}")
  elif n.op == ProgramOp.STORE_PAIR:
    target = _format_view(n.args[0])
    lines.append(f"{pad}{target} <- pair({_format_scalar(n.args[1])}, {_format_scalar(n.args[2])})")
  elif n.op == ProgramOp.ASSIGN:
    declaration = f"{n.dtype.name} " if n.attrs.get("declare") else ""
    lines.append(f"{pad}{declaration}{n.attrs['target']} = {_format_scalar(n.args[0])}")
  elif n.op == ProgramOp.CALL:
    args = ", ".join(_format_scalar_or_view(a) for a in n.args)
    rets = n.attrs.get("returns", ())
    ret_prefix = f"{', '.join(rets)} = " if rets else ""
    lines.append(f"{pad}{ret_prefix}call {n.attrs['callee']}({args})")
  else:
    # fallback for unknown / scalar at statement scope
    lines.append(f"{pad}{n.op.value}")


def _format_view(v: ProgramNode) -> str:
  if v.op != ProgramOp.VIEW:
    return v.op.value
  idx = ", ".join(_format_scalar(a) for a in v.args)
  return f"{v.attrs['buffer']}[{idx}]"


def _format_scalar(n: ProgramNode) -> str:
  if n.op == ProgramOp.CONST_INT:
    return str(n.attrs["value"])
  if n.op == ProgramOp.CONST_FLOAT:
    return f"{n.attrs['value']:g}"
  if n.op == ProgramOp.VAR:
    return str(n.attrs["name"])
  if n.op == ProgramOp.LOAD:
    return _format_view(n.args[0])
  if n.op in (ProgramOp.ADD, ProgramOp.SUB, ProgramOp.MUL, ProgramOp.DIV, ProgramOp.MOD):
    sym = {ProgramOp.ADD: "+", ProgramOp.SUB: "-", ProgramOp.MUL: "*", ProgramOp.DIV: "/", ProgramOp.MOD: "%"}[n.op]
    return f"({_format_scalar(n.args[0])} {sym} {_format_scalar(n.args[1])})"
  if n.op == ProgramOp.NEG:
    return f"(-{_format_scalar(n.args[0])})"
  if n.op in UNARY_FN_OPS:
    return f"{n.op.value}({_format_scalar(n.args[0])})"
  if n.op in BINARY_FN_OPS:
    return f"{n.op.value}({_format_scalar(n.args[0])}, {_format_scalar(n.args[1])})"
  return f"<{n.op.value}>"


def _format_scalar_or_view(n: ProgramNode) -> str:
  if n.op == ProgramOp.VIEW:
    return _format_view(n)
  return _format_scalar(n)
