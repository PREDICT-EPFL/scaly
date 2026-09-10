"""Graph JSON for the visualizer: nodes, edges, labels, and per-op colors.

Presentation only. The stable textual forms of both dialects are ``ir/text.py``; this module is
what the recorder and the served UI consume.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ..function import Function
from ..ir.expr import Expr, ExprOp, topo
from ..ir.program import ProgramNode, ProgramOp


_EXPR_COLORS = {
  ExprOp.INPUT: "#c0c0ff",
  ExprOp.CONST: "#e0e0e0",
  ExprOp.CALL: "#00b7c8",
  ExprOp.VMAP: "#f6ccff",
  ExprOp.ADD: "#ffffc0",
  ExprOp.SUB: "#ffffc0",
  ExprOp.MUL: "#ffffc0",
  ExprOp.DIV: "#ffffc0",
  ExprOp.NEG: "#ffffc0",
  ExprOp.SUM: "#ffb0b0",
  ExprOp.MATMUL: "#ffb0b0",
  ExprOp.RESHAPE: "#d8f9e4",
  ExprOp.TRANSPOSE: "#d8f9e4",
  ExprOp.SLICE: "#e5eaff",
  ExprOp.GATHER: "#e5eaff",
  ExprOp.SCATTER: "#e5eaff",
  ExprOp.STACK: "#ffc14d",
  ExprOp.CONCAT: "#ffc14d",
}

_PROGRAM_NODE_COLORS = {
  ProgramOp.PROGRAM: "#c07788",
  ProgramOp.PROC: "#c07788",
  ProgramOp.KERNEL: "#c07788",
  ProgramOp.BUFFER: "#b0bdff",
  ProgramOp.VIEW: "#e5eaff",
  ProgramOp.RANGE: "#c8a0e0",
  ProgramOp.FOR: "#c8a0e0",
  ProgramOp.STORE: "#87ceeb",
  ProgramOp.STORE_PAIR: "#87ceeb",
  ProgramOp.ASSIGN: "#87ceeb",
  ProgramOp.LOAD: "#ffc0c0",
  ProgramOp.CALL: "#00b7c8",
  ProgramOp.LAUNCH: "#00b7c8",
  ProgramOp.CONST_INT: "#e0e0e0",
  ProgramOp.CONST_FLOAT: "#e0e0e0",
  ProgramOp.VAR: "#cef263",
}


def _program_node_topo(root: ProgramNode) -> list[ProgramNode]:
  seen: set[int] = set()
  out: list[ProgramNode] = []

  def visit(n: ProgramNode) -> None:
    if id(n) in seen:
      return
    seen.add(id(n))
    for arg in n.args:
      visit(arg)
    out.append(n)

  visit(root)
  return out


def expr_graph(obj: Function | Expr | Iterable[Expr]) -> dict[str, Any]:
  """An expression graph as JSON-serializable nodes and edges, with labels and per-op colors.

  Presentation for tooling. For text meant to be diffed or asserted on, use
  ``render_expr_assembly``.
  """
  outs = tuple(obj.outputs) if isinstance(obj, Function) else ((obj,) if isinstance(obj, Expr) else tuple(obj))
  nodes = topo(outs)
  loc = {e.id: f"e{i}" for i, e in enumerate(nodes)}
  graph_nodes = []
  edges = []
  for i, e in enumerate(nodes):
    op = ExprOp(e.op)
    label = f"{op.value.upper()}\n%{i}\n{e.type.dtype.name}{e.shape}"
    if e.name:
      label += f"\n{e.name}"
    if op == ExprOp.CONST and e.value is not None and e.value.size <= 4:
      label += "\n" + np.array2string(e.value, separator=", ")
    graph_nodes.append({"id": loc[e.id], "label": label, "color": _EXPR_COLORS.get(op, "#ffffff")})
    for pos, arg in enumerate(e.args):
      edges.append({"from": loc[arg.id], "to": loc[e.id], "label": str(pos)})
  return {"nodes": graph_nodes, "edges": edges, "outputs": [loc[e.id] for e in outs]}


def program_graph(root: ProgramNode) -> dict[str, Any]:
  """A lowered program as JSON-serializable nodes and edges, for tooling."""
  nodes = _program_node_topo(root)
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
    graph_nodes.append({"id": loc[id(n)], "label": label, "color": _PROGRAM_NODE_COLORS.get(n.op, "#ffffff")})
    for pos, arg in enumerate(n.args):
      edges.append({"from": loc[id(arg)], "to": loc[id(n)], "label": str(pos)})
  return {"nodes": graph_nodes, "edges": edges, "outputs": [loc[id(root)]]}


__all__ = ["expr_graph", "program_graph"]
