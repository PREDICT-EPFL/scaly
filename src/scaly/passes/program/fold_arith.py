"""Arithmetic cleanup of loop bodies: constant-buffer reads become constants, then the shared identities fold."""

from __future__ import annotations

from ...ir import program as p
from ...ir.match import Pattern, rewrite
from ...ir.program import ProgramNode, ProgramOp, RangeKind, walk_program
from ...ir.types import dtypes
from ...utils.names import NameScope
from ..arith import constant, fold_program
from .fold_tiles import _period, _scalar_safe
from ._common import (
  _map_procs,
  _proc_parts,
  _rebuild_proc,
  name_scope,
  prune_dead_buffers,
  rebuild_program,
  substitute_var,
)


def fold_arith(prog: ProgramNode) -> ProgramNode:
  """Fold arithmetic inside every procedure body, including ``FOR`` bodies.

  Runs after fusion, whose index substitution is what exposes ``x * 1``, ``x + 0`` and reads of a
  uniform constant buffer (a broadcast scalar constant) inside loop bodies.
  """
  return _map_procs(prog, _fold_proc)


def _fold_proc(proc: ProgramNode) -> ProgramNode:
  params, body = _proc_parts(proc)
  names = name_scope(proc)
  changed = False
  while True:
    new_body = _fold_body(body, names)
    if all(a is b for a, b in zip(new_body, body, strict=True)):
      break
    body, changed = new_body, True
  rebuilt = _rebuild_proc(proc, params, body) if changed else proc
  return prune_dead_buffers(rebuilt)


def _index_bounds(node: ProgramNode, bounds: dict[str, tuple[int, int]]) -> tuple[int, int] | None:
  if node.dtype != dtypes.int64:
    return None
  if node.op == ProgramOp.CONST_INT:
    value = int(node.attrs["value"])
    return value, value
  if node.op == ProgramOp.VAR:
    return bounds.get(node.attrs["name"])
  if node.op not in (ProgramOp.ADD, ProgramOp.SUB, ProgramOp.MUL):
    return None
  left, right = (_index_bounds(arg, bounds) for arg in node.args)
  if left is None or right is None:
    return None
  if node.op == ProgramOp.ADD:
    result = left[0] + right[0], left[1] + right[1]
  elif node.op == ProgramOp.SUB:
    result = left[0] - right[1], left[1] - right[0]
  else:
    products = [a * b for a in left for b in right]
    result = min(products), max(products)
  return result if -(1 << 63) <= result[0] <= result[1] < (1 << 63) else None


def _index_quotient(node: ProgramNode, divisor: int) -> tuple[ProgramNode, ProgramNode]:
  """Split integer arithmetic as ``divisor * quotient + remainder`` without changing its value."""
  if node.op == ProgramOp.ADD:
    left, right = (_index_quotient(arg, divisor) for arg in node.args)
    return p.add(left[0], right[0]), p.add(left[1], right[1])
  if node.op == ProgramOp.MUL:
    for coeff, value in (node.args, node.args[::-1]):
      if coeff.op == ProgramOp.CONST_INT and coeff.attrs["value"] % divisor == 0:
        return p.mul(p.const_int(coeff.attrs["value"] // divisor), value), p.const_int(0)
  if node.op == ProgramOp.CONST_INT:
    return p.const_int(node.attrs["value"] // divisor), p.const_int(node.attrs["value"] % divisor)
  return p.const_int(0), node


def _fold_index(node: ProgramNode, bounds: dict[str, tuple[int, int]]) -> ProgramNode | None:
  left, right = node.args
  if node.dtype != dtypes.int64 or right.op != ProgramOp.CONST_INT or right.attrs["value"] <= 0:
    return None
  interval = _index_bounds(left, bounds)
  if interval is None or interval[0] < 0:
    return None
  divisor = int(right.attrs["value"])
  quotient, remainder = _index_quotient(left, divisor)
  interval = _index_bounds(remainder, bounds)
  if interval is not None and 0 <= interval[0] <= interval[1] < divisor:
    return quotient if node.op == ProgramOp.DIV else remainder
  return None


def _fold_body(body: list[ProgramNode], names: NameScope) -> list[ProgramNode]:
  declarations = {s.attrs["name"]: s for s in body if s.op == ProgramOp.BUFFER and "values" in s.attrs}
  constants = {name: decl.attrs["values"] for name, decl in declarations.items()}
  uniform = {name for name, decl in declarations.items() if _period(decl) == 1 and _scalar_safe(constants[name][0])}

  def constant_read(n: ProgramNode) -> bool:
    view = n.args[0]
    values = constants.get(view.attrs["buffer"])
    if not values or len(view.args) != 1:
      return False
    index = view.args[0]
    return (index.op == ProgramOp.CONST_INT and 0 <= index.attrs["value"] < len(values) and _scalar_safe(values[index.attrs["value"]])) or view.attrs[
      "buffer"
    ] in uniform

  def read(n: ProgramNode) -> ProgramNode:
    index = n.args[0].args[0]
    values = constants[n.args[0].attrs["buffer"]]
    return constant(values[index.attrs["value"] if index.op == ProgramOp.CONST_INT else 0], n.dtype)

  patterns = [Pattern(ProgramOp.LOAD, constant_read, read), Pattern(None, lambda n: bool(n.args), fold_program)]

  def scoped(stmt: ProgramNode, bounds: dict[str, tuple[int, int]], memo: dict[ProgramNode, ProgramNode]) -> ProgramNode:
    if stmt.op == ProgramOp.FOR:
      rng, *nested = stmt.args
      rng = rewrite(rng, patterns, rebuild=rebuild_program)
      assigned = {node.attrs["target"] for child in nested for node in walk_program(child) if node.op == ProgramOp.ASSIGN}
      inner = {name: interval for name, interval in bounds.items() if name != rng.attrs["name"] and name not in assigned}
      if all(arg.op == ProgramOp.CONST_INT for arg in rng.args):
        start, stop, step = (int(arg.attrs["value"]) for arg in rng.args)
        if start < stop and step > 0 and rng.attrs["name"] not in assigned:
          inner[rng.attrs["name"]] = start, stop - 1
          nodes = [node for child in nested for node in walk_program(child)]
          stores = [node for node in nodes if node.op == ProgramOp.STORE]
          outer_store = any(node.op == ProgramOp.VAR and node.attrs["name"] in bounds for store in stores for node in walk_program(store.args[0]))
          # Keep top-level selectors intact for pack_masks; expose inner copy coordinates.
          if (
            start == 0
            and step == 1
            and not rng.attrs.get("carried")
            and rng.attrs["kind"] != RangeKind.REDUCE
            and outer_store
            and not any(node.op == ProgramOp.FOR for node in nodes)
          ):
            divisors = {
              int(node.args[1].attrs["value"])
              for node in nodes
              if node.op in (ProgramOp.DIV, ProgramOp.MOD)
              and node.dtype == dtypes.int64
              and node.args[1].op == ProgramOp.CONST_INT
              and 1 < node.args[1].attrs["value"] < stop
              and stop % node.args[1].attrs["value"] == 0
              and any(v.op == ProgramOp.VAR and v.attrs["name"] == rng.attrs["name"] for v in walk_program(node.args[0]))
            }
            if divisors:
              divisor = min(divisors)
              block = names.allocate(f"{rng.attrs['name']}_block")
              index = p.add(p.mul(p.var(block), p.const_int(divisor)), p.var(rng.attrs["name"]))
              body = [substitute_var(child, rng.attrs["name"], index) for child in nested]
              inside = p.for_(p.range_(rng.attrs["name"], 0, divisor, kind=rng.attrs["kind"]), body)
              return scoped(p.for_(p.range_(block, 0, stop // divisor, kind=rng.attrs["kind"]), [inside]), bounds, memo)
      inner_memo: dict[ProgramNode, ProgramNode] = {}
      return rebuild_program(stmt, (rng, *(scoped(child, inner, inner_memo) for child in nested)))
    indices = Pattern[ProgramNode](None, lambda n: n.op in (ProgramOp.DIV, ProgramOp.MOD), lambda n: _fold_index(n, bounds) or fold_program(n))
    return rewrite(stmt, [indices, *patterns], rebuild=rebuild_program, memo=memo)

  memo: dict[ProgramNode, ProgramNode] = {}
  return [scoped(stmt, {}, memo) for stmt in body]
