"""Expand small procedures into shared scalar values before buffer fusion and workspace packing."""

from __future__ import annotations

import math
import operator
from collections import Counter
from dataclasses import dataclass

from ...ir import program as p
from ...ir.program import ProgramNode, ProgramOp
from ...ir.types import DType


_CONSTANTS = {ProgramOp.CONST_INT, ProgramOp.CONST_FLOAT}
_ARITHMETIC = {ProgramOp.ADD: operator.add, ProgramOp.SUB: operator.sub, ProgramOp.MUL: operator.mul, ProgramOp.DIV: operator.truediv}
_MATH = {op: getattr(math, op.value) for op in (*p.UNARY_FN_OPS, ProgramOp.POW, ProgramOp.ATAN2) if op not in {ProgramOp.ABS}}
_MATH[ProgramOp.ABS] = abs

AUTO_SCALAR_OPS_PER_PROC = 4096
AUTO_SCALAR_GROWTH_PER_PROGRAM = 16_384
AUTO_EXPANSION_WORK_PER_PROC = 65_536


def _constant(value: int | float, dtype: DType) -> ProgramNode:
  return ProgramNode(
    ProgramOp.CONST_INT if dtype.is_integer else ProgramOp.CONST_FLOAT, attrs={"value": int(value) if dtype.is_integer else float(value)}, dtype=dtype
  )


def _fold(op: ProgramOp, args: tuple[ProgramNode, ...], dtype: DType) -> ProgramNode:
  node = ProgramNode(op, args, dtype=dtype)
  if all(a.op in _CONSTANTS for a in args):
    vals = [a.attrs["value"] for a in args]
    try:
      if op in {ProgramOp.DIV, ProgramOp.MOD} and dtype.is_integer:
        x, y = vals
        q = (abs(x) // abs(y)) * (-1 if (x < 0) != (y < 0) else 1)
        value = q if op == ProgramOp.DIV else x - q * y
      elif op == ProgramOp.NEG:
        value = -vals[0]
      elif op in _ARITHMETIC:
        value = _ARITHMETIC[op](*vals)
      elif op in _MATH:
        value = _MATH[op](*vals)
      elif op in {ProgramOp.MINIMUM, ProgramOp.MAXIMUM} and all(math.isfinite(v) for v in vals):
        value = (min if op == ProgramOp.MINIMUM else max)(vals)
      else:
        return node
      return _constant(value, dtype)
    except (ValueError, OverflowError, ZeroDivisionError):
      return node

  def is_const(a: ProgramNode, value: float) -> bool:
    return a.op in _CONSTANTS and a.attrs["value"] == value

  x = args[0]
  if op == ProgramOp.NEG:
    return x.args[0] if x.op == ProgramOp.NEG else node
  if len(args) != 2:
    return node
  y = args[1]
  if op == ProgramOp.ADD:
    if is_const(x, 0):
      return y
    if is_const(y, 0):
      return x
    if y.op == ProgramOp.NEG:
      return _fold(ProgramOp.SUB, (x, y.args[0]), dtype)
    if x.op == ProgramOp.NEG:
      return _fold(ProgramOp.SUB, (y, x.args[0]), dtype)
  elif op == ProgramOp.SUB:
    if is_const(y, 0):
      return x
    if is_const(x, 0):
      return _fold(ProgramOp.NEG, (y,), dtype)
    if y.op == ProgramOp.NEG:
      return _fold(ProgramOp.ADD, (x, y.args[0]), dtype)
  elif op == ProgramOp.MUL:
    if is_const(x, 0) or is_const(y, 0):
      return _constant(0, dtype)
    if is_const(x, 1):
      return y
    if is_const(y, 1):
      return x
    if is_const(x, -1):
      return _fold(ProgramOp.NEG, (y,), dtype)
    if is_const(y, -1):
      return _fold(ProgramOp.NEG, (x,), dtype)
    if x.op == y.op == ProgramOp.NEG:
      return _fold(op, (x.args[0], y.args[0]), dtype)
  elif op == ProgramOp.DIV:
    if is_const(x, 0):
      return x
    if is_const(y, 1):
      return x
  elif op == ProgramOp.POW:
    if is_const(y, 0) or is_const(x, 1):
      return _constant(1, dtype)
    if is_const(y, 1):
      return x
    if is_const(y, 2):
      return _fold(ProgramOp.MUL, (x, x), dtype)
  return node


@dataclass(slots=True)
class _Pointer:
  values: list[ProgramNode | None]
  offset: int = 0


class _Frame:
  def __init__(self, proc: ProgramNode, pointers: list[_Pointer], procs: dict[str, ProgramNode]) -> None:
    self.procs = procs
    self.buffers = {param.attrs["name"]: ptr for param, ptr in zip(proc.args[: proc.attrs["param_count"]], pointers, strict=True)}
    self.variables: dict[str, ProgramNode] = {}

  def pointer(self, node: ProgramNode) -> _Pointer:
    if node.op == ProgramOp.BUFFER:
      return self.buffers[node.attrs["name"]]
    ptr = self.buffers[node.attrs["buffer"]]
    offset = self.scalar(node.args[0], {}).attrs["value"] if node.args else 0
    return _Pointer(ptr.values, ptr.offset + offset)

  def scalar(self, node: ProgramNode, memo: dict[ProgramNode, ProgramNode]) -> ProgramNode:
    if node in memo:
      return memo[node]
    if node.op in _CONSTANTS:
      result = node
    elif node.op == ProgramOp.VAR:
      result = self.variables[node.attrs["name"]]
    elif node.op == ProgramOp.LOAD:
      ptr = self.pointer(node.args[0])
      result = ptr.values[ptr.offset]
      assert result is not None, "scalarization read an uninitialized buffer element"
    else:
      result = _fold(node.op, tuple(self.scalar(a, memo) for a in node.args), node.dtype)
    memo[node] = result
    return result

  def run(self, body: tuple[ProgramNode, ...]) -> None:
    for stmt in body:
      if stmt.op == ProgramOp.BUFFER:
        if "alias_of" in stmt.attrs:
          src = self.buffers[stmt.attrs["alias_of"]]
          ptr = _Pointer(src.values, src.offset + stmt.attrs["alias_offset"])
        else:
          values = stmt.attrs.get("values")
          ptr = _Pointer([_constant(v, stmt.dtype) for v in values] if values is not None else [None] * math.prod(stmt.attrs["shape"]))
        self.buffers[stmt.attrs["name"]] = ptr
      elif stmt.op == ProgramOp.STORE:
        ptr = self.pointer(stmt.args[0])
        ptr.values[ptr.offset] = self.scalar(stmt.args[1], {})
      elif stmt.op == ProgramOp.FOR:
        rng = stmt.args[0]
        start, stop, step = (self.scalar(a, {}).attrs["value"] for a in rng.args)
        for i in range(start, stop, step):
          self.variables[rng.attrs["name"]] = p.const_int(i)
          self.run(stmt.args[1:])
      elif stmt.op == ProgramOp.CALL:
        callee = self.procs[stmt.attrs["callee"]]
        frame = _Frame(callee, [self.pointer(a) for a in stmt.args], self.procs)
        frame.run(callee.args[callee.attrs["param_count"] :])
      else:
        raise NotImplementedError(f"scalarization does not support {stmt.op}")


def _schedule(outputs: list[tuple[ProgramNode, ProgramNode]], reserved: set[str]) -> list[ProgramNode]:
  order: list[ProgramNode] = []
  seen: set[ProgramNode] = set()
  pending = [(value, False) for _, value in reversed(outputs)]
  while pending:
    node, ready = pending.pop()
    if node in seen:
      continue
    if not ready:
      pending.append((node, True))
      pending.extend((a, False) for a in reversed(node.args))
      continue
    seen.add(node)
    order.append(node)
  uses = Counter(a for n in order for a in n.args)
  uses.update(value for _, value in outputs)
  values: dict[ProgramNode, ProgramNode] = {}
  depth: dict[ProgramNode, int] = {}
  body: list[ProgramNode] = []
  serial = 0
  for node in order:
    args = tuple(values[a] for a in node.args)
    value = ProgramNode(node.op, args, node.attrs, node.dtype)
    d = 1 + max((depth[a] for a in node.args), default=0)
    if node.op not in {*_CONSTANTS, ProgramOp.VIEW} and (uses[node] > 1 or d >= 32):
      while (name := f"v{serial}") in reserved:
        serial += 1
      serial += 1
      body.append(p.assign(name, value, declare=True))
      value, d = p.var(name, node.dtype), 0
    values[node], depth[node] = value, d
  body.extend(p.store(target, values[value]) for target, value in outputs)
  return body


def _scalarize_proc(proc: ProgramNode, procs: dict[str, ProgramNode]) -> ProgramNode:
  params = proc.args[: proc.attrs["param_count"]]
  n_in = proc.attrs["input_count"]
  pointers = [
    _Pointer([p.load(p.view(param, [p.const_int(i)])) if k < n_in else None for i in range(math.prod(param.attrs["shape"]))])
    for k, param in enumerate(params)
  ]
  frame = _Frame(proc, pointers, procs)
  frame.run(proc.args[len(params) :])
  outputs: list[tuple[ProgramNode, ProgramNode]] = []
  for param, ptr in zip(params[n_in:], pointers[n_in:], strict=True):
    for i, value in enumerate(ptr.values):
      assert value is not None, "scalarization left an output element uninitialized"
      outputs.append((p.view(param, [p.const_int(i)]), value))
  body = _schedule(outputs, {param.attrs["name"] for param in params})
  return ProgramNode(proc.op, (*params, *body), {**proc.attrs, "scalarized": True}, proc.dtype)


def scalarize_program(prog: ProgramNode) -> ProgramNode:
  """Expand selected procedures, bounded under auto and only through eligible pure callees."""
  procs = {pr.attrs["name"]: pr for pr in prog.args[: prog.attrs["proc_count"]]}
  expansion_work: dict[str, int | None] = {}

  def work(node: ProgramNode) -> int | None:
    if node.op == ProgramOp.CALL:
      return expansion_work.get(node.attrs["callee"])
    if node.op == ProgramOp.FOR:
      rng = node.args[0]
      if any(a.op != ProgramOp.CONST_INT for a in rng.args):
        return None
      start, stop, step = (a.attrs["value"] for a in rng.args)
      total = [work(s) for s in node.args[1:]]
      return len(range(start, stop, step)) * sum(c for c in total if c is not None) if all(c is not None for c in total) and step > 0 else None
    if node.op == ProgramOp.BUFFER:
      return math.prod(node.attrs["shape"])
    return 1 if node.op == ProgramOp.STORE else None

  def scalar_cost(proc: ProgramNode) -> tuple[int, int]:
    seen: set[ProgramNode] = set()
    stack = list(proc.args[proc.attrs["param_count"] :])
    while stack:
      node = stack.pop()
      if node in seen:
        continue
      seen.add(node)
      stack.extend(node.args)
    ops = sum(node.op in p.SCALAR_OPS - {*_CONSTANTS, ProgramOp.VAR, ProgramOp.LOAD} for node in seen)
    statements = sum(node.op in {ProgramOp.ASSIGN, ProgramOp.STORE} for node in seen)
    return ops, ops + statements

  scalar_growth = 0
  replacements: dict[str, ProgramNode] = {}
  for name, proc in procs.items():
    body_work = [work(stmt) for stmt in proc.args[proc.attrs["param_count"] :]]
    param_work = sum(math.prod(param.attrs["shape"]) for param in proc.args[: proc.attrs["param_count"]])
    total_work = param_work + sum(value for value in body_work if value is not None)
    eligible = proc.attrs.get("scalarize") and all(value is not None for value in body_work)
    if eligible and proc.attrs["lowering"] == "auto" and total_work > AUTO_EXPANSION_WORK_PER_PROC:
      eligible = False
    if not eligible:
      expansion_work[name] = None
      continue
    candidate = _scalarize_proc(proc, procs)
    ops, growth = scalar_cost(candidate)
    if proc.attrs["lowering"] == "auto" and (ops > AUTO_SCALAR_OPS_PER_PROC or scalar_growth + growth > AUTO_SCALAR_GROWTH_PER_PROGRAM):
      expansion_work[name] = None
      continue
    replacements[name] = candidate
    expansion_work[name] = total_work
    scalar_growth += growth
  args = tuple(replacements.get(pr.attrs.get("name"), pr) for pr in prog.args)
  return ProgramNode(prog.op, args, prog.attrs, prog.dtype)
