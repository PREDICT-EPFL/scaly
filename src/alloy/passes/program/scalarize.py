"""Expand small procedures into shared scalar values before buffer fusion and workspace packing."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from ...ir import program as p
from ...ir.program import ProgramNode, ProgramOp
from ..arith import CONSTANTS, constant, fold_program


AUTO_SCALAR_OPS_PER_PROC = 4096
AUTO_SCALAR_GROWTH_PER_PROGRAM = 16_384
AUTO_EXPANSION_WORK_PER_PROC = 65_536
MAX_SCALAR_DEPTH = 32


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
    stack = [(node, False)]
    while stack:
      n, ready = stack.pop()
      if n in memo:
        continue
      if n.op in CONSTANTS:
        memo[n] = n
      elif n.op == ProgramOp.VAR:
        memo[n] = self.variables[n.attrs["name"]]
      elif n.op == ProgramOp.LOAD:
        ptr = self.pointer(n.args[0])
        value = ptr.values[ptr.offset]
        assert value is not None, "scalarization read an uninitialized buffer element"
        memo[n] = value
      elif ready:
        memo[n] = fold_program(ProgramNode(n.op, tuple(memo[a] for a in n.args), dtype=n.dtype))
      else:
        stack.append((n, True))
        stack.extend((a, False) for a in reversed(n.args) if a not in memo)
    return memo[node]

  def run(self, body: tuple[ProgramNode, ...]) -> None:
    for stmt in body:
      if stmt.op == ProgramOp.BUFFER:
        if "alias_of" in stmt.attrs:
          src = self.buffers[stmt.attrs["alias_of"]]
          ptr = _Pointer(src.values, src.offset + stmt.attrs["alias_offset"])
        else:
          values = stmt.attrs.get("values")
          ptr = _Pointer([constant(v, stmt.dtype) for v in values] if values is not None else [None] * math.prod(stmt.attrs["shape"]))
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
    if node.op not in {*CONSTANTS, ProgramOp.VIEW} and (uses[node] > 1 or d >= MAX_SCALAR_DEPTH):
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

  def work(stmt: ProgramNode) -> int | None:
    total = 0
    stack = [(stmt, 1)]
    while stack:
      node, factor = stack.pop()
      if node.op == ProgramOp.CALL:
        callee_work = expansion_work.get(node.attrs["callee"])
        if callee_work is None:
          return None
        total += factor * callee_work
      elif node.op == ProgramOp.FOR:
        rng = node.args[0]
        if any(a.op != ProgramOp.CONST_INT for a in rng.args):
          return None
        start, stop, step = (a.attrs["value"] for a in rng.args)
        if step <= 0:
          return None
        stack.extend((s, factor * len(range(start, stop, step))) for s in node.args[1:])
      elif node.op == ProgramOp.BUFFER:
        total += factor * math.prod(node.attrs["shape"])
      elif node.op == ProgramOp.STORE:
        total += factor
      else:
        return None
    return total

  def scalar_cost(proc: ProgramNode) -> tuple[int, int]:
    seen: set[ProgramNode] = set()
    stack = list(proc.args[proc.attrs["param_count"] :])
    while stack:
      node = stack.pop()
      if node in seen:
        continue
      seen.add(node)
      stack.extend(node.args)
    ops = sum(node.op in p.SCALAR_OPS - {*CONSTANTS, ProgramOp.VAR, ProgramOp.LOAD} for node in seen)
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
    if proc.attrs["scalarize"] == "callee":  # inlined into expanding callers, never a root
      expansion_work[name] = total_work
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
