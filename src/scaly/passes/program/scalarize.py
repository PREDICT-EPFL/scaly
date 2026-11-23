"""Expand small procedures into shared scalar values before buffer fusion and workspace packing."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ...ir import program as p
from ...ir.program import ProgramNode, ProgramOp, walk_program
from ...ir.support import Support
from ...utils.names import NameScope, c_ident
from ..arith import CONSTANTS, constant, fold_program
from ._common import name_scope
from .scheduling import ScalarNameAllocator, schedule_values


AUTO_SCALAR_OPS_PER_PROC = 4096
AUTO_SCALAR_GROWTH_PER_PROGRAM = 16_384
AUTO_EXPANSION_WORK_PER_PROC = 65_536


class _InputValues(dict[int, ProgramNode | None]):
  def __init__(self, buffer: ProgramNode) -> None:
    super().__init__()
    self.buffer = buffer

  def __missing__(self, index: int) -> ProgramNode:
    values = self.buffer.attrs.get("values")
    value = constant(values[index], self.buffer.dtype) if values is not None else p.load(p.view(self.buffer, [p.const_int(index)]))
    self[index] = value
    return value


@dataclass(slots=True)
class _Pointer:
  values: list[ProgramNode | None] | _InputValues
  offset: int = 0
  support: Support = None


class _DataDependent(Exception):
  """An address, range bound or branch selector that does not fold to a constant."""


class _OverBudget(Exception):
  pass


@dataclass(slots=True)
class _Budget:
  """The values a static loop may still compute, and the work it may still do, before expansion gives up.

  A value counts when it is not a constant after unrolling: index arithmetic and masked entries fold
  away and never reach the generated code. Work counts every statement, range trip and allocated
  element, and is charged before it is done.
  """

  values: int
  work: int = AUTO_EXPANSION_WORK_PER_PROC

  def spend(self, value: ProgramNode) -> None:
    self.values -= value.op not in CONSTANTS
    if self.values < 0:
      raise _OverBudget

  def charge(self, work: int) -> None:
    self.work -= work
    if self.work < 0:
      raise _OverBudget


class _Frame:
  def __init__(
    self,
    proc: ProgramNode,
    pointers: list[_Pointer],
    procs: dict[str, ProgramNode],
    folded: dict[tuple[ProgramNode, tuple[ProgramNode, ...]], ProgramNode],
    *,
    strict: bool = False,
    budget: _Budget | None = None,
  ) -> None:
    self.strict = strict
    self.budget = budget
    self.procs = procs
    self.folded = folded
    self.buffers = {
      param.attrs["name"]: _Pointer(ptr.values, ptr.offset, param.attrs.get("support"))
      for param, ptr in zip(proc.args[: proc.attrs["param_count"]], pointers, strict=True)
    }
    self.variables: dict[str, ProgramNode] = {}

  def pointer(self, node: ProgramNode) -> _Pointer:
    if node.op == ProgramOp.BUFFER:
      return self.buffers[node.attrs["name"]]
    ptr = self.buffers[node.attrs["buffer"]]
    index = self.scalar(node.args[0], {})
    if index.op != ProgramOp.CONST_INT:
      raise _DataDependent
    offset = index.attrs["value"]
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
        args = tuple(memo[a] for a in n.args)
        key = n, args
        value = self.folded.get(key)
        if value is None:
          attrs = {**n.attrs, "strict": True} if self.strict and n.dtype.is_floating else n.attrs
          value = self.folded[key] = fold_program(ProgramNode(n.op, args, attrs, dtype=n.dtype))
        memo[n] = value
      else:
        stack.append((n, True))
        stack.extend((a, False) for a in reversed(n.args) if a not in memo)
    return memo[node]

  def run(self, body: tuple[ProgramNode, ...]) -> None:
    for stmt in body:
      if self.budget is not None:
        self.budget.charge(1)
      if stmt.op == ProgramOp.BUFFER:
        if "alias_of" in stmt.attrs:
          src = self.buffers[stmt.attrs["alias_of"]]
          ptr = _Pointer(src.values, src.offset + stmt.attrs["alias_offset"])
        else:
          if self.budget is not None:
            self.budget.charge(math.prod(stmt.attrs["shape"]))
          values = stmt.attrs.get("values")
          ptr = _Pointer([constant(v, stmt.dtype) for v in values] if values is not None else [None] * math.prod(stmt.attrs["shape"]))
        ptr.support = stmt.attrs.get("support")
        self.buffers[stmt.attrs["name"]] = ptr
      elif stmt.op == ProgramOp.STORE:
        ptr = self.pointer(stmt.args[0])
        base = self.buffers[stmt.args[0].attrs["buffer"]]
        value = ptr.values[ptr.offset] = (
          constant(0, stmt.dtype) if base.support is not None and ptr.offset - base.offset not in base.support else self.scalar(stmt.args[1], {})
        )
        if self.budget is not None:
          self.budget.spend(value)
      elif stmt.op == ProgramOp.ASSIGN:
        value = self.variables[stmt.attrs["target"]] = self.scalar(stmt.args[0], {})
        if self.budget is not None:
          self.budget.spend(value)
      elif stmt.op == ProgramOp.FOR:
        rng = stmt.args[0]
        start, stop, step = (self._constant(a) for a in rng.args)
        if self.budget is not None:
          self.budget.charge(len(range(start, stop, step)))
        for i in range(start, stop, step):
          self.variables[rng.attrs["name"]] = p.const_int(i)
          self.run(stmt.args[1:])
      elif stmt.op == ProgramOp.IF:
        self.run(stmt.args[1 if self._constant(stmt.args[0]) else 2].args)
      elif stmt.op == ProgramOp.CALL:
        callee = self.procs[stmt.attrs["callee"]]
        frame = _Frame(callee, [self.pointer(a) for a in stmt.args], self.procs, self.folded, strict=self.strict, budget=self.budget)
        frame.run(callee.args[callee.attrs["param_count"] :])
      else:
        raise NotImplementedError(f"scalarization does not support {stmt.op}")

  def _constant(self, node: ProgramNode) -> int:
    value = self.scalar(node, {})
    if value.op not in CONSTANTS:
      raise _DataDependent
    return value.attrs["value"]


def expand_static_loop(
  params: list[ProgramNode],
  body: list[ProgramNode],
  results: list[tuple[ProgramNode, int]],
  procs: dict[str, ProgramNode],
  names: NameScope,
  *,
  budget: int | None = None,
) -> list[ProgramNode]:
  """Interpret an ordered static loop, preserving floating arithmetic and writing through mutable storage.

  Bounds and branches inside the loop must fold to constants at each step. With ``budget``, the
  values the carried range stores or assigns through every callee that are not constants, and the
  constant results, are counted, and ``_OverBudget`` is raised once they exceed it or once the
  interpretation's work exceeds ``AUTO_EXPANSION_WORK_PER_PROC``.
  """
  proc = p.proc("static_loop", params, body)
  frame = _Frame(proc, [_Pointer(_InputValues(buf)) for buf in params], procs, {}, strict=True)
  loop_budget = None if budget is None else _Budget(budget)
  for stmt in body:
    carried = stmt.op == ProgramOp.FOR and stmt.args[0].attrs.get("carried")
    frame.budget = loop_budget if carried else None
    frame.run((stmt,))
  buffers = {buf.attrs["name"]: buf for buf in (*params, *body) if buf.op == ProgramOp.BUFFER}
  targets, values = [], []
  for buf, size in results:
    if not size:
      continue
    ptr = frame.buffers[buf.attrs["name"]]
    offset = 0
    while "alias_of" in buf.attrs:
      offset += buf.attrs["alias_offset"]
      buf = buffers[buf.attrs["alias_of"]]
    for at in range(size):
      value = ptr.values[ptr.offset + at]
      assert value is not None
      targets.append(p.view(buf, [p.const_int(offset + at)]))
      values.append(value)
      # A constant result is never counted inside the loop, yet its store still reaches the generated code.
      if loop_budget is not None and value.op in CONSTANTS:
        loop_budget.values -= 1
  if loop_budget is not None and loop_budget.values < 0:
    raise _OverBudget
  declarations, values = schedule_values(values, ScalarNameAllocator(names))
  return [
    *(stmt for stmt in body if stmt.op == ProgramOp.BUFFER),
    *declarations,
    *(p.store(target, value) for target, value in zip(targets, values, strict=True)),
  ]


def _scalarize_proc(proc: ProgramNode, procs: dict[str, ProgramNode]) -> ProgramNode:
  params = proc.args[: proc.attrs["param_count"]]
  if any(param.attrs.get("intent") == "inout" for param in params):
    raise NotImplementedError("scalarize refuses inout parameters")
  n_in = proc.attrs["input_count"]
  pointers = [
    _Pointer([p.load(p.view(param, [p.const_int(i)])) if k < n_in else None for i in range(math.prod(param.attrs["shape"]))])
    for k, param in enumerate(params)
  ]
  frame = _Frame(proc, pointers, procs, {})
  frame.run(proc.args[len(params) :])
  outputs: list[tuple[ProgramNode, ProgramNode]] = []
  for param, ptr in zip(params[n_in:], pointers[n_in:], strict=True):
    for i in range(math.prod(param.attrs["shape"])):
      value = ptr.values[i]
      assert value is not None, "scalarization left an output element uninitialized"
      outputs.append((p.view(param, [p.const_int(i)]), value))
  names = ScalarNameAllocator({c_ident(param.attrs["name"]) for param in params})
  declarations, values = schedule_values([value for _, value in outputs], names)
  body = [*declarations, *(p.store(target, value) for (target, _), value in zip(outputs, values, strict=True))]
  return ProgramNode(proc.op, (*params, *body), {**proc.attrs, "scalarized": True}, dtype=proc.dtype)


def _scalar_cost(proc: ProgramNode) -> tuple[int, int]:
  """The scalar operations a procedure emits, and those plus its statements, counting products repeated at their uses."""
  ops = statements = 0
  stack = list(proc.args[proc.attrs["param_count"] :])
  while stack:
    node = stack.pop()
    ops += node.op in p.SCALAR_OPS - {*CONSTANTS, ProgramOp.VAR, ProgramOp.LOAD}
    statements += node.op in {ProgramOp.ASSIGN, ProgramOp.STORE}
    stack.extend(node.args)
  return ops, ops + statements


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
        if rng.attrs.get("carried") or any(a.op != ProgramOp.CONST_INT for a in rng.args):
          return None
        start, stop, step = (a.attrs["value"] for a in rng.args)
        if step <= 0:
          return None
        stack.extend((s, factor * len(range(start, stop, step))) for s in node.args[1:])
      elif node.op == ProgramOp.BUFFER:
        total += factor * math.prod(node.attrs["shape"])
      elif node.op in (ProgramOp.STORE, ProgramOp.ASSIGN):
        total += factor
      else:
        return None
    return total

  scalar_growth = 0
  replacements: dict[str, ProgramNode] = {}
  for name, proc in procs.items():
    body_work = [work(stmt) for stmt in proc.args[proc.attrs["param_count"] :]]
    param_work = sum(math.prod(param.attrs["shape"]) for param in proc.args[: proc.attrs["param_count"]])
    total_work = param_work + sum(value for value in body_work if value is not None)
    mode = proc.attrs.get("scalarize_mode", "disabled")
    eligible = (
      mode != "disabled"
      and all(value is not None for value in body_work)
      and not any(param.attrs.get("intent") == "inout" for param in proc.args[: proc.attrs["param_count"]])
    )
    if eligible and proc.attrs["lowering"] == "auto" and total_work > AUTO_EXPANSION_WORK_PER_PROC:
      eligible = False
    if not eligible:
      expansion_work[name] = None
      continue
    if mode == "inline":
      expansion_work[name] = total_work
      continue
    try:
      candidate = _scalarize_proc(proc, procs)
    except _DataDependent:
      expansion_work[name] = None
      continue
    ops, growth = _scalar_cost(candidate)
    if proc.attrs["lowering"] == "auto" and (ops > AUTO_SCALAR_OPS_PER_PROC or scalar_growth + growth > AUTO_SCALAR_GROWTH_PER_PROGRAM):
      expansion_work[name] = None
      continue
    replacements[name] = candidate
    expansion_work[name] = total_work
    scalar_growth += growth
  args = [replacements.get(pr.attrs.get("name"), pr) for pr in prog.args]
  if args and args[-1].attrs.get("fold_reorders"):
    root, specialized = _fold_reorders(args[-1], {pr.attrs["name"]: pr for pr in args}, set(replacements), name_scope(prog))
    # A specialization whose callee something else still calls is a second copy of its body.
    called = {node.attrs["callee"] for pr in (*args[:-1], root) for node in walk_program(pr) if node.op == ProgramOp.CALL}
    copies = sum(_scalar_cost(pr)[1] for pr in specialized if pr.attrs["reordered_from"] in called)
    if specialized and scalar_growth + copies <= AUTO_SCALAR_GROWTH_PER_PROGRAM:
      args = [*args[:-1], *specialized, root]
      return ProgramNode(prog.op, tuple(args), {**prog.attrs, "proc_count": len(args)}, dtype=prog.dtype)
  return ProgramNode(prog.op, tuple(args), prog.attrs, dtype=prog.dtype)


# An entry point converts each sparse argument from its declared layout to canonical order, and each
# sparse result back, with copy loops around its calls. Once a callee is expanded, its loads and
# stores have constant indices, so a copy can instead change the indices of a specialized callee:
# the loop, its index table and its temporary leave the generated code.


def _references(stmt: ProgramNode, procs: dict[str, ProgramNode]) -> tuple[set[str], set[str]]:
  """The buffers a statement reads and writes, through its loops and calls."""
  reads: set[str] = set()
  writes: set[str] = set()
  stack = [stmt]
  while stack:
    node = stack.pop()
    if node.op in (ProgramOp.STORE, ProgramOp.STORE_PAIR):
      writes.add(node.args[0].attrs["buffer"])
      stack.extend((*node.args[0].args, *node.args[1:]))
    elif node.op == ProgramOp.CALL:
      # An external function's arguments may be read and written.
      callee = procs.get(node.attrs["callee"])
      n_in = callee.attrs["input_count"] if callee is not None else 0
      for i, arg in enumerate(node.args):
        name = _buffer_name(arg)
        (reads if i < n_in else writes).add(name)
        if callee is None:
          reads.add(name)
        stack.extend(arg.args)
    elif node.op == ProgramOp.VIEW:
      reads.add(node.attrs["buffer"])
      stack.extend(node.args)
    elif node.op == ProgramOp.BUFFER:
      continue
    else:
      stack.extend(node.args)
  return reads, writes


def _buffer_name(arg: ProgramNode) -> str:
  return arg.attrs["name"] if arg.op == ProgramOp.BUFFER else arg.attrs["buffer"]


def _copy_loop(stmt: ProgramNode, bound: frozenset[str] = frozenset()) -> bool:
  """Whether ``stmt`` is a loop with constant bounds whose every statement copies one loaded value.

  Its indices may use only its own loop variables, so the copy is the same at every call.
  """
  if stmt.op != ProgramOp.FOR:
    return False
  rng = stmt.args[0]
  if rng.attrs.get("carried") or any(a.op != ProgramOp.CONST_INT for a in rng.args):
    return False
  bound = bound | {rng.attrs["name"]}
  for s in stmt.args[1:]:
    if s.op == ProgramOp.FOR:
      if not _copy_loop(s, bound):
        return False
    elif s.op != ProgramOp.STORE or s.args[1].op != ProgramOp.LOAD:
      return False
    elif any(n.op == ProgramOp.VAR and n.attrs["name"] not in bound for n in walk_program(s)):
      return False
  return True


def _copied(
  loop: ProgramNode, target: ProgramNode, source: str, buffers: dict[str, ProgramNode], procs: dict[str, ProgramNode]
) -> dict[int, int] | None:
  """For each element of ``target`` that ``loop`` writes, the element of ``source`` it copies, or None."""
  frame = _Frame(p.proc("copy", (), ()), [], procs, {})
  for name, buf in buffers.items():
    if buf.attrs.get("values") is not None:
      frame.run((buf,))
    elif name != target.attrs["name"]:
      frame.buffers[name] = _Pointer(_InputValues(buf))
  size = math.prod(target.attrs["shape"])
  frame.buffers[target.attrs["name"]] = _Pointer([None] * size)
  try:
    frame.run((loop,))
  except (_DataDependent, NotImplementedError):
    return None
  written = frame.buffers[target.attrs["name"]].values
  assert isinstance(written, list)
  copied: dict[int, int] = {}
  for at, value in enumerate(written):
    if value is None:
      continue
    view = value.args[0] if value.op == ProgramOp.LOAD else None
    if view is None or view.attrs["buffer"] != source or view.args[0].op != ProgramOp.CONST_INT:
      return None
    copied[at] = view.args[0].attrs["value"]
  return copied


def _fold_reorders(proc: ProgramNode, procs: dict[str, ProgramNode], expanded: set[str], names: NameScope) -> tuple[ProgramNode, list[ProgramNode]]:
  """Move the copy loops around ``proc``'s calls to expanded procedures into specialized callees."""
  procs, expanded = dict(procs), set(expanded)
  n_params = proc.attrs["param_count"]
  params, body = proc.args[:n_params], list(proc.args[n_params:])
  specialized: dict[str, ProgramNode] = {}
  while True:
    folded = None
    if any(stmt.op == ProgramOp.FOR for stmt in body):
      scope = _Scope(params, body, procs)
      for at, stmt in enumerate(body):
        # A noinline callee stays the call frame it is, a workaround for a compiler bug.
        if stmt.op == ProgramOp.CALL and stmt.attrs["callee"] in expanded and not procs[stmt.attrs["callee"]].attrs.get("noinline"):
          folded = _fold_call(at, body, procs, scope)
          if folded is not None:
            break
    if folded is None:
      # A specialization that a later fold specialized again is no longer called.
      called = {node.attrs["callee"] for stmt in body for node in walk_program(stmt) if node.op == ProgramOp.CALL}
      return ProgramNode(proc.op, (*params, *body), proc.attrs, dtype=proc.dtype), [pr for n, pr in specialized.items() if n in called]
    callee, args, removed = folded
    old = callee.attrs["name"]
    stem = callee.attrs.get("reordered_from", old)
    # Calls that fold the same copies into the same callee share one specialization.
    name = next((n for n, pr in specialized.items() if pr.args == callee.args and pr.attrs["reordered_from"] == stem), None)
    if name is None:
      # A specialization only this call uses is specialized again under its own name.
      others = sum(s.op == ProgramOp.CALL and s.attrs["callee"] == old for s in body) > 1
      name = old if old in specialized and not others else names.procedure(f"{stem}_reordered")
      expanded.add(name)
      specialized[name] = procs[name] = ProgramNode(
        callee.op, callee.args, {**callee.attrs, "name": name, "reordered_from": stem}, dtype=callee.dtype
      )
    call = ProgramNode(ProgramOp.CALL, tuple(args), {**body[at].attrs, "callee": name}, dtype=body[at].dtype)
    body = [call if i == at else s for i, s in enumerate(body) if i not in removed]


class _Scope:
  """The buffers of a procedure body and the statements that read or write each one."""

  def __init__(self, params: tuple[ProgramNode, ...], body: list[ProgramNode], procs: dict[str, ProgramNode]) -> None:
    self.buffers = {b.attrs["name"]: b for b in (*params, *body) if b.op == ProgramOp.BUFFER}
    self.declared = {
      s.attrs["name"]: i for i, s in enumerate(body) if s.op == ProgramOp.BUFFER and s.attrs.get("values") is None and "alias_of" not in s.attrs
    }
    self.tables = {name for name, b in self.buffers.items() if b.attrs.get("values") is not None}
    # A buffer another one aliases is reached under two names, so it keeps its copies.
    self.aliased = {b.attrs["alias_of"] for b in self.buffers.values() if "alias_of" in b.attrs}
    self.refs = [_references(s, procs) for s in body]
    self.users: dict[str, list[int]] = {}
    for i, (reads, writes) in enumerate(self.refs):
      for name in reads | writes:
        self.users.setdefault(name, []).append(i)


def _fold_call(
  at: int, body: list[ProgramNode], procs: dict[str, ProgramNode], scope: _Scope
) -> tuple[ProgramNode, list[ProgramNode], set[int]] | None:
  """The callee of ``body[at]`` with the copies into its inputs and out of its outputs folded in, its arguments, and the statements it replaces."""
  call = body[at]
  callee = procs[call.attrs["callee"]]
  n_params, n_in = callee.attrs["param_count"], callee.attrs["input_count"]
  formals = list(callee.args[:n_params])
  buffers, declared, tables, aliased, refs = scope.buffers, scope.declared, scope.tables, scope.aliased, scope.refs
  args = list(call.args)
  loads: dict[str, tuple[ProgramNode, dict[int, int]]] = {}
  stores: dict[str, tuple[ProgramNode, dict[int, list[int]]]] = {}
  removed: set[int] = set()
  for k, arg in enumerate(call.args):
    temp = _buffer_name(arg)
    if arg.op != ProgramOp.BUFFER or temp not in declared or temp in aliased or sum(_buffer_name(a) == temp for a in call.args) != 1:
      continue
    users = scope.users.get(temp, [])
    writers = [i for i in users if temp in refs[i][1]]
    formal = formals[k]
    if k < n_in:
      # The temporary is filled by one copy loop from one buffer, which nothing writes before the call reads it.
      if len(writers) != 1 or set(users) != {writers[0], at} or writers[0] > at:
        continue
      loop = writers[0]
      sources = refs[loop][0] - tables
      if not _copy_loop(body[loop]) or refs[loop][1] != {temp} or len(sources) != 1:
        continue
      (source,) = sources
      if source in aliased or buffers[source].dtype != formal.dtype or any(loop < i <= at and source in refs[i][1] for i in scope.users[source]):
        continue
      copied = _copied(body[loop], buffers[temp], source, buffers, procs)
      if copied is None or len(copied) != math.prod(buffers[temp].attrs["shape"]):
        continue
      formals[k] = _formal(formal, buffers[source])
      loads[formal.attrs["name"]] = formals[k], copied
      args[k] = buffers[source]
    else:
      # The temporary is read by one copy loop into one buffer, which nothing reads or writes in between.
      readers = [i for i in users if i != at]
      if writers != [at] or len(readers) != 1 or readers[0] < at:
        continue
      loop = readers[0]
      if not _copy_loop(body[loop]) or len(refs[loop][1]) != 1 or refs[loop][0] - tables != {temp}:
        continue
      (target,) = refs[loop][1]
      if target in aliased or buffers[target].dtype != formal.dtype or any(_buffer_name(a) == target for a in args):
        continue
      if any(at <= i < loop for i in scope.users[target]):
        continue
      copied = _copied(body[loop], buffers[target], temp, buffers, procs)
      if not copied:
        continue
      targets: dict[int, list[int]] = {}
      for to, frm in copied.items():
        targets.setdefault(frm, []).append(to)
      formals[k] = _formal(formal, buffers[target])
      stores[formal.attrs["name"]] = formals[k], targets
      args[k] = buffers[target]
    removed |= {loop, declared[temp]}
  if not removed:
    return None
  try:
    statements = _rewrite(callee.args[n_params:], loads, stores)
  except (_DataDependent, KeyError):
    return None
  if statements is None:
    return None
  return ProgramNode(callee.op, (*formals, *statements), callee.attrs, dtype=callee.dtype), args, removed


def _formal(formal: ProgramNode, actual: ProgramNode) -> ProgramNode:
  """``formal`` with the shape of the buffer now passed to it."""
  attrs = {key: value for key, value in formal.attrs.items() if key != "support"}
  return ProgramNode(ProgramOp.BUFFER, (), {**attrs, "shape": actual.attrs["shape"]}, dtype=formal.dtype)


def _rewrite(
  statements: tuple[ProgramNode, ...],
  loads: dict[str, tuple[ProgramNode, dict[int, int]]],
  stores: dict[str, tuple[ProgramNode, dict[int, list[int]]]],
) -> list[ProgramNode] | None:
  """Expanded statements reading and writing through the copies, or None for a body that is not expanded."""
  memo: dict[ProgramNode, ProgramNode] = {}

  def scalar(node: ProgramNode) -> ProgramNode:
    if node in memo:
      return memo[node]
    if node.op == ProgramOp.LOAD and node.args[0].attrs["buffer"] in loads:
      formal, copied = loads[node.args[0].attrs["buffer"]]
      out = p.load(p.view(formal, [p.const_int(copied[_index(node.args[0])])]))
    else:
      args = tuple(scalar(a) for a in node.args)
      out = node if args == node.args else ProgramNode(node.op, args, node.attrs, dtype=node.dtype)
    memo[node] = out
    return out

  out: list[ProgramNode] = []
  for stmt in statements:
    if stmt.op == ProgramOp.ASSIGN:
      out.append(scalar(stmt))
    elif stmt.op == ProgramOp.STORE:
      view, value = stmt.args
      if view.attrs["buffer"] in stores:
        formal, targets = stores[view.attrs["buffer"]]
        out.extend(p.store(p.view(formal, [p.const_int(to)]), scalar(value)) for to in targets.get(_index(view), ()))
      else:
        out.append(ProgramNode(stmt.op, (scalar(view), scalar(value)), stmt.attrs, dtype=stmt.dtype))
    else:
      return None
  return out


def _index(view: ProgramNode) -> int:
  index = view.args[0]
  if index.op != ProgramOp.CONST_INT:
    raise _DataDependent
  return index.attrs["value"]
