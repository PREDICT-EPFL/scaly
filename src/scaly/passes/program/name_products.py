"""Name a product of two locals or constants that several statements of one block repeat."""

from __future__ import annotations

from collections import Counter

from ...ir import program as p
from ...ir.program import ProgramNode, ProgramOp, walk_program
from ...utils.names import c_ident
from ..arith import CONSTANTS
from .scheduling import ScalarNameAllocator

_STATEMENTS = {ProgramOp.ASSIGN, ProgramOp.STORE, ProgramOp.STORE_PAIR}


def name_products(prog: ProgramNode) -> ProgramNode:
  """Write once, in a local, each product of two locals or constants that a block repeats.

  ``schedule_values`` leaves a floating product with two uses at both uses, so that the register
  estimate of ``widen_ranges`` sees no extra local (#166). Once widening has run, that estimate is
  done: outside a widened range the product is named before the first statement that repeats it.
  A group ends where a statement assigns one of its operands, so every use reads the same value.
  """
  args: list[ProgramNode] = []
  for node in prog.args:
    if node.op == ProgramOp.PROC:
      pc = node.attrs["param_count"]
      body = node.args[pc:]
      names = ScalarNameAllocator({c_ident(name) for name in _reserved(node.args[:pc], body)})
      node = ProgramNode(node.op, (*node.args[:pc], *_rewrite(body, names)), node.attrs, dtype=node.dtype)
    args.append(node)
  return ProgramNode(prog.op, tuple(args), prog.attrs, dtype=prog.dtype)


def _reserved(params: tuple[ProgramNode, ...], body: tuple[ProgramNode, ...]) -> set[str]:
  reserved = {param.attrs["name"] for param in params}
  for node in walk_program(p.block(*body)):
    if node.op == ProgramOp.BUFFER:
      reserved.add(node.attrs["name"])
    elif node.op == ProgramOp.ASSIGN:
      reserved.add(node.attrs["target"])
    elif node.op == ProgramOp.FOR:
      reserved.add(node.args[0].attrs["name"])
    elif node.op == ProgramOp.VAR:
      reserved.add(node.attrs["name"])
  return reserved


def _is_product(node: ProgramNode) -> bool:
  return node.op == ProgramOp.MUL and node.dtype.is_floating and all(arg.op in {*CONSTANTS, ProgramOp.VAR} for arg in node.args)


def _products(stmt: ProgramNode) -> list[ProgramNode]:
  """The products a statement computes, once per occurrence in its expression tree."""
  found: list[ProgramNode] = []
  stack = list(stmt.args[1:] if stmt.op != ProgramOp.ASSIGN else stmt.args)
  while stack:
    node = stack.pop()
    if _is_product(node):
      found.append(node)
    elif node.op != ProgramOp.VIEW:
      stack.extend(node.args)
  return found


def _assigned(stmt: ProgramNode) -> set[str]:
  return {node.attrs["target"] for node in walk_program(stmt) if node.op == ProgramOp.ASSIGN}


def _widened(stmt: ProgramNode) -> bool:
  return any(node.op == ProgramOp.RANGE and "lanes" in node.attrs for node in walk_program(stmt))


def _rewrite(body: tuple[ProgramNode, ...], names: ScalarNameAllocator) -> tuple[ProgramNode, ...]:
  # Nested blocks first, each on its own: a name declared there is not visible here.
  body = tuple(_nested(stmt, names) for stmt in body)
  # A group is a product until a statement assigns one of its operands.
  generation: Counter[str] = Counter()
  groups: dict[tuple[ProgramNode, tuple[int, ...]], list[int]] = {}
  members: list[list[tuple[ProgramNode, tuple[int, ...]]]] = [[] for _ in body]
  for at, stmt in enumerate(body):
    if stmt.op in _STATEMENTS:
      for product in _products(stmt):
        key = (product, tuple(generation[arg.attrs["name"]] for arg in product.args if arg.op == ProgramOp.VAR))
        groups.setdefault(key, []).append(at)
        members[at].append(key)
    for name in _assigned(stmt):
      generation[name] += 1
  named = {key: names.fresh() for key, uses in groups.items() if len(uses) > 1}
  if not named:
    return body
  out: list[ProgramNode] = []
  declared: set[tuple[ProgramNode, tuple[int, ...]]] = set()
  for at, stmt in enumerate(body):
    keys = [key for key in dict.fromkeys(members[at]) if key in named]
    for key in keys:
      if key not in declared:
        out.append(p.assign(named[key], key[0], declare=True))
        declared.add(key)
    if keys:
      stmt = _substitute(stmt, {key[0]: p.var(named[key], key[0].dtype) for key in keys})
    out.append(stmt)
  return tuple(out)


def _nested(stmt: ProgramNode, names: ScalarNameAllocator) -> ProgramNode:
  if stmt.op == ProgramOp.FOR and not _widened(stmt):
    return ProgramNode(stmt.op, (stmt.args[0], *_rewrite(stmt.args[1:], names)), stmt.attrs, dtype=stmt.dtype)
  if stmt.op == ProgramOp.IF:
    branches = tuple(p.block(*_rewrite(branch.args, names)) for branch in stmt.args[1:])
    return ProgramNode(stmt.op, (stmt.args[0], *branches), stmt.attrs, dtype=stmt.dtype)
  return stmt


def _substitute(stmt: ProgramNode, replace: dict[ProgramNode, ProgramNode]) -> ProgramNode:
  memo: dict[ProgramNode, ProgramNode] = {}

  def visit(node: ProgramNode) -> ProgramNode:
    if node in replace:
      return replace[node]
    if node in memo:
      return memo[node]
    if node.op == ProgramOp.VIEW or not node.args:
      result = node
    else:
      result = ProgramNode(node.op, tuple(visit(arg) for arg in node.args), node.attrs, dtype=node.dtype)
    memo[node] = result
    return result

  if stmt.op == ProgramOp.ASSIGN:
    return ProgramNode(stmt.op, tuple(visit(arg) for arg in stmt.args), stmt.attrs, dtype=stmt.dtype)
  return ProgramNode(stmt.op, (stmt.args[0], *(visit(arg) for arg in stmt.args[1:])), stmt.attrs, dtype=stmt.dtype)
