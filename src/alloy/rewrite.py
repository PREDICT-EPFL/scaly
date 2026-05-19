from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Callable, Iterable

import numpy as np

from .expr import Expr, _attrs_key, stack, topo, zeros_like
from .ops import Ops


@dataclass(frozen=True, slots=True)
class Pattern:
  op: Ops | None
  predicate: Callable[[Expr], bool]
  replacement: Callable[[Expr], Expr]

  def matches(self, expr: Expr) -> bool:
    return (self.op is None or expr.op == self.op) and self.predicate(expr)


class PatternMatcher:
  def __init__(self, patterns: Iterable[Pattern]):
    self.any: list[Pattern] = []
    self.by_op: dict[Ops, list[Pattern]] = defaultdict(list)
    for pattern in patterns:
      if pattern.op is None:
        self.any.append(pattern)
      else:
        self.by_op[pattern.op].append(pattern)

  def candidates(self, op: Ops) -> Iterable[Pattern]:
    yield from self.any
    yield from self.by_op.get(op, ())

  def rewrite(self, expr: Expr) -> Expr | None:
    for pattern in self.candidates(Ops(expr.op)):
      if pattern.matches(expr):
        ret = pattern.replacement(expr)
        if ret is not expr:
          return ret
    return None


def rewrite(expr: Expr, patterns: Iterable[Pattern] | PatternMatcher) -> Expr:
  matcher = patterns if isinstance(patterns, PatternMatcher) else PatternMatcher(patterns)
  replacements: dict[int, Expr] = {}
  for node in topo((expr,)):
    cur = _replace_args(node, replacements)
    # tinygrad-style graph rewrite invariant: each original node is processed once
    # in topological order, with already-rewritten children cached in replacements.
    while (new := matcher.rewrite(cur)) is not None:
      cur = new
    replacements[node.id] = cur
  return replacements[expr.id]


def simplify(expr: Expr) -> Expr:
  for _ in range(8):
    new = rewrite(expr, SIMPLIFY_PATTERNS)
    if new is expr:
      return expr
    expr = new
  return expr


def cse(expr: Expr) -> Expr:
  return cse_many([expr])[0]


def cse_many(outputs: Iterable[Expr]) -> tuple[Expr, ...]:
  outputs = tuple(outputs)
  memo: dict[tuple[Any, ...], Expr] = {}
  keys: dict[int, tuple[Any, ...]] = {}
  replacements: dict[int, Expr] = {}
  for node in topo(outputs):
    cur = _replace_args(node, replacements)
    key = _structural_key(cur, keys)
    cur = memo.setdefault(key, cur)
    keys[node.id] = key
    keys[cur.id] = key
    replacements[node.id] = cur
  return tuple(replacements[out.id] for out in outputs)


def _replace_args(expr: Expr, replacements: dict[int, Expr]) -> Expr:
  args = tuple(replacements.get(arg.id, arg) for arg in expr.args)
  return (
    expr
    if all(a is b for a, b in zip(args, expr.args, strict=True))
    else Expr(expr.op, args, expr.type, expr.name, expr.value, dict(expr.attrs), expr.lowering)
  )


def _structural_key(expr: Expr, child_keys: dict[int, tuple[Any, ...]]) -> tuple[Any, ...]:
  value_key = None if expr.value is None else (expr.value.shape, str(expr.value.dtype), expr.value.tobytes())
  children = tuple(child_keys[arg.id] for arg in expr.args)
  if expr.op in {Ops.ADD, Ops.MUL}:
    # hash gives a fast total order even when children carry heterogeneous attrs (e.g. SLICE
    # index tuples with `(int, slice)` vs `(slice,)`); raw tuple `<` fails on int-vs-tuple at
    # the same position. CSE only needs determinism for commutativity, not lexical correctness.
    children = tuple(sorted(children, key=hash))
  return (
    Ops(expr.op).value,
    expr.name,
    expr.type.shape,
    expr.type.dtype,
    expr.type.diff,
    expr.lowering,
    _attrs_key(expr.attrs),
    value_key,
    children,
  )


def _all_args_const(e: Expr) -> bool:
  return e.op != Ops.CALL and bool(e.args) and all(a.op == Ops.CONST for a in e.args)


def _constant_fold(e: Expr) -> Expr:
  return Expr.const(e.eval({}), lowering=e.lowering)


def _const_value(e: Expr) -> np.ndarray | None:
  return e.value if e.op == Ops.CONST else None


def _is_zero(e: Expr) -> bool:
  value = _const_value(e)
  return value is not None and bool(np.all(value == 0))


def _is_one(e: Expr) -> bool:
  value = _const_value(e)
  return value is not None and bool(np.all(value == 1))


def _same_shape_as_result(arg: Expr, result: Expr) -> bool:
  return arg.shape == result.shape


def _add_identity(e: Expr) -> Expr:
  x, y = e.args
  if x is y:
    return Expr.const(2.0, lowering=e.lowering) * x
  if _is_zero(y) and _same_shape_as_result(x, e):
    return x
  if _is_zero(x) and _same_shape_as_result(y, e):
    return y
  return e


def _sub_identity(e: Expr) -> Expr:
  x, y = e.args
  if x is y:
    return zeros_like(e)
  if _is_zero(y) and _same_shape_as_result(x, e):
    return x
  return e


def _mul_identity_or_zero(e: Expr) -> Expr:
  x, y = e.args
  if _is_zero(x) or _is_zero(y):
    return zeros_like(e)
  if _is_one(y) and _same_shape_as_result(x, e):
    return x
  if _is_one(x) and _same_shape_as_result(y, e):
    return y
  return e


def _div_identity(e: Expr) -> Expr:
  x, y = e.args
  if x is y:
    return Expr.const(np.ones(e.shape, dtype=np.float64), lowering=e.lowering)
  if _is_one(y) and _same_shape_as_result(x, e):
    return x
  return e


def _reshape_identity(e: Expr) -> Expr:
  return e.args[0] if e.args[0].shape == e.shape else e


def _transpose_identity(e: Expr) -> Expr:
  axes = e.attrs["axes"]
  return e.args[0] if axes == tuple(range(len(axes))) else e


def _matmul_zero(e: Expr) -> Expr:
  return zeros_like(e) if _is_zero(e.args[0]) or _is_zero(e.args[1]) else e


def _slice_of_slice_step1(e: Expr) -> bool:
  inner = e.args[0]
  if inner.op != Ops.SLICE:
    return False
  for item in inner.attrs["index"]:
    if isinstance(item, slice) and item.step not in (None, 1):
      return False
  for item in e.attrs["index"]:
    if isinstance(item, slice) and item.step not in (None, 1):
      return False
  return True


def _slice_of_slice(e: Expr) -> Expr:
  inner = e.args[0]
  base = inner.args[0]
  index1 = inner.attrs["index"]
  index2 = e.attrs["index"]
  combined: list[Any] = []
  outer_iter = iter(index2)
  for dim, item in enumerate(index1):
    if isinstance(item, int):
      combined.append(item if item >= 0 else base.shape[dim] + item)
      continue
    start1, stop1, _ = item.indices(base.shape[dim])
    outer = next(outer_iter, slice(None))
    if isinstance(outer, int):
      pos = outer if outer >= 0 else (stop1 - start1) + outer
      combined.append(start1 + pos)
    else:
      a2, b2, _ = outer.indices(stop1 - start1)
      combined.append(slice(start1 + a2, start1 + b2))
  return base[tuple(combined)]


def _slice_of_stack_full(e: Expr) -> bool:
  stack_node = e.args[0]
  if stack_node.op != Ops.STACK:
    return False
  axis = stack_node.attrs.get("axis", 0)
  index = e.attrs["index"]
  if axis >= len(index) or not isinstance(index[axis], slice):
    return False
  start, stop, step = index[axis].indices(stack_node.shape[axis])
  return start == 0 and stop == stack_node.shape[axis] and step == 1


def _slice_of_stack(e: Expr) -> Expr:
  stack_node = e.args[0]
  axis = stack_node.attrs.get("axis", 0)
  index = e.attrs["index"]
  rest = index[:axis] + index[axis + 1 :]
  if not rest or all(isinstance(item, slice) and item.indices(d) == (0, d, 1) for item, d in zip(rest, stack_node.args[0].shape, strict=True)):
    return stack_node
  new_args = [arg[rest] for arg in stack_node.args]
  new_axis = sum(1 for item in index[:axis] if isinstance(item, slice))
  return stack(new_args, axis=new_axis)


SIMPLIFY_PATTERNS: tuple[Pattern, ...] = (
  Pattern(None, _all_args_const, _constant_fold),
  Pattern(Ops.ADD, lambda e: e.args[0] is e.args[1] or _is_zero(e.args[0]) or _is_zero(e.args[1]), _add_identity),
  Pattern(Ops.SUB, lambda e: e.args[0] is e.args[1] or _is_zero(e.args[1]), _sub_identity),
  Pattern(Ops.MUL, lambda e: _is_zero(e.args[0]) or _is_zero(e.args[1]) or _is_one(e.args[0]) or _is_one(e.args[1]), _mul_identity_or_zero),
  Pattern(Ops.DIV, lambda e: e.args[0] is e.args[1] or _is_one(e.args[1]), _div_identity),
  Pattern(Ops.RESHAPE, lambda e: e.args[0].shape == e.shape, _reshape_identity),
  Pattern(Ops.TRANSPOSE, lambda e: e.attrs["axes"] == tuple(range(len(e.attrs["axes"]))), _transpose_identity),
  Pattern(Ops.MATMUL, lambda e: _is_zero(e.args[0]) or _is_zero(e.args[1]), _matmul_zero),
  Pattern(Ops.SLICE, _slice_of_stack_full, _slice_of_stack),
  Pattern(Ops.SLICE, _slice_of_slice_step1, _slice_of_slice),
)
