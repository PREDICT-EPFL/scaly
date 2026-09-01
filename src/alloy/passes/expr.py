"""Concrete expression-dialect rewrites: algebraic simplification, constant folding, CSE.

How a rewrite is defined and applied — ``Pattern``, ``PatternMatcher``, the walk-rebuild — is
``ir/match.py``; this module is the rule set built on it.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..ir.expr import Expr, ExprOp, OP_INFO, _attrs_key, stack, topo, zeros_like
from ..ir.match import Pattern, _replace_args, rewrite


def simplify(expr: Expr) -> Expr:
  """Apply algebraic identities and constant folding until the graph stops changing.

  Covers ``x + 0``, ``x * 1``, ``x * 0``, ``x ** 0``, ``x ** 1``, identity reshape and transpose, slice of slice, slice
  of stack, and folding of all-constant subgraphs.
  """
  for _ in range(8):
    new = rewrite(expr, SIMPLIFY_PATTERNS)
    if new is expr:
      return expr
    expr = new
  return expr


def cse(expr: Expr) -> Expr:
  """Merge structurally equal subgraphs so each distinct computation appears once."""
  return cse_many([expr])[0]


def simplify_cse_fixpoint(expr: Expr, max_rounds: int = 4) -> Expr:
  """Run simplify ↔ cse to a fixpoint.

  CSE folds structurally-identical subgraphs, which can unlock simplifications (e.g. ``x - x``
  rewrites to zero only after both sides become the same node). simplify can in turn create
  newly-identical subgraphs (``x + 0`` rewrites both sides to ``x``). The fixpoint alternates
  the two passes until the graph stops shrinking.
  """
  for _ in range(max_rounds):
    new = simplify(cse(expr))
    if new is expr:
      return expr
    expr = new
  return expr


def cse_many(outputs: Iterable[Expr]) -> tuple[Expr, ...]:
  """Common-subexpression elimination across several outputs at once.

  Sharing is found *between* the outputs as well as inside each, which is the point of asking
  for several derivatives from one ``Function.factory`` call.
  """
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


def _structural_key(expr: Expr, child_keys: dict[int, tuple[Any, ...]]) -> tuple[Any, ...]:
  value_key = None if expr.value is None else (expr.value.shape, str(expr.value.dtype), expr.value.tobytes())
  children = tuple(child_keys[arg.id] for arg in expr.args)
  if expr.op in {ExprOp.ADD, ExprOp.MUL}:
    # hash gives a fast total order even when children carry heterogeneous attrs (e.g. SLICE
    # index tuples with `(int, slice)` vs `(slice,)`); raw tuple `<` fails on int-vs-tuple at
    # the same position. CSE only needs determinism for commutativity, not lexical correctness.
    children = tuple(sorted(children, key=hash))
  return (
    ExprOp(expr.op).value,
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
  return e.op != ExprOp.CALL and bool(e.args) and all(a.op == ExprOp.CONST for a in e.args)


def _constant_fold(e: Expr) -> Expr:
  vals = [a.value for a in e.args]
  if any(v is None for v in vals):
    return e
  args = [v for v in vals if v is not None]
  if e.op == ExprOp.RESHAPE:
    out = args[0].reshape(e.attrs["shape"])
  elif e.op == ExprOp.TRANSPOSE:
    out = np.transpose(args[0], axes=e.attrs["axes"])
  elif e.op == ExprOp.SLICE:
    out = args[0][e.attrs["index"]]
  elif e.op == ExprOp.GATHER:
    indices = e.attrs["indices"]
    out = np.take(args[0].reshape(-1), indices).reshape(indices.shape)
  elif e.op == ExprOp.SCATTER:
    out = np.zeros(e.shape, dtype=np.float64).reshape(-1)
    out[e.attrs["indices"].reshape(-1)] = args[0].reshape(-1)
    out = out.reshape(e.shape)
  elif e.op == ExprOp.STACK:
    out = np.stack(args, axis=e.attrs.get("axis", 0))
  elif e.op == ExprOp.CONCAT:
    out = np.concatenate(args, axis=e.attrs.get("axis", 0))
  elif e.op == ExprOp.SUM:
    out = np.asarray(np.sum(args[0]), dtype=np.float64)
  elif e.op == ExprOp.MATMUL:
    out = args[0] @ args[1]
  else:
    info = OP_INFO[ExprOp(e.op)]
    if info.numpy is None:
      return e
    out = info.numpy(*args)
  return Expr.const(out, dtype=e.type.dtype, lowering=e.lowering)


def _const_value(e: Expr) -> np.ndarray | None:
  return e.value if e.op == ExprOp.CONST else None


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
  if _is_zero(x) and _same_shape_as_result(y, e):
    return -y
  return e


def _neg_of_neg(e: Expr) -> Expr:
  return e.args[0].args[0]


def _zero_unary(e: Expr) -> Expr:
  return zeros_like(e)


def _all_args_zero(e: Expr) -> bool:
  return bool(e.args) and all(_is_zero(a) for a in e.args)


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


def _pow_identity(e: Expr) -> Expr:
  base, exponent = e.args
  if _is_zero(exponent):
    return Expr.const(np.ones(e.shape, dtype=np.float64), lowering=e.lowering)
  if _is_one(exponent) and _same_shape_as_result(base, e):
    return base
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
  if inner.op != ExprOp.SLICE:
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
  if stack_node.op != ExprOp.STACK:
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
  Pattern(ExprOp.ADD, lambda e: e.args[0] is e.args[1] or _is_zero(e.args[0]) or _is_zero(e.args[1]), _add_identity),
  Pattern(ExprOp.SUB, lambda e: e.args[0] is e.args[1] or _is_zero(e.args[0]) or _is_zero(e.args[1]), _sub_identity),
  Pattern(ExprOp.MUL, lambda e: _is_zero(e.args[0]) or _is_zero(e.args[1]) or _is_one(e.args[0]) or _is_one(e.args[1]), _mul_identity_or_zero),
  Pattern(ExprOp.DIV, lambda e: e.args[0] is e.args[1] or _is_one(e.args[1]), _div_identity),
  Pattern(ExprOp.POW, lambda e: _is_zero(e.args[1]) or _is_one(e.args[1]), _pow_identity),
  Pattern(ExprOp.NEG, lambda e: e.args[0].op == ExprOp.NEG, _neg_of_neg),
  Pattern(ExprOp.RESHAPE, lambda e: e.args[0].shape == e.shape, _reshape_identity),
  Pattern(ExprOp.TRANSPOSE, lambda e: e.attrs["axes"] == tuple(range(len(e.attrs["axes"]))), _transpose_identity),
  Pattern(ExprOp.MATMUL, lambda e: _is_zero(e.args[0]) or _is_zero(e.args[1]), _matmul_zero),
  Pattern(ExprOp.SUM, lambda e: _is_zero(e.args[0]), _zero_unary),
  Pattern(ExprOp.GATHER, lambda e: _is_zero(e.args[0]), _zero_unary),
  Pattern(ExprOp.SCATTER, lambda e: _is_zero(e.args[0]), _zero_unary),
  Pattern(ExprOp.STACK, _all_args_zero, _zero_unary),
  Pattern(ExprOp.CONCAT, _all_args_zero, _zero_unary),
  Pattern(ExprOp.SLICE, _slice_of_stack_full, _slice_of_stack),
  Pattern(ExprOp.SLICE, _slice_of_slice_step1, _slice_of_slice),
)
