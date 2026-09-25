"""Concrete expression-dialect rewrites: algebraic simplification, constant folding, CSE.

How a rewrite is defined and applied — ``Pattern``, ``PatternMatcher``, the walk-rebuild — is
``ir/match.py``; this module is the rule set built on it.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..ir.expr import Expr, ExprOp, OP_INFO, _attrs_key, matmul, stack, topo, zeros_like
from ..ir.types import dtypes
from ..ir.match import Pattern, _replace_args, rewrite
from .arith import ARITH_EXPR, fold


def simplify(expr: Expr) -> Expr:
  """Apply algebraic identities and constant folding until the graph stops changing.

  Covers the shared arithmetic identities of ``passes/arith.py`` (neutral elements, zero
  annihilation, ``x - x``, negation normalization, small constant powers), folding of all-constant
  subgraphs, identity reshape, transpose and gather, slice of slice, slice of stack, ``A.T @ v`` as
  ``v @ A`` (and ``v @ A.T`` as ``A @ v``), and a matmul with an all-ones vector as sums.
  """
  return rewrite(expr, SIMPLIFY_PATTERNS, fixpoint=False, revisit=True, max_steps=100_000)


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
  replacements: dict[int, Expr] = {}
  for node in topo(outputs):
    cur = _replace_args(node, replacements)
    key = _structural_key(cur)
    cur = memo.setdefault(key, cur)
    replacements[node.id] = cur
  return tuple(replacements[out.id] for out in outputs)


def _structural_key(expr: Expr) -> tuple[Any, ...]:
  value_key = None if expr.value is None else (expr.value.shape, str(expr.value.dtype), expr.value.tobytes())
  children = tuple(arg.id for arg in expr.args)
  if expr.op in {ExprOp.ADD, ExprOp.MUL}:
    children = tuple(sorted(children))
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
  try:
    with np.errstate(divide="raise", invalid="raise", over="raise"):
      out = _evaluate(e, args)
  except FloatingPointError:
    return e
  return e if out is None else Expr.const(out, dtype=e.type.dtype, lowering=e.lowering)


def _evaluate(e: Expr, args: list[np.ndarray]) -> np.ndarray | np.generic | None:
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
  elif e.op == ExprOp.CAST:
    out = args[0].astype(e.type.dtype.numpy())
  else:
    info = OP_INFO[ExprOp(e.op)]
    if info.numpy is None:
      return None
    out = info.numpy(*args)
  return out


def _const_value(e: Expr) -> np.ndarray | None:
  return e.value if e.op == ExprOp.CONST else None


def _is_zero(e: Expr) -> bool:
  value = _const_value(e)
  return value is not None and bool(np.all(value == 0))


def _is_one(e: Expr) -> bool:
  value = _const_value(e)
  return value is not None and bool(np.all(value == 1))


def _arith(e: Expr) -> Expr:
  return fold(ARITH_EXPR, e)


def _zero_unary(e: Expr) -> Expr:
  return zeros_like(e)


def _all_args_zero(e: Expr) -> bool:
  return bool(e.args) and all(_is_zero(a) for a in e.args)


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


def _is_matrix_transpose(e: Expr) -> bool:
  return e.op == ExprOp.TRANSPOSE and len(e.shape) == 2 and e.attrs["axes"] == (1, 0)


def _matmul_of_transpose_and_vector(e: Expr) -> bool:
  a, b = e.args
  return (len(b.shape) == 1 and _is_matrix_transpose(a)) or (len(a.shape) == 1 and _is_matrix_transpose(b))


def _matmul_transpose_fold(e: Expr) -> Expr:
  """``A.T @ v -> v @ A`` and ``v @ A.T -> A @ v``: the same product with no transpose materialized."""
  a, b = e.args
  return matmul(b, a.args[0]) if len(b.shape) == 1 else matmul(b.args[0], a)


def _matmul_ones_vector(e: Expr) -> bool:
  a, b = e.args
  return len(a.shape) == len(b.shape) == 1 and (_is_one(a) or _is_one(b))


def _matmul_ones(e: Expr) -> Expr:
  """``v @ 1 -> sum(v)``. The matrix forms wait for an axis reduction in the IR: as stacked row sums
  they lower to one loop per row and lose the fused producer, which is slower in loop form."""
  a, b = e.args
  return (b if _is_one(a) else a).sum()


def _select_folds(e: Expr) -> bool:
  cond, a, b = e.args
  value = _const_value(cond)
  uniform = value is not None and value.size > 0 and bool(np.all(value == value.reshape(-1)[0]))
  if uniform:
    return (a if value.reshape(-1)[0] else b).shape == e.shape  # type: ignore[union-attr]
  return (a is b and a.shape == e.shape) or (_is_zero(a) and _is_zero(b) and e.type.dtype == dtypes.float64)


def _select_fold(e: Expr) -> Expr:
  cond, a, b = e.args
  value = _const_value(cond)
  if value is not None and value.size > 0:
    return a if value.reshape(-1)[0] else b
  return a if a is b else zeros_like(e)


def _gather_identity(e: Expr) -> bool:
  indices = e.attrs["indices"]
  return indices.size == e.args[0].size and bool(np.array_equal(indices.reshape(-1), np.arange(indices.size)))


SIMPLIFY_PATTERNS: tuple[Pattern, ...] = (
  Pattern(None, _all_args_const, _constant_fold),
  *(Pattern(op, lambda e: not _all_args_const(e), _arith) for op in (ExprOp.ADD, ExprOp.SUB, ExprOp.MUL, ExprOp.DIV, ExprOp.NEG, ExprOp.POW)),
  Pattern(ExprOp.RESHAPE, lambda e: e.args[0].shape == e.shape and not e.attrs.get("lowering_identity"), _reshape_identity),
  Pattern(ExprOp.TRANSPOSE, lambda e: e.attrs["axes"] == tuple(range(len(e.attrs["axes"]))), _transpose_identity),
  Pattern(ExprOp.MATMUL, lambda e: _is_zero(e.args[0]) or _is_zero(e.args[1]), _matmul_zero),
  Pattern(ExprOp.MATMUL, _matmul_of_transpose_and_vector, _matmul_transpose_fold),
  Pattern(ExprOp.MATMUL, _matmul_ones_vector, _matmul_ones),
  Pattern(ExprOp.SUM, lambda e: _is_zero(e.args[0]), _zero_unary),
  Pattern(ExprOp.GATHER, lambda e: _is_zero(e.args[0]), _zero_unary),
  Pattern(ExprOp.GATHER, _gather_identity, lambda e: e.args[0].reshape(e.shape)),
  Pattern(ExprOp.SCATTER, lambda e: _is_zero(e.args[0]), _zero_unary),
  Pattern(ExprOp.STACK, _all_args_zero, _zero_unary),
  Pattern(ExprOp.CONCAT, _all_args_zero, _zero_unary),
  Pattern(ExprOp.SLICE, _slice_of_stack_full, _slice_of_stack),
  Pattern(ExprOp.SLICE, _slice_of_slice_step1, _slice_of_slice),
  Pattern(ExprOp.SELECT, lambda e: not _all_args_const(e) and _select_folds(e), _select_fold),
  Pattern(ExprOp.NOT, lambda e: e.args[0].op == ExprOp.NOT, lambda e: e.args[0].args[0]),
)
