"""Concrete expression-dialect rewrites: algebraic simplification, constant folding, CSE.

How a rewrite is defined and applied — ``Pattern``, ``PatternMatcher``, the walk-rebuild — is
``ir/match.py``; this module is the rule set built on it.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..ir.expr import Expr, ExprOp, _attrs_key, define_rules, gather, matmul, op_def, scatter, stack, topo, zeros_like
from ..ir.match import Pattern, _replace_args, rewrite
from .arith import ARITH_EXPR, fold


def simplify(expr: Expr) -> Expr:
  """Apply algebraic identities and constant folding until the graph stops changing.

  Covers the shared arithmetic identities of ``passes/arith.py`` (neutral elements, zero
  annihilation, ``x - x``, negation normalization, small constant powers), folding of all-constant
  subgraphs, identity reshape, transpose and gather, slice of slice, slice of stack, ``A.T @ v`` as
  ``v @ A`` (and ``v @ A.T`` as ``A @ v``), a matmul with an all-ones vector as sums, a product
  with a mostly-zero constant as a scatter of the entries it keeps, and the composition of a
  gather with what it reads: another gather, a transpose, a slice, a concatenation, a scatter, or
  a sum of such, so that an array which is only placed and picked from is never formed.
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
    str(expr.op),
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
  info = op_def(e.op)
  if info.fold is not None:
    return info.fold(e, args)
  return None if info.numpy is None else info.numpy(*args)


_SEGMENT_UFUNCS = {"add": np.add, "max": np.maximum, "min": np.minimum}


def _fold_segment_reduce(e: Expr, args: list[np.ndarray]) -> np.ndarray:
  out = np.full(e.size, e.attrs["fill"], dtype=np.float64)
  _SEGMENT_UFUNCS[e.attrs["reduce"]].at(out, e.attrs["indices"].reshape(-1), args[0].reshape(-1))
  return out.reshape(e.shape)


def _fold_put(e: Expr, args: list[np.ndarray]) -> np.ndarray:
  """A put or put_add of constants, lane by lane in order; a lane outside the last axis drops."""
  base, idx, values = args
  n = e.shape[-1]
  if not e.size:
    return base.astype(np.float64).reshape(e.shape)
  out = base.astype(np.float64, copy=True).reshape(-1, n)
  vals = np.asarray(values, dtype=np.float64).reshape(out.shape[0], -1)
  for j, i in enumerate(np.asarray(idx, dtype=np.int64).reshape(-1)):
    if 0 <= i < n:
      out[:, i] = out[:, i] + vals[:, j] if e.op == ExprOp.PUT_ADD else vals[:, j]
  return out.reshape(e.shape)


def _fold_gather(e: Expr, args: list[np.ndarray]) -> np.ndarray:
  indices = e.attrs["indices"]
  return np.take(args[0].reshape(-1), indices).reshape(indices.shape)


# How the builtin ops that need their attributes fold (``OpDef.fold``); the rest apply ``OpDef.numpy``.
_FOLD_RULES = {
  ExprOp.RESHAPE: lambda e, args: args[0].reshape(e.attrs["shape"]),
  ExprOp.TRANSPOSE: lambda e, args: np.transpose(args[0], axes=e.attrs["axes"]),
  ExprOp.SLICE: lambda e, args: args[0][e.attrs["index"]],
  ExprOp.GATHER: _fold_gather,
  ExprOp.SEGMENT_REDUCE: _fold_segment_reduce,
  ExprOp.PUT_ADD: _fold_put,
  ExprOp.PUT: _fold_put,
  ExprOp.STACK: lambda e, args: np.stack(args, axis=e.attrs.get("axis", 0)),
  ExprOp.CONCAT: lambda e, args: np.concatenate(args, axis=e.attrs.get("axis", 0)),
  ExprOp.SUM: lambda e, args: np.asarray(np.sum(args[0]), dtype=np.float64),
  ExprOp.MATMUL: lambda e, args: args[0] @ args[1],
  ExprOp.CAST: lambda e, args: args[0].astype(e.type.dtype.numpy()),
}
for _op, _rule in _FOLD_RULES.items():
  define_rules(_op, fold=_rule)


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
  return (a is b and a.shape == e.shape) or _same_constant(a, b, e.shape)


def _same_constant(a: Expr, b: Expr, shape: tuple[int, ...]) -> bool:
  """Whether ``a`` and ``b`` are constants with the same bits once broadcast to ``shape``: ``-0.0`` is not ``0.0``."""
  va, vb = _const_value(a), _const_value(b)
  return va is not None and vb is not None and np.broadcast_to(va, shape).tobytes() == np.broadcast_to(vb, shape).tobytes()


def _select_fold(e: Expr) -> Expr:
  cond, a, b = e.args
  value = _const_value(cond)
  if value is not None and value.size > 0:
    return a if value.reshape(-1)[0] else b
  if a is b and a.shape == e.shape:
    return a
  assert a.value is not None
  return Expr.const(np.broadcast_to(a.value, e.shape).copy(), dtype=e.type.dtype, lowering=e.lowering)


def _gather_identity(e: Expr) -> bool:
  indices = e.attrs["indices"]
  return indices.size == e.args[0].size and bool(np.array_equal(indices.reshape(-1), np.arange(indices.size)))


def _gathered(e: Expr) -> Expr:
  """What a gather reads, looking through reshapes (which keep the flat order)."""
  src = e.args[0]
  while src.op == ExprOp.RESHAPE and not src.attrs.get("lowering_identity"):
    src = src.args[0]
  return src


def _compose_gathers(e: Expr) -> Expr:
  """``gather(gather(x, a), b) == gather(x, a[b])``: one index table, one loop, and one index
  expression instead of two composed ones (each ``k // n`` of the inner map inside the outer's)."""
  inner = _gathered(e)
  return gather(inner.args[0], np.asarray(inner.attrs["indices"]).reshape(-1)[np.asarray(e.attrs["indices"])])


def _source(e: Expr) -> Expr:
  """``e`` looking through reshapes, which keep the flat order."""
  while e.op == ExprOp.RESHAPE and not e.attrs.get("lowering_identity"):
    e = e.args[0]
  return e


def _transpose_of_gather(e: Expr) -> Expr:
  """A transpose of gathered values is one gather, with the index table transposed."""
  inner = _source(e.args[0])
  table = np.asarray(inner.attrs["indices"]).reshape(e.args[0].shape).transpose(e.attrs["axes"])
  return gather(inner.args[0], np.ascontiguousarray(table))


def _is_scatter(e: Expr) -> bool:
  """A ``segment_reduce`` that adds into zeros: what ``scatter`` builds."""
  return e.attrs["reduce"] == "add" and e.attrs["fill"] == 0.0


def _is_permutation(e: Expr) -> bool:
  indices = np.asarray(e.attrs["indices"]).reshape(-1)
  return indices.size == e.size and bool(np.array_equal(np.sort(indices), np.arange(e.size)))


def _scatter_to_gather(e: Expr) -> Expr:
  """A scatter that writes every entry once is a gather by the inverse permutation, which needs no
  zeroing pass and composes with the gathers and transposes around it."""
  indices = np.asarray(e.attrs["indices"]).reshape(-1)
  inverse = np.empty_like(indices)
  inverse[indices] = np.arange(indices.size)
  return gather(e.args[0].reshape((e.args[0].size,)), inverse.reshape(e.shape))


def _gather_of_transpose(e: Expr) -> Expr:
  """A gather of a transpose is a gather of what was transposed, at the transposed places."""
  moved = _gathered(e)
  axes = moved.attrs["axes"]
  coords = np.unravel_index(np.asarray(e.attrs["indices"]), moved.shape)
  back = [coords[axes.index(axis)] for axis in range(len(axes))]  # coordinate ``axis`` of the source is the transposed one at its place in ``axes``
  flat = np.zeros_like(back[0])
  for coord, size in zip(back, moved.args[0].shape, strict=True):
    flat = flat * size + coord
  return gather(moved.args[0], flat)


def _opens(e: Expr) -> bool:
  """Whether a gather of ``e`` simplifies further: ``e`` is, through reshapes, transposes and
  sums, built of scatters and gathers, whose index tables compose with the gather's."""
  pending = [e]
  while pending:
    cur = _source(pending.pop())
    if cur.op == ExprOp.GATHER or (cur.op == ExprOp.SEGMENT_REDUCE and _is_scatter(cur)):
      return True
    if cur.op == ExprOp.TRANSPOSE or (cur.op == ExprOp.ADD and all(arg.shape == cur.shape for arg in cur.args)):
      pending.extend(cur.args)
  return False


def _picks_once(e: Expr) -> bool:
  """Whether a gather reads no entry twice: then each term of a sum under it does no more work
  gathered than it did summed. A gather that repeats entries (a broadcast, a tiling) would make
  every term repeat them, where the sum is formed once and then repeated: a dense stage Hessian
  built that way ran 1.3x slower with its terms distributed."""
  indices = np.asarray(e.attrs["indices"]).reshape(-1)
  return np.unique(indices).size == indices.size


def _gather_of_sum(e: Expr) -> Expr:
  """A gather of a sum whose terms are placed or picked arrays is the sum of the terms' gathers,
  each of which composes with its own table: the assembly of a sparse derivative, which sums
  blocks scattered into a compressed matrix and then gathers the nonzeros, never forms the matrix.
  Only for a gather that reads each entry at most once (``_picks_once``)."""
  total = _gathered(e)
  flat = (total.size,)
  x, y = (gather(arg.reshape(flat), e.attrs["indices"]) for arg in total.args)
  return x + y


def _gather_of_scatter(e: Expr) -> Expr:
  """A gather of scattered values reads the values directly: each place gathered takes the values
  scattered to it (none: zero; several: their sum), so the zero array they were placed in is
  never filled. Every place taking exactly one value is a plain gather."""
  placed = _gathered(e)
  at = np.asarray(placed.attrs["indices"]).reshape(-1)
  wanted = np.asarray(e.attrs["indices"]).reshape(-1)
  order = np.argsort(at, kind="stable")
  first, last = np.searchsorted(at[order], wanted, side="left"), np.searchsorted(at[order], wanted, side="right")
  counts = last - first
  into = np.repeat(np.arange(wanted.size), counts)
  source = order[np.concatenate([np.arange(a, b) for a, b in zip(first, last, strict=True) if b > a])] if into.size else into
  values = placed.args[0].reshape((placed.args[0].size,))
  if not into.size:
    return zeros_like(e)
  if np.all(counts == 1):
    return gather(values, source.reshape(e.shape))
  return scatter(gather(values, source), into, e.shape)


def _gather_of_slice(e: Expr) -> Expr:
  """A gather of a slice reads what was sliced, at the places the slice kept."""
  cut = _gathered(e)
  whole = cut.args[0]
  kept = np.arange(whole.size).reshape(whole.shape)[cut.attrs["index"]].reshape(-1)
  return gather(whole, kept[np.asarray(e.attrs["indices"])])


# The parts of a concatenation a gather may read from and still be taken apart: each part read
# becomes a term of its own.
GATHERED_PARTS = 4


def _parts_read(e: Expr) -> tuple[np.ndarray, np.ndarray]:
  """For a gather of a concatenation: the part each index reads, and the place in that part."""
  joined = _gathered(e)
  axis = joined.attrs.get("axis", 0)
  owner = np.concatenate([np.full(part.shape, at) for at, part in enumerate(joined.args)], axis=axis).reshape(-1)
  place = np.concatenate([np.arange(part.size).reshape(part.shape) for part in joined.args], axis=axis).reshape(-1)
  indices = np.asarray(e.attrs["indices"]).reshape(-1)
  return owner[indices], place[indices]


def _placed(e: Expr) -> bool:
  """Whether ``e`` is a constant, an array built of placed and picked arrays (``_opens``), or a
  concatenation of such: what a gather reads through without computing anything."""
  e = _source(e)
  if e.op == ExprOp.CONCAT:
    return all(_placed(part) for part in e.args)
  return e.op == ExprOp.CONST or _opens(e)


def _takes_apart(e: Expr) -> bool:
  """Whether a gather of a concatenation is better read from the parts: always from one part; from
  a few when it reads no entry twice and every part read is a constant or built of placed and
  picked arrays, which compose with the gather. A part that is computed stays whole: gathered, it
  would be computed through an index table, where the race cars' Jacobian computes it in a
  vector loop and gathers after (1.13x slower taken apart)."""
  owner = np.unique(_parts_read(e)[0])
  if owner.size == 1:
    return True
  parts = [_gathered(e).args[int(at)] for at in owner]
  return owner.size <= GATHERED_PARTS and _picks_once(e) and all(_placed(part) for part in parts)


def _gather_of_concat(e: Expr) -> Expr:
  """A gather of a concatenation reads the parts: one part, a gather of it; a few, each part's
  values placed where the gather wanted them."""
  joined = _gathered(e)
  owner, place = _parts_read(e)
  total: Expr | None = None
  for at in np.unique(owner):
    into = np.flatnonzero(owner == at)
    part = joined.args[int(at)]
    picked = gather(part.reshape((part.size,)), place[into])
    if into.size == owner.size:
      return picked.reshape(e.shape)
    term = scatter(picked, into, e.shape)
    total = term if total is None else total + term
  assert total is not None
  return total


# A constant factor with at most one nonzero entry in this many is a scatter of its nonzeros.
SPARSE_FACTOR = 8


def _sparse_factor(e: Expr) -> int | None:
  """Which operand of the product ``e`` is a constant of the product's shape that is mostly zeros."""
  for at, arg in enumerate(e.args):
    if arg.op == ExprOp.CONST and arg.shape == e.shape and e.size >= SPARSE_FACTOR and arg.value is not None:
      if np.count_nonzero(arg.value) * SPARSE_FACTOR <= e.size:
        return at
  return None


def _product_with_sparse_constant(e: Expr) -> Expr:
  """A product with a constant that is mostly zeros (a seed matrix, a selector) computes only the
  entries the constant keeps and places them: the zeros are never multiplied, and a gather of the
  result composes with the placement. A zero times anything is zero here, as in ``x * 0``."""
  at = _sparse_factor(e)
  assert at is not None
  factor, other = e.args[at], e.args[1 - at]
  values = np.asarray(factor.value).reshape(-1)
  kept = np.flatnonzero(values)
  if not kept.size:
    return zeros_like(e)
  source = np.broadcast_to(np.arange(other.size).reshape(other.shape), e.shape).reshape(-1)[kept]
  picked = gather(other.reshape((other.size,)), source)
  if not np.all(values[kept] == 1.0):
    picked = picked * Expr.const(values[kept], dtype=factor.type.dtype)
  return scatter(picked, kept, e.shape)


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
  Pattern(ExprOp.GATHER, lambda e: _gathered(e).op == ExprOp.GATHER, _compose_gathers),
  Pattern(ExprOp.GATHER, lambda e: _gathered(e).op == ExprOp.TRANSPOSE, _gather_of_transpose),
  Pattern(ExprOp.GATHER, lambda e: _gathered(e).op == ExprOp.SLICE, _gather_of_slice),
  Pattern(ExprOp.GATHER, lambda e: _gathered(e).op == ExprOp.CONCAT and _takes_apart(e), _gather_of_concat),
  Pattern(ExprOp.GATHER, lambda e: _gathered(e).op == ExprOp.ADD and _picks_once(e) and _opens(_gathered(e)), _gather_of_sum),
  Pattern(ExprOp.GATHER, lambda e: _gathered(e).op == ExprOp.SEGMENT_REDUCE and _is_scatter(_gathered(e)), _gather_of_scatter),
  Pattern(ExprOp.SEGMENT_REDUCE, lambda e: _is_scatter(e) and not _is_zero(e.args[0]) and _is_permutation(e), _scatter_to_gather),
  Pattern(ExprOp.MUL, lambda e: not _all_args_const(e) and _sparse_factor(e) is not None, _product_with_sparse_constant),
  Pattern(ExprOp.TRANSPOSE, lambda e: _source(e.args[0]).op == ExprOp.GATHER, _transpose_of_gather),
  Pattern(ExprOp.SEGMENT_REDUCE, lambda e: _is_scatter(e) and _is_zero(e.args[0]), _zero_unary),
  Pattern(ExprOp.STACK, _all_args_zero, _zero_unary),
  Pattern(ExprOp.CONCAT, _all_args_zero, _zero_unary),
  Pattern(ExprOp.SLICE, _slice_of_stack_full, _slice_of_stack),
  Pattern(ExprOp.SLICE, _slice_of_slice_step1, _slice_of_slice),
  Pattern(ExprOp.SELECT, lambda e: not _all_args_const(e) and _select_folds(e), _select_fold),
  Pattern(ExprOp.NOT, lambda e: e.args[0].op == ExprOp.NOT, lambda e: e.args[0].args[0]),
)
