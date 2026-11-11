"""The elementwise table: each op's local partials, NumPy fold, program op and C spelling, read by both AD modes."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..ir.expr import Expr, ExprOp, _full_like, _uniform, cast, equal, gather, not_equal, pack, scatter, to_dense, values, where, zeros_like
from ..ir.program import ProgramOp
from ..ir.support import project_entries
from ..ir.types import TensorType, stored_indices


@dataclass(frozen=True, slots=True)
class Mask:
  """A partial selected by ``condition``, optionally scaled by ``weight``, and zero elsewhere.

  A product with a mask is a select of the product, so the branch not chosen contributes an exact
  zero even where it, or its derivative, is NaN or infinite.
  """

  condition: Expr
  weight: Expr | None = None


type Local = Expr | int | tuple[Expr | int, Expr] | Mask
type Partial = Callable[[Expr], Local]


@dataclass(frozen=True, slots=True)
class Elementwise:
  """One row of the elementwise table.

  ``partials[i]`` builds the partial of a node with respect to operand ``i``, and is called only
  when that operand is active. A partial of ``1`` or ``-1`` passes the tangent through or negates
  it. A pair ``(numerator, denominator)`` is a quotient: the tangent is multiplied by the numerator
  before the division, so a finite product does not overflow or underflow through the partial. A
  ``Mask`` selects. An operand that is not floating is never active.
  ``partials`` is ``None`` for an op without a derivative. ``jvp`` replaces the sum of partials
  times tangents with a prescribed form, more stable or cheaper; it receives the tangents aligned
  with the output.
  ``c`` is the C spelling, a libm function, an operator or a template over the operands ``{0}``,
  ``{1}``, ``{2}`` and the result's C type ``{type}``, and ``c_integer`` a different spelling for
  integer operands. ``expensive`` marks a libm call that fusion must not duplicate.
  """

  program: ProgramOp
  numpy: Callable[..., np.ndarray | np.generic]
  c: str
  partials: tuple[Partial, ...] | None = None
  jvp: Callable[[Expr, Sequence[Expr | None]], Expr] | None = None
  c_integer: str | None = None
  expensive: bool = False


def zero_partial(e: Expr) -> int:
  """A structurally zero partial, shared by the derivative and sparsity traversals."""
  return 0


def _const(value: float, like: Expr) -> Expr:
  return Expr.const(value, dtype=like.type.dtype)


def _zero(like: Expr) -> Expr:
  return zeros_like(like) if like.type.pattern is not None else _const(0, like)


def _sqrt_one_minus_square(e: Expr) -> Expr:
  return (_const(1, e) - e.args[0] ** _const(2, e)).sqrt()


def _atan2_denominator(e: Expr) -> Expr:
  y, x = e.args
  return x * x + y * y


def _mul_jvp(e: Expr, tangents: Sequence[Expr | None]) -> Expr:
  (x, y), (dx, dy) = e.args, tangents
  if x is y:
    return _seed_axis(_const(2, e) * x) * dx
  left = None if dx is None else dx * _seed_axis(y, e)
  right = None if dy is None else _seed_axis(x, e) * dy
  if left is None:
    assert right is not None
    return right
  return left if right is None else left + right


def _div_jvp(e: Expr, tangents: Sequence[Expr | None]) -> Expr:
  dx, db = tangents
  quotient = _seed_axis(e)
  numerator = dx if db is None else -(quotient * db) if dx is None else dx - quotient * db
  return numerator / _seed_axis(e.args[1], e)


def _inactive(e: Expr) -> Local:
  raise AssertionError("a bool condition is never active")


def _select_jvp(e: Expr, tangents: Sequence[Expr | None]) -> Expr:
  _, dx, dy = tangents
  zero = _const(0, e)
  condition = _seed_axis(e.args[0], e)
  result = where(condition, zero if dx is None else dx, zero if dy is None else dy)
  masks = [None if t is None else cotangent_mask(t) for t in (dx, dy)]
  if any(mask is not None for mask in masks):
    activity = [Expr.const(t is not None, dtype=condition.type.dtype) if mask is None else mask for t, mask in zip((dx, dy), masks, strict=True)]
    result = where(where(condition, *activity), result, zeros_like(result))
  return result


def _extremum_partial(e: Expr, position: int) -> Mask:
  x, y = e.args[position], e.args[1 - position]
  chosen = x >= y if e.op == ExprOp.MAXIMUM else x <= y
  return Mask(chosen, where(equal(x, y), _const(0.5, e), _const(1, e)))


def reduction_partial(e: Expr) -> Mask:
  """Equal shares among all logical extrema, including implicit sparse zeros."""
  x = to_dense(e.args[0])
  shape = tuple(1 if i in e.attrs["axes"] else n for i, n in enumerate(x.shape))
  chosen = equal(x, to_dense(e).reshape(shape))
  count = cast(chosen, e.type.dtype).sum(e.attrs["axes"], keepdims=True)
  return Mask(chosen, _const(1, e) / count)


def times_partial(local: Mask, seed: Expr) -> Expr:
  """Apply a masked partial while preserving an outer mask on the seed."""
  return _masked(seed, lambda inner: _times_partial(local, inner))


# NumPy has no erf; frompyfunc keeps constant folding vectorized without adding SciPy.
def _erf(x: np.ndarray) -> np.ndarray:
  return np.asarray(np.frompyfunc(math.erf, 1, 1)(x), dtype=np.float64)


ELEMENTWISE: dict[ExprOp, Elementwise] = {
  ExprOp.NEG: Elementwise(ProgramOp.NEG, np.negative, "-", (lambda e: -1,)),
  ExprOp.SIN: Elementwise(ProgramOp.SIN, np.sin, "sin", (lambda e: e.args[0].cos(),), expensive=True),
  ExprOp.COS: Elementwise(ProgramOp.COS, np.cos, "cos", (lambda e: -e.args[0].sin(),), expensive=True),
  ExprOp.TAN: Elementwise(ProgramOp.TAN, np.tan, "tan", (lambda e: (1, e.args[0].cos() ** _const(2, e)),), expensive=True),
  ExprOp.ASIN: Elementwise(ProgramOp.ASIN, np.arcsin, "asin", (lambda e: (1, _sqrt_one_minus_square(e)),), expensive=True),
  ExprOp.ACOS: Elementwise(ProgramOp.ACOS, np.arccos, "acos", (lambda e: (-1, _sqrt_one_minus_square(e)),), expensive=True),
  ExprOp.ATAN: Elementwise(ProgramOp.ATAN, np.arctan, "atan", (lambda e: (1, _const(1, e) + e.args[0] ** _const(2, e)),), expensive=True),
  ExprOp.SINH: Elementwise(ProgramOp.SINH, np.sinh, "sinh", (lambda e: e.args[0].cosh(),), expensive=True),
  ExprOp.COSH: Elementwise(ProgramOp.COSH, np.cosh, "cosh", (lambda e: e.args[0].sinh(),), expensive=True),
  ExprOp.TANH: Elementwise(ProgramOp.TANH, np.tanh, "tanh", (lambda e: _const(1, e) - e * e,), expensive=True),
  ExprOp.ERF: Elementwise(
    ProgramOp.ERF, _erf, "erf", (lambda e: _const(2 / math.sqrt(math.pi), e) * (-(e.args[0] ** _const(2, e))).exp(),), expensive=True
  ),
  ExprOp.EXP: Elementwise(ProgramOp.EXP, np.exp, "exp", (lambda e: e,), expensive=True),
  ExprOp.LOG: Elementwise(ProgramOp.LOG, np.log, "log", (lambda e: (1, e.args[0]),), expensive=True),
  ExprOp.SQRT: Elementwise(ProgramOp.SQRT, np.sqrt, "sqrt", (lambda e: _const(0.5, e) / e,), expensive=True),
  ExprOp.ABS: Elementwise(
    ProgramOp.ABS,
    np.abs,
    "fabs",
    (lambda e: Mask(not_equal(e.args[0], 0), where(e.args[0] > 0, _const(1, e), where(e.args[0] < 0, _const(-1, e), e.args[0]))),),
    c_integer="(({type})({0} < 0 ? 0 - (uint64_t)({0}) : (uint64_t)({0})))",
  ),
  ExprOp.FLOOR: Elementwise(ProgramOp.FLOOR, np.floor, "floor", (zero_partial,)),
  ExprOp.CEIL: Elementwise(ProgramOp.CEIL, np.ceil, "ceil", (zero_partial,)),
  ExprOp.ADD: Elementwise(ProgramOp.ADD, np.add, "+", (lambda e: 1, lambda e: 1)),
  ExprOp.SUB: Elementwise(ProgramOp.SUB, np.subtract, "-", (lambda e: 1, lambda e: -1)),
  ExprOp.MUL: Elementwise(ProgramOp.MUL, np.multiply, "*", (lambda e: e.args[1], lambda e: e.args[0]), _mul_jvp),
  ExprOp.DIV: Elementwise(ProgramOp.DIV, np.divide, "/", (lambda e: (1, e.args[1]), lambda e: (-e, e.args[1])), _div_jvp),
  ExprOp.POW: Elementwise(
    ProgramOp.POW,
    np.power,
    "pow",
    (lambda e: e.args[1] * e.args[0] ** (e.args[1] - _const(1, e)), lambda e: e * e.args[0].log()),
    expensive=True,
  ),
  ExprOp.ATAN2: Elementwise(
    ProgramOp.ATAN2,
    np.arctan2,
    "atan2",
    (lambda e: (e.args[1], _atan2_denominator(e)), lambda e: (-e.args[0], _atan2_denominator(e))),
    expensive=True,
  ),
  ExprOp.MINIMUM: Elementwise(
    ProgramOp.MINIMUM,
    np.minimum,
    "scaly_minimum",
    (lambda e: _extremum_partial(e, 0), lambda e: _extremum_partial(e, 1)),
    c_integer="({0} < {1} ? {0} : {1})",
  ),
  ExprOp.MAXIMUM: Elementwise(
    ProgramOp.MAXIMUM,
    np.maximum,
    "scaly_maximum",
    (lambda e: _extremum_partial(e, 0), lambda e: _extremum_partial(e, 1)),
    c_integer="({0} > {1} ? {0} : {1})",
  ),
  ExprOp.LT: Elementwise(ProgramOp.LT, np.less, "<"),
  ExprOp.LE: Elementwise(ProgramOp.LE, np.less_equal, "<="),
  ExprOp.EQ: Elementwise(ProgramOp.EQ, np.equal, "=="),
  ExprOp.NE: Elementwise(ProgramOp.NE, np.not_equal, "!="),
  ExprOp.AND: Elementwise(ProgramOp.AND, np.logical_and, "&&"),
  ExprOp.OR: Elementwise(ProgramOp.OR, np.logical_or, "||"),
  ExprOp.NOT: Elementwise(ProgramOp.NOT, np.logical_not, "!"),
  # isfinite may return any nonzero int; a bool holds 0 or 1.
  ExprOp.ISFINITE: Elementwise(ProgramOp.ISFINITE, np.isfinite, "(isfinite({0}) != 0)"),
  ExprOp.SELECT: Elementwise(
    ProgramOp.SELECT,
    np.where,
    "({0} ? {1} : {2})",
    (_inactive, lambda e: Mask(e.args[0]), lambda e: Mask(~e.args[0])),
    _select_jvp,
    # Integer literals are C ints: the cast keeps a select between two of them in its own width.
    c_integer="(({type})({0} ? {1} : {2}))",
  ),
  # Constant folding converts to the node's dtype, so the fold itself is the identity.
  ExprOp.CAST: Elementwise(ProgramOp.CAST, lambda x: x, "(({type})({0}))", (lambda e: 1,), lambda e, t: cast(t[0], e.type.dtype)),
  # NumPy's floor semantics, which C's / and % do not have; codegen/c.py defines the helpers.
  ExprOp.FLOORDIV: Elementwise(ProgramOp.FLOORDIV, np.floor_divide, "(({type})scaly_floordiv({0}, {1}))"),
  ExprOp.MOD: Elementwise(ProgramOp.FLOORMOD, np.remainder, "(({type})scaly_floormod({0}, {1}))"),
}


def tangent(expr: Expr, tangents: Sequence[Expr | None], nseed: int) -> Expr:
  """The tangent of an elementwise ``expr``, shaped ``(nseed, ...)``, from its operands' tangents.

  Each tangent is ``(nseed, *operand.shape)``, or ``None`` for an inactive operand, whose partial
  is never built.
  """
  if expr.type.pattern is not None:
    compact = _compact_elementwise(expr)
    seed_type = TensorType((nseed, *expr.shape), expr.type.dtype, pattern=expr.type.pattern)
    seeds = [
      None if t is None else _sample_stored(_broadcast_tangent(t, arg, expr, nseed), seed_type) for arg, t in zip(expr.args, tangents, strict=True)
    ]
    return pack(tangent(compact, seeds, nseed), expr.type.pattern, shape=seed_type.shape)
  row = ELEMENTWISE[ExprOp(expr.op)]
  assert row.partials is not None
  aligned = [None if t is None else _broadcast_tangent(t, arg, expr, nseed) for arg, t in zip(expr.args, tangents, strict=True)]
  if row.jvp is not None and (expr.op == ExprOp.SELECT or not any(t is not None and _mask_of(t) is not None for t in aligned)):
    return row.jvp(expr, aligned)
  total: Expr | None = None
  for partial, dx in zip(row.partials, aligned, strict=True):
    if dx is None:
      continue
    local = partial(expr)
    if _mask_of(dx) is not None:
      term = _masked(dx, lambda inner: _times_partial(local, inner, lambda partial: _seed_axis(partial, expr)))
      total = term if total is None else accumulate(total, term)
      continue
    negate = isinstance(local, int) and local < 0
    term = dx if negate else _times_partial(local, dx, lambda partial: _seed_axis(partial, expr))
    total = (-term if negate else term) if total is None else total - term if negate else accumulate(total, term)
  assert total is not None
  return total


def cotangent(expr: Expr, cot: Expr, position: int) -> Expr:
  """The cotangent of operand ``position`` of an elementwise ``expr``: ``cot`` times the partial,
  summed over the axes the operand was broadcast along."""
  if expr.type.pattern is not None:
    compact = _compact_elementwise(expr)
    partials = ELEMENTWISE[ExprOp(expr.op)].partials
    assert partials is not None
    local = partials[position](compact)
    contribution = _masked(_sample_stored(cot, expr.type), lambda t: _times_partial(local, t))
    operand = expr.args[position]
    projected = project_entries(stored_indices(expr.type), operand.shape, expr.shape)

    def transpose(c: Expr) -> tuple[Expr]:
      if operand.type.pattern is None:
        return (scatter(c, projected, operand.shape),)
      entries = stored_indices(operand.type)
      slots = np.searchsorted(entries, projected)
      keep = slots < len(entries)
      keep[keep] &= entries[slots[keep]] == projected[keep]
      result = scatter(gather(c, np.flatnonzero(keep)), slots[keep], operand.type.storage_shape)
      return (pack(result, operand.type.pattern, shape=operand.shape),)

    (result,) = carry_mask(contribution, transpose)
    assert result is not None
    return result
  partials = ELEMENTWISE[ExprOp(expr.op)].partials
  assert partials is not None
  local = partials[position](expr)
  # A cotangent a select masked keeps its mask outside the product, so a NaN or infinite partial
  # where the select did not choose this value stays out: where(c, t * partial, 0).
  product = _masked(cot, lambda inner: _times_partial(local, inner))
  arg = expr.args[position]
  (summed,) = carry_mask(product, lambda c: (_unbroadcast(c, arg.shape, expr.shape),))
  assert summed is not None
  return _masked(summed, lambda inner: cast(inner, arg.type.dtype))


def carry_mask(cot: Expr, transpose: Callable[[Expr], tuple[Expr | None, ...]]) -> tuple[Expr | None, ...]:
  """``transpose(cot)`` for a linear map whose weights are 0 or 1, such as a sum, a slice or a
  reshape, each result masked where no entry the mask on ``cot`` keeps reaches it."""
  results = transpose(cot)
  mask = _mask_of(cot)
  if mask is None:
    return results
  ones = _full_like(1, cot) if cot.type.pattern is not None else Expr.const(np.ones(cot.shape, dtype=cot.type.dtype.numpy()), dtype=cot.type.dtype)
  weights = transpose(_masked(cot, lambda inner: ones))
  return tuple(
    r if r is None or w is None or r.op == ExprOp.CONST or _mask_of(r) is not None else where(_nonzero_mask(w), r, _zero(r))
    for r, w in zip(results, weights, strict=True)
  )


def _nonzero_mask(expr: Expr) -> Expr:
  if expr.type.pattern is not None:
    return pack(not_equal(values(expr), 0), expr.type.pattern, shape=expr.shape)
  return not_equal(expr, 0)


def accumulate(total: Expr, cot: Expr) -> Expr:
  """Two cotangents of one node summed, under the union of their masks when both have one."""
  a, b = _mask_of(total), _mask_of(cot)
  if a is None or b is None:
    return total + cot
  if a[0] is b[0]:
    return where(a[0], a[1] + b[1], _zero(total))
  if total.type.pattern is not None:
    combined = pack(_sample_stored(a[0], total.type) | _sample_stored(b[0], total.type), total.type.pattern, shape=total.shape)
    return where(combined, total + cot, zeros_like(total))
  return where(a[0] | b[0], total + cot, _const(0, total))


def _mask_of(cot: Expr) -> tuple[Expr, Expr] | None:
  """The chosen condition and inner value of a select with a constant zero branch."""
  if ExprOp(cot.op) != ExprOp.SELECT:
    return None
  condition, x, y = cot.args
  for inner, other in ((x, y), (y, x)):
    zero = other.args[0] if other.op == ExprOp.PACK else other
    if zero.op == ExprOp.CONST and zero.value is not None and not np.any(zero.value) and inner.shape == cot.shape:
      if inner is y:
        condition = ~condition if condition.type.pattern is None else pack(~values(condition), condition.type.pattern, shape=condition.shape)
      return condition, inner
  return None


def cotangent_mask(cot: Expr) -> Expr | None:
  """The conjunction of the select conditions enclosing a cotangent, independent of its value."""
  condition: Expr | None = None
  while (mask := _mask_of(cot)) is not None:
    condition = mask[0] if condition is None else condition & mask[0]
    cot = mask[1]
  return condition


def _masked(cot: Expr, product: Callable[[Expr], Expr]) -> Expr:
  mask = _mask_of(cot)
  if mask is None:
    return product(cot)
  inner = _masked(mask[1], product)
  return where(mask[0], inner, _zero(inner))


def _compact_elementwise(expr: Expr) -> Expr:
  return Expr(
    expr.op,
    tuple(_sample_stored(arg, expr.type) for arg in expr.args),
    TensorType(expr.type.storage_shape, expr.type.dtype, expr.type.diff),
    lowering=expr.lowering,
  )


def _sample_stored(expr: Expr, output: TensorType) -> Expr:
  (sampled,) = carry_mask(expr, lambda e: (_sample_stored_unmasked(e, output),))
  assert sampled is not None
  return sampled


def _sample_stored_unmasked(expr: Expr, output: TensorType) -> Expr:
  if expr.shape == output.shape and expr.type.pattern is output.pattern:
    return values(expr)
  logical = project_entries(stored_indices(output), expr.shape, output.shape)
  if expr.type.pattern is None:
    return gather(expr, logical).reshape(output.storage_shape)
  entries = stored_indices(expr.type)
  slots = np.searchsorted(entries, logical)
  keep = slots < len(entries)
  keep[keep] &= entries[slots[keep]] == logical[keep]
  sampled = gather(values(expr), slots[keep])
  return scatter(sampled, np.flatnonzero(keep), output.storage_shape)


def _times_partial(local: Local, t: Expr, align: Callable[[Expr], Expr] | None = None) -> Expr:
  """``t`` times a partial. Forward mode passes ``align`` to put the partial on the seed axis."""
  if isinstance(local, Mask):
    condition = local.condition if align is None else align(local.condition)
    weight = local.weight if align is None or local.weight is None else align(local.weight)
    product = t if weight is None else t * weight
    shape = np.broadcast_shapes(condition.shape, product.shape)
    if product.shape != shape:
      product = product * _uniform(1, shape, dtype=t.type.dtype)
    return where(condition, product, _const(0, t))
  if isinstance(local, tuple):
    numerator, denominator = local
    return _times_partial(numerator, t, align) / (denominator if align is None else align(denominator))
  if isinstance(local, int):
    return -t if local < 0 else t if local else _uniform(0, t.shape, dtype=t.type.dtype)
  return t * local if align is None else align(local) * t


def _seed_axis(expr: Expr, output: Expr | None = None) -> Expr:
  missing = 0 if output is None else len(output.shape) - len(expr.shape)
  return expr.reshape((1, *(1,) * missing, *expr.shape))


def _broadcast_tangent(tangent: Expr, operand: Expr, output: Expr, nseed: int) -> Expr:
  missing = len(output.shape) - len(operand.shape)
  if missing and tangent.type.pattern is not None:
    return pack(
      values(tangent).reshape((nseed, *(1,) * missing, *operand.type.storage_shape)),
      tangent.type.pattern,
      shape=(nseed, *(1,) * missing, *operand.shape),
    )
  return tangent if missing == 0 else tangent.reshape((nseed, *(1,) * missing, *operand.shape))


def _unbroadcast(cot: Expr, in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> Expr:
  if in_shape == out_shape:
    return cot
  padded = (1,) * (len(out_shape) - len(in_shape)) + in_shape
  axes = tuple(
    i for i, (small, large) in enumerate(zip(padded, out_shape, strict=True)) if i < len(out_shape) - len(in_shape) or small == 1 and large != 1
  )
  return cot.sum(axes).reshape(in_shape)
