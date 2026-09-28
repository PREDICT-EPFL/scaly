"""``ragged_add`` and ``ragged_dot``, the run-time-range expression ops a sparse factorization's
column updates are made of, with their derivatives, structural sparsity, verification, loop
lowering and the positions they touch per step (what lets a loop carry be updated in place)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse

from ...ad.forward import JVPManyUnsupported, is_zero_const
from ...ad.sparsity import empty_mask, incidence, mask_compose, mask_or
from ...ir import program as p
from ...ir.expr import Expr, as_expr, common_lowering, diff_any, promote_dtype, register_op, stack, zeros_like
from ...ir.program import ProgramNode, RangeKind
from ...ir.spec import Rule
from ...ir.types import TensorType, dtypes

if TYPE_CHECKING:
  from ...passes.lowering import LowerCtx, Positions, StepValue

RAGGED_ADD, RAGGED_DOT = "ragged_add", "ragged_dot"


def _ragged_map(table: Any, what: str) -> np.ndarray | None:
  if table is None:
    return None
  arr = np.asarray(table, dtype=np.int64).reshape(-1)
  if arr.size and arr.min() < 0:
    raise ValueError(f"{what} entries must be non-negative")
  return arr


def _ragged_bounds(lo: Any, hi: Any) -> tuple[Expr, Expr]:
  lo, hi = as_expr(lo), as_expr(hi)
  for e, what in ((lo, "lo"), (hi, "hi")):
    if e.type.dtype != dtypes.int64 or len(e.shape) != 1:
      raise TypeError(f"ragged {what} must be a rank-1 int64 vector, got {e.type.dtype}{e.shape}")
  if lo.shape != hi.shape:
    raise ValueError(f"ragged lo and hi must have one entry per group, got {lo.shape} and {hi.shape}")
  return lo, hi


def ragged_add(base: Any, src: Any, lo: Any, hi: Any, scale: Any, *, dst_map: Any = None, src_map: Any = None) -> Expr:
  """``base`` plus, for every group ``g`` and every ``p`` with ``lo[g] <= p < hi[g]``,
  ``src[src_map[p]] * scale[g]`` added at ``dst_map[p]`` (``None`` maps are the identity).

  Ranges of run-time length: the inner loop of a sparse update such as a left-looking column
  update (``w[rows[p]] -= L[p] * s_k`` over a contiguous run of a column). ``lo``/``hi`` are ``int64``
  vectors, ``scale`` a float vector, all with one entry per group; a group with ``lo == hi`` does
  nothing. The maps are fixed tables. Nothing is checked at run time: every ``p`` must lie inside
  the maps and every mapped index inside ``base`` and ``src``. Library code builds the ranges from
  its own tables.
  """
  base, src, scale = as_expr(base), as_expr(src), as_expr(scale)
  lo, hi = _ragged_bounds(lo, hi)
  if len(base.shape) != 1 or len(src.shape) != 1 or scale.shape != lo.shape:
    raise ValueError(f"ragged_add needs vectors and one scale per group, got base {base.shape}, src {src.shape}, scale {scale.shape}")
  promote_dtype(base, src, scale)
  attrs = {"dst_map": _ragged_map(dst_map, "dst_map"), "src_map": _ragged_map(src_map, "src_map")}
  return Expr(
    RAGGED_ADD,
    (base, src, lo, hi, scale),
    TensorType(base.shape, dtype=base.type.dtype, diff=diff_any(base, src, scale)),
    attrs=attrs,
    lowering=common_lowering(base, src, lo, hi, scale),
  )


def ragged_dot(a: Any, b: Any, lo: Any, hi: Any, *, a_map: Any = None, b_map: Any = None) -> Expr:
  """One dot product per group: ``out[g] = sum over lo[g] <= p < hi[g] of a[a_map[p]] * b[b_map[p]]``
  (``None`` maps are the identity). The counterpart of ``ragged_add``, with the same unchecked
  contract; ``ragged_add``'s derivative with respect to its scale is one of these."""
  a, b = as_expr(a), as_expr(b)
  lo, hi = _ragged_bounds(lo, hi)
  if len(a.shape) != 1 or len(b.shape) != 1:
    raise ValueError(f"ragged_dot needs vectors, got {a.shape} and {b.shape}")
  dtype = promote_dtype(a, b)
  return Expr(
    RAGGED_DOT,
    (a, b, lo, hi),
    TensorType(lo.shape, dtype=dtype, diff=diff_any(a, b)),
    attrs={"a_map": _ragged_map(a_map, "a_map"), "b_map": _ragged_map(b_map, "b_map")},
    lowering=common_lowering(a, b, lo, hi),
  )


def _ragged_shapes(expr: Expr) -> str | None:
  lo, hi = expr.args[2], expr.args[3]
  if lo.shape != hi.shape or len(lo.shape) != 1 or lo.type.dtype.name != "int64" or hi.type.dtype.name != "int64":
    return f"{expr.op} needs int64 lo and hi vectors of one shape, got {lo.type.dtype}{lo.shape} and {hi.type.dtype}{hi.shape}"
  if expr.op == RAGGED_ADD and (expr.args[4].shape != lo.shape or expr.shape != expr.args[0].shape):
    return "RAGGED_ADD needs one scale per group and keeps its base's shape"
  if expr.op == RAGGED_DOT and expr.shape != lo.shape:
    return "RAGGED_DOT gives one value per group"
  return None


def ragged_tangent(expr: Expr, d: list[Expr | None]) -> Expr:
  """``ragged_add`` and ``ragged_dot`` are linear in each floating operand: the tangent is the same op
  with one operand replaced by its tangent at a time (``None`` for a zero tangent)."""
  if expr.op == RAGGED_ADD:
    base, src, lo, hi, scale = expr.args
    maps = {"dst_map": expr.attrs["dst_map"], "src_map": expr.attrs["src_map"]}
    out = d[0] if d[0] is not None else zeros_like(expr)
    if d[1] is not None:
      out = ragged_add(out, d[1], lo, hi, scale, **maps)
    if d[4] is not None:
      out = ragged_add(out, src, lo, hi, d[4], **maps)
    return out
  a, b, lo, hi = expr.args
  maps = {"a_map": expr.attrs["a_map"], "b_map": expr.attrs["b_map"]}
  terms = [ragged_dot(d[0], b, lo, hi, **maps)] if d[0] is not None else []
  if d[1] is not None:
    terms.append(ragged_dot(a, d[1], lo, hi, **maps))
  return (terms[0] + terms[1] if len(terms) == 2 else terms[0]) if terms else zeros_like(expr)


def _jvp_ragged(expr: Expr, d: list[Expr]) -> Expr:
  return ragged_tangent(expr, [None if is_zero_const(t) else t for t in d])


def _jvp_many_ragged(expr: Expr, tan: Callable[[Expr], Expr], nseed: int) -> Expr:
  args = expr.args
  d = [tan(arg) for arg in args]
  if nseed > 64:
    raise JVPManyUnsupported(str(expr.op))  # one tangent op per seed would outgrow the per-seed fallback
  rows = [ragged_tangent(expr, [None if is_zero_const(t) else t[k] for t in d]) for k in range(nseed)]
  return stack(rows, axis=0)


def _vjp_ragged_add(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  base, src, lo, hi, scale = args
  dmap, smap = expr.attrs["dst_map"], expr.attrs["src_map"]
  src_bar = ragged_add(zeros_like(src), cot, lo, hi, scale, dst_map=smap, src_map=dmap)
  scale_bar = ragged_dot(src, cot, lo, hi, a_map=smap, b_map=dmap)
  return (cot, src_bar, zeros_like(lo), zeros_like(hi), scale_bar)


def _vjp_ragged_dot(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  a, b, lo, hi = args
  amap, bmap = expr.attrs["a_map"], expr.attrs["b_map"]
  a_bar = ragged_add(zeros_like(a), b, lo, hi, cot, dst_map=amap, src_map=bmap)
  b_bar = ragged_add(zeros_like(b), a, lo, hi, cot, dst_map=bmap, src_map=amap)
  return (a_bar, b_bar, zeros_like(lo), zeros_like(hi))


def _sparsity_ragged(expr: Expr, mask: Callable[[Expr], sparse.csr_array], ncols: int) -> sparse.csr_array:
  # Run-time ranges: any output entry may depend on any entry of the floating operands (and a
  # ragged_add's entry on its own base entry).
  floats = [expr.args[1], expr.args[4]] if expr.op == RAGGED_ADD else [expr.args[0], expr.args[1]]
  out = mask(expr.args[0]) if expr.op == RAGGED_ADD else empty_mask((expr.size, ncols))
  for arg in floats:
    dense = incidence((expr.size, arg.size), np.repeat(np.arange(expr.size), arg.size), np.tile(np.arange(arg.size), expr.size))
    out = mask_or(out, mask_compose(dense, mask(arg)))
  return out


def _mapped(ctx: LowerCtx, table: np.ndarray | None, p_var: ProgramNode) -> ProgramNode:
  """``table[p]`` from a ``static const`` table, or ``p`` itself for the identity map."""
  if table is None:
    return p_var
  return p.load(p.view(ctx.new_const_index(table), [p_var]))


def _lower_ragged_add(ctx: LowerCtx, node: Expr) -> None:
  """Copy the base (unless this is a link of a proven in-place chain), then for each group a loop of
  run-time length ``hi[g] - lo[g]`` adding ``src[src_map[p]] * scale[g]`` at ``dst_map[p]``: the
  inner loop of a sparse column update, one contiguous run of the source per group."""
  base, src, lo, hi, scale = node.args
  if ctx.in_place is not None and node.id in ctx.in_place:
    out = ctx.output_buffer(0)
    ctx.bind(node, out.attrs["name"])
  else:
    out = ctx.new_private(node.type.dtype, node.shape)
    ctx.bind(node, out.attrs["name"])
    ctx.emit(ctx.copy_loop(ctx.buf_of(base), out, node.shape))
  groups = lo.size
  if not groups:
    return
  nm = f"{out.attrs['name']}_{ctx.fresh_id()}"
  g, q = p.var(f"rg_{nm}"), p.var(f"rp_{nm}")
  dst = _mapped(ctx, node.attrs["dst_map"], q)
  src_i = _mapped(ctx, node.attrs["src_map"], q)
  weight = p.load(p.view(ctx.buf_of(scale), [g]))
  view = p.view(out, [dst])
  update = p.store(view, p.add(p.load(view), p.mul(p.load(p.view(ctx.buf_of(src), [src_i])), weight)))
  inner = p.for_(p.range_(q.attrs["name"], p.load(p.view(ctx.buf_of(lo), [g])), p.load(p.view(ctx.buf_of(hi), [g])), kind=RangeKind.REDUCE), [update])
  ctx.emit(p.for_(p.range_(g.attrs["name"], 0, groups, kind=RangeKind.SERIAL), [inner]))


def _lower_ragged_dot(ctx: LowerCtx, node: Expr) -> None:
  """One accumulation per group over its run: four partial sums, as for the dense dot products."""
  a, b, lo, hi = node.args
  out = ctx.alloc_tmp(node)
  groups = lo.size
  if not groups:
    return
  nm = f"{out.attrs['name']}_{ctx.fresh_id()}"
  g = p.var(f"rg_{nm}")
  a_map, b_map = node.attrs["a_map"], node.attrs["b_map"]
  a_tab = None if a_map is None else ctx.new_const_index(a_map)
  b_tab = None if b_map is None else ctx.new_const_index(b_map)

  def term(q: ProgramNode) -> ProgramNode:
    ai = q if a_tab is None else p.load(p.view(a_tab, [q]))
    bi = q if b_tab is None else p.load(p.view(b_tab, [q]))
    return p.mul(p.load(p.view(ctx.buf_of(a), [ai])), p.load(p.view(ctx.buf_of(b), [bi])))

  sums, total = ctx.blocked_sum(f"r_{nm}", p.load(p.view(ctx.buf_of(lo), [g])), p.load(p.view(ctx.buf_of(hi), [g])), term, node.type.dtype)
  body = [*sums, p.store(p.view(out, [g]), total)]
  ctx.emit(p.for_(p.range_(g.attrs["name"], 0, groups, kind=RangeKind.SERIAL), body))


class _Ragged:
  """The positions a ragged op visits per step: ``table[p]`` (or ``p``) for ``p`` in the step's
  ranges. Kept as ranges, since enumerating them costs as much as the factorization they describe;
  bounds come from the ranges and the table's extremes."""

  def __init__(self, lo: np.ndarray, hi: np.ndarray, table: np.ndarray | None) -> None:
    self.lo, self.hi, self.table = lo, hi, table

  def bounds(self) -> tuple[np.ndarray, np.ndarray]:
    """Per step, an interval ``[low, high]`` holding every position (``low > high`` when none)."""
    live = self.hi > self.lo
    any_live = live.any(axis=1)
    if self.table is None:
      low = np.where(live, self.lo, np.iinfo(np.int64).max).min(axis=1, initial=np.iinfo(np.int64).max)
      high = np.where(live, self.hi - 1, -1).max(axis=1, initial=-1)
    else:
      low = np.full(self.lo.shape[0], int(self.table.min()) if self.table.size else 0)
      high = np.full(self.lo.shape[0], int(self.table.max()) if self.table.size else -1)
    return np.where(any_live, low, 1), np.where(any_live, high, 0)

  def explicit(self) -> np.ndarray:
    return _ragged_positions(self.lo, self.hi, self.table)


def _ragged_positions(lo: np.ndarray, hi: np.ndarray, table: np.ndarray | None) -> np.ndarray:
  """Per step, every ``table[p]`` (or ``p``) for ``p`` in the step's ranges ``[lo[g], hi[g])``, padded
  with -1 to the widest step."""
  length = lo.shape[0]
  per_step = [np.concatenate([np.arange(a, b) for a, b in zip(lo[t], hi[t], strict=True)] or [np.zeros(0, np.int64)]) for t in range(length)]
  width = max((x.size for x in per_step), default=0)
  out = np.full((length, max(width, 1)), -1, dtype=np.int64)
  for t, x in enumerate(per_step):
    out[t, : x.size] = x if table is None else table[x]
  return out


def _ragged_ranges(node: Expr, table: np.ndarray | None, value: StepValue, length: int) -> _Ragged | None:
  """The mapped positions a ragged op visits at each step, kept as ranges until needed."""
  lo, hi = value(node.args[2]), value(node.args[3])
  if lo is None or hi is None:
    return None
  return _Ragged(lo.reshape(length, -1), hi.reshape(length, -1), table)


def _ragged_reads(node: Expr, position: int, value: StepValue, length: int) -> Positions | None:
  if node.op == RAGGED_ADD:
    return _ragged_ranges(node, node.attrs["src_map"], value, length) if position == 1 else None
  return _ragged_ranges(node, node.attrs["a_map" if position == 0 else "b_map"], value, length) if position in (0, 1) else None


def _ragged_writes(node: Expr, value: StepValue, length: int) -> Positions | None:
  return _ragged_ranges(node, node.attrs["dst_map"], value, length)


register_op(
  RAGGED_ADD,
  arity=5,
  jvp=_jvp_ragged,
  jvp_many=_jvp_many_ragged,
  vjp=_vjp_ragged_add,
  sparsity=_sparsity_ragged,
  verify=(Rule(RAGGED_ADD, "ragged-shapes", _ragged_shapes),),
  lower=_lower_ragged_add,
  traits={"runtime_index": True, "update": _ragged_writes, "reads": _ragged_reads},
)
register_op(
  RAGGED_DOT,
  arity=4,
  jvp=_jvp_ragged,
  jvp_many=_jvp_many_ragged,
  vjp=_vjp_ragged_dot,
  sparsity=_sparsity_ragged,
  verify=(Rule(RAGGED_DOT, "ragged-shapes", _ragged_shapes),),
  lower=_lower_ragged_dot,
  traits={"runtime_index": True, "reads": _ragged_reads},
)
