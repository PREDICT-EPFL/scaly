"""``solve_triangular``, the triangular-solve expression op, with its derivatives, structural
sparsity, verification and loop lowering; and what the dense factorizations share with it."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import sparse

from ...ad.forward import is_zero_const, zeros_many
from ...ad.sparsity import incidence, mask_compose, mask_or
from ...ir import program as p
from ...ir.expr import Expr, as_expr, common_lowering, diff_any, promote_dtype, register_op, zeros_like
from ...ir.program import ProgramNode, RangeKind
from ...ir.spec import Rule
from ...ir.types import DType, TensorType
from ...utils.options import get_options

if TYPE_CHECKING:
  from ...passes.lowering import LowerCtx

TRISOLVE = "trisolve"

type Unroll = bool | Literal["auto"]
"""A factorization's or triangular solve's choice between straight-line code and loops: fixed when
its graph is built (``sc.options(linalg=dict(dense_unroll=...))``), or ``"auto"``, made when it is
lowered, for the target, by the size of its straight-line body (``straight_line``)."""


def _square(a: Any, what: str) -> Expr:
  a = as_expr(a)
  if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
    raise ValueError(f"{what} needs a square matrix, got shape {a.shape}")
  if not a.type.dtype.is_floating:
    raise TypeError(f"{what} needs a floating-point matrix, got {a.type.dtype}")
  return a


def _unroll_attr(n: int, unroll: Unroll | None = None) -> dict[str, Unroll]:
  """Whether a dense factorization or solve of order ``n`` becomes straight-line code: ``unroll``
  when given, else what the ``linalg`` option in force when the graph is built says, orders up to
  ``dense_unroll`` straight-line and larger ones loops, or, with the option unset, ``"auto"``. A
  derivative passes the choice of the node it differentiates, so that no option in force when it is
  built changes it."""
  if unroll is None:
    limit = get_options().namespace("linalg").dense_unroll
    return {"unroll": "auto" if limit is None else n <= limit}
  return {"unroll": "auto" if unroll == "auto" else bool(unroll)}


def straight_line(ctx: LowerCtx, node: Expr, ops: int) -> bool:
  """Whether ``node`` lowers to straight-line code: the choice its graph recorded, or, for
  ``"auto"``, whether its straight-line body of ``ops`` operations is under the target's
  ``straight_line_ops``."""
  choice = node.attrs.get("unroll", "auto")
  return ops < ctx.target.straight_line_ops if choice == "auto" else bool(choice)


def _entry(buf: ProgramNode, n: int, i: ProgramNode, j: ProgramNode) -> ProgramNode:
  """The view of entry ``(i, j)`` of a row-major matrix with ``n`` columns."""
  return p.view(buf, [p.add(p.mul(i, p.const_int(n)), j)])


def _tri_mask(n: int, lower: bool, unit: bool) -> Expr:
  """Ones on the triangle a triangular solve reads (its diagonal too unless unit), zeros elsewhere."""
  mask = np.tril(np.ones((n, n))) if lower else np.triu(np.ones((n, n)))
  if unit:
    np.fill_diagonal(mask, 0.0)
  return Expr.const(mask)


def solve_triangular(t: Any, b: Any, *, lower: bool = True, trans: bool = False, unit_diagonal: bool = False, unroll: Unroll | None = None) -> Expr:
  """``X`` with ``op(T) X = B``, ``op(T) = T`` or ``T^T``, for a triangular ``T``; ``B`` a vector or a
  matrix of right-hand sides. Only the triangle named by ``lower`` is read, and its diagonal only
  when ``unit_diagonal`` is false. ``unroll`` overrides the ``linalg`` option's choice between
  straight-line code (True), loops (False) and the target's choice (``"auto"``), as a derivative
  does to keep its factorization's."""
  t = _square(t, "solve_triangular")
  b = as_expr(b)
  if len(b.shape) not in (1, 2) or b.shape[0] != t.shape[0]:
    raise ValueError(f"solve_triangular with a {t.shape} matrix needs a right-hand side of {t.shape[0]} rows, got shape {b.shape}")
  return Expr(
    TRISOLVE,
    (t, b),
    TensorType(b.shape, dtype=promote_dtype(t, b), diff=diff_any(t, b)),
    attrs={"lower": bool(lower), "trans": bool(trans), "unit": bool(unit_diagonal), **_unroll_attr(t.shape[0], unroll)},
    lowering=common_lowering(t, b),
  )


def _trisolve_shapes(expr: Expr) -> str | None:
  t, b = expr.args
  if len(t.shape) != 2 or t.shape[0] != t.shape[1] or len(b.shape) not in (1, 2) or b.shape[0] != t.shape[0] or expr.shape != b.shape:
    return f"TRISOLVE of {t.shape} by {b.shape} -> {expr.shape} is inconsistent"
  if any(k not in expr.attrs for k in ("lower", "trans", "unit")):
    return "TRISOLVE needs 'lower', 'trans' and 'unit' attrs"
  return None


def trisolve_tangent(expr: Expr, dt: Expr | None, db: Expr | None) -> Expr:
  """``op(T) dX = dB - op(dT) X`` with ``dT`` restricted to the triangle the solve reads."""
  t, _ = expr.args
  lower, trans, unit = (bool(expr.attrs[k]) for k in ("lower", "trans", "unit"))
  rhs = db
  if dt is not None:
    masked = dt * _tri_mask(t.shape[0], lower, unit)
    term = (masked.T if trans else masked) @ expr
    rhs = -term if rhs is None else rhs - term
  assert rhs is not None
  return solve_triangular(t, rhs, lower=lower, trans=trans, unit_diagonal=unit, unroll=expr.attrs["unroll"])


def _jvp_trisolve(expr: Expr, d: list[Expr]) -> Expr:
  dt, db = (None if is_zero_const(t) else t for t in d)
  return zeros_like(expr) if dt is None and db is None else trisolve_tangent(expr, dt, db)


def _seeds_as_columns(x: Expr) -> tuple[Expr, tuple[int, ...]]:
  """A seeded matrix operand ``(nseed, n[, m])`` as one matrix ``(n, nseed * m)``, with what undoes it."""
  nseed, n = x.shape[0], x.shape[1]
  m = 1 if len(x.shape) == 2 else x.shape[2]
  cols = x.reshape((nseed, n, m)).transpose((1, 0, 2)).reshape((n, nseed * m))
  return cols, (nseed, n, m)


def _columns_as_seeds(cols: Expr, layout: tuple[int, ...], shape: tuple[int, ...]) -> Expr:
  nseed, n, m = layout
  return cols.reshape((n, nseed, m)).transpose((1, 0, 2)).reshape(shape)


def _seed_solve(t: Expr, rhs: Expr, **flags: Any) -> Expr:
  """One triangular solve for every seed: the seeds become right-hand-side columns."""
  cols, layout = _seeds_as_columns(rhs)
  return _columns_as_seeds(solve_triangular(t, cols, **flags), layout, rhs.shape)


def _seed_transpose(x: Expr) -> Expr:
  return x.transpose((0, 2, 1))


def _jvp_many_trisolve(expr: Expr, tan: Callable[[Expr], Expr], nseed: int) -> Expr:
  """Multi-seed tangents: the single-seed rule with every seed a column of one solve or product."""
  d = [tan(arg) for arg in expr.args]
  n = expr.shape[0]
  t, _ = expr.args
  lower, trans, unit = (bool(expr.attrs[k]) for k in ("lower", "trans", "unit"))
  rhs = None if is_zero_const(d[1]) else d[1]
  if not is_zero_const(d[0]):
    masked = d[0] * _tri_mask(n, lower, unit)
    op = _seed_transpose(masked) if trans else masked
    x = expr if len(expr.shape) == 2 else expr.reshape((n, 1))
    term = (op.reshape((nseed * n, n)) @ x).reshape((nseed, *expr.shape))
    rhs = -term if rhs is None else rhs - term
  if rhs is None:
    return zeros_many(expr, nseed)
  return _seed_solve(t, rhs, lower=lower, trans=trans, unit_diagonal=unit, unroll=expr.attrs["unroll"])


def _vjp_trisolve(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  args = expr.args
  t, b = args
  lower, trans, unit = (bool(expr.attrs[k]) for k in ("lower", "trans", "unit"))
  b_bar = solve_triangular(t, cot, lower=lower, trans=not trans, unit_diagonal=unit, unroll=expr.attrs["unroll"])
  x2, bb2 = (expr.reshape((expr.size, 1)), b_bar.reshape((b_bar.size, 1))) if len(expr.shape) == 1 else (expr, b_bar)
  outer = x2 @ bb2.T if trans else bb2 @ x2.T
  return (-(outer * _tri_mask(t.shape[0], lower, unit)), b_bar)


def _sparsity_trisolve(expr: Expr, mask: Callable[[Expr], sparse.csr_array], ncols: int) -> sparse.csr_array:
  # Column c of the solution may depend on all of column c of the right-hand side and on every
  # entry of the triangle the solve reads.
  t, b = expr.args
  n = t.shape[0]
  m = 1 if len(b.shape) == 1 else b.shape[1]
  tri = np.tril(np.ones((n, n), dtype=bool)) if expr.attrs["lower"] else np.triu(np.ones((n, n), dtype=bool))
  if expr.attrs["unit"]:
    np.fill_diagonal(tri, False)
  read = np.flatnonzero(tri.reshape(-1))
  t_rows, t_cols = np.repeat(np.arange(expr.size), read.size), np.tile(read, expr.size)
  r, r2, col = np.meshgrid(np.arange(n), np.arange(n), np.arange(m), indexing="ij")
  b_rows, b_cols = (r * m + col).reshape(-1), (r2 * m + col).reshape(-1)
  from_t = mask_compose(incidence((expr.size, t.size), t_rows, t_cols), mask(t))
  return mask_or(from_t, mask_compose(incidence((expr.size, b.size), b_rows, b_cols), mask(b)))


def _lower_trisolve(ctx: LowerCtx, node: Expr) -> None:
  """Substitution reading the triangle by rows. ``T X = B`` takes each unknown as its right-hand side
  minus a dot product with the unknowns already found (row ``i`` of ``T``, contiguous); ``T^T X = B``
  sweeps columns of ``T^T``, which are rows of ``T``: each unknown, once found, is subtracted from
  the right-hand sides still open. A matrix of right-hand sides does each step for a whole row of
  ``X``, contiguous. Small orders are unrolled."""
  t, b = node.args
  n = t.shape[0]
  m = 1 if len(b.shape) == 1 else b.shape[1]
  tb, bb, out, dt = ctx.buf_of(t), ctx.buf_of(b), ctx.alloc_tmp(node), node.type.dtype
  lower, trans, unit = (bool(node.attrs[key]) for key in ("lower", "trans", "unit"))
  c = p.const_int
  nm = out.attrs["name"]

  def x(row: ProgramNode, cc: ProgramNode) -> ProgramNode:
    return p.view(out, [p.add(p.mul(row, c(m)), cc) if m > 1 else row])

  def rhs(row: ProgramNode, cc: ProgramNode) -> ProgramNode:
    return p.view(bb, [p.add(p.mul(row, c(m)), cc) if m > 1 else row])

  def per_col(body: Any) -> list[ProgramNode]:
    """``body(column)`` for every right-hand side: one statement for a vector, a loop otherwise."""
    if m == 1:
      return body(c(0))
    name = f"sc_{nm}_{ctx.fresh_id()}"
    return [p.for_(p.range_(name, 0, m, kind=RangeKind.GLOBAL), body(p.var(name)))]

  def scale(i: ProgramNode) -> list[ProgramNode]:
    if unit:
      return []
    return per_col(lambda cc: [p.store(x(i, cc), p.div(p.load(x(i, cc)), p.load(_entry(tb, n, i, i))))])

  # ``order(s)`` is the row handled at step ``s``; ``others(i)`` the range of the other index.
  forward = lower != trans
  row_at = (lambda s: s) if forward else (lambda s: p.sub(c(n - 1), s))  # noqa: E731
  # The body loops over the right-hand sides, but scalar expansion can unroll that loop too.
  if straight_line(ctx, node, n * n * m):
    steps = range(n) if forward else range(n - 1, -1, -1)
    if trans:
      ctx.emit(ctx.copy_loop(bb, out, b.shape))
      for r in steps:
        ctx.emit(*scale(c(r)))
        for k in range(r) if lower else range(r + 1, n):
          ctx.emit(
            *per_col(
              lambda cc, r=r, k=k: [p.store(x(c(k), cc), p.sub(p.load(x(c(k), cc)), p.mul(p.load(_entry(tb, n, c(r), c(k))), p.load(x(c(r), cc)))))]
            )
          )
      return
    for r in steps:

      def solve_row(cc: ProgramNode, r: int = r) -> list[ProgramNode]:
        value = p.load(rhs(c(r), cc))
        for k in range(r) if lower else range(r + 1, n):
          value = p.sub(value, p.mul(p.load(_entry(tb, n, c(r), c(k))), p.load(x(c(k), cc))))
        if not unit:
          value = p.div(value, p.load(_entry(tb, n, c(r), c(r))))
        return [p.store(x(c(r), cc), value)]

      ctx.emit(*per_col(solve_row))
    return
  rows, width = ctx.target.product_tile
  if rows > 1 and width % 4 == 0 and width % rows == 0 and n >= 4 * width and m >= width:
    _trisolve_blocked(ctx, tb, bb, out, n, m, lower, trans, unit, dt)
    return
  s, k = p.var(f"ss_{nm}"), p.var(f"sk_{nm}")
  i = row_at(s)
  k_rng = (
    (lambda kind: p.range_(k.attrs["name"], 0, i, kind=kind)) if lower else (lambda kind: p.range_(k.attrs["name"], p.add(i, c(1)), n, kind=kind))
  )  # noqa: E731
  if trans:
    ctx.emit(ctx.copy_loop(bb, out, b.shape))
    sweep = per_col(lambda cc: [p.store(x(k, cc), p.sub(p.load(x(k, cc)), p.mul(p.load(_entry(tb, n, i, k)), p.load(x(i, cc)))))])
    body = [*scale(i), p.for_(k_rng(RangeKind.GLOBAL), sweep)]
  elif m == 1:
    lo, hi = (c(0), i) if lower else (p.add(i, c(1)), c(n))
    sums, total = ctx.blocked_sum(f"t_{nm}", lo, hi, lambda kk: p.mul(p.load(_entry(tb, n, i, kk)), p.load(x(kk, c(0)))), dt)
    value = p.sub(p.load(rhs(i, c(0))), total)
    body = [*sums, p.store(x(i, c(0)), value if unit else p.div(value, p.load(_entry(tb, n, i, i))))]
  else:
    # Four rows of ``X`` per pass over the row being solved: a quarter of its loads and stores.
    lo, hi = (c(0), i) if lower else (p.add(i, c(1)), c(n))
    tail = p.sub(hi, p.mod(p.sub(hi, lo), c(4)))
    kb = p.var(f"kb_{nm}")

    def four(cc: ProgramNode) -> list[ProgramNode]:
      terms = [p.mul(p.load(_entry(tb, n, i, p.add(kb, c(q)))), p.load(x(p.add(kb, c(q)), cc))) for q in range(4)]
      return [p.store(x(i, cc), p.sub(p.load(x(i, cc)), p.add(p.add(terms[0], terms[1]), p.add(terms[2], terms[3]))))]

    def one(cc: ProgramNode) -> list[ProgramNode]:
      return [p.store(x(i, cc), p.sub(p.load(x(i, cc)), p.mul(p.load(_entry(tb, n, i, k)), p.load(x(k, cc)))))]

    body = [
      *per_col(lambda cc: [p.store(x(i, cc), p.load(rhs(i, cc)))]),
      p.for_(p.range_(kb.attrs["name"], lo, tail, step=4, kind=RangeKind.SERIAL), per_col(four)),
      p.for_(p.range_(k.attrs["name"], tail, hi, kind=RangeKind.SERIAL), per_col(one)),
      *scale(i),
    ]
  ctx.emit(p.for_(p.range_(s.attrs["name"], 0, n, kind=RangeKind.SERIAL), body))


def _trisolve_blocked(
  ctx: LowerCtx, tb: ProgramNode, bb: ProgramNode, out: ProgramNode, n: int, m: int, lower: bool, trans: bool, unit: bool, dt: DType
) -> None:
  """Substitution in blocks of the register tile's width of unknowns (``Target.product_tile``, 8 on
  the reference machine), for ``m`` right-hand sides of a tile's width or more. For each block, in
  the order the substitution needs: subtract from its right-hand sides the product of its rows of
  the triangle with the unknowns already found, register tiles over the right-hand sides; then solve
  the block itself row by row, dividing by its diagonal. The rows of ``T^T``
  are read down the columns of ``T``, one entry at a time, which is how a tile reads its left
  operand anyway. As in the blocked Cholesky, each unknown's dot product runs in four quarters of
  the unknowns before its block, each summed from zero and added pairwise before the right-hand
  side subtracts it."""
  c = p.const_int
  nm = out.attrs["name"]
  rows, width = ctx.target.product_tile
  full, tail = divmod(n, width)
  forward = lower != trans
  upper = ctx.new_private(dt, (n * m,))  # the last two quarters of the dot products
  segments = ctx.tile_segments(m)

  def coef(i: ProgramNode, k: ProgramNode) -> ProgramNode:
    """Entry ``(i, k)`` of the triangle solved against: ``T``'s, or ``T^T``'s."""
    return p.load(_entry(tb, n, k, i) if trans else _entry(tb, n, i, k))

  def x(row: ProgramNode, col: ProgramNode) -> ProgramNode:
    return p.view(out, [p.add(p.mul(row, c(m)), col)])

  def block(first: ProgramNode, w: int, solved: ProgramNode, quarter: ProgramNode, tag: str) -> list[ProgramNode]:
    """Rows ``first`` to ``first + w``, whose solved unknowns are the ``4 * quarter`` from ``solved``."""
    stmts: list[ProgramNode] = []
    # Each unknown's dot product in four quarters, added as the Cholesky adds them: the first two into
    # the block's own unknowns, the last two into ``upper``, then the right-hand side minus both.
    for part in range(4):
      offset = p.add(solved, p.mul(quarter, c(part)))
      into = out if part < 2 else upper

      for number, (start, cols, count) in enumerate(segments):
        jb = p.var(f"tj{tag}{part}{number}_{nm}")
        left = p.add(c(start), p.mul(jb, c(cols))) if count > 1 else c(start)

        def term(row: ProgramNode, step: ProgramNode, col: ProgramNode, offset: ProgramNode = offset, left: ProgramNode = left) -> ProgramNode:
          k = p.add(offset, step)
          return p.mul(coef(row, k), p.load(x(k, p.add(left, col))))

        def out_at(row: ProgramNode, col: ProgramNode, left: ProgramNode = left, into: ProgramNode = into) -> ProgramNode:
          return p.view(into, [p.add(p.mul(row, c(m)), p.add(left, col))])

        def finish(row: ProgramNode, col: ProgramNode, total: ProgramNode, out_at: Any = out_at) -> ProgramNode:
          return p.add(p.load(out_at(row, col)), total)

        done = None if part % 2 == 0 else finish

        ib, it = p.var(f"ti{tag}{part}{number}_{nm}"), p.var(f"tl{tag}{part}{number}_{nm}")
        tile_rows = [p.add(first, p.add(p.mul(ib, c(rows)), c(r))) for r in range(rows)]
        tiles = [
          p.for_(
            p.range_(ib.attrs["name"], 0, w // rows, kind=RangeKind.GLOBAL),
            ctx.tile(f"{nm}_{tag}{part}{number}t", tile_rows, cols, quarter, term, out_at, dt, done),
          )
        ]
        if w % rows:  # the rows a tile leaves, one at a time
          tiles.append(
            p.for_(
              p.range_(it.attrs["name"], p.add(first, c(w - w % rows)), p.add(first, c(w)), kind=RangeKind.GLOBAL),
              ctx.tile(f"{nm}_{tag}{part}{number}l", [it], cols, quarter, term, out_at, dt, done),
            )
          )
        stmts += [p.for_(p.range_(jb.attrs["name"], 0, count, kind=RangeKind.GLOBAL), tiles)] if count > 1 else tiles
    r, col = p.var(f"tr{tag}_{nm}"), p.var(f"tq{tag}_{nm}")
    at = p.add(p.mul(r, c(m)), col)
    total = p.add(p.load(p.view(out, [at])), p.load(p.view(upper, [at])))
    stmts.append(
      p.for_(
        p.range_(r.attrs["name"], first, p.add(first, c(w)), kind=RangeKind.GLOBAL),
        [p.for_(p.range_(col.attrs["name"], 0, m, kind=RangeKind.GLOBAL), [p.store(p.view(out, [at]), p.sub(p.load(p.view(bb, [at])), total))])],
      )
    )
    # The block itself, row by row in the substitution's order.
    order = list(range(w)) if forward else list(range(w - 1, -1, -1))
    for place, a in enumerate(order):
      i = p.add(first, c(a))
      col = p.var(f"tc{tag}{a}_{nm}")
      value = p.load(x(i, col))
      for r in order[:place]:
        value = p.sub(value, p.mul(coef(i, p.add(first, c(r))), p.load(x(p.add(first, c(r)), col))))
      if not unit:
        value = p.div(value, coef(i, i))
      stmts.append(p.for_(p.range_(col.attrs["name"], 0, m, kind=RangeKind.GLOBAL), [p.store(x(i, col), value)]))
    return stmts

  # Forward, block ``j`` is rows ``j * width`` on and the unknowns before them; backward, it is the
  # rows ``width`` above ``n - j * width`` and the unknowns from there down. The tail comes last.
  if full:
    jb = p.var(f"tb_{nm}")
    first = p.mul(jb, c(width)) if forward else p.sub(c(n - width), p.mul(jb, c(width)))
    solved = c(0) if forward else p.sub(c(n), p.mul(jb, c(width)))
    ctx.emit(p.for_(p.range_(jb.attrs["name"], 0, full, kind=RangeKind.SERIAL), block(first, width, solved, p.mul(jb, c(width // 4)), "f")))
  if tail:
    first, solved = (c(full * width), c(0)) if forward else (c(0), c(tail))
    ctx.emit(*block(first, tail, solved, c(full * width // 4), "e"))


register_op(
  TRISOLVE,
  arity=2,
  jvp=_jvp_trisolve,
  jvp_many=_jvp_many_trisolve,
  vjp=_vjp_trisolve,
  sparsity=_sparsity_trisolve,
  verify=(Rule(TRISOLVE, "trisolve-shapes", _trisolve_shapes),),
  lower=_lower_trisolve,
)
