"""``cholesky``, ``ldl`` and ``lu``, the dense factorization expression ops, with their derivatives
(``lu`` has none: ``linalg.solve`` differentiates implicitly), structural sparsity, verification
and loop lowering."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse

from ...ad.forward import is_zero_const, zeros_many
from ...ad.sparsity import incidence, mask_compose
from ...ir import program as p
from ...ir.expr import Expr, gather, register_op, zeros_like
from ...ir.program import ProgramNode, ProgramOp, RangeKind
from ...ir.spec import Rule
from ...ir.types import DType, TensorType, dtypes
from .trisolve import (
  DENSE_UNROLL,
  _columns_as_seeds,
  _entry,
  _seed_solve,
  _seed_transpose,
  _seeds_as_columns,
  _square,
  _unroll_attr,
  solve_triangular,
)

if TYPE_CHECKING:
  from ...passes.lowering import LowerCtx

CHOLESKY, LDL, LU = "cholesky", "ldl", "lu"

LU_NO_DERIVATIVE = (
  "the dense LU factorization has no derivative: linalg.solve(a, b, assume='gen') solves with it and "
  "differentiates implicitly, so its derivative never reaches the factorization"
)
"""Why an ``lu`` node refuses a nonzero tangent or cotangent."""


def cholesky(a: Any) -> Expr:
  """The lower Cholesky factor ``L`` of a symmetric positive definite matrix, ``A = L L^T``.

  Only the lower triangle of ``a`` is read; the upper triangle of the result is zero. No check is
  made: a matrix that is not positive definite gives NaN (a square root of a negative number).
  Differentiable, reading the derivative of the lower triangle as that of a symmetric matrix.
  """
  a = _square(a, "cholesky")
  return Expr(CHOLESKY, (a,), TensorType(a.shape, dtype=a.type.dtype, diff=a.type.diff), attrs=_unroll_attr(a.shape[0]), lowering=a.lowering)


def ldl(a: Any) -> Expr:
  """``A = L D L^T`` without pivoting, packed in one matrix: ``L`` (unit lower) below the diagonal,
  ``D`` on it, zeros above. For quasi-definite matrices (positive and negative definite diagonal
  blocks), where every leading pivot is nonzero; a zero pivot gives inf or NaN. Only the lower
  triangle of ``a`` is read."""
  a = _square(a, "ldl")
  return Expr(LDL, (a,), TensorType(a.shape, dtype=a.type.dtype, diff=a.type.diff), attrs=_unroll_attr(a.shape[0]), lowering=a.lowering)


def lu(a: Any) -> Expr:
  """``P A = L U`` with partial pivoting (the row of largest magnitude in each column), packed in one
  ``(n + 1, n)`` array: rows ``0 .. n-1`` hold ``L`` (unit lower) below the diagonal and ``U`` on and
  above it, and row ``n`` holds the permutation, ``perm[i]`` the row of ``A`` that became row ``i``,
  as a float. A singular matrix gives a zero pivot, and inf or NaN in what follows it.

  The factorization has no derivative of its own: ``linalg.solve(a, b, assume="gen")`` solves with
  it and differentiates implicitly, as ``SparseLDL.solve`` does."""
  a = _square(a, "lu")
  n = a.shape[0]
  return Expr(LU, (a,), TensorType((n + 1, n), dtype=a.type.dtype, diff=a.type.diff), attrs=_unroll_attr(n), lowering=a.lowering)


def _factor_shape(expr: Expr) -> str | None:
  a = expr.args[0]
  if len(a.shape) != 2 or a.shape[0] != a.shape[1] or expr.shape != a.shape:
    return f"{expr.op} needs a square matrix and keeps its shape, got {a.shape} -> {expr.shape}"
  return None


def _lu_shape(expr: Expr) -> str | None:
  a = expr.args[0]
  if len(a.shape) != 2 or a.shape[0] != a.shape[1] or expr.shape != (a.shape[0] + 1, a.shape[0]):
    return f"lu needs a square matrix and gives the factors and the permutation, (n + 1, n); got {a.shape} -> {expr.shape}"
  return None


def _lower_as_symmetric(d: Expr) -> Expr:
  """The symmetric matrix whose lower triangle is that of ``d``: what a factorization reads."""
  n, nd = d.shape[-1], len(d.shape)
  strict = d * Expr.const(np.tril(np.ones((n, n)), -1))
  return d * Expr.const(np.tril(np.ones((n, n)))) + strict.transpose((*range(nd - 2), nd - 1, nd - 2))


def _sandwich(t: Expr, s: Expr, *, unit: bool) -> Expr:
  """``L^{-1} S L^{-T}`` for a symmetric ``S`` and the lower triangle of ``t``, a factorization's
  result, whose choice between straight-line code and loops the solves keep."""
  unroll = bool(t.attrs["unroll"])
  z = solve_triangular(t, s, lower=True, unit_diagonal=unit, unroll=unroll)
  return solve_triangular(t, z.T, lower=True, unit_diagonal=unit, unroll=unroll).T


def factor_tangent(expr: Expr, d: Expr) -> Expr:
  """The tangent of ``cholesky`` or ``ldl`` along a tangent ``d`` of the matrix.

  ``A = L L^T``: with ``X = L^{-1} S L^{-T}``, ``dL = L (tril(X) - diag(X) / 2)``.
  ``A = L D L^T`` (packed ``F``): ``dD = diag(X)`` and ``dL = L stril(X) D^{-1}`` for the unit ``L``.
  ``S`` is the symmetric matrix whose lower triangle is that of ``d``."""
  n = expr.shape[0]
  s = _lower_as_symmetric(d)
  if expr.op == CHOLESKY:
    phi = np.tril(np.ones((n, n)))
    np.fill_diagonal(phi, 0.5)
    return expr @ (_sandwich(expr, s, unit=False) * Expr.const(phi))
  x = _sandwich(expr, s, unit=True)
  unit_l = expr * Expr.const(np.tril(np.ones((n, n)), -1)) + Expr.const(np.eye(n))
  inv_d = 1.0 / gather(expr.reshape((n * n,)), np.arange(n) * (n + 1))
  return (unit_l @ (x * Expr.const(np.tril(np.ones((n, n)), -1)))) * inv_d.reshape((1, n)) + x * Expr.const(np.eye(n))


def _jvp_factor(expr: Expr, d: list[Expr]) -> Expr:
  return zeros_like(expr) if is_zero_const(d[0]) else factor_tangent(expr, d[0])


def _seed_left(m: Expr, x: Expr) -> Expr:
  """``m @ x_s`` for every seed of a seeded matrix ``x``."""
  cols, layout = _seeds_as_columns(x)
  return _columns_as_seeds(m @ cols, layout, x.shape)


def _jvp_many_factor(expr: Expr, tan: Callable[[Expr], Expr], nseed: int) -> Expr:
  """Multi-seed tangents of ``cholesky`` and ``ldl``: the single-seed rule with every seed a column
  of one solve or product."""
  d = [tan(arg) for arg in expr.args]
  n = expr.shape[0]
  if is_zero_const(d[0]):
    return zeros_many(expr, nseed)
  s = _lower_as_symmetric(d[0])
  unit = expr.op == LDL
  unroll = bool(expr.attrs["unroll"])
  z = _seed_solve(expr, s, lower=True, unit_diagonal=unit, unroll=unroll)
  x = _seed_transpose(_seed_solve(expr, _seed_transpose(z), lower=True, unit_diagonal=unit, unroll=unroll))
  if expr.op == CHOLESKY:
    phi = np.tril(np.ones((n, n)))
    np.fill_diagonal(phi, 0.5)
    return _seed_left(expr, x * Expr.const(phi))
  unit_l = expr * Expr.const(np.tril(np.ones((n, n)), -1)) + Expr.const(np.eye(n))
  inv_d = 1.0 / gather(expr.reshape((n * n,)), np.arange(n) * (n + 1))
  return _seed_left(unit_l, x * Expr.const(np.tril(np.ones((n, n)), -1))) * inv_d.reshape((1, 1, n)) + x * Expr.const(np.eye(n))


def _factor_cotangent(expr: Expr, cot: Expr) -> Expr:
  """The cotangent of the matrix under ``cholesky`` or ``ldl`` (which read its lower triangle).

  With ``G = L^{-T} P L^{-1}``, the cotangent is ``tril(G) + stril(G^T)``, where
  ``P = Phi(L^T Lbar)`` (``tril`` with the diagonal halved) for ``L L^T``, and for ``L D L^T``
  ``P = stril(L^T stril(Fbar) D^{-1}) + diag(Fbar)`` with the unit ``L`` of the packed factor."""
  n = expr.shape[0]
  tril, stril, eye = np.tril(np.ones((n, n))), np.tril(np.ones((n, n)), -1), np.eye(n)
  if expr.op == CHOLESKY:
    phi = tril.copy()
    np.fill_diagonal(phi, 0.5)
    inner, unit = (expr.T @ cot) * Expr.const(phi), False
  else:
    unit_l = expr * Expr.const(stril) + Expr.const(eye)
    inv_d = 1.0 / gather(expr.reshape((n * n,)), np.arange(n) * (n + 1))
    inner = (unit_l.T @ ((cot * Expr.const(stril)) * inv_d.reshape((1, n)))) * Expr.const(stril) + cot * Expr.const(eye)
    unit = True
  unroll = bool(expr.attrs["unroll"])
  w = solve_triangular(expr, inner, lower=True, trans=True, unit_diagonal=unit, unroll=unroll)
  g = solve_triangular(expr, w.T, lower=True, trans=True, unit_diagonal=unit, unroll=unroll).T
  return g * Expr.const(tril) + (g * Expr.const(stril.T)).T


def _vjp_factor(expr: Expr, cot: Expr) -> tuple[Expr, ...]:
  return (_factor_cotangent(expr, cot),)


def _sparsity_factor(expr: Expr, mask: Callable[[Expr], sparse.csr_array], ncols: int) -> sparse.csr_array:
  # Every entry of the lower triangle of the factor may depend on every entry the factorization reads.
  a = expr.args[0]
  n = a.shape[0]
  lower = np.flatnonzero(np.tril(np.ones((n, n), dtype=bool)).reshape(-1))
  rows, cols = np.repeat(lower, lower.size), np.tile(lower, lower.size)
  return mask_compose(incidence((expr.size, a.size), rows, cols), mask(a))


def _sparsity_lu(expr: Expr, mask: Callable[[Expr], sparse.csr_array], ncols: int) -> sparse.csr_array:
  # Pivoting moves any row anywhere: every entry of the factors and the permutation may depend on every entry.
  a = expr.args[0]
  dense = incidence((expr.size, a.size), np.repeat(np.arange(expr.size), a.size), np.tile(np.arange(a.size), expr.size))
  return mask_compose(dense, mask(a))


def _unary_node(op: ProgramOp, x: ProgramNode) -> ProgramNode:
  return ProgramNode(op, (x,), dtype=x.dtype)


def _lower_factor(ctx: LowerCtx, node: Expr) -> None:
  """Row-by-row (Crout) ``L L^T`` or ``L D L^T``: entry ``(i, j)``, ``j <= i``, is the matrix entry
  minus a dot product of two rows already computed, both contiguous in row-major storage. For
  ``L D L^T`` the row being computed is kept scaled by ``D`` in a scratch vector, so every update
  is one multiply-add. Small orders are unrolled into straight-line code."""
  a = node.args[0]
  n = a.shape[0]
  src, out, dt = ctx.buf_of(a), ctx.alloc_tmp(node), node.type.dtype
  chol = node.op == CHOLESKY
  scaled = None if chol else ctx.new_private(dt, (n,))
  zero = p.const_float(0.0, dtype=dt)
  c = p.const_int

  def term(i: ProgramNode, j: ProgramNode, k: ProgramNode) -> ProgramNode:
    """The ``k`` term subtracted for entry ``(i, j)``."""
    if chol:
      return p.mul(p.load(_entry(out, n, i, k)), p.load(_entry(out, n, j, k)))
    assert scaled is not None
    return p.mul(p.load(p.view(scaled, [k])), p.load(_entry(out, n, j, k)))

  def finish(i: ProgramNode, j: ProgramNode, value: ProgramNode, diagonal: bool) -> list[ProgramNode]:
    if diagonal:
      return [p.store(_entry(out, n, i, i), _unary_node(ProgramOp.SQRT, value) if chol else value)]
    stores = [p.store(p.view(scaled, [j]), value)] if scaled is not None else []
    return [*stores, p.store(_entry(out, n, i, j), p.div(value, p.load(_entry(out, n, j, j))))]

  if node.attrs.get("unroll", n <= DENSE_UNROLL):
    for i in range(n):
      for j in range(i + 1):
        value = p.load(_entry(src, n, c(i), c(j)))
        for k in range(j):
          value = p.sub(value, term(c(i), c(j), c(k)))
        ctx.emit(*finish(c(i), c(j), value, i == j))
      ctx.emit(*(p.store(_entry(out, n, c(i), c(z)), zero) for z in range(i + 1, n)))
    return
  if chol:
    _cholesky_tiles(ctx, src, out, n, dt)
    return
  nm = out.attrs["name"]
  i, j, z = (p.var(f"{v}_{nm}") for v in ("fi", "fj", "fz"))
  off_sum, off_total = ctx.blocked_sum(f"o_{nm}", c(0), j, lambda kk: term(i, j, kk), dt)
  diag_sum, diag_total = ctx.blocked_sum(f"d_{nm}", c(0), i, lambda kk: term(i, i, kk), dt)
  off = [*off_sum, *finish(i, j, p.sub(p.load(_entry(src, n, i, j)), off_total), False)]
  diag = [*diag_sum, *finish(i, i, p.sub(p.load(_entry(src, n, i, i)), diag_total), True)]
  zeros = p.for_(p.range_(z.attrs["name"], p.add(i, c(1)), n, kind=RangeKind.GLOBAL), [p.store(_entry(out, n, i, z), zero)])
  row = [p.for_(p.range_(j.attrs["name"], 0, i, kind=RangeKind.SERIAL), off), *diag, zeros]
  ctx.emit(p.for_(p.range_(i.attrs["name"], 0, n, kind=RangeKind.SERIAL), row))


def _lower_lu(ctx: LowerCtx, node: Expr) -> None:
  """Right-looking ``P A = L U`` with partial pivoting, in place in the ``(n + 1, n)`` result: rows
  ``0 .. n-1`` start as ``A`` and row ``n`` as the identity permutation; column ``k`` finds the row
  of largest magnitude at or below the diagonal (the first on a tie, as LAPACK), swaps it with row
  ``k`` across every column and the permutation, divides the column below the pivot by it, and
  updates the trailing block.

  Small orders are straight-line code with every index a constant: the pivot row is a run-time
  value, so the swap reads and writes through selects on it (``n - k`` per entry of row ``k``, one
  per entry below it), which keeps every access at a fixed address for scalar expansion. Larger
  orders loop, and swap through loads and stores at the pivot row's run-time address."""
  a = node.args[0]
  n = a.shape[0]
  src, out, dt = ctx.buf_of(a), ctx.alloc_tmp(node), node.type.dtype
  c = p.const_int
  pivot, largest, keep = (p.view(ctx.new_private(t, ()), [c(0)]) for t in (dtypes.int64, dt, dt))

  def at(i: ProgramNode, j: ProgramNode) -> ProgramNode:
    return _entry(out, n, i, j)

  def magnitude(i: ProgramNode, k: ProgramNode) -> ProgramNode:
    return _unary_node(ProgramOp.ABS, p.load(at(i, k)))

  def search(k: ProgramNode, i: ProgramNode) -> list[ProgramNode]:
    """Row ``i`` becomes column ``k``'s pivot when strictly larger than the best so far."""
    value = magnitude(i, k)
    better = p.compare(ProgramOp.LT, p.load(largest), value)
    return [p.store(pivot, p.select(better, i, p.load(pivot))), p.store(largest, p.select(better, value, p.load(largest)))]

  def eliminate(k: ProgramNode, i: ProgramNode, j: ProgramNode) -> ProgramNode:
    return p.store(at(i, j), p.sub(p.load(at(i, j)), p.mul(p.load(at(i, k)), p.load(at(k, j)))))

  def scale(k: ProgramNode, i: ProgramNode) -> ProgramNode:
    return p.store(at(i, k), p.div(p.load(at(i, k)), p.load(at(k, k))))

  if node.attrs.get("unroll", n <= DENSE_UNROLL):
    for i in range(n):
      ctx.emit(*(p.store(at(c(i), c(j)), p.load(_entry(src, n, c(i), c(j)))) for j in range(n)))
    ctx.emit(*(p.store(at(c(n), c(j)), p.const_float(float(j), dtype=dt)) for j in range(n)))
    for k in range(n - 1):
      ctx.emit(p.store(pivot, c(k)), p.store(largest, magnitude(c(k), c(k))))
      for i in range(k + 1, n):
        ctx.emit(*search(c(k), c(i)))
      is_pivot = {i: p.compare(ProgramOp.EQ, p.load(pivot), c(i)) for i in range(k + 1, n)}
      for j in range(n + 1):  # every column, then (j = n) the permutation, entry i of row n
        cells = {i: at(c(i), c(j)) if j < n else p.view(out, [c(n * n + i)]) for i in range(k, n)}
        chosen = p.load(cells[k])
        for i in range(k + 1, n):
          chosen = p.select(is_pivot[i], p.load(cells[i]), chosen)
        ctx.emit(p.store(keep, p.load(cells[k])), p.store(cells[k], chosen))
        ctx.emit(*(p.store(cells[i], p.select(is_pivot[i], p.load(keep), p.load(cells[i]))) for i in range(k + 1, n)))
      for i in range(k + 1, n):
        ctx.emit(scale(c(k), c(i)))
        ctx.emit(*(eliminate(c(k), c(i), c(j)) for j in range(k + 1, n)))
    return
  nm = out.attrs["name"]
  i, j, k = (p.var(f"{v}_{nm}") for v in ("ui", "uj", "uk"))
  copy = p.for_(p.range_(j.attrs["name"], 0, n, kind=RangeKind.GLOBAL), [p.store(at(i, j), p.load(_entry(src, n, i, j)))])
  ctx.emit(p.for_(p.range_(i.attrs["name"], 0, n, kind=RangeKind.GLOBAL), [copy]))
  ctx.emit(p.for_(p.range_(j.attrs["name"], 0, n, kind=RangeKind.GLOBAL), [p.store(at(c(n), j), p.cast(j, dt))]))

  def swap(cell_k: ProgramNode, cell_p: ProgramNode) -> list[ProgramNode]:
    return [p.store(keep, p.load(cell_k)), p.store(cell_k, p.load(cell_p)), p.store(cell_p, p.load(keep))]

  row = p.load(pivot)
  update = p.for_(p.range_(j.attrs["name"], p.add(k, c(1)), n, kind=RangeKind.GLOBAL), [eliminate(k, i, j)])
  column = [
    p.store(pivot, k),
    p.store(largest, magnitude(k, k)),
    p.for_(p.range_(i.attrs["name"], p.add(k, c(1)), n), search(k, i)),
    p.for_(p.range_(j.attrs["name"], 0, n), swap(at(k, j), at(row, j))),
    *swap(at(c(n), k), at(c(n), row)),
    p.for_(p.range_(i.attrs["name"], p.add(k, c(1)), n, kind=RangeKind.GLOBAL), [scale(k, i), update]),
  ]
  ctx.emit(p.for_(p.range_(k.attrs["name"], 0, n - 1), column))


CHOLESKY_TILE = 4
"""The side of the register tiles of the looped dense Cholesky."""


def _cholesky_tiles(ctx: LowerCtx, src: ProgramNode, out: ProgramNode, n: int, dt: DType) -> None:
  """Crout ``L L^T`` by square tiles of ``CHOLESKY_TILE`` rows and columns of ``L``, in row-major
  order: a tile's dot products over the columns left of it run together, each row and column
  entry loaded once for the whole tile and the sums kept in registers (16 multiply-adds for 8
  loads, where one entry at a time takes two loads each); then the terms inside the tile, column
  by column. A tile needs the rows above it and the tiles left of it, which the order provides.
  The last rows and columns form narrower tiles when ``n`` is not a multiple of the side.

  Each dot product is four partial sums over contiguous quarters of its columns, added pairwise,
  as accurate as the four interleaved sums of the entry-at-a-time kernel. One running sum per
  entry is faster still, but late in an interior-point solve the condensed matrix's last pivots
  are rounding noise, and its larger error let one Maros-Meszaros problem (QSHARE1B) fail a
  factorization and double its iterations."""
  c = p.const_int
  nm = out.attrs["name"]
  side = CHOLESKY_TILE
  full, rest = divmod(n, side)
  zero = p.const_float(0.0, dtype=dt)

  def scalars() -> list[list[ProgramNode]]:
    return [[p.view(ctx.new_private(dt, ()), [c(0)]) for _ in range(side)] for _ in range(side)]

  sums, parts = scalars(), [scalars() for _ in range(side)]

  def tile(bi: ProgramNode, bj: ProgramNode, rows: int, cols: int, diagonal: bool) -> list[ProgramNode]:
    """The tile at block row ``bi`` and block column ``bj``: rows ``side * bi`` on, columns ``side * bj`` on."""
    i0, j0 = p.mul(bi, c(side)), p.mul(bj, c(side))
    pairs = [(a, b) for a in range(rows) for b in range(cols) if not diagonal or a >= b]
    stmts: list[ProgramNode] = []
    for quarter in range(side):  # the dot products over columns [quarter * bj, (quarter + 1) * bj)
      k = p.var(f"tk{ctx.fresh_id()}_{nm}")
      dots = [
        p.store(sums[a][b], p.add(p.load(sums[a][b]), p.mul(p.load(_entry(out, n, p.add(i0, c(a)), k)), p.load(_entry(out, n, p.add(j0, c(b)), k)))))
        for a, b in pairs
      ]
      stmts += [p.store(sums[a][b], zero) for a, b in pairs]
      stmts.append(p.for_(p.range_(k.attrs["name"], p.mul(bj, c(quarter)), p.mul(bj, c(quarter + 1))), dots))
      stmts += [p.store(parts[quarter][a][b], p.load(sums[a][b])) for a, b in pairs]
    for b in range(cols):
      j = p.add(j0, c(b))
      for a in range(rows):
        if diagonal and a < b:
          continue
        i = p.add(i0, c(a))
        dot = p.add(p.add(p.load(parts[0][a][b]), p.load(parts[1][a][b])), p.add(p.load(parts[2][a][b]), p.load(parts[3][a][b])))
        value = p.sub(p.load(_entry(src, n, i, j)), dot)
        for inner in range(b):
          value = p.sub(value, p.mul(p.load(_entry(out, n, i, p.add(j0, c(inner)))), p.load(_entry(out, n, j, p.add(j0, c(inner))))))
        if diagonal and a == b:
          stmts.append(p.store(_entry(out, n, i, i), _unary_node(ProgramOp.SQRT, value)))
        else:
          stmts.append(p.store(_entry(out, n, i, j), p.div(value, p.load(_entry(out, n, j, j)))))
    return stmts

  bi, bj = p.var(f"tbi_{nm}"), p.var(f"tbj_{nm}")
  if full:
    left = p.for_(p.range_(bj.attrs["name"], 0, bi), tile(bi, bj, side, side, False))
    ctx.emit(p.for_(p.range_(bi.attrs["name"], 0, full), [left, *tile(bi, bi, side, side, True)]))
  if rest:
    if full:
      ctx.emit(p.for_(p.range_(bj.attrs["name"], 0, full), tile(c(full), bj, rest, side, False)))
    ctx.emit(*tile(c(full), c(full), rest, rest, True))
  zi, zj = p.var(f"tzi_{nm}"), p.var(f"tzj_{nm}")
  upper = p.for_(p.range_(zj.attrs["name"], p.add(zi, c(1)), n, kind=RangeKind.GLOBAL), [p.store(_entry(out, n, zi, zj), zero)])
  ctx.emit(p.for_(p.range_(zi.attrs["name"], 0, n, kind=RangeKind.SERIAL), [upper]))


def _refuse_lu(*_: Any) -> Any:
  raise NotImplementedError(LU_NO_DERIVATIVE)


# cholesky and ldl share every rule; each tells the two apart by the node's op.
for _op in (CHOLESKY, LDL):
  register_op(
    _op,
    arity=1,
    jvp=_jvp_factor,
    jvp_many=_jvp_many_factor,
    vjp=_vjp_factor,
    sparsity=_sparsity_factor,
    verify=(Rule(_op, "factor-shape", _factor_shape),),
    lower=_lower_factor,
  )
register_op(
  LU,
  arity=1,
  jvp=_refuse_lu,
  jvp_many=_refuse_lu,
  vjp=_refuse_lu,
  sparsity=_sparsity_lu,
  verify=(Rule(LU, "lu-shape", _lu_shape),),
  lower=_lower_lu,
)
