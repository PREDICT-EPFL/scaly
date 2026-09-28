"""Dense factorizations and triangular solves as expression ops, lowered to plain loops.

``cholesky``, ``ldl``, ``lu`` and ``solve_triangular`` are expression ops (``ir/expr.py``) with their
own loop lowering and structural sparsity, and all but ``lu`` with derivatives in both modes; this
module adds the solves built from them. Orders up to ``DENSE_UNROLL`` become straight-line code.
Nothing calls an external library: the loops are generated C like everything else.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ..function.model import ConcreteFunction
from ..function.sugar import custom_derivative, vmap
from ..ir.expr import Expr, as_expr, cast, cholesky, gather, ldl, lu, put, solve_triangular, take
from ..ir.types import DType, dtypes
from ..utils.options import LinalgOptions, get_options

__all__ = ["cho_solve", "cholesky", "ldl", "ldl_solve", "ldl_unpack", "lu", "lu_solve", "solve", "solve_triangular"]


def ldl_unpack(f: Any) -> tuple[Expr, Expr]:
  """``(L, d)`` from a packed ``ldl`` factor: the unit lower ``L`` and the diagonal of ``D``."""
  f = as_expr(f)
  n, dtype = f.shape[0], f.type.dtype
  unit_l = f * Expr.const(np.tril(np.ones((n, n)), -1), dtype=dtype) + Expr.const(np.eye(n), dtype=dtype)
  return unit_l, gather(f.reshape((n * n,)), np.arange(n) * (n + 1))


def cho_solve(factor: Any, b: Any) -> Expr:
  """``A^{-1} b`` from the Cholesky factor ``L`` of ``A``: two triangular solves."""
  return solve_triangular(factor, solve_triangular(factor, b, lower=True), lower=True, trans=True)


def ldl_solve(f: Any, b: Any) -> Expr:
  """``A^{-1} b`` from a packed ``ldl`` factor: a unit lower solve, a diagonal scaling and a unit upper solve."""
  f, b = as_expr(f), as_expr(b)
  _, d = ldl_unpack(f)
  y = solve_triangular(f, b, lower=True, unit_diagonal=True)
  y = y / (d if len(b.shape) == 1 else d.reshape((d.size, 1)))
  return solve_triangular(f, y, lower=True, trans=True, unit_diagonal=True)


def lu_solve(f: Any, b: Any, *, trans: bool = False) -> Expr:
  """``A^{-1} b``, or ``A^{-T} b`` with ``trans``, from ``f = lu(A)``; ``b`` a vector or a matrix of
  right-hand sides. Differentiable in ``b``; in ``f`` only through the factorization, which has no
  derivative, so differentiate in ``A`` through ``solve(a, b, assume="gen")``."""
  f, b = as_expr(f), as_expr(b)
  n = f.shape[1] if len(f.shape) == 2 else -1
  if f.shape != (n + 1, n):
    raise ValueError(f"lu_solve needs a factor from lu, shaped (n + 1, n), got {f.shape}")
  if len(b.shape) not in (1, 2) or b.shape[0] != n:
    raise ValueError(f"lu_solve with a factor of order {n} needs a right-hand side of {n} rows, got shape {b.shape}")
  factors, perm = f[:n], cast(f[n], dtypes.int64)
  if not trans:  # L U x = P b
    pb = take(b, perm, in_range=True) if len(b.shape) == 1 else take(b.T, perm, in_range=True).T
    return solve_triangular(factors, solve_triangular(factors, pb, lower=True, unit_diagonal=True), lower=False)
  # U^T L^T P x = b: two transposed sweeps, then x = P^T w, entry perm[i] of x being w[i].
  w = solve_triangular(factors, solve_triangular(factors, b, lower=False, trans=True), lower=True, trans=True, unit_diagonal=True)
  if len(b.shape) == 1:
    return put(Expr.const(np.zeros(n), dtype=w.type.dtype), perm, w, in_range=True)
  return put(Expr.const(np.zeros(b.shape[::-1]), dtype=w.type.dtype), perm, w.T, in_range=True).T


def solve(a: Any, b: Any, *, assume: Literal["pos", "sym", "gen"] = "pos") -> Expr:
  """``A^{-1} b``: by Cholesky for a symmetric positive definite matrix (``assume="pos"``), by
  ``L D L^T`` without pivoting for a symmetric quasi-definite one (``assume="sym"``), both reading
  the lower triangle, or by ``lu`` with partial pivoting for any nonsingular matrix
  (``assume="gen"``).

  The general solve differentiates implicitly, ``dx = A^{-1} (db - dA x)``, and in reverse one
  transposed solve and an outer product, all with the one factorization; its second derivatives are
  implicit too."""
  if assume == "pos":
    return cho_solve(cholesky(a), b)
  if assume == "sym":
    return ldl_solve(ldl(a), b)
  if assume != "gen":
    raise ValueError(f"assume must be 'pos', 'sym' or 'gen', got {assume!r}")
  a, b = as_expr(a), as_expr(b)
  if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
    raise ValueError(f"solve needs a square matrix, got shape {a.shape}")
  n = a.shape[0]
  if len(b.shape) not in (1, 2) or b.shape[0] != n:
    raise ValueError(f"solve with a {a.shape} matrix needs a right-hand side of {n} rows, got shape {b.shape}")
  f = lu(a)
  fn = _general_solvers(n, a.type.dtype)[0]
  if len(b.shape) == 1:
    return _call(fn, a, f, b)
  m = b.shape[1]
  cols = vmap(fn, m, [(a.reshape((n * n,)), 0, 0), (f.reshape(((n + 1) * n,)), 0, 0), (b.T.reshape((m * n,)), 0, n)])
  return cols.reshape((m, n)).T


def _call(fn: ConcreteFunction, *args: Expr) -> Expr:
  out = fn.symbolic_call(tuple(args))
  return out[0] if isinstance(out, tuple) else out


_GENERAL: dict[tuple[int, DType, bool], tuple[ConcreteFunction, ConcreteFunction]] = {}


def _general_solvers(n: int, dtype: DType) -> tuple[ConcreteFunction, ConcreteFunction]:
  """``(A, f, b) -> A^{-1} b`` and ``-> A^{-T} b`` with ``f = lu(A)`` computed outside, as Functions
  whose derivatives are implicit: the factor's own derivative is taken to be zero, and the rules
  solve with the same factor through the other Function. Two levels of rules, as ``SparseLDL``'s:
  second derivatives are implicit too, and only a third would reach the factorization.

  The triangular solves inside are straight-line code or loops as ``sc.options(linalg=dict(dense_unroll=...))``
  decides when the solve is built, so each decision has its own pair, named apart from the default's."""
  unroll = n <= get_options().namespace("linalg").dense_unroll
  key = (n, dtype, unroll)
  if key not in _GENERAL:
    tag = f"lu_solve{n}" + ("" if dtype == dtypes.float64 else f"_{dtype.name}")
    if unroll != (n <= LinalgOptions().dense_unroll):
      tag += "_unrolled" if unroll else "_looped"

    def syms(*names: str) -> list[Expr]:
      shapes = {"a": (n, n), "f": (n + 1, n)}
      return [Expr.sym(v, shapes.get(v[-1] if v.startswith("d") else v, (n,)), dtype=dtype) for v in names]

    def base(trans: bool) -> ConcreteFunction:
      a, f, b = syms("a", "f", "b")
      return ConcreteFunction.from_exprs(f"{tag}{'_t' if trans else ''}", [a, f, b], [lu_solve(f, b, trans=trans)], ["a", "f", "b"], ["x"])

    def jvp(inner: ConcreteFunction, trans: bool, level: int) -> ConcreteFunction:
      a, f, b, da, df, db = syms("a", "f", "b", "da", "df", "db")
      x = _call(inner, a, f, b)
      dx = _call(inner, a, f, db - (da.T if trans else da) @ x)
      return ConcreteFunction.from_exprs(f"{inner.name}_jvp{level}", [a, f, b, da, df, db], [dx], ["a", "f", "b", "da", "df", "db"], ["dx"])

    def vjp(other: ConcreteFunction, trans: bool, level: int, name: str) -> ConcreteFunction:
      a, f, b = syms("a", "f", "b")
      x, xbar = Expr.sym("xo", (n,), dtype=dtype), Expr.sym("xbar", (n,), dtype=dtype)
      bbar = _call(other, a, f, xbar)
      outer = x.reshape((n, 1)) @ bbar.reshape((1, n)) if trans else bbar.reshape((n, 1)) @ x.reshape((1, n))
      outs = [-outer, Expr.const(np.zeros((n + 1, n)), dtype=dtype), bbar]
      return ConcreteFunction.from_exprs(f"{name}_vjp{level}", [a, f, b, x, xbar], outs, ["a", "f", "b", "xo", "xbar"], ["abar", "fbar", "bbar"])

    plain, transposed = base(False), base(True)
    inner, inner_t = plain, transposed
    for level in (1, 2):
      inner, inner_t = (
        custom_derivative(plain, jvp=jvp(inner, False, level), vjp=vjp(inner_t, False, level, plain.name)),
        custom_derivative(transposed, jvp=jvp(inner_t, True, level), vjp=vjp(inner, True, level, transposed.name)),
      )
    _GENERAL[key] = (inner, inner_t)
  return _GENERAL[key]
