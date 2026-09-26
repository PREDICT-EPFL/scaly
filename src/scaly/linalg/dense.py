"""Dense factorizations and triangular solves as expression ops, lowered to plain loops.

``cholesky``, ``ldl`` and ``solve_triangular`` are expression ops (``ir/expr.py``) with their own
loop lowering, derivatives in both modes and structural sparsity; this module adds the solves built
from them. Orders up to ``DENSE_UNROLL`` become straight-line code. Nothing calls an external
library: the loops are generated C like everything else.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ..ir.expr import Expr, as_expr, cholesky, gather, ldl, solve_triangular

__all__ = ["cho_solve", "cholesky", "ldl", "ldl_solve", "ldl_unpack", "solve", "solve_triangular"]


def ldl_unpack(f: Any) -> tuple[Expr, Expr]:
  """``(L, d)`` from a packed ``ldl`` factor: the unit lower ``L`` and the diagonal of ``D``."""
  f = as_expr(f)
  n = f.shape[0]
  unit_l = f * Expr.const(np.tril(np.ones((n, n)), -1)) + Expr.const(np.eye(n))
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


def solve(a: Any, b: Any, *, assume: Literal["pos", "sym"] = "pos") -> Expr:
  """``A^{-1} b`` for a symmetric ``A``, reading its lower triangle: by Cholesky for a positive
  definite matrix (``assume="pos"``), by ``L D L^T`` without pivoting for a quasi-definite one
  (``assume="sym"``)."""
  if assume == "pos":
    return cho_solve(cholesky(a), b)
  if assume == "sym":
    return ldl_solve(ldl(a), b)
  raise ValueError(f"assume must be 'pos' or 'sym', got {assume!r}")
