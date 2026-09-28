"""Banded linear systems whose matrix is known when the graph is built: tridiagonal and cyclic
tridiagonal solves by the Thomas algorithm, as ``scan``s over the rows of the right-hand side."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import solve_banded

from ..function.model import ConcreteFunction
from ..function.sugar import scan
from ..ir.expr import Expr, as_expr, gather

__all__ = ["solve_cyclic_tridiagonal", "solve_tridiagonal"]


def _bands(lower: Any, diag: Any, upper: Any, rhs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, Expr]:
  diag = np.array(diag, dtype=np.float64).reshape(-1)
  lower, upper = (np.array(v, dtype=np.float64).reshape(-1) for v in (lower, upper))
  rhs = as_expr(rhs)
  if lower.size != diag.size or upper.size != diag.size or diag.size == 0:
    raise ValueError(f"the three bands need one entry per row, got {lower.size}, {diag.size} and {upper.size}")
  if not rhs.shape or rhs.shape[0] != diag.size:
    raise ValueError(f"the right-hand side needs {diag.size} rows along its first axis, got shape {rhs.shape}")
  return lower, diag, upper, rhs


def solve_tridiagonal(lower: Any, diag: Any, upper: Any, rhs: Any) -> Expr:
  """``A x = rhs`` for a tridiagonal ``A`` known when the graph is built (``lower[i] = A[i, i-1]``,
  ``diag[i] = A[i, i]``, ``upper[i] = A[i, i+1]``; ``lower[0]`` and ``upper[-1]`` are not read) and
  a right-hand side ``rhs`` with one row per row of ``A`` along its first axis, any trailing shape.

  The Thomas algorithm without pivoting, its pivots and multipliers computed now: the generated
  code is a forward and a backward ``scan`` over the rows, linear in ``rhs`` and differentiable in
  it. A zero pivot gives inf or NaN; a diagonally dominant ``A`` has none."""
  lower, diag, upper, rhs = _bands(lower, diag, upper, rhs)
  n = diag.size
  width = rhs.size // n
  pivot, ratio = np.empty(n), np.empty(n)
  pivot[0], ratio[0] = diag[0], upper[0] / diag[0]
  for i in range(1, n):
    pivot[i] = diag[i] - lower[i] * ratio[i - 1]
    ratio[i] = upper[i] / pivot[i]
  forward = np.stack([lower, 1.0 / pivot], axis=1).reshape(-1)
  flat = rhs.reshape((rhs.size,))
  (_, reduced) = scan(_sweep(width, "forward"), Expr.const(np.zeros(width)), [(flat, 0, width), (Expr.const(forward), 0, 2)], length=n)
  back = np.arange(n)[::-1]
  rows = (back[:, None] * width + np.arange(width)[None, :]).reshape(-1)
  (_, out) = scan(
    _sweep(width, "backward"), Expr.const(np.zeros(width)), [(gather(reduced, rows), 0, width), (Expr.const(ratio[back]), 0, 1)], length=n
  )
  return gather(out, rows).reshape(rhs.shape)


def solve_cyclic_tridiagonal(lower: Any, diag: Any, upper: Any, rhs: Any) -> Expr:
  """``A x = rhs`` for a cyclic tridiagonal ``A`` known when the graph is built: the bands of
  ``solve_tridiagonal``, with ``lower[0] = A[0, n-1]`` and ``upper[-1] = A[n-1, 0]`` in the corners,
  as a periodic spline's system has them. Sherman and Morrison: one tridiagonal solve of ``rhs``
  and a correction along a vector found now."""
  lower, diag, upper, rhs = _bands(lower, diag, upper, rhs)
  n = diag.size
  alpha, beta, gamma = upper[-1], lower[0], -diag[0]  # A[n-1, 0], A[0, n-1]
  corrected = diag.copy()
  corrected[0], corrected[-1] = diag[0] - gamma, diag[-1] - alpha * beta / gamma
  inner_lower, inner_upper = lower.copy(), upper.copy()
  inner_lower[0], inner_upper[-1] = 0.0, 0.0
  u = np.zeros(n)
  u[0], u[-1] = gamma, alpha
  banded = np.stack([np.concatenate([[0.0], inner_upper[:-1]]), corrected, np.concatenate([inner_lower[1:], [0.0]])])
  z = solve_banded((1, 1), banded, u)
  x = solve_tridiagonal(inner_lower, corrected, inner_upper, rhs)
  factor = z / (1.0 + z[0] + beta * z[-1] / gamma)
  weight = x[0:1] + (beta / gamma) * x[-1:]
  return x - Expr.const(factor.reshape(-1, *(1,) * (len(rhs.shape) - 1))) * weight


_SWEEPS: dict[tuple[int, str], ConcreteFunction[Any, Any, Any, Any]] = {}


def _sweep(width: int, direction: str) -> ConcreteFunction[Any, Any, Any, Any]:
  """One row of the Thomas algorithm, as a scan body: forward ``r_i = (b_i - l_i r_{i-1}) / p_i``,
  backward ``s_i = r_i - c_i s_{i+1}``; the carry is the previous row."""
  key = (width, direction)
  if key not in _SWEEPS:
    prev, row = Expr.sym("prev", (width,)), Expr.sym("row", (width,))
    if direction == "forward":
      coef = Expr.sym("coef", (2,))
      nxt = (row - coef[0] * prev) * coef[1]
    else:
      coef = Expr.sym("coef", (1,))
      nxt = row - coef[0] * prev
    _SWEEPS[key] = ConcreteFunction.from_exprs(
      f"linalg_thomas_{direction}_{width}", [prev, row, coef], [nxt, nxt], ["prev", "row", "coef"], ["next", "out"]
    )
  return _SWEEPS[key]
