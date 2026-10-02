"""Symmetric positive definite block-tridiagonal systems: the Cholesky factorization as a ``scan`` over the blocks, and solves with it.

The matrix has ``K`` diagonal blocks ``D_k``, each ``B x B``, and below each but the last a block
``E_k`` (the rows of block ``k + 1``, the columns of block ``k``) that is zero outside its last
``c`` columns:

    [ D_0  E_0'            ]
    [ E_0  D_1  E_1'       ]
    [      E_1  D_2  ...   ]
    [           ...        ]

It is what the normal equations of a problem with stages look like once the variables are in stage
order, where a stage couples with the next through a few of its variables only. The factor is
``L_k L_k' = D_k - W_{k-1} W_{k-1}'`` with ``W_k = E_k L_k^{-T}``, which is zero outside its last
``c`` columns as ``E_k`` is, so only those are computed and stored. One step is a Cholesky of a
block, a triangular solve against its trailing ``c x c`` triangle and a product, each a dense
kernel, and the generated code is that step in a loop: it does not grow with ``K``.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from ..function.model import ConcreteFunction
from ..function.sugar import scan
from ..ir.expr import Expr, as_expr, concat, gather
from .dense import cholesky, solve_triangular

__all__ = ["BlockTridiagonalCholesky"]

_NAMES = itertools.count()


def _backward(flat: Expr, steps: int, size: int) -> Expr:
  """The ``steps`` rows of ``size`` values in ``flat``, last first: a backward scan's output in block order."""
  rows = (np.arange(steps)[::-1, None] * size + np.arange(size)[None, :]).reshape(-1)
  return gather(flat, rows)


class BlockTridiagonalCholesky:
  """The Cholesky factorization ``L L'`` of a symmetric positive definite block-tridiagonal matrix.

  ``diag`` is ``(K, B, B)``, the diagonal blocks, of which the lower triangles are read; ``below``
  is ``(K - 1, B, c)``, the last ``c`` columns of each block below the diagonal, the others being
  zero (``c = B`` for a full block). With one block ``below`` is not given.

  ``values`` holds the factor, flat: the ``K`` triangular blocks ``L_k`` (their upper triangles
  zero), then the ``K - 1`` blocks ``W_k`` of ``c`` columns. A pivot that is not positive leaves
  that entry of the factor's diagonal, and what follows it, not positive or not a number;
  ``diagonal`` is where to look. ``solve`` and ``solve_with`` solve with the factor, and
  everything is differentiable through the loops.
  """

  def __init__(self, diag: Any, below: Any = None, *, name: str | None = None) -> None:
    diag = as_expr(diag)
    if len(diag.shape) != 3 or diag.shape[1] != diag.shape[2] or not diag.shape[0] or not diag.shape[1]:
      raise ValueError(f"diag must be a stack of K square blocks, (K, B, B), got shape {diag.shape}")
    self.K, self.B = diag.shape[0], diag.shape[1]
    if below is None:
      if self.K != 1:
        raise ValueError(f"{self.K} diagonal blocks need the {self.K - 1} blocks below them")
      self.c = 0
    else:
      below = as_expr(below)
      if len(below.shape) != 3 or below.shape[:2] != (self.K - 1, self.B) or not 1 <= below.shape[2] <= self.B:
        raise ValueError(f"below must have shape ({self.K - 1}, {self.B}, c) with 1 <= c <= {self.B}, got {below.shape}")
      self.c = below.shape[2] if self.K > 1 else 0
    self.name = name or f"blocks{next(_NAMES)}"
    self.values = self._factor(diag, below)

  # --- the layout of ``values`` -------------------------------------------------------------------

  @property
  def n(self) -> int:
    """The order of the matrix, ``K * B``."""
    return self.K * self.B

  @property
  def size(self) -> int:
    """The length of ``values``."""
    return self.K * self.B * self.B + (self.K - 1) * self.B * self.c

  def _parts(self, values: Expr) -> tuple[Expr, Expr]:
    split = self.K * self.B * self.B
    return values[:split], values[split:]

  def diagonal_of(self, values: Expr) -> Expr:
    """The diagonal of the factor in ``values``, ``(K * B,)``."""
    k, b = np.arange(self.K)[:, None], np.arange(self.B)[None, :]
    return gather(values, (k * self.B * self.B + b * (self.B + 1)).reshape(-1))

  @property
  def diagonal(self) -> Expr:
    """The diagonal of the factor, ``(K * B,)``: every entry positive when the factorization went through."""
    return self.diagonal_of(self.values)

  # --- the factorization --------------------------------------------------------------------------

  def _factor(self, diag: Expr, below: Expr | None) -> Expr:
    K, B, c = self.K, self.B, self.c
    flat = diag.reshape((K * B * B,))
    if K == 1:
      return cholesky(diag.reshape((B, B))).reshape((B * B,))
    assert below is not None
    taken, d, e = Expr.sym("S", (B * B,)), Expr.sym("D", (B, B)), Expr.sym("E", (B, c))
    lower = cholesky(d - taken.reshape((B, B)))
    # W L' = E with W zero outside its last c columns: the trailing triangle alone decides them.
    w = solve_triangular(lower[B - c :, B - c :], e.T, lower=True).T
    step = ConcreteFunction.from_exprs(
      f"{self.name}_factor_step", [taken, d, e], [(w @ w.T).reshape((B * B,)), lower, w], ["S", "D", "E"], ["S_next", "L", "W"]
    )
    slices = [(flat, 0, B * B), (below.reshape(((K - 1) * B * c,)), 0, B * c)]
    last, lowers, ws = scan(step, Expr.const(np.zeros(B * B)), slices, length=K - 1)
    final = cholesky(flat[(K - 1) * B * B :].reshape((B, B)) - last.reshape((B, B)))
    return concat([lowers, final.reshape((B * B,)), ws])

  # --- the solves ---------------------------------------------------------------------------------

  def solve(self, rhs: Any) -> Expr:
    """``A^{-1} rhs`` for a right-hand side of ``K * B`` values."""
    return self.solve_with(self.values, rhs)

  def solve_with(self, values: Any, rhs: Any) -> Expr:
    """``A^{-1} rhs`` with the factor in ``values``, which may be this factorization's own or one
    kept from another call: a forward pass over the blocks, then a backward one."""
    K, B, c = self.K, self.B, self.c
    values, rhs = as_expr(values), as_expr(rhs)
    if values.shape != (self.size,):
      raise ValueError(f"values must have shape ({self.size},), got {values.shape}")
    if rhs.shape != (self.n,):
      raise ValueError(f"the right-hand side must have shape ({self.n},), got {rhs.shape}")
    lowers, ws = self._parts(values)
    last = lowers[(K - 1) * B * B :].reshape((B, B))
    if K == 1:
      return solve_triangular(last, solve_triangular(last, rhs, lower=True), lower=True, trans=True)
    lead = Expr.const(np.zeros(B - c))
    carry, lk, wk, rk = Expr.sym("t", (B,)), Expr.sym("L", (B, B)), Expr.sym("W", (B, c)), Expr.sym("r", (B,))
    y = solve_triangular(lk, rk - carry, lower=True)
    forward = ConcreteFunction.from_exprs(
      f"{self.name}_solve_forward", [carry, lk, wk, rk], [wk @ y[B - c :], y], ["t", "L", "W", "r"], ["t_next", "y"]
    )
    taken, ys = scan(forward, Expr.const(np.zeros(B)), [(lowers, 0, B * B), (ws, 0, B * c), (rhs, 0, B)], length=K - 1)
    y_last = solve_triangular(last, rhs[(K - 1) * B :] - taken, lower=True)
    x_last = solve_triangular(last, y_last, lower=True, trans=True)
    # Row ``k`` of ``L'`` reads the next block through ``W_k'``, which reaches the last c entries only.
    nxt, yk = Expr.sym("x_next", (B,)), Expr.sym("y", (B,))
    x = solve_triangular(lk, yk - concat([lead, wk.T @ nxt]), lower=True, trans=True)
    backward = ConcreteFunction.from_exprs(f"{self.name}_solve_backward", [nxt, lk, wk, yk], [x, x], ["x_next", "L", "W", "y"], ["x", "x_out"])
    slices = [(lowers, (K - 2) * B * B, -B * B), (ws, (K - 2) * B * c, -B * c), (ys, (K - 2) * B, -B)]
    _, xs = scan(backward, x_last, slices, length=K - 1)
    return concat([_backward(xs, K - 1, B), x_last])
