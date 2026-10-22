"""Whole-derivative constructions assembled from the two modes.

``jacobian`` batches forward mode over the identity, ``gradient`` is one reverse sweep, and
``hessian`` composes the two.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..ir.expr import Expr
from ..passes.expr import simplify_cse_fixpoint
from .forward import jvp_many
from .reverse import _ones_like, vjp


def jacobian(expr: Expr, wrt: Expr) -> Expr:
  """Dense Jacobian ``d expr / d wrt``, shape ``(expr.size, wrt.size)``.

  Column ``j`` is the derivative with respect to ``wrt[j]``. Computed by pushing the whole
  identity through forward mode in one batched pass, then simplifying.
  """
  if wrt.size == 0:
    return Expr.const(np.zeros((expr.size, 0), dtype=expr.type.dtype.numpy()), dtype=expr.type.dtype)
  seed_arr = np.eye(wrt.size, dtype=wrt.type.dtype.numpy()).reshape((wrt.size, *wrt.shape))
  return simplify_cse_fixpoint(jvp_many(expr, wrt, Expr.const(seed_arr, dtype=wrt.type.dtype)).reshape((wrt.size, expr.size)).transpose((1, 0)))


def gradient(expr: Expr, wrt: Expr) -> Expr:
  """Gradient of a scalar ``expr`` with respect to ``wrt``, as one reverse sweep."""
  if expr.size != 1:
    raise ValueError("gradient expects a scalar expression")
  return vjp((expr,), (wrt,), (_ones_like(expr),))[0]


def hessian(expr: Expr, wrt: Expr) -> Expr:
  """Second derivatives of a scalar ``expr``: the Jacobian of its gradient."""
  return jacobian(gradient(expr, wrt).reshape((wrt.size,)), wrt)


def finite_difference(fun: Any, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
  """Approximate the Jacobian of the numerical ``fun`` at ``x`` by central differences.

  Returns shape ``(output size, x.size)``. Useful as an independent check of a symbolic derivative.
  """
  x = np.asarray(x, dtype=np.float64)
  y0 = np.asarray(fun(x), dtype=np.float64).reshape(-1)
  jac = np.empty((y0.size, x.size), dtype=np.float64)
  flat = x.reshape(-1)
  for i in range(flat.size):
    xp = flat.copy()
    xm = flat.copy()
    xp[i] += eps
    xm[i] -= eps
    jac[:, i] = (np.asarray(fun(xp.reshape(x.shape))).reshape(-1) - np.asarray(fun(xm.reshape(x.shape))).reshape(-1)) / (2 * eps)
  return jac
