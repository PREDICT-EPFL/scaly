"""Exact second derivatives of small NumPy functions, for dense Lagrangian Hessian references.

A hyper-dual number carries a value, two first-order perturbations and their cross term, so one
evaluation of ``f`` at ``x + e1*d1 + e2*d2`` yields ``d1' H d2`` exactly, with no step size. The
second direction is a whole vector, so one evaluation gives one row of the Hessian.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


class HyperDual:
  __slots__ = ("v", "e1", "e2", "e12")

  def __init__(self, v: float, e1: float = 0.0, e2=0.0, e12=0.0):
    self.v, self.e1, self.e2, self.e12 = v, e1, e2, e12

  @staticmethod
  def _coerce(other) -> HyperDual | None:
    # Arrays are left to NumPy, which then applies the operator elementwise.
    if isinstance(other, np.ndarray):
      return None
    return other if isinstance(other, HyperDual) else HyperDual(float(other))

  def __add__(self, other):
    o = self._coerce(other)
    if o is None:
      return NotImplemented
    return HyperDual(self.v + o.v, self.e1 + o.e1, self.e2 + o.e2, self.e12 + o.e12)

  __radd__ = __add__

  def __neg__(self) -> HyperDual:
    return HyperDual(-self.v, -self.e1, -self.e2, -self.e12)

  def __sub__(self, other):
    return NotImplemented if isinstance(other, np.ndarray) else self + (-self._coerce(other))

  def __rsub__(self, other):
    return NotImplemented if isinstance(other, np.ndarray) else self._coerce(other) + (-self)

  def __mul__(self, other):
    o = self._coerce(other)
    if o is None:
      return NotImplemented
    return HyperDual(
      self.v * o.v, self.e1 * o.v + self.v * o.e1, self.e2 * o.v + self.v * o.e2, self.e12 * o.v + self.e1 * o.e2 + self.e2 * o.e1 + self.v * o.e12
    )

  __rmul__ = __mul__

  def _apply(self, f: float, df: float, d2f: float) -> HyperDual:
    return HyperDual(f, df * self.e1, df * self.e2, df * self.e12 + d2f * self.e1 * self.e2)

  def __truediv__(self, other):
    o = self._coerce(other)
    if o is None:
      return NotImplemented
    return self * o._apply(1.0 / o.v, -1.0 / o.v**2, 2.0 / o.v**3)

  def __rtruediv__(self, other):
    return NotImplemented if isinstance(other, np.ndarray) else self._coerce(other) / self

  def __pow__(self, n) -> HyperDual:
    return self._apply(self.v**n, n * self.v ** (n - 1), n * (n - 1) * self.v ** (n - 2))

  def sin(self) -> HyperDual:
    return self._apply(np.sin(self.v), np.cos(self.v), -np.sin(self.v))

  def cos(self) -> HyperDual:
    return self._apply(np.cos(self.v), -np.sin(self.v), -np.cos(self.v))

  def tanh(self) -> HyperDual:
    t = np.tanh(self.v)
    return self._apply(t, 1.0 - t * t, -2.0 * t * (1.0 - t * t))

  def exp(self) -> HyperDual:
    e = np.exp(self.v)
    return self._apply(e, e, e)

  def sqrt(self) -> HyperDual:
    s = np.sqrt(self.v)
    return self._apply(s, 0.5 / s, -0.25 / (s * self.v))


def lagrangian_hessian_np(fn: Callable[[np.ndarray], Sequence], x: np.ndarray, lam: np.ndarray) -> np.ndarray:
  """Dense Hessian of ``dot(lam, fn(x))`` at ``x``.

  ``fn`` receives an object array of ``HyperDual`` and must be written with NumPy operators and the
  ``np.sin``-style ufuncs, which dispatch to the methods above elementwise.
  """
  x, lam = np.asarray(x, dtype=np.float64), np.asarray(lam, dtype=np.float64).reshape(-1)
  n = x.size
  hess = np.zeros((n, n), dtype=np.float64)
  for i in range(n):
    xs = np.empty(n, dtype=object)
    for j in range(n):
      e2 = np.zeros(n)
      e2[j] = 1.0
      xs[j] = HyperDual(float(x[j]), 1.0 if j == i else 0.0, e2, np.zeros(n))
    total = HyperDual(0.0)
    for weight, out in zip(lam, fn(xs), strict=True):
      total = total + float(weight) * out
    hess[i] = total.e12
  return hess
