"""Unit quaternions as rotations: Hamilton's product, stored scalar last ``(x, y, z, w)`` as SymForce and SciPy store them, with the exponential and logarithm of SO(3)."""

from __future__ import annotations

import numpy as np

from ..ir.expr import Expr, less, minimum, stack, where

EPS = 10 * float(np.finfo(np.float64).eps)
"""The regularization of ``exp`` and ``log`` near the identity, SymForce's ``10 eps``."""


def mul(a: Expr, b: Expr) -> Expr:
  """The Hamilton product ``a b``: the rotation ``b`` then ``a``."""
  ax, ay, az, aw = a[0], a[1], a[2], a[3]
  bx, by, bz, bw = b[0], b[1], b[2], b[3]
  return stack(
    [
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
      aw * bw - ax * bx - ay * by - az * bz,
    ]
  )


def conj(q: Expr) -> Expr:
  """The conjugate, the inverse of a unit quaternion."""
  return stack([-q[0], -q[1], -q[2], q[3]])


def _rows(q: Expr) -> list[list[Expr]]:
  x, y, z, w = q[0], q[1], q[2], q[3]
  return [
    [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
    [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
    [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
  ]


def matrix(q: Expr) -> Expr:
  """The rotation matrix ``R(q)``, ``(3, 3)``, of a unit quaternion."""
  return stack([stack(row) for row in _rows(q)])


def rotate(q: Expr, v: Expr) -> Expr:
  """``R(q) v``, through the rotation matrix's entries as SymForce's generated code computes it."""
  r = _rows(q)
  return stack([r[i][0] * v[0] + r[i][1] * v[1] + r[i][2] * v[2] for i in range(3)])


def exp(v: Expr, eps: float = EPS) -> Expr:
  """The rotation by the vector ``v`` (axis times angle): ``(sin(t/2) v / t, cos(t/2))`` with
  ``t = sqrt(|v|^2 + eps^2)``, smooth through the identity."""
  theta = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + eps * eps).sqrt()
  s = (0.5 * theta).sin() / theta
  return stack([s * v[0], s * v[1], s * v[2], (0.5 * theta).cos()])


def log(q: Expr, eps: float = EPS) -> Expr:
  """The rotation vector of a unit quaternion, SymForce's ``Rot3.to_tangent``: ``w`` clamped to
  ``1 - eps`` and its sign taken with ``sign(0) = 1``, so ``q`` and ``-q`` give the same vector."""
  w = q[3]
  sign = where(less(w, 0.0), -1.0, 1.0)
  ws = minimum(w.abs(), 1.0 - eps)
  scale = sign * 2.0 * ws.acos() / (1.0 - ws * ws).sqrt()
  return stack([scale * q[0], scale * q[1], scale * q[2]])


__all__ = ["EPS", "conj", "exp", "log", "matrix", "mul", "rotate"]
