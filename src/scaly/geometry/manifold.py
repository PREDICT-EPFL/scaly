"""Manifolds for optimization: a point's size, its tangent space's, and the retraction and local coordinates between them, for vectors, rotations and poses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..ir.expr import Expr, concat
from . import quaternion as quat


@dataclass(frozen=True)
class Euclidean:
  """``R^n``: a point is its own coordinates, ``retract(x, d) = x + d``, ``local(a, b) = b - a``."""

  n: int

  @property
  def size(self) -> int:
    return self.n

  @property
  def tangent(self) -> int:
    return self.n

  def retract(self, x: Expr, delta: Expr) -> Expr:
    """``x + delta``."""
    return x + delta

  def local(self, a: Expr, b: Expr) -> Expr:
    """``b - a``."""
    return b - a


@dataclass(frozen=True)
class SO3:
  """Rotations as unit quaternions (``quaternion``: Hamilton, scalar last), perturbed on the right as
  SymForce does: ``retract(q, d) = q Exp(d)``, ``local(a, b) = Log(a^-1 b)``."""

  size: ClassVar[int] = 4
  tangent: ClassVar[int] = 3
  eps: float = quat.EPS

  def retract(self, q: Expr, delta: Expr) -> Expr:
    """``q Exp(delta)``."""
    return quat.mul(q, quat.exp(delta, self.eps))

  def local(self, a: Expr, b: Expr) -> Expr:
    """``Log(a^-1 b)``."""
    return quat.log(quat.mul(quat.conj(a), b), self.eps)


@dataclass(frozen=True)
class Pose3:
  """Poses as SymForce's ``Pose3`` retracts them: the product of ``SO3`` and ``R^3``, a point
  ``(qx, qy, qz, qw, tx, ty, tz)``, a tangent ``(rotation (3), translation (3))``, the rotation
  perturbed on the right and the translation added (not the SE(3) exponential)."""

  size: ClassVar[int] = 7
  tangent: ClassVar[int] = 6
  eps: float = quat.EPS

  def retract(self, pose: Expr, delta: Expr) -> Expr:
    """``(q Exp(delta_R), t + delta_t)``."""
    return concat([quat.mul(pose[:4], quat.exp(delta[:3], self.eps)), pose[4:] + delta[3:]])

  def local(self, a: Expr, b: Expr) -> Expr:
    """``(Log(a_R^-1 b_R), b_t - a_t)``."""
    return concat([quat.log(quat.mul(quat.conj(a[:4]), b[:4]), self.eps), b[4:] - a[4:]])


__all__ = ["SO3", "Euclidean", "Pose3"]
