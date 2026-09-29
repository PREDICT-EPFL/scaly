"""Three-vectors: the cross product and its skew-symmetric matrix."""

from __future__ import annotations

from ..ir.expr import Expr, stack


def cross(a: Expr, b: Expr) -> Expr:
  """``a x b`` of two 3-vectors."""
  return stack([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def skew(v: Expr) -> Expr:
  """The matrix ``[v]_x`` with ``[v]_x w = v x w``."""
  zero = 0.0 * v[0]
  return stack([stack([zero, -v[2], v[1]]), stack([v[2], zero, -v[0]]), stack([-v[1], v[0], zero])])


__all__ = ["cross", "skew"]
