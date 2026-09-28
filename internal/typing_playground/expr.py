"""Stand-ins for ``scaly.Expr`` and ``np.ndarray``: shapes, and the polynomial degree the QP proof uses."""

from __future__ import annotations

from dataclasses import dataclass
from types import EllipsisType

import numpy as np

type Shape = tuple[int, ...]
type ShapeDecl = Shape | EllipsisType


def as_shape(shape: int | Shape) -> Shape:
  return (shape,) if isinstance(shape, int) else shape


def _size(shape: Shape) -> int:
  return int(np.prod(shape)) if shape else 1


def _degree_add(a: int | None, b: int | None) -> int | None:
  return None if a is None or b is None else max(a, b)


def _degree_mul(a: int | None, b: int | None) -> int | None:
  return None if a is None or b is None else a + b


@dataclass(frozen=True)
class Expr:
  """A symbolic tensor. ``degree`` is its polynomial degree in the problem variables (``None`` when
  not polynomial); it stands in for the structural dependency analysis the real QP proof uses."""

  shape: Shape
  name: str | None = None
  degree: int | None = 0

  @property
  def size(self) -> int:
    return _size(self.shape)

  def _binary(self, other: Expr | float, degree: int | None) -> Expr:
    other_shape = other.shape if isinstance(other, Expr) else ()
    if self.shape and other_shape and self.shape != other_shape:
      raise ValueError(f"shape mismatch {self.shape} vs {other_shape}")
    return Expr(self.shape or other_shape, degree=degree)

  def _degree_of(self, other: Expr | float) -> int | None:
    return other.degree if isinstance(other, Expr) else 0

  def __add__(self, other: Expr | float) -> Expr:
    return self._binary(other, _degree_add(self.degree, self._degree_of(other)))

  __radd__ = __add__

  def __sub__(self, other: Expr | float) -> Expr:
    return self._binary(other, _degree_add(self.degree, self._degree_of(other)))

  __rsub__ = __sub__

  def __neg__(self) -> Expr:
    return Expr(self.shape, degree=self.degree)

  def __mul__(self, other: Expr | float) -> Expr:
    return self._binary(other, _degree_mul(self.degree, self._degree_of(other)))

  __rmul__ = __mul__

  def __truediv__(self, other: Expr | float) -> Expr:
    other_degree = self._degree_of(other)
    return self._binary(other, self.degree if other_degree == 0 else None)

  def __pow__(self, power: int) -> Expr:
    return Expr(self.shape, degree=None if self.degree is None else self.degree * power)

  def __matmul__(self, other: Expr) -> Expr:
    if len(self.shape) == 2 and len(other.shape) == 1 and self.shape[1] == other.shape[0]:
      shape: Shape = (self.shape[0],)
    elif len(self.shape) == 1 and len(other.shape) == 1 and self.shape == other.shape:
      shape = ()
    elif len(self.shape) == 1 and len(other.shape) == 2 and self.shape[0] == other.shape[0]:
      shape = (other.shape[1],)
    else:
      raise ValueError(f"cannot matmul {self.shape} @ {other.shape}")
    return Expr(shape, degree=_degree_mul(self.degree, other.degree))

  def __getitem__(self, index: int | slice) -> Expr:
    if not self.shape:
      raise ValueError("cannot index a scalar")
    if isinstance(index, slice):
      return Expr((len(range(*index.indices(self.shape[0]))), *self.shape[1:]), degree=self.degree)
    return Expr(self.shape[1:], degree=self.degree)

  def sum(self) -> Expr:
    return Expr((), degree=self.degree)

  def sin(self) -> Expr:
    return Expr(self.shape, degree=None if self.degree else 0)


def const(value: float | np.ndarray) -> Expr:
  return Expr(np.shape(value), degree=0)


def concat(parts: tuple[Expr, ...]) -> Expr:
  degree: int | None = 0
  for part in parts:
    degree = _degree_add(degree, part.degree)
  return Expr((sum(part.size for part in parts),), degree=degree)


@dataclass(frozen=True)
class Buffer:
  """A numeric tensor (``np.ndarray`` in the real thing)."""

  shape: Shape

  @property
  def size(self) -> int:
    return _size(self.shape)
