from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul
from collections.abc import Sequence
from typing import Literal

import numpy as np

DType = Literal["float64"]
Lowering = Literal["auto", "scalar", "block", "opaque"]


def _check_shape(name: str, shape: tuple[int, ...]) -> None:
  if any(d < 0 for d in shape):
    raise ValueError(f"{name} shape cannot contain negative dimensions, got {shape}")


@dataclass(frozen=True, slots=True)
class ScalarType:
  dtype: DType = "float64"
  diff: bool = True


@dataclass(frozen=True, slots=True)
class SparsityType:
  shape: tuple[int, int]
  rows: tuple[int, ...]
  cols: tuple[int, ...]

  def __post_init__(self) -> None:
    if len(self.shape) != 2:
      raise ValueError(f"sparsity shape must be rank-2, got {self.shape}")
    _check_shape("sparsity", self.shape)
    if len(self.rows) != len(self.cols):
      raise ValueError("sparsity rows and cols must have the same length")
    if any(r < 0 or r >= self.shape[0] for r in self.rows) or any(c < 0 or c >= self.shape[1] for c in self.cols):
      raise ValueError(f"sparsity indices out of bounds for shape {self.shape}")
    if len(set(zip(self.rows, self.cols))) != len(self.rows):
      raise ValueError("sparsity indices must be unique")

  @property
  def nnz(self) -> int:
    return len(self.rows)

  @staticmethod
  def empty(shape: tuple[int, int]) -> SparsityType:
    return SparsityType(shape, (), ())

  @staticmethod
  def dense(shape: tuple[int, int]) -> SparsityType:
    rows, cols = np.nonzero(np.ones(shape, dtype=bool))
    return SparsityType(shape, tuple(int(x) for x in rows), tuple(int(x) for x in cols))

  @staticmethod
  def from_mask(mask: np.ndarray) -> SparsityType:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
      raise ValueError(f"sparsity mask must be rank-2, got {mask.shape}")
    rows, cols = np.nonzero(mask)
    shape = (int(mask.shape[0]), int(mask.shape[1]))
    return SparsityType(shape, tuple(int(x) for x in rows), tuple(int(x) for x in cols))

  @staticmethod
  def from_csr(shape: tuple[int, int], row_ptr: Sequence[int], col_ind: Sequence[int]) -> SparsityType:
    row_ptr = tuple(int(x) for x in row_ptr)
    col_ind = tuple(int(x) for x in col_ind)
    _check_compressed_ptr("row_ptr", row_ptr, shape[0], len(col_ind))
    if any(c < 0 or c >= shape[1] for c in col_ind):
      raise ValueError(f"CSR column indices out of bounds for shape {shape}")
    rows = tuple(r for r in range(shape[0]) for _ in range(row_ptr[r + 1] - row_ptr[r]))
    return SparsityType(shape, rows, col_ind)

  @staticmethod
  def from_csc(shape: tuple[int, int], col_ptr: Sequence[int], row_ind: Sequence[int]) -> SparsityType:
    col_ptr = tuple(int(x) for x in col_ptr)
    row_ind = tuple(int(x) for x in row_ind)
    _check_compressed_ptr("col_ptr", col_ptr, shape[1], len(row_ind))
    if any(r < 0 or r >= shape[0] for r in row_ind):
      raise ValueError(f"CSC row indices out of bounds for shape {shape}")
    cols = tuple(c for c in range(shape[1]) for _ in range(col_ptr[c + 1] - col_ptr[c]))
    return SparsityType(shape, row_ind, cols)

  def to_mask(self) -> np.ndarray:
    mask = np.zeros(self.shape, dtype=bool)
    mask[list(self.rows), list(self.cols)] = True
    return mask

  def to_csr(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
    order = sorted(range(self.nnz), key=lambda i: (self.rows[i], self.cols[i]))
    row_ptr = [0] * (self.shape[0] + 1)
    for i in order:
      row_ptr[self.rows[i] + 1] += 1
    for r in range(self.shape[0]):
      row_ptr[r + 1] += row_ptr[r]
    return tuple(row_ptr), tuple(self.cols[i] for i in order)

  def to_csc(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
    order = sorted(range(self.nnz), key=lambda i: (self.cols[i], self.rows[i]))
    col_ptr = [0] * (self.shape[1] + 1)
    for i in order:
      col_ptr[self.cols[i] + 1] += 1
    for c in range(self.shape[1]):
      col_ptr[c + 1] += col_ptr[c]
    return tuple(col_ptr), tuple(self.rows[i] for i in order)


def _check_compressed_ptr(name: str, ptr: tuple[int, ...], n_outer: int, nnz: int) -> None:
  if len(ptr) != n_outer + 1:
    raise ValueError(f"{name} must have length {n_outer + 1}, got {len(ptr)}")
  if not ptr or ptr[0] != 0 or ptr[-1] != nnz:
    raise ValueError(f"{name} must start at 0 and end at nnz={nnz}")
  if any(a > b for a, b in zip(ptr, ptr[1:])):
    raise ValueError(f"{name} must be nondecreasing")


@dataclass(frozen=True, slots=True)
class TensorType:
  shape: tuple[int, ...] = ()
  dtype: DType = "float64"
  sparsity: SparsityType | None = None
  diff: bool = True

  def __post_init__(self) -> None:
    _check_shape("tensor", self.shape)
    if self.sparsity is not None and self.shape != self.sparsity.shape:
      raise ValueError(f"tensor shape {self.shape} does not match sparsity shape {self.sparsity.shape}")

  @property
  def ndim(self) -> int:
    return len(self.shape)

  @property
  def size(self) -> int:
    return reduce(mul, self.shape, 1)

  @property
  def is_scalar(self) -> bool:
    return self.shape == () or self.shape == (1,) or self.size == 1


def as_shape(shape: int | tuple[int, ...] | list[int] | None = None) -> tuple[int, ...]:
  if shape is None:
    return ()
  if isinstance(shape, int):
    ret = (shape,)
  else:
    ret = tuple(int(x) for x in shape)
  _check_shape("tensor", ret)
  return ret


def broadcast_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
  if not a:
    return b
  if not b:
    return a
  out: list[int] = []
  for da, db in zip(reversed(a), reversed(b), strict=False):
    if da == 1:
      out.append(db)
    elif db == 1:
      out.append(da)
    elif da == db:
      out.append(da)
    else:
      raise ValueError(f"cannot broadcast shapes {a} and {b}")
  longer = a if len(a) > len(b) else b
  out.extend(reversed(longer[: abs(len(a) - len(b))]))
  return tuple(reversed(out))
