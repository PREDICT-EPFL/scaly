"""Declared dense and sparse value orders and immutable boundary index arrays."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from math import prod
from threading import Lock
from typing import Any, Literal
from weakref import WeakValueDictionary

import numpy as np
from scipy import sparse

from .types import CanonicalPattern, SparsityPattern


def _readonly(value: Any) -> np.ndarray:
  return np.frombuffer(np.asarray(value, dtype=np.int64).tobytes(), dtype=np.int64)


def bind_sparse(value: Any) -> tuple[CanonicalPattern, np.ndarray]:
  """Canonicalize a copied SciPy argument and recover its shared matrix pattern."""
  if not np.can_cast(value.dtype, np.float64, "same_kind"):
    raise TypeError(f"cannot cast {value.dtype} to float64")
  copied = sparse.coo_array(value, copy=True)
  copied.sum_duplicates()
  shape = copied.shape
  if len(shape) < 2:
    raise ValueError("a pattern hole needs a matrix or a batch of matrices")
  count = prod(shape[:-2])
  if count == 0:
    raise ValueError("an empty batch cannot bind a pattern hole; instantiate it with a TensorType")
  nnz, remainder = divmod(copied.nnz, count)
  if remainder or (
    shape[:-2] and copied.nnz and not np.array_equal(np.ravel_multi_index(copied.coords[:-2], shape[:-2]), np.arange(copied.nnz) // nnz)
  ):
    raise ValueError("sparse batches must share one stored pattern")
  rows, cols = copied.coords[-2:]
  pattern = CanonicalPattern(shape[-2:], tuple(map(int, rows[:nnz])), tuple(map(int, cols[:nnz])))
  if any(
    not np.array_equal(actual, expected) for actual, expected in zip(copied.coords, sparse_layout(pattern, "csr").coordinates(shape), strict=True)
  ):
    raise ValueError("sparse batches must share one stored pattern")
  return pattern, np.ascontiguousarray(copied.data.reshape((*shape[:-2], nnz)), dtype=np.float64)


@dataclass(frozen=True)
class DenseLayout:
  """Column-major order within each dense matrix, with leading batch axes unchanged."""

  shape: tuple[int, ...] | None

  @cached_property
  def permutation(self) -> np.ndarray:
    """Indices mapping row-major matrix values to column-major values."""
    assert self.shape is not None
    shape = self.shape[-2:]
    return _readonly(np.arange(prod(shape)).reshape(shape).T.ravel())

  @cached_property
  def inverse_permutation(self) -> np.ndarray:
    """Indices restoring row-major matrix values."""
    return _readonly(np.argsort(self.permutation))

  def numerical_values(self, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Keep logical coordinates while making each matrix column-contiguous."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
      raise ValueError(f"dense input has shape {array.shape}, expected {shape}")
    return np.ascontiguousarray(array.swapaxes(-1, -2)).swapaxes(-1, -2)

  def dense_array(self, values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """View a column-major value buffer with its mathematical shape."""
    return values.reshape((*shape[:-2], shape[-1], shape[-2])).swapaxes(-1, -2)


@dataclass(frozen=True, eq=False)
class SparseLayout:
  """One declared order of a canonical pattern's stored values."""

  pattern: CanonicalPattern
  format: Literal["csc", "csr", "coo"]
  order: tuple[int, ...]
  _batch_coordinates: dict[tuple[int, ...], tuple[np.ndarray, ...]] = field(init=False, default_factory=dict, repr=False)

  @cached_property
  def permutation(self) -> np.ndarray:
    """Indices such that declared values equal canonical values at these slots."""
    return _readonly(self.order)

  @cached_property
  def inverse_permutation(self) -> np.ndarray:
    """Indices that restore canonical row-major order from the declared order."""
    return _readonly(np.argsort(self.permutation))

  @cached_property
  def rows(self) -> np.ndarray:
    """Stored row coordinates in the declared order."""
    return _readonly(np.asarray(self.pattern.rows)[self.permutation])

  @cached_property
  def cols(self) -> np.ndarray:
    """Stored column coordinates in the declared order."""
    return _readonly(np.asarray(self.pattern.cols)[self.permutation])

  @cached_property
  def indptr(self) -> np.ndarray:
    """Compressed row or column pointers for this layout."""
    outer, extent = (self.cols, self.pattern.shape[1]) if self.format == "csc" else (self.rows, self.pattern.shape[0])
    return _readonly(np.r_[0, np.cumsum(np.bincount(outer, minlength=extent))])

  @property
  def indices(self) -> np.ndarray:
    """The minor indices for a compressed layout."""
    return self.rows if self.format == "csc" else self.cols

  def coordinates(self, shape: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    """Read-only coordinates shared by all results of this batch shape."""
    if shape not in self._batch_coordinates:
      batch = shape[:-2]
      count = prod(batch)
      if batch and (count == 0 or self.pattern.nnz == 0):
        self._batch_coordinates[shape] = tuple(_readonly(np.empty(0)) for _ in shape)
        return self._batch_coordinates[shape]
      axes = np.unravel_index(np.arange(count), batch) if batch else ()
      self._batch_coordinates[shape] = (
        (
          *(_readonly(np.repeat(axis, self.pattern.nnz)) for axis in axes),
          _readonly(np.tile(self.rows, count)),
          _readonly(np.tile(self.cols, count)),
        )
        if batch
        else (self.rows, self.cols)
      )
    return self._batch_coordinates[shape]

  def numerical_values(self, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Copy and canonicalize sparse input, checking every stored coordinate."""
    storage_shape = (*shape[:-2], self.pattern.nnz)
    if not sparse.issparse(value):
      array = np.asarray(value)
      if array.shape != storage_shape:
        raise ValueError(f"expected a sparse array of shape {shape} or stored values of shape {storage_shape}, got {array.shape}")
      if not np.can_cast(array.dtype, np.float64, "same_kind"):
        raise TypeError(f"cannot cast {array.dtype} to float64")
      return np.require(array, dtype=np.float64, requirements="C")
    if value.shape != shape:
      raise ValueError(f"sparse input has shape {value.shape}, expected {shape}")
    if not np.can_cast(value.dtype, np.float64, "same_kind"):
      raise TypeError(f"cannot cast {value.dtype} to float64")
    copied = sparse.coo_array(value, copy=True)
    copied.sum_duplicates()
    coordinates = copied.coords
    order = np.lexsort(tuple(reversed(coordinates)))
    expected = self.coordinates(shape)
    expected_order = np.lexsort(tuple(reversed(expected)))
    if copied.nnz != prod(storage_shape) or any(
      not np.array_equal(axis[order], target[expected_order]) for axis, target in zip(coordinates, expected, strict=True)
    ):
      raise ValueError("sparse input has a different stored pattern")
    canonical = copied.data[order].reshape(storage_shape)
    return np.ascontiguousarray(canonical[..., self.permutation], dtype=np.float64)

  def sparse_array(self, values: np.ndarray, shape: tuple[int, ...]) -> sparse.sparray:
    """Attach the shared read-only indices to a fresh declared-order value buffer."""
    if len(shape) == 2 and self.format != "coo":
      constructor = sparse.csc_array if self.format == "csc" else sparse.csr_array
      result = constructor((values.reshape(-1), self.indices, self.indptr), shape=shape, copy=False)
      result.indices, result.indptr = self.indices, self.indptr
      return result
    return sparse.coo_array((values.reshape(-1), self.coordinates(shape)), shape=shape, copy=False)


type LeafLayout = DenseLayout | SparseLayout | None
type Layout = str | SparsityPattern | SparseLayout | DenseLayout | None
_LAYOUTS: WeakValueDictionary[tuple[Any, ...], SparseLayout] = WeakValueDictionary()
_LAYOUT_LOCK = Lock()


def sparse_layout(pattern: CanonicalPattern, layout: Layout = "csc") -> SparseLayout:
  """Resolve and share the immutable index arrays for a declared sparse order."""
  if isinstance(layout, SparseLayout):
    if layout.pattern is not pattern:
      raise ValueError("layout has a different stored pattern")
    return layout
  key = (pattern, layout)
  with _LAYOUT_LOCK:
    cached = _LAYOUTS.get(key)
    if cached is not None:
      return cached
    if isinstance(layout, SparsityPattern):
      canonical, permutation = layout.canonicalize()
      if canonical is not pattern:
        raise ValueError("coordinate layout has a different stored pattern")
      format_, order = "coo", tuple(map(int, np.argsort(permutation)))
    elif layout in ("csc", "csr"):
      format_ = layout
      order = tuple(map(int, np.lexsort((pattern.rows, pattern.cols)))) if layout == "csc" else tuple(range(pattern.nnz))
    else:
      raise ValueError("sparse layout must be 'csc', 'csr', or a coordinate pattern")
    cached = SparseLayout(pattern, format_, order)
    _LAYOUTS[key] = cached
    return cached
