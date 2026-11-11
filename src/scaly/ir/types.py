"""The type vocabulary both dialects share: dtypes, device placement, tensor types and sparsity patterns."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from hashlib import sha256
from operator import index, mul
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from threading import Lock
from typing import Any, Literal
import weakref

import numpy as np

Lowering = Literal["auto", "scalar", "block", "opaque"]


@dataclass(frozen=True, slots=True)
class DType:
  """Interned dtype descriptor.

  See ``dtypes`` for the canonical instances. A ``DType`` compares equal to its string ``name``,
  so ``dtypes.float64 == "float64"`` holds.
  """

  name: str
  bits: int
  c_type: str
  is_floating: bool = False
  is_integer: bool = False
  is_bool: bool = False
  is_unsigned: bool = False

  @property
  def itemsize(self) -> int:
    return self.bits // 8

  def numpy(self) -> np.dtype:
    return np.dtype(self.name)

  def __str__(self) -> str:
    return self.name

  def __eq__(self, other: object) -> bool:  # backward compat with string dtype
    if isinstance(other, DType):
      return (self.name, self.bits, self.c_type, self.is_floating, self.is_integer, self.is_bool, self.is_unsigned) == (
        other.name,
        other.bits,
        other.c_type,
        other.is_floating,
        other.is_integer,
        other.is_bool,
        other.is_unsigned,
      )
    if isinstance(other, str):
      return self.name == other
    return NotImplemented

  def __hash__(self) -> int:
    return hash(self.name)


class dtypes:
  """Canonical interned dtype instances. Mirrors the small tinygrad-style registry."""

  bool_ = DType("bool", 8, "uint8_t", is_bool=True)
  uint8 = DType("uint8", 8, "uint8_t", is_integer=True, is_unsigned=True)
  int32 = DType("int32", 32, "int32_t", is_integer=True)
  int64 = DType("int64", 64, "int64_t", is_integer=True)
  float32 = DType("float32", 32, "float", is_floating=True)
  float64 = DType("float64", 64, "double", is_floating=True)

  _BY_NAME: dict[str, DType] = {}

  @classmethod
  def from_name(cls, name: str) -> DType:
    if not cls._BY_NAME:
      cls._BY_NAME.update({d.name: d for d in cls.all()})
    try:
      return cls._BY_NAME[name]
    except KeyError as e:
      raise ValueError(f"unknown dtype {name!r}; known names: {sorted(cls._BY_NAME)}") from e

  @classmethod
  def all(cls) -> tuple[DType, ...]:
    return (cls.bool_, cls.uint8, cls.int32, cls.int64, cls.float32, cls.float64)


def as_dtype(value: DType | str | None) -> DType:
  """Coerce a dtype name, a ``DType`` or ``None`` into a canonical ``DType``.

  ``None`` means ``float64``, scaly's default.
  """
  if value is None:
    return dtypes.float64
  if isinstance(value, DType):
    return value
  if isinstance(value, str):
    return dtypes.from_name(value)
  raise TypeError(f"cannot interpret {value!r} as a DType")


@dataclass(frozen=True, slots=True)
class DeviceSpec:
  """Where a region/function should run. ``host`` is the only kind."""

  kind: str = "host"

  def __post_init__(self) -> None:
    if self.kind != "host":
      raise ValueError(f"unsupported device kind {self.kind!r}; expected host")

  def __str__(self) -> str:
    return self.kind

  @staticmethod
  def parse(spec: "DeviceSpec | str | None") -> "DeviceSpec":
    if spec is None:
      return DeviceSpec()
    if isinstance(spec, DeviceSpec):
      return spec
    if isinstance(spec, str):
      return DeviceSpec(spec)
    raise TypeError(f"cannot interpret {spec!r} as a DeviceSpec")


def _check_shape(name: str, shape: tuple[int, ...]) -> None:
  if any(d < 0 for d in shape):
    raise ValueError(f"{name} shape cannot contain negative dimensions, got {shape}")


@dataclass(frozen=True, slots=True)
class SparsityPattern:
  """The structurally nonzero entries of a matrix, as coordinate lists ``rows`` and ``cols``.

  The entry order is the order of the values stored alongside the pattern. It is not necessarily
  row- or column-major. ``to_csr`` and ``to_csc`` return the permutation into either order.
  """

  shape: tuple[int, int]
  rows: tuple[int, ...]
  cols: tuple[int, ...]

  def __post_init__(self) -> None:
    for name in ("shape", "rows", "cols"):
      object.__setattr__(self, name, tuple(getattr(self, name)))
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

  def canonicalize(self) -> tuple[CanonicalPattern, tuple[int, ...]]:
    """Return the shared row-major pattern and the permutation that reorders its values.

    ``values_canonical[k] = values[permutation[k]]``. Stored zeros keep their coordinates. The
    pattern is what ``TensorType(shape, pattern=...)`` takes.
    """
    order = tuple(sorted(range(self.nnz), key=lambda i: (self.rows[i], self.cols[i])))
    return CanonicalPattern(self.shape, tuple(self.rows[i] for i in order), tuple(self.cols[i] for i in order)), order

  @staticmethod
  def empty(shape: tuple[int, int]) -> SparsityPattern:
    return SparsityPattern(shape, (), ())

  @staticmethod
  def dense(shape: tuple[int, int]) -> SparsityPattern:
    rows, cols = np.nonzero(np.ones(shape, dtype=bool))
    return SparsityPattern(shape, tuple(int(x) for x in rows), tuple(int(x) for x in cols))

  @staticmethod
  def from_mask(mask: np.ndarray) -> SparsityPattern:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
      raise ValueError(f"sparsity mask must be rank-2, got {mask.shape}")
    rows, cols = np.nonzero(mask)
    shape = (int(mask.shape[0]), int(mask.shape[1]))
    return SparsityPattern(shape, tuple(int(x) for x in rows), tuple(int(x) for x in cols))

  @staticmethod
  def from_csr(shape: tuple[int, int], row_ptr: Sequence[int], col_ind: Sequence[int]) -> SparsityPattern:
    row_ptr = tuple(int(x) for x in row_ptr)
    col_ind = tuple(int(x) for x in col_ind)
    _check_compressed_ptr("row_ptr", row_ptr, shape[0], len(col_ind))
    if any(c < 0 or c >= shape[1] for c in col_ind):
      raise ValueError(f"CSR column indices out of bounds for shape {shape}")
    rows = tuple(r for r in range(shape[0]) for _ in range(row_ptr[r + 1] - row_ptr[r]))
    return SparsityPattern(shape, rows, col_ind)

  @staticmethod
  def from_csc(shape: tuple[int, int], col_ptr: Sequence[int], row_ind: Sequence[int]) -> SparsityPattern:
    col_ptr = tuple(int(x) for x in col_ptr)
    row_ind = tuple(int(x) for x in row_ind)
    _check_compressed_ptr("col_ptr", col_ptr, shape[1], len(row_ind))
    if any(r < 0 or r >= shape[0] for r in row_ind):
      raise ValueError(f"CSC row indices out of bounds for shape {shape}")
    cols = tuple(c for c in range(shape[1]) for _ in range(col_ptr[c + 1] - col_ptr[c]))
    return SparsityPattern(shape, row_ind, cols)

  def to_mask(self) -> np.ndarray:
    mask = np.zeros(self.shape, dtype=bool)
    mask[list(self.rows), list(self.cols)] = True
    return mask

  def to_csr(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """``(row_ptr, col_ind, val_perm)``: ``val_perm[k]`` is the COO position of CSR slot ``k``, so
    ``values_csr[k] = values[val_perm[k]]`` pairs a compact COO-ordered value buffer with the CSR
    indices. The COO order is arbitrary, for example piece by piece for a mapped sparse Jacobian."""
    order = sorted(range(self.nnz), key=lambda i: (self.rows[i], self.cols[i]))
    row_ptr = [0] * (self.shape[0] + 1)
    for i in order:
      row_ptr[self.rows[i] + 1] += 1
    for r in range(self.shape[0]):
      row_ptr[r + 1] += row_ptr[r]
    return tuple(row_ptr), tuple(self.cols[i] for i in order), tuple(order)

  def to_csc(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """``(col_ptr, row_ind, val_perm)``, the compressed-column counterpart of ``to_csr``."""
    order = sorted(range(self.nnz), key=lambda i: (self.cols[i], self.rows[i]))
    col_ptr = [0] * (self.shape[1] + 1)
    for i in order:
      col_ptr[self.cols[i] + 1] += 1
    for c in range(self.shape[1]):
      col_ptr[c + 1] += col_ptr[c]
    return tuple(col_ptr), tuple(self.rows[i] for i in order), tuple(order)


def _check_compressed_ptr(name: str, ptr: tuple[int, ...], n_outer: int, nnz: int) -> None:
  if len(ptr) != n_outer + 1:
    raise ValueError(f"{name} must have length {n_outer + 1}, got {len(ptr)}")
  if not ptr or ptr[0] != 0 or ptr[-1] != nnz:
    raise ValueError(f"{name} must start at 0 and end at nnz={nnz}")
  if any(a > b for a, b in zip(ptr, ptr[1:])):
    raise ValueError(f"{name} must be nondecreasing")


_PATTERN_CACHE: weakref.WeakValueDictionary[tuple, CanonicalPattern] = weakref.WeakValueDictionary()
_PATTERN_LOCK = Lock()


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False, init=False)
class CanonicalPattern:
  """The stored coordinates of a sparse matrix, in row-major order.

  Equal coordinates give the same object, so ``is`` compares two patterns. Build one with
  ``SparsityPattern.canonicalize``. The digest identifies the shape and coordinates.
  """

  shape: tuple[int, int]
  rows: tuple[int, ...]
  cols: tuple[int, ...]
  digest: str

  def __new__(cls, shape: tuple[int, int], rows: tuple[int, ...], cols: tuple[int, ...]) -> CanonicalPattern:
    pattern = SparsityPattern(shape, rows, cols)
    coordinates = sorted((index(r), index(c)) for r, c in zip(pattern.rows, pattern.cols))
    key = ((index(pattern.shape[0]), index(pattern.shape[1])), tuple(r for r, _ in coordinates), tuple(c for _, c in coordinates))
    with _PATTERN_LOCK:
      if (cached := _PATTERN_CACHE.get(key)) is not None:
        return cached
      instance = object.__new__(cls)
      for name, value in zip(("shape", "rows", "cols"), key, strict=True):
        object.__setattr__(instance, name, value)
      object.__setattr__(instance, "digest", sha256(repr(key).encode("ascii")).hexdigest())
      _PATTERN_CACHE[key] = instance
      return instance

  @property
  def nnz(self) -> int:
    return len(self.rows)

  def union(self, other: CanonicalPattern) -> CanonicalPattern:
    """Return the coordinates stored in either pattern."""
    return self._combine(other, intersection=False)

  def intersection(self, other: CanonicalPattern) -> CanonicalPattern:
    """Return the coordinates stored in both patterns."""
    return self._combine(other, intersection=True)

  def _combine(self, other: CanonicalPattern, *, intersection: bool) -> CanonicalPattern:
    if self.shape != other.shape:
      raise ValueError(f"pattern shapes differ: {self.shape} and {other.shape}")
    a, b = set(zip(self.rows, self.cols)), set(zip(other.rows, other.cols))
    coordinates = a & b if intersection else a | b
    return CanonicalPattern(self.shape, tuple(r for r, _ in coordinates), tuple(c for _, c in coordinates))

  def matmul(self, other: CanonicalPattern) -> CanonicalPattern:
    """Return the Boolean matrix product of two patterns."""
    from scipy import sparse

    if self.shape[1] != other.shape[0]:
      raise ValueError(f"cannot multiply patterns {self.shape} and {other.shape}")
    a = sparse.coo_array((np.ones(self.nnz, dtype=bool), (self.rows, self.cols)), shape=self.shape).tocsr()
    b = sparse.coo_array((np.ones(other.nnz, dtype=bool), (other.rows, other.cols)), shape=other.shape).tocsr()
    product = (a @ b).tocoo()
    return CanonicalPattern((self.shape[0], other.shape[1]), tuple(map(int, product.row)), tuple(map(int, product.col)))


@dataclass(frozen=True, slots=True)
class TensorType:
  """The shape and dtype of an expression, and whether it is differentiable (``diff``).

  A ``pattern`` makes the tensor sparse: it lists the coordinates stored in each matrix over the
  last two axes, and every other entry is zero. Leading axes are batch axes sharing the pattern.
  """

  shape: tuple[int, ...] = ()
  dtype: DType = dtypes.float64
  diff: bool = True
  pattern: CanonicalPattern | None = None

  def __post_init__(self) -> None:
    _check_shape("tensor", self.shape)
    if not isinstance(self.dtype, DType):
      object.__setattr__(self, "dtype", as_dtype(self.dtype))
    if self.dtype.name == "uint8" or self.dtype.is_unsigned and self.dtype.bits == 8:
      raise ValueError("unsupported expression dtype uint8")
    if self.pattern is not None:
      if not isinstance(self.pattern, CanonicalPattern):
        raise TypeError("tensor pattern must be canonical; use SparsityPattern.canonicalize()")
      if self.ndim < 2 or self.shape[-2:] != self.pattern.shape:
        raise ValueError(f"tensor last two axes {self.shape[-2:]} do not match pattern shape {self.pattern.shape}")

  @property
  def ndim(self) -> int:
    return len(self.shape)

  @property
  def size(self) -> int:
    return reduce(mul, self.shape, 1)

  @property
  def storage_shape(self) -> tuple[int, ...]:
    """The shape of the stored values: ``shape`` when dense, the batch axes and ``nnz`` when sparse."""
    return self.shape if self.pattern is None else (*self.shape[:-2], self.pattern.nnz)

  @property
  def storage_size(self) -> int:
    """The number of stored values, including explicit zeros."""
    return reduce(mul, self.storage_shape, 1)

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


def stored_indices(type_: TensorType) -> np.ndarray:
  """Map canonical stored slots to flat logical coordinates."""
  pattern = type_.pattern
  if pattern is None:
    return np.arange(type_.size, dtype=np.int64)
  local = np.asarray(pattern.rows, dtype=np.int64) * pattern.shape[1] + np.asarray(pattern.cols, dtype=np.int64)
  batches = reduce(mul, type_.shape[:-2], 1)
  return (np.arange(batches, dtype=np.int64)[:, None] * (pattern.shape[0] * pattern.shape[1]) + local).reshape(-1)


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


def _array_data(value: np.ndarray) -> np.ndarray:
  return value[tuple(slice(0, 1) if stride == 0 else slice(None) for stride in value.strides)] if value.ndim else value


def _uniform_value(value: np.ndarray) -> np.generic | None:
  return value.flat[0] if value.size and all(stride == 0 or dim <= 1 for stride, dim in zip(value.strides, value.shape, strict=True)) else None


def frozen(value: Any) -> Any:
  """``value`` as an immutable copy, for an interned node's ``attrs``: a mapping becomes a read-only
  view, a list a tuple and an array a read-only copy, all the way down."""
  if isinstance(value, np.ndarray):
    compact = _array_data(value)
    out = compact.copy()
    out.flags.writeable = False
    return np.broadcast_to(out, value.shape) if compact.shape != value.shape else out
  if isinstance(value, Mapping):
    return MappingProxyType({k: frozen(v) for k, v in value.items()})
  if isinstance(value, list) or type(value) is tuple:
    return tuple(frozen(v) for v in value)
  return value
