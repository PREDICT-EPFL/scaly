from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from collections.abc import Sequence
from typing import Literal

import numpy as np

Lowering = Literal["auto", "scalar", "block", "opaque"]


@dataclass(frozen=True, slots=True)
class DType:
  """Interned dtype descriptor.

  See ``dtypes`` for the canonical instances. ``DType`` instances compare equal
  to their string ``name`` (e.g. ``DType("float64", ...) == "float64"``) so older
  call sites that round-trip the dtype through ``str`` keep working through the
  Phase 1 migration.
  """

  name: str
  bits: int
  c_type: str
  is_floating: bool = False
  is_integer: bool = False
  is_bool: bool = False

  @property
  def itemsize(self) -> int:
    return self.bits // 8

  def numpy(self) -> np.dtype:
    return np.dtype(self.name)

  def __str__(self) -> str:
    return self.name

  def __eq__(self, other: object) -> bool:  # backward compat with string dtype
    if isinstance(other, DType):
      return self.name == other.name
    if isinstance(other, str):
      return self.name == other
    return NotImplemented

  def __hash__(self) -> int:
    return hash(self.name)


class dtypes:
  """Canonical interned dtype instances. Mirrors the small tinygrad-style registry."""

  bool_ = DType("bool", 8, "uint8_t", is_bool=True)
  int32 = DType("int32", 32, "int32_t", is_integer=True)
  int64 = DType("int64", 64, "int64_t", is_integer=True)
  float32 = DType("float32", 32, "float", is_floating=True)
  float64 = DType("float64", 64, "double", is_floating=True)

  _BY_NAME: dict[str, DType] = {}

  @classmethod
  def from_name(cls, name: str) -> DType:
    if not cls._BY_NAME:
      cls._BY_NAME.update({d.name: d for d in (cls.bool_, cls.int32, cls.int64, cls.float32, cls.float64)})
    try:
      return cls._BY_NAME[name]
    except KeyError as e:
      raise ValueError(f"unknown dtype {name!r}; supported: {sorted(cls._BY_NAME)}") from e

  @classmethod
  def all(cls) -> tuple[DType, ...]:
    return (cls.bool_, cls.int32, cls.int64, cls.float32, cls.float64)


def as_dtype(value: DType | str | None) -> DType:
  if value is None:
    return dtypes.float64
  if isinstance(value, DType):
    return value
  if isinstance(value, str):
    return dtypes.from_name(value)
  raise TypeError(f"cannot interpret {value!r} as a DType")


@dataclass(frozen=True, slots=True)
class DeviceSpec:
  """Where a region/function should run.

  ``kind`` is one of ``host``, ``cuda``, ``opencl``, ``metal``. ``index`` is the
  device index for backends that have one. Backends register their
  ``BackendSupport`` separately; ``DeviceSpec`` is only the policy value.
  """

  kind: str = "host"
  index: int = 0

  def __post_init__(self) -> None:
    if self.kind not in {"host", "cuda", "opencl", "metal"}:
      raise ValueError(f"unsupported device kind {self.kind!r}; expected host/cuda/opencl/metal")
    if self.index < 0:
      raise ValueError(f"device index must be non-negative, got {self.index}")

  def __str__(self) -> str:
    return self.kind if self.kind == "host" else f"{self.kind}:{self.index}"

  @staticmethod
  def parse(spec: "DeviceSpec | str | None") -> "DeviceSpec":
    if spec is None:
      return DeviceSpec()
    if isinstance(spec, DeviceSpec):
      return spec
    if isinstance(spec, str):
      if spec == "host":
        return DeviceSpec("host", 0)
      if ":" in spec:
        kind, idx = spec.split(":", 1)
        return DeviceSpec(kind, int(idx))
      return DeviceSpec(spec, 0)
    raise TypeError(f"cannot interpret {spec!r} as a DeviceSpec")


@dataclass(frozen=True, slots=True)
class BackendSupport:
  name: str
  dtypes: frozenset[DType] = field(default_factory=frozenset)

  def supports(self, dtype: DType) -> bool:
    return dtype in self.dtypes


BACKEND_SUPPORT: dict[str, BackendSupport] = {
  "host": BackendSupport("host", frozenset(dtypes.all())),
  # placeholder capability tables for the lowering policy: backends that exist
  # at policy time but cannot lower yet still record their dtype constraints
  # so an early diagnostic can reject e.g. float64 on Metal.
  "cuda": BackendSupport("cuda", frozenset({dtypes.float32, dtypes.float64, dtypes.int32, dtypes.int64, dtypes.bool_})),
  "opencl": BackendSupport("opencl", frozenset({dtypes.float32, dtypes.float64, dtypes.int32, dtypes.int64, dtypes.bool_})),
  "metal": BackendSupport("metal", frozenset({dtypes.float32, dtypes.int32, dtypes.int64, dtypes.bool_})),
}


def backend_supports(device: DeviceSpec, dtype: DType) -> bool:
  support = BACKEND_SUPPORT.get(device.kind)
  return support is not None and support.supports(dtype)


def _check_shape(name: str, shape: tuple[int, ...]) -> None:
  if any(d < 0 for d in shape):
    raise ValueError(f"{name} shape cannot contain negative dimensions, got {shape}")


@dataclass(frozen=True, slots=True)
class ScalarType:
  dtype: DType = dtypes.float64
  diff: bool = True

  def __post_init__(self) -> None:
    if not isinstance(self.dtype, DType):
      object.__setattr__(self, "dtype", as_dtype(self.dtype))


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

  def to_csr(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """``(row_ptr, col_ind, val_perm)``: ``val_perm[k]`` is the COO position of CSR slot ``k``, so
    ``values_csr[k] = values[val_perm[k]]`` pairs a compact COO-ordered value buffer with the CSR
    indices (the COO nnz order is arbitrary — e.g. piece-ordered on the structured MAP spjac path)."""
    order = sorted(range(self.nnz), key=lambda i: (self.rows[i], self.cols[i]))
    row_ptr = [0] * (self.shape[0] + 1)
    for i in order:
      row_ptr[self.rows[i] + 1] += 1
    for r in range(self.shape[0]):
      row_ptr[r + 1] += row_ptr[r]
    return tuple(row_ptr), tuple(self.cols[i] for i in order), tuple(order)

  def to_csc(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """``(col_ptr, row_ind, val_perm)`` — see ``to_csr``."""
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


@dataclass(frozen=True, slots=True)
class TensorType:
  shape: tuple[int, ...] = ()
  dtype: DType = dtypes.float64
  sparsity: SparsityType | None = None
  diff: bool = True

  def __post_init__(self) -> None:
    _check_shape("tensor", self.shape)
    if not isinstance(self.dtype, DType):
      object.__setattr__(self, "dtype", as_dtype(self.dtype))
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
