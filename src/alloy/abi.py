from __future__ import annotations

from dataclasses import dataclass

from .types import DType, as_dtype

C_API_SIGNATURE = "int f(const double** arg, double** res, int* iw, double* w, void* mem)"


@dataclass(frozen=True, slots=True)
class BufferType:
  dtype: DType
  shape: tuple[int, ...]
  name: str | None = None

  def __post_init__(self) -> None:
    if not isinstance(self.dtype, DType):
      object.__setattr__(self, "dtype", as_dtype(self.dtype))

  @property
  def size(self) -> int:
    n = 1
    for d in self.shape:
      n *= d
    return n

  def c_type(self) -> str:
    dims = "".join(f"[{d}]" for d in self.shape)
    name = "data" if self.name is None else self.name
    return f"struct {{ {self.dtype.c_type} {name}{dims}; }}"


def c_api_signature(symbol: str = "f") -> str:
  return C_API_SIGNATURE.replace(" f(", f" {symbol}(")
