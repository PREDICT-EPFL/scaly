"""The universal C ABI: the entry signature, the status codes, the symbol mangling, and the typed
buffer struct."""

from __future__ import annotations

from dataclasses import dataclass

from ..ir.types import DType, as_dtype
from ..utils.names import c_ident as c_ident

C_API_SIGNATURE = "int f(const double** arg, double** res, int* iw, double* w, void* mem)"

# The status codes every generated entry returns (docs/how_it_works/c_abi.md).
ABI_STATUS = {"SCALY_SUCCESS": 0, "SCALY_ERR_NULL_ABI": 1, "SCALY_ERR_NULL_WORK": 2, "SCALY_ERR_NULL_RESULT": 3, "SCALY_ERR_NULL_INPUT": 4}


def abi_status_defines(*, guarded: bool = False) -> list[str]:
  """``ABI_STATUS`` as C defines: plain for a generated source, ``#ifndef``-guarded for a header,
  which several generated modules may include into one translation unit."""
  if not guarded:
    return [f"#define {name} {code}" for name, code in ABI_STATUS.items()]
  return [line for name, code in ABI_STATUS.items() for line in (f"#ifndef {name}", f"#define {name} {code}", "#endif")]


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
  """The universal ABI entry signature, spelled for a given symbol name."""
  return C_API_SIGNATURE.replace(" f(", f" {symbol}(")
