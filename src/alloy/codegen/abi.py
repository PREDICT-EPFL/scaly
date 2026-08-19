"""The universal C ABI: the entry signature, the status codes, the symbol mangling, and the typed
buffer struct."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..ir.types import DType, as_dtype

C_API_SIGNATURE = "int f(const double** arg, double** res, int* iw, double* w, void* mem)"

# The status codes every generated entry returns (docs/how_it_works/c_abi.md).
ABI_STATUS = {"ALLOY_SUCCESS": 0, "ALLOY_ERR_NULL_ABI": 1, "ALLOY_ERR_NULL_WORK": 2, "ALLOY_ERR_NULL_RESULT": 3, "ALLOY_ERR_NULL_INPUT": 4}


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


def c_ident(name: str) -> str:
  """Sanitize an alloy name into the C identifier codegen exports it under.

  Function, buffer, var and callee names may contain ``:`` (derivative names like ``fwd:eq:z``) or
  other non-identifier characters. Names colliding with the emitters' own identifiers (the ``w``
  workspace tail, the ``arg``/``res``/``iw``/``mem`` ABI params) are suffixed with ``_``. Every
  renderer and the JIT's symbol lookup go through this one function, so they cannot disagree.
  """
  ident = re.sub(r"\W", "_", name)
  if ident in ("w", "arg", "res", "iw", "mem"):
    ident += "_"
  return f"_{ident}" if ident[:1].isdigit() else ident
