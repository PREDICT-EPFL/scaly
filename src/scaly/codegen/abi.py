"""The pointer ABI shared by every generated function: the entry signature, the status codes and
the symbol mangling (``docs/how_it_works/generated_interface.md``). The typed structs and the C++
``Buffer`` layered on top are an API, not part of the ABI; they render in ``aot.py`` and ``cpp.py``."""

from __future__ import annotations

from dataclasses import dataclass

from ..function import ConcreteFunction
from ..ir.types import DType, as_dtype
from ..utils.names import c_ident as c_ident

# ``mem`` follows CasADi 3.8: an ``int`` memory handle a stateful function would index a pool with.
# Every scaly function is stateless and ignores it.
C_API_SIGNATURE = "int f(const double** arg, double** res, int* iw, double* w, int mem)"

# The status codes every generated entry returns (docs/how_it_works/generated_interface.md).
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


def buffer_idents(fun: ConcreteFunction) -> tuple[list[str], list[str]]:
  """The identifiers the typed wrappers use for ``fun``'s inputs and outputs. A name is ``c_ident``
  of itself, with ``_in``/``_out`` appended when the same name is both an input and an output (a
  solver Function's warm start and solution), and a trailing underscore when it would shadow the
  entry the wrapper calls or its workspace."""
  symbol = c_ident(fun.name)
  shared = set(fun.input_names) & set(fun.output_names)

  def ident(name: str, suffix: str) -> str:
    out = c_ident(name + suffix if name in shared else name)
    return f"{out}_" if out in (symbol, "workspace") else out

  return [ident(n, "_in") for n in fun.input_names], [ident(n, "_out") for n in fun.output_names]


def c_api_signature(symbol: str = "f") -> str:
  """The universal ABI entry signature, spelled for a given symbol name."""
  return C_API_SIGNATURE.replace(" f(", f" {symbol}(")
