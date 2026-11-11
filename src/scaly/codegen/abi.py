"""The pointer ABI shared by every generated function: the entry signature, the status codes and
the symbol mangling (``docs/how_it_works/generated_interface.md``). The typed structs and the C++
``Buffer`` layered on top are an API, not part of the ABI; they render in ``aot.py`` and ``cpp.py``."""

from __future__ import annotations

from ..function.concrete import ConcreteFunction
from ..function.model import Function, as_concrete
from ..ir.layout import SparseLayout
from ..ir.types import DType, dtypes
from ..passes.lowering import workspace_dtype
from ..utils.names import c_ident as c_ident, NameScope

# The pointee of ``arg``, ``res`` and ``w`` for each workspace dtype; a uint8 workspace is addressed in bytes.
ENTRY_C_TYPES = {dtypes.float64: "double", dtypes.float32: "float", dtypes.uint8: "void"}

# The status codes every generated entry returns (docs/how_it_works/generated_interface.md).
ABI_STATUS = {"SCALY_SUCCESS": 0, "SCALY_ERR_NULL_ABI": 1, "SCALY_ERR_NULL_WORK": 2, "SCALY_ERR_NULL_RESULT": 3, "SCALY_ERR_NULL_INPUT": 4}


def abi_status_defines(*, guarded: bool = False) -> list[str]:
  """``ABI_STATUS`` as C defines: plain for a generated source, ``#ifndef``-guarded for a header,
  which several generated modules may include into one translation unit."""
  if not guarded:
    return [f"#define {name} {code}" for name, code in ABI_STATUS.items()]
  return [line for name, code in ABI_STATUS.items() for line in (f"#ifndef {name}", f"#define {name} {code}", "#endif")]


def buffer_idents(fun: ConcreteFunction) -> tuple[list[str], list[str]]:
  """The identifiers the typed wrappers use for ``fun``'s inputs and outputs. A name is ``c_ident``
  of itself, with ``_in``/``_out`` appended when the same name is both an input and an output (a
  solver ConcreteFunction's warm start and solution), and a trailing underscore when it would shadow the
  entry the wrapper calls, its workspace or its option context."""
  symbol = c_ident(fun.name)
  shared = set(fun.input_names) & set(fun.output_names)
  names = NameScope(
    (symbol, f"{symbol}_with_options", "arg", "res", "workspace", "workspace_t", "solver_options", "call", "sz_arg", "sz_res", "sz_iw", "sz_w"),
    header=True,
  )
  names.occupied.add(f"{symbol}_FRAME_BYTES")

  def ident(name: str, suffix: str) -> str:
    out = c_ident(name + suffix if name in shared else name)
    if names.contains(out):
      out += "_"
    return names.allocate(out, suffixes=("_t",))

  return [ident(n, "_in") for n in fun.input_names], [ident(n, "_out") for n in fun.output_names]


def wrapper_param_idents(symbol: str, inputs: list[str], outputs: list[str], *, type_prefix: str) -> tuple[list[str], list[str], str]:
  """Allocate typed-wrapper parameters without hiding entry points, local arrays or typedefs."""
  names = NameScope(
    (
      symbol,
      f"{symbol}_with_options",
      "arg",
      "res",
      "solver_options",
      "scaly_solver_option",
      f"{type_prefix}workspace_t",
      *(f"{type_prefix}{ident}_t" for ident in (*inputs, *outputs)),
    ),
    header=True,
  )
  return [names.allocate(n) for n in inputs], [names.allocate(n) for n in outputs], names.allocate("workspace")


def c_api_signature(fun: Function | ConcreteFunction, symbol: str | None = None, *, solver_options: bool = False) -> str:
  """The entry signature of ``fun``, named ``symbol`` or after ``fun``. ``arg``, ``res`` and ``w``
  point to ``double`` when every leaf is float64, to ``float`` when every leaf is float32, and to
  ``void`` otherwise, with ``SZ_W`` then counting bytes. ``mem`` follows CasADi 3.8: an ``int``
  memory handle a stateful function would index a pool with. Every Scaly function is stateless and
  ignores it."""
  fun = as_concrete(fun)
  t = ENTRY_C_TYPES[workspace_dtype(fun)]
  options = ", const scaly_solver_option* const* solver_options" if solver_options else ""
  return f"int {symbol or c_ident(fun.name)}(const {t}** arg, {t}** res, int* iw, {t}* w, int mem{options})"


def leaf_c_type(dtype: DType) -> str:
  """The C type a leaf's values cross the entry as. A bool leaf is C's ``bool``, one byte holding 0
  or 1, which the kernel reads and writes as ``uint8_t``."""
  return "bool" if dtype.is_bool else dtype.c_type


def workspace_c_type(fun: ConcreteFunction) -> str:
  """The element type of a typed header's workspace buffer: the entry's pointee, or bytes."""
  t = ENTRY_C_TYPES[workspace_dtype(fun)]
  return "unsigned char" if t == "void" else t


def sparse_layout_tables(layout: SparseLayout) -> tuple[tuple[str, tuple[int, ...]], ...]:
  """The local index tables in a sparse leaf's declared value order, without value permutations."""
  if layout.format == "coo":
    tables = (("rows", layout.rows), ("cols", layout.cols))
  elif layout.format == "csr":
    tables = (("csr_row_ptr", layout.indptr), ("csr_col_ind", layout.indices))
  else:
    tables = (("csc_col_ptr", layout.indptr), ("csc_row_ind", layout.indices))
  return tuple((name, tuple(map(int, values))) for name, values in tables)
