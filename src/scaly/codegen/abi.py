"""Generated entry signatures, buffer and workspace declarations, and status codes."""

from __future__ import annotations

from ..function.concrete import ConcreteFunction
from ..function.model import Function, as_concrete
from ..ir.layout import SparseLayout
from ..ir.types import DType, dtypes
from ..passes.program.pack_workspace import WORKSPACE_ALIGNMENT, WorkspaceLayout
from ..utils.names import c_ident as c_ident, NameScope

WORKSPACE_MEMBERS = {dtypes.float64: "f64", dtypes.float32: "f32", dtypes.int64: "i64", dtypes.int32: "i32", dtypes.bool_: "b", dtypes.uint8: "u8"}
ALIGNAS = [
  "#ifndef SCALY_ALIGNAS",
  "#ifdef __cplusplus",
  "#define SCALY_ALIGNAS(n) alignas(n)",
  "#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L",
  "#define SCALY_ALIGNAS(n) _Alignas(n)",
  "#elif defined(__GNUC__)",
  "#define SCALY_ALIGNAS(n) __attribute__((aligned(n)))",
  "#else",
  "#define SCALY_ALIGNAS(n)",
  "#endif",
  "#endif",
]

# The status codes every generated entry returns (docs/how_it_works/generated_interface.md).
ABI_STATUS = {
  "SCALY_SUCCESS": 0,
  "SCALY_ERR_NULL_ABI": 1,
  "SCALY_ERR_NULL_WORK": 2,
  "SCALY_ERR_NULL_RESULT": 3,
  "SCALY_ERR_NULL_INPUT": 4,
  "SCALY_ERR_ALLOC_WORK": 5,
}


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
  names.occupied.update((f"{symbol}_FRAME_BYTES", f"{symbol}_WORKSPACE_BYTES", f"{symbol}_abi"))

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
      "w",
      "solver_options",
      "scaly_solver_option",
      *(dtype.c_type for dtype in dtypes.all()),
      f"{type_prefix}workspace_t",
      *(f"{type_prefix}{ident}_t" for ident in (*inputs, *outputs)),
    ),
    header=True,
  )
  return [names.allocate(n) for n in inputs], [names.allocate(n) for n in outputs], names.allocate("workspace")


def c_api_signature(fun: Function | ConcreteFunction, symbol: str | None = None) -> str:
  """The typed C entry signature, with one parameter per leaf and a workspace pointer."""
  fun = as_concrete(fun)
  symbol = symbol or c_ident(fun.name)
  inputs, outputs = buffer_idents(fun)
  ins, outs, _ = wrapper_param_idents(symbol, inputs, outputs, type_prefix=f"{symbol}_")
  params = [
    *(f"const {symbol}_{ident}_t* {name}" for ident, name in zip(inputs, ins, strict=True)),
    *(f"{symbol}_{ident}_t* {name}" for ident, name in zip(outputs, outs, strict=True)),
    f"{symbol}_workspace_t* w",
  ]
  return f"void {symbol}({', '.join(params)})"


def pointer_signature(fun: ConcreteFunction, *, casadi: bool = False, solver_options: bool = False) -> str:
  """The pointer-array entry for the JIT and C++ wrappers, or the CasADi entry."""
  symbol = c_ident(fun.name)
  options = ", const scaly_solver_option* const* solver_options" if solver_options else ""
  if casadi:
    suffix = "_with_options" if solver_options else ""
    return f"int {symbol}{suffix}(const double** arg, double** res, casadi_int* iw, void* w, int mem{options})"
  suffix = "_abi_with_options" if solver_options else "_abi"
  return f"int {symbol}{suffix}(const void* const* arg, void* const* res, {symbol}_workspace_t* w{options})"


def leaf_c_type(dtype: DType) -> str:
  """The C type a leaf's values cross the entry as. A bool leaf is C's ``bool``, one byte holding 0
  or 1, which the kernel reads and writes as ``uint8_t``."""
  return "bool" if dtype.is_bool else dtype.c_type


def workspace_members(layout: WorkspaceLayout) -> list[str]:
  """The typed workspace fields in layout order."""
  return [f"  SCALY_ALIGNAS({WORKSPACE_ALIGNMENT}) {leaf_c_type(dtype)} {WORKSPACE_MEMBERS[dtype]}[{count}];" for dtype, count in layout] or [
    "  unsigned char unused;"
  ]


def buffer_structs(fun: ConcreteFunction, layout: WorkspaceLayout) -> list[str]:
  """The same type definitions for standalone source and the C and C++ headers."""
  symbol = c_ident(fun.name)
  guard = f"SCALY_TYPES_{symbol}"
  return [
    f"#ifndef {guard}",
    f"#define {guard}",
    *(
      f"typedef struct {{ SCALY_ALIGNAS(16) {leaf_c_type(e.type.dtype)} data[{max(1, e.type.storage_size)}]; }} {symbol}_{ident}_t;"
      for ident, e in zip(sum(buffer_idents(fun), []), (*fun.inputs, *fun.outputs), strict=True)
    ),
    "typedef struct {",
    *workspace_members(layout),
    f"}} {symbol}_workspace_t;",
    "#endif",
  ]


def sparse_layout_tables(layout: SparseLayout) -> tuple[tuple[str, tuple[int, ...]], ...]:
  """The local index tables in a sparse leaf's declared value order, without value permutations."""
  if layout.format == "coo":
    tables = (("rows", layout.rows), ("cols", layout.cols))
  elif layout.format == "csr":
    tables = (("csr_row_ptr", layout.indptr), ("csr_col_ind", layout.indices))
  else:
    tables = (("csc_col_ptr", layout.indptr), ("csc_row_ind", layout.indices))
  return tuple((name, tuple(map(int, values))) for name, values in tables)
