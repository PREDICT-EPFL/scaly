"""The C++ header (``lang="cpp"``): the guarded ``Buffer<T, Ns...>`` template with inline aligned
storage and a ``constexpr`` shape, and one namespace per function holding the buffer aliases,
``constexpr`` sizes and sparsity tables, and ``call``. It declares the same C kernel the C header
does (``docs/how_it_works/generated_interface.md``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .abi import abi_status_defines, buffer_idents, c_api_signature, c_ident
from .casadi import casadi_declarations, casadi_defines
from .solver import solver_stats_symbols
from ..function import Function
from ..solvers.stats import stats_c_defs

if TYPE_CHECKING:
  from ..ir.types import SparsityPattern

BUFFER_TEMPLATE = """#ifndef SCALY_BUFFER_HPP
#define SCALY_BUFFER_HPP
// A fixed-shape, row-major, 16-byte aligned buffer with inline storage: an aggregate to place on
// the stack, in a static, or behind make_unique. shape is the Expr shape; data is its flattening.
template <typename T, std::size_t... Ns>
struct alignas(16) Buffer {
  static constexpr std::size_t ndim = sizeof...(Ns);
  static constexpr std::array<std::size_t, ndim> shape = {Ns...};
  static constexpr std::size_t size = (Ns * ... * std::size_t{1});
  T data[size > 0 ? size : 1];
  T* ptr() { return data; }
  const T* ptr() const { return data; }
  template <typename... I> T& operator()(I... idx) { return data[offset(idx...)]; }
  template <typename... I> const T& operator()(I... idx) const { return data[offset(idx...)]; }
  template <typename... I> static constexpr std::size_t offset(I... idx) {
    static_assert(sizeof...(I) == ndim, "Buffer: one index per dimension");
    const std::size_t index[] = {static_cast<std::size_t>(idx)..., 0};
    std::size_t off = 0;
    for (std::size_t k = 0; k < ndim; ++k) {
      assert(index[k] < shape[k]);
      off = off * shape[k] + index[k];
    }
    return off;
  }
};
#endif"""


def _buffer_alias(ident: str, shape: tuple[int, ...]) -> str:
  dims = "".join(f", {d}" for d in shape)
  return f"using {ident}_t = Buffer<double{dims}>;"


def _std_array(name: str, values: tuple[int, ...], size: str) -> str:
  return f"constexpr std::array<int, {size}> {name} = {{{', '.join(str(v) for v in values)}}};"


def _sparse_namespace(name: str, sp: SparsityPattern) -> list[str]:
  row_ptr, col_ind, csr_perm = sp.to_csr()
  col_ptr, row_ind, csc_perm = sp.to_csc()
  return [
    f"namespace {c_ident(name)} {{",
    f"constexpr int nrow = {sp.shape[0]};",
    f"constexpr int ncol = {sp.shape[1]};",
    f"constexpr int nnz = {sp.nnz};",
    _std_array("rows", sp.rows, "nnz"),
    _std_array("cols", sp.cols, "nnz"),
    _std_array("csr_row_ptr", row_ptr, "nrow + 1"),
    _std_array("csr_col_ind", col_ind, "nnz"),
    _std_array("csr_val_perm", csr_perm, "nnz"),
    _std_array("csc_col_ptr", col_ptr, "ncol + 1"),
    _std_array("csc_row_ind", row_ind, "nnz"),
    _std_array("csc_val_perm", csc_perm, "nnz"),
    "}",
  ]


def render_cpp_header(fun: Function, backends: tuple[str, ...], sz_w: int, *, casadi: bool, sparsities: tuple[SparsityPattern | None, ...]) -> str:
  """The ``.hpp`` for ``fun``. The kernel symbols are declared ``extern "C"`` inside the function's
  namespace, since a namespace and a function cannot share the global name. C linkage keeps the
  symbol unmangled, so ``f::f`` is the same entry a C caller reaches as ``f``."""
  symbol = c_ident(fun.name)
  inputs, outputs = buffer_idents(fun)
  params = [f"const {i}_t& {i}" for i in inputs] + [f"{o}_t& {o}" for o in outputs] + ["workspace_t& workspace"]
  arg_init = ", ".join(f"{i}.ptr()" for i in inputs) or "nullptr"
  res_init = ", ".join(f"{o}.ptr()" for o in outputs) or "nullptr"
  lines = [
    "#pragma once",
    "",
    "#include <array>",
    "#include <cassert>",
    "#include <cstddef>",
    *(["#include <cstdint>", "", *stats_c_defs()] if backends else []),
    "",
    *abi_status_defines(guarded=True),
    *(["", *casadi_defines()] if casadi else []),
    "",
    f"#define {symbol}_SZ_ARG {len(fun.inputs)}",
    f"#define {symbol}_SZ_RES {len(fun.outputs)}",
    f"#define {symbol}_SZ_IW 0",
    f"#define {symbol}_SZ_W {sz_w}",
    "",
    BUFFER_TEMPLATE,
    "",
    f"namespace {symbol} {{",
    f"// The pointer ABI for {fun.name}; the same C symbols the C header declares.",
    'extern "C" ' + c_api_signature(symbol) + ";",
    *(f'extern "C" int {s}_stats(scaly_solver_stats* out);' for s in solver_stats_symbols(fun)),
    *(f'extern "C" {decl}' for decl in (casadi_declarations(symbol) if casadi else [])),
    "",
    *(_buffer_alias(i, e.shape) for i, e in zip(inputs, fun.inputs, strict=True)),
    *(_buffer_alias(o, e.shape) for o, e in zip(outputs, fun.outputs, strict=True)),
    f"using workspace_t = Buffer<double, {symbol}_SZ_W>;",
    f"constexpr int sz_arg = {symbol}_SZ_ARG;",
    f"constexpr int sz_res = {symbol}_SZ_RES;",
    f"constexpr int sz_iw = {symbol}_SZ_IW;",
    f"constexpr int sz_w = {symbol}_SZ_W;",
    "",
    f"inline int call({', '.join(params)}) {{",
    f"  const double* arg[sz_arg > 0 ? sz_arg : 1] = {{{arg_init}}};",
    f"  double* res[sz_res > 0 ? sz_res : 1] = {{{res_init}}};",
    f"  return {symbol}(arg, res, nullptr, sz_w ? workspace.ptr() : nullptr, 0);",
    "}",
  ]
  for name, sp in zip(fun.output_names, sparsities, strict=True):
    if sp is not None:
      lines += ["", *_sparse_namespace(name, sp)]
  lines += [f"}}  // namespace {symbol}"]
  return "\n".join(lines) + "\n"
