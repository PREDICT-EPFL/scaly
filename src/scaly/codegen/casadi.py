"""The CasADi 3.8 compatible layer (``casadi=True``): the ``casadi_int``/``casadi_real`` defines, the
query functions CasADi's ``external`` and acados resolve, the compressed-column sparsity encoding,
the render-time layout check, and the gather that hands a compact sparse output to the caller in
compressed-column order (``docs/how_it_works/generated_interface.md``)."""

from __future__ import annotations

from dataclasses import dataclass

from .abi import c_ident
from ..function.concrete import ConcreteFunction
from ..ir.types import SparsityPattern

CASADI_QUERIES = (
  "n_in",
  "n_out",
  "name_in",
  "name_out",
  "default_in",
  "sparsity_in",
  "sparsity_out",
  "work",
  "work_bytes",
  "checkout",
  "release",
  "incref",
  "decref",
)
"""The suffixes ``casadi=True`` exports under ``<symbol>_``. acados needs the entry plus ``work``,
``sparsity_in``, ``sparsity_out``, ``n_in`` and ``n_out``; the rest make ``casadi.external`` work."""


def casadi_defines() -> list[str]:
  """``casadi_int`` and ``casadi_real`` as CasADi spells them: guarded macros with CasADi's own
  defaults, so ``casadi.external`` (which reads ``long long``) loads the library as built, and an
  acados build (which reads ``int``) compiles the same ``.c`` with ``-Dcasadi_int=int``. The two
  consumers disagree on the width, so no single binary serves both; the macro is the switch."""
  return ["#ifndef casadi_real", "#define casadi_real double", "#endif", "#ifndef casadi_int", "#define casadi_int long long int", "#endif"]


def casadi_declarations(symbol: str) -> list[str]:
  """The query prototypes, for a header."""
  return [
    f"casadi_int {symbol}_n_in(void);",
    f"casadi_int {symbol}_n_out(void);",
    f"const char* {symbol}_name_in(casadi_int i);",
    f"const char* {symbol}_name_out(casadi_int i);",
    f"casadi_real {symbol}_default_in(casadi_int i);",
    f"const casadi_int* {symbol}_sparsity_in(casadi_int i);",
    f"const casadi_int* {symbol}_sparsity_out(casadi_int i);",
    f"int {symbol}_work(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w);",
    f"int {symbol}_work_bytes(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w);",
    f"int {symbol}_checkout(void);",
    f"void {symbol}_release(int mem);",
    f"void {symbol}_incref(void);",
    f"void {symbol}_decref(void);",
  ]


def check_casadi_layout(fun: ConcreteFunction) -> None:
  """CasADi buffers are column-major and scaly's are row-major. Scalars, vectors and compact sparse
  outputs agree. A dense matrix with both dimensions above one would need a transpose, so it is
  rejected here rather than hand the caller a transposed matrix."""
  for kind, names, exprs, sparsities in (
    ("input", fun.input_names, fun.inputs, (None,) * len(fun.inputs)),
    ("output", fun.output_names, fun.outputs, fun.output_sparsities),
  ):
    for name, expr, sp in zip(names, exprs, sparsities, strict=True):
      if sp is None and (len(expr.shape) > 2 or (len(expr.shape) == 2 and min(expr.shape) > 1)):
        raise ValueError(
          f"casadi=True: {kind} {name!r} of {fun.name!r} has dense matrix shape {expr.shape}; CasADi is column-major, so only scalars, vectors and compact sparse outputs are supported"
        )


def _dense_dims(shape: tuple[int, ...]) -> tuple[int, int]:
  if len(shape) == 0:
    return (1, 1)
  if len(shape) == 1:
    return (shape[0], 1)
  return (shape[0], shape[1])


def casadi_sparsity(shape: tuple[int, ...], sp: SparsityPattern | None) -> tuple[int, ...]:
  """CasADi's compressed encoding: ``{nrow, ncol, 1}`` for dense, ``{nrow, ncol, colind..., row...}``
  for sparse. ``check_casadi_layout`` has already ruled out dense matrices."""
  if sp is None:
    return (*_dense_dims(shape), 1)
  col_ptr, row_ind, _ = sp.to_csc()
  return (*sp.shape, *col_ptr, *row_ind)


def csc_ordered(sp: SparsityPattern) -> SparsityPattern:
  """``sp`` with its nonzeros listed in compressed-column order, which is the order a CasADi caller
  reads the value buffer in under ``casadi=True``. The header's tables are rendered from this
  pattern so they describe the buffer actually written; its ``csc_val_perm`` is the identity."""
  _, _, perm = sp.to_csc()
  return SparsityPattern(sp.shape, tuple(sp.rows[i] for i in perm), tuple(sp.cols[i] for i in perm))


def casadi_output_sparsities(fun: ConcreteFunction) -> tuple[SparsityPattern | None, ...]:
  return tuple(None if sp is None else csc_ordered(sp) for sp in fun.output_sparsities)


def _needs_gather(sp: SparsityPattern | None) -> bool:
  return sp is not None and sp.to_csc()[2] != tuple(range(sp.nnz))


def casadi_scratch(fun: ConcreteFunction) -> int:
  """Doubles added to ``SZ_W`` for the compressed-column gather: ``nnz`` per compact sparse output
  whose native order is not already compressed-column (the QP path's patterns are, so they add 0)."""
  return sum(sp.nnz for sp in fun.output_sparsities if _needs_gather(sp) and sp is not None)


@dataclass(frozen=True, slots=True)
class CasadiGather:
  """How an entry hands compact sparse outputs over in compressed-column order: ``ptr`` redirects
  the body's writes for those outputs into ``w`` past the packed workspace, ``setup`` declares
  the scratch pointers, and ``epilogue`` permutes each into ``res[i]`` through ``csc_val_perm``."""

  ptr: dict[str, str]
  setup: list[str]
  epilogue: list[str]


def casadi_gather(fun: ConcreteFunction, base_sz_w: int) -> CasadiGather:
  ptr: dict[str, str] = {}
  setup: list[str] = []
  epilogue: list[str] = []
  offset = base_sz_w
  for i, (name, sp) in enumerate(zip(fun.output_names, fun.output_sparsities, strict=True)):
    if sp is None or not _needs_gather(sp):
      continue
    ident = c_ident(name)
    _, _, perm = sp.to_csc()
    ptr[name] = f"{ident}_native"
    setup.append(f"  double* {ident}_native = w + {offset};")
    epilogue += [
      f"  static const int {ident}_csc_val_perm[{sp.nnz}] = {{{', '.join(str(p) for p in perm)}}};",
      f"  for (int k = 0; k < {sp.nnz}; ++k) res[{i}][k] = {ident}_native[{ident}_csc_val_perm[k]];",
    ]
    offset += sp.nnz
  return CasadiGather(ptr, setup, epilogue)


def _switch(symbol: str, ret: str, name: str, cases: list[str], default: str) -> list[str]:
  return [
    f"{ret} {symbol}_{name}(casadi_int i) {{",
    "  switch (i) {",
    *(f"    case {k}: return {value};" for k, value in enumerate(cases)),
    f"    default: return {default};",
    "  }",
    "}",
  ]


def _c_string(text: str) -> str:
  return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_casadi_queries(fun: ConcreteFunction, sz_w: int) -> list[str]:
  """Definitions of the query functions for the translation unit. ``sz_w`` is the total ``SZ_W``
  the header quotes, gather scratch included."""
  symbol = c_ident(fun.name)
  n_in, n_out = len(fun.inputs), len(fun.outputs)
  tables: list[str] = []
  names_in: list[str] = []
  names_out: list[str] = []
  for kind, names, exprs, sparsities, into in (
    ("in", fun.input_names, fun.inputs, (None,) * n_in, names_in),
    ("out", fun.output_names, fun.outputs, fun.output_sparsities, names_out),
  ):
    for k, (name, expr, sp) in enumerate(zip(names, exprs, sparsities, strict=True)):
      values = casadi_sparsity(expr.shape, sp)
      tables.append(f"static const casadi_int {symbol}_s{kind}{k}[{len(values)}] = {{{', '.join(str(v) for v in values)}}};")
      into.append(f"{symbol}_s{kind}{k}")
  return [
    *tables,
    "",
    f"casadi_int {symbol}_n_in(void) {{ return {n_in}; }}",
    f"casadi_int {symbol}_n_out(void) {{ return {n_out}; }}",
    *_switch(symbol, "const char*", "name_in", [_c_string(n) for n in fun.input_names], "0"),
    *_switch(symbol, "const char*", "name_out", [_c_string(n) for n in fun.output_names], "0"),
    f"casadi_real {symbol}_default_in(casadi_int i) {{ (void)i; return 0; }}",
    *_switch(symbol, "const casadi_int*", "sparsity_in", names_in, "0"),
    *_switch(symbol, "const casadi_int*", "sparsity_out", names_out, "0"),
    f"int {symbol}_work(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w) {{",
    f"  if (sz_arg) *sz_arg = {n_in};",
    f"  if (sz_res) *sz_res = {n_out};",
    "  if (sz_iw) *sz_iw = 0;",
    f"  if (sz_w) *sz_w = {sz_w};",
    "  return 0;",
    "}",
    f"int {symbol}_work_bytes(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w) {{",
    f"  if (sz_arg) *sz_arg = {n_in} * sizeof(const casadi_real*);",
    f"  if (sz_res) *sz_res = {n_out} * sizeof(casadi_real*);",
    "  if (sz_iw) *sz_iw = 0;",
    f"  if (sz_w) *sz_w = {sz_w} * sizeof(casadi_real);",
    "  return 0;",
    "}",
    f"int {symbol}_checkout(void) {{ return 0; }}",
    f"void {symbol}_release(int mem) {{ (void)mem; }}",
    f"void {symbol}_incref(void) {{}}",
    f"void {symbol}_decref(void) {{}}",
  ]
