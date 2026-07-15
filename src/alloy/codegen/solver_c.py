"""C codegen for ``Ops.SOLVER_CALL`` — emits a raw function that drives the
vendored PIQP / IPOPT C interfaces.

This is the one sanctioned non-Program-IR renderer (rule 6 in
``docs/program_ir_migration.md``): a solver Function's body is shape-specific
enough that the cleanest implementation is a small hand-written template per
backend, parameterised by the descriptor. The oracle Functions the template
drives are *not* hand-written — they lower through Program IR like any other
host Function and are rendered as ``<oracle>_raw`` by ``codegen/program_c``.

Outer functions that contain a solver as a callee lower through Program IR with
the ``SolverFunction`` callee treated as opaque (``lowering.lower_function``):
the solver renders to a ``static void qp_xxx_raw(...)`` body here, and the
caller's lowered ``CALL`` emits a ``qp_xxx_raw(...)`` invocation.
``codegen/c.render_c_source`` orchestrates the whole translation unit, ordering
the solver wrapper after its (Program-IR) oracle PROCs. The JIT detects the
PIQP/IPOPT linkage need through ``uses_piqp(...)`` / ``uses_ipopt(...)``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from alloy.expr import topo
from alloy.function import Function
from alloy.ops import Ops
from alloy.toolchain import SolverLibraryError, solver_compile_flags as _toolchain_solver_compile_flags, solver_diagnostic, solver_paths
from alloy.types import SparsityType

if TYPE_CHECKING:
  from alloy.solvers.solver_function import SolverDescriptor


def solver_include_dir() -> Path:
  """Return the first discovered solver C header directory."""
  paths = solver_paths()
  if not paths.include_dirs:
    raise SolverLibraryError(solver_diagnostic())
  return paths.include_dirs[0]


def solver_lib_dir() -> Path:
  """Return the first discovered solver shared-library directory."""
  paths = solver_paths()
  if not paths.lib_dirs:
    raise SolverLibraryError(solver_diagnostic())
  return paths.lib_dirs[0]


def _c_ident(name: str) -> str:
  """Must match ``codegen.c._c_ident`` and ``codegen.program_c._c_ident``."""
  ident = re.sub(r"\W", "_", name)
  if ident in ("w", "arg", "res", "iw", "mem"):
    ident += "_"
  return f"_{ident}" if ident[:1].isdigit() else ident


def _raw_symbol(fun: Function) -> str:
  return f"{_c_ident(fun.name)}_raw"


def is_solver_function(fun: Function) -> bool:
  return hasattr(fun, "descriptor") and getattr(getattr(fun, "descriptor", None), "backend", None) in {"piqp", "ipopt"}


def _descriptor(fun: Function) -> SolverDescriptor:
  return fun.descriptor  # ty: ignore[unresolved-attribute]


# ---------------------------------------------------------------------------
# Predicates / dependencies the outer renderer consumes.
# ---------------------------------------------------------------------------


def solver_callees(fun: Function) -> list[Function]:
  """Return the inner Functions a solver Function depends on at codegen time."""
  if not is_solver_function(fun):
    return []
  desc = _descriptor(fun)
  out: list[Function] = []
  for cand in (desc.oracle, desc.base, desc.grad, desc.jac, desc.hess, desc.bounds):
    if cand is not None and cand not in out:
      out.append(cand)
  return out


def uses_piqp(fun: Function) -> bool:
  return _uses_backend(fun, "piqp", set())


def uses_ipopt(fun: Function) -> bool:
  return _uses_backend(fun, "ipopt", set())


def _uses_backend(fun: Function, backend: str, seen: set[int]) -> bool:
  if id(fun) in seen:
    return False
  seen.add(id(fun))
  if is_solver_function(fun) and _descriptor(fun).backend == backend:
    return True
  for node in topo(fun.outputs):
    if node.op in {Ops.CALL, Ops.MAP}:
      if _uses_backend(node.attrs["callee"], backend, seen):
        return True
    if node.op == Ops.SOLVER_CALL:
      if node.attrs["solver"].backend == backend:
        return True
  return False


def solver_includes(fun: Function) -> list[str]:
  out: list[str] = []
  if uses_piqp(fun):
    out.append('#include "piqp/piqp.h"')
  if uses_ipopt(fun):
    out.append('#include "coin-or/IpStdCInterface.h"')
  return out


def solver_compile_flags(fun: Function, *, rpath: bool = True) -> list[str]:
  """Compiler/linker flags an AOT consumer needs for ``fun``.

  Returns ``[]`` when ``fun`` does not transitively reach any solver.
  Otherwise: plugin package ``-I`` / ``-L`` paths, ``-lpiqpc`` /
  ``-lipopt`` (whichever apply), and an ``-Wl,-rpath`` pointing at the
  vendored lib directory so the resulting binary finds the shared libs at
  load time without ``LD_LIBRARY_PATH`` / ``DYLD_LIBRARY_PATH`` overrides.

  Set ``rpath=False`` if the consumer plans to bundle the libs elsewhere and
  will set the rpath / install_name themselves.
  """
  flags: list[str] = []
  needs_piqp = uses_piqp(fun)
  needs_ipopt = uses_ipopt(fun)
  if not (needs_piqp or needs_ipopt):
    return flags
  return _toolchain_solver_compile_flags(needs_piqp, needs_ipopt, rpath=rpath)


def solver_stats_symbols(fun: Function) -> tuple[str, ...]:
  """C identifiers for every solver wrapper reachable from ``fun``."""
  found: dict[str, Function] = {}
  seen: set[int] = set()

  def visit(fn: Function) -> None:
    if id(fn) in seen:
      return
    seen.add(id(fn))
    if is_solver_function(fn):
      symbol = _c_ident(fn.name)
      if symbol in found and found[symbol] is not fn:
        raise ValueError(f"duplicate solver symbol {symbol!r} in one generated translation unit")
      found[symbol] = fn
      for callee in solver_callees(fn):
        visit(callee)
      return
    for node in topo(fn.outputs):
      if node.op in {Ops.CALL, Ops.MAP}:
        visit(node.attrs["callee"])

  visit(fun)
  return tuple(found)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def render_solver_raw(fun: Function) -> list[str]:
  desc = _descriptor(fun)
  if desc.backend == "piqp":
    body = _render_piqp_raw(fun, desc)
  elif desc.backend == "ipopt":
    body = _render_ipopt_raw(fun, desc)
  else:
    raise NotImplementedError(f"C codegen for solver backend {desc.backend!r} not yet implemented")
  symbol = _c_ident(fun.name)
  return [
    f"static alloy_solver_stats {symbol}_stats_data;",
    *body,
    "",
    f"int {symbol}_stats(alloy_solver_stats* out) {{",
    "  if (!out) return 1;",
    f"  *out = {symbol}_stats_data;",
    "  return 0;",
    "}",
  ]


def _csc_tables(name: str, sp: SparsityType | None) -> list[str]:
  """Static CSC pattern tables for one QP matrix. The pattern is constructed
  in CSC order by ``qp._qp_matrix_sparsity`` so the compact value buffer needs
  no runtime permutation — enforced here."""
  assert sp is not None
  col_ptr, row_ind, val_perm = sp.to_csc()
  assert val_perm == tuple(range(len(val_perm))), "sparse QP pattern must be CSC-ordered (identity value permutation)"
  return [
    f"  static piqp_int {name}_p[{len(col_ptr)}] = {{ {', '.join(str(v) for v in col_ptr)} }};",
    f"  static piqp_int {name}_i[{len(row_ind)}] = {{ {', '.join(str(v) for v in row_ind)} }};",
  ]


def _render_piqp_raw(fun: Function, desc: SolverDescriptor) -> list[str]:
  symbol = _c_ident(fun.name)
  raw = _raw_symbol(fun)
  n, p, m = desc.n, desc.n_eq, desc.n_ineq
  oracle = desc.oracle
  assert oracle is not None, "PIQP solver descriptor must carry an oracle Function"
  sparse = desc.sparse
  if sparse:
    assert desc.P_sparsity is not None
    assert (desc.A_sparsity is not None) == bool(p) and (desc.G_sparsity is not None) == bool(m)
  nnz_P = desc.P_sparsity.nnz if sparse and desc.P_sparsity is not None else 0
  nnz_A = desc.A_sparsity.nnz if sparse and desc.A_sparsity is not None else 0
  nnz_G = desc.G_sparsity.nnz if sparse and desc.G_sparsity is not None else 0

  # Map solver input order to C parameter names: in0..in{n-1}.
  # Inputs: x0, lam_eq0, lam_ineq0, then params in order.
  param_count = len(desc.param_names)
  oracle_param_args = [f"in{3 + i}" for i in range(param_count)]

  # Oracle output ordering (set by qp.py): P, c, [A_eq, b_eq], [G_ineq, l_ineq, u_ineq], x_lb, x_ub.
  # Sparse wrappers get compact CSC-ordered value buffers for P/A/G.
  oracle_outs: list[tuple[str, int]] = []  # (buffer C name, size)
  oracle_outs.append(("P_buf", nnz_P if sparse else n * n))
  oracle_outs.append(("c_buf", n))
  if p:
    oracle_outs.append(("A_buf", nnz_A if sparse else p * n))
    oracle_outs.append(("b_buf", p))
  if m:
    oracle_outs.append(("G_buf", nnz_G if sparse else m * n))
    oracle_outs.append(("lineq_buf", m))
    oracle_outs.append(("uineq_buf", m))
  oracle_outs.append(("xlb_buf", n))
  oracle_outs.append(("xub_buf", n))

  # Parameters for the raw signature.
  c_inputs = [f"const double* in{i}" for i in range(len(desc.input_signature))]
  c_outputs = [f"double* out{i}" for i in range(len(desc.output_signature))]
  params = [*c_inputs, *c_outputs, "double* w"]

  lines: list[str] = []
  if sparse:
    lines.append(f"// PIQP sparse solver wrapper for {fun.name} (n={n}, p={p}, m={m}, nnz P/A/G = {nnz_P}/{nnz_A}/{nnz_G}).")
  else:
    lines.append(f"// PIQP dense solver wrapper for {fun.name} (n={n}, p={p}, m={m}).")
  lines.append(f"static void {raw}({', '.join(params)}) {{")
  # Suppress unused warnings (warm-start inputs are not consumed yet).
  lines.append("  (void)in0;")  # x0
  lines.append("  (void)in1;")  # lam_eq0
  lines.append("  (void)in2;")  # lam_ineq0
  lines.append("  double stats_t0 = alloy_clock_s();")

  # 1. Local QP data buffers (static: row-major P/A/G are O(n^2) in the dense
  # case and would overflow the stack at larger sizes; the wrapper is
  # non-reentrant anyway).
  for buf, size in oracle_outs:
    lines.append(f"  static double {buf}[{size}];")
  if not sparse:
    # Column-major staging buffers used for actual PIQP calls (transpose).
    lines.append(f"  static double Pcol[{n * n}];")
    if p:
      lines.append(f"  static double Acol[{p * n}];")
    if m:
      lines.append(f"  static double Gcol[{m * n}];")

  # 2. Call the oracle. The oracle's input order is the param list; its
  # output order matches oracle_outs above.
  oracle_raw = _raw_symbol(oracle)
  oracle_call = ", ".join([*oracle_param_args, *(name for name, _ in oracle_outs), "w"])
  lines.append("  double fe_t0 = alloy_clock_s();")
  lines.append(f"  {oracle_raw}({oracle_call});")
  lines.append("  double stats_t_fe = alloy_clock_s() - fe_t0;")

  if sparse:
    # 3. Static CSC handles over the baked patterns and compact value buffers.
    lines += _csc_tables("P", desc.P_sparsity)
    lines.append(f"  static piqp_csc P_csc = {{ {n}, {n}, {nnz_P}, P_p, P_i, P_buf }};")
    if p:
      lines += _csc_tables("A", desc.A_sparsity)
      lines.append(f"  static piqp_csc A_csc = {{ {p}, {n}, {nnz_A}, A_p, A_i, A_buf }};")
    if m:
      lines += _csc_tables("G", desc.G_sparsity)
      lines.append(f"  static piqp_csc G_csc = {{ {m}, {n}, {nnz_G}, G_p, G_i, G_buf }};")
  else:
    # 3. Row-major -> column-major transpose for P, A, G.
    lines.append(f"  for (int j = 0; j < {n}; ++j) for (int i = 0; i < {n}; ++i) Pcol[i + j * {n}] = P_buf[i * {n} + j];")
    if p:
      lines.append(f"  for (int j = 0; j < {n}; ++j) for (int i = 0; i < {p}; ++i) Acol[i + j * {p}] = A_buf[i * {n} + j];")
    if m:
      lines.append(f"  for (int j = 0; j < {n}; ++j) for (int i = 0; i < {m}; ++i) Gcol[i + j * {m}] = G_buf[i * {n} + j];")

  # 4. Static PIQP workspace, lazily set up on first call.
  interface = "sparse" if sparse else "dense"
  lines.append(f"  static piqp_workspace* {symbol}_ws = NULL;")
  lines.append(f"  static piqp_settings {symbol}_settings;")
  lines.append("  double solver_t0 = alloy_clock_s();")
  lines.append(f"  if ({symbol}_ws == NULL) {{")
  lines.append(f"    piqp_set_default_settings_{interface}(&{symbol}_settings);")
  for key, val in desc.options:
    if isinstance(val, bool):
      lines.append(f"    {symbol}_settings.{key} = {1 if val else 0};")
    elif isinstance(val, (int, float)):
      lines.append(f"    {symbol}_settings.{key} = {val};")
    else:
      # String options are not part of PIQP's settings struct.
      raise NotImplementedError(f"PIQP option {key}={val!r} cannot be lowered to C")
  # Build setup data struct.
  lines.append(f"    piqp_data_{interface} setup_data;")
  lines.append(f"    setup_data.n = {n};")
  lines.append(f"    setup_data.p = {p};")
  lines.append(f"    setup_data.m = {m};")
  lines.append(f"    setup_data.P = {'&P_csc' if sparse else 'Pcol'};")
  lines.append("    setup_data.c = c_buf;")
  lines.append(f"    setup_data.A = {('&A_csc' if sparse else 'Acol') if p else 'NULL'};")
  lines.append(f"    setup_data.b = {'b_buf' if p else 'NULL'};")
  lines.append(f"    setup_data.G = {('&G_csc' if sparse else 'Gcol') if m else 'NULL'};")
  lines.append(f"    setup_data.h_l = {'lineq_buf' if m else 'NULL'};")
  lines.append(f"    setup_data.h_u = {'uineq_buf' if m else 'NULL'};")
  lines.append("    setup_data.x_l = xlb_buf;")
  lines.append("    setup_data.x_u = xub_buf;")
  lines.append(f"    piqp_setup_{interface}(&{symbol}_ws, &setup_data, &{symbol}_settings);")
  lines.append("  } else {")
  if sparse:
    lines.append(
      f"    piqp_update_sparse({symbol}_ws, &P_csc, c_buf, "
      f"{'&A_csc' if p else 'NULL'}, {'b_buf' if p else 'NULL'}, "
      f"{'&G_csc' if m else 'NULL'}, {'lineq_buf' if m else 'NULL'}, {'uineq_buf' if m else 'NULL'}, "
      "xlb_buf, xub_buf);"
    )
  else:
    lines.append(
      f"    piqp_update_dense({symbol}_ws, Pcol, c_buf, "
      f"{'Acol' if p else 'NULL'}, {'b_buf' if p else 'NULL'}, "
      f"{'Gcol' if m else 'NULL'}, {'lineq_buf' if m else 'NULL'}, {'uineq_buf' if m else 'NULL'}, "
      "xlb_buf, xub_buf);"
    )
  lines.append("  }")

  # 5. Solve and read results.
  lines.append(f"  piqp_solve({symbol}_ws);")
  lines.append("  double stats_t_solver = alloy_clock_s() - solver_t0;")
  lines.append(f"  piqp_result* res = {symbol}_ws->result;")
  # Outputs: x, cost, lam_eq, lam_ineq, lam_box.
  lines.append(f"  for (int i = 0; i < {n}; ++i) out0[i] = res->x[i];")
  lines.append("  out1[0] = res->info.primal_obj;")
  if p:
    lines.append(f"  for (int i = 0; i < {p}; ++i) out2[i] = res->y[i];")
  if m:
    lines.append(f"  for (int i = 0; i < {m}; ++i) out3[i] = res->z_u[i] - res->z_l[i];")
  lines.append(f"  for (int i = 0; i < {n}; ++i) out4[i] = res->z_bu[i] - res->z_bl[i];")

  lines.append("  int32_t stats_status;")
  lines.append("  switch (res->info.status) {")
  lines.append("    case PIQP_SOLVED: stats_status = ALLOY_SOLVE_OK; break;")
  lines.append("    case PIQP_MAX_ITER_REACHED: stats_status = ALLOY_SOLVE_MAX_ITER; break;")
  lines.append("    case PIQP_PRIMAL_INFEASIBLE: stats_status = ALLOY_SOLVE_PRIMAL_INFEASIBLE; break;")
  lines.append("    case PIQP_DUAL_INFEASIBLE: stats_status = ALLOY_SOLVE_DUAL_INFEASIBLE; break;")
  lines.append("    case PIQP_NUMERICS: stats_status = ALLOY_SOLVE_NUMERICS; break;")
  lines.append("    default: stats_status = ALLOY_SOLVE_ERROR; break;")
  lines.append("  }")
  lines.append(f"  {symbol}_stats_data.version = ALLOY_SOLVER_STATS_VERSION;")
  lines.append(f"  {symbol}_stats_data.status = stats_status;")
  lines.append(f"  {symbol}_stats_data.native_status = (int32_t)res->info.status;")
  lines.append(f"  {symbol}_stats_data.iter = (int32_t)res->info.iter;")
  lines.append(f"  {symbol}_stats_data.obj = res->info.primal_obj;")
  lines.append(f"  {symbol}_stats_data.t_fe = stats_t_fe;")
  lines.append(f"  {symbol}_stats_data.t_solver = stats_t_solver;")
  lines.append(f"  {symbol}_stats_data.n_eval_f = 1;")
  lines.append(f"  {symbol}_stats_data.n_eval_grad_f = 0;")
  lines.append(f"  {symbol}_stats_data.n_eval_g = 0;")
  lines.append(f"  {symbol}_stats_data.n_eval_jac_g = 0;")
  lines.append(f"  {symbol}_stats_data.n_eval_h = 0;")
  lines.append(f"  {symbol}_stats_data._pad0 = 0;")
  lines.append("  double stats_t_total = alloy_clock_s() - stats_t0;")
  lines.append(f"  {symbol}_stats_data.t_total = stats_t_total;")
  lines.append(f"  {symbol}_stats_data.t_glue = stats_t_total - stats_t_fe - stats_t_solver;")

  lines.append("}")
  return lines


# ---------------------------------------------------------------------------
# IPOPT
# ---------------------------------------------------------------------------


_IPOPT_INF = 2e19


_IPOPT_OPTION_TOKEN = re.compile(r"[A-Za-z0-9_./+-]+\Z")


def _ipopt_option_call(key: str, val: object) -> str:
  # IPOPT's StdCInterface declares ``char*`` (not ``const char*``) for option
  # keys and string values. The casts keep C++ consumers happy under
  # ``-Wwritable-strings`` without changing C semantics. Keys/values are
  # interpolated into C string literals, so restrict them to safe tokens.
  if not _IPOPT_OPTION_TOKEN.match(key):
    raise NotImplementedError(f"IPOPT option key {key!r} cannot be lowered to a C string literal")
  if isinstance(val, bool):
    return f'AddIpoptIntOption(problem, (char*)"{key}", {1 if val else 0})'
  if isinstance(val, int):
    return f'AddIpoptIntOption(problem, (char*)"{key}", {val})'
  if isinstance(val, float):
    return f'AddIpoptNumOption(problem, (char*)"{key}", {val})'
  if isinstance(val, str):
    if not _IPOPT_OPTION_TOKEN.match(val):
      raise NotImplementedError(f"IPOPT option {key}={val!r} cannot be lowered to a C string literal")
    return f'AddIpoptStrOption(problem, (char*)"{key}", (char*)"{val}")'
  raise NotImplementedError(f"IPOPT option {key}={val!r} cannot be lowered to C")


def _render_ipopt_raw(fun: Function, desc: SolverDescriptor) -> list[str]:
  """Render the solver wrapper for an IPOPT NLP.

  Strategy:

  - All ``in*`` param pointers, the caller-provided workspace ``w``, and the
    per-solve stats counters are stored in a static context struct so the eval
    callbacks can reach them through ``UserDataPtr``. Passing ``w`` through is
    load-bearing: the oracle ``_raw`` kernels require their packed scratch
    workspace and crash on NULL at any nontrivial problem size.
  - Sparse Jacobian rows/cols and the lower-triangle Hessian rows/cols are
    emitted as static const arrays.
  - Five ``eval_*`` static functions bridge IPOPT into the generated
    base / grad / jac / hess kernels, timing each call (the FE side of the
    stats split) and counting evaluations like the Python backend does.
  - The wrapper body computes bounds via ``bounds_raw``, clamps them to
    IPOPT's ±2e19 infinity convention, builds an ``IpoptProblem``, applies
    options, registers the iteration-counting intermediate callback, seeds
    the ``lam_eq0``/``lam_ineq0`` warm start, runs ``IpoptSolve``, writes
    outputs, and fills the alloy stats struct (status mapped via the vendored
    ``ApplicationReturnStatus`` enum so upstream drift breaks at compile time).
  """
  symbol = _c_ident(fun.name)
  raw = _raw_symbol(fun)
  n, n_h, n_g = desc.n, desc.n_eq, desc.n_ineq
  m = n_h + n_g
  base = desc.base
  grad = desc.grad
  jac = desc.jac
  hess = desc.hess
  bounds = desc.bounds
  assert base is not None and grad is not None and hess is not None and bounds is not None
  base_raw = _raw_symbol(base)
  grad_raw = _raw_symbol(grad)
  jac_raw = _raw_symbol(jac) if jac is not None else None
  hess_raw = _raw_symbol(hess)
  bounds_raw = _raw_symbol(bounds)
  param_count = len(desc.param_names)
  param_args = [f"ctx->p{i}" for i in range(param_count)]

  jac_sp = desc.jac_sparsity
  hess_sp = desc.hess_sparsity
  assert hess_sp is not None
  jac_rows = list(jac_sp.rows) if jac_sp is not None else []
  jac_cols = list(jac_sp.cols) if jac_sp is not None else []
  nnz_jac = len(jac_rows)
  hess_rows_full = list(hess_sp.rows)
  hess_cols_full = list(hess_sp.cols)
  lower_mask = list(desc.hess_lower_mask)
  lower_indices = [i for i, b in enumerate(lower_mask) if b]
  nnz_hess_full = len(hess_rows_full)
  nnz_hess = len(lower_indices)
  hess_rows = [hess_rows_full[i] for i in lower_indices]
  hess_cols = [hess_cols_full[i] for i in lower_indices]

  lines: list[str] = []
  lines.append(f"// IPOPT NLP wrapper for {fun.name} (n={n}, n_h={n_h}, n_g={n_g}).")
  # Context struct holding param pointers, the workspace, and per-solve stats.
  lines.append("typedef struct {")
  for i in range(param_count):
    lines.append(f"  const double* p{i};")
  lines.append("  double* w;")
  lines.append("  double t_fe;")
  lines.append("  int32_t n_eval_f, n_eval_grad_f, n_eval_g, n_eval_jac_g, n_eval_h, iter;")
  lines.append(f"}} {symbol}_ctx_t;")
  lines.append(f"static {symbol}_ctx_t {symbol}_ctx;")
  # Static evaluation buffers shared by the callbacks (file-scope so callbacks
  # see them; static so large problems cannot overflow the stack).
  if m:
    lines.append(f"static double {symbol}_g_scratch[{m}];")
  if nnz_hess:
    lines.append(f"static double {symbol}_h_scratch[{nnz_hess_full}];")
  # Sparsity patterns.
  if nnz_jac:
    lines.append(f"static const int {symbol}_jac_rows[{nnz_jac}] = {{ {', '.join(str(r) for r in jac_rows)} }};")
    lines.append(f"static const int {symbol}_jac_cols[{nnz_jac}] = {{ {', '.join(str(c) for c in jac_cols)} }};")
  if nnz_hess:
    lines.append(f"static const int {symbol}_hess_rows[{nnz_hess}] = {{ {', '.join(str(r) for r in hess_rows)} }};")
    lines.append(f"static const int {symbol}_hess_cols[{nnz_hess}] = {{ {', '.join(str(c) for c in hess_cols)} }};")
    lines.append(f"static const int {symbol}_hess_lower_idx[{nnz_hess}] = {{ {', '.join(str(i) for i in lower_indices)} }};")

  # eval_f
  lines.append(f"static bool {symbol}_eval_f(ipindex N, ipnumber* x, bool new_x, ipnumber* obj_value, UserDataPtr ud) {{")
  lines.append("  (void)N; (void)new_x;")
  lines.append(f"  {symbol}_ctx_t* ctx = ({symbol}_ctx_t*)ud;")
  lines.append("  double fe_t0 = alloy_clock_s();")
  lines.append("  double f_buf[1];")
  lines.append(f"  {base_raw}(x, {', '.join([*param_args, 'f_buf', *([f'{symbol}_g_scratch'] if m else [])])}, ctx->w);")
  lines.append("  ctx->t_fe += alloy_clock_s() - fe_t0;")
  lines.append("  ctx->n_eval_f += 1;")
  lines.append("  obj_value[0] = f_buf[0];")
  lines.append("  return true;")
  lines.append("}")

  # eval_grad_f
  lines.append(f"static bool {symbol}_eval_grad_f(ipindex N, ipnumber* x, bool new_x, ipnumber* grad_f, UserDataPtr ud) {{")
  lines.append("  (void)N; (void)new_x;")
  lines.append(f"  {symbol}_ctx_t* ctx = ({symbol}_ctx_t*)ud;")
  lines.append("  double fe_t0 = alloy_clock_s();")
  lines.append(f"  {grad_raw}(x, {', '.join([*param_args, 'grad_f'])}, ctx->w);")
  lines.append("  ctx->t_fe += alloy_clock_s() - fe_t0;")
  lines.append("  ctx->n_eval_grad_f += 1;")
  lines.append("  return true;")
  lines.append("}")

  # eval_g
  lines.append(f"static bool {symbol}_eval_g(ipindex N, ipnumber* x, bool new_x, ipindex M, ipnumber* g, UserDataPtr ud) {{")
  lines.append("  (void)N; (void)M; (void)new_x;")
  if not m:
    lines.append("  (void)x; (void)g; (void)ud;")
    lines.append("  return true;")
  else:
    lines.append(f"  {symbol}_ctx_t* ctx = ({symbol}_ctx_t*)ud;")
    lines.append("  double fe_t0 = alloy_clock_s();")
    lines.append("  double f_buf[1];")
    lines.append(f"  {base_raw}(x, {', '.join([*param_args, 'f_buf', 'g'])}, ctx->w);")
    lines.append("  ctx->t_fe += alloy_clock_s() - fe_t0;")
    lines.append("  ctx->n_eval_g += 1;")
    lines.append("  return true;")
  lines.append("}")

  # eval_jac_g
  lines.append(
    f"static bool {symbol}_eval_jac_g(ipindex N, ipnumber* x, bool new_x, ipindex M, ipindex nele_jac, "
    "ipindex* iRow, ipindex* jCol, ipnumber* values, UserDataPtr ud) {"
  )
  lines.append("  (void)N; (void)M; (void)new_x; (void)nele_jac;")
  if not nnz_jac:
    lines.append("  (void)x; (void)iRow; (void)jCol; (void)values; (void)ud;")
    lines.append("  return true;")
  else:
    lines.append(f"  {symbol}_ctx_t* ctx = ({symbol}_ctx_t*)ud;")
    lines.append("  if (values == NULL) {")
    lines.append(f"    for (int k = 0; k < {nnz_jac}; ++k) {{ iRow[k] = {symbol}_jac_rows[k]; jCol[k] = {symbol}_jac_cols[k]; }}")
    lines.append("  } else {")
    assert jac_raw is not None
    lines.append("    double fe_t0 = alloy_clock_s();")
    lines.append(f"    {jac_raw}(x, {', '.join([*param_args, 'values'])}, ctx->w);")
    lines.append("    ctx->t_fe += alloy_clock_s() - fe_t0;")
    lines.append("    ctx->n_eval_jac_g += 1;")
    lines.append("  }")
    lines.append("  return true;")
  lines.append("}")

  # eval_h
  lines.append(
    f"static bool {symbol}_eval_h(ipindex N, ipnumber* x, bool new_x, ipnumber obj_factor, ipindex M, "
    "ipnumber* lambda, bool new_lambda, ipindex nele_hess, ipindex* iRow, ipindex* jCol, "
    "ipnumber* values, UserDataPtr ud) {"
  )
  lines.append("  (void)N; (void)M; (void)new_x; (void)new_lambda; (void)nele_hess;")
  if not nnz_hess:
    lines.append("  (void)x; (void)obj_factor; (void)lambda; (void)iRow; (void)jCol; (void)values; (void)ud;")
    lines.append("  return true;")
  else:
    lines.append(f"  {symbol}_ctx_t* ctx = ({symbol}_ctx_t*)ud;")
    lines.append("  if (values == NULL) {")
    lines.append(f"    for (int k = 0; k < {nnz_hess}; ++k) {{ iRow[k] = {symbol}_hess_rows[k]; jCol[k] = {symbol}_hess_cols[k]; }}")
    lines.append("  } else {")
    lines.append("    double fe_t0 = alloy_clock_s();")
    lines.append("    double obj_buf[1]; obj_buf[0] = obj_factor;")
    hess_args = ["x", "obj_buf"]
    if m:
      hess_args.append("lambda")
    hess_args.extend(param_args)
    hess_args.append(f"{symbol}_h_scratch")
    lines.append(f"    {hess_raw}({', '.join(hess_args)}, ctx->w);")
    lines.append(f"    for (int k = 0; k < {nnz_hess}; ++k) values[k] = {symbol}_h_scratch[{symbol}_hess_lower_idx[k]];")
    lines.append("    ctx->t_fe += alloy_clock_s() - fe_t0;")
    lines.append("    ctx->n_eval_h += 1;")
    lines.append("  }")
    lines.append("  return true;")
  lines.append("}")

  # Intermediate callback: iteration counting (L3 parity with the Python path).
  lines.append(
    f"static bool {symbol}_intermediate(ipindex alg_mod, ipindex iter_count, ipnumber obj_value, "
    "ipnumber inf_pr, ipnumber inf_du, ipnumber mu, ipnumber d_norm, ipnumber regularization_size, "
    "ipnumber alpha_du, ipnumber alpha_pr, ipindex ls_trials, UserDataPtr ud) {"
  )
  lines.append("  (void)alg_mod; (void)obj_value; (void)inf_pr; (void)inf_du; (void)mu; (void)d_norm;")
  lines.append("  (void)regularization_size; (void)alpha_du; (void)alpha_pr; (void)ls_trials;")
  lines.append(f"  (({symbol}_ctx_t*)ud)->iter = (int32_t)iter_count;")
  lines.append("  return true;")
  lines.append("}")

  # Wrapper body.
  c_inputs = [f"const double* in{i}" for i in range(len(desc.input_signature))]
  c_outputs = [f"double* out{i}" for i in range(len(desc.output_signature))]
  params = [*c_inputs, *c_outputs, "double* w"]
  lines.append(f"static void {raw}({', '.join(params)}) {{")
  lines.append("  double stats_t0 = alloy_clock_s();")
  # Stash params + workspace in the static context, reset per-solve stats.
  # NLP inputs: x0, lam_eq0, lam_ineq0, lam_box0, then params from in4.
  for i in range(param_count):
    lines.append(f"  {symbol}_ctx.p{i} = in{4 + i};")
  lines.append(f"  {symbol}_ctx.w = w;")
  lines.append(f"  {symbol}_ctx.t_fe = 0.0;")
  lines.append(f"  {symbol}_ctx.n_eval_f = 0; {symbol}_ctx.n_eval_grad_f = 0; {symbol}_ctx.n_eval_g = 0;")
  lines.append(f"  {symbol}_ctx.n_eval_jac_g = 0; {symbol}_ctx.n_eval_h = 0; {symbol}_ctx.iter = 0;")
  # Compute bounds (counted as FE time), then clamp to IPOPT's ±2e19
  # infinity convention like the Python backend's _replace_nonfinite.
  # Deliberate divergence: `!(x > lim)` also maps NaN bounds to the infinity
  # limit, where the Python path forwards NaN to IPOPT (invalid either way).
  lines.append(f"  static double x_L[{n}]; static double x_U[{n}];")
  if n_g:
    lines.append(f"  static double l_in[{n_g}]; static double u_in[{n_g}];")
  bounds_outs = ["x_L", "x_U"]
  if n_g:
    bounds_outs.extend(["l_in", "u_in"])
  bounds_call = ", ".join([*[f"in{4 + i}" for i in range(param_count)], *bounds_outs])
  lines.append("  double bounds_t0 = alloy_clock_s();")
  lines.append(f"  {bounds_raw}({bounds_call}, w);")
  lines.append(f"  {symbol}_ctx.t_fe += alloy_clock_s() - bounds_t0;")
  lines.append(f"  for (int i = 0; i < {n}; ++i) {{")
  lines.append(f"    if (!(x_L[i] > -{_IPOPT_INF})) x_L[i] = -{_IPOPT_INF};")
  lines.append(f"    if (!(x_U[i] < {_IPOPT_INF})) x_U[i] = {_IPOPT_INF};")
  lines.append("  }")
  # Combine g_L / g_U: stack zeros for equalities, then l_in/u_in for inequalities.
  if m:
    lines.append(f"  static double g_L[{m}]; static double g_U[{m}];")
    if n_h:
      lines.append(f"  for (int i = 0; i < {n_h}; ++i) {{ g_L[i] = 0.0; g_U[i] = 0.0; }}")
    if n_g:
      lines.append(f"  for (int i = 0; i < {n_g}; ++i) {{")
      lines.append(f"    g_L[{n_h} + i] = !(l_in[i] > -{_IPOPT_INF}) ? -{_IPOPT_INF} : l_in[i];")
      lines.append(f"    g_U[{n_h} + i] = !(u_in[i] < {_IPOPT_INF}) ? {_IPOPT_INF} : u_in[i];")
      lines.append("  }")
  # CreateIpoptProblem copies bounds internally, so the problem is recreated
  # every call (bounds are parameter-dependent). Matches the Python path.
  jac_n = nnz_jac
  hess_n = nnz_hess
  if m:
    lines.append(
      f"  IpoptProblem problem = CreateIpoptProblem({n}, x_L, x_U, {m}, g_L, g_U, "
      f"{jac_n}, {hess_n}, 0, "
      f"{symbol}_eval_f, {symbol}_eval_g, {symbol}_eval_grad_f, {symbol}_eval_jac_g, {symbol}_eval_h);"
    )
  else:
    lines.append(
      f"  IpoptProblem problem = CreateIpoptProblem({n}, x_L, x_U, 0, NULL, NULL, "
      f"0, {hess_n}, 0, "
      f"{symbol}_eval_f, {symbol}_eval_g, {symbol}_eval_grad_f, {symbol}_eval_jac_g, {symbol}_eval_h);"
    )
  # Apply options and register the intermediate callback, checking every
  # return (the Python reference path raises on a rejected option; here a
  # failure surfaces as ALLOY_SOLVE_ERROR stats with defined outputs).
  lines.append("  int setup_ok = problem != NULL;")
  for key, val in desc.options:
    lines.append(f"  setup_ok = setup_ok && {_ipopt_option_call(key, val)};")
  lines.append(f"  setup_ok = setup_ok && SetIntermediateCallback(problem, {symbol}_intermediate);")
  lines.append("  if (!setup_ok) {")
  lines.append(f"    for (int i = 0; i < {n}; ++i) out0[i] = in0[i];")
  lines.append("    out1[0] = 0.0;")
  if n_h:
    lines.append(f"    for (int i = 0; i < {n_h}; ++i) {{ out2[i] = 0.0; out4[i] = 0.0; }}")
  if n_g:
    lines.append(f"    for (int i = 0; i < {n_g}; ++i) {{ out3[i] = 0.0; out5[i] = 0.0; }}")
  lines.append(f"    for (int i = 0; i < {n}; ++i) out6[i] = 0.0;")
  lines.append(f"    {symbol}_stats_data.version = ALLOY_SOLVER_STATS_VERSION;")
  lines.append(f"    {symbol}_stats_data.status = ALLOY_SOLVE_ERROR;")
  lines.append(f"    {symbol}_stats_data.native_status = (int32_t)(problem ? Invalid_Option : Invalid_Problem_Definition);")
  lines.append(f"    {symbol}_stats_data.iter = 0;")
  lines.append(f"    {symbol}_stats_data.obj = 0.0;")
  lines.append(f"    {symbol}_stats_data.t_fe = {symbol}_ctx.t_fe;")
  lines.append(f"    {symbol}_stats_data.t_solver = 0.0;")
  lines.append(f"    {symbol}_stats_data.n_eval_f = 0; {symbol}_stats_data.n_eval_grad_f = 0; {symbol}_stats_data.n_eval_g = 0;")
  lines.append(f"    {symbol}_stats_data.n_eval_jac_g = 0; {symbol}_stats_data.n_eval_h = 0; {symbol}_stats_data._pad0 = 0;")
  lines.append("    double fail_t_total = alloy_clock_s() - stats_t0;")
  lines.append(f"    {symbol}_stats_data.t_total = fail_t_total;")
  lines.append(f"    {symbol}_stats_data.t_glue = fail_t_total - {symbol}_ctx.t_fe;")
  lines.append("    if (problem) FreeIpoptProblem(problem);")
  lines.append("    return;")
  lines.append("  }")
  # Working buffers: primal seeded from x0, constraint multipliers from
  # lam_eq0/lam_ineq0, box multipliers sign-split from the signed lam_box0
  # (lam_box = z_U - z_L, so z_L = max(-lam_box0, 0), z_U = max(lam_box0, 0)).
  lines.append(f"  static double xv[{n}]; for (int i = 0; i < {n}; ++i) xv[i] = in0[i];")
  if m:
    lines.append(f"  static double g_val[{m}];")
  lines.append("  double obj_val = 0.0;")
  lines.append(f"  static double mult_g[{m if m else 1}];")
  lines.append(f"  static double mult_x_L[{n}]; static double mult_x_U[{n}];")
  if n_h:
    lines.append(f"  for (int i = 0; i < {n_h}; ++i) mult_g[i] = in1[i];")
  else:
    lines.append("  (void)in1;")
  if n_g:
    lines.append(f"  for (int i = 0; i < {n_g}; ++i) mult_g[{n_h} + i] = in2[i];")
  else:
    lines.append("  (void)in2;")
  lines.append(f"  for (int i = 0; i < {n}; ++i) {{ mult_x_L[i] = in3[i] < 0.0 ? -in3[i] : 0.0; mult_x_U[i] = in3[i] > 0.0 ? in3[i] : 0.0; }}")
  lines.append("  double fe_before_solve = " + f"{symbol}_ctx.t_fe;")
  lines.append("  double solver_t0 = alloy_clock_s();")
  if m:
    lines.append(f"  enum ApplicationReturnStatus ip_status = IpoptSolve(problem, xv, g_val, &obj_val, mult_g, mult_x_L, mult_x_U, &{symbol}_ctx);")
  else:
    lines.append(f"  enum ApplicationReturnStatus ip_status = IpoptSolve(problem, xv, NULL, &obj_val, NULL, mult_x_L, mult_x_U, &{symbol}_ctx);")
  lines.append("  double t_ipopt = alloy_clock_s() - solver_t0;")
  lines.append("  FreeIpoptProblem(problem);")
  # Write outputs: x, f, h_eq, g_ineq, lam_eq, lam_ineq, lam_box.
  lines.append(f"  for (int i = 0; i < {n}; ++i) out0[i] = xv[i];")
  lines.append("  out1[0] = obj_val;")
  if n_h:
    lines.append(f"  for (int i = 0; i < {n_h}; ++i) out2[i] = g_val[i];")
  if n_g:
    lines.append(f"  for (int i = 0; i < {n_g}; ++i) out3[i] = g_val[{n_h} + i];")
  if n_h:
    lines.append(f"  for (int i = 0; i < {n_h}; ++i) out4[i] = mult_g[i];")
  if n_g:
    lines.append(f"  for (int i = 0; i < {n_g}; ++i) out5[i] = mult_g[{n_h} + i];")
  lines.append(f"  for (int i = 0; i < {n}; ++i) out6[i] = mult_x_U[i] - mult_x_L[i];")

  # Stats. Status mapped through the vendored ApplicationReturnStatus enum
  # constants so upstream renames/renumbers break at compile time.
  lines.append("  int32_t stats_status;")
  lines.append("  switch (ip_status) {")
  lines.append("    case Solve_Succeeded: stats_status = ALLOY_SOLVE_OK; break;")
  lines.append("    case Solved_To_Acceptable_Level: stats_status = ALLOY_SOLVE_ACCEPTABLE; break;")
  lines.append("    case Feasible_Point_Found: stats_status = ALLOY_SOLVE_ACCEPTABLE; break;")
  lines.append("    case Maximum_Iterations_Exceeded: stats_status = ALLOY_SOLVE_MAX_ITER; break;")
  lines.append("    case Maximum_CpuTime_Exceeded: stats_status = ALLOY_SOLVE_MAX_ITER; break;")
  lines.append("    case Maximum_WallTime_Exceeded: stats_status = ALLOY_SOLVE_MAX_ITER; break;")
  lines.append("    case Infeasible_Problem_Detected: stats_status = ALLOY_SOLVE_PRIMAL_INFEASIBLE; break;")
  # Diverging iterates suggest unboundedness but are not a dual-infeasibility
  # certificate; report the weaker NUMERICS instead of DUAL_INFEASIBLE.
  lines.append("    case Diverging_Iterates: stats_status = ALLOY_SOLVE_NUMERICS; break;")
  lines.append("    case User_Requested_Stop: stats_status = ALLOY_SOLVE_USER_STOP; break;")
  lines.append("    case Search_Direction_Becomes_Too_Small: stats_status = ALLOY_SOLVE_NUMERICS; break;")
  lines.append("    case Restoration_Failed: stats_status = ALLOY_SOLVE_NUMERICS; break;")
  lines.append("    case Error_In_Step_Computation: stats_status = ALLOY_SOLVE_NUMERICS; break;")
  lines.append("    case Invalid_Number_Detected: stats_status = ALLOY_SOLVE_NUMERICS; break;")
  lines.append("    default: stats_status = ALLOY_SOLVE_ERROR; break;")
  lines.append("  }")
  lines.append(f"  {symbol}_stats_data.version = ALLOY_SOLVER_STATS_VERSION;")
  lines.append(f"  {symbol}_stats_data.status = stats_status;")
  lines.append(f"  {symbol}_stats_data.native_status = (int32_t)ip_status;")
  lines.append(f"  {symbol}_stats_data.iter = {symbol}_ctx.iter;")
  lines.append(f"  {symbol}_stats_data.obj = obj_val;")
  lines.append(f"  {symbol}_stats_data.t_fe = {symbol}_ctx.t_fe;")
  lines.append(f"  double stats_t_solver = t_ipopt - ({symbol}_ctx.t_fe - fe_before_solve);")
  lines.append("  if (stats_t_solver < 0.0) stats_t_solver = 0.0;")
  lines.append(f"  {symbol}_stats_data.t_solver = stats_t_solver;")
  lines.append(f"  {symbol}_stats_data.n_eval_f = {symbol}_ctx.n_eval_f;")
  lines.append(f"  {symbol}_stats_data.n_eval_grad_f = {symbol}_ctx.n_eval_grad_f;")
  lines.append(f"  {symbol}_stats_data.n_eval_g = {symbol}_ctx.n_eval_g;")
  lines.append(f"  {symbol}_stats_data.n_eval_jac_g = {symbol}_ctx.n_eval_jac_g;")
  lines.append(f"  {symbol}_stats_data.n_eval_h = {symbol}_ctx.n_eval_h;")
  lines.append(f"  {symbol}_stats_data._pad0 = 0;")
  lines.append("  double stats_t_total = alloy_clock_s() - stats_t0;")
  lines.append(f"  {symbol}_stats_data.t_total = stats_t_total;")
  lines.append(f"  {symbol}_stats_data.t_glue = stats_t_total - {symbol}_stats_data.t_fe - stats_t_solver;")
  lines.append("}")
  return lines
