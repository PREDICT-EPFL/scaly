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
  ident = re.sub(r"\W", "_", name)
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


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def render_solver_raw(fun: Function) -> list[str]:
  desc = _descriptor(fun)
  if desc.backend == "piqp":
    return _render_piqp_raw(fun, desc)
  if desc.backend == "ipopt":
    return _render_ipopt_raw(fun, desc)
  raise NotImplementedError(f"C codegen for solver backend {desc.backend!r} not yet implemented")


def _render_piqp_raw(fun: Function, desc: SolverDescriptor) -> list[str]:
  symbol = _c_ident(fun.name)
  raw = _raw_symbol(fun)
  n, p, m = desc.n, desc.n_eq, desc.n_ineq
  oracle = desc.oracle
  assert oracle is not None, "PIQP solver descriptor must carry an oracle Function"

  # Map solver input order to C parameter names: in0..in{n-1}.
  # Inputs: x0, lam_eq0, lam_ineq0, then params in order.
  param_count = len(desc.param_names)
  oracle_param_args = [f"in{3 + i}" for i in range(param_count)]

  # Oracle output ordering (set by qp.py): P, c, [A_eq, b_eq], [G_ineq, l_ineq, u_ineq], x_lb, x_ub.
  oracle_outs: list[tuple[str, int]] = []  # (buffer C name, size)
  oracle_outs.append(("P_buf", n * n))
  oracle_outs.append(("c_buf", n))
  if p:
    oracle_outs.append(("A_buf", p * n))
    oracle_outs.append(("b_buf", p))
  if m:
    oracle_outs.append(("G_buf", m * n))
    oracle_outs.append(("lineq_buf", m))
    oracle_outs.append(("uineq_buf", m))
  oracle_outs.append(("xlb_buf", n))
  oracle_outs.append(("xub_buf", n))

  # Parameters for the raw signature.
  c_inputs = [f"const double* in{i}" for i in range(len(desc.input_signature))]
  c_outputs = [f"double* out{i}" for i in range(len(desc.output_signature))]
  params = [*c_inputs, *c_outputs, "double* w"]

  lines: list[str] = []
  lines.append(f"// PIQP dense solver wrapper for {fun.name} (n={n}, p={p}, m={m}).")
  lines.append(f"static void {raw}({', '.join(params)}) {{")
  # Suppress unused warnings (warm-start inputs are not consumed yet).
  lines.append("  (void)w;")
  lines.append("  (void)in0;")  # x0
  lines.append("  (void)in1;")  # lam_eq0
  lines.append("  (void)in2;")  # lam_ineq0

  # 1. Local QP data buffers (stack-allocated; row-major from the oracle).
  for buf, size in oracle_outs:
    lines.append(f"  double {buf}[{size}];")
  # Column-major staging buffers used for actual PIQP calls (transpose).
  lines.append(f"  double Pcol[{n * n}];")
  if p:
    lines.append(f"  double Acol[{p * n}];")
  if m:
    lines.append(f"  double Gcol[{m * n}];")

  # 2. Call the oracle. The oracle's input order is the param list; its
  # output order matches oracle_outs above.
  oracle_raw = _raw_symbol(oracle)
  oracle_call = ", ".join([*oracle_param_args, *(name for name, _ in oracle_outs), "w"])
  lines.append(f"  {oracle_raw}({oracle_call});")

  # 3. Row-major -> column-major transpose for P, A, G.
  lines.append(f"  for (int j = 0; j < {n}; ++j) for (int i = 0; i < {n}; ++i) Pcol[i + j * {n}] = P_buf[i * {n} + j];")
  if p:
    lines.append(f"  for (int j = 0; j < {n}; ++j) for (int i = 0; i < {p}; ++i) Acol[i + j * {p}] = A_buf[i * {n} + j];")
  if m:
    lines.append(f"  for (int j = 0; j < {n}; ++j) for (int i = 0; i < {m}; ++i) Gcol[i + j * {m}] = G_buf[i * {n} + j];")

  # 4. Static PIQP workspace, lazily set up on first call.
  lines.append(f"  static piqp_workspace* {symbol}_ws = NULL;")
  lines.append(f"  static piqp_settings {symbol}_settings;")
  lines.append(f"  if ({symbol}_ws == NULL) {{")
  lines.append(f"    piqp_set_default_settings_dense(&{symbol}_settings);")
  for key, val in desc.options:
    if isinstance(val, bool):
      lines.append(f"    {symbol}_settings.{key} = {1 if val else 0};")
    elif isinstance(val, (int, float)):
      lines.append(f"    {symbol}_settings.{key} = {val};")
    else:
      # String options are not part of PIQP's settings struct.
      raise NotImplementedError(f"PIQP option {key}={val!r} cannot be lowered to C")
  # Build setup data struct.
  lines.append("    piqp_data_dense setup_data;")
  lines.append(f"    setup_data.n = {n};")
  lines.append(f"    setup_data.p = {p};")
  lines.append(f"    setup_data.m = {m};")
  lines.append("    setup_data.P = Pcol;")
  lines.append("    setup_data.c = c_buf;")
  lines.append(f"    setup_data.A = {'Acol' if p else 'NULL'};")
  lines.append(f"    setup_data.b = {'b_buf' if p else 'NULL'};")
  lines.append(f"    setup_data.G = {'Gcol' if m else 'NULL'};")
  lines.append(f"    setup_data.h_l = {'lineq_buf' if m else 'NULL'};")
  lines.append(f"    setup_data.h_u = {'uineq_buf' if m else 'NULL'};")
  lines.append("    setup_data.x_l = xlb_buf;")
  lines.append("    setup_data.x_u = xub_buf;")
  lines.append(f"    piqp_setup_dense(&{symbol}_ws, &setup_data, &{symbol}_settings);")
  lines.append("  } else {")
  lines.append(
    f"    piqp_update_dense({symbol}_ws, Pcol, c_buf, "
    f"{'Acol' if p else 'NULL'}, {'b_buf' if p else 'NULL'}, "
    f"{'Gcol' if m else 'NULL'}, {'lineq_buf' if m else 'NULL'}, {'uineq_buf' if m else 'NULL'}, "
    "xlb_buf, xub_buf);"
  )
  lines.append("  }")

  # 5. Solve and read results.
  lines.append(f"  piqp_solve({symbol}_ws);")
  lines.append(f"  piqp_result* res = {symbol}_ws->result;")
  # Outputs: x, cost, lam_eq, lam_ineq, lam_box.
  lines.append(f"  for (int i = 0; i < {n}; ++i) out0[i] = res->x[i];")
  lines.append("  out1[0] = res->info.primal_obj;")
  if p:
    lines.append(f"  for (int i = 0; i < {p}; ++i) out2[i] = res->y[i];")
  if m:
    lines.append(f"  for (int i = 0; i < {m}; ++i) out3[i] = res->z_u[i] - res->z_l[i];")
  lines.append(f"  for (int i = 0; i < {n}; ++i) out4[i] = res->z_bu[i] - res->z_bl[i];")

  lines.append("}")
  return lines


# ---------------------------------------------------------------------------
# IPOPT
# ---------------------------------------------------------------------------


_IPOPT_INF = 2e19


def _ipopt_option_call(key: str, val: object) -> str:
  # IPOPT's StdCInterface declares ``char*`` (not ``const char*``) for option
  # keys and string values. The casts keep C++ consumers happy under
  # ``-Wwritable-strings`` without changing C semantics.
  if isinstance(val, bool):
    return f'AddIpoptIntOption(problem, (char*)"{key}", {1 if val else 0})'
  if isinstance(val, int):
    return f'AddIpoptIntOption(problem, (char*)"{key}", {val})'
  if isinstance(val, float):
    return f'AddIpoptNumOption(problem, (char*)"{key}", {val})'
  if isinstance(val, str):
    return f'AddIpoptStrOption(problem, (char*)"{key}", (char*)"{val}")'
  raise NotImplementedError(f"IPOPT option {key}={val!r} cannot be lowered to C")


def _render_ipopt_raw(fun: Function, desc: SolverDescriptor) -> list[str]:
  """Render the solver wrapper for an IPOPT NLP.

  Strategy:

  - All ``in*`` param pointers are stored in a static context struct so the
    eval callbacks can reach them through ``UserDataPtr``.
  - Sparse Jacobian rows/cols and the lower-triangle Hessian rows/cols are
    emitted as static const arrays.
  - Five ``eval_*`` static functions bridge IPOPT into the JIT-compiled
    base / grad / jac / hess Functions. The Hessian gather respects the
    same lower-triangle mask used by the Python backend.
  - The wrapper body computes bounds via ``bounds_raw``, builds an
    ``IpoptProblem`` (cached statically across calls), applies options,
    runs ``IpoptSolve``, and writes outputs.
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
  # Context struct holding pointers to param arrays.
  lines.append("typedef struct {")
  for i in range(param_count):
    lines.append(f"  const double* p{i};")
  lines.append(f"}} {symbol}_ctx_t;")
  lines.append(f"static {symbol}_ctx_t {symbol}_ctx;")
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
  if m:
    lines.append(f"  double g_buf[{m}];")
  lines.append("  double f_buf[1];")
  lines.append(f"  {base_raw}(x, {', '.join([*param_args, 'f_buf', *(['g_buf'] if m else [])])}, NULL);")
  lines.append("  obj_value[0] = f_buf[0];")
  lines.append("  return true;")
  lines.append("}")

  # eval_grad_f
  lines.append(f"static bool {symbol}_eval_grad_f(ipindex N, ipnumber* x, bool new_x, ipnumber* grad_f, UserDataPtr ud) {{")
  lines.append("  (void)N; (void)new_x;")
  lines.append(f"  {symbol}_ctx_t* ctx = ({symbol}_ctx_t*)ud;")
  lines.append(f"  {grad_raw}(x, {', '.join([*param_args, 'grad_f'])}, NULL);")
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
    lines.append("  double f_buf[1];")
    lines.append(f"  {base_raw}(x, {', '.join([*param_args, 'f_buf', 'g'])}, NULL);")
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
    lines.append(f"    {jac_raw}(x, {', '.join([*param_args, 'values'])}, NULL);")
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
    lines.append("    double obj_buf[1]; obj_buf[0] = obj_factor;")
    lines.append(f"    double hbuf[{nnz_hess_full}];")
    hess_args = ["x", "obj_buf"]
    if m:
      hess_args.append("lambda")
    hess_args.extend(param_args)
    hess_args.append("hbuf")
    lines.append(f"    {hess_raw}({', '.join(hess_args)}, NULL);")
    lines.append(f"    for (int k = 0; k < {nnz_hess}; ++k) values[k] = hbuf[{symbol}_hess_lower_idx[k]];")
    lines.append("  }")
    lines.append("  return true;")
  lines.append("}")

  # Wrapper body.
  c_inputs = [f"const double* in{i}" for i in range(len(desc.input_signature))]
  c_outputs = [f"double* out{i}" for i in range(len(desc.output_signature))]
  params = [*c_inputs, *c_outputs, "double* w"]
  lines.append(f"static void {raw}({', '.join(params)}) {{")
  lines.append("  (void)w;")
  lines.append("  (void)in1; (void)in2;")  # initial duals unused (no warm start yet)
  # Stash params in static context.
  for i in range(param_count):
    lines.append(f"  {symbol}_ctx.p{i} = in{3 + i};")
  # Compute bounds.
  lines.append(f"  double x_L[{n}]; double x_U[{n}];")
  if m:
    lines.append(f"  double l_in[{n_g}]; double u_in[{n_g}];") if n_g else None
  bounds_outs = ["x_L", "x_U"]
  if n_g:
    bounds_outs.extend(["l_in", "u_in"])
  bounds_call = ", ".join([*[f"in{3 + i}" for i in range(param_count)], *bounds_outs])
  lines.append(f"  {bounds_raw}({bounds_call}, NULL);")
  # Combine g_L / g_U: stack zeros for equalities, then l_in/u_in for inequalities.
  if m:
    lines.append(f"  double g_L[{m}]; double g_U[{m}];")
    if n_h:
      lines.append(f"  for (int i = 0; i < {n_h}; ++i) {{ g_L[i] = 0.0; g_U[i] = 0.0; }}")
    if n_g:
      lines.append(f"  for (int i = 0; i < {n_g}; ++i) {{ g_L[{n_h} + i] = l_in[i]; g_U[{n_h} + i] = u_in[i]; }}")
  # Cache the IpoptProblem statically across calls. We only rebuild bounds via
  # IPOPT options — but actually CreateIpoptProblem copies bounds internally,
  # so we need to recreate when bounds change. For now, recreate every call;
  # this matches the Python ctypes path and keeps the C code straightforward.
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
  # Apply options.
  for key, val in desc.options:
    lines.append(f"  {_ipopt_option_call(key, val)};")
  # Allocate working buffers.
  lines.append(f"  double xv[{n}]; for (int i = 0; i < {n}; ++i) xv[i] = in0[i];")
  if m:
    lines.append(f"  double g_val[{m}];")
  lines.append("  double obj_val;")
  lines.append(f"  double mult_g[{m if m else 1}];")
  lines.append(f"  double mult_x_L[{n}]; double mult_x_U[{n}];")
  lines.append(f"  for (int i = 0; i < {m if m else 1}; ++i) mult_g[i] = 0.0;")
  lines.append(f"  for (int i = 0; i < {n}; ++i) {{ mult_x_L[i] = 0.0; mult_x_U[i] = 0.0; }}")
  if m:
    lines.append(f"  IpoptSolve(problem, xv, g_val, &obj_val, mult_g, mult_x_L, mult_x_U, &{symbol}_ctx);")
  else:
    lines.append(f"  IpoptSolve(problem, xv, NULL, &obj_val, NULL, mult_x_L, mult_x_U, &{symbol}_ctx);")
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
  lines.append("}")
  return lines
