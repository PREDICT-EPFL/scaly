"""IPOPT C wrapper template — the plugin half of scaly's solver codegen contract.

Called by ``scaly.codegen.solver.render_solver_raw`` through the backend's
``render_wrapper`` hook. Emits the IPOPT eval callbacks bridging into the
generated base/grad/jac/hess kernels plus a ``static void <ctx.raw_symbol>(...)``
that builds the ``IpoptProblem``, runs ``IpoptSolve``, and fills
``ctx.stats_symbol``. Contract: ``docs/dev/solver_plugins.md``.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from scaly.codegen.solver import SolverWrapperCtx
  from scaly.function.concrete import ConcreteFunction
  from scaly.solvers.model import SolverDescriptor

_IPOPT_INF = 2e19


_IPOPT_OPTION_TOKEN = re.compile(r"[A-Za-z0-9_./+-]+\Z")


def _validate_hessian_triangle(rows: list[int], cols: list[int]) -> None:
  """Reject a Hessian pattern that mixes entries from both triangles."""
  has_lower = any(row > col for row, col in zip(rows, cols, strict=True))
  has_upper = any(row < col for row, col in zip(rows, cols, strict=True))
  if has_lower and has_upper:
    raise ValueError("IPOPT Hessian sparsity must contain exactly one triangle")


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


def render_wrapper(fun: ConcreteFunction, ctx: SolverWrapperCtx) -> list[str]:
  """Render the solver wrapper for an IPOPT NLP.

  Strategy:

  - All ``in*`` param pointers, the caller-provided workspace ``w``, and the
    per-solve stats counters are stored in a static context struct so the eval
    callbacks can reach them through ``UserDataPtr``. Passing ``w`` through is
    load-bearing: the oracle ``_raw`` kernels require their packed scratch
    workspace and crash on NULL at any nontrivial problem size.
  - Sparse Jacobian rows/cols and the handed Hessian-triangle rows/cols are
    emitted as static const arrays.
  - Five ``eval_*`` static functions bridge IPOPT into the generated
    base / grad / jac / hess kernels, timing each call (the FE side of the
    stats split) and counting evaluations.
  - The wrapper body computes bounds via ``bounds_raw``, clamps them to
    IPOPT's ±2e19 infinity convention, builds an ``IpoptProblem``, applies
    options, registers the iteration-counting intermediate callback, seeds
    the ``lam_eq0``/``lam_ineq0`` warm start, runs ``IpoptSolve``, writes
    outputs, and fills the scaly stats struct (status mapped via the vendored
    ``ApplicationReturnStatus`` enum so upstream drift breaks at compile time).
  """
  desc: SolverDescriptor = fun.descriptor
  symbol = ctx.symbol
  raw = ctx.raw_symbol
  n, n_h, n_g = desc.n, desc.n_eq, desc.n_ineq
  m = n_h + n_g
  base = desc.base
  grad = desc.grad
  jac = desc.jac
  hess = desc.hess
  bounds = desc.bounds
  assert base is not None and grad is not None and hess is not None and bounds is not None
  base_raw = ctx.raw_symbol_of(base)
  grad_raw = ctx.raw_symbol_of(grad)
  jac_raw = ctx.raw_symbol_of(jac) if jac is not None else None
  hess_raw = ctx.raw_symbol_of(hess)
  bounds_raw = ctx.raw_symbol_of(bounds)
  param_count = len(desc.param_names)
  nv = desc.n_var_blocks
  if nv < 1:
    raise ValueError("typed NLP descriptors need at least one variable block")
  var_sizes = [math.prod(shape) for _, shape in desc.input_signature[:nv]]
  var_offsets = [sum(var_sizes[:i]) for i in range(nv)]
  eq_input, ineq_input, param_start = 2 * nv, 2 * nv + 1, 2 * nv + 2
  eq_output, ineq_output = 2 * nv, 2 * nv + 1
  param_args = [f"ctx->p{i}" for i in range(param_count)]

  jac_sp = desc.jac_sparsity
  hess_sp = desc.hess_sparsity
  assert hess_sp is not None
  jac_rows = list(jac_sp.rows) if jac_sp is not None else []
  jac_cols = list(jac_sp.cols) if jac_sp is not None else []
  nnz_jac = len(jac_rows)
  hess_rows = list(hess_sp.rows)
  hess_cols = list(hess_sp.cols)
  _validate_hessian_triangle(hess_rows, hess_cols)
  nnz_hess = len(hess_rows)

  lines: list[str] = []
  lines.append(f"// IPOPT NLP wrapper for {fun.name} (n={n}, n_h={n_h}, n_g={n_g}).")
  # Context struct holding param pointers, the workspace, and per-solve stats.
  lines.append("typedef struct {")
  for i in range(param_count):
    lines.append(f"  const double* p{i};")
  lines.append("  double* w;")
  lines.append("  double t_fe;")
  lines.append("  int32_t n_eval_f, n_eval_grad_f, n_eval_g, n_eval_jac_g, n_eval_h, iter;")
  lines.append("  double inf_pr, step_inf, alpha;")
  lines.append("  int32_t backtracks;")
  lines.append(f"}} {symbol}_ctx_t;")
  lines.append(f"static {symbol}_ctx_t {symbol}_ctx;")
  # Static evaluation buffers shared by the callbacks (file-scope so callbacks
  # see them; static so large problems cannot overflow the stack).
  if m:
    lines.append(f"static double {symbol}_g_scratch[{m}];")
  # Sparsity patterns.
  if nnz_jac:
    lines.append(f"static const int {symbol}_jac_rows[{nnz_jac}] = {{ {', '.join(str(r) for r in jac_rows)} }};")
    lines.append(f"static const int {symbol}_jac_cols[{nnz_jac}] = {{ {', '.join(str(c) for c in jac_cols)} }};")
  if nnz_hess:
    lines.append(f"static const int {symbol}_hess_rows[{nnz_hess}] = {{ {', '.join(str(r) for r in hess_rows)} }};")
    lines.append(f"static const int {symbol}_hess_cols[{nnz_hess}] = {{ {', '.join(str(c) for c in hess_cols)} }};")

  # eval_f
  lines.append(f"static bool {symbol}_eval_f(ipindex N, ipnumber* x, bool new_x, ipnumber* obj_value, UserDataPtr ud) {{")
  lines.append("  (void)N; (void)new_x;")
  lines.append(f"  {symbol}_ctx_t* ctx = ({symbol}_ctx_t*)ud;")
  lines.append("  double fe_t0 = scaly_clock_s();")
  lines.append("  double f_buf[1];")
  lines.append(f"  {base_raw}(x, {', '.join([*param_args, 'f_buf', *([f'{symbol}_g_scratch'] if m else [])])}, ctx->w);")
  lines.append("  ctx->t_fe += scaly_clock_s() - fe_t0;")
  lines.append("  ctx->n_eval_f += 1;")
  lines.append("  obj_value[0] = f_buf[0];")
  lines.append("  return true;")
  lines.append("}")

  # eval_grad_f
  lines.append(f"static bool {symbol}_eval_grad_f(ipindex N, ipnumber* x, bool new_x, ipnumber* grad_f, UserDataPtr ud) {{")
  lines.append("  (void)N; (void)new_x;")
  lines.append(f"  {symbol}_ctx_t* ctx = ({symbol}_ctx_t*)ud;")
  lines.append("  double fe_t0 = scaly_clock_s();")
  lines.append(f"  {grad_raw}(x, {', '.join([*param_args, 'grad_f'])}, ctx->w);")
  lines.append("  ctx->t_fe += scaly_clock_s() - fe_t0;")
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
    lines.append("  double fe_t0 = scaly_clock_s();")
    lines.append("  double f_buf[1];")
    lines.append(f"  {base_raw}(x, {', '.join([*param_args, 'f_buf', 'g'])}, ctx->w);")
    lines.append("  ctx->t_fe += scaly_clock_s() - fe_t0;")
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
    lines.append("    double fe_t0 = scaly_clock_s();")
    lines.append(f"    {jac_raw}(x, {', '.join([*param_args, 'values'])}, ctx->w);")
    lines.append("    ctx->t_fe += scaly_clock_s() - fe_t0;")
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
    lines.append("    double fe_t0 = scaly_clock_s();")
    lines.append("    double obj_buf[1]; obj_buf[0] = obj_factor;")
    hess_args = ["x", *param_args, "obj_buf"]
    if m:
      hess_args.append("lambda")
    hess_args.append("values")
    lines.append(f"    {hess_raw}({', '.join(hess_args)}, ctx->w);")
    lines.append("    ctx->t_fe += scaly_clock_s() - fe_t0;")
    lines.append("    ctx->n_eval_h += 1;")
    lines.append("  }")
    lines.append("  return true;")
  lines.append("}")

  # Intermediate callback: iteration counting (L3 parity) plus the v3
  # diagnostics IPOPT hands over for free — primal infeasibility, step inf
  # norm, last primal step size, and line-search trial counts (trials minus
  # the accepted one, matching the SQP backtrack semantics).
  lines.append(
    f"static bool {symbol}_intermediate(ipindex alg_mod, ipindex iter_count, ipnumber obj_value, "
    "ipnumber inf_pr, ipnumber inf_du, ipnumber mu, ipnumber d_norm, ipnumber regularization_size, "
    "ipnumber alpha_du, ipnumber alpha_pr, ipindex ls_trials, UserDataPtr ud) {"
  )
  lines.append("  (void)obj_value; (void)inf_du; (void)mu;")
  lines.append("  (void)regularization_size; (void)alpha_du;")
  lines.append(f"  {symbol}_ctx_t* ctx = ({symbol}_ctx_t*)ud;")
  lines.append("  ctx->iter = (int32_t)iter_count;")
  lines.append("  // alg_mod 1 is the restoration phase, whose inf_pr/d_norm/alpha_pr describe the restoration subproblem.")
  lines.append("  if (alg_mod == 0) {")
  lines.append("    ctx->inf_pr = inf_pr;")
  lines.append("    if (iter_count > 0) { ctx->step_inf = d_norm; ctx->alpha = alpha_pr; }")
  lines.append("    if (ls_trials > 1) ctx->backtracks += (int32_t)ls_trials - 1;")
  lines.append("  }")
  lines.append("  return true;")
  lines.append("}")

  # Wrapper body.
  c_inputs = [f"const double* in{i}" for i in range(len(desc.input_signature))]
  c_outputs = [f"double* out{i}" for i in range(len(desc.output_signature))]
  params = [*c_inputs, *c_outputs, "double* w"]
  lines.append(f"static void {raw}({', '.join(params)}) {{")
  lines.append("  double stats_t0 = scaly_clock_s();")
  # Stash params + workspace in the static context, reset per-solve stats.
  for i in range(param_count):
    lines.append(f"  {symbol}_ctx.p{i} = in{param_start + i};")
  lines.append(f"  {symbol}_ctx.w = w;")
  lines.append(f"  {symbol}_ctx.t_fe = 0.0;")
  lines.append(f"  {symbol}_ctx.n_eval_f = 0; {symbol}_ctx.n_eval_grad_f = 0; {symbol}_ctx.n_eval_g = 0;")
  lines.append(f"  {symbol}_ctx.n_eval_jac_g = 0; {symbol}_ctx.n_eval_h = 0; {symbol}_ctx.iter = 0;")
  lines.append(f"  {symbol}_ctx.inf_pr = 0.0; {symbol}_ctx.step_inf = 0.0; {symbol}_ctx.alpha = 0.0; {symbol}_ctx.backtracks = 0;")
  # Compute bounds (counted as FE time), then translate core's IEEE infinities to IPOPT's ±2e19
  # infinity convention. Deliberate choice: `!(x > lim)` also maps NaN bounds
  # to the infinity limit (invalid either way).
  lines.append(f"  static double x_L[{n}]; static double x_U[{n}];")
  if n_g:
    lines.append(f"  static double l_in[{n_g}]; static double u_in[{n_g}];")
  bounds_outs = ["x_L", "x_U"]
  if n_g:
    bounds_outs.extend(["l_in", "u_in"])
  bounds_call = ", ".join([*[f"in{param_start + i}" for i in range(param_count)], *bounds_outs])
  lines.append("  double bounds_t0 = scaly_clock_s();")
  lines.append(f"  {bounds_raw}({bounds_call}, w);")
  lines.append(f"  {symbol}_ctx.t_fe += scaly_clock_s() - bounds_t0;")
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
  # every call (bounds are parameter-dependent).
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
  # return (a rejected option surfaces as SCALY_SOLVE_ERROR stats with
  # defined outputs).
  lines.append("  int setup_ok = problem != NULL;")
  for key, val in desc.options:
    lines.append(f"  setup_ok = setup_ok && {_ipopt_option_call(key, val)};")
  lines.append(f"  setup_ok = setup_ok && SetIntermediateCallback(problem, {symbol}_intermediate);")
  lines.append("  if (!setup_ok) {")
  for block, size in enumerate(var_sizes):
    lines.append(f"    for (int i = 0; i < {size}; ++i) {chr(123)} out{block}[i] = in{block}[i]; out{nv + block}[i] = 0.0; {chr(125)}")
  if n_h:
    lines.append(f"    for (int i = 0; i < {n_h}; ++i) out{eq_output}[i] = 0.0;")
  if n_g:
    lines.append(f"    for (int i = 0; i < {n_g}; ++i) out{ineq_output}[i] = 0.0;")
  lines.append(f"    {ctx.stats_symbol}.version = SCALY_SOLVER_STATS_VERSION;")
  lines.append(f"    {ctx.stats_symbol}.status = SCALY_SOLVE_ERROR;")
  lines.append(f"    {ctx.stats_symbol}.native_status = (int32_t)(problem ? Invalid_Option : Invalid_Problem_Definition);")
  lines.append(f"    {ctx.stats_symbol}.iter = 0;")
  lines.append(f"    {ctx.stats_symbol}.obj = 0.0;")
  lines.append(f"    {ctx.stats_symbol}.t_fe = {symbol}_ctx.t_fe;")
  lines.append(f"    {ctx.stats_symbol}.t_solver = 0.0;")
  lines.append(f"    {ctx.stats_symbol}.t_qp = 0.0;")
  lines.append(f"    {ctx.stats_symbol}.t_globalization = 0.0;")
  lines.append(f"    {ctx.stats_symbol}.n_eval_f = 0; {ctx.stats_symbol}.n_eval_grad_f = 0; {ctx.stats_symbol}.n_eval_g = 0;")
  lines.append(f"    {ctx.stats_symbol}.n_eval_jac_g = 0; {ctx.stats_symbol}.n_eval_h = 0; {ctx.stats_symbol}._pad0 = 0;")
  lines.append(f"    {ctx.stats_symbol}.primal_viol = 0.0; {ctx.stats_symbol}.step_inf = 0.0; {ctx.stats_symbol}.alpha = 0.0;")
  lines.append(f"    {ctx.stats_symbol}.merit_penalty = 0.0; {ctx.stats_symbol}.backtracks = 0; {ctx.stats_symbol}.qp_iter = 0;")
  lines.append("    double fail_t_total = scaly_clock_s() - stats_t0;")
  lines.append(f"    {ctx.stats_symbol}.t_total = fail_t_total;")
  lines.append(f"    {ctx.stats_symbol}.t_glue = fail_t_total - {symbol}_ctx.t_fe;")
  lines.append("    if (problem) FreeIpoptProblem(problem);")
  lines.append("    return;")
  lines.append("  }")
  # Working buffers: primal seeded from x0, constraint multipliers from
  # lam_eq0/lam_ineq0, box multipliers sign-split from the signed lam_box0
  # (lam_box = z_U - z_L, so z_L = max(-lam_box0, 0), z_U = max(lam_box0, 0)).
  lines.append(f"  static double xv[{n}];")
  for block, (size, offset) in enumerate(zip(var_sizes, var_offsets, strict=True)):
    lines.append(f"  for (int i = 0; i < {size}; ++i) xv[{offset} + i] = in{block}[i];")
  if m:
    lines.append(f"  static double g_val[{m}];")
  lines.append("  double obj_val = 0.0;")
  lines.append(f"  static double mult_g[{m if m else 1}];")
  lines.append(f"  static double mult_x_L[{n}]; static double mult_x_U[{n}];")
  if n_h:
    lines.append(f"  for (int i = 0; i < {n_h}; ++i) mult_g[i] = in{eq_input}[i];")
  if n_g:
    lines.append(f"  for (int i = 0; i < {n_g}; ++i) mult_g[{n_h} + i] = in{ineq_input}[i];")
  for block, (size, offset) in enumerate(zip(var_sizes, var_offsets, strict=True)):
    lines.append(
      f"  for (int i = 0; i < {size}; ++i) {chr(123)} double lam = in{nv + block}[i]; mult_x_L[{offset} + i] = lam < 0.0 ? -lam : 0.0; mult_x_U[{offset} + i] = lam > 0.0 ? lam : 0.0; {chr(125)}"
    )
  lines.append("  double fe_before_solve = " + f"{symbol}_ctx.t_fe;")
  lines.append("  double solver_t0 = scaly_clock_s();")
  if m:
    lines.append(f"  enum ApplicationReturnStatus ip_status = IpoptSolve(problem, xv, g_val, &obj_val, mult_g, mult_x_L, mult_x_U, &{symbol}_ctx);")
  else:
    lines.append(f"  enum ApplicationReturnStatus ip_status = IpoptSolve(problem, xv, NULL, &obj_val, NULL, mult_x_L, mult_x_U, &{symbol}_ctx);")
  lines.append("  double t_ipopt = scaly_clock_s() - solver_t0;")
  lines.append("  FreeIpoptProblem(problem);")
  # Write variables and multipliers in the declared tree order.
  for block, (size, offset) in enumerate(zip(var_sizes, var_offsets, strict=True)):
    lines.append(
      f"  for (int i = 0; i < {size}; ++i) {chr(123)} out{block}[i] = xv[{offset} + i]; out{nv + block}[i] = mult_x_U[{offset} + i] - mult_x_L[{offset} + i]; {chr(125)}"
    )
  if n_h:
    lines.append(f"  for (int i = 0; i < {n_h}; ++i) out{eq_output}[i] = mult_g[i];")
  if n_g:
    lines.append(f"  for (int i = 0; i < {n_g}; ++i) out{ineq_output}[i] = mult_g[{n_h} + i];")

  # Stats. Status mapped through the vendored ApplicationReturnStatus enum
  # constants so upstream renames/renumbers break at compile time.
  lines.append("  int32_t stats_status;")
  lines.append("  switch (ip_status) {")
  lines.append("    case Solve_Succeeded: stats_status = SCALY_SOLVE_OK; break;")
  lines.append("    case Solved_To_Acceptable_Level: stats_status = SCALY_SOLVE_ACCEPTABLE; break;")
  lines.append("    case Feasible_Point_Found: stats_status = SCALY_SOLVE_ACCEPTABLE; break;")
  lines.append("    case Maximum_Iterations_Exceeded: stats_status = SCALY_SOLVE_MAX_ITER; break;")
  lines.append("    case Maximum_CpuTime_Exceeded: stats_status = SCALY_SOLVE_MAX_ITER; break;")
  lines.append("    case Maximum_WallTime_Exceeded: stats_status = SCALY_SOLVE_MAX_ITER; break;")
  lines.append("    case Infeasible_Problem_Detected: stats_status = SCALY_SOLVE_PRIMAL_INFEASIBLE; break;")
  # Diverging iterates suggest unboundedness but are not a dual-infeasibility
  # certificate; report the weaker NUMERICS instead of DUAL_INFEASIBLE.
  lines.append("    case Diverging_Iterates: stats_status = SCALY_SOLVE_NUMERICS; break;")
  lines.append("    case User_Requested_Stop: stats_status = SCALY_SOLVE_USER_STOP; break;")
  lines.append("    case Search_Direction_Becomes_Too_Small: stats_status = SCALY_SOLVE_NUMERICS; break;")
  lines.append("    case Restoration_Failed: stats_status = SCALY_SOLVE_NUMERICS; break;")
  lines.append("    case Error_In_Step_Computation: stats_status = SCALY_SOLVE_NUMERICS; break;")
  lines.append("    case Invalid_Number_Detected: stats_status = SCALY_SOLVE_NUMERICS; break;")
  lines.append("    default: stats_status = SCALY_SOLVE_ERROR; break;")
  lines.append("  }")
  lines.append(f"  {ctx.stats_symbol}.version = SCALY_SOLVER_STATS_VERSION;")
  lines.append(f"  {ctx.stats_symbol}.status = stats_status;")
  lines.append(f"  {ctx.stats_symbol}.native_status = (int32_t)ip_status;")
  lines.append(f"  {ctx.stats_symbol}.iter = {symbol}_ctx.iter;")
  lines.append(f"  {ctx.stats_symbol}.obj = obj_val;")
  lines.append(f"  {ctx.stats_symbol}.t_fe = {symbol}_ctx.t_fe;")
  lines.append(f"  double stats_t_solver = t_ipopt - ({symbol}_ctx.t_fe - fe_before_solve);")
  lines.append("  if (stats_t_solver < 0.0) stats_t_solver = 0.0;")
  lines.append(f"  {ctx.stats_symbol}.t_solver = stats_t_solver;")
  lines.append(f"  {ctx.stats_symbol}.t_qp = 0.0;")
  lines.append(f"  {ctx.stats_symbol}.t_globalization = 0.0;")
  lines.append(f"  {ctx.stats_symbol}.n_eval_f = {symbol}_ctx.n_eval_f;")
  lines.append(f"  {ctx.stats_symbol}.n_eval_grad_f = {symbol}_ctx.n_eval_grad_f;")
  lines.append(f"  {ctx.stats_symbol}.n_eval_g = {symbol}_ctx.n_eval_g;")
  lines.append(f"  {ctx.stats_symbol}.n_eval_jac_g = {symbol}_ctx.n_eval_jac_g;")
  lines.append(f"  {ctx.stats_symbol}.n_eval_h = {symbol}_ctx.n_eval_h;")
  lines.append(f"  {ctx.stats_symbol}._pad0 = 0;")
  # v3 diagnostics from the intermediate callback; merit penalty and QP
  # iterations have no IPOPT equivalent (filter line search, interior point).
  lines.append(f"  {ctx.stats_symbol}.primal_viol = {symbol}_ctx.inf_pr;")
  lines.append(f"  {ctx.stats_symbol}.step_inf = {symbol}_ctx.step_inf;")
  lines.append(f"  {ctx.stats_symbol}.alpha = {symbol}_ctx.alpha;")
  lines.append(f"  {ctx.stats_symbol}.merit_penalty = 0.0;")
  lines.append(f"  {ctx.stats_symbol}.backtracks = {symbol}_ctx.backtracks;")
  lines.append(f"  {ctx.stats_symbol}.qp_iter = 0;")
  lines.append("  double stats_t_total = scaly_clock_s() - stats_t0;")
  lines.append(f"  {ctx.stats_symbol}.t_total = stats_t_total;")
  lines.append(f"  {ctx.stats_symbol}.t_glue = stats_t_total - {ctx.stats_symbol}.t_fe - stats_t_solver;")
  lines.append("}")
  return lines
