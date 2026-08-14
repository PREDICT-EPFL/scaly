"""PIQP-backed sequential quadratic programming wrapper.

The QP subproblem is assembled through PIQP's sparse interface from the
descriptor's Jacobian and Hessian sparsity, with static CSC index tables baked
at codegen time and only the value arrays refilled per SQP iteration.
``qp="dense"`` assembles the same QP for PIQP's dense interface instead; it is
slower at every canonical benchmark point, so nothing selects it by default.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from alloy.codegen.solver_c import SolverWrapperCtx
  from alloy.function import Function
  from alloy.solvers.solver_function import SolverDescriptor


def _option_map(options: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
  out = dict(options)
  for ignored in ("print_level", "sb", "warm_start_init_point"):
    out.pop(ignored, None)
  allowed = {
    "dual_tol",
    "globalization",
    "hessian",
    "line_search_beta",
    "max_iter",
    "merit",
    "qp",
    "qp_max_iter",
    "qp_tol",
    "regularization",
    "tol",
    "trace",
    "watchdog",
  }
  unknown = sorted(set(out) - allowed)
  if unknown:
    raise NotImplementedError(f"alloy-sqp options are not supported: {unknown}")
  return out


def _positive_int(options: dict[str, Any], name: str, default: int) -> int:
  value = options.get(name, default)
  if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
    raise ValueError(f"alloy-sqp {name} must be a positive integer, got {value!r}")
  return value


def _nonnegative_int(options: dict[str, Any], name: str, default: int) -> int:
  value = options.get(name, default)
  if isinstance(value, bool) or not isinstance(value, int) or value < 0:
    raise ValueError(f"alloy-sqp {name} must be a non-negative integer, got {value!r}")
  return value


def _positive_float(options: dict[str, Any], name: str, default: float) -> float:
  value = options.get(name, default)
  if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
    raise ValueError(f"alloy-sqp {name} must be finite and positive, got {value!r}")
  return float(value)


def _csc(entries: list[tuple[int, int, int]], n_cols: int) -> tuple[list[int], list[int], list[int], list[int]]:
  """Order ``(row, col, src)`` triples column-major and return
  ``(col_ptr, rows, cols, srcs)``; ``src`` is the entry's index in the oracle's
  compact value buffer, or -1 for a structural-only entry."""
  ordered = sorted(entries, key=lambda e: (e[1], e[0]))
  col_ptr = [0] * (n_cols + 1)
  for _, c, _ in ordered:
    col_ptr[c + 1] += 1
  for c in range(n_cols):
    col_ptr[c + 1] += col_ptr[c]
  return col_ptr, [e[0] for e in ordered], [e[1] for e in ordered], [e[2] for e in ordered]


def _int_table(ctype: str, name: str, values: list[int]) -> str:
  return f"  static {ctype} {name}[{max(len(values), 1)}] = {{ {', '.join(str(v) for v in values) if values else '0'} }};"


def _ldl_symbolic(n: int, col_ptr: list[int], rows: list[int]) -> tuple[list[int], list[int]]:
  """Elimination tree and column counts of the LDL^T factor of an upper-triangular
  CSC pattern (Davis's ``ldl_symbolic``). Returns ``(parent, col_ptr_of_L)``."""
  parent, flag, nz = [-1] * n, [-1] * n, [0] * n
  for k in range(n):
    flag[k] = k
    for p in range(col_ptr[k], col_ptr[k + 1]):
      i = rows[p]
      while i < k and flag[i] != k:
        if parent[i] == -1:
          parent[i] = k
        nz[i] += 1
        flag[i] = k
        i = parent[i]
  l_ptr = [0] * (n + 1)
  for k in range(n):
    l_ptr[k + 1] = l_ptr[k] + nz[k]
  return parent, l_ptr


def render_wrapper(fun: Function, ctx: SolverWrapperCtx) -> list[str]:
  desc: SolverDescriptor = fun.descriptor  # ty: ignore[unresolved-attribute]
  base, grad, jac, hess, bounds = desc.base, desc.grad, desc.jac, desc.hess, desc.bounds
  assert base is not None and grad is not None and hess is not None and bounds is not None
  assert jac is not None or not (desc.n_eq + desc.n_ineq)
  assert desc.jac_sparsity is not None and desc.hess_sparsity is not None
  n, nh, ng, m = desc.n, desc.n_eq, desc.n_ineq, desc.n_eq + desc.n_ineq
  jrows, jcols = desc.jac_sparsity.rows, desc.jac_sparsity.cols
  hrows, hcols = desc.hess_sparsity.rows, desc.hess_sparsity.cols
  opts = _option_map(desc.options)
  max_iter = _positive_int(opts, "max_iter", 50)
  tol = _positive_float(opts, "tol", 1e-6)
  dual_tol = _positive_float(opts, "dual_tol", 1e-4)
  beta = _positive_float(opts, "line_search_beta", 0.7)
  merit_offset = _positive_float(opts, "merit", 10.0)
  regularization = _positive_float(opts, "regularization", 1e-6)
  qp_max_iter = _positive_int(opts, "qp_max_iter", 50)
  qp_tol = _positive_float(opts, "qp_tol", 1e-6)
  watchdog = _nonnegative_int(opts, "watchdog", 0)
  if beta >= 1.0:
    raise ValueError(f"alloy-sqp line_search_beta must be below 1, got {beta!r}")
  globalization = opts.get("globalization", "filter")
  if globalization not in ("filter", "l1"):
    raise ValueError(f"alloy-sqp globalization must be 'filter' or 'l1', got {globalization!r}")
  if globalization == "filter" and "watchdog" in opts:
    raise ValueError("alloy-sqp watchdog applies only to globalization='l1'")
  hessian_mode = opts.get("hessian", "exact")
  if hessian_mode not in ("exact", "objective"):
    raise ValueError(f"alloy-sqp hessian must be 'exact' or 'objective', got {hessian_mode!r}")
  qp_mode = opts.get("qp", "sparse")
  if qp_mode not in ("sparse", "dense"):
    raise ValueError(f"alloy-sqp qp must be 'sparse' or 'dense', got {qp_mode!r}")
  sparse = qp_mode == "sparse"
  trace = opts.get("trace", False)
  if not isinstance(trace, bool):
    raise ValueError(f"alloy-sqp trace must be a bool, got {trace!r}")
  trace_prefix = f"[alloy-sqp {re.sub(r'\W', '_', fun.name)}]"

  # QP patterns, fixed across SQP iterations: P is the upper triangle of the
  # Lagrangian Hessian unioned with the full diagonal (regularization writes
  # every diagonal entry) and, when there are equalities, with the pattern of
  # the constraint-normal term A.T @ A. A and G are the equality and inequality
  # row blocks of the constraint Jacobian.
  upper: dict[tuple[int, int], int] = {}
  for k, (r, c) in enumerate(zip(hrows, hcols, strict=True)):
    upper.setdefault((min(r, c), max(r, c)), k)
  for i in range(n):
    upper.setdefault((i, i), -1)
  eq_rows: dict[int, list[tuple[int, int]]] = {}
  for k, (r, c) in enumerate(zip(jrows, jcols, strict=True)):
    if r < nh:
      eq_rows.setdefault(r, []).append((c, k))
  normal_terms = [(min(ca, cb), max(ca, cb), ka, kb) for row in eq_rows.values() for ca, ka in row for cb, kb in row if ca <= cb]
  for r, c, _, _ in normal_terms:
    upper.setdefault((r, c), -1)
  p_ptr, p_rows, p_cols, p_src = _csc([(r, c, k) for (r, c), k in upper.items()], n)
  p_slot = {(r, c): k for k, (r, c) in enumerate(zip(p_rows, p_cols, strict=True))}
  p_diag = [p_slot[(i, i)] for i in range(n)]
  a_ptr, a_rows, _, a_src = _csc([(r, c, k) for k, (r, c) in enumerate(zip(jrows, jcols, strict=True)) if r < nh], n)
  g_ptr, g_rows, _, g_src = _csc([(r - nh, c, k) for k, (r, c) in enumerate(zip(jrows, jcols, strict=True)) if r >= nh], n)
  nnz_j, nnz_h, nnz_p = len(jrows), len(hrows), len(p_rows)
  ldl_parent, ldl_ptr = _ldl_symbolic(n, p_ptr, p_rows)
  nnz_l = ldl_ptr[n]
  normal_slot = [p_slot[(r, c)] for r, c, _, _ in normal_terms]
  normal_a = [ka for _, _, ka, _ in normal_terms]
  normal_b = [kb for _, _, _, kb in normal_terms]

  param_args = [f"in{4 + i}" for i in range(len(desc.param_names))]
  base_args = lambda x, f, g: ", ".join([x, *param_args, f, *((g,) if m else ()), "w"])  # noqa: E731
  one_out_args = lambda x, out: ", ".join([x, *param_args, out, "w"])  # noqa: E731
  hess_args = lambda x, lf, lg, out: ", ".join([x, lf, *((lg,) if m else ()), *param_args, out, "w"])  # noqa: E731
  bounds_args = ", ".join([*param_args, "xlb", "xub", *(("gl", "gu") if ng else ()), "w"])
  signature = [
    *(f"const double* in{i}" for i in range(len(desc.input_signature))),
    *(f"double* out{i}" for i in range(len(desc.output_signature))),
    "double* w",
  ]
  base_raw, grad_raw = ctx.raw_symbol_of(base), ctx.raw_symbol_of(grad)
  jac_raw = ctx.raw_symbol_of(jac) if jac is not None else ""
  hess_raw, bounds_raw = ctx.raw_symbol_of(hess), ctx.raw_symbol_of(bounds)
  interface = "sparse" if sparse else "dense"

  lines = [
    *(("#include <stdio.h>",) if trace else ()),
    f"// PIQP-backed SQP wrapper for {fun.name} (n={n}, equality={nh}, inequality={ng}, qp={qp_mode},"
    f" nnz jac/hess = {nnz_j}/{nnz_h}, nnz P/A/G = {nnz_p}/{len(a_rows)}/{len(g_rows)}).",
    f"static void {ctx.raw_symbol}({', '.join(signature)}) {{",
    "  double stats_t0 = alloy_clock_s();",
    f"  static double x[{n}], trial[{n}], step[{n}], grad_buf[{n}], dual_buf[{n}], cq[{n}];",
    f"  static double g[{max(m, 1)}], trial_g[{max(m, 1)}], jac_buf[{max(nnz_j, 1)}], hess_buf[{max(nnz_h, 1)}];",
    f"  static double lam_g[{max(m, 1)}], lam_box[{n}], qp_lam_g[{max(m, 1)}], qp_lam_box[{n}], zero_lam[{max(m, 1)}];",
    f"  static double xlb[{n}], xub[{n}], dl[{n}], du[{n}];",
    f"  static double gl[{max(ng, 1)}], gu[{max(ng, 1)}], b[{max(nh, 1)}], hl[{max(ng, 1)}], hu[{max(ng, 1)}];",
    # Sparsity tables. The oracle value buffers are in the sparsity's own COO
    # order, so every fill goes through a source-index table rather than
    # unrolled per-entry assignments.
    _int_table("const int", "jac_rows", list(jrows)),
    _int_table("const int", "jac_cols", list(jcols)),
    _int_table("piqp_int", "P_p", p_ptr),
    _int_table("piqp_int", "P_i", p_rows),
    _int_table("const int", "P_j", p_cols),
    _int_table("const int", "P_src", p_src),
    _int_table("const int", "P_diag", p_diag),
    f"  static double P_x[{nnz_p}];",
    # Symbolic LDL^T of the P pattern (Davis's ldl_symbolic), for the modified
    # factorization that convexifies the Hessian.
    _int_table("const int", "ldl_parent", ldl_parent),
    _int_table("const int", "ldl_p", ldl_ptr),
    *(
      (
        _int_table("const int", "normal_slot", normal_slot),
        _int_table("const int", "normal_a", normal_a),
        _int_table("const int", "normal_b", normal_b),
      )
      if nh
      else ()
    ),
    f"  static int ldl_i[{max(nnz_l, 1)}], ldl_nz[{n}], ldl_flag[{n}], ldl_pattern[{n}];",
    f"  static double ldl_x[{max(nnz_l, 1)}], ldl_y[{n}], ldl_d[{n}];",
  ]
  if sparse:
    lines += [f"  static piqp_csc P_csc = {{ {n}, {n}, {nnz_p}, P_p, P_i, P_x }};"]
    if nh:
      lines += [
        _int_table("piqp_int", "A_p", a_ptr),
        _int_table("piqp_int", "A_i", a_rows),
        _int_table("const int", "A_src", a_src),
        f"  static double A_x[{len(a_rows)}];",
        f"  static piqp_csc A_csc = {{ {nh}, {n}, {len(a_rows)}, A_p, A_i, A_x }};",
      ]
    if ng:
      lines += [
        _int_table("piqp_int", "G_p", g_ptr),
        _int_table("piqp_int", "G_i", g_rows),
        _int_table("const int", "G_src", g_src),
        f"  static double G_x[{len(g_rows)}];",
        f"  static piqp_csc G_csc = {{ {ng}, {n}, {len(g_rows)}, G_p, G_i, G_x }};",
      ]
  else:
    lines += [f"  static double Pcol[{n * n}];"]
    if nh:
      lines += [f"  static double Acol[{nh * n}];", _int_table("const int", "A_src", a_src), _int_table("piqp_int", "A_i", a_rows)]
    if ng:
      lines += [f"  static double Gcol[{ng * n}];", _int_table("const int", "G_src", g_src), _int_table("piqp_int", "G_i", g_rows)]
  lines += [
    *(("  static double filter_f[20], filter_v[20]; int filter_count = 0;",) if globalization == "filter" else ()),
    *(
      (
        f"  static double checkpoint_x[{n}], checkpoint_lam_g[{max(m, 1)}], checkpoint_lam_box[{n}];",
        f"  static double checkpoint_step[{n}], checkpoint_qp_lam_g[{max(m, 1)}], checkpoint_qp_lam_box[{n}];",
        f"  static double checkpoint_g[{max(m, 1)}], checkpoint_grad[{n}];",
        "  double checkpoint_f = 0.0, watchdog_phi = 0.0, watchdog_dphi = 0.0, watchdog_mu = 0.0; int watchdog_step = 0;",
      )
      if globalization == "l1" and watchdog
      else ()
    ),
    "  double stats_t_fe = 0.0, stats_t_qp = 0.0, stats_t_globalization = 0.0;",
    "  double stats_step_inf = 0.0, stats_alpha = 0.0, stats_merit_penalty = 0.0;",
    "  int stats_backtracks = 0, stats_qp_iter = 0;",
    "  double bounds_fe0 = alloy_clock_s();",
    f"  {bounds_raw}({bounds_args});",
    "  stats_t_fe += alloy_clock_s() - bounds_fe0;",
    "  int inputs_ok = 1;",
    f"  for (int i = 0; i < {n}; ++i) inputs_ok = inputs_ok && !isnan(xlb[i]) && !isnan(xub[i]) && xlb[i] <= xub[i] && isfinite(in0[i]) && isfinite(in3[i]);",
    f"  for (int i = 0; i < {nh}; ++i) inputs_ok = inputs_ok && isfinite(in1[i]);",
    f"  for (int i = 0; i < {ng}; ++i) inputs_ok = inputs_ok && !isnan(gl[i]) && !isnan(gu[i]) && gl[i] <= gu[i] && isfinite(in2[i]);",
    f"  for (int i = 0; i < {n}; ++i) {{ x[i] = fmax(xlb[i], fmin(xub[i], in0[i])); lam_box[i] = in3[i]; }}",
    f"  for (int i = 0; i < {nh}; ++i) lam_g[i] = in1[i];",
    f"  for (int i = 0; i < {ng}; ++i) lam_g[{nh} + i] = in2[i];",
    f"  for (int i = 0; i < {m}; ++i) {{ g[i] = 0.0; zero_lam[i] = 0.0; }}",
    "  int n_eval_f = 0, n_eval_grad_f = 0, n_eval_g = 0, n_eval_jac_g = 0, n_eval_h = 0;",
    "  double f = 0.0, trial_f = 0.0, primal = 0.0, complementarity = 0.0, stationarity = 0.0;",
    "  int status = ALLOY_SOLVE_MAX_ITER, native_status = PIQP_SOLVED, iterations = 0;",
    f"  piqp_workspace* qp = NULL; piqp_settings settings; piqp_set_default_settings_{interface}(&settings);",
    f"  settings.max_iter = {qp_max_iter}; settings.eps_abs = {qp_tol}; settings.eps_rel = {qp_tol}; settings.eps_duality_gap_abs = {qp_tol}; settings.eps_duality_gap_rel = {qp_tol}; settings.verbose = 0; settings.preconditioner_reuse_on_update = 0;",
    f"  const double sqp_tol = {tol}, sqp_dual_tol = {dual_tol}, ls_beta = {beta}, merit_offset = {merit_offset}, min_reg = {regularization};",
    "  if (!inputs_ok) status = ALLOY_SOLVE_NUMERICS;",
    f"  for (int sqp_iter = 0; sqp_iter <= {max_iter} && inputs_ok; ++sqp_iter) {{",
    "    iterations = sqp_iter;",
    "    double fe0 = alloy_clock_s();",
    f"    {base_raw}({base_args('x', '&f', 'g')}); n_eval_f++;" + (" n_eval_g++;" if m else ""),
    f"    {grad_raw}({one_out_args('x', 'grad_buf')}); n_eval_grad_f++;",
    *((f"    {jac_raw}({one_out_args('x', 'jac_buf')}); n_eval_jac_g++;",) if m else ()),
    "    stats_t_fe += alloy_clock_s() - fe0;",
    "    int oracle_ok = isfinite(f);",
    f"    for (int i = 0; i < {n}; ++i) oracle_ok = oracle_ok && isfinite(grad_buf[i]);",
    f"    for (int i = 0; i < {m}; ++i) oracle_ok = oracle_ok && isfinite(g[i]);",
    f"    for (int i = 0; i < {nnz_j}; ++i) oracle_ok = oracle_ok && isfinite(jac_buf[i]);",
    "    if (!oracle_ok) { status = ALLOY_SOLVE_NUMERICS; break; }",
    "    primal = 0.0; complementarity = 0.0; stationarity = 0.0;",
  ]
  if nh:
    lines += [f"    for (int i = 0; i < {nh}; ++i) if (fabs(g[i]) > primal) primal = fabs(g[i]);"]
  if ng:
    lines += [
      f"    for (int i = 0; i < {ng}; ++i) {{ double value = g[{nh} + i], v = value < gl[i] ? gl[i] - value : (value > gu[i] ? value - gu[i] : 0.0); if (v > primal) primal = v; double lam = lam_g[{nh} + i], comp = lam > 0.0 ? (gu[i] >= 1e19 ? fabs(lam) : fabs(lam) * fabs(gu[i] - value)) : (lam < 0.0 ? (gl[i] <= -1e19 ? fabs(lam) : fabs(lam) * fabs(value - gl[i])) : 0.0); if (comp > complementarity) complementarity = comp; }}"
    ]
  lines += [
    f"    for (int i = 0; i < {n}; ++i) {{ double v = x[i] < xlb[i] ? xlb[i] - x[i] : (x[i] > xub[i] ? x[i] - xub[i] : 0.0); if (v > primal) primal = v; double lam = lam_box[i], comp = lam > 0.0 ? (xub[i] >= 1e19 ? fabs(lam) : fabs(lam) * fabs(xub[i] - x[i])) : (lam < 0.0 ? (xlb[i] <= -1e19 ? fabs(lam) : fabs(lam) * fabs(x[i] - xlb[i])) : 0.0); if (comp > complementarity) complementarity = comp; }}",
    f"    for (int i = 0; i < {n}; ++i) dual_buf[i] = grad_buf[i] + lam_box[i];",
    f"    for (int k = 0; k < {nnz_j}; ++k) dual_buf[jac_cols[k]] += jac_buf[k] * lam_g[jac_rows[k]];",
    f"    for (int i = 0; i < {n}; ++i) if (fabs(dual_buf[i]) > stationarity) stationarity = fabs(dual_buf[i]);",
    *(
      (
        f'    fprintf(stderr, "{trace_prefix} iter=%d primal=%.3e complementarity=%.3e stationarity=%.3e\\n", sqp_iter, primal, complementarity, stationarity);',
      )
      if trace
      else ()
    ),
    "    if (primal <= sqp_tol && complementarity <= sqp_dual_tol && stationarity <= sqp_dual_tol) { status = ALLOY_SOLVE_OK; break; }",
    f"    if (sqp_iter == {max_iter}) break;",
    "    double hess_fe0 = alloy_clock_s();",
    f"    {hess_raw}({hess_args('x', '&(double){1.0}', 'lam_g' if hessian_mode == 'exact' else 'zero_lam', 'hess_buf')}); n_eval_h++;",
    "    stats_t_fe += alloy_clock_s() - hess_fe0;",
    "    int hess_ok = 1;",
    f"    for (int i = 0; i < {nnz_h}; ++i) hess_ok = hess_ok && isfinite(hess_buf[i]);",
    "    if (!hess_ok) { status = ALLOY_SOLVE_NUMERICS; break; }",
  ]

  def convexify_lines(indent: str) -> list[str]:
    """Assemble P = H + normal_shift * A.T @ A + min_reg * I and convexify it
    with a modified sparse LDL^T: factorize, and whenever a pivot is not at
    least min_reg raise it to its own magnitude. Because only diagonal entries
    are touched, L*D*L^T equals the assembled P plus that diagonal, so the
    correction is an exact diagonal shift and `shift` is zero wherever the
    assembled matrix is already positive definite. The whole pass costs
    O(nnz(L)) instead of a dense O(n^3) probe."""
    return [
      f"{indent}for (int k = 0; k < {nnz_p}; ++k) {{ int src = P_src[k]; P_x[k] = src >= 0 ? hess_buf[src] : 0.0; }}",
      *(
        (
          f"{indent}if (normal_shift > 0.0) for (int k = 0; k < {len(normal_terms)}; ++k)"
          f" P_x[normal_slot[k]] += normal_shift * jac_buf[normal_a[k]] * jac_buf[normal_b[k]];",
        )
        if nh
        else ()
      ),
      f"{indent}for (int i = 0; i < {n}; ++i) P_x[P_diag[i]] += min_reg;",
      f"{indent}shift = 0.0;",
      f"{indent}for (int i = 0; i < {n}; ++i) {{ ldl_flag[i] = -1; ldl_y[i] = 0.0; }}",
      f"{indent}for (int k = 0; k < {n}; ++k) {{",
      f"{indent}  int top = {n}; ldl_flag[k] = k; ldl_nz[k] = 0;",
      f"{indent}  for (int p = P_p[k]; p < P_p[k + 1]; ++p) {{",
      f"{indent}    int i = (int)P_i[p]; if (i > k) continue;",
      f"{indent}    ldl_y[i] += P_x[p];",
      f"{indent}    int len = 0;",
      f"{indent}    for (; ldl_flag[i] != k; i = ldl_parent[i]) {{ ldl_pattern[len++] = i; ldl_flag[i] = k; }}",
      f"{indent}    while (len > 0) ldl_pattern[--top] = ldl_pattern[--len];",
      f"{indent}  }}",
      f"{indent}  double dk = ldl_y[k]; ldl_y[k] = 0.0;",
      f"{indent}  for (; top < {n}; ++top) {{",
      f"{indent}    int i = ldl_pattern[top]; double yi = ldl_y[i]; ldl_y[i] = 0.0;",
      f"{indent}    for (int p = ldl_p[i]; p < ldl_p[i] + ldl_nz[i]; ++p) ldl_y[ldl_i[p]] -= ldl_x[p] * yi;",
      f"{indent}    double lki = yi / ldl_d[i]; dk -= lki * yi;",
      f"{indent}    ldl_i[ldl_p[i] + ldl_nz[i]] = k; ldl_x[ldl_p[i] + ldl_nz[i]] = lki; ldl_nz[i]++;",
      f"{indent}  }}",
      # A pivot below min_reg is raised to its own magnitude, never clamped to
      # min_reg: clamping divides that column by a near-zero pivot, and the
      # resulting L entries drive the next pivots further negative until the
      # shift runs away (measured 9.4e19 -> 2.0e46 -> 4.8e89 on canonical race).
      f"{indent}  double target = fabs(dk); if (target < min_reg) target = min_reg;",
      f"{indent}  if (target > dk) {{ double e = target - dk; P_x[P_diag[k]] += e; if (e > shift) shift = e; dk = target; }}",
      f"{indent}  ldl_d[k] = dk;",
      f"{indent}}}",
    ]

  # With equalities, damp curvature normal to the linearized constraint
  # manifold before damping the diagonal: A.T @ A is constant over that
  # manifold, so it convexifies the QP without shrinking the feasible SQP
  # direction, which a diagonal shift does (it costs canonical race its
  # superlinear rate — stationarity then decays by only ~0.7 per iteration).
  if nh:
    lines += [
      "    double normal_shift = 0.0, shift = 0.0;",
      "    for (int attempt = 0; attempt < 10; ++attempt) {",
      *convexify_lines("      "),
      "      if (shift == 0.0) break;",
      "      normal_shift = normal_shift == 0.0 ? 1.0 : 10.0 * normal_shift;",
      "    }",
    ]
  else:
    lines += ["    double shift = 0.0;", *convexify_lines("    ")]
  if hessian_mode == "exact" and nh:
    # Negative reduced curvature survived every normal-space attempt, so the
    # exact Lagrangian model itself is the problem: drop the constraint
    # multipliers and convexify the objective Hessian instead.
    lines += [
      "    if (shift > 0.0) {",
      "      double objective_hess_fe0 = alloy_clock_s();",
      f"      {hess_raw}({hess_args('x', '&(double){1.0}', 'zero_lam', 'hess_buf')}); n_eval_h++;",
      "      stats_t_fe += alloy_clock_s() - objective_hess_fe0; hess_ok = 1;",
      f"      for (int i = 0; i < {nnz_h}; ++i) hess_ok = hess_ok && isfinite(hess_buf[i]);",
      "      if (!hess_ok) { status = ALLOY_SOLVE_NUMERICS; break; }",
      "      normal_shift = 0.0;",
      *convexify_lines("      "),
      "    }",
    ]
  lines += [f"    for (int i = 0; i < {n}; ++i) cq[i] = grad_buf[i];"]
  if not sparse:
    lines += [
      f"    for (int i = 0; i < {n * n}; ++i) Pcol[i] = 0.0;",
      f"    for (int k = 0; k < {nnz_p}; ++k) {{ Pcol[P_i[k] + P_j[k] * {n}] = P_x[k]; Pcol[P_j[k] + P_i[k] * {n}] = P_x[k]; }}",
    ]
  if nh:
    if sparse:
      lines += [f"    for (int k = 0; k < {len(a_rows)}; ++k) A_x[k] = jac_buf[A_src[k]];"]
    else:
      lines += [
        f"    for (int i = 0; i < {nh * n}; ++i) Acol[i] = 0.0;",
        f"    for (int k = 0; k < {len(a_rows)}; ++k) Acol[A_i[k] * {n} + jac_cols[A_src[k]]] = jac_buf[A_src[k]];",
      ]
    lines += [f"    for (int i = 0; i < {nh}; ++i) b[i] = -g[i];"]
  if ng:
    if sparse:
      lines += [f"    for (int k = 0; k < {len(g_rows)}; ++k) G_x[k] = jac_buf[G_src[k]];"]
    else:
      lines += [
        f"    for (int i = 0; i < {ng * n}; ++i) Gcol[i] = 0.0;",
        f"    for (int k = 0; k < {len(g_rows)}; ++k) Gcol[G_i[k] * {n} + jac_cols[G_src[k]]] = jac_buf[G_src[k]];",
      ]
    lines += [
      f"    for (int i = 0; i < {ng}; ++i) {{ hl[i] = gl[i] <= -1e19 ? -PIQP_INF : gl[i] - g[{nh} + i]; hu[i] = gu[i] >= 1e19 ? PIQP_INF : gu[i] - g[{nh} + i]; }}"
    ]
  qp_P = "&P_csc" if sparse else "Pcol"
  qp_A = ("&A_csc" if sparse else "Acol") if nh else "NULL"
  qp_G = ("&G_csc" if sparse else "Gcol") if ng else "NULL"
  lines += [
    f"    for (int i = 0; i < {n}; ++i) {{ dl[i] = xlb[i] <= -1e19 ? -PIQP_INF : xlb[i] - x[i]; du[i] = xub[i] >= 1e19 ? PIQP_INF : xub[i] - x[i]; }}",
    f"    piqp_data_{interface} data = {{ {n}, {nh}, {ng}, {qp_P}, cq, {qp_A}, {'b' if nh else 'NULL'}, {qp_G}, {'hl' if ng else 'NULL'}, {'hu' if ng else 'NULL'}, dl, du }};",
    "    double qp0 = alloy_clock_s();",
    f"    if (qp == NULL) piqp_setup_{interface}(&qp, &data, &settings);",
    f"    else piqp_update_{interface}(qp, {qp_P}, cq, {qp_A}, {'b' if nh else 'NULL'}, {qp_G}, {'hl' if ng else 'NULL'}, {'hu' if ng else 'NULL'}, dl, du);",
    "    piqp_status qp_status = piqp_solve(qp); stats_t_qp += alloy_clock_s() - qp0; native_status = (int)qp_status;",
    "    stats_qp_iter += (int)qp->result->info.iter;",
    *(
      (
        f'    if (qp_status != PIQP_SOLVED) fprintf(stderr, "{trace_prefix} iter=%d qp_status=%d qp_iter=%d primal_res=%.3e dual_res=%.3e gap=%.3e rel_gap=%.3e qp not solved\\n", sqp_iter + 1, (int)qp_status, (int)qp->result->info.iter, qp->result->info.primal_res, qp->result->info.dual_res, qp->result->info.duality_gap, qp->result->info.duality_gap_rel);',
      )
      if trace
      else ()
    ),
    "    // like laopt, continue with PIQP's best iterate on max-iter or infeasibility; globalization and the KKT test judge it",
    "    if (qp_status != PIQP_SOLVED && qp_status != PIQP_MAX_ITER_REACHED && qp_status != PIQP_PRIMAL_INFEASIBLE && qp_status != PIQP_DUAL_INFEASIBLE) { status = ALLOY_SOLVE_NUMERICS; break; }",
    f"    for (int i = 0; i < {n}; ++i) step[i] = qp->result->x[i];",
  ]
  if nh:
    lines += [f"    for (int i = 0; i < {nh}; ++i) qp_lam_g[i] = qp->result->y[i];"]
  if ng:
    lines += [f"    for (int i = 0; i < {ng}; ++i) qp_lam_g[{nh} + i] = qp->result->z_u[i] - qp->result->z_l[i];"]
  lines += [
    f"    for (int i = 0; i < {n}; ++i) qp_lam_box[i] = qp->result->z_bu[i] - qp->result->z_bl[i];",
    "    double step_inf = 0.0, grad_step = 0.0;",
    f"    for (int i = 0; i < {n}; ++i) {{ if (fabs(step[i]) > step_inf) step_inf = fabs(step[i]); grad_step += grad_buf[i] * step[i]; }}",
    "    stats_step_inf = step_inf;",
    *(
      (
        f'    fprintf(stderr, "{trace_prefix} iter=%d qp_status=%d qp_iter=%d primal=%.3e step_inf=%.3e shift=%.3e\\n", sqp_iter + 1, (int)qp_status, (int)qp->result->info.iter, primal, step_inf, shift);',
      )
      if trace
      else ()
    ),
    "    double globalization0 = alloy_clock_s(), globalization_fe = 0.0, alpha = 1.0; int accepted = 0, ls_backtracks = 0;",
  ]

  def violation_lines(indent: str, values: str, point: str, target: str) -> list[str]:
    result = [f"{indent}double {target} = 0.0;"]
    if nh:
      result += [f"{indent}for (int i = 0; i < {nh}; ++i) {target} += fabs({values}[i]);"]
    if ng:
      result += [
        f"{indent}for (int i = 0; i < {ng}; ++i) {{ double value = {values}[{nh} + i]; {target} += fmax(0.0, gl[i] - value) + fmax(0.0, value - gu[i]); }}"
      ]
    result += [f"{indent}for (int i = 0; i < {n}; ++i) {target} += fmax(0.0, xlb[i] - {point}[i]) + fmax(0.0, {point}[i] - xub[i]);"]
    return result

  def trial_lines(indent: str) -> list[str]:
    return [
      f"{indent}for (int i = 0; i < {n}; ++i) trial[i] = x[i] + alpha * step[i];",
      f"{indent}double trial_fe0 = alloy_clock_s();",
      f"{indent}{base_raw}({base_args('trial', '&trial_f', 'trial_g')}); n_eval_f++;" + (" n_eval_g++;" if m else ""),
      f"{indent}double trial_fe = alloy_clock_s() - trial_fe0; stats_t_fe += trial_fe; globalization_fe += trial_fe;",
      f"{indent}int trial_ok = isfinite(trial_f);",
      f"{indent}for (int i = 0; i < {m}; ++i) trial_ok = trial_ok && isfinite(trial_g[i]);",
      *violation_lines(indent, "trial_g", "trial", "trial_violation"),
    ]

  if globalization == "filter":
    lines += [
      *violation_lines("    ", "g", "x", "violation"),
      "    if (filter_count == 0) { filter_f[0] = f; filter_v[0] = violation; filter_count = 1; }",
      "    for (int attempt = 0; attempt < 100 && alpha >= 1e-4; ++attempt) {",
      *trial_lines("      "),
      "      int reject = !trial_ok;",
      "      for (int j = 0; j < filter_count && !reject; ++j) if (trial_f > filter_f[j] && trial_violation > filter_v[j]) reject = 1;",
      "      double model = alpha * grad_step;",
      "      if (!reject && model < 0.0 && trial_f > f + 0.25 * model) reject = 1;",
      "      if (!reject && model >= 0.0) for (int j = 0; j < filter_count && !reject; ++j) if (trial_f > filter_f[j] - 1e-5 * filter_v[j] && trial_violation > (1.0 - 1e-5) * filter_v[j]) reject = 1;",
      "      if (!reject) {",
      "        int keep = 0;",
      "        for (int j = 0; j < filter_count; ++j) if (trial_f > filter_f[j] || trial_violation > filter_v[j]) { filter_f[keep] = filter_f[j]; filter_v[keep++] = filter_v[j]; }",
      "        if (keep < 20) { filter_f[keep] = trial_f; filter_v[keep++] = trial_violation; }",
      "        else { filter_f[0] = trial_f; filter_v[0] = trial_violation; }",
      "        filter_count = keep > 0 ? keep : 1; accepted = 1; break;",
      "      }",
      "      alpha *= ls_beta; ls_backtracks++;",
      "    }",
    ]
  else:
    lines += violation_lines("    ", "g", "x", "violation")
    if watchdog:
      lines += [
        "    int fallback = 0;",
        "    if (watchdog_step == 0) {",
        "      watchdog_mu = merit_offset;",
        f"      for (int i = 0; i < {m}; ++i) if (fabs(qp_lam_g[i]) + merit_offset > watchdog_mu) watchdog_mu = fabs(qp_lam_g[i]) + merit_offset;",
        f"      for (int i = 0; i < {n}; ++i) if (fabs(qp_lam_box[i]) + merit_offset > watchdog_mu) watchdog_mu = fabs(qp_lam_box[i]) + merit_offset;",
        "      watchdog_phi = f + watchdog_mu * violation; watchdog_dphi = grad_step - watchdog_mu * violation;",
        f"      for (int i = 0; i < {n}; ++i) {{ checkpoint_x[i] = x[i]; checkpoint_lam_box[i] = lam_box[i]; checkpoint_step[i] = step[i]; checkpoint_qp_lam_box[i] = qp_lam_box[i]; checkpoint_grad[i] = grad_buf[i]; }}",
        f"      for (int i = 0; i < {m}; ++i) {{ checkpoint_lam_g[i] = lam_g[i]; checkpoint_qp_lam_g[i] = qp_lam_g[i]; checkpoint_g[i] = g[i]; }}",
        "      checkpoint_f = f;",
        *trial_lines("      "),
        "      double trial_phi = trial_f + watchdog_mu * trial_violation;",
        "      if (trial_ok && trial_phi <= watchdog_phi + 0.25 * watchdog_dphi) accepted = 1;",
        "      else if (trial_ok && trial_phi < 1e3 * fmax(1.0, fabs(watchdog_phi))) { watchdog_step = 1; accepted = 1; }",
        "      else fallback = 1;",
        "    } else if (watchdog_step < " + str(watchdog) + ") {",
        "      double current_phi = f + watchdog_mu * violation;",
        "      if (current_phi <= watchdog_phi) {",
        f"        for (int i = 0; i < {n}; ++i) {{ checkpoint_x[i] = x[i]; checkpoint_lam_box[i] = lam_box[i]; checkpoint_step[i] = step[i]; checkpoint_qp_lam_box[i] = qp_lam_box[i]; checkpoint_grad[i] = grad_buf[i]; }}",
        f"        for (int i = 0; i < {m}; ++i) {{ checkpoint_lam_g[i] = lam_g[i]; checkpoint_qp_lam_g[i] = qp_lam_g[i]; checkpoint_g[i] = g[i]; }}",
        "        checkpoint_f = f;",
        "      }",
        *trial_lines("      "),
        "      double trial_phi = trial_f + watchdog_mu * trial_violation;",
        "      if (trial_ok && trial_phi <= watchdog_phi + 0.25 * watchdog_dphi) { watchdog_step = 0; accepted = 1; }",
        "      else if (trial_ok && trial_phi < 1e3 * fmax(1.0, fabs(watchdog_phi))) { watchdog_step++; accepted = 1; }",
        "      else fallback = 1;",
        "    } else fallback = 1;",
        "    if (fallback) {",
        f"      for (int i = 0; i < {n}; ++i) {{ x[i] = checkpoint_x[i]; lam_box[i] = checkpoint_lam_box[i]; step[i] = checkpoint_step[i]; qp_lam_box[i] = checkpoint_qp_lam_box[i]; grad_buf[i] = checkpoint_grad[i]; }}",
        f"      for (int i = 0; i < {m}; ++i) {{ lam_g[i] = checkpoint_lam_g[i]; qp_lam_g[i] = checkpoint_qp_lam_g[i]; g[i] = checkpoint_g[i]; }}",
        "      f = checkpoint_f; grad_step = 0.0;",
        f"      for (int i = 0; i < {n}; ++i) grad_step += grad_buf[i] * step[i];",
        *violation_lines("      ", "g", "x", "fallback_violation"),
        "      double mu = merit_offset; if (fallback_violation > 0.0) mu += fabs(grad_step) / (0.5 * fallback_violation);",
        "      stats_merit_penalty = mu; double merit = f + mu * fallback_violation, dmerit = grad_step - mu * fallback_violation; alpha = 1.0; accepted = 0;",
        "      for (int attempt = 0; attempt < 100 && alpha >= 1e-4; ++attempt) {",
        *trial_lines("        "),
        "        double trial_merit = trial_f + mu * trial_violation;",
        "        if (trial_ok && trial_merit <= merit + alpha * 0.25 * dmerit) { accepted = 1; break; }",
        "        alpha *= ls_beta; ls_backtracks++;",
        "      }",
        "      watchdog_step = 0;",
        "    } else stats_merit_penalty = watchdog_mu;",
      ]
    else:
      lines += [
        "    double mu = merit_offset; if (violation > 0.0) mu += fabs(grad_step) / (0.5 * violation);",
        "    stats_merit_penalty = mu; double merit = f + mu * violation, dmerit = grad_step - mu * violation;",
        "    for (int attempt = 0; attempt < 100 && alpha >= 1e-4; ++attempt) {",
        *trial_lines("      "),
        "      double trial_merit = trial_f + mu * trial_violation;",
        "      if (trial_ok && trial_merit <= merit + alpha * 0.25 * dmerit) { accepted = 1; break; }",
        "      alpha *= ls_beta; ls_backtracks++;",
        "    }",
      ]
  lines += [
    "    double globalization_elapsed = alloy_clock_s() - globalization0 - globalization_fe; if (globalization_elapsed > 0.0) stats_t_globalization += globalization_elapsed;",
    "    stats_backtracks += ls_backtracks; if (accepted) stats_alpha = alpha;",
    *(
      (
        f'    fprintf(stderr, "{trace_prefix} iter=%d globalization={globalization} trial_f=%.9e alpha=%.3e accepted=%d backtracks=%d penalty=%.3e\\n", sqp_iter + 1, trial_f, alpha, accepted, ls_backtracks, stats_merit_penalty);',
      )
      if trace
      else ()
    ),
    "    if (!accepted) { status = ALLOY_SOLVE_NUMERICS; break; }",
    f"    for (int i = 0; i < {n}; ++i) {{ x[i] += alpha * step[i]; lam_box[i] += alpha * (qp_lam_box[i] - lam_box[i]); }}",
    f"    for (int i = 0; i < {m}; ++i) lam_g[i] += alpha * (qp_lam_g[i] - lam_g[i]);",
    "  }",
    f"  for (int i = 0; i < {n}; ++i) out0[i] = x[i];",
    "  out1[0] = f;",
  ]
  if nh:
    lines += [f"  for (int i = 0; i < {nh}; ++i) {{ out2[i] = g[i]; out4[i] = lam_g[i]; }}"]
  if ng:
    lines += [f"  for (int i = 0; i < {ng}; ++i) {{ out3[i] = g[{nh} + i]; out5[i] = lam_g[{nh} + i]; }}"]
  lines += [f"  for (int i = 0; i < {n}; ++i) out6[i] = lam_box[i];"]
  lines += [
    f"  {ctx.stats_symbol}.version = ALLOY_SOLVER_STATS_VERSION;",
    f"  {ctx.stats_symbol}.status = status; {ctx.stats_symbol}.native_status = native_status; {ctx.stats_symbol}.iter = iterations;",
    f"  {ctx.stats_symbol}.obj = f; {ctx.stats_symbol}.t_fe = stats_t_fe; {ctx.stats_symbol}.t_solver = 0.0;",
    f"  {ctx.stats_symbol}.t_qp = stats_t_qp; {ctx.stats_symbol}.t_globalization = stats_t_globalization;",
    f"  {ctx.stats_symbol}.n_eval_f = n_eval_f; {ctx.stats_symbol}.n_eval_grad_f = n_eval_grad_f; {ctx.stats_symbol}.n_eval_g = n_eval_g;",
    f"  {ctx.stats_symbol}.n_eval_jac_g = n_eval_jac_g; {ctx.stats_symbol}.n_eval_h = n_eval_h; {ctx.stats_symbol}._pad0 = 0;",
    f"  {ctx.stats_symbol}.primal_viol = primal; {ctx.stats_symbol}.step_inf = stats_step_inf;",
    f"  {ctx.stats_symbol}.alpha = stats_alpha; {ctx.stats_symbol}.merit_penalty = stats_merit_penalty;",
    f"  {ctx.stats_symbol}.backtracks = stats_backtracks; {ctx.stats_symbol}.qp_iter = stats_qp_iter;",
    "  if (qp) { piqp_cleanup(qp); qp = NULL; }",
    "  double stats_t_total = alloy_clock_s() - stats_t0;",
    f"  {ctx.stats_symbol}.t_total = stats_t_total;",
    f"  {ctx.stats_symbol}.t_glue = stats_t_total - stats_t_fe - stats_t_qp - stats_t_globalization;",
    "}",
  ]
  return lines
