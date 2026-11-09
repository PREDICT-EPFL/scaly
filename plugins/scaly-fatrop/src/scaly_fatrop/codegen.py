"""Fill Fatrop's stage matrices from whole-problem or mapped stage oracles.

The five full_eval callbacks evaluate the whole horizon and return 1 to skip
stage evaluations. Static tables transfer compact derivatives into BLASFEO
matrices ordered [u_k, x_k].
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .stage_codegen import render_callbacks

if TYPE_CHECKING:
  from scaly.codegen.solver import SolverWrapperCtx
  from scaly.function.concrete import ConcreteFunction
  from scaly.solvers.model import SolverDescriptor


def render_wrapper(fun: ConcreteFunction, ctx: SolverWrapperCtx) -> list[str]:
  """Generate one OCPCInterface using the descriptor's checked stage layout."""
  desc: SolverDescriptor = fun.descriptor
  layout = desc.stage_layout
  n, m, stages = desc.n, desc.n_eq + desc.n_ineq, layout.N + 1
  nv = desc.n_var_blocks
  s, stats = ctx.symbol, ctx.stats_symbol
  stage = desc.stage_oracles
  jac, hess = desc.jac_sparsity, desc.hess_sparsity
  assert stage is not None or (jac is not None and hess is not None and desc.jac is not None and desc.hess is not None)
  assert desc.base is not None and desc.grad is not None and desc.bounds is not None
  sizes = [math.prod(shape) for _, shape in desc.input_signature[:nv]]
  offsets = [sum(sizes[:i]) for i in range(nv)]
  starts = [sum(layout.nx[j] + layout.nu[j] for j in range(k)) for k in range(stages)]
  var_stage, var_local, permutation = [], [], []
  for k, (nx, nu, start) in enumerate(zip(layout.nx, layout.nu, starts, strict=True)):
    for i in range(nx + nu):
      var_stage.append(k)
      local = i + nu if i < nx else i - nx
      var_local.append(local)
      permutation.append(start + local)
  row_stage, dyn_col = [-1] * m, [-1] * m
  pivots = set()
  for k, rows in enumerate(layout.dynamics):
    for i, row in enumerate(rows):
      row_stage[row], dyn_col[row] = k, i
      pivots.add((row, starts[k + 1] + i))
  order = []
  order_start = [0]
  for k in range(stages):
    for row in (*layout.eq[k], *layout.ineq[k]):
      row_stage[row] = k
      order.append(row)
    order.extend(m + i for i in range(starts[k], starts[k] + layout.nx[k] + layout.nu[k]))
    order_start.append(len(order))
  pivot_indices = [] if jac is None else [i for i, pair in enumerate(zip(jac.rows, jac.cols, strict=True)) if pair in pivots]

  lines: list[str] = []

  def emit(source: str) -> None:
    lines.extend(source.strip("\n").splitlines())

  def table(name: str, values) -> None:
    values = list(values)
    emit(f"static const int {s}_{name}[{max(1, len(values))}] = {{ {', '.join(map(str, values)) or '0'} }};")

  for name, values in (
    ("nx", layout.nx),
    ("nu", layout.nu),
    ("start", starts),
    ("perm", permutation),
    ("vstage", var_stage),
    ("vlocal", var_local),
    ("rstage", row_stage + var_stage),
    ("dyncol", dyn_col),
    ("order", order),
    ("order_start", order_start),
    ("jr", () if jac is None else jac.rows),
    ("jc", () if jac is None else jac.cols),
    ("pivot", pivot_indices),
    ("hr", () if hess is None else hess.rows),
    ("hc", () if hess is None else hess.cols),
  ):
    table(name, values)
  emit(f"""
  typedef struct {{
    double x[{n}], initial[{n}], g[{max(1, m)}], grad[{n}];
    double lower[{m + n}], upper[{m + n}], lam[{m + n}], factor[{max(1, m)}], obj;
    int kind[{m + n}], column[{m + n}], native[{m + n}], ng[{stages}], ngi[{stages}], invalid;
    double *w, t_fe, t_callback;
    const scaly_solver_option* const* solver_options;
    int n_f, n_g, n_grad, n_jac, n_hess;
""")
  if stage is None:
    assert jac is not None and hess is not None
    emit(f"    double j[{max(1, jac.nnz)}], h[{max(1, hess.nnz)}];")
  else:
    for name, output in zip(
      ("stage_dyn", "stage_initial", "stage_path", "stage_terminal", "stage_h", "terminal_h"), (*stage.jac.outputs, *stage.hess.outputs), strict=True
    ):
      emit(f"    double {name}[{max(1, output.size)}];")
  for i in range(len(desc.param_names)):
    emit(f"    const double* p{i};")
  emit(f"""
  }} {s}_data;
  static {s}_data {s}_ctx;
  static void {s}_read({s}_data* d, const double* x) {{
    for (int i = 0; i < {n}; ++i) d->x[i] = x[{s}_perm[i]];
  }}
""")

  params = [f"d->p{i}" for i in range(len(desc.param_names))]

  def call(oracle, inputs: list[str], outputs: list[str], count: str = "") -> str:
    args = [*inputs, *outputs, "d->w"]
    if ctx.oracle_options(oracle):
      args.append("d->solver_options")
    return f"double t = scaly_clock_s();\n  {ctx.raw_symbol_of(oracle)}({', '.join(args)});\n  d->t_fe += scaly_clock_s() - t;" + (
      f" ++d->{count};" if count else ""
    )

  for name, oracle, outputs, count in (
    ("base", desc.base, ["&d->obj", *(["d->g"] if m else [])], "n_f"),
    ("grad", desc.grad, ["d->grad"], "n_grad"),
    ("jac", desc.jac, ["d->j"], "n_jac"),
  ):
    if oracle is None:
      continue
    emit(f"static void {s}_{name}({s}_data* d) {{\n  {call(oracle, ['d->x', *params], outputs, count)}")
    if name == "base":
      emit("  ++d->n_g;")
    if name == "jac":
      emit(f"""  for (int i = 0; i < {len(pivot_indices)}; ++i) {{
      int entry = {s}_pivot[i], row = {s}_jr[entry];
      if (!isfinite(d->j[entry]) || (d->factor[row] != 0.0 && fabs(d->j[entry]*d->factor[row]+1.0) > 1e-12)) d->invalid = 1;
    }}""")
    emit("}")

  for name, expression in (("nx", f"{s}_nx[k]"), ("nu", f"{s}_nu[k]"), ("ng", "d->ng[k]"), ("ngi", "d->ngi[k]")):
    emit(f"static int {s}_get_{name}(int k, void* ud) {{ {s}_data* d = ({s}_data*)ud; return {expression}; }}")
  emit(f"""
  static int {s}_horizon(void* ud) {{ return {stages}; }}
  static int {s}_initial_x(double* x, int k, void* ud) {{
    {s}_data* d = ({s}_data*)ud;
    for (int i = 0; i < {s}_nx[k]; ++i) x[i] = d->initial[{s}_start[k]+i];
    return 0;
  }}
  static int {s}_initial_u(double* u, int k, void* ud) {{
    {s}_data* d = ({s}_data*)ud;
    for (int i = 0; i < {s}_nu[k]; ++i) u[i] = d->initial[{s}_start[k]+{s}_nx[k]+i];
    return 0;
  }}
  static int {s}_bounds(double* lower, double* upper, int k, void* ud) {{
    {s}_data* d = ({s}_data*)ud;
    for (int j = {s}_order_start[k]; j < {s}_order_start[k+1]; ++j) {{
      int i = {s}_order[j];
      if (d->kind[i] == 2) {{ lower[d->column[i]] = d->lower[i]; upper[d->column[i]] = d->upper[i]; }}
    }}
    return 0;
  }}
  static double {s}_value({s}_data* d, int i) {{ return i < {m} ? d->g[i] : d->x[i-{m}]; }}
  static void {s}_multipliers({s}_data* d, const double* lam) {{
    for (int i = 0; i < {m + n}; ++i)
      d->lam[i] = d->native[i] < 0 ? 0.0 : (d->kind[i] == 0 ? d->factor[i] : 1.0) * lam[d->native[i]];
  }}
  static int {s}_objective(double scale, const double* x, const double* p, const double* gp,
                          double* out, const struct FatropOcpCDims* dims, void* ud) {{
    {s}_data* d = ({s}_data*)ud; double begin = scaly_clock_s();
    {s}_read(d, x); {s}_base(d); *out = scale*d->obj;
    d->t_callback += scaly_clock_s()-begin; return 1;
  }}
  static int {s}_gradient(double scale, const double* x, const double* p, const double* gp,
                         double* out, const struct FatropOcpCDims* dims, void* ud) {{
    {s}_data* d = ({s}_data*)ud; double begin = scaly_clock_s();
    {s}_read(d, x); {s}_grad(d);
    for (int i = 0; i < {n}; ++i) out[{s}_perm[i]] = scale*d->grad[i];
    d->t_callback += scaly_clock_s()-begin; return 1;
  }}
  static int {s}_violation(const double* x, const double* p, const double* gp, double* out,
                          const struct FatropOcpCDims* dims, void* ud) {{
    {s}_data* d = ({s}_data*)ud; double begin = scaly_clock_s();
    {s}_read(d, x); {s}_base(d);
    for (int i = 0; i < {m + n}; ++i) if (d->native[i] >= 0) {{
      double value = {s}_value(d, i);
      out[d->native[i]] = d->kind[i] == 0 ? d->factor[i]*value : value - (d->kind[i] == 1 ? d->lower[i] : 0.0);
    }}
    d->t_callback += scaly_clock_s()-begin; return 1;
  }}
""")
  if stage is not None:
    lines.extend(render_callbacks(desc, ctx, call))
  else:
    assert jac is not None and hess is not None and desc.hess is not None
    emit(f"""
  static int {s}_jacobian(const double* x, const double* p, const double* gp,
                         struct blasfeo_dmat* dyn, struct blasfeo_dmat* eq, struct blasfeo_dmat* ineq,
                         const struct FatropOcpCDims* dims, void* ud) {{
    {s}_data* d = ({s}_data*)ud; double begin = scaly_clock_s();
    {s}_read(d, x); {s}_base(d); {s}_jac(d);
    for (int k = 0; k < {stages}; ++k) {{
      int width = {s}_nx[k]+{s}_nu[k];
      for (int j = 0; j < d->ng[k]; ++j) for (int i = 0; i <= width; ++i) BLASFEO_DMATEL(&eq[k], i, j) = 0.0;
      for (int j = 0; j < d->ngi[k]; ++j) for (int i = 0; i <= width; ++i) BLASFEO_DMATEL(&ineq[k], i, j) = 0.0;
      if (k < {stages - 1}) for (int j = 0; j < {s}_nx[k+1]; ++j)
        for (int i = 0; i <= width; ++i) BLASFEO_DMATEL(&dyn[k], i, j) = 0.0;
    }}
    for (int i = 0; i < {jac.nnz}; ++i) {{
      int row = {s}_jr[i], col = {s}_jc[i], k = {s}_rstage[row], kind = d->kind[row];
      if (kind < 0 || (kind == 0 && {s}_vstage[col] != k)) continue;
      struct blasfeo_dmat* block = kind == 0 ? dyn+k : kind == 1 ? eq+k : ineq+k;
      BLASFEO_DMATEL(block, {s}_vlocal[col], d->column[row]) = (kind == 0 ? d->factor[row] : 1.0)*d->j[i];
    }}
    for (int i = 0; i < {m + n}; ++i) if (d->kind[i] >= 0) {{
      int k = {s}_rstage[i], kind = d->kind[i], column = d->column[i];
      struct blasfeo_dmat* block = kind == 0 ? dyn+k : kind == 1 ? eq+k : ineq+k;
      if (i >= {m}) BLASFEO_DMATEL(block, {s}_vlocal[i-{m}], column) = 1.0;
      double value = {s}_value(d, i);
      BLASFEO_DMATEL(block, {s}_nx[k]+{s}_nu[k], column) = kind == 0 ? d->factor[i]*value : value-(kind == 1 ? d->lower[i] : 0.0);
    }}
    d->t_callback += scaly_clock_s()-begin; return 1;
  }}
  static int {s}_hessian(double scale, const double* x, const double* lam, const double* p, const double* gp,
                        struct blasfeo_dmat* out, const struct FatropOcpCDims* dims, void* ud) {{
    {s}_data* d = ({s}_data*)ud; double begin = scaly_clock_s();
    {s}_read(d, x); {s}_multipliers(d, lam); {s}_grad(d); {s}_jac(d);
    {call(desc.hess, ["d->x", *params, "&scale", *(["d->lam"] if m else [])], ["d->h"], "n_hess")}
    for (int k = 0; k < {stages}; ++k) {{
      int width = {s}_nx[k]+{s}_nu[k];
      for (int j = 0; j < width; ++j) for (int i = 0; i <= width; ++i) BLASFEO_DMATEL(&out[k], i, j) = 0.0;
    }}
    for (int i = 0; i < {hess.nnz}; ++i) {{
      int row = {s}_hr[i], col = {s}_hc[i], k = {s}_vstage[row];
      BLASFEO_DMATEL(&out[k], {s}_vlocal[row], {s}_vlocal[col]) = d->h[i];
      BLASFEO_DMATEL(&out[k], {s}_vlocal[col], {s}_vlocal[row]) = d->h[i];
    }}
    for (int i = 0; i < {n}; ++i) d->grad[i] = scale*d->grad[i] + d->lam[{m}+i];
    // Fatrop's stage gradient omits the incoming transition's -x_next term.
    for (int i = 0; i < {jac.nnz}; ++i) {{
      int row = {s}_jr[i], col = {s}_jc[i];
      if (d->kind[row] != 0 || {s}_vstage[col] == {s}_rstage[row]) d->grad[col] += d->j[i]*d->lam[row];
    }}
    for (int i = 0; i < {n}; ++i) {{
      int k = {s}_vstage[i];
      BLASFEO_DMATEL(&out[k], {s}_nx[k]+{s}_nu[k], {s}_vlocal[i]) = d->grad[i];
    }}
    d->t_callback += scaly_clock_s()-begin; return 1;
  }}
""")
  signature = [
    *(f"const double* in{i}" for i in range(len(desc.input_signature))),
    *(f"double* out{i}" for i in range(len(desc.output_signature))),
    "double* w",
    "const scaly_solver_option* const* solver_options",
  ]
  emit(f"""
  static void {ctx.raw_symbol}({", ".join(signature)}) {{
    double begin = scaly_clock_s();
    {s}_data* d = &{s}_ctx;
    memset(d, 0, sizeof(*d));
    {stats} = (scaly_solver_stats){{0}};
    {stats}.version = SCALY_SOLVER_STATS_VERSION;
    {stats}.status = SCALY_SOLVE_ERROR; {stats}.native_status = -1;
    d->w = w; d->solver_options = solver_options;
""")
  for i in range(len(desc.param_names)):
    emit(f"  d->p{i} = in{2 * nv + 2 + i};")
  for block, (size, offset) in enumerate(zip(sizes, offsets, strict=True)):
    emit(
      f"  for (int i = 0; i < {size}; ++i) {{ d->initial[{offset}+i] = d->x[{offset}+i] = out{block}[i] = in{block}[i]; out{nv + block}[i] = 0.0; }}"
    )
  emit(f"  for (int i = 0; i < {desc.n_eq}; ++i) out{2 * nv}[i] = 0.0;")
  emit(f"  for (int i = 0; i < {desc.n_ineq}; ++i) out{2 * nv + 1}[i] = 0.0;")
  bounds_outputs = [f"d->lower+{m}", f"d->upper+{m}"]
  if desc.n_ineq:
    bounds_outputs += [f"d->lower+{desc.n_eq}", f"d->upper+{desc.n_eq}"]
  emit(f"""
    {call(desc.bounds, params, bounds_outputs)}
    for (int i = 0; i < {m + n}; ++i) {{ d->kind[i] = -1; d->native[i] = -1; }}
    for (int i = 0; i < {m}; ++i) if ({s}_dyncol[i] >= 0) {{ d->kind[i] = 0; d->column[i] = {s}_dyncol[i]; }}
    int setup_ok = 1;
    for (int j = 0; j < {len(order)}; ++j) {{
      int i = {s}_order[j], k = {s}_rstage[i];
      double lower = d->lower[i], upper = d->upper[i];
      if (isnan(lower) || isnan(upper) || lower > upper || lower == INFINITY || upper == -INFINITY) setup_ok = 0;
      if (lower == -INFINITY && upper == INFINITY) continue;
      d->kind[i] = lower == upper ? 1 : 2;
      d->column[i] = d->kind[i] == 1 ? d->ng[k]++ : d->ngi[k]++;
    }}
    for (int k = 0; k < {stages}; ++k) if (d->ng[k] > {s}_nx[k]+{s}_nu[k]) setup_ok = 0;
    {f"{s}_jac(d);" if stage is None else f"for (int i = 0; i < {m}; ++i) if (d->kind[i] == 0) d->factor[i] = 1.0;"}
""")
  if stage is None:
    emit(f"""
    for (int i = 0; i < {len(pivot_indices)}; ++i) {{
      int entry = {s}_pivot[i], row = {s}_jr[entry];
      double coefficient = d->j[entry];
      if (!isfinite(coefficient) || coefficient == 0.0) setup_ok = 0;
      else {{
        d->factor[row] = -1.0/coefficient;
        if (!isfinite(d->factor[row])) setup_ok = 0;
      }}
    }}
""")
  emit(f"""
    setup_ok = setup_ok && !d->invalid;
    struct FatropOcpCInterface interface = {{0}};
    interface.get_nx = {s}_get_nx; interface.get_nu = {s}_get_nu;
    interface.get_ng = {s}_get_ng; interface.get_ng_ineq = {s}_get_ngi;
    interface.get_horizon_length = {s}_horizon; interface.get_bounds = {s}_bounds;
    interface.get_initial_xk = {s}_initial_x; interface.get_initial_uk = {s}_initial_u;
    interface.full_eval_lag_hess = {s}_hessian; interface.full_eval_constr_jac = {s}_jacobian;
    interface.full_eval_contr_viol = {s}_violation; interface.full_eval_obj_grad = {s}_gradient;
    interface.full_eval_obj = {s}_objective; interface.user_data = d;
    struct FatropOcpCSolver* solver = setup_ok ? fatrop_ocp_c_create(&interface, NULL, NULL) : NULL;
    setup_ok = setup_ok && solver != NULL;
    for (const scaly_solver_option* option = solver_options[{ctx.options_index}]; setup_ok && option->name; ++option) {{
      int type = fatrop_ocp_c_option_type(option->name);
      if (type == 0 && option->kind == 1 && isfinite(option->number) &&
          option->number > 0.0)
        setup_ok = fatrop_ocp_c_set_option_double(solver, option->name, option->number) == 0;
      else if (type == 1 && option->kind == 0 && option->integer >= 0 && option->integer <= INT32_MAX)
        setup_ok = fatrop_ocp_c_set_option_int(solver, option->name, (int)option->integer) == 0;
      else if (type == 2 && option->kind == 0 && (option->integer == 0 || option->integer == 1))
        setup_ok = fatrop_ocp_c_set_option_bool(solver, option->name, (int)option->integer) == 0;
      else setup_ok = 0;
    }}
    double t_solver = 0.0;
    if (setup_ok) {{
      const struct FatropOcpCDims* dims = fatrop_ocp_c_get_dims(solver);
      for (int i = 0; i < {m + n}; ++i) if (d->kind[i] >= 0) {{
        int k = {s}_rstage[i];
        d->native[i] = d->column[i] + (d->kind[i] == 0 ? dims->dyn_eq_offs[k] : d->kind[i] == 1 ? dims->g_offs[k] : dims->g_ineq_offs[k]);
      }}
      double solve_begin = scaly_clock_s(), callback_before = d->t_callback;
      int result = fatrop_ocp_c_solve(solver);
      t_solver = fmax(0.0, scaly_clock_s()-solve_begin-(d->t_callback-callback_before));
      const struct FatropOcpCStats* native = fatrop_ocp_c_get_stats(solver);
      {stats}.iter = native->iterations_count;
      {stats}.native_status = result == -1 ? -1 : native->return_flag;
      // Fatrop 1.2.0 exposes IpSolverReturnFlag as an integer, without a C enum.
      switch ({stats}.native_status) {{
        case 0: {stats}.status = SCALY_SOLVE_OK; break;
        case 1: {stats}.status = SCALY_SOLVE_MAX_ITER; break;
        case 2: {stats}.status = SCALY_SOLVE_ACCEPTABLE; break;
        case 3: {stats}.status = SCALY_SOLVE_PRIMAL_INFEASIBLE; break;
        case 4: case 5: {stats}.status = SCALY_SOLVE_NUMERICS; break;
        default: {stats}.status = SCALY_SOLVE_ERROR; break;
      }}
      {s}_read(d, fatrop_ocp_c_get_primal(solver)->pa);
      {s}_multipliers(d, fatrop_ocp_c_get_dual(solver)->pa);
      if (d->invalid) {stats}.status = SCALY_SOLVE_ERROR;
""")
  for block, (size, offset) in enumerate(zip(sizes, offsets, strict=True)):
    emit(f"  for (int i = 0; i < {size}; ++i) {{ out{block}[i] = d->x[{offset}+i]; out{nv + block}[i] = d->lam[{m + offset}+i]; }}")
  emit(f"""
      for (int i = 0; i < {desc.n_eq}; ++i) out{2 * nv}[i] = d->lam[i];
      for (int i = 0; i < {desc.n_ineq}; ++i) out{2 * nv + 1}[i] = d->lam[{desc.n_eq}+i];
    }}
    if (solver) fatrop_ocp_c_destroy(solver);
    {s}_base(d);
    {stats}.obj = d->obj;
    for (int i = 0; i < {m + n}; ++i) {{
      double value = {s}_value(d, i);
      {stats}.primal_viol = fmax({stats}.primal_viol, fmax(d->lower[i]-value, value-d->upper[i]));
      if ({stats}.status != SCALY_SOLVE_ERROR && !isfinite(value)) {stats}.status = SCALY_SOLVE_NUMERICS;
    }}
    if ({stats}.status != SCALY_SOLVE_ERROR && !isfinite(d->obj)) {stats}.status = SCALY_SOLVE_NUMERICS;
    {stats}.n_eval_f = d->n_f; {stats}.n_eval_g = d->n_g;
    {stats}.n_eval_grad_f = d->n_grad; {stats}.n_eval_jac_g = d->n_jac; {stats}.n_eval_h = d->n_hess;
    {stats}.t_fe = d->t_fe; {stats}.t_solver = t_solver;
    {stats}.t_total = scaly_clock_s()-begin;
    {stats}.t_glue = {stats}.t_total-d->t_fe-t_solver;
  }}
""")
  return lines
