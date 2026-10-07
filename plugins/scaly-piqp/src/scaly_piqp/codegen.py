"""PIQP C wrapper template — the plugin half of scaly's solver codegen contract.

Called by ``scaly.codegen.solver.render_solver_raw`` through the backend's
``render_wrapper`` hook. Emits a ``static void <ctx.raw_symbol>(...)`` that
calls the generated QP-data oracle, drives ``piqp_c`` (dense or sparse
interface), and fills ``ctx.stats_symbol``. Contract: ``docs/dev/solver_plugins.md``.
"""

from __future__ import annotations

import math

from . import _SETTINGS, _INTEGER_SETTINGS

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from scaly.codegen.solver import SolverWrapperCtx
  from scaly.function.concrete import ConcreteFunction
  from scaly.solvers.model import SolverDescriptor
  from scaly.ir.types import SparsityPattern


def _csc_tables(name: str, sp: SparsityPattern | None) -> list[str]:
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


def render_wrapper(fun: ConcreteFunction, ctx: SolverWrapperCtx) -> list[str]:
  desc: SolverDescriptor = fun.descriptor
  symbol = ctx.symbol
  raw = ctx.raw_symbol
  n, p, m = desc.n, desc.n_eq, desc.n_ineq
  oracle = desc.oracle
  assert oracle is not None, "PIQP solver descriptor must carry an oracle Function"
  sparse = dict(desc.compile_options)["sparse"]
  interface = "sparse" if sparse else "dense"
  nnz_P = desc.P_sparsity.nnz if desc.P_sparsity is not None else n * n
  nnz_A = desc.A_sparsity.nnz if desc.A_sparsity is not None else p * n
  nnz_G = desc.G_sparsity.nnz if desc.G_sparsity is not None else m * n

  param_count = len(desc.param_names)
  nv = desc.n_var_blocks
  if nv < 1:
    raise ValueError("typed QP descriptors need at least one variable block")
  var_sizes = [math.prod(shape) for _, shape in desc.input_signature[:nv]]
  var_offsets = [sum(var_sizes[:i]) for i in range(nv)]
  eq_output, ineq_output, param_start = 2 * nv, 2 * nv + 1, 2 * nv + 2
  oracle_param_args = [f"in{param_start + i}" for i in range(param_count)]

  # Oracle output ordering (set by qp.py): P, c, [A_eq, b_eq], [G_ineq, l_ineq, u_ineq], x_lb, x_ub.
  # Sparse wrappers get compact CSC-ordered value buffers for P/A/G.
  oracle_outs: list[tuple[str, int]] = []  # (buffer C name, size)
  oracle_outs.append(("P_buf", nnz_P))
  oracle_outs.append(("c_buf", n))
  if p:
    oracle_outs.append(("A_buf", nnz_A))
    oracle_outs.append(("b_buf", p))
  if m:
    oracle_outs.append(("G_buf", nnz_G))
    oracle_outs.append(("lineq_buf", m))
    oracle_outs.append(("uineq_buf", m))
  oracle_outs.append(("xlb_buf", n))
  oracle_outs.append(("xub_buf", n))

  # Parameters for the raw signature.
  c_inputs = [f"const double* in{i}" for i in range(len(desc.input_signature))]
  c_outputs = [f"double* out{i}" for i in range(len(desc.output_signature))]
  params = [*c_inputs, *c_outputs, "double* w", "const scaly_solver_option* const* solver_options"]

  lines: list[str] = []
  lines.append(f"// PIQP solver wrapper for {fun.name} (n={n}, p={p}, m={m}, nnz P/A/G = {nnz_P}/{nnz_A}/{nnz_G}).")
  lines.append(f"static void {raw}({', '.join(params)}) {{")
  # PIQP does not consume warm starts yet.
  for input_index in range(param_start):
    lines.append(f"  (void)in{input_index};")
  lines.append(f"  const scaly_solver_option* options = solver_options[{ctx.options_index}];")
  lines.append("  double stats_t0 = scaly_clock_s();")

  for buf, size in oracle_outs:
    lines.append(f"  static double {buf}[{size}];")

  # 2. Call the oracle. The oracle's input order is the param list; its
  # output order matches oracle_outs above.
  oracle_raw = ctx.raw_symbol_of(oracle)
  oracle_call = ", ".join([*oracle_param_args, *(name for name, _ in oracle_outs), "w", *ctx.oracle_options(oracle)])
  lines.append("  double fe_t0 = scaly_clock_s();")
  lines.append(f"  {oracle_raw}({oracle_call});")
  # Core oracles use IEEE infinities for open sides; PIQP uses its finite sentinel.
  if m:
    lines.append(
      f"  for (int i = 0; i < {m}; ++i) {{ if (isinf(lineq_buf[i]) && lineq_buf[i] < 0.0) lineq_buf[i] = -PIQP_INF; if (isinf(uineq_buf[i]) && uineq_buf[i] > 0.0) uineq_buf[i] = PIQP_INF; }}"
    )
  lines.append(
    f"  for (int i = 0; i < {n}; ++i) {{ if (isinf(xlb_buf[i]) && xlb_buf[i] < 0.0) xlb_buf[i] = -PIQP_INF; if (isinf(xub_buf[i]) && xub_buf[i] > 0.0) xub_buf[i] = PIQP_INF; }}"
  )
  lines.append("  double stats_t_fe = scaly_clock_s() - fe_t0;")

  for matrix, pattern, rows, cols, nnz in (
    ("P", desc.P_sparsity, n, n, nnz_P),
    ("A", desc.A_sparsity, p, n, nnz_A),
    ("G", desc.G_sparsity, m, n, nnz_G),
  ):
    if not sparse or not rows:
      continue
    assert pattern is not None
    lines += _csc_tables(matrix, pattern)
    lines.append(f"  static piqp_csc {matrix}_csc = {{ {rows}, {cols}, {nnz}, {matrix}_p, {matrix}_i, {matrix}_buf }};")

  lines.append(f"  static piqp_workspace* {symbol}_ws = NULL;")
  lines.append("  static piqp_settings previous_settings;")
  lines.append("  piqp_settings settings;")
  lines.append("  double solver_t0 = scaly_clock_s();")
  lines.append(f"  piqp_set_default_settings_{interface}(&settings);")
  lines.append("  static const struct { const char* name; size_t offset; int kind; } fields[] = {")
  for key in sorted(_SETTINGS):
    kind = 2 if key == "kkt_solver" else int(key in _INTEGER_SETTINGS)
    lines.append(f'    {{ "{key}", offsetof(piqp_settings, {key}), {kind} }},')
  lines += [
    "  };",
    "  int changed = 0;",
    "  for (size_t k = 0; k < sizeof fields / sizeof *fields; ++k) {",
    "    char* field = (char*)&settings + fields[k].offset;",
    "    const char* previous = (const char*)&previous_settings + fields[k].offset;",
    "    for (const scaly_solver_option* option = options; option->name; ++option) {",
    "      if (strcmp(option->name, fields[k].name)) continue;",
    "      if (fields[k].kind == 1) *(piqp_int*)field = (piqp_int)option->integer;",
    "      else if (fields[k].kind == 2) *(piqp_kkt_solver*)field = (piqp_kkt_solver)option->integer;",
    "      else *(piqp_float*)field = option->kind == 0 ? (piqp_float)option->integer : option->number;",
    "    }",
    "    if (fields[k].kind == 1) changed |= *(const piqp_int*)previous != *(piqp_int*)field;",
    "    else if (fields[k].kind == 2) changed |= *(const piqp_kkt_solver*)previous != *(piqp_kkt_solver*)field;",
    "    else changed |= *(const piqp_float*)previous != *(piqp_float*)field;",
    "  }",
  ]
  lines.append(f"  if ({symbol}_ws && changed) {{ piqp_cleanup({symbol}_ws); {symbol}_ws = NULL; }}")
  lines.append("  previous_settings = settings;")
  P = "&P_csc" if sparse else "P_buf"
  A = ("&A_csc" if sparse else "A_buf") if p else "NULL"
  G = ("&G_csc" if sparse else "G_buf") if m else "NULL"
  args = f"{P}, c_buf, {A}, {'b_buf' if p else 'NULL'}, {G}, {'lineq_buf' if m else 'NULL'}, {'uineq_buf' if m else 'NULL'}, xlb_buf, xub_buf"
  lines.append(f"  piqp_data_{interface} data = {{ {n}, {p}, {m}, {args} }};")
  lines.append(f"  if (!{symbol}_ws) piqp_setup_{interface}(&{symbol}_ws, &data, &settings);")
  lines.append(f"  else piqp_update_{interface}({symbol}_ws, {args});")

  # 5. Solve and read results.
  lines.append(f"  piqp_solve({symbol}_ws);")
  lines.append("  double stats_t_solver = scaly_clock_s() - solver_t0;")
  lines.append(f"  piqp_result* res = {symbol}_ws->result;")
  for block, (size, offset) in enumerate(zip(var_sizes, var_offsets, strict=True)):
    lines.append(f"  for (int i = 0; i < {size}; ++i) out{block}[i] = res->x[{offset} + i];")
    lines.append(f"  for (int i = 0; i < {size}; ++i) out{nv + block}[i] = res->z_bu[{offset} + i] - res->z_bl[{offset} + i];")
  if p:
    lines.append(f"  for (int i = 0; i < {p}; ++i) out{eq_output}[i] = res->y[i];")
  if m:
    lines.append(f"  for (int i = 0; i < {m}; ++i) out{ineq_output}[i] = res->z_u[i] - res->z_l[i];")

  lines.append("  int32_t stats_status;")
  lines.append("  switch (res->info.status) {")
  lines.append("    case PIQP_SOLVED: stats_status = SCALY_SOLVE_OK; break;")
  lines.append("    case PIQP_MAX_ITER_REACHED: stats_status = SCALY_SOLVE_MAX_ITER; break;")
  lines.append("    case PIQP_PRIMAL_INFEASIBLE: stats_status = SCALY_SOLVE_PRIMAL_INFEASIBLE; break;")
  lines.append("    case PIQP_DUAL_INFEASIBLE: stats_status = SCALY_SOLVE_DUAL_INFEASIBLE; break;")
  lines.append("    case PIQP_NUMERICS: stats_status = SCALY_SOLVE_NUMERICS; break;")
  lines.append("    default: stats_status = SCALY_SOLVE_ERROR; break;")
  lines.append("  }")
  lines.append(f"  {ctx.stats_symbol}.version = SCALY_SOLVER_STATS_VERSION;")
  lines.append(f"  {ctx.stats_symbol}.status = stats_status;")
  lines.append(f"  {ctx.stats_symbol}.native_status = (int32_t)res->info.status;")
  lines.append(f"  {ctx.stats_symbol}.iter = (int32_t)res->info.iter;")
  lines.append(f"  {ctx.stats_symbol}.obj = res->info.primal_obj;")
  lines.append(f"  {ctx.stats_symbol}.t_fe = stats_t_fe;")
  lines.append(f"  {ctx.stats_symbol}.t_solver = 0.0;")
  lines.append(f"  {ctx.stats_symbol}.t_qp = stats_t_solver;")
  lines.append(f"  {ctx.stats_symbol}.t_globalization = 0.0;")
  lines.append(f"  {ctx.stats_symbol}.n_eval_f = 1;")
  lines.append(f"  {ctx.stats_symbol}.n_eval_grad_f = 0;")
  lines.append(f"  {ctx.stats_symbol}.n_eval_g = 0;")
  lines.append(f"  {ctx.stats_symbol}.n_eval_jac_g = 0;")
  lines.append(f"  {ctx.stats_symbol}.n_eval_h = 0;")
  lines.append(f"  {ctx.stats_symbol}._pad0 = 0;")
  # v3 diagnostics: PIQP has a native primal residual and iteration count;
  # line-search/merit fields do not apply to a direct QP solve.
  lines.append(f"  {ctx.stats_symbol}.primal_viol = res->info.primal_res;")
  lines.append(f"  {ctx.stats_symbol}.step_inf = 0.0;")
  lines.append(f"  {ctx.stats_symbol}.alpha = 0.0;")
  lines.append(f"  {ctx.stats_symbol}.merit_penalty = 0.0;")
  lines.append(f"  {ctx.stats_symbol}.backtracks = 0;")
  lines.append(f"  {ctx.stats_symbol}.qp_iter = (int32_t)res->info.iter;")
  lines.append("  double stats_t_total = scaly_clock_s() - stats_t0;")
  lines.append(f"  {ctx.stats_symbol}.t_total = stats_t_total;")
  lines.append(f"  {ctx.stats_symbol}.t_glue = stats_t_total - stats_t_fe - stats_t_solver;")

  lines.append("}")
  return lines
