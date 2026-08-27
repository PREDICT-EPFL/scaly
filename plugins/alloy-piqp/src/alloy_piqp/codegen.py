"""PIQP C wrapper template — the plugin half of alloy's solver codegen contract.

Called by ``alloy.codegen.solver.render_solver_raw`` through the backend's
``render_wrapper`` hook. Emits a ``static void <ctx.raw_symbol>(...)`` that
calls the generated QP-data oracle, drives ``piqp_c`` (dense or sparse
interface), and fills ``ctx.stats_symbol``. Contract: ``docs/dev/solver_plugins.md``.
"""

from __future__ import annotations

import math

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from alloy.codegen.solver import SolverWrapperCtx
  from alloy.function import Function
  from alloy.solvers.model import SolverDescriptor
  from alloy.ir.types import SparsityType


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


def render_wrapper(fun: Function, ctx: SolverWrapperCtx) -> list[str]:
  desc: SolverDescriptor = fun.descriptor
  symbol = ctx.symbol
  raw = ctx.raw_symbol
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
  # PIQP does not consume warm starts yet.
  for input_index in range(param_start):
    lines.append(f"  (void)in{input_index};")
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
  oracle_raw = ctx.raw_symbol_of(oracle)
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
    # 3. PIQP's C wrapper consumes contiguous row-major dense matrices.
    lines.append(f"  for (int j = 0; j < {n}; ++j) for (int i = 0; i < {n}; ++i) Pcol[i + j * {n}] = P_buf[i * {n} + j];")
    if p:
      lines.append(f"  for (int i = 0; i < {p * n}; ++i) Acol[i] = A_buf[i];")
    if m:
      lines.append(f"  for (int i = 0; i < {m * n}; ++i) Gcol[i] = G_buf[i];")

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
  for block, (size, offset) in enumerate(zip(var_sizes, var_offsets, strict=True)):
    lines.append(f"  for (int i = 0; i < {size}; ++i) out{block}[i] = res->x[{offset} + i];")
    lines.append(f"  for (int i = 0; i < {size}; ++i) out{nv + block}[i] = res->z_bu[{offset} + i] - res->z_bl[{offset} + i];")
  if p:
    lines.append(f"  for (int i = 0; i < {p}; ++i) out{eq_output}[i] = res->y[i];")
  if m:
    lines.append(f"  for (int i = 0; i < {m}; ++i) out{ineq_output}[i] = res->z_u[i] - res->z_l[i];")

  lines.append("  int32_t stats_status;")
  lines.append("  switch (res->info.status) {")
  lines.append("    case PIQP_SOLVED: stats_status = ALLOY_SOLVE_OK; break;")
  lines.append("    case PIQP_MAX_ITER_REACHED: stats_status = ALLOY_SOLVE_MAX_ITER; break;")
  lines.append("    case PIQP_PRIMAL_INFEASIBLE: stats_status = ALLOY_SOLVE_PRIMAL_INFEASIBLE; break;")
  lines.append("    case PIQP_DUAL_INFEASIBLE: stats_status = ALLOY_SOLVE_DUAL_INFEASIBLE; break;")
  lines.append("    case PIQP_NUMERICS: stats_status = ALLOY_SOLVE_NUMERICS; break;")
  lines.append("    default: stats_status = ALLOY_SOLVE_ERROR; break;")
  lines.append("  }")
  lines.append(f"  {ctx.stats_symbol}.version = ALLOY_SOLVER_STATS_VERSION;")
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
  lines.append("  double stats_t_total = alloy_clock_s() - stats_t0;")
  lines.append(f"  {ctx.stats_symbol}.t_total = stats_t_total;")
  lines.append(f"  {ctx.stats_symbol}.t_glue = stats_t_total - stats_t_fe - stats_t_solver;")

  lines.append("}")
  return lines
