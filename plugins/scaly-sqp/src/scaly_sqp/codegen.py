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

from jinja2 import Environment, PackageLoader, StrictUndefined

from scaly.opt.external.graph import solver_descriptor

if TYPE_CHECKING:
  from scaly.opt.external.wrapper import SolverWrapperCtx
  from scaly.function import ConcreteFunction


_TEMPLATE = Environment(
  loader=PackageLoader("scaly_sqp", "."),
  undefined=StrictUndefined,
  autoescape=False,
  trim_blocks=True,
  lstrip_blocks=True,
).get_template("sqp.c.jinja")


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
    raise NotImplementedError(f"scaly-sqp options are not supported: {unknown}")
  return out


def _positive_int(options: dict[str, Any], name: str, default: int) -> int:
  value = options.get(name, default)
  if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
    raise ValueError(f"scaly-sqp {name} must be a positive integer, got {value!r}")
  return value


def _nonnegative_int(options: dict[str, Any], name: str, default: int) -> int:
  value = options.get(name, default)
  if isinstance(value, bool) or not isinstance(value, int) or value < 0:
    raise ValueError(f"scaly-sqp {name} must be a non-negative integer, got {value!r}")
  return value


def _positive_float(options: dict[str, Any], name: str, default: float) -> float:
  value = options.get(name, default)
  if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
    raise ValueError(f"scaly-sqp {name} must be finite and positive, got {value!r}")
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


def render_wrapper(fun: ConcreteFunction, ctx: SolverWrapperCtx) -> list[str]:
  desc = solver_descriptor(fun)
  base, grad, jac, hess, bounds = desc.base, desc.grad, desc.jac, desc.hess, desc.bounds
  assert base is not None and grad is not None and hess is not None and bounds is not None
  assert jac is not None or not (desc.n_eq + desc.n_ineq)
  assert desc.jac_sparsity is not None and desc.hess_sparsity is not None
  n, nh, ng, m = desc.n, desc.n_eq, desc.n_ineq, desc.n_eq + desc.n_ineq
  nv = desc.n_var_blocks
  if nv < 1:
    raise ValueError("typed NLP descriptors need at least one variable block")
  var_sizes = [math.prod(shape) for _, shape in desc.input_signature[:nv]]
  var_offsets = [sum(var_sizes[:i]) for i in range(nv)]
  eq_input, ineq_input, param_start = 2 * nv, 2 * nv + 1, 2 * nv + 2
  eq_output, ineq_output = 2 * nv, 2 * nv + 1
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
    raise ValueError(f"scaly-sqp line_search_beta must be below 1, got {beta!r}")
  globalization = opts.get("globalization", "filter")
  if globalization not in ("filter", "l1"):
    raise ValueError(f"scaly-sqp globalization must be 'filter' or 'l1', got {globalization!r}")
  if globalization == "filter" and "watchdog" in opts:
    raise ValueError("scaly-sqp watchdog applies only to globalization='l1'")
  hessian_mode = opts.get("hessian", "exact")
  if hessian_mode not in ("exact", "objective"):
    raise ValueError(f"scaly-sqp hessian must be 'exact' or 'objective', got {hessian_mode!r}")
  qp_mode = opts.get("qp", "sparse")
  if qp_mode not in ("sparse", "dense"):
    raise ValueError(f"scaly-sqp qp must be 'sparse' or 'dense', got {qp_mode!r}")
  trace = opts.get("trace", False)
  if not isinstance(trace, bool):
    raise ValueError(f"scaly-sqp trace must be a bool, got {trace!r}")
  trace_prefix = f"[scaly-sqp {re.sub(r'\W', '_', fun.name)}]"

  # QP patterns, fixed across SQP iterations: P is the canonical upper-triangle
  # view of the handed Lagrangian Hessian pattern, unioned with the full
  # diagonal (regularization writes every diagonal entry) and, when there are
  # equalities, with the pattern of the constraint-normal term A.T @ A. A and
  # G are the equality and inequality row blocks of the constraint Jacobian.
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

  param_args = [f"in{param_start + i}" for i in range(len(desc.param_names))]
  base_args = lambda x, f, g: ", ".join([x, *param_args, f, *((g,) if m else ()), "w"])  # noqa: E731
  one_out_args = lambda x, out: ", ".join([x, *param_args, out, "w"])  # noqa: E731
  hess_args = lambda x, lf, lg, out: ", ".join([x, *param_args, lf, *((lg,) if m else ()), out, "w"])  # noqa: E731
  bounds_args = ", ".join([*param_args, "xlb", "xub", *(("gl", "gu") if ng else ()), "w"])
  signature = [
    *(f"const double* in{i}" for i in range(len(desc.input_signature))),
    *(f"double* out{i}" for i in range(len(desc.output_signature))),
    "double* w",
  ]
  base_raw, grad_raw = ctx.raw_symbol_of(base), ctx.raw_symbol_of(grad)
  jac_raw = ctx.raw_symbol_of(jac) if jac is not None else ""
  hess_raw, bounds_raw = ctx.raw_symbol_of(hess), ctx.raw_symbol_of(bounds)
  return _TEMPLATE.render(
    name=fun.name,
    raw_symbol=ctx.raw_symbol,
    stats_symbol=ctx.stats_symbol,
    signature=", ".join(signature),
    n=n,
    nh=nh,
    ng=ng,
    m=m,
    nv=nv,
    var_blocks=list(zip(var_sizes, var_offsets, strict=True)),
    eq_input=eq_input,
    ineq_input=ineq_input,
    eq_output=eq_output,
    ineq_output=ineq_output,
    nnz_j=nnz_j,
    nnz_h=nnz_h,
    nnz_p=nnz_p,
    nnz_l=nnz_l,
    n_normal=len(normal_terms),
    jrows=list(jrows),
    jcols=list(jcols),
    p_ptr=p_ptr,
    p_rows=p_rows,
    p_cols=p_cols,
    p_src=p_src,
    p_diag=p_diag,
    a_ptr=a_ptr,
    a_rows=a_rows,
    a_src=a_src,
    g_ptr=g_ptr,
    g_rows=g_rows,
    g_src=g_src,
    ldl_parent=ldl_parent,
    ldl_ptr=ldl_ptr,
    normal_slot=normal_slot,
    normal_a=normal_a,
    normal_b=normal_b,
    base_raw=base_raw,
    grad_raw=grad_raw,
    jac_raw=jac_raw,
    hess_raw=hess_raw,
    bounds_raw=bounds_raw,
    base_args=base_args("x", "&f", "g"),
    trial_args=base_args("trial", "&trial_f", "trial_g"),
    grad_args=one_out_args("x", "grad_buf"),
    jac_args=one_out_args("x", "jac_buf"),
    hess_args=hess_args("x", "&(double){1.0}", "lam_g" if hessian_mode == "exact" else "zero_lam", "hess_buf"),
    objective_hess_args=hess_args("x", "&(double){1.0}", "zero_lam", "hess_buf"),
    bounds_args=bounds_args,
    qp_mode=qp_mode,
    globalization=globalization,
    hessian_mode=hessian_mode,
    watchdog=watchdog,
    trace=trace,
    trace_prefix=trace_prefix,
    max_iter=max_iter,
    tol=tol,
    dual_tol=dual_tol,
    beta=beta,
    merit_offset=merit_offset,
    regularization=regularization,
    qp_max_iter=qp_max_iter,
    qp_tol=qp_tol,
  ).split("\n")
