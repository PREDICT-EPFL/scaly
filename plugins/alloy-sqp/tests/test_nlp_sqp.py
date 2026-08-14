from __future__ import annotations

import re

import numpy as np
import pytest

import alloy as al
from alloy.codegen.c import render_c_source


def _problem(*, hessian: str = "exact", max_iter: int = 30, trace: bool = False, **options) -> al.SolverFunction:
  x = al.sym("x", 2)
  target = al.sym("target", 2, diff=False)
  suffix = "".join(f"_{key}_{value}" for key, value in sorted(options.items()))
  return al.nlp(
    x=x,
    p=target,
    f=0.5 * al.dot(x - target, x - target),
    h_eq=al.stack([x[0] + x[1] - 1.0]),
    g_ineq=al.stack([x[0] - x[1]]),
    l_ineq=np.array([-0.5]),
    u_ineq=np.array([0.5]),
    x_lb=np.zeros(2),
    x_ub=np.ones(2),
    solver="sqp",
    name=f"test_sqp_{hessian}_{max_iter}{'_trace' if trace else ''}{suffix}",
    options={"hessian": hessian, "max_iter": max_iter, "tol": 1e-7, "qp_tol": 1e-8, "trace": trace, **options},
  )


@pytest.mark.parametrize("interface", ["sparse", "dense"])
def test_sqp_generated_wrapper_reuses_one_qp_workspace(interface: str) -> None:
  source = render_c_source(_problem() if interface == "sparse" else _problem(qp="dense"))
  assert source.count(f"piqp_setup_{interface}") == 1
  assert source.count(f"piqp_update_{interface}") == 1
  assert source.count("piqp_cleanup") == 1
  assert f"if (qp == NULL) piqp_setup_{interface}" in source
  assert f"else piqp_update_{interface}" in source
  assert "settings.preconditioner_reuse_on_update = 0" in source
  assert "settings.eps_duality_gap_abs = 1e-08" in source
  assert "settings.eps_duality_gap_rel = 1e-08" in source


@pytest.mark.solver("sqp")
@pytest.mark.parametrize("hessian", ["exact", "objective"])
def test_sqp_hessian_modes_solve_constrained_quadratic(hessian: str) -> None:
  solver = _problem(hessian=hessian)
  out = solver(np.zeros(2), np.zeros(1), np.zeros(1), np.zeros(2), np.array([0.2, 0.8]))
  assert solver.last_status is not None and solver.last_status.ok
  np.testing.assert_allclose(out["x"], [0.25, 0.75], atol=2e-6)
  np.testing.assert_allclose(out["h_eq"], 0.0, atol=1e-7)
  assert out["g_ineq"][0] == pytest.approx(-0.5, abs=2e-6)


@pytest.mark.solver("sqp")
def test_sqp_stats_split_is_additive() -> None:
  solver = _problem()
  solver(np.zeros(2), np.zeros(1), np.zeros(1), np.zeros(2), np.array([0.2, 0.8]))
  stats = solver.last_stats
  assert stats is not None and stats.status == al.AlloySolveStatus.OK
  assert stats.t_fe >= 0.0 and stats.t_qp > 0.0 and stats.t_globalization >= 0.0
  assert stats.t_solver == 0.0
  assert stats.t_total == pytest.approx(stats.t_fe + stats.t_solver + stats.t_qp + stats.t_globalization + stats.t_glue, rel=1e-10, abs=1e-12)
  assert stats.n_eval_h > 0 and stats.n_eval_jac_g > 0


@pytest.mark.solver("sqp")
def test_sqp_diagnostics_stats_fields() -> None:
  solver = _problem()
  solver(np.zeros(2), np.zeros(1), np.zeros(1), np.zeros(2), np.array([0.2, 0.8]))
  stats = solver.last_stats
  assert stats is not None and stats.status == al.AlloySolveStatus.OK
  assert stats.version == al.ALLOY_SOLVER_STATS_VERSION
  assert stats.qp_iter > 0 and stats.backtracks >= 0
  assert 0.0 <= stats.primal_viol <= 1e-7  # converged: violation within tol
  assert stats.step_inf >= 0.0
  assert stats.alpha > 0.0  # at least one accepted line-search step
  assert stats.merit_penalty == 0.0  # the default filter has no merit penalty


@pytest.mark.solver("sqp")
def test_sqp_rejects_nonfinite_warm_starts_before_the_kkt_check() -> None:
  solver = _problem()
  valid = [np.zeros(2), np.zeros(1), np.zeros(1), np.zeros(2), np.array([0.2, 0.8])]
  for slot in range(4):
    args = [value.copy() for value in valid]
    args[slot].reshape(-1)[0] = np.nan
    solver(*args)
    assert solver.last_stats is not None and solver.last_stats.status == al.AlloySolveStatus.NUMERICS


def test_sqp_trace_is_off_by_default() -> None:
  source = render_c_source(_problem())
  assert "fprintf" not in source and "stdio.h" not in source


def test_sqp_trace_prefix_sanitizes_hostile_names() -> None:
  # the prefix lands inside C format strings: %, quotes, and escapes in the name must not survive
  x = al.sym("x", 2)
  target = al.sym("target", 2, diff=False)
  fun = al.nlp(
    x=x,
    p=target,
    f=0.5 * al.dot(x - target, x - target),
    x_lb=np.zeros(2),
    x_ub=np.ones(2),
    solver="sqp",
    name='pct%s "quote',
    options={"trace": True},
  )
  source = render_c_source(fun)
  assert "[alloy-sqp pct_s__quote]" in source
  # the raw name may appear in comments, but never inside a format string
  assert all("pct%s" not in line and '"quote' not in line for line in source.splitlines() if "fprintf" in line)


@pytest.mark.solver("sqp")
def test_sqp_trace_prints_per_iteration_lines_to_stderr(capfd: pytest.CaptureFixture[str]) -> None:
  solver = _problem(trace=True)
  solver(np.zeros(2), np.zeros(1), np.zeros(1), np.zeros(2), np.array([0.2, 0.8]))
  err = capfd.readouterr().err
  assert "[alloy-sqp test_sqp_exact_30_trace] iter=1 qp_status=" in err
  assert "globalization=filter" in err and "alpha=" in err and "penalty=" in err and "backtracks=" in err


def test_sqp_generated_globalization_mechanics_are_explicit() -> None:
  filter_source = render_c_source(_problem())
  assert "static double filter_f[20], filter_v[20]" in filter_source
  assert "trial_f > filter_f[j] && trial_violation > filter_v[j]" in filter_source
  assert "filter_f[keep] = filter_f[j]" in filter_source
  assert "trial_f > filter_f[j] - 1e-5 * filter_v[j]" in filter_source
  assert "x[i] += alpha * step[i]" in filter_source
  assert "lam_g[i] += alpha * (qp_lam_g[i] - lam_g[i])" in filter_source

  l1_source = render_c_source(_problem(globalization="l1", watchdog=2))
  assert "trial_merit <= merit + alpha * 0.25 * dmerit" in l1_source
  for state in ("checkpoint_x", "checkpoint_lam_g", "checkpoint_step", "checkpoint_qp_lam_g"):
    assert state in l1_source
  assert "x[i] = checkpoint_x[i]" in l1_source


def _c_table(source: str, name: str) -> list[int]:
  match = re.search(rf"\b{name}\[\d+\] = {{ ([^}}]*) }};", source)
  assert match is not None, f"{name} table missing from generated source"
  return [int(value) for value in match.group(1).split(",")]


def _coupled_problem(name: str, *, solver: str = "sqp", **options) -> al.SolverFunction:
  """Scrambled Jacobian/Hessian patterns: the objective couples (0,3) and (1,2),
  the constraints touch non-adjacent variables, so a wrong COO-to-CSC
  permutation lands values in the wrong rows instead of cancelling out."""
  x = al.sym("x", 4)
  return al.nlp(
    x=x,
    f=0.5 * ((x[0] - 1.0) ** 2 + 2.0 * (x[1] - 2.0) ** 2 + 3.0 * (x[2] + 1.0) ** 2 + 4.0 * (x[3] - 0.5) ** 2) + 0.3 * x[0] * x[3] + 0.2 * x[1] * x[2],
    h_eq=al.stack([x[0] + 2.0 * x[2] - 1.0, x[1] - x[3]]),
    g_ineq=al.stack([x[0] * x[3], x[1] ** 2 + x[2]]),
    l_ineq=np.array([-1.0, -np.inf]),
    u_ineq=np.array([1.0, 3.0]),
    x_lb=np.full(4, -2.0),
    x_ub=np.full(4, 2.0),
    solver=solver,
    name=name,
    options={"tol": 1e-9, "qp_tol": 1e-10, **options} if solver == "sqp" else {"tol": 1e-10},
  )


def test_sqp_bakes_csc_patterns_from_the_descriptor_sparsity() -> None:
  solver = _coupled_problem("sqp_csc_pattern")
  desc = solver.descriptor
  assert desc.jac_sparsity is not None and desc.hess_sparsity is not None
  jac_sparsity, hess_sparsity = desc.jac_sparsity, desc.hess_sparsity
  source = render_c_source(solver)
  n, nh = desc.n, desc.n_eq

  p_ptr, p_row = _c_table(source, "P_p"), _c_table(source, "P_i")
  assert p_ptr[0] == 0 and p_ptr[-1] == len(p_row) and len(p_ptr) == n + 1
  p_col = [j for j in range(n) for _ in range(p_ptr[j + 1] - p_ptr[j])]
  # P is the upper triangle of the Hessian pattern, unioned with the full
  # diagonal and with the constraint-normal term's A.T @ A pattern
  eq_cols: dict[int, list[int]] = {}
  for r, c in zip(jac_sparsity.rows, jac_sparsity.cols):
    if r < nh:
      eq_cols.setdefault(r, []).append(c)
  expected = {(min(r, c), max(r, c)) for r, c in zip(hess_sparsity.rows, hess_sparsity.cols)} | {(i, i) for i in range(n)}
  expected |= {(min(a, b), max(a, b)) for cols in eq_cols.values() for a in cols for b in cols}
  assert set(zip(p_row, p_col)) == expected
  assert all(r <= c for r, c in zip(p_row, p_col))
  assert [_c_table(source, "P_diag")[i] for i in range(n)] == [list(zip(p_row, p_col)).index((i, i)) for i in range(n)]

  # A and G are the equality and inequality row blocks of the Jacobian, and
  # their value slots point back at the oracle's own COO buffer
  jac = list(zip(jac_sparsity.rows, jac_sparsity.cols))
  a_src, a_row = _c_table(source, "A_src"), _c_table(source, "A_i")
  assert [jac[k][0] for k in a_src] == a_row
  assert sorted(jac[k] for k in a_src) == sorted(entry for entry in jac if entry[0] < nh)
  g_src, g_row = _c_table(source, "G_src"), _c_table(source, "G_i")
  assert [jac[k][0] - nh for k in g_src] == g_row
  assert sorted(jac[k] for k in g_src) == sorted(entry for entry in jac if entry[0] >= nh)


@pytest.mark.solver("sqp")
@pytest.mark.solver("ipopt")
def test_sqp_sparse_assembly_reaches_the_same_solution_as_ipopt() -> None:
  """The sparse assembly's only external reference: a scrambled COO-to-CSC
  permutation still produces a self-consistent QP, so pattern assertions alone
  cannot catch it — an independent solver on the same NLP can."""
  start = np.array([0.5, 1.0, -0.5, 0.25])
  sqp = _coupled_problem("sqp_vs_ipopt_sparse")
  sqp_out = sqp(start, np.zeros(2), np.zeros(2), np.zeros(4))
  assert sqp.last_status is not None and sqp.last_status.ok

  ipopt = _coupled_problem("sqp_vs_ipopt_ipopt", solver="ipopt")
  ipopt_out = ipopt(start, np.zeros(2), np.zeros(2), np.zeros(4))
  assert ipopt.last_status is not None and ipopt.last_status.ok
  np.testing.assert_allclose(sqp_out["x"], ipopt_out["x"], rtol=1e-6, atol=1e-7)
  assert sqp_out["f"] == pytest.approx(float(ipopt_out["f"]), abs=1e-8)
  np.testing.assert_allclose(sqp_out["h_eq"], 0.0, atol=1e-9)


@pytest.mark.solver("sqp")
def test_sqp_sparse_and_dense_qp_interfaces_agree() -> None:
  """The dense interface assembles the same QP from the same patterns, so it
  is a differential check on the sparse assembly. Benchmarks never select it —
  it is slower at every canonical point (race N=40: 20.5 ms against 2.0 ms)."""
  start = np.array([0.5, 1.0, -0.5, 0.25])
  results = {}
  for interface in ("sparse", "dense"):
    solver = _coupled_problem(f"sqp_interface_{interface}") if interface == "sparse" else _coupled_problem("sqp_interface_dense", qp="dense")
    out = solver(start, np.zeros(2), np.zeros(2), np.zeros(4))
    assert solver.last_status is not None and solver.last_status.ok
    results[interface] = (out, solver.last_stats)
  sparse_out, sparse_stats = results["sparse"]
  dense_out, dense_stats = results["dense"]
  for key in ("x", "f", "h_eq", "g_ineq", "lam_eq", "lam_ineq", "lam_box"):
    np.testing.assert_allclose(sparse_out[key], dense_out[key], rtol=1e-6, atol=1e-8, err_msg=key)
  assert sparse_stats is not None and dense_stats is not None
  assert sparse_stats.iter == dense_stats.iter


def test_sqp_convexifies_with_a_modified_ldl_factorization() -> None:
  source = render_c_source(_problem())
  # the modification is a diagonal shift, so it goes straight onto P's diagonal
  assert "double target = fabs(dk); if (target < min_reg) target = min_reg;" in source
  assert "if (target > dk) { double e = target - dk; P_x[P_diag[k]] += e;" in source
  # a negative pivot becomes its magnitude, never a clamp to min_reg: clamping
  # runs the shift away (9.4e19 -> 2.0e46 -> 4.8e89 on canonical race)
  assert "dk = min_reg;" not in source
  # the dense Cholesky probe and its J^T J normal-curvature branch are gone
  assert "JTJ" not in source and "probe_shift" not in source and "factor_ok" not in source


def test_sqp_ldl_symbolic_matches_a_dense_reference() -> None:
  from alloy_sqp.codegen import _ldl_symbolic

  # arrow pattern: column 0 couples to every row, so eliminating it fills the
  # whole trailing block — parent[i] = i + 1 and 4 + 3 + 2 + 1 factor entries
  n = 5
  entries = sorted([(0, 0)] + [(i, i) for i in range(1, n)] + [(0, c) for c in range(1, n)], key=lambda e: (e[1], e[0]))
  col_ptr = [0] * (n + 1)
  for _, c in entries:
    col_ptr[c + 1] += 1
  for c in range(n):
    col_ptr[c + 1] += col_ptr[c]
  parent, l_ptr = _ldl_symbolic(n, col_ptr, [r for r, _ in entries])
  assert parent == [1, 2, 3, 4, -1]
  assert l_ptr == [0, 4, 7, 9, 10, 10]

  # a diagonal Hessian needs no factor entries at all
  diagonal = [0] * (n + 1)
  for c in range(n):
    diagonal[c + 1] = c + 1
  parent, l_ptr = _ldl_symbolic(n, diagonal, list(range(n)))
  assert parent == [-1] * n and l_ptr == [0] * (n + 1)


@pytest.mark.solver("sqp")
def test_sqp_solves_an_indefinite_coupled_hessian() -> None:
  # H = [[1, 2], [2, 1]] has eigenvalues -1 and 3, so PIQP cannot take it
  # unregularized; the modified factorization shifts it positive definite
  x = al.sym("x", 2)
  solver = al.nlp(
    x=x,
    f=0.5 * x[0] ** 2 + 0.5 * x[1] ** 2 + 2.0 * x[0] * x[1],
    x_lb=np.full(2, -1.0),
    x_ub=np.full(2, 1.0),
    solver="sqp",
    name="sqp_indefinite_coupled",
    options={"tol": 1e-8, "max_iter": 40},
  )
  out = solver(np.array([0.3, -0.2]), np.zeros(0), np.zeros(0), np.zeros(2))
  assert solver.last_status is not None and solver.last_status.ok
  assert out["f"] == pytest.approx(-1.0, abs=1e-7)
  np.testing.assert_allclose(np.abs(out["x"]), 1.0, atol=1e-7)
  assert out["x"][0] * out["x"][1] < 0.0


@pytest.mark.solver("sqp")
@pytest.mark.parametrize("globalization", ["filter", "l1"])
def test_sqp_globalizations_backtrack_before_accepting(globalization: str) -> None:
  x = al.sym("x", 1)
  solver = al.nlp(
    x=x,
    f=0.25 * x[0] ** 4 - 0.5 * x[0] ** 2,
    solver="sqp",
    name=f"sqp_{globalization}_backtrack",
    options={"globalization": globalization, "regularization": 0.3, "tol": 1e-8, "max_iter": 30},
  )
  out = solver(np.array([0.5]), np.zeros(0), np.zeros(0), np.zeros(1))
  stats = solver.last_stats
  assert stats is not None and stats.status == al.AlloySolveStatus.OK
  assert stats.backtracks > 0 and 0.0 < stats.alpha <= 1.0
  np.testing.assert_allclose(np.abs(out["x"]), 1.0, atol=2e-5)
  if globalization == "l1":
    assert stats.merit_penalty == pytest.approx(10.0)


@pytest.mark.solver("sqp")
def test_sqp_fails_when_filter_has_no_acceptable_trial() -> None:
  x = al.sym("x", 1)
  solver = al.nlp(
    x=x,
    f=0.25 * x[0] ** 4 - 0.5 * x[0] ** 2,
    solver="sqp",
    name="sqp_filter_rejects_every_trial",
    options={"line_search_beta": 1e-5, "regularization": 0.03, "max_iter": 2},
  )
  solver(np.array([0.5]), np.zeros(0), np.zeros(0), np.zeros(1))
  stats = solver.last_stats
  assert stats is not None and stats.status == al.AlloySolveStatus.NUMERICS
  assert stats.alpha == 0.0 and stats.backtracks == 1


@pytest.mark.solver("sqp")
def test_sqp_watchdog_falls_back_to_checkpoint_line_search() -> None:
  x = al.sym("x", 1)
  solver = al.nlp(
    x=x,
    f=0.25 * x[0] ** 4 - 0.5 * x[0] ** 2,
    solver="sqp",
    name="sqp_watchdog_fallback",
    options={"globalization": "l1", "watchdog": 1, "regularization": 0.3, "tol": 1e-8, "max_iter": 30},
  )
  out = solver(np.array([0.5]), np.zeros(0), np.zeros(0), np.zeros(1))
  stats = solver.last_stats
  assert stats is not None and stats.status == al.AlloySolveStatus.OK
  assert stats.backtracks > 0 and 0.0 < stats.alpha <= 1.0
  np.testing.assert_allclose(np.abs(out["x"]), 1.0, atol=2e-5)


@pytest.mark.solver("sqp")
def test_sqp_kkt_terminates_at_initial_bound_optima_with_signed_multipliers() -> None:
  x = al.sym("x", 1)
  upper = al.nlp(x=x, f=-x[0], x_lb=np.array([0.0]), x_ub=np.array([1.0]), solver="sqp", name="sqp_upper_kkt")
  lower = al.nlp(x=x, f=x[0], x_lb=np.array([0.0]), x_ub=np.array([1.0]), solver="sqp", name="sqp_lower_kkt")
  for solver, x0, lam in ((upper, 1.0, 1.0), (lower, 0.0, -1.0)):
    solver(np.array([x0]), np.zeros(0), np.zeros(0), np.array([lam]))
    stats = solver.last_stats
    assert stats is not None and stats.status == al.AlloySolveStatus.OK
    assert stats.iter == 0 and stats.qp_iter == 0 and stats.alpha == 0.0


@pytest.mark.solver("sqp")
def test_sqp_bound_complementarity_uses_the_slack_selected_by_multiplier_sign() -> None:
  x = al.sym("x", 1)
  solver = al.nlp(
    x=x,
    f=x[0],
    x_lb=np.array([0.0]),
    x_ub=np.array([1.0]),
    solver="sqp",
    name="sqp_wrong_bound_sign",
    options={"max_iter": 1},
  )
  solver(np.array([1.0]), np.zeros(0), np.zeros(0), np.array([-1.0]))
  stats = solver.last_stats
  assert stats is not None and stats.iter == 1


@pytest.mark.solver("sqp")
def test_sqp_inequality_complementarity_uses_signed_two_sided_multiplier() -> None:
  x = al.sym("x", 1)
  solver = al.nlp(
    x=x,
    f=-x[0],
    g_ineq=al.stack([x[0]]),
    l_ineq=np.array([0.0]),
    u_ineq=np.array([1.0]),
    solver="sqp",
    name="sqp_upper_ineq_kkt",
  )
  solver(np.array([1.0]), np.zeros(0), np.array([1.0]), np.zeros(1))
  stats = solver.last_stats
  assert stats is not None and stats.status == al.AlloySolveStatus.OK and stats.iter == 0


@pytest.mark.solver("sqp")
def test_sqp_nested_in_host_function() -> None:
  solver = _problem()
  target = al.sym("target", 2, diff=False)
  x = solver.call(
    x0=al.const(np.zeros(2)),
    lam_eq0=al.const(np.zeros(1)),
    lam_ineq0=al.const(np.zeros(1)),
    lam_box0=al.const(np.zeros(2)),
    target=target,
  )[0]
  host = al.Function("nested_sqp_host", [target], [al.dot(x, x)], ["target"], ["norm"])
  np.testing.assert_allclose(host(np.array([0.2, 0.8])), 0.625, atol=3e-6)


@pytest.mark.solver("sqp")
def test_sqp_continues_with_best_iterate_after_qp_max_iter() -> None:
  # the iteration cap makes PIQP stop just short of its tolerance with a polished
  # iterate; like laopt, the SQP must use it and still converge
  solver = _problem(qp_max_iter=4)
  out = solver(np.zeros(2), np.zeros(1), np.zeros(1), np.zeros(2), np.array([0.2, 0.8]))
  stats = solver.last_stats
  assert stats is not None and stats.status == al.AlloySolveStatus.OK
  assert stats.native_status == -1  # PIQP_MAX_ITER_REACHED on the last QP
  np.testing.assert_allclose(out["x"], [0.25, 0.75], atol=2e-6)


@pytest.mark.solver("sqp")
def test_sqp_max_iter_status() -> None:
  x = al.sym("x", 1)
  solver = al.nlp(
    x=x,
    f=(x[0] - 2.0) ** 4,
    h_eq=al.stack([x[0] * x[0] - 1.0]),
    solver="sqp",
    name="sqp_max_iter",
    options={"max_iter": 1, "tol": 1e-12},
  )
  solver(np.array([0.5]), np.zeros(1), np.zeros(0), np.zeros(1))
  assert solver.last_stats is not None and solver.last_stats.status == al.AlloySolveStatus.MAX_ITER


@pytest.mark.solver("sqp")
def test_sqp_enforces_coupled_constraints_and_reports_box_multiplier() -> None:
  x = al.sym("x", 2)
  solver = al.nlp(
    x=x,
    f=0.5 * al.dot(x - al.const(np.array([2.0, 0.0])), x - al.const(np.array([2.0, 0.0]))),
    h_eq=al.stack([x[0] + x[1] - 1.0]),
    g_ineq=al.stack([x[0] - x[1]]),
    l_ineq=np.array([-np.inf]),
    u_ineq=np.array([0.0]),
    x_lb=np.zeros(2),
    x_ub=np.ones(2),
    solver="sqp",
    name="sqp_coupled_constraints",
    options={"tol": 1e-8},
  )
  out = solver(np.array([2.0, -1.0]), np.zeros(1), np.zeros(1), np.zeros(2))
  assert solver.last_status is not None and solver.last_status.ok
  np.testing.assert_allclose(out["x"], [0.5, 0.5], atol=2e-7)
  np.testing.assert_allclose(out["h_eq"], 0.0, atol=1e-8)
  assert out["g_ineq"][0] <= 1e-8

  y = al.sym("y", 1)
  bound = al.nlp(x=y, f=0.5 * (y[0] - 2.0) ** 2, x_lb=np.array([0.0]), x_ub=np.array([1.0]), solver="sqp", name="sqp_bound", options={"qp_tol": 1e-8})
  bound_out = bound(np.array([2.0]), np.zeros(0), np.zeros(0), np.zeros(1))
  assert bound.last_status is not None and bound.last_status.ok
  np.testing.assert_allclose(bound_out["x"], [1.0], atol=1e-8)
  assert bound_out["lam_box"][0] > 0.0


@pytest.mark.solver("sqp")
def test_sqp_does_not_accept_unconverged_unconstrained_iterate() -> None:
  x = al.sym("x", 1)
  solver = al.nlp(x=x, f=(x[0] - 2.0) ** 4, solver="sqp", name="sqp_unconverged", options={"max_iter": 1, "tol": 1e-12})
  solver(np.array([0.0]), np.zeros(0), np.zeros(0), np.zeros(1))
  assert solver.last_stats is not None and solver.last_stats.status == al.AlloySolveStatus.MAX_ITER


@pytest.mark.parametrize(
  "options,match",
  [
    ({"line_search_beta": 1.0}, "line_search_beta"),
    ({"regularization": 0.0}, "regularization"),
    ({"max_iter": True}, "max_iter"),
    ({"dual_tol": 0.0}, "dual_tol"),
    ({"globalization": "full"}, "globalization"),
    ({"watchdog": -1}, "watchdog"),
    ({"watchdog": 0}, "only to globalization='l1'"),
    ({"qp": "banded"}, "qp must be 'sparse' or 'dense'"),
  ],
)
def test_sqp_rejects_invalid_options(options: dict[str, str | int | float | bool], match: str) -> None:
  x = al.sym("x", 1)
  solver = al.nlp(x=x, f=x[0] ** 2, solver="sqp", name=f"sqp_invalid_{match}", options=options)
  with pytest.raises(ValueError, match=match):
    solver(np.zeros(1), np.zeros(0), np.zeros(0), np.zeros(1))


@pytest.mark.solver("sqp")
def test_same_sqp_wrapper_accepts_casadi_codegen_oracles() -> None:
  import casadi as ca

  from alloy_sqp.casadi import build_casadi_external_sqp

  x, p = ca.MX.sym("x", 2), ca.MX.sym("p", 2)
  f = 0.5 * ca.dot(x - p, x - p)
  g = ca.vertcat(x[0] + x[1] - 1.0)
  lam_f, lam_g = ca.MX.sym("lam_f"), ca.MX.sym("lam_g", 1)
  base = ca.Function("external_fixture_base", [x, p], [f, g])
  grad = ca.Function("external_fixture_grad", [x, p], [ca.gradient(f, x)])
  jac = ca.Function("external_fixture_jac", [x, p], [ca.jacobian(g, x)])
  hess = ca.Function("external_fixture_hess", [x, lam_f, lam_g, p], [ca.hessian(lam_f * f + ca.dot(lam_g, g), x)[0]])
  solver = build_casadi_external_sqp(
    name="external_fixture_sqp",
    base=base,
    grad=grad,
    jac=jac,
    hess=hess,
    n_eq=1,
    n_ineq=0,
    x_lb=np.full(2, -np.inf),
    x_ub=np.full(2, np.inf),
    l_ineq=np.zeros(0),
    u_ineq=np.zeros(0),
  )
  out = solver(np.zeros(2), np.zeros(1), np.zeros(0), np.zeros(2), np.array([0.2, 0.8]))
  assert solver.last_status is not None and solver.last_status.ok
  np.testing.assert_allclose(out["x"], [0.2, 0.8], atol=2e-6)


def test_casadi_external_sqp_validates_oracle_and_bound_shapes() -> None:
  import casadi as ca

  from alloy_sqp.casadi import build_casadi_external_sqp

  x, p = ca.MX.sym("shape_x", 2), ca.MX.sym("shape_p", 1)
  f = ca.dot(x, x)
  lam_f = ca.MX.sym("shape_lam_f")
  base = ca.Function("shape_base", [x, p], [f])
  grad = ca.Function("shape_grad", [x, p], [ca.gradient(f, x)])
  bad_grad = ca.Function("shape_bad_grad", [x, p], [f])
  hess = ca.Function("shape_hess", [x, lam_f, p], [ca.hessian(lam_f * f, x)[0]])
  with pytest.raises(ValueError, match="shape_bad_grad"):
    build_casadi_external_sqp(
      name="shape_external_sqp",
      base=base,
      grad=bad_grad,
      jac=None,
      hess=hess,
      n_eq=0,
      n_ineq=0,
      x_lb=np.full(2, -np.inf),
      x_ub=np.full(2, np.inf),
      l_ineq=np.zeros(0),
      u_ineq=np.zeros(0),
    )
  with pytest.raises(ValueError, match="x_lb"):
    build_casadi_external_sqp(
      name="shape_external_sqp",
      base=base,
      grad=grad,
      jac=None,
      hess=hess,
      n_eq=0,
      n_ineq=0,
      x_lb=np.zeros(1),
      x_ub=np.full(2, np.inf),
      l_ineq=np.zeros(0),
      u_ineq=np.zeros(0),
    )


@pytest.mark.solver("sqp")
def test_casadi_external_sqp_supports_unconstrained_problem_without_jacobian() -> None:
  import casadi as ca

  from alloy_sqp.casadi import build_casadi_external_sqp

  x, p = ca.MX.sym("free_x", 1), ca.MX.sym("free_p", 1)
  f = 0.5 * (x[0] - p[0]) ** 2
  lam_f = ca.MX.sym("free_lam_f")
  solver = build_casadi_external_sqp(
    name="free_external_sqp",
    base=ca.Function("free_base", [x, p], [f]),
    grad=ca.Function("free_grad", [x, p], [ca.gradient(f, x)]),
    jac=None,
    hess=ca.Function("free_hess", [x, lam_f, p], [ca.hessian(lam_f * f, x)[0]]),
    n_eq=0,
    n_ineq=0,
    x_lb=np.array([-np.inf]),
    x_ub=np.array([np.inf]),
    l_ineq=np.zeros(0),
    u_ineq=np.zeros(0),
    options={"qp_tol": 1e-8},
  )
  out = solver(np.zeros(1), np.zeros(0), np.zeros(0), np.zeros(1), np.array([0.75]))
  assert solver.last_status is not None and solver.last_status.ok
  np.testing.assert_allclose(out["x"], [0.75], atol=1e-6)
