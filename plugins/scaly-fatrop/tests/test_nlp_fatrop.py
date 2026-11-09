"""Fatrop's whole-problem wrapper agrees with reference NLP solves."""

import numpy as np
import pytest

import scaly as sc
from scaly.solvers.registry import require_backend


def test_backend_registration():
  assert require_backend("fatrop", "nlp").hess_triangle == "lower"


def _problem():
  @sc.problem(vars=sc.group(sc.arg("a", 3), sc.arg("b", 3)), params=sc.arg("p", 4))
  def problem(v, p):
    z = sc.concat(v)
    return sc.ProblemSpec(
      minimize=sc.sumsqr(z - sc.const(np.array([0.3, 1.0, 0.7, -0.6, 1.1, 0.2]))) + 0.1 * (z[0] * z[1]) ** 2,
      eq=(z[0] - p[0], z[2] - z[0] ** 2 - z[1] - 0.2 * z[0] * z[1], z[4] - z[2] ** 2 - z[3]),
      ineq=(sc.bounded(z[1] ** 2, hi=0.25), sc.bounded(z[4], lo=p[2], hi=p[3])),
      lb=(sc.const(np.full(3, -np.inf)), sc.stack([-0.3, -np.inf, p[1]])),
      ub=(sc.const(np.full(3, np.inf)), sc.stack([np.inf, np.inf, p[1]])),
    )

  return problem


@pytest.mark.solver("fatrop")
@pytest.mark.solver("ipopt")
@pytest.mark.parametrize("last_bounds", [(0.0, 2.0), (0.4, 0.4), (-np.inf, np.inf)])
def test_nonlinear_stage_blocks_bounds_duals_and_stats(last_bounds):
  problem = _problem()
  fatrop = sc.solver(problem, "fatrop", options={"tol": 1e-11, "constr_viol_tol": 1e-12})
  ipopt = sc.solver(problem, "ipopt", options={"tol": 1e-11})
  start = (np.full(3, 0.2), np.full(3, 0.2))
  for initial in (0.3, 0.4):
    params = np.array([initial, 0.1, *last_bounds])
    actual, reference = fatrop(params, x0=start), ipopt(params, x0=start)
    for actual_group, reference_group in zip(actual, reference, strict=True):
      np.testing.assert_allclose(
        np.concatenate(actual_group) if isinstance(actual_group, tuple) else actual_group,
        np.concatenate(reference_group) if isinstance(reference_group, tuple) else reference_group,
        rtol=0.0,
        atol=1e-8,
      )
    stats = fatrop.stats()
    assert stats.status == sc.ScalySolveStatus.OK and stats.native_status == 0 and stats.iter > 0
    assert stats.primal_viol <= 1.1e-8
    assert stats.obj == pytest.approx(ipopt.stats().obj, abs=1e-8)
    assert min(stats.n_eval_f, stats.n_eval_grad_f, stats.n_eval_g, stats.n_eval_jac_g, stats.n_eval_h) > 0
    assert min(stats.t_total, stats.t_fe, stats.t_solver, stats.t_glue) >= 0.0
    assert stats.t_total == pytest.approx(stats.t_fe + stats.t_solver + stats.t_glue, abs=1e-12)
    assert stats.t_qp == stats.t_globalization == stats.step_inf == stats.alpha == stats.merit_penalty == stats.backtracks == stats.qp_iter == 0


@pytest.mark.solver("fatrop")
def test_iteration_limit_and_runtime_options():
  problem = _problem()
  solver = sc.solver(problem, "fatrop", options={"max_iter": 0})
  solver(np.array([0.3, 0.1, 0.0, 2.0]))
  assert solver.stats().status == sc.ScalySolveStatus.MAX_ITER
  assert solver.stats().native_status == 1


@pytest.mark.solver("fatrop")
def test_nested_solver_preserves_workspace_and_parameters():
  problem = _problem()
  solve = sc.solver(problem, "fatrop", options={"tol": 1e-11})

  @sc.function(sc.arg("p", 4))
  def outer(p):
    return sc.concat(solve(p)[0])

  params = np.array([0.3, 0.1, 0.0, 2.0])
  np.testing.assert_allclose(outer(params), np.concatenate(solve(params)[0]), atol=1e-10)


@pytest.mark.parametrize(
  "options,error",
  [
    ({"unknown": 1}, ValueError),
    ({"tol": -1.0}, ValueError),
    ({"tol": np.inf}, ValueError),
    ({"max_iter": 1.5}, TypeError),
    ({"max_iter": -1}, ValueError),
    ({"linsol_it_ref": "yes"}, TypeError),
  ],
)
def test_invalid_options(options, error):
  with pytest.raises(error):
    sc.solver(_problem(), "fatrop", options=options)


@pytest.mark.solver("fatrop")
def test_bound_classification_changes_between_calls_and_errors_reset_stats():
  solve = sc.solver(_problem(), "fatrop", options={"tol": 1e-11})
  for lo, hi in ((0.0, 2.0), (0.4, 0.4), (-np.inf, np.inf)):
    result = solve(np.array([0.3, 0.1, lo, hi]))
    assert solve.stats().status == sc.ScalySolveStatus.OK
    if lo == hi:
      assert result[0][1][1] == pytest.approx(lo, abs=1e-10)
    if lo == -np.inf:
      assert result[3][1] == 0.0
  result = solve(np.array([0.3, 0.1, 2.0, 0.0]))
  stats = solve.stats()
  assert stats.status == sc.ScalySolveStatus.ERROR and stats.native_status == -1
  assert stats.iter == stats.n_eval_h == stats.t_solver == 0
  assert np.isfinite(np.concatenate(result[0])).all()
  assert stats.version == sc.SCALY_SOLVER_STATS_VERSION


@pytest.mark.solver("fatrop")
@pytest.mark.solver("ipopt")
def test_changing_state_sizes_and_zero_controls():
  @sc.problem(vars=sc.arg("z", 6))
  def varying(z):
    return sc.ProblemSpec(minimize=sc.sumsqr(z - 1.0), eq=(z[:1] - 0.2, z[2:5] - z[0] ** 2 - z[1], z[5:6] - z[2:5].sum()))

  for backend in ("fatrop", "ipopt"):
    solve = sc.solver(varying, backend, options={"tol": 1e-11})
    result = solve(())
    assert solve.stats().to_solver_status().ok
    if backend == "fatrop":
      actual = result
    else:
      for a, b in zip(actual, result, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-8, rtol=0)


@pytest.mark.solver("fatrop")
def test_empty_hessian_and_scaled_transition():
  @sc.problem(vars=sc.arg("z", 3), params=sc.arg("coefficient", ()))
  def linear(z, coefficient):
    return sc.ProblemSpec(
      minimize=-z[2], eq=(z[0] - 1.0, coefficient * (z[2] - z[0] - z[1])), lb=sc.const([0.0, 0.0, 0.0]), ub=sc.const([2.0, 1.0, 3.0])
    )

  solve = sc.solver(linear, "fatrop", options={"tol": 1e-10})
  for coefficient in (1.0, -1.0, 2.0):
    np.testing.assert_allclose(solve(np.array(coefficient))[0], [1.0, 1.0, 2.0], atol=2e-8, rtol=0)
    assert solve.stats().status == sc.ScalySolveStatus.OK
  solve(np.array(0.0))
  assert solve.stats().status == sc.ScalySolveStatus.ERROR
