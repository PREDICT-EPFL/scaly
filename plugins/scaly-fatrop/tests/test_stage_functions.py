"""Mapped stage derivatives and Fatrop solves agree with whole-problem oracles."""

from dataclasses import replace

import numpy as np
import pytest

import scaly as sc
from tests.solvers.test_stage_oracles import _ocp


@pytest.mark.solver("fatrop")
@pytest.mark.parametrize("N,globals_", [(1, True), (4, True), (4, False)])
def test_builder_paths_take_the_same_iterations_and_return_the_same_duals(N, globals_):
  ocp = _ocp(N, globals_=globals_)
  problem = ocp.stage_problem()
  options = {"tol": 1e-10}
  mapped = sc.solver(problem, "fatrop", name="mapped", options=options)
  whole = sc.solver(replace(problem, stage_functions=None), "fatrop", name="whole", options=options)
  for initial in (np.array([0.2, -0.1]), np.array([0.1, 0.05])):
    params = (initial, np.array(0.8), np.zeros((N + 1, 2))) if globals_ else (initial, np.array(1.0), np.array(0.8), np.zeros((N + 1, 2)))
    guess = ocp.pack(initial, 0.0, *((1.0,) if globals_ else ()), stage_form=True)
    actual, expected = mapped(params, x0=guess), whole(params, x0=guess)
    assert mapped.stats().status == whole.stats().status == sc.ScalySolveStatus.OK
    assert mapped.stats().iter == whole.stats().iter
    for a, b in zip(actual, expected, strict=True):
      for left, right in zip(a, b, strict=True) if isinstance(a, tuple) else ((a, b),):
        np.testing.assert_allclose(left, right, atol=1e-8, rtol=0.0)
    stats = mapped.stats()
    assert min(stats.t_fe, stats.t_solver, stats.t_glue) >= 0
    assert stats.t_total == pytest.approx(stats.t_fe + stats.t_solver + stats.t_glue, abs=1e-12)


@pytest.mark.solver("fatrop")
@pytest.mark.solver("ipopt")
def test_free_final_time_solves_through_stage_functions():
  from tests.solvers.test_ocp import _free_time

  ocp = _free_time()
  staged = ocp.stage_problem()
  mapped = sc.solver(staged, "fatrop", options={"tol": 1e-10})
  whole = sc.solver(replace(staged, stage_functions=None), "fatrop", name="free_time_whole", options={"tol": 1e-10})
  reference = sc.solver(ocp.problem(), "ipopt", options={"tol": 1e-10})
  initial = np.zeros(2)
  guess = ocp.pack(initial, 0.0, 3.0, stage_form=True)
  actual = ocp.unpack(mapped(initial, x0=guess)[0], stage_form=True)
  expected = ocp.unpack(reference(initial, x0=ocp.pack(initial, 0.0, 3.0))[0])
  whole(initial, x0=guess)
  assert mapped.stats().status == whole.stats().status == reference.stats().status == sc.ScalySolveStatus.OK
  assert mapped.stats().iter == whole.stats().iter
  for a, b in zip(actual, expected, strict=True):
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
  assert 2.0 <= float(actual[-1]) < 2.5


@pytest.mark.solver("fatrop")
@pytest.mark.parametrize("terminal_only,nu", [(True, 1), (False, 1), (True, 0)])
def test_missing_costs_constraints_and_zero_controls(terminal_only, nu):
  @sc.function(sc.arg("x", 1), sc.arg("u", nu))
  def step(x, u):
    return 0.8 * x + u.sum()

  @sc.function(sc.arg("x", 1), sc.arg("u", nu))
  def cost(x, u):
    return sc.sumsqr(x) + sc.sumsqr(u)

  @sc.function(sc.arg("x", 1))
  def end(x):
    return sc.sumsqr(x)

  ocp = sc.DiscreteOCP(N=2, dynamics=step, stage_cost=None if terminal_only else cost, terminal_cost=end if terminal_only else None)
  problem = ocp.stage_problem()
  mapped = sc.solver(problem, "fatrop", name="missing_mapped")
  whole = sc.solver(replace(problem, stage_functions=None), "fatrop", name="missing_whole")
  initial = np.array([0.3])
  a, b = mapped(initial), whole(initial)
  assert mapped.stats().status == whole.stats().status == sc.ScalySolveStatus.OK
  assert mapped.stats().iter == whole.stats().iter
  for left, right in zip(a[0], b[0], strict=True):
    np.testing.assert_allclose(left, right, atol=1e-9)


@pytest.mark.solver("fatrop")
def test_nested_stage_solver_preserves_workspace_and_parameters():
  ocp = _ocp()
  solve = sc.solver(ocp.stage_problem(), "fatrop")
  params = (np.array([0.2, -0.1]), np.array(0.8), np.zeros((4, 2)))

  @sc.function(sc.arg("initial", 2), sc.arg("gain", ()), sc.arg("ref", (4, 2)))
  def outer(initial, gain, ref):
    z, x_N = solve((initial, gain, ref), x0=tuple(sc.const(v) for v in ocp.pack(0.0, 0.0, 1.0, stage_form=True)))[0]
    return sc.concat([z.vec(), x_N])

  z, x_N = solve(params, x0=ocp.pack(0.0, 0.0, 1.0, stage_form=True))[0]
  np.testing.assert_allclose(outer(*params), np.concatenate([z.ravel(), x_N]), atol=1e-9)
