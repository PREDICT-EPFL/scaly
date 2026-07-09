"""Smoke tests: every problem builds and runs; QP solvers agree with IPOPT."""
import numpy as np
import pytest

from fastbench.core.registry import get_problem, get_solver, list_problems
from fastbench.core.simulator import run_closed_loop
from fastbench.core.metrics import episode_metrics, closed_loop_cost


@pytest.mark.parametrize("pname", list_problems())
def test_ipopt_runs_each_problem(pname):
    problem = get_problem(pname)
    adapter = get_solver("casadi_ipopt")
    assert adapter.available()
    adapter.build(problem)
    x0 = problem.x0_nominal()
    log = run_closed_loop(problem, adapter, x0, np.random.default_rng(0))
    m = episode_metrics(problem, log)
    assert np.isfinite(m["closed_loop_cost"])
    assert 0.0 <= m["step_success_rate"] <= 1.0
    assert len(log.U) == problem.meta.n_sim


@pytest.mark.parametrize("solver", ["osqp", "piqp", "proxqp"])
def test_qp_solvers_match_ipopt_on_di(solver):
    """Each QP backend reproduces the IPOPT cost on the double integrator."""
    problem = get_problem("double_integrator")
    a_ref = get_solver("casadi_ipopt"); a_ref.build(problem)
    ref = closed_loop_cost(problem, run_closed_loop(
        problem, a_ref, problem.x0_nominal(), np.random.default_rng(1)))
    a = get_solver(solver)
    if not a.available():
        pytest.skip(f"{solver} unavailable")
    a.build(problem)
    c = closed_loop_cost(problem, run_closed_loop(
        problem, a, problem.x0_nominal(), np.random.default_rng(1)))
    assert abs(c - ref) / abs(ref) < 1e-3


def test_fastsqp_matches_ipopt_on_lti():
    """FastSQP (codegen C++ SQP) must match IPOPT on a linear problem."""
    a = get_solver("fastsqp")
    if not a.available():
        pytest.skip("fastsqp toolchain unavailable")
    problem = get_problem("double_integrator")
    a_ref = get_solver("casadi_ipopt"); a_ref.build(problem)
    ref = closed_loop_cost(problem, run_closed_loop(
        problem, a_ref, problem.x0_nominal(), np.random.default_rng(2)))
    a.build(problem)
    c = closed_loop_cost(problem, run_closed_loop(
        problem, a, problem.x0_nominal(), np.random.default_rng(2)))
    assert abs(c - ref) / abs(ref) < 1e-3


def test_problem_registry_nonempty():
    assert set(["pendulum_swingup", "chain_mass", "kinematic_vehicle"]).issubset(
        set(list_problems()))


def test_qp_solvers_match_ipopt_on_lti():
    """OSQP and PIQP must reproduce the IPOPT closed-loop cost on the QP."""
    problem = get_problem("chain_mass")
    x0 = problem.x0_nominal()
    costs = {}
    for s in ["casadi_ipopt", "osqp", "piqp"]:
        a = get_solver(s)
        if not a.available() or not a.supports(problem):
            continue
        a.build(problem)
        log = run_closed_loop(problem, a, x0, np.random.default_rng(1))
        costs[s] = closed_loop_cost(problem, log)
    assert "casadi_ipopt" in costs
    ref = costs["casadi_ipopt"]
    for s, c in costs.items():
        assert abs(c - ref) / abs(ref) < 1e-3, f"{s} cost {c} != IPOPT {ref}"


def test_qp_solver_skips_nonlinear():
    problem = get_problem("pendulum_swingup")
    assert not get_solver("osqp").supports(problem)
    assert not get_solver("piqp").supports(problem)


def test_clean_is_scoped_and_safe():
    """clean() only ever targets paths inside the project root or cwd."""
    import os
    from fastbench.core import clean as cl
    roots = [os.path.abspath(cl.ROOT), os.path.abspath(os.getcwd())]
    for deep in (False, True):
        for t in cl.collect_targets(deep=deep):
            assert any(os.path.abspath(t).startswith(r + os.sep) for r in roots)
            assert os.path.abspath(t) not in roots          # never a root itself
    # source files must never be selected for deletion
    targets = set(cl.collect_targets(deep=True))
    assert cl.ROOT + "/fastbench/cli.py" not in targets
    assert cl.ROOT + "/cpp/fastsqp.cpp" not in targets
