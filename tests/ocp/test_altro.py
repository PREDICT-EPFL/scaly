"""ALTRO (experimental) as an OCP method: bounds with a terminal equality, a hard path constraint and a
terminal set, each against the direct method on IPOPT, with and without the projected Newton phase; a
varying reference; the warm start and its shift; its statuses; what it refuses. The case study
``examples/case_studies/altro`` checks it against Altro.jl's recorded iterates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import scaly as sc
from scaly import integrators as si
from scaly import ocp, sets

from .support import Controller

A = np.array([[1.0, 0.1], [0.0, 1.0]])
B = np.array([[0.005], [0.1]])
X0 = np.array([-1.0, 0.0])


@sc.function(2, 1, output="g", name="altro_speed")
def speed(x, u):
  return x[1:]


def _problem(name: str, **kwargs) -> ocp.DiscreteOCP:
  spec: dict[str, Any] = {
    "step": si.affine(A, B, name=f"{name}_map"),
    "N": 30,
    "stage_cost": ocp.Quadratic(np.eye(2), 0.1 * np.eye(1)),
    "terminal_cost": ocp.Quadratic(10 * np.eye(2)),
    "u_bounds": (-1.0, 1.0),
    "name": name,
  }
  return ocp.DiscreteOCP(**{**spec, **kwargs})


CASES = {
  "equality": ({"x_bounds": ([-5.0, -0.4], [5.0, 0.4]), "terminal": ocp.TerminalEquality(np.zeros(2))}, 1e-5, 1e-7),
  "path": ({"constraints": [ocp.Path(speed, lo=-0.4, hi=0.4)]}, 1e-3, 1e-5),
  "set": ({"terminal": sets.Polytope.box([-0.05, -0.05], [0.05, 0.05])}, 1e-5, 1e-7),
}


@pytest.mark.solver("ipopt")
@pytest.mark.parametrize("projected_newton", [False, True], ids=["al", "pn"])
@pytest.mark.parametrize("case", list(CASES))
def test_the_constrained_problems_optimum(case: str, projected_newton: bool) -> None:
  spec, du, dcost = CASES[case]
  problem = _problem(f"altro_{case}_{int(projected_newton)}", **spec)
  altro = Controller(problem, ocp.ALTRO(projected_newton=projected_newton)).solve(X0)
  nlp = Controller(problem, ocp.Direct(sc.opt.IPOPT(options={"tol": 1e-12}))).solve(X0)
  assert altro.status == sc.Status.OK and nlp.status.ok
  np.testing.assert_allclose(altro.us, nlp.us, atol=du)
  np.testing.assert_allclose(altro.cost, nlp.cost, rtol=dcost)
  assert np.abs(altro.us).max() <= 1 + 1e-6  # the control bounds


@pytest.mark.solver("ipopt")
def test_the_bounds_and_the_equality_bind_and_hold() -> None:
  problem = _problem("altro_binding", **CASES["equality"][0])
  solution = Controller(problem, ocp.ALTRO(projected_newton=True)).solve(X0)
  assert np.abs(solution.us).max() > 1 - 1e-6 and solution.xs[:, 1].max() > 0.4 - 1e-6  # both kinds of bound bind
  assert solution.xs[1:, 1].max() <= 0.4 + 1e-8 and np.abs(solution.xs[-1]).max() < 1e-8  # the projection's tolerance


@pytest.mark.solver("ipopt")
def test_a_continuous_model_scales_its_running_cost_by_dt() -> None:
  pendulum = sc.function(2, 1, output="xdot", name="altro_swing")(lambda x, u: sc.stack([x[1], -x[0].sin() - 0.2 * x[1] + u[0]]))
  continuous = ocp.ContinuousOCP(
    pendulum,
    T=3.0,
    stage_cost=ocp.Quadratic(np.diag([1.0, 0.1]), 0.05 * np.eye(1), x_ref=np.array([np.pi, 0.0])),
    terminal_cost=ocp.Quadratic(20.0 * np.eye(2), x_ref=np.array([np.pi, 0.0])),  # weighed against dt times the running cost
    u_bounds=(-2.0, 2.0),
    name="altro_swing_ocp",
  )
  problem = ocp.transcribe(continuous, ocp.MultipleShooting(si.RK4()), N=30)
  altro = Controller(problem, ocp.ALTRO(), name="altro_swing_solve").solve(np.zeros(2), warm=np.full(30, 0.5))
  nlp = Controller(problem, ocp.Direct(sc.opt.IPOPT(options={"tol": 1e-12}))).solve(np.zeros(2), warm=None)
  assert altro.status == sc.Status.OK and nlp.status.ok
  np.testing.assert_allclose(altro.cost, nlp.cost, rtol=1e-6)
  np.testing.assert_allclose(altro.us, nlp.us, atol=1e-4)


@pytest.mark.solver("ipopt")
def test_a_varying_reference_through_a_parameter() -> None:
  ref = np.stack([np.linspace(-1.0, 0.5, 31), np.full(31, 0.05)], axis=1).ravel()
  problem = _problem("altro_tracking", stage_cost=ocp.Quadratic(np.eye(2), 0.1 * np.eye(1), x_ref="r"), terminal_cost=None, varying=("r",))
  altro = Controller(problem, ocp.ALTRO()).solve(X0, r=ref)
  nlp = Controller(problem, ocp.Direct(sc.opt.IPOPT(options={"tol": 1e-12}))).solve(X0, r=ref)
  np.testing.assert_allclose(altro.us, nlp.us, atol=1e-5)


def test_the_warm_start_is_the_controls() -> None:
  problem = _problem("altro_shift", N=3)
  method = ocp.ALTRO()
  assert method.warm_size(problem) == 3
  np.testing.assert_array_equal(ocp.initial_guess(problem, method, X0, np.array([0.2])), [0.2] * 3)
  np.testing.assert_array_equal(ocp.shift(problem, method)(np.arange(3.0)), [1.0, 2.0, 2.0])


def test_the_iteration_limit() -> None:
  problem = _problem("altro_limit", **CASES["equality"][0])
  assert Controller(problem, ocp.ALTRO(iterations_outer=1)).solve(X0).status == sc.Status.MAX_ITER


def test_what_altro_refuses() -> None:
  @sc.function(2, 1, output="l", name="altro_function_cost")
  def function_cost(x, u):
    return (x * x).sum() + (u * u).sum()

  pendulum = sc.function(2, 1, output="xdot", name="altro_pendulum")(lambda x, u: sc.stack([x[1], -x[0].sin() + u[0]]))
  cases = {
    "hard path constraints": (_problem("altro_soft", constraints=[ocp.Path(speed, hi=0.4, soft=10.0)]), ocp.ALTRO()),
    "variables of its own": (ocp.transcribe(ocp.ContinuousOCP(pendulum, T=1.0), ocp.Collocation(2), N=3), ocp.ALTRO()),
    "diagonal weights": (_problem("altro_dense", stage_cost=ocp.Quadratic(np.ones((2, 2)) + np.eye(2), np.eye(1))), ocp.ALTRO(projected_newton=True)),
    "Quadratic costs": (_problem("altro_function", stage_cost=function_cost), ocp.ALTRO(projected_newton=True)),
  }
  for reason, (problem, method) in cases.items():
    with pytest.raises(ValueError, match=reason):
      ocp.solver(problem, method)
  assert ocp.ALTRO().supports(_problem("altro_function_al", stage_cost=function_cost)).ok  # the AL phase takes any cost
