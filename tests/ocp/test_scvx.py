"""SCvx (experimental) as an OCP method: a nonlinear model to a terminal equality, a linear one with a
path constraint and a terminal polytope, each against the direct method on IPOPT; the warm start and
its shift; its statuses; what it refuses."""

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
IPOPT = ocp.Direct(sc.opt.IPOPT(options={"tol": 1e-12}))


@sc.function(2, 1, output="xdot", name="scvx_pendulum")
def pendulum(x, u):
  return sc.stack([x[1], -x[0].sin() - 0.2 * x[1] + u[0]])


@sc.function(2, 1, output="g", name="scvx_speed")
def speed(x, u):
  return x[1:]


def _linear(name: str, **kwargs) -> ocp.DiscreteOCP:
  spec: dict[str, Any] = {
    "step": si.affine(A, B, name=f"{name}_map"),
    "N": 30,
    "stage_cost": ocp.Quadratic(np.eye(2), 0.1 * np.eye(1)),
    "terminal_cost": ocp.Quadratic(10 * np.eye(2)),
    "u_bounds": (-1.0, 1.0),
    "name": name,
  }
  return ocp.DiscreteOCP(**{**spec, **kwargs})


@pytest.mark.method("opt.ipopt")
def test_a_nonlinear_model_reaches_the_direct_methods_optimum() -> None:
  continuous = ocp.ContinuousOCP(
    pendulum,
    T=2.0,
    stage_cost=ocp.Quadratic(np.diag([1.0, 0.1]), 0.1 * np.eye(1)),
    terminal_cost=ocp.Quadratic(20.0 * np.eye(2), x_ref=np.array([1.0, 0.0])),  # weighed against dt times the running cost
    u_bounds=(-1.5, 1.5),
    name="scvx_pendulum_ocp",
  )
  problem = ocp.transcribe(continuous, ocp.MultipleShooting(si.RK4()), N=20)
  scvx = Controller(problem, ocp.SCvx()).solve(np.zeros(2))
  nlp = Controller(problem, IPOPT).solve(np.zeros(2))
  assert scvx.status == sc.Status.OK and nlp.status.ok
  np.testing.assert_allclose(scvx.us, nlp.us, atol=1e-5)
  np.testing.assert_allclose(scvx.cost, nlp.cost, rtol=1e-8)
  assert np.abs(nlp.us).max() > 1.5 - 1e-6  # the bound binds


@pytest.mark.method("opt.ipopt")
@pytest.mark.parametrize("constraint", ["path", "terminal set"], ids=["path", "terminal_set"])
def test_linear_constraints_hold_at_the_direct_methods_optimum(constraint: str) -> None:
  spec = (
    {"constraints": [ocp.Path(speed, lo=-0.4, hi=0.4)]} if constraint == "path" else {"terminal": sets.Polytope.box([-0.05, -0.05], [0.05, 0.05])}
  )
  problem = _linear(f"scvx_{constraint.replace(' ', '_')}", **spec)
  x0 = np.array([-1.0, 0.0])
  scvx = Controller(problem, ocp.SCvx()).solve(x0)
  nlp = Controller(problem, IPOPT).solve(x0)
  assert scvx.status == sc.Status.OK
  np.testing.assert_allclose(scvx.us, nlp.us, atol=1e-4)
  np.testing.assert_allclose(scvx.cost, nlp.cost, rtol=1e-6)


def test_the_warm_start_is_the_reference_trajectory() -> None:
  problem = _linear("scvx_shift", N=2)
  method = ocp.SCvx()
  assert method.warm_size(problem) == 3 * 2 + 2
  np.testing.assert_array_equal(ocp.initial_guess(problem, method, np.array([1.0, 2.0]), np.array([0.5])), [1, 2, 1, 2, 1, 2, 0.5, 0.5])
  np.testing.assert_array_equal(ocp.shift(problem, method)(np.arange(8.0)), [2, 3, 4, 5, 4, 5, 7, 7])


def test_the_iteration_limit() -> None:
  solution = Controller(_linear("scvx_limit"), ocp.SCvx(max_iter=1)).solve(np.array([-1.0, 0.0]))
  assert solution.status == sc.Status.MAX_ITER and solution.iterations == 1


def test_what_scvx_refuses() -> None:
  @sc.function(2, 1, output="l", name="scvx_function_cost")
  def function_cost(x, u):
    return (x * x).sum() + (u * u).sum()

  cases = {
    "convex Quadratic costs": _linear("scvx_function", stage_cost=function_cost),
    "hard path constraints": _linear("scvx_soft", constraints=[ocp.Path(speed, hi=0.4, soft=10.0)]),
    "constraints are linear": _linear("scvx_ellipsoid", terminal=sets.Ellipsoid(np.eye(2), 1.0)),
    "a discrete map or multiple shooting": ocp.transcribe(
      ocp.ContinuousOCP(pendulum, T=1.0, stage_cost=ocp.Quadratic(np.eye(2), np.eye(1))), ocp.Collocation(2), N=3
    ),
  }
  for reason, problem in cases.items():
    with pytest.raises(ValueError, match=reason):
      ocp.solver(problem, ocp.SCvx())
