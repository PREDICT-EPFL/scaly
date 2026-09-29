"""DiscreteOCP conformance: every installed OCP method, on each reference problem it supports, reaches
the direct method's optimum (IPOPT, tight) to its own tolerance, and a problem it refuses is refused
by ``sc.ocp.solver`` too. A method missing from ``METHODS`` fails the test: a method is supported
once it passes here."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import scaly as sc
from scaly import integrators as si
from scaly import ocp

from .support import Controller

A = np.array([[1.0, 0.1], [0.0, 1.0]])
B = np.array([[0.005], [0.1]])
X0 = np.array([0.8, -0.3])

# The method with the options its tolerance below needs, and that tolerance on the controls.
METHODS: dict[str, tuple[Any, float]] = {
  "direct": (ocp.Direct(sc.opt.IPOPT(options={"tol": 1e-10})), 1e-6),
  "ilqr": (ocp.ILQR(mu=1e-8, mu_min=1e-12), 1e-6),
  "tinyadmm": (ocp.TinyADMM(rho=2.0, abs_pri_tol=1e-11, abs_dua_tol=1e-11, max_iter=50000), 1e-6),
  "altro": (ocp.ALTRO(projected_newton=True), 1e-4),
  "scvx": (ocp.SCvx(), 1e-4),
}


@sc.function(2, 1, output="xdot", name="conf_pendulum")
def pendulum(x, u):
  return sc.stack([x[1], -x[0].sin() - 0.2 * x[1] + u[0]])


def _linear(name: str, **kwargs) -> ocp.DiscreteOCP:
  spec: dict[str, Any] = {
    "step": si.affine(A, B, name=f"{name}_map"),
    "N": 20,
    "stage_cost": ocp.Quadratic(np.eye(2), 0.1 * np.eye(1)),
    "terminal_cost": ocp.Quadratic(10 * np.eye(2)),
    "name": name,
  }
  return ocp.DiscreteOCP(**{**spec, **kwargs})


def _problems() -> dict[str, tuple[ocp.DiscreteOCP, dict[str, np.ndarray]]]:
  horizon = 20
  ref = np.stack([np.linspace(0.8, 0.0, horizon + 1), np.full(horizon + 1, -0.04)], axis=1).ravel()
  continuous = ocp.ContinuousOCP(
    pendulum,
    T=2.0,
    stage_cost=ocp.Quadratic(np.diag([1.0, 0.1]), 0.1 * np.eye(1)),
    terminal_cost=ocp.Quadratic(10.0 * np.eye(2)),
    name="conf_pendulum_ocp",
  )
  return {
    "lq": (_linear("conf_lq"), {}),
    "boxed": (_linear("conf_boxed", u_bounds=(-0.5, 0.5), x_bounds=([-5.0, -0.35], [5.0, 0.35])), {}),
    "tracking": (
      _linear(
        "conf_tracking",
        stage_cost=ocp.Quadratic(np.eye(2), 0.1 * np.eye(1), x_ref="r"),
        terminal_cost=ocp.Quadratic(10 * np.eye(2), x_ref="r"),
        varying=("r",),
      ),
      {"r": ref},
    ),
    "nonlinear": (ocp.transcribe(continuous, ocp.MultipleShooting(si.RK4()), N=horizon), {}),
  }


PROBLEMS = _problems()


def test_every_installed_method_is_in_the_conformance_table() -> None:
  assert sorted(ocp.REGISTRY.installed()) == sorted(METHODS)


@pytest.mark.solver("ipopt")
@pytest.mark.parametrize("problem_name", list(PROBLEMS))
@pytest.mark.parametrize("method_name", list(METHODS))
def test_the_method_reaches_the_direct_optimum(method_name: str, problem_name: str) -> None:
  method, tolerance = METHODS[method_name]
  problem, params = PROBLEMS[problem_name]
  support = method.supports(problem)
  if not support:
    with pytest.raises(ValueError, match="cannot solve this problem"):
      ocp.solver(problem, method)
    pytest.skip(f"{method_name} refuses {problem_name}: {', '.join(support.reasons)}")
  reference = Controller(problem, ocp.Direct(sc.opt.IPOPT(options={"tol": 1e-12})), name=f"{problem.name}_reference").solve(X0, **params)
  solution = Controller(problem, method, name=f"{problem.name}_{method_name}_conformance").solve(X0, **params)
  assert reference.status.ok and solution.status.ok, solution.status
  np.testing.assert_allclose(solution.us, reference.us, atol=tolerance)
  np.testing.assert_allclose(solution.xs, reference.xs, atol=tolerance)
  np.testing.assert_allclose(solution.cost, reference.cost, rtol=tolerance)
