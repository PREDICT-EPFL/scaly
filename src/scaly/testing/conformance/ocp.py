"""The DiscreteOCP suite: reference problems (linear-quadratic, boxed, tracking a varying reference, nonlinear by multiple shooting) and the contract an OCP method meets on them: the direct method's optimum on IPOPT, to the method's tolerance."""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np

import scaly as sc
from scaly import integrators as si
from scaly import ocp

A = np.array([[1.0, 0.1], [0.0, 1.0]])
B = np.array([[0.005], [0.1]])
X0 = np.array([0.8, -0.3])
"""The initial state every problem is solved from."""


@cache
def _pendulum() -> Any:
  @sc.function(2, 1, output="xdot", name="conformance_pendulum")
  def pendulum(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    return sc.stack([x[1], -x[0].sin() - 0.2 * x[1] + u[0]])

  return pendulum


def _linear(name: str, **kwargs: Any) -> ocp.DiscreteOCP:
  spec: dict[str, Any] = {
    "step": si.affine(A, B, name=f"{name}_map"),
    "N": 20,
    "stage_cost": ocp.Quadratic(np.eye(2), 0.1 * np.eye(1)),
    "terminal_cost": ocp.Quadratic(10 * np.eye(2)),
    "name": name,
  }
  return ocp.DiscreteOCP(**{**spec, **kwargs})


@cache
def problems() -> dict[str, tuple[ocp.DiscreteOCP, dict[str, np.ndarray]]]:
  """The reference problems by name, each with the parameter values it is solved at."""
  horizon = 20
  ref = np.stack([np.linspace(0.8, 0.0, horizon + 1), np.full(horizon + 1, -0.04)], axis=1).ravel()
  continuous = ocp.ContinuousOCP(
    _pendulum(),
    T=2.0,
    stage_cost=ocp.Quadratic(np.diag([1.0, 0.1]), 0.1 * np.eye(1)),
    terminal_cost=ocp.Quadratic(10.0 * np.eye(2)),
    name="conformance_pendulum_ocp",
  )
  return {
    "lq": (_linear("conformance_lq"), {}),
    "boxed": (_linear("conformance_boxed", u_bounds=(-0.5, 0.5), x_bounds=([-5.0, -0.35], [5.0, 0.35])), {}),
    "tracking": (
      _linear(
        "conformance_tracking",
        stage_cost=ocp.Quadratic(np.eye(2), 0.1 * np.eye(1), x_ref="r"),
        terminal_cost=ocp.Quadratic(10 * np.eye(2), x_ref="r"),
        varying=("r",),
      ),
      {"r": ref},
    ),
    "nonlinear": (ocp.transcribe(continuous, ocp.MultipleShooting(si.RK4()), N=horizon), {}),
  }


def solve(problem: ocp.DiscreteOCP, method: Any, params: dict[str, np.ndarray], *, name: str) -> tuple[np.ndarray, np.ndarray, sc.Status, float]:
  """``problem`` solved by ``method`` from ``X0`` and its cold start: the states, the controls, the
  status and the objective."""
  solve_fn = ocp.solver(problem, method, name=name)
  values = [np.ravel(np.asarray(params[p.name], dtype=np.float64)) for p in problem.params]
  xs, us, _, info = solve_fn(X0, *values, ocp.initial_guess(problem, method, X0))
  return np.asarray(xs), np.asarray(us), sc.Status(int(info.status)), float(info.objective)


@cache
def reference(problem_name: str) -> tuple[np.ndarray, np.ndarray, float]:
  """The direct method's optimum on IPOPT, tight: the states, the controls, the objective."""
  problem, params = problems()[problem_name]
  xs, us, status, cost = solve(problem, ocp.Direct(sc.opt.IPOPT(options={"tol": 1e-12})), params, name=f"{problem.name}_reference")
  assert status.ok, status
  return xs, us, cost


def check(method: Any, problem_name: str, tolerance: float, *, label: str) -> None:
  """``method`` reaches the reference optimum of ``problem_name`` to ``tolerance``: the controls and
  the states absolutely, the objective relatively. ``label`` names its Function."""
  problem, params = problems()[problem_name]
  xs_ref, us_ref, cost_ref = reference(problem_name)
  xs, us, status, cost = solve(problem, method, params, name=f"{problem.name}_{label}_conformance")
  assert status.ok, status
  np.testing.assert_allclose(us, us_ref, atol=tolerance)
  np.testing.assert_allclose(xs, xs_ref, atol=tolerance)
  np.testing.assert_allclose(cost, cost_ref, rtol=tolerance)


__all__ = ["X0", "check", "problems", "reference", "solve"]
