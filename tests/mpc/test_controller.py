"""The controller: the control law's warm start and shift, its agreement with whole solves, the
closed loop, and the generated C."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly import integrators as si
from scaly import mpc
from scaly.codegen import render_c_module

A = np.array([[1.0, 0.1], [0.0, 1.0]])
B = np.array([[0.005], [0.1]])


@sc.function(2, 1, output="xnext")
def double(x, u):
  return sc.const(A) @ x + sc.const(B) @ u


@sc.function(2, 1, output="g")
def speed(x, u):
  return x[1:]


def _ocp(name: str, horizon: int = 10) -> mpc.OCP:
  return mpc.OCP(
    step=double,
    horizon=horizon,
    stage_cost=mpc.Quadratic(np.eye(2), 0.1 * np.eye(1)),
    terminal_cost=mpc.Quadratic(10 * np.eye(2)),
    u_bounds=(-1, 1),
    constraints=[mpc.Path(speed, lo=-0.8, hi=0.8, soft=50.0)],
    name=name,
  )


def test_the_shift_moves_every_stage_block_up_one() -> None:
  controller = mpc.MPC(_ocp("shift_layout", horizon=3), "ipopt")
  layout = controller.ocp.layout
  assert layout.var_names == ("xs", "us", "slack") and layout.var_sizes == (8, 3, 3)
  n_vars, n_eq, n_ineq = 14, controller.n_eq, controller.n_ineq
  assert (n_eq, n_ineq) == (2 + 3 * 2, 2 * 3)
  guess = np.arange(controller.guess_size, dtype=float)
  moved = np.asarray(controller.shift(guess))
  xs, us, slack = guess[:8], guess[8:11], guess[11:14]
  np.testing.assert_array_equal(moved[:8], [*xs[2:], *xs[-2:]])
  np.testing.assert_array_equal(moved[8:11], [*us[1:], us[-1]])
  np.testing.assert_array_equal(moved[11:14], [*slack[1:], slack[-1]])
  np.testing.assert_array_equal(
    moved[n_vars : 2 * n_vars],
    np.concatenate([[*g[2:], *g[-2:]] if i == 0 else [*g[1:], g[-1]] for i, g in enumerate(np.split(guess[n_vars : 2 * n_vars], [8, 11]))]),
  )
  eq = guess[2 * n_vars : 2 * n_vars + n_eq]
  np.testing.assert_array_equal(moved[2 * n_vars : 2 * n_vars + n_eq], [*eq[:2], *eq[4:], *eq[-2:]])  # x0's stay, the dynamics' move
  ineq = guess[2 * n_vars + n_eq :]
  np.testing.assert_array_equal(moved[2 * n_vars + n_eq :], [*ineq[1:3], ineq[2], *ineq[4:6], ineq[5]])  # each soft side on its own


@pytest.mark.solver("ipopt")
def test_the_control_law_is_the_solve_and_its_shifted_warm_start() -> None:
  law = mpc.MPC(_ocp("law_side"), "ipopt", options={"tol": 1e-10})
  whole = mpc.MPC(_ocp("solve_side"), "ipopt", options={"tol": 1e-10})
  x_law = x_solve = np.array([2.0, 0.0])
  for _ in range(4):
    u = law(x_law)
    solution = whole.solve(x_solve)
    np.testing.assert_allclose(u, solution.us[0], rtol=1e-9, atol=1e-10)
    assert law._guess is not None and whole._guess is not None
    np.testing.assert_allclose(law._guess, whole._guess, rtol=1e-9, atol=1e-9)
    x_law = A @ x_law + B @ u
    x_solve = A @ x_solve + B @ solution.us[0]
  assert law.status.ok


@pytest.mark.solver("piqp")
def test_a_closed_loop_reaches_the_origin_within_its_limits() -> None:
  controller = mpc.MPC(_ocp("closed_loop", horizon=20), "piqp", options={"eps_abs": 1e-9, "eps_rel": 1e-9})
  run = mpc.simulate(controller, lambda x, u: A @ x + B @ u, np.array([2.0, 0.0]), 80)
  assert run.xs.shape == (81, 2) and run.us.shape == (80, 1) and set(run.statuses) == {"OK"}
  assert np.abs(run.xs[-1]).max() < 1e-2 and np.abs(run.us).max() <= 1 + 1e-6
  assert np.abs(run.xs[:, 1]).max() < 0.8 + 1e-3  # the soft speed limit holds, since it can


@pytest.mark.solver("ipopt")
def test_a_continuous_plant_through_an_adaptive_map() -> None:
  @sc.function(2, 1, output="xdot")
  def pendulum(x, u):
    return sc.stack([x[1], -x[0].sin() - 0.1 * x[1] + u[0]])

  ocp = mpc.OCP(
    ode=pendulum,
    dt=0.1,
    horizon=15,
    stage_cost=mpc.Quadratic(np.eye(2), 0.1 * np.eye(1)),
    terminal_cost=mpc.Quadratic(10 * np.eye(2)),
    u_bounds=(-2, 2),
    name="pendulum_mpc",
  )
  controller = mpc.MPC(ocp, "ipopt")
  plant = si.adaptive(pendulum, dt=0.1, rtol=1e-10, atol=1e-12, name="pendulum_plant")
  run = mpc.simulate(controller, plant, np.array([1.0, 0.0]), 40)
  assert np.abs(run.xs[-1]).max() < 1e-2 and np.all(run.iterations > 0) and np.all(run.solve_times > 0)


@pytest.mark.solver("piqp")
def test_a_closed_loop_parameter_may_change_with_the_step() -> None:
  ocp = mpc.OCP(step=double, horizon=10, stage_cost=mpc.Quadratic(np.eye(2), 0.1 * np.eye(1), x_ref="target"), u_bounds=(-1, 1), name="moving_target")
  controller = mpc.MPC(ocp, "piqp")
  run = mpc.simulate(controller, lambda x, u: A @ x + B @ u, np.zeros(2), 120, target=lambda k: np.array([1.0 if k < 60 else -1.0, 0.0]))
  assert abs(run.xs[60, 0] - 1.0) < 0.1 and abs(run.xs[-1, 0] + 1.0) < 0.1  # the target the step gave is tracked


def test_the_law_generates_one_c_module_with_its_solver() -> None:
  controller = mpc.MPC(_ocp("generated"), "ipopt")
  module = render_c_module(controller.law)
  assert "generated_law" in module.body and "generated_ipopt" in module.body
  assert controller.law.input_names == ("x0", "guess") and controller.law.output_names == ("u", "guess_next")
  assert controller.law.inputs[1].shape == (controller.guess_size,)


def test_the_initial_guess() -> None:
  ocp = mpc.OCP(ode=double_ode, dt=0.1, horizon=3, transcription=sc.ocp.Collocation(2), name="guess_layout")
  controller = mpc.MPC(ocp, "ipopt")
  guess = controller.initial_guess(np.array([1.0, 2.0]), np.array([0.5]))
  n_vars = ocp.layout.n_vars
  np.testing.assert_array_equal(guess[:8], np.tile([1.0, 2.0], 4))
  np.testing.assert_array_equal(guess[8:11], [0.5] * 3)
  np.testing.assert_array_equal(guess[11:n_vars], np.tile([1.0, 2.0], 3))  # one internal state per Radau(2) interval
  assert not guess[n_vars:].any()


@sc.function(2, 1, output="xdot")
def double_ode(x, u):
  return sc.stack([x[1], u[0]])
