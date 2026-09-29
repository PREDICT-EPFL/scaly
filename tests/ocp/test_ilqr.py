"""iLQR as an OCP method: the LQR in one step on a linear-quadratic problem, a nonlinear problem's
optimum against the direct method, a continuous model through multiple shooting, varying
parameters, the warm start and its shift, its statuses, and what it refuses."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly import integrators as si
from scaly import ocp
from scaly.codegen import render_c_module

from .support import Controller, closed_loop

A = np.array([[1.0, 0.1], [0.0, 1.0]])
B = np.array([[0.005], [0.1]])
Q, R = np.diag([1.0, 0.5]), 0.1 * np.eye(1)
K, P = ocp.lqr(A, B, Q, R)


def _linear(name: str, horizon: int = 15, **kwargs) -> ocp.DiscreteOCP:
  return ocp.DiscreteOCP(
    step=si.affine(A, B, name=f"{name}_map"), N=horizon, stage_cost=ocp.Quadratic(Q, R), terminal_cost=ocp.Quadratic(P), name=name, **kwargs
  )


def test_on_a_linear_quadratic_problem_ilqr_is_the_lqr() -> None:
  """The LQ problem's cost is its own quadratic model, so the first step is the solution; with the LQR
  cost to go at the end, ``u_0 = K x`` and the optimal cost is ``x'Px``."""
  problem = _linear("ilqr_lq")
  controller = Controller(problem, ocp.ILQR(mu=1e-12, mu_min=1e-12))
  for x in (np.array([1.0, -0.5]), np.array([-3.0, 2.0])):
    solution = controller.solve(x, warm=controller.initial_guess(x))
    assert solution.status == sc.Status.OK and solution.iterations <= 3
    np.testing.assert_allclose(solution.us[0], K @ x, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(solution.cost, x @ P @ x, rtol=1e-9)
    np.testing.assert_allclose(solution.xs[1:], (A @ solution.xs[:-1].T + B @ solution.us.T).T, atol=1e-14)  # a rollout: dynamics exact


NX, NU, N, DT = 4, 2, 40, 0.1
OBSTACLE = np.array([2.0, 0.6, 0.6])


@sc.function(NX, NU, output="xnext", name="ilqr_car")
def car(x, u):
  px, py, h, v = x[0], x[1], x[2], x[3]
  return sc.stack([px + DT * v * h.cos(), py + DT * v * h.sin(), h + DT * v * u[1], v + DT * u[0]])


@sc.function(NX, NU, sc.L("goal", NX), output="l", name="ilqr_car_stage")
def car_stage(x, u, goal):
  e = x - goal
  gap = (x[0] - OBSTACLE[0]) ** 2 + (x[1] - OBSTACLE[1]) ** 2 - OBSTACLE[2] ** 2
  return DT * (0.05 * e[3] * e[3] + 0.05 * u[0] * u[0] + 0.5 * u[1] * u[1] + 20.0 * (-gap / 0.1).exp())


@sc.function(NX, sc.L("goal", NX), output="vf", name="ilqr_car_terminal")
def car_terminal(x, goal):
  e = x - goal
  return 50.0 * (e[0] * e[0] + e[1] * e[1]) + 25.0 * e[2] * e[2] + 5.0 * e[3] * e[3]


CAR = ocp.DiscreteOCP(step=car, N=N, stage_cost=car_stage, terminal_cost=car_terminal, name="ilqr_car_ocp")
GOAL = np.array([4.0, 1.5, np.pi / 2, 0.0])


@pytest.mark.solver("ipopt")
def test_a_nonlinear_problem_reaches_the_direct_methods_optimum() -> None:
  x0 = np.zeros(NX)
  ilqr = Controller(CAR, ocp.ILQR()).solve(x0, goal=GOAL)
  direct = Controller(CAR, ocp.Direct(sc.opt.IPOPT(options={"tol": 1e-12}))).solve(x0, goal=GOAL)
  assert ilqr.status == sc.Status.OK and direct.status.ok
  np.testing.assert_allclose(ilqr.cost, direct.cost, rtol=1e-8)
  np.testing.assert_allclose(ilqr.us, direct.us, atol=1e-4)
  assert np.hypot(ilqr.xs[:, 0] - OBSTACLE[0], ilqr.xs[:, 1] - OBSTACLE[1]).min() > OBSTACLE[2]  # around the obstacle


@sc.function(sc.L("us", N * NU), sc.L("x0", NX), sc.L("goal", NX), output="cost", name="ilqr_car_shooting")
def shooting(us, x0, goal):
  x, total = x0, 0.0
  for k in range(N):
    u = us[k * NU : (k + 1) * NU]
    total = total + car_stage(x, u, goal)
    x = car(x, u)
  return total + car_terminal(x, goal)


def test_the_solution_is_a_stationary_point_of_the_shooting_cost() -> None:
  x0 = np.zeros(NX)
  solution = Controller(CAR, ocp.ILQR()).solve(x0, goal=GOAL)
  grad = sc.gradient(shooting, wrt="us")(solution.point, x0, GOAL)
  assert np.abs(grad).max() < 1e-5
  np.testing.assert_allclose(shooting(solution.point, x0, GOAL), solution.cost, rtol=1e-14)


@sc.function(2, 1, output="xdot", name="ilqr_pendulum")
def pendulum(x, u):
  return sc.stack([x[1], -x[0].sin() - 0.1 * x[1] + u[0]])


@pytest.mark.solver("ipopt")
def test_a_continuous_model_by_multiple_shooting_solves_the_direct_methods_problem() -> None:
  """The running cost at the points is ``dt`` times the sum; the map is the transcription's integrator."""
  continuous = ocp.ContinuousOCP(
    pendulum, T=2.0, stage_cost=ocp.Quadratic(np.eye(2), 0.1 * np.eye(1)), terminal_cost=ocp.Quadratic(10 * np.eye(2)), name="ilqr_pendulum_ocp"
  )
  problem = ocp.transcribe(continuous, ocp.MultipleShooting(si.RK4(steps=2)), N=20)
  x0 = np.array([1.2, 0.0])
  ilqr = Controller(problem, ocp.ILQR()).solve(x0)
  direct = Controller(problem, ocp.Direct(sc.opt.IPOPT(options={"tol": 1e-12}))).solve(x0)
  np.testing.assert_allclose(ilqr.cost, direct.cost, rtol=1e-9)
  np.testing.assert_allclose(ilqr.us, direct.us, atol=1e-5)
  np.testing.assert_allclose(ilqr.xs, direct.xs, atol=1e-6)


def test_a_varying_reference_is_read_stage_by_stage_and_at_the_end() -> None:
  """``x+ = x + u`` from ``r_0`` tracking ``r_k``: every reference is reachable, at no state cost."""

  @sc.function(1, 1, output="xnext", name="ilqr_integrator")
  def integrator(x, u):
    return x + u

  ref = np.array([0.0, 0.5, 0.2, 1.0, 1.3, 0.7, 0.9])
  problem = ocp.DiscreteOCP(
    step=integrator,
    N=6,
    stage_cost=ocp.Quadratic(np.eye(1), 1e-9 * np.eye(1), x_ref="r"),
    terminal_cost=ocp.Quadratic(np.eye(1), x_ref="r"),
    varying=("r",),
    name="ilqr_tracking",
  )
  solution = Controller(problem, ocp.ILQR(mu=1e-8)).solve(np.array([0.0]), r=ref)
  np.testing.assert_allclose(solution.xs.ravel(), ref, atol=1e-7)
  np.testing.assert_allclose(solution.us.ravel(), np.diff(ref), atol=1e-7)


def test_the_warm_start_is_the_controls_and_the_shift_moves_them_up() -> None:
  problem = _linear("ilqr_shift", horizon=4)
  method = ocp.ILQR()
  assert method.warm_size(problem) == 4
  np.testing.assert_array_equal(ocp.initial_guess(problem, method, np.zeros(2), np.array([0.3])), np.full(4, 0.3))
  np.testing.assert_array_equal(ocp.shift(problem, method)(np.arange(4.0)), [1.0, 2.0, 3.0, 3.0])


def test_a_receding_horizon_warm_starts_from_the_shifted_controls() -> None:
  controller = Controller(CAR, ocp.ILQR())
  x0 = np.zeros(NX)
  xs, _, statuses, iterations = closed_loop(controller, lambda x, u: car(x, u), x0, 6, goal=GOAL)
  assert set(statuses) == {sc.Status.OK}
  assert iterations[1:].max() < iterations[0]  # the shifted plan starts closer; mu starts afresh
  assert np.abs(xs[-1] - x0).max() > 0.1  # the car moves


def test_the_statuses() -> None:
  x0 = np.zeros(NX)
  assert Controller(CAR, ocp.ILQR(max_iter=2), name="ilqr_car_two").solve(x0, goal=GOAL).status == sc.Status.MAX_ITER

  @sc.function(1, 1, output="l", name="ilqr_undefined")
  def undefined(x, u):
    return (-(1.0 + u[0] * u[0])).sqrt()  # NaN everywhere: every step fails, mu grows past its limit

  @sc.function(1, 1, output="xnext", name="ilqr_scalar_map")
  def scalar_map(x, u):
    return x + u

  broken = ocp.DiscreteOCP(step=scalar_map, N=3, stage_cost=undefined, name="ilqr_broken")
  assert Controller(broken, ocp.ILQR()).solve(np.zeros(1)).status == sc.Status.NUMERICS


def test_the_solver_and_its_shift_generate_one_c_module() -> None:
  problem = _linear("ilqr_law")
  method = ocp.ILQR()
  solve, shift = ocp.solver(problem, method), ocp.shift(problem, method)

  @sc.function(sc.L("x0", 2), sc.L("warm", method.warm_size(problem)), output=sc.G("u", "warm_next"), name="ilqr_law_law")
  def law(x0, warm):
    _, us, point, _ = solve(x0, warm)
    return us[0], shift(point)

  assert solve.name == "ilqr_law_ilqr"
  body = render_c_module(law).body
  assert "ilqr_law_ilqr" in body and "ilqr_law_shift_ilqr" in body


def test_what_ilqr_refuses() -> None:
  @sc.function(2, 1, output="g", name="ilqr_speed")
  def speed(x, u):
    return x[1:]

  cases = {
    "no path constraints": _linear("ilqr_paths", constraints=[ocp.Path(speed, hi=1.0)]),
    "no bounds": _linear("ilqr_bounds", u_bounds=(-1, 1)),
    "no terminal set or equality": _linear("ilqr_terminal", terminal=ocp.TerminalEquality()),
    "not a transcription with variables of its own": ocp.transcribe(ocp.ContinuousOCP(pendulum, T=1.0), ocp.Collocation(2), N=3),
    "not integrated": ocp.transcribe(
      ocp.ContinuousOCP(pendulum, T=1.0, stage_cost=ocp.Quadratic(np.eye(2), np.eye(1)), cost="integral", name="ilqr_integral"), N=3
    ),
  }
  for reason, problem in cases.items():
    with pytest.raises(ValueError, match=reason):
      ocp.solver(problem, ocp.ILQR())
  assert ocp.ILQR().supports(_linear("ilqr_open", u_bounds=(None, np.inf))).ok  # sides that bound nothing
  with pytest.raises(ValueError, match="mu_min <= mu <= mu_max"):
    ocp.ILQR(mu=0.0)
