"""The direct method: a cart-pole transcribed by the library against the same problem written by hand,
the transcriptions against each other, parameters by name, costs, terminal equalities, soft
constraints and bounds; the warm start's shift, the control law as one C module, closed loops."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly import integrators as si
from scaly import ocp
from scaly.codegen import render_c_module

from .support import Controller, closed_loop

NX, NU, N, DT = 4, 1, 20, 0.05
M_CART, M_POLE, LENGTH, G = 1.0, 0.3, 0.5, 9.81
U_MAX, P_MAX, SOFT = 15.0, 0.35, 1e3
Q = np.array([2.0, 20.0, 0.1, 0.1])
R, QN = 0.02, 10.0
X0 = np.array([0.3, 0.4, 0.0, 0.0])
IPOPT = sc.opt.IPOPT(options={"tol": 1e-12})


def dynamics(x, u):
  theta, pd, td = x[1], x[2], x[3]
  s, c = theta.sin(), theta.cos()
  den = M_CART + M_POLE * s * s
  pdd = (u[0] + M_POLE * LENGTH * td * td * s - M_POLE * G * s * c) / den
  tdd = (G * s * (M_CART + M_POLE) - c * (u[0] + M_POLE * LENGTH * td * td * s)) / (LENGTH * den)
  return sc.stack([pd, td, pdd, tdd])


def error(x):
  return sc.stack([x[0], 1.0 - x[1].cos(), x[2], x[3]])


@sc.function(NX, NU, output="xdot")
def cartpole(x, u):
  return dynamics(x, u)


@sc.function(NX, NU, output="l")
def stage(x, u):
  e = error(x)
  return (sc.const(Q) * e * e).sum() + R * u[0] * u[0]


@sc.function(NX, output="vf")
def terminal(x):
  e = error(x)
  return QN * (sc.const(Q) * e * e).sum()


@sc.function(NX, NU, output="p")
def position(x, u):
  return x[:1]


def hand_written():
  """The formulation of `examples/nmpc_cartpole.py`, written out without `scaly.ocp` or
  `scaly.integrators`, the track limit at the stages the library puts path constraints (0 .. N-1)."""

  def rk4(x, u, h):
    k1 = dynamics(x, u)
    k2 = dynamics(x + 0.5 * h * k1, u)
    k3 = dynamics(x + 0.5 * h * k2, u)
    k4 = dynamics(x + h * k3, u)
    return x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

  @sc.function(NX, NU, NX, name="hand_defect")
  def defect(x, u, x_next):
    return rk4(rk4(x, u, DT / 2), u, DT / 2) - x_next

  @sc.function(NX, NU, name="hand_stage")
  def stage_cost(x, u):
    e = error(x)
    return ((sc.const(Q) * e * e).sum() + R * u[0] * u[0]).reshape((1,)) * DT

  @sc.opt.problem(vars=sc.G(sc.L("xs", (N + 1) * NX), sc.L("us", N * NU), sc.L("slack", N)), params=sc.L("x0", NX), name="hand_cartpole")
  def swing(variables, x0):
    xs, us, slack = variables
    defects = sc.vmap(defect, N, [(xs, 0, NX), (us, 0, NU), (xs, NX, NX)])
    costs = sc.vmap(stage_cost, N, [(xs, 0, NX), (us, 0, NU)])
    positions = xs.reshape((N + 1, NX))[:N, 0]
    e = error(xs[N * NX :])
    return sc.opt.ProblemSpec(
      minimize=costs.sum() + QN * (sc.const(Q) * e * e).sum() + SOFT * slack.sum(),
      eq=(xs[:NX] - x0, defects),
      ineq=(sc.opt.bounded(positions - slack, hi=P_MAX), sc.opt.bounded(positions + slack, lo=-P_MAX)),
      lb=(sc.opt.NO_LB, sc.const(np.full(N * NU, -U_MAX)), sc.const(np.zeros(N))),
      ub=(sc.opt.NO_UB, sc.const(np.full(N * NU, U_MAX)), sc.opt.NO_UB),
    )

  return swing


def library(transcription=None, name="lib_cartpole", *, track: bool = True, cost=None) -> ocp.DiscreteOCP:
  continuous = ocp.ContinuousOCP(
    cartpole,
    T=N * DT,
    stage_cost=stage,
    terminal_cost=terminal,
    u_bounds=(-U_MAX, U_MAX),
    constraints=[ocp.Path(position, lo=-P_MAX, hi=P_MAX, soft=SOFT)] if track else [],
    cost=cost,
    name=name,
  )
  return ocp.transcribe(continuous, transcription or ocp.MultipleShooting(si.RK4(steps=2)), N=N)


@pytest.mark.method("opt.ipopt")
def test_the_library_transcribes_the_cart_pole_as_by_hand() -> None:
  hand = sc.opt.solver(hand_written(), IPOPT, name="hand_ipopt")
  n_eq, n_ineq = (N + 1) * NX, 2 * N
  guess = (np.tile(X0, N + 1), np.zeros(N), np.zeros(N))
  (xs, us, slack), *_ = hand(guess, tuple(np.zeros(g.size) for g in guess), np.zeros(n_eq), np.zeros(n_ineq), X0)
  problem = library()
  nlp, layout = ocp.to_problem(problem)
  assert layout.var_names == ("xs", "us", "slack") and nlp.n_eq == n_eq and nlp.n_ineq == n_ineq
  solution = Controller(problem, ocp.Direct(IPOPT)).solve(X0)
  assert solution.status.ok
  np.testing.assert_allclose(solution.xs.ravel(), xs, rtol=1e-9, atol=1e-10)
  np.testing.assert_allclose(solution.us.ravel(), us, rtol=1e-9, atol=1e-10)
  assert solution.slack is not None
  np.testing.assert_allclose(solution.slack, slack, atol=1e-9)
  assert solution.slack.max() > 0.1  # the track limit gives way on the way up
  assert np.all(solution.slack >= -1e-7)  # IPOPT relaxes bounds by 1e-8


@pytest.mark.method("opt.ipopt")
def test_shooting_and_collocation_solve_the_same_problem_and_pseudospectral_improves_on_it() -> None:
  method = ocp.Direct(sc.opt.IPOPT(options={"tol": 1e-11}))
  # A path constraint binds only at the grid points, which pseudospectral controls can dodge.
  shooting = Controller(library(ocp.MultipleShooting(si.RK4(steps=4)), name="agree_shooting", track=False, cost="integral"), method).solve(X0)
  collocation = Controller(library(ocp.Collocation(3), name="agree_collocation", track=False, cost="integral"), method).solve(X0)
  pseudo = Controller(library(ocp.Pseudospectral(4), name="agree_pseudo", track=False), method).solve(X0)
  assert shooting.status.ok and collocation.status.ok and pseudo.status.ok
  # The same held controls, integrated two ways: the solutions differ by the integrators' errors.
  np.testing.assert_allclose(collocation.us, shooting.us, rtol=1e-3, atol=1e-4)
  np.testing.assert_allclose(collocation.xs, shooting.xs, rtol=1e-3, atol=1e-5)
  assert pseudo.cost <= shooting.cost * (1 + 1e-6)  # controls free inside each interval can only do better
  assert pseudo.cost > 0.95 * shooting.cost


@sc.function(1, 1, output="xnext")
def integrator(x, u):
  return x + u


TIGHT = ocp.Direct(sc.opt.PIQP(sparse=True, options={"eps_abs": 1e-10, "eps_rel": 1e-10}))


@pytest.mark.method("opt.piqp")
def test_a_varying_reference_is_read_stage_by_stage() -> None:
  """``x+ = x + u`` tracking ``r_k`` from ``x0 = r_0`` can reach every reference exactly, at zero cost."""
  ref = np.array([0.0, 0.5, 0.2, 1.0, 1.3, 0.7, 0.9])
  problem = ocp.DiscreteOCP(
    step=integrator,
    N=6,
    stage_cost=ocp.Quadratic(np.eye(1), 1e-6 * np.eye(1), x_ref="r"),
    terminal_cost=ocp.Quadratic(np.eye(1), x_ref="r"),
    varying=("r",),
    name="tracking",
  )
  assert [p.name for p in problem.params] == ["r"]
  solution = Controller(problem, TIGHT).solve(np.array([0.0]), r=ref)
  np.testing.assert_allclose(solution.xs.ravel(), ref, atol=1e-5)
  np.testing.assert_allclose(solution.us.ravel(), np.diff(ref), atol=1e-5)


@sc.function(2, 1, (), output="xdot")
def spring(x, u, stiffness):
  return sc.stack([x[1], -stiffness * x[0] + u[0]])


@sc.function(2, 1, 2, output="l")
def tracking(x, u, target):
  e = x - target
  return (e * e).sum() + 0.01 * u[0] * u[0]


@pytest.mark.method("opt.ipopt")
def test_parameters_are_the_union_of_what_the_functions_name() -> None:
  continuous = ocp.ContinuousOCP(spring, T=1.0, stage_cost=tracking, terminal=ocp.TerminalEquality(x_ref="target"), name="spring_ocp")
  problem = ocp.transcribe(continuous, N=10)
  assert [(p.name, p.type.shape) for p in problem.params] == [("stiffness", ()), ("target", (2,))]
  controller = Controller(problem, ocp.Direct(sc.opt.IPOPT(options={"tol": 1e-10})))
  assert controller.solve_fn.input_names == ("x0", "stiffness", "target", "warm")
  solution = controller.solve(np.array([1.0, 0.0]), stiffness=np.array(4.0), target=np.array([0.2, 0.0]))
  assert solution.status.ok
  np.testing.assert_allclose(solution.xs[-1], [0.2, 0.0], atol=1e-8)  # the terminal equality, at the parameter
  stiffer = controller.solve(
    np.array([1.0, 0.0]), warm=controller.initial_guess(np.array([1.0, 0.0])), stiffness=np.array(9.0), target=np.array([0.2, 0.0])
  )
  assert np.abs(stiffer.us - solution.us).max() > 1e-3  # the model reads its parameter


@pytest.mark.method("opt.piqp")
def test_a_discrete_map_sums_its_costs_and_the_terminal_equality_holds() -> None:
  problem = ocp.DiscreteOCP(
    step=integrator, N=4, stage_cost=ocp.Quadratic(np.zeros((1, 1)), np.eye(1)), terminal=ocp.TerminalEquality(np.array([2.0])), name="reach"
  )
  solution = Controller(problem, TIGHT).solve(np.array([0.0]))
  np.testing.assert_allclose(solution.us.ravel(), 0.5, atol=1e-6)  # the cheapest way to 2 in 4 steps, cost 4 * 0.5^2
  np.testing.assert_allclose(solution.cost, 1.0, atol=1e-6)  # no dt: a discrete map's costs are summed


@pytest.mark.method("opt.ipopt")
def test_a_soft_constraint_holds_when_it_can_and_gives_way_when_it_cannot() -> None:
  @sc.function(1, 1, output="g")
  def level(x, u):
    return x

  def solve(bound: float):
    problem = ocp.DiscreteOCP(
      step=integrator,
      N=3,
      stage_cost=ocp.Quadratic(np.eye(1), 0.1 * np.eye(1)),
      u_bounds=(-0.1, 0.1),
      constraints=[ocp.Path(level, hi=bound, soft=100.0)],
      name=f"soft_{int(bound * 10)}",
    )
    return Controller(problem, ocp.Direct(sc.opt.IPOPT(options={"tol": 1e-10}))).solve(np.array([1.0]))

  feasible, infeasible = solve(1.5), solve(0.5)
  assert feasible.slack is not None
  np.testing.assert_allclose(feasible.slack, 0.0, atol=1e-7)
  assert infeasible.status.ok and infeasible.slack is not None
  # x0 = 1 moves at most 0.1 a step: the violations are 0.5, 0.4, 0.3 and the slacks take them exactly.
  np.testing.assert_allclose(infeasible.slack, [0.5, 0.4, 0.3], atol=1e-6)


@pytest.mark.method("opt.piqp")
def test_state_bounds_hold_after_the_initial_state_and_a_control_reference() -> None:
  """From ``x0 = 1.8`` towards 3, with ``x <= 1.5``: the initial state is data and may lie outside, every
  later state stops at the bound. ``u`` is drawn to ``u_ref = 0.3`` where the state cost allows."""
  target = np.array([3.0])
  problem = ocp.DiscreteOCP(
    step=integrator,
    N=4,
    stage_cost=ocp.Quadratic(np.eye(1), 1e-3 * np.eye(1), x_ref=target),
    terminal_cost=ocp.Quadratic(np.eye(1), x_ref=target),  # without it nothing pulls the last state
    x_bounds=(None, 1.5),
    name="state_bounds",
  )
  solution = Controller(problem, TIGHT).solve(np.array([1.8]))
  np.testing.assert_allclose(solution.xs.ravel(), [1.8, 1.5, 1.5, 1.5, 1.5], atol=1e-6)
  steady = ocp.DiscreteOCP(
    step=integrator, N=3, stage_cost=ocp.Quadratic(np.zeros((1, 1)), np.eye(1), u_ref=np.array([0.3])), name="control_reference"
  )
  np.testing.assert_allclose(Controller(steady, ocp.Direct(sc.opt.PIQP(sparse=True))).solve(np.array([0.0])).us.ravel(), 0.3, atol=1e-6)


@pytest.mark.method("opt.ipopt")
def test_control_bounds_reach_every_pseudospectral_control() -> None:
  problem = library(ocp.Pseudospectral(4), name="bounded_nodes", track=False)
  controller = Controller(problem, ocp.Direct(sc.opt.IPOPT(options={"tol": 1e-10})))
  for start, side in ((X0, U_MAX), (-X0, -U_MAX)):  # the mirrored start saturates the other side
    solution = controller.solve(start, warm=controller.initial_guess(start))
    assert solution.zs is not None
    controls = solution.zs[:, problem.interval.state_times.size * NX :]
    assert np.abs(controls).max() <= U_MAX + 1e-6
    assert np.abs(controls - side).min() < 1e-3  # they reach the bound and stop there


# The tangent of a product or quotient with a constant leaves the constant's zero term out
# (``tests/core/ad/test_zero_tangent_products.py``), so the model stays provably affine to PIQP.
@sc.function(1, 1, output="xnext")
def _times_step(x, u):
  return x + 0.1 * u


@sc.function(1, 1, output="xnext")
def _over_step(x, u):
  return x + u / 10.0


@pytest.mark.method("opt.piqp")
@pytest.mark.parametrize("step", [_times_step, _over_step], ids=["times", "over"])
def test_an_ocp_takes_a_model_that_scales_its_control(step: sc.Function) -> None:
  # The OCP wraps the model in one more Function, so the scaling sits two calls below the constraint.
  problem = ocp.DiscreteOCP(
    step=step, N=4, stage_cost=ocp.Quadratic(np.zeros((1, 1)), np.eye(1)), terminal=ocp.TerminalEquality(np.array([2.0])), name=f"scaled_{step.name}"
  )
  solution = Controller(problem, TIGHT).solve(np.array([0.0]))
  np.testing.assert_allclose(solution.us.ravel(), 5.0, atol=1e-6)  # the cheapest way to 2 in 4 steps of 0.1 u
  np.testing.assert_allclose(solution.cost, 100.0, atol=1e-5)


@pytest.mark.method("opt.ipopt")
def test_the_condensed_form_of_a_continuous_model_rolls_out_its_shooting_method() -> None:
  # The condensed form eliminates the states by a scan of the transcription's own integrator method.
  # From near upright the problem is nearly linear, with one optimum both forms must reach.
  options, x0 = sc.opt.IPOPT(options={"tol": 1e-11}), np.array([0.05, 0.05, 0.0, 0.0])
  problem = library(ocp.MultipleShooting(si.RK4(steps=2)), name="rollout", track=False)
  sparse = Controller(problem, ocp.Direct(options)).solve(x0)
  condensed = Controller(problem, ocp.Direct(options, form="condensed")).solve(x0)
  assert sparse.status.ok and condensed.status.ok
  np.testing.assert_allclose(condensed.us, sparse.us, rtol=1e-6, atol=1e-7)
  np.testing.assert_allclose(condensed.xs, sparse.xs, rtol=1e-6, atol=1e-8)


A = np.array([[1.0, 0.1], [0.0, 1.0]])
B = np.array([[0.005], [0.1]])


@sc.function(2, 1, output="xnext")
def double(x, u):
  return sc.const(A) @ x + sc.const(B) @ u


@sc.function(2, 1, output="g")
def speed(x, u):
  return x[1:]


def _soft(name: str, horizon: int = 10) -> ocp.DiscreteOCP:
  return ocp.DiscreteOCP(
    step=double,
    N=horizon,
    stage_cost=ocp.Quadratic(np.eye(2), 0.1 * np.eye(1)),
    terminal_cost=ocp.Quadratic(10 * np.eye(2)),
    u_bounds=(-1, 1),
    constraints=[ocp.Path(speed, lo=-0.8, hi=0.8, soft=50.0)],
    name=name,
  )


def test_the_shift_moves_every_stage_block_up_one() -> None:
  problem = _soft("shift_layout", horizon=3)
  method = ocp.Direct("ipopt")
  nlp, layout = ocp.to_problem(problem)
  assert layout.var_names == ("xs", "us", "slack") and layout.var_sizes == (8, 3, 3)
  n_vars, n_eq, n_ineq = 14, nlp.n_eq, nlp.n_ineq
  assert (n_eq, n_ineq) == (2 + 3 * 2, 2 * 3) and method.warm_size(problem) == 2 * n_vars + n_eq + n_ineq
  guess = np.arange(method.warm_size(problem), dtype=float)
  moved = np.asarray(ocp.shift(problem, method)(guess))
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


def _law(problem: ocp.DiscreteOCP, method: ocp.Direct, name: str) -> sc.Function:
  """The control law as one Function: the solve, its first control and its shifted point."""
  solve, shift = ocp.solver(problem, method, name=name), ocp.shift(problem, method)

  @sc.function(sc.L("x0", problem.nx), sc.L("warm", method.warm_size(problem)), output=sc.G("u", "warm_next"), name=f"{name}_law")
  def law(x0, warm):
    _, us, point, _ = solve(x0, warm)
    return us[0], shift(point)

  return law


@pytest.mark.method("opt.ipopt")
def test_the_control_law_is_the_solve_and_its_shifted_warm_start() -> None:
  method = ocp.Direct(sc.opt.IPOPT(options={"tol": 1e-10}))
  problem = _soft("law_side")
  law, whole = _law(problem, method, "law_side"), Controller(problem, method, name="solve_side")
  x_law = x_solve = np.array([2.0, 0.0])
  warm = ocp.initial_guess(problem, method, x_law)
  for _ in range(4):
    u, warm = law(x_law, warm)
    solution = whole.solve(x_solve)
    np.testing.assert_allclose(u, solution.us[0], rtol=1e-9, atol=1e-10)
    assert whole.warm is not None
    np.testing.assert_allclose(warm, whole.warm, rtol=1e-9, atol=1e-9)
    x_law = A @ x_law + B @ u
    x_solve = A @ x_solve + B @ solution.us[0]


def test_the_law_generates_one_c_module_with_its_solver() -> None:
  problem = _soft("generated")
  method = ocp.Direct("ipopt")
  law = _law(problem, method, "generated")
  module = render_c_module(law)
  assert "generated_law" in module.body and "generated_solve" in module.body
  assert law.input_names == ("x0", "warm") and law.output_names == ("u", "warm_next")
  assert law.inputs[1].shape == (method.warm_size(problem),)


@pytest.mark.method("opt.piqp")
def test_a_closed_loop_reaches_the_origin_within_its_limits() -> None:
  controller = Controller(_soft("closed_loop", horizon=20), ocp.Direct(sc.opt.PIQP(sparse=True, options={"eps_abs": 1e-9, "eps_rel": 1e-9})))
  xs, us, statuses, _ = closed_loop(controller, lambda x, u: A @ x + B @ u, np.array([2.0, 0.0]), 80)
  assert xs.shape == (81, 2) and us.shape == (80, 1) and set(statuses) == {sc.Status.OK}
  assert np.abs(xs[-1]).max() < 1e-2 and np.abs(us).max() <= 1 + 1e-6
  assert np.abs(xs[:, 1]).max() < 0.8 + 1e-3  # the soft speed limit holds, since it can


@pytest.mark.method("opt.ipopt")
def test_a_continuous_plant_through_an_adaptive_map() -> None:
  @sc.function(2, 1, output="xdot")
  def pendulum(x, u):
    return sc.stack([x[1], -x[0].sin() - 0.1 * x[1] + u[0]])

  continuous = ocp.ContinuousOCP(
    pendulum,
    T=1.5,
    stage_cost=ocp.Quadratic(np.eye(2), 0.1 * np.eye(1)),
    terminal_cost=ocp.Quadratic(10 * np.eye(2)),
    u_bounds=(-2, 2),
    name="pendulum_mpc",
  )
  controller = Controller(ocp.transcribe(continuous, N=15), ocp.Direct("ipopt"))
  plant = si.adaptive(pendulum, dt=0.1, rtol=1e-10, atol=1e-12, name="pendulum_plant")
  xs, _, _, iterations = closed_loop(controller, plant, np.array([1.0, 0.0]), 40)
  assert np.abs(xs[-1]).max() < 1e-2 and np.all(iterations > 0)


@pytest.mark.method("opt.piqp")
def test_a_closed_loop_parameter_may_change_with_the_step() -> None:
  problem = ocp.DiscreteOCP(
    step=double, N=10, stage_cost=ocp.Quadratic(np.eye(2), 0.1 * np.eye(1), x_ref="target"), u_bounds=(-1, 1), name="moving_target"
  )
  controller = Controller(problem, ocp.Direct(sc.opt.PIQP(sparse=True)))
  xs, *_ = closed_loop(controller, lambda x, u: A @ x + B @ u, np.zeros(2), 120, target=lambda k: np.array([1.0 if k < 60 else -1.0, 0.0]))
  assert abs(xs[60, 0] - 1.0) < 0.1 and abs(xs[-1, 0] + 1.0) < 0.1  # the target the step gave is tracked


@sc.function(2, 1, output="xdot")
def double_ode(x, u):
  return sc.stack([x[1], u[0]])


def test_the_initial_guess() -> None:
  problem = ocp.transcribe(ocp.ContinuousOCP(double_ode, T=0.3, name="guess_layout"), ocp.Collocation(2), N=3)
  method = ocp.Direct("ipopt")
  guess = ocp.initial_guess(problem, method, np.array([1.0, 2.0]), np.array([0.5]))
  n_vars = method.layout(problem).n_vars
  np.testing.assert_array_equal(guess[:8], np.tile([1.0, 2.0], 4))
  np.testing.assert_array_equal(guess[8:11], [0.5] * 3)
  np.testing.assert_array_equal(guess[11:n_vars], np.tile([1.0, 2.0], 3))  # one internal state per Radau(2) interval
  assert not guess[n_vars:].any()


def test_what_the_direct_method_refuses() -> None:
  with pytest.raises(ValueError, match="form is 'sparse' or 'condensed'"):
    ocp.Direct(form="dense")  # ty: ignore[invalid-argument-type]
  collocated = ocp.transcribe(ocp.ContinuousOCP(double_ode, T=0.3), ocp.Collocation(2), N=3)
  with pytest.raises(ValueError, match="the condensed form takes a discrete map"):
    ocp.solver(collocated, ocp.Direct(form="condensed"))
  continuous = ocp.ContinuousOCP(double_ode, T=0.3)
  for build in (ocp.solver, ocp.shift):
    with pytest.raises(TypeError, match="is a ContinuousOCP, which no method solves: transcribe it first"):
      build(continuous, ocp.Direct())  # ty: ignore[invalid-argument-type]
  with pytest.raises(TypeError, match="ocp.direct solves DiscreteOCP, not ContinuousOCP"):
    ocp.REGISTRY.resolve(ocp.Direct(), continuous)  # the registry's own check, below the pointer to transcribe
