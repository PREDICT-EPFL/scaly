"""The discrete-time OCP builder: its two lowerings against hand-written problems, the stage form's
layout and Functions, a free final time through both lowerings, and what the builder refuses."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.function.model import as_concrete
from scaly.solvers.stages import StageLayout

A = np.array([[1.0, 0.1], [0.0, 1.0]])
B = np.array([[0.005], [0.1]])
X0 = np.array([1.0, -0.5])
PIQP = {"sparse": True, "eps_abs": 1e-10, "eps_rel": 1e-10}


@sc.function(sc.arg("x", 2), sc.arg("u", 1), outputs=sc.arg("x_next", 2))
def double_integrator(x: sc.Expr, u: sc.Expr) -> sc.Expr:
  return sc.const(A) @ x + sc.const(B) @ u


@sc.function(sc.arg("x", 2), sc.arg("u", 1), outputs=sc.arg("l", ()))
def effort(x: sc.Expr, u: sc.Expr) -> sc.Expr:
  return sc.sumsqr(x) + 0.1 * sc.sumsqr(u)


@sc.function(sc.arg("x", 2), outputs=sc.arg("vf", ()))
def final(x: sc.Expr) -> sc.Expr:
  return 10.0 * sc.sumsqr(x)


@sc.function(sc.arg("x", 1), sc.arg("u", 1), outputs=sc.arg("x_next", 1))
def integrator(x: sc.Expr, u: sc.Expr) -> sc.Expr:
  return x + u


def _solve(problem: sc.Problem, backend: str, guess: object, params: object, options: dict | None = None) -> tuple:
  solve = sc.solver(problem, backend, options=options)
  result = solve(params, x0=guess)
  assert solve.stats().to_solver_status().ok
  return result


@pytest.mark.solver("piqp")
def test_the_general_lowering_matches_an_unrolled_problem() -> None:
  n = 10
  ocp = sc.DiscreteOCP(
    N=n,
    dynamics=double_integrator,
    stage_cost=effort,
    terminal_cost=final,
    u_bounds=(-1.0, 1.0),
    x_bounds=([-5.0, -0.6], [5.0, 0.6]),
    name="unrolled_di",
  )

  @sc.problem(vars=sc.group(sc.arg("xs", (n + 1, 2)), sc.arg("us", (n, 1))), params=sc.arg("x_init", 2))
  def by_hand(variables: tuple[sc.Expr, sc.Expr], x_init: sc.Expr) -> sc.ProblemSpec[tuple[sc.Expr, sc.Expr]]:
    xs, us = variables
    cost = sum((sc.sumsqr(xs[k]) + 0.1 * sc.sumsqr(us[k]) for k in range(n)), start=10.0 * sc.sumsqr(xs[n]))
    lb, ub = np.tile([-5.0, -0.6], (n + 1, 1)), np.tile([5.0, 0.6], (n + 1, 1))
    lb[0], ub[0] = -np.inf, np.inf
    return sc.ProblemSpec(
      minimize=cost,
      eq=(xs[0] - x_init, *(xs[k + 1] - sc.const(A) @ xs[k] - sc.const(B) @ us[k] for k in range(n))),
      lb=(sc.const(lb), sc.const(np.full((n, 1), -1.0))),
      ub=(sc.const(ub), sc.const(np.full((n, 1), 1.0))),
    )

  (xs, us), *_ = _solve(by_hand, "piqp", (np.zeros((n + 1, 2)), np.zeros((n, 1))), X0, PIQP)
  variables, *_ = _solve(ocp.problem(), "piqp", ocp.pack(X0, 0.0), X0, PIQP)
  actual_xs, actual_us = ocp.unpack(variables)
  np.testing.assert_allclose(actual_xs, xs, atol=1e-7)
  np.testing.assert_allclose(actual_us, us, atol=1e-7)
  assert np.abs(actual_xs[1:, 1]).max() <= 0.6 + 1e-7  # the speed bound holds after the start


def test_the_stage_layout_and_variables() -> None:
  ocp = sc.DiscreteOCP(N=4, dynamics=double_integrator, stage_cost=effort, name="layout_di")
  general, staged = ocp.problem(), ocp.stage_problem()
  layout = StageLayout(
    nx=(2,) * 5, nu=(1,) * 4 + (0,), dynamics=tuple(tuple(range(2 + 2 * k, 4 + 2 * k)) for k in range(4)), eq=((0, 1), (), (), (), ()), ineq=((),) * 5
  )
  assert staged.stage_layout == layout and general.stage_layout is None
  assert as_concrete(sc.solver(general, "ipopt").function).descriptor.stage_layout == layout
  assert general.vars.names == staged.vars.names == ("z", "x_N")
  assert general.vars.shapes == ((4, 3), (2,)) and general.params.names == ("initial_state",)
  assert (general.n_eq, general.n_ineq) == (10, 0)


def _free_time(N: int = 20) -> sc.DiscreteOCP:
  @sc.function(sc.arg("x", 2), sc.arg("u", 1), sc.arg("T", ()), outputs=sc.arg("x_next", 2))
  def euler(x: sc.Expr, u: sc.Expr, T: sc.Expr) -> sc.Expr:
    return x + (T / N) * sc.stack([x[1], u[0]])

  @sc.function(sc.arg("x", 2), sc.arg("u", 1), sc.arg("T", ()), outputs=sc.arg("l", ()))
  def energy(x: sc.Expr, u: sc.Expr, T: sc.Expr) -> sc.Expr:
    return 0.1 * (T / N) * sc.sumsqr(u)

  @sc.function(sc.arg("x", 2), sc.arg("T", ()), outputs=sc.arg("time", ()))
  def time(x: sc.Expr, T: sc.Expr) -> sc.Expr:
    return T

  @sc.function(sc.arg("x", 2), outputs=sc.arg("error", 2))
  def arrive(x: sc.Expr) -> sc.Expr:
    return x - sc.const(np.array([1.0, 0.0]))

  return sc.DiscreteOCP(
    N=N,
    dynamics=euler,
    stage_cost=energy,
    terminal_cost=time,
    terminal=(sc.StageConstraint(arrive, lo=0.0, hi=0.0), sc.StageConstraint(time, lo=0.5)),
    u_bounds=(-1.0, 1.0),
    global_vars=("T",),
    name="free_time",
  )


@pytest.mark.solver("ipopt")
def test_a_free_final_time_reaches_the_same_solution_through_both_lowerings() -> None:
  ocp, n = _free_time(), 20
  general, staged = ocp.problem(), ocp.stage_problem()
  assert general.vars.names == ("z", "x_N", "T") and staged.vars.names == ("z", "x_N")
  assert staged.stage_layout == StageLayout(
    nx=(3,) * (n + 1),
    nu=(1,) * n + (0,),
    dynamics=tuple(tuple(range(2 + 3 * k, 5 + 3 * k)) for k in range(n)),
    eq=((0, 1),) + ((),) * n,
    ineq=((),) * n + ((staged.n_eq, staged.n_eq + 1, staged.n_eq + 2),),
  )
  descriptor = as_concrete(sc.solver(general, "ipopt").function).descriptor
  with pytest.raises(ValueError, match="stage form"):
    _ = descriptor.stage_layout

  start = np.zeros(2)
  options = {"tol": 1e-10, "print_level": 0, "sb": "yes"}
  shared = ocp.unpack(_solve(general, "ipopt", ocp.pack(start, 0.0, 3.0), start, options)[0])
  carried = ocp.unpack(_solve(staged, "ipopt", ocp.pack(start, 0.0, 3.0, stage_form=True), start, options)[0], stage_form=True)
  for left, right in zip(shared, carried, strict=True):
    np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
  xs, us, T = shared
  np.testing.assert_allclose(xs[-1], [1.0, 0.0], atol=1e-8)
  assert 2.0 <= float(T) < 2.5 and np.abs(us).max() > 0.99  # near the bang-bang minimum time of 2


def test_the_stage_functions_compute_every_row_of_the_stage_form() -> None:
  n = 3

  @sc.function(sc.arg("x", 2), sc.arg("u", 1), sc.arg("T", ()), sc.arg("gain", ()), outputs=sc.arg("x_next", 2))
  def step(x: sc.Expr, u: sc.Expr, T: sc.Expr, gain: sc.Expr) -> sc.Expr:
    return x + T * sc.stack([x[1].sin(), gain * u[0] - x[0] ** 3])

  @sc.function(sc.arg("x", 2), sc.arg("u", 1), sc.arg("ref", 2), outputs=sc.arg("l", ()))
  def track(x: sc.Expr, u: sc.Expr, ref: sc.Expr) -> sc.Expr:
    return sc.sumsqr(x - ref) + u[0] ** 4

  @sc.function(sc.arg("x", 2), sc.arg("T", ()), sc.arg("ref", 2), outputs=sc.arg("vf", ()))
  def arrive(x: sc.Expr, T: sc.Expr, ref: sc.Expr) -> sc.Expr:
    return T * sc.sumsqr(x - ref)

  @sc.function(sc.arg("x", 2), sc.arg("u", 1), outputs=sc.arg("g", 2))
  def corridor(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    return sc.stack([x[0] * u[0], x[1].cos()])

  @sc.function(sc.arg("x", 2), sc.arg("u", 1), sc.arg("gain", ()), outputs=sc.arg("g", ()))
  def power(x: sc.Expr, u: sc.Expr, gain: sc.Expr) -> sc.Expr:
    return gain * x[1] * u[0]

  @sc.function(sc.arg("x", 2), sc.arg("T", ()), outputs=sc.arg("g", ()))
  def duration(x: sc.Expr, T: sc.Expr) -> sc.Expr:
    return T + x[0]

  ocp = sc.DiscreteOCP(
    N=n,
    dynamics=step,
    stage_cost=track,
    terminal_cost=arrive,
    path=(sc.StageConstraint(corridor, hi=1.0), sc.StageConstraint(power, lo=-2.0, hi=2.0)),
    terminal=(sc.StageConstraint(duration, lo=0.0),),
    x_bounds=(-5.0, [4.0, 6.0]),
    u_bounds=(None, 1.0),
    stage_params=("ref",),
    global_vars=("T",),
    name="contract",
  )
  staged = ocp.stage_problem()
  layout, functions = staged.stage_layout, staged.stage_functions
  assert layout is not None and functions is not None
  assert staged.params.names == ("initial_state", "gain", "ref") and functions.stage_params == ("ref",)
  assert functions.path is not None and functions.terminal is not None and functions.cost is not None and functions.terminal_cost is not None

  rng = np.random.default_rng(0)
  z, x_last = rng.normal(size=(n, 4)), rng.normal(size=3)
  initial, gain, ref = rng.normal(size=2), np.array(0.7), rng.normal(size=(n + 1, 2))
  descriptor = as_concrete(sc.solver(staged, "ipopt").function).descriptor
  assert descriptor.base is not None
  f, g = descriptor.base(np.concatenate([z.ravel(), x_last]), (initial, gain, ref))
  # The bounds leave x_0 and the carried global variable free and bound u at every stage.
  assert descriptor.bounds is not None
  x_lb, x_ub, _, _ = descriptor.bounds(initial, gain, ref)
  inf = np.inf
  np.testing.assert_array_equal(x_lb, [-inf, -inf, -inf, -inf, *[-5.0, -5.0, -inf, -inf] * (n - 1), -5.0, -5.0, -inf])
  np.testing.assert_array_equal(x_ub, [inf, inf, inf, 1.0, *[4.0, 6.0, inf, 1.0] * (n - 1), 4.0, 6.0, inf])

  states, controls = [*z[:, :3], x_last], z[:, 3:]
  expected_f = sum(float(functions.cost(states[k], controls[k], gain, ref[k])) for k in range(n)) + float(
    functions.terminal_cost(x_last, gain, ref[n])
  )
  np.testing.assert_allclose(f, expected_f, rtol=1e-12)
  np.testing.assert_allclose(g[list(layout.eq[0])], functions.initial(states[0], initial), rtol=1e-12)
  for k in range(n):
    np.testing.assert_allclose(g[list(layout.dynamics[k])], functions.dynamics(states[k], controls[k], gain, ref[k]) - states[k + 1], rtol=1e-12)
    np.testing.assert_allclose(g[list(layout.ineq[k])], functions.path(states[k], controls[k], gain, ref[k]), rtol=1e-12)
  np.testing.assert_allclose(g[list(layout.ineq[n])], functions.terminal(x_last, gain, ref[n]), rtol=1e-12)
  # Mapped over the stages, with the global parameters broadcast and the per-stage ones sliced.
  mapped = sc.vmap(functions.dynamics, n)(np.stack(states[:n]), controls, sc.broadcast(gain), ref[:n])
  np.testing.assert_allclose(mapped, [functions.dynamics(states[k], controls[k], gain, ref[k]) for k in range(n)], rtol=1e-12)
  # The state carries the global variable unchanged.
  np.testing.assert_allclose(np.asarray(mapped)[:, 2], z[:, 2])


def test_pack_and_unpack() -> None:
  ocp, n = _free_time(4), 4
  xs, us, T = np.arange(10.0).reshape(5, 2), np.arange(4.0).reshape(4, 1), np.array(2.5)
  z, x_last, packed_T = ocp.pack(xs, us, T)
  np.testing.assert_array_equal(z, np.hstack([xs[:n], us]))
  np.testing.assert_array_equal(x_last, xs[n])
  for got, want in zip(ocp.unpack((z, x_last, packed_T)), (xs, us, T), strict=True):
    np.testing.assert_array_equal(got, want)
  z, x_last = ocp.pack(xs, us, T, stage_form=True)
  np.testing.assert_array_equal(z, np.hstack([xs[:n], np.full((n, 1), 2.5), us]))
  np.testing.assert_array_equal(x_last, [*xs[n], 2.5])
  for got, want in zip(ocp.unpack((z, x_last), stage_form=True), (xs, us, T), strict=True):
    np.testing.assert_array_equal(got, want)
  with pytest.raises(ValueError, match="1 global variable"):
    ocp.pack(xs, us)


# Ported from origin/devrush tests/ocp/test_direct.py, "a varying reference is read stage by stage".
@pytest.mark.solver("piqp")
def test_a_stage_parameter_is_read_stage_by_stage() -> None:
  """``x+ = x + u`` tracking ``r_k`` from ``x0 = r_0`` can reach every reference exactly, at zero cost."""

  @sc.function(sc.arg("x", 1), sc.arg("u", 1), sc.arg("r", 1), outputs=sc.arg("l", ()))
  def follow(x: sc.Expr, u: sc.Expr, r: sc.Expr) -> sc.Expr:
    return sc.sumsqr(x - r) + 1e-6 * sc.sumsqr(u)

  @sc.function(sc.arg("x", 1), sc.arg("r", 1), outputs=sc.arg("vf", ()))
  def end(x: sc.Expr, r: sc.Expr) -> sc.Expr:
    return sc.sumsqr(x - r)

  ref = np.array([0.0, 0.5, 0.2, 1.0, 1.3, 0.7, 0.9])
  ocp = sc.DiscreteOCP(N=6, dynamics=integrator, stage_cost=follow, terminal_cost=end, stage_params=("r",), name="tracking")
  problem = ocp.problem()
  assert problem.params.names == ("initial_state", "r") and problem.params.shapes == ((1,), (7, 1))
  xs, us = ocp.unpack(_solve(problem, "piqp", ocp.pack(0.0, 0.0), (np.zeros(1), ref[:, None]), PIQP)[0])
  np.testing.assert_allclose(xs.ravel(), ref, atol=1e-5)
  np.testing.assert_allclose(us.ravel(), np.diff(ref), atol=1e-5)


# Ported from origin/devrush tests/ocp/test_direct.py, "parameters are the union of what the functions name".
@pytest.mark.solver("ipopt")
def test_parameters_are_the_union_of_what_the_functions_name() -> None:
  @sc.function(sc.arg("x", 2), sc.arg("u", 1), sc.arg("stiffness", ()), outputs=sc.arg("x_next", 2))
  def spring(x: sc.Expr, u: sc.Expr, stiffness: sc.Expr) -> sc.Expr:
    return x + 0.1 * sc.stack([x[1], -stiffness * x[0] + u[0]])

  @sc.function(sc.arg("x", 2), sc.arg("u", 1), sc.arg("target", 2), outputs=sc.arg("l", ()))
  def tracking(x: sc.Expr, u: sc.Expr, target: sc.Expr) -> sc.Expr:
    return sc.sumsqr(x - target) + 0.01 * sc.sumsqr(u)

  @sc.function(sc.arg("x", 2), sc.arg("target", 2), outputs=sc.arg("error", 2))
  def reach(x: sc.Expr, target: sc.Expr) -> sc.Expr:
    return x - target

  ocp = sc.DiscreteOCP(N=10, dynamics=spring, stage_cost=tracking, terminal=(sc.StageConstraint(reach, lo=0.0, hi=0.0),), name="spring_ocp")
  problem = ocp.problem()
  assert [(name, shape) for name, shape in zip(problem.params.names, problem.params.shapes, strict=True)] == [
    ("initial_state", (2,)),
    ("stiffness", ()),
    ("target", (2,)),
  ]
  start, target = np.array([1.0, 0.0]), np.array([0.2, 0.0])
  options = {"tol": 1e-10, "print_level": 0, "sb": "yes"}
  xs, us = ocp.unpack(_solve(problem, "ipopt", ocp.pack(start, 0.0), (start, np.array(4.0), target), options)[0])
  np.testing.assert_allclose(xs[-1], target, atol=1e-8)  # the terminal constraint, at the parameter
  _, stiffer = ocp.unpack(_solve(problem, "ipopt", ocp.pack(start, 0.0), (start, np.array(9.0), target), options)[0])
  assert np.abs(stiffer - us).max() > 1e-3  # the model reads its parameter


@sc.function(sc.arg("x", 1), sc.arg("u", 1), outputs=sc.arg("l", ()))
def _control_effort(x: sc.Expr, u: sc.Expr) -> sc.Expr:
  return sc.sumsqr(u)


@sc.function(sc.arg("x", 1), outputs=sc.arg("error", 1))
def _at_two(x: sc.Expr) -> sc.Expr:
  return x - 2.0


# Ported from origin/devrush tests/ocp/test_direct.py, "a discrete map sums its costs and the terminal equality holds".
@pytest.mark.solver("piqp")
def test_a_discrete_map_sums_its_costs_and_the_terminal_constraint_holds() -> None:
  ocp = sc.DiscreteOCP(N=4, dynamics=integrator, stage_cost=_control_effort, terminal=(sc.StageConstraint(_at_two, lo=0.0, hi=0.0),), name="reach")
  solve = sc.solver(ocp.problem(), "piqp", options=PIQP)
  variables, *_ = solve(np.zeros(1))
  assert solve.stats().to_solver_status().ok
  _, us = ocp.unpack(variables)
  np.testing.assert_allclose(us.ravel(), 0.5, atol=1e-6)  # the cheapest way to 2 in 4 steps, cost 4 * 0.5^2
  np.testing.assert_allclose(solve.stats().obj, 1.0, atol=1e-6)  # a discrete map's costs are summed


# Ported from origin/devrush tests/ocp/test_direct.py, "state bounds hold after the initial state and a control reference".
@pytest.mark.solver("piqp")
def test_state_bounds_hold_after_the_initial_state() -> None:
  """From ``x0 = 1.8`` towards 3, with ``x <= 1.5``: the initial state is data and may lie outside, every
  later state stops at the bound."""

  @sc.function(sc.arg("x", 1), sc.arg("u", 1), outputs=sc.arg("l", ()))
  def towards_three(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    return sc.sumsqr(x - 3.0) + 1e-3 * sc.sumsqr(u)

  @sc.function(sc.arg("x", 1), outputs=sc.arg("vf", ()))
  def end(x: sc.Expr) -> sc.Expr:
    return sc.sumsqr(x - 3.0)

  ocp = sc.DiscreteOCP(N=4, dynamics=integrator, stage_cost=towards_three, terminal_cost=end, x_bounds=(None, 1.5), name="state_bounds")
  xs, _ = ocp.unpack(_solve(ocp.problem(), "piqp", ocp.pack(1.8, 0.0), np.array([1.8]), PIQP)[0])
  np.testing.assert_allclose(xs.ravel(), [1.8, 1.5, 1.5, 1.5, 1.5], atol=1e-6)


@sc.function(sc.arg("x", 1), sc.arg("u", 1), outputs=sc.arg("x_next", 1))
def _times_step(x: sc.Expr, u: sc.Expr) -> sc.Expr:
  return x + 0.1 * u


@sc.function(sc.arg("x", 1), sc.arg("u", 1), outputs=sc.arg("x_next", 1))
def _over_step(x: sc.Expr, u: sc.Expr) -> sc.Expr:
  return x + u / 10.0


# Ported from origin/devrush tests/ocp/test_direct.py, "an ocp takes a model that scales its control".
@pytest.mark.solver("piqp")
@pytest.mark.parametrize("step", [_times_step, _over_step], ids=["times", "over"])
def test_an_ocp_takes_a_model_that_scales_its_control(step: sc.Function) -> None:
  # The lowering wraps the model in a mapped call, so PIQP's proof that the problem is quadratic sees through it.
  ocp = sc.DiscreteOCP(
    N=4, dynamics=step, stage_cost=_control_effort, terminal=(sc.StageConstraint(_at_two, lo=0.0, hi=0.0),), name=f"scaled{step.name}"
  )
  solve = sc.solver(ocp.problem(), "piqp", options=PIQP)
  variables, *_ = solve(np.zeros(1))
  assert solve.stats().to_solver_status().ok
  np.testing.assert_allclose(ocp.unpack(variables)[1].ravel(), 5.0, atol=1e-6)  # the cheapest way to 2 in 4 steps of 0.1 u
  np.testing.assert_allclose(solve.stats().obj, 100.0, atol=1e-5)


NX, NU, N, DT = 4, 1, 20, 0.05
M_CART, M_POLE, LENGTH, G = 1.0, 0.3, 0.5, 9.81
U_MAX, P_MIN, P_MAX = 15.0, -0.05, 0.35
Q = np.array([2.0, 20.0, 0.1, 0.1])
R, QN = 0.02, 10.0
CART_X0 = np.array([0.3, 0.4, 0.0, 0.0])


def _cartpole(x: sc.Expr, u: sc.Expr) -> sc.Expr:
  theta, pd, td = x[1], x[2], x[3]
  s, c = theta.sin(), theta.cos()
  den = M_CART + M_POLE * s * s
  pdd = (u[0] + M_POLE * LENGTH * td * td * s - M_POLE * G * s * c) / den
  tdd = (G * s * (M_CART + M_POLE) - c * (u[0] + M_POLE * LENGTH * td * td * s)) / (LENGTH * den)
  return sc.stack([pd, td, pdd, tdd])


def _rk4(x: sc.Expr, u: sc.Expr, h: float) -> sc.Expr:
  k1 = _cartpole(x, u)
  k2 = _cartpole(x + 0.5 * h * k1, u)
  k3 = _cartpole(x + 0.5 * h * k2, u)
  k4 = _cartpole(x + h * k3, u)
  return x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def _error(x: sc.Expr) -> sc.Expr:
  return sc.stack([x[0], 1.0 - x[1].cos(), x[2], x[3]])


# Ported from origin/devrush tests/ocp/test_direct.py, "the library transcribes the cart pole as by hand",
# as a discrete map with a hard track limit: transcriptions and soft constraints are not part of the builder.
@pytest.mark.solver("ipopt")
def test_the_builder_states_the_cart_pole_as_by_hand() -> None:
  @sc.function(sc.arg("x", NX), sc.arg("u", NU), outputs=sc.arg("x_next", NX))
  def step(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    return _rk4(_rk4(x, u, DT / 2), u, DT / 2)

  @sc.function(sc.arg("x", NX), sc.arg("u", NU), outputs=sc.arg("l", ()))
  def stage(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    e = _error(x)
    return DT * ((sc.const(Q) * e * e).sum() + R * u[0] * u[0])

  @sc.function(sc.arg("x", NX), outputs=sc.arg("vf", ()))
  def terminal(x: sc.Expr) -> sc.Expr:
    e = _error(x)
    return QN * (sc.const(Q) * e * e).sum()

  @sc.function(sc.arg("x", NX), sc.arg("u", NU), outputs=sc.arg("p", 1))
  def position(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    return x[:1]

  @sc.problem(vars=sc.group(sc.arg("xs", (N + 1) * NX), sc.arg("us", N * NU)), params=sc.arg("x0", NX), name="hand_cartpole")
  def by_hand(variables: tuple[sc.Expr, sc.Expr], x0: sc.Expr) -> sc.ProblemSpec[tuple[sc.Expr, sc.Expr]]:
    xs, us = variables
    states = [xs[k * NX : (k + 1) * NX] for k in range(N + 1)]
    cost = sum((stage(states[k], us[k : k + 1]) for k in range(N)), start=terminal(states[N]))
    return sc.ProblemSpec(
      minimize=cost,
      eq=(states[0] - x0, *(step(states[k], us[k : k + 1]) - states[k + 1] for k in range(N))),
      ineq=(sc.bounded(sc.stack([states[k][0] for k in range(N)]), lo=P_MIN, hi=P_MAX),),
      lb=(sc.NO_LB, sc.const(-U_MAX)),
      ub=(sc.NO_UB, sc.const(U_MAX)),
    )

  ocp = sc.DiscreteOCP(
    N=N,
    dynamics=step,
    stage_cost=stage,
    terminal_cost=terminal,
    path=(sc.StageConstraint(position, lo=P_MIN, hi=P_MAX),),
    u_bounds=(-U_MAX, U_MAX),
    name="lib_cartpole",
  )
  options = {"tol": 1e-12, "print_level": 0, "sb": "yes"}
  (xs, us), *_ = _solve(by_hand, "ipopt", (np.tile(CART_X0, N + 1), np.zeros(N)), CART_X0, options)
  problem = ocp.problem()
  assert (problem.n_eq, problem.n_ineq) == (by_hand.n_eq, by_hand.n_ineq)
  actual_xs, actual_us = ocp.unpack(_solve(problem, "ipopt", ocp.pack(CART_X0, 0.0), CART_X0, options)[0])
  np.testing.assert_allclose(actual_xs.ravel(), xs, rtol=1e-9, atol=1e-10)
  np.testing.assert_allclose(actual_us.ravel(), us, rtol=1e-9, atol=1e-10)
  assert actual_xs[:, 0].min() < P_MIN + 1e-6  # the track limit binds


# Ported from origin/devrush tests/ocp/test_problem.py, "what an ocp refuses", for the discrete builder.
def test_what_an_ocp_refuses() -> None:
  with pytest.raises(ValueError, match="N must be a positive integer"):
    sc.DiscreteOCP(N=0, dynamics=double_integrator)
  assert sc.DiscreteOCP(N=np.int64(3), dynamics=double_integrator).problem().vars.shapes == ((3, 3), (2,))  # ty: ignore[invalid-argument-type]
  with pytest.raises(TypeError, match="take StageConstraint values"):
    sc.DiscreteOCP(N=3, dynamics=double_integrator, path=[effort])  # ty: ignore[invalid-argument-type]
  with pytest.raises(ValueError, match=r"x_bounds has shape \(3,\), expected a number or \(2,\)"):
    sc.DiscreteOCP(N=3, dynamics=double_integrator, x_bounds=(np.zeros(3), None))
  with pytest.raises(TypeError, match="a bound is a number or an array"):
    sc.DiscreteOCP(N=3, dynamics=double_integrator, terminal=[sc.StageConstraint(final, hi=sc.const(1.0))])

  @sc.function(sc.arg("u", 1), sc.arg("x", 1), outputs=sc.arg("l", ()))
  def swapped(u: sc.Expr, x: sc.Expr) -> sc.Expr:
    return sc.sumsqr(x) + sc.sumsqr(u)

  with pytest.raises(ValueError, match="swapped takes the state first"):
    sc.DiscreteOCP(N=3, dynamics=integrator, stage_cost=swapped)

  @sc.function(sc.arg("x", 2), outputs=sc.arg("x_next", 2))
  def one_input(x: sc.Expr) -> sc.Expr:
    return x

  with pytest.raises(ValueError, match="takes the state, then the control"):
    sc.DiscreteOCP(N=3, dynamics=one_input)

  @sc.function(sc.arg("x", 2), sc.arg("u", 1), outputs=sc.arg("x_next", 3))
  def grows(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    return sc.concat([x, u])

  with pytest.raises(ValueError, match="returns the next state"):
    sc.DiscreteOCP(N=3, dynamics=grows)

  @sc.function(sc.arg("x", 3), sc.arg("u", 1), outputs=sc.arg("l", ()))
  def wide(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    return x.sum()

  with pytest.raises(ValueError, match="wide takes the state"):
    sc.DiscreteOCP(N=3, dynamics=double_integrator, stage_cost=wide)

  @sc.function(sc.arg("x", 2), sc.arg("u", 1), outputs=sc.arg("l", 2))
  def vector_cost(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    return x

  with pytest.raises(ValueError, match="one value"):
    sc.DiscreteOCP(N=3, dynamics=double_integrator, stage_cost=vector_cost)

  @sc.function(sc.arg("x", 2), sc.arg("u", 1), sc.arg("stiffness", 3), outputs=sc.arg("l", ()))
  def clash(x: sc.Expr, u: sc.Expr, stiffness: sc.Expr) -> sc.Expr:
    return x.sum()

  @sc.function(sc.arg("x", 2), sc.arg("u", 1), sc.arg("stiffness", ()), outputs=sc.arg("x_next", 2))
  def stiff(x: sc.Expr, u: sc.Expr, stiffness: sc.Expr) -> sc.Expr:
    return stiffness * x

  with pytest.raises(ValueError, match="'stiffness' is"):
    sc.DiscreteOCP(N=3, dynamics=stiff, stage_cost=clash)

  @sc.function(sc.arg("x", 2), sc.arg("u", 1), sc.arg("initial_state", 2), outputs=sc.arg("x_next", 2))
  def taken(x: sc.Expr, u: sc.Expr, initial_state: sc.Expr) -> sc.Expr:
    return x + initial_state

  with pytest.raises(ValueError, match="may not be named 'initial_state'"):
    sc.DiscreteOCP(N=3, dynamics=taken)
  with pytest.raises(ValueError, match="stage_params names 'r'"):
    sc.DiscreteOCP(N=3, dynamics=double_integrator, stage_params=("r",))
  with pytest.raises(ValueError, match="both a stage parameter and a global variable"):
    sc.DiscreteOCP(N=3, dynamics=stiff, stage_params=("stiffness",), global_vars=("stiffness",))
  with pytest.raises(ValueError, match="needs at least one of lo / hi"):
    sc.StageConstraint(final)
