"""OCPs: the library's transcription of a cart-pole against a hand-written one, the transcriptions
against each other, parameters by name, costs, terminal equalities and soft constraints."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly import integrators as si
from scaly import mpc

NX, NU, N, DT = 4, 1, 20, 0.05
M_CART, M_POLE, LENGTH, G = 1.0, 0.3, 0.5, 9.81
U_MAX, P_MAX, SOFT = 15.0, 0.35, 1e3
Q = np.array([2.0, 20.0, 0.1, 0.1])
R, QN = 0.02, 10.0
X0 = np.array([0.3, 0.4, 0.0, 0.0])


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
  """The formulation of `examples/nmpc_cartpole.py`, written out without `scaly.mpc` or
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


def library(transcription=None, name="lib_cartpole", *, track: bool = True, **kw) -> mpc.OCP:
  return mpc.OCP(
    ode=cartpole,
    dt=DT,
    horizon=N,
    transcription=transcription or si.MultipleShooting(si.rk4, steps=2),
    stage_cost=stage,
    terminal_cost=terminal,
    u_bounds=(-U_MAX, U_MAX),
    constraints=[mpc.Path(position, lo=-P_MAX, hi=P_MAX, soft=SOFT)] if track else [],
    name=name,
    **kw,
  )


@pytest.mark.solver("ipopt")
def test_the_library_transcribes_the_cart_pole_as_by_hand() -> None:
  options = {"tol": 1e-12}
  hand = sc.opt.solver(hand_written(), sc.opt.IPOPT(options=options), name="hand_ipopt")
  n_eq, n_ineq = (N + 1) * NX, 2 * N
  guess = (np.tile(X0, N + 1), np.zeros(N), np.zeros(N))
  (xs, us, slack), *_ = hand(guess, tuple(np.zeros(g.size) for g in guess), np.zeros(n_eq), np.zeros(n_ineq), X0)
  ocp = library()
  assert ocp.layout.var_names == ("xs", "us", "slack") and ocp.problem.n_eq == n_eq and ocp.problem.n_ineq == n_ineq
  solution = mpc.MPC(ocp, "ipopt", options=options).solve(X0)
  assert solution.status.ok
  np.testing.assert_allclose(solution.xs.ravel(), xs, rtol=1e-9, atol=1e-10)
  np.testing.assert_allclose(solution.us.ravel(), us, rtol=1e-9, atol=1e-10)
  assert solution.slack is not None
  np.testing.assert_allclose(solution.slack, slack, atol=1e-9)
  assert solution.slack is not None and solution.slack.max() > 0.1  # the track limit gives way on the way up
  assert np.all(solution.slack >= -1e-7)  # IPOPT relaxes bounds by 1e-8


@pytest.mark.solver("ipopt")
def test_shooting_and_collocation_solve_the_same_problem_and_pseudospectral_improves_on_it() -> None:
  options = {"tol": 1e-11}
  kw = {"cost": "integral", "track": False}  # a path constraint binds only at the grid points, which pseudospectral controls can dodge
  shooting = mpc.MPC(library(si.MultipleShooting(si.rk4, steps=4), name="agree_shooting", **kw), "ipopt", options=options).solve(X0)
  collocation = mpc.MPC(library(si.Collocation(3), name="agree_collocation", **kw), "ipopt", options=options).solve(X0)
  pseudo = mpc.MPC(library(si.Pseudospectral(4), name="agree_pseudo", track=False), "ipopt", options=options).solve(X0)
  assert shooting.status.ok and collocation.status.ok and pseudo.status.ok
  # The same held controls, integrated two ways: the solutions differ by the integrators' errors.
  np.testing.assert_allclose(collocation.us, shooting.us, rtol=1e-3, atol=1e-4)
  np.testing.assert_allclose(collocation.xs, shooting.xs, rtol=1e-3, atol=1e-5)
  assert pseudo.cost <= shooting.cost * (1 + 1e-6)  # controls free inside each interval can only do better
  assert pseudo.cost > 0.95 * shooting.cost


@sc.function(1, 1, output="xnext")
def integrator(x, u):
  return x + u


@pytest.mark.solver("piqp")
def test_a_varying_reference_is_read_stage_by_stage() -> None:
  """``x+ = x + u`` tracking ``r_k`` from ``x0 = r_0`` can reach every reference exactly, at zero cost."""
  horizon = 6
  ref = np.array([0.0, 0.5, 0.2, 1.0, 1.3, 0.7, 0.9])
  ocp = mpc.OCP(
    step=integrator,
    horizon=horizon,
    stage_cost=mpc.Quadratic(np.eye(1), 1e-6 * np.eye(1), x_ref="r"),
    terminal_cost=mpc.Quadratic(np.eye(1), x_ref="r"),
    varying=("r",),
    name="tracking",
  )
  assert [p.name for p in ocp.params] == ["r"]
  solution = mpc.MPC(ocp, "piqp", options={"eps_abs": 1e-10, "eps_rel": 1e-10}).solve(np.array([0.0]), r=ref)
  np.testing.assert_allclose(solution.xs.ravel(), ref, atol=1e-5)
  np.testing.assert_allclose(solution.us.ravel(), np.diff(ref), atol=1e-5)


@sc.function(2, 1, (), output="xdot")
def spring(x, u, stiffness):
  return sc.stack([x[1], -stiffness * x[0] + u[0]])


@sc.function(2, 1, 2, output="l")
def tracking(x, u, target):
  e = x - target
  return (e * e).sum() + 0.01 * u[0] * u[0]


@pytest.mark.solver("ipopt")
def test_parameters_are_the_union_of_what_the_functions_name() -> None:
  ocp = mpc.OCP(ode=spring, dt=0.1, horizon=10, stage_cost=tracking, terminal=mpc.TerminalEquality(x_ref="target"), name="spring_ocp")
  assert [(p.name, p.type.shape) for p in ocp.params] == [("stiffness", ()), ("target", (2,))]
  controller = mpc.MPC(ocp, "ipopt", options={"tol": 1e-10})
  solution = controller.solve(np.array([1.0, 0.0]), stiffness=np.array(4.0), target=np.array([0.2, 0.0]))
  assert solution.status.ok
  np.testing.assert_allclose(solution.xs[-1], [0.2, 0.0], atol=1e-8)  # the terminal equality, at the parameter
  stiffer = controller.solve(
    np.array([1.0, 0.0]), guess=controller.initial_guess(np.array([1.0, 0.0])), stiffness=np.array(9.0), target=np.array([0.2, 0.0])
  )
  assert np.abs(stiffer.us - solution.us).max() > 1e-3  # the model reads its parameter
  with pytest.raises(TypeError, match=r"takes the parameters \['stiffness', 'target'\]"):
    controller.solve(np.array([1.0, 0.0]), stiffness=np.array(4.0))


@pytest.mark.solver("piqp")
def test_a_discrete_map_sums_its_costs_and_the_terminal_equality_holds() -> None:
  ocp = mpc.OCP(
    step=integrator, horizon=4, stage_cost=mpc.Quadratic(np.zeros((1, 1)), np.eye(1)), terminal=mpc.TerminalEquality(np.array([2.0])), name="reach"
  )
  solution = mpc.MPC(ocp, "piqp", options={"eps_abs": 1e-10, "eps_rel": 1e-10}).solve(np.array([0.0]))
  np.testing.assert_allclose(solution.us.ravel(), 0.5, atol=1e-6)  # the cheapest way to 2 in 4 steps, cost 4 * 0.5^2
  np.testing.assert_allclose(solution.cost, 1.0, atol=1e-6)  # no dt: a discrete map's costs are summed


@pytest.mark.solver("ipopt")
def test_a_soft_constraint_holds_when_it_can_and_gives_way_when_it_cannot() -> None:
  @sc.function(1, 1, output="g")
  def level(x, u):
    return x

  def solve(bound: float) -> mpc.Solution:
    ocp = mpc.OCP(
      step=integrator,
      horizon=3,
      stage_cost=mpc.Quadratic(np.eye(1), 0.1 * np.eye(1)),
      u_bounds=(-0.1, 0.1),
      constraints=[mpc.Path(level, hi=bound, soft=100.0)],
      name=f"soft_{int(bound * 10)}",
    )
    return mpc.MPC(ocp, "ipopt", options={"tol": 1e-10}).solve(np.array([1.0]))

  feasible, infeasible = solve(1.5), solve(0.5)
  assert feasible.slack is not None
  np.testing.assert_allclose(feasible.slack, 0.0, atol=1e-7)
  assert infeasible.status.ok and infeasible.slack is not None
  # x0 = 1 moves at most 0.1 a step: the violations are 0.5, 0.4, 0.3 and the slacks take them exactly.
  np.testing.assert_allclose(infeasible.slack, [0.5, 0.4, 0.3], atol=1e-6)


def test_refusals() -> None:
  with pytest.raises(ValueError, match="exactly one of ode="):
    mpc.OCP(horizon=3)
  with pytest.raises(ValueError, match="exactly one of ode="):
    mpc.OCP(ode=cartpole, step=integrator, horizon=3, dt=0.1)
  with pytest.raises(ValueError, match="needs a positive dt"):
    mpc.OCP(ode=cartpole, horizon=3)
  with pytest.raises(ValueError, match="horizon must be a positive integer"):
    mpc.OCP(step=integrator, horizon=0)
  with pytest.raises(ValueError, match="a transcription goes with ode="):
    mpc.OCP(step=integrator, horizon=3, transcription=si.Collocation())
  with pytest.raises(ValueError, match="cost is 'points', or 'integral'"):
    mpc.OCP(step=integrator, horizon=3, cost="integral")
  with pytest.raises(ValueError, match="Q must be 1x1"):
    mpc.OCP(step=integrator, horizon=3, stage_cost=mpc.Quadratic(np.eye(2)))
  with pytest.raises(ValueError, match="varying names"):
    mpc.OCP(step=integrator, horizon=3, varying=("r",))

  @sc.function(1, 1, 3, output="l")
  def clash(x, u, stiffness):
    return (x * x).sum()

  with pytest.raises(ValueError, match="parameter 'stiffness' is"):
    mpc.OCP(ode=spring, dt=0.1, horizon=3, stage_cost=clash)

  @sc.function(1, 1, 1, output="xnext")
  def taken(x, u, x0):
    return x + u + x0

  with pytest.raises(ValueError, match="may not be named 'x0'"):
    mpc.OCP(step=taken, horizon=3)


@pytest.mark.solver("piqp")
def test_state_bounds_hold_after_the_initial_state_and_a_control_reference() -> None:
  """From ``x0 = 1.8`` towards 3, with ``x <= 1.5``: the initial state is data and may lie outside, every
  later state stops at the bound. ``u`` is drawn to ``u_ref = 0.3`` where the state cost allows."""
  target = np.array([3.0])
  ocp = mpc.OCP(
    step=integrator,
    horizon=4,
    stage_cost=mpc.Quadratic(np.eye(1), 1e-3 * np.eye(1), x_ref=target),
    terminal_cost=mpc.Quadratic(np.eye(1), x_ref=target),  # without it nothing pulls the last state
    x_bounds=(None, 1.5),
    name="state_bounds",
  )
  solution = mpc.MPC(ocp, "piqp", options={"eps_abs": 1e-10, "eps_rel": 1e-10}).solve(np.array([1.8]))
  np.testing.assert_allclose(solution.xs.ravel(), [1.8, 1.5, 1.5, 1.5, 1.5], atol=1e-6)
  steady = mpc.OCP(step=integrator, horizon=3, stage_cost=mpc.Quadratic(np.zeros((1, 1)), np.eye(1), u_ref=np.array([0.3])), name="control_reference")
  np.testing.assert_allclose(mpc.MPC(steady, "piqp").solve(np.array([0.0])).us.ravel(), 0.3, atol=1e-6)


@pytest.mark.solver("ipopt")
def test_control_bounds_reach_every_pseudospectral_control() -> None:
  ocp = library(si.Pseudospectral(4), name="bounded_nodes", track=False)
  controller = mpc.MPC(ocp, "ipopt", options={"tol": 1e-10})
  for start, side in ((X0, U_MAX), (-X0, -U_MAX)):  # the mirrored start saturates the other side
    solution = controller.solve(start, guess=controller.initial_guess(start))
    controls = solution.zs[:, ocp.interval.state_times.size * NX :]  # ty: ignore[not-subscriptable]
    assert np.abs(controls).max() <= U_MAX + 1e-6
    assert np.abs(controls - side).min() < 1e-3  # they reach the bound and stop there


# The tangent of a product or quotient with a constant leaves the constant's zero term out
# (``tests/ad/test_zero_tangent_products.py``), so the model stays provably affine to PIQP.
@sc.function(1, 1, output="xnext")
def _times_step(x, u):
  return x + 0.1 * u


@sc.function(1, 1, output="xnext")
def _over_step(x, u):
  return x + u / 10.0


@pytest.mark.solver("piqp")
@pytest.mark.parametrize("step", [_times_step, _over_step], ids=["times", "over"])
def test_mpc_takes_a_model_that_scales_its_control(step: sc.Function) -> None:
  # OCP wraps the model in one more Function, so the scaling sits two calls below the constraint.
  ocp = mpc.OCP(
    step=step,
    horizon=4,
    stage_cost=mpc.Quadratic(np.zeros((1, 1)), np.eye(1)),
    terminal=mpc.TerminalEquality(np.array([2.0])),
    name=f"scaled_{step.name}",
  )
  solution = mpc.MPC(ocp, "piqp", options={"eps_abs": 1e-10, "eps_rel": 1e-10}).solve(np.array([0.0]))
  np.testing.assert_allclose(solution.us.ravel(), 5.0, atol=1e-6)  # the cheapest way to 2 in 4 steps of 0.1 u
  np.testing.assert_allclose(solution.cost, 100.0, atol=1e-5)
