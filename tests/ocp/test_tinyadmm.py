"""TinyADMM as an OCP method: the problem's own solution against the direct method on PIQP, with box
bounds, references per stage and an affine offset; the LQ problem in one primal step; the finite
horizon's Riccati cache; the warm start and its shift; its statuses; what it refuses."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import scaly as sc
from scaly import integrators as si
from scaly import ocp
from scaly.linalg.stagewise import Riccati
from scaly.ocp.tinyadmm import finite_cache

from .support import Controller, closed_loop

A = np.array([[1.0, 0.1], [0.0, 1.0]])
B = np.array([[0.005], [0.1]])
Q, R, QN = np.diag([1.0, 0.1]), 0.05 * np.eye(1), np.diag([10.0, 1.0])
TIGHT: dict[str, Any] = {"abs_pri_tol": 1e-10, "abs_dua_tol": 1e-10, "max_iter": 20000}
PIQP = ocp.Direct(sc.opt.PIQP(sparse=True, options={"eps_abs": 1e-11, "eps_rel": 1e-11}))


def _boxed(name: str, horizon: int = 20, **kwargs) -> ocp.DiscreteOCP:
  return ocp.DiscreteOCP(
    step=si.affine(A, B, name=f"{name}_map"),
    N=horizon,
    stage_cost=kwargs.pop("stage_cost", ocp.Quadratic(Q, R)),
    terminal_cost=ocp.Quadratic(QN, x_ref=kwargs.pop("terminal_ref", None)),
    u_bounds=(-1.0, 1.0),
    x_bounds=([-5.0, -0.6], [5.0, 0.6]),
    name=name,
    **kwargs,
  )


@pytest.mark.method("opt.piqp")
def test_a_boxed_linear_problem_is_the_direct_methods_solution() -> None:
  problem = _boxed("tiny_boxed")
  x0 = np.array([2.0, 0.0])
  admm = Controller(problem, ocp.TinyADMM(rho=5.0, **TIGHT)).solve(x0)
  qp = Controller(problem, PIQP).solve(x0)
  assert admm.status == sc.Status.OK and qp.status.ok
  np.testing.assert_allclose(admm.us, qp.us, atol=1e-7)
  np.testing.assert_allclose(admm.xs, qp.xs, atol=1e-7)
  np.testing.assert_allclose(admm.cost, qp.cost, rtol=1e-8)
  assert np.abs(qp.us).max() > 1 - 1e-6 and np.abs(qp.xs[:, 1]).max() > 0.6 - 1e-6  # both kinds of bound bind
  outside = np.array([0.0, 0.65])  # the initial state is data, outside the speed bound; one step brings it back
  from_outside = Controller(problem, ocp.TinyADMM(rho=5.0, **TIGHT)).solve(outside)
  assert from_outside.status == sc.Status.OK  # its knot is not clipped, or the residual would never vanish
  np.testing.assert_allclose(from_outside.us, Controller(problem, PIQP).solve(outside).us, atol=1e-7)


@pytest.mark.method("opt.ipopt")
def test_references_per_stage_are_the_problems_not_the_librarys() -> None:
  """A varying state reference and a fixed control reference: TinyMPC's library would weigh the
  reference by ``Q + rho I``; the method solves the stated problem."""
  horizon = 12
  # Small enough that no bound binds: the references alone decide the solution.
  ref = np.stack([np.linspace(0.0, 0.3, horizon + 1), np.full(horizon + 1, 0.05)], axis=1).ravel()
  problem = _boxed(
    "tiny_tracking",
    horizon=horizon,
    stage_cost=ocp.Quadratic(Q, R, x_ref="r", u_ref=np.array([0.1])),
    terminal_ref="r",
    varying=("r",),
  )
  x0 = np.array([0.1, 0.0])
  admm = Controller(problem, ocp.TinyADMM(rho=2.0, **TIGHT)).solve(x0, r=ref)
  nlp = Controller(problem, ocp.Direct(sc.opt.IPOPT(options={"tol": 1e-13}))).solve(x0, r=ref)
  assert admm.status == sc.Status.OK and np.abs(nlp.us).max() < 0.9 and np.abs(nlp.xs[:, 1]).max() < 0.5
  np.testing.assert_allclose(admm.us, nlp.us, atol=1e-7)
  np.testing.assert_allclose(admm.cost, nlp.cost, rtol=1e-8)


@sc.function(2, 1, output="xdot", name="tiny_damped")
def damped(x, u):
  return sc.const(np.array([[0.0, 1.0], [-2.0, -0.3]])) @ x + sc.const(np.array([[0.0], [1.0]])) @ u + sc.const(np.array([0.0, 0.4]))


@pytest.mark.method("opt.piqp")
def test_a_linear_model_by_multiple_shooting_with_an_offset() -> None:
  """The map is RK4 of an affine ODE, ``A x + B u + f`` with ``f`` from the offset; the running cost
  is ``dt`` times the sum at the points."""
  continuous = ocp.ContinuousOCP(
    damped, T=2.0, stage_cost=ocp.Quadratic(Q, R), terminal_cost=ocp.Quadratic(QN), u_bounds=(-2.0, 2.0), name="tiny_damped_ocp"
  )
  problem = ocp.transcribe(continuous, ocp.MultipleShooting(si.RK4()), N=20)
  x0 = np.array([1.0, 0.5])
  admm = Controller(problem, ocp.TinyADMM(rho=1.0, **TIGHT)).solve(x0)
  qp = Controller(problem, PIQP).solve(x0)
  np.testing.assert_allclose(admm.us, qp.us, atol=1e-7)
  np.testing.assert_allclose(admm.xs, qp.xs, atol=1e-8)
  np.testing.assert_allclose(admm.cost, qp.cost, rtol=1e-8)


@pytest.mark.method("opt.piqp")
def test_without_bounds_the_slacks_clip_nothing_and_the_lq_solution_is_reached() -> None:
  problem = ocp.DiscreteOCP(
    step=si.affine(A, B, name="tiny_free_map"), N=15, stage_cost=ocp.Quadratic(Q, R), terminal_cost=ocp.Quadratic(QN), name="tiny_free"
  )
  x0 = np.array([1.0, -0.5])
  admm = Controller(problem, ocp.TinyADMM(rho=3.0, **TIGHT)).solve(x0)
  qp = Controller(problem, PIQP).solve(x0)
  assert admm.status == sc.Status.OK
  np.testing.assert_allclose(admm.us, qp.us, rtol=1e-8, atol=1e-9)


def test_the_finite_caches_gains_are_the_riccati_recursions() -> None:
  rho, knots = 0.7, 8
  c = finite_cache(A, B, Q, R, QN, rho, knots)
  gains = Riccati(A, B, Q + rho * np.eye(2), R + rho * np.eye(1), QN + rho * np.eye(2), N=knots - 1).gains
  (expected,) = sc.Function.from_exprs("tiny_riccati_gains", [], [gains], [], ["k"])._flat_numerical_call()
  assert c.K.shape == (knots - 1, 1, 2) and c.varying
  np.testing.assert_allclose(-np.asarray(expected), c.K, rtol=1e-12, atol=1e-14)  # u = -K x there, u = K x here


def test_the_warm_start_and_its_shift() -> None:
  problem = _boxed("tiny_shift", horizon=3)
  method = ocp.TinyADMM()
  size = method.warm_size(problem)
  assert size == 4 * (4 * 2) + 4 * (3 * 1)  # x, v, vnew, g over 4 knots; u, z, znew, y over 3 stages
  guess = ocp.initial_guess(problem, method, np.array([1.0, 2.0]), np.array([0.5]))
  np.testing.assert_array_equal(guess[:8], np.tile([1.0, 2.0], 4))
  np.testing.assert_array_equal(guess[8:11], [0.5] * 3)
  moved = np.asarray(ocp.shift(problem, method)(np.arange(float(size))))
  np.testing.assert_array_equal(moved[:8], [2, 3, 4, 5, 6, 7, 6, 7])  # x by one knot, the last repeated
  np.testing.assert_array_equal(moved[8:11], [9, 10, 10])  # u by one stage


def test_a_receding_horizon_warm_starts_from_the_shifted_state() -> None:
  problem = _boxed("tiny_loop")
  method = ocp.TinyADMM(rho=5.0, abs_pri_tol=1e-6, abs_dua_tol=1e-6, max_iter=5000)
  controller = Controller(problem, method)
  xs, us, statuses, iterations = closed_loop(controller, lambda x, u: A @ x + B @ u, np.array([2.0, 0.0]), 40)
  assert set(statuses) == {sc.Status.OK}
  assert np.median(iterations[1:]) < iterations[0]  # the shifted state starts closer
  assert np.abs(us).max() <= 1 + 1e-5 and np.abs(xs[-1]).max() < 0.2  # from 2 m, at the force limit, in 4 s


def test_the_iteration_limit() -> None:
  solution = Controller(_boxed("tiny_limit"), ocp.TinyADMM(max_iter=2)).solve(np.array([2.0, 0.0]))
  assert solution.status == sc.Status.MAX_ITER and solution.iterations == 2


def test_what_tinyadmm_refuses() -> None:
  @sc.function(2, 1, output="xnext", name="tiny_pendulum")
  def pendulum(x, u):
    return sc.stack([x[0] + 0.1 * x[1], x[1] - 0.1 * x[0].sin() + 0.1 * u[0]])

  @sc.function(2, 1, (), output="xnext", name="tiny_massive")
  def massive(x, u, mass):
    return sc.const(A) @ x + sc.const(B) @ u / mass

  @sc.function(2, 1, output="l", name="tiny_function_cost")
  def function_cost(x, u):
    return (x * x).sum() + (u * u).sum()

  @sc.function(2, 1, output="g", name="tiny_path")
  def path(x, u):
    return x[:1]

  quadratic = ocp.Quadratic(Q, R)
  cases = {
    "affine in the state and the control": ocp.DiscreteOCP(step=pendulum, N=4, stage_cost=quadratic, name="tiny_nonlinear"),
    "are not references": ocp.DiscreteOCP(step=massive, N=4, stage_cost=quadratic, name="tiny_parametric"),
    "takes Quadratic costs": ocp.DiscreteOCP(step=si.affine(A, B, name="tiny_fc_map"), N=4, stage_cost=function_cost, name="tiny_fc"),
    "no path constraints": _boxed("tiny_paths", constraints=[ocp.Path(path, hi=1.0)]),
    "no terminal set or equality": _boxed("tiny_equality", terminal=ocp.TerminalEquality()),
    "needs a control weight R": ocp.DiscreteOCP(step=si.affine(A, B, name="tiny_nor_map"), N=4, stage_cost=ocp.Quadratic(Q), name="tiny_nor"),
  }
  for reason, problem in cases.items():
    with pytest.raises(ValueError, match=reason):
      ocp.solver(problem, ocp.TinyADMM())
  with pytest.raises(ValueError, match="rho > 0"):
    ocp.TinyADMM(rho=0.0)
