"""OCPs as problems: a continuous one transcribed and formulated, a discrete one formulated sparse and
condensed, each solved as an ``sc.opt`` problem; the stage structure and layout they carry; what
they refuse."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly import integrators as si
from scaly import ocp

A = np.array([[1.0, 0.1], [0.0, 1.0]])
B = np.array([[0.005], [0.1]])
X0 = np.array([1.0, -0.5])


@sc.function(2, 1, output="xnext")
def double_integrator(x: sc.Expr, u: sc.Expr) -> sc.Expr:
  return sc.const(A) @ x + sc.const(B) @ u


@sc.function(2, 1, output="xdot")
def spring(x: sc.Expr, u: sc.Expr) -> sc.Expr:
  return sc.stack([x[1], -4.0 * x[0] - 0.3 * x[1] + u[0]])


@sc.function(2, 1, output="l")
def effort(x: sc.Expr, u: sc.Expr) -> sc.Expr:
  return (x * x).sum() + 0.1 * u[0] * u[0]


def _solve(problem: sc.opt.NLP, layout: ocp.Layout, method: object, x0: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
  solve = sc.opt.solver(problem, method, name=f"{problem.name}_{len(layout.var_names)}_{type(method).__name__}")
  zeros = problem.vars.unflatten(tuple(np.zeros(s) for s in problem.vars.shapes))
  primal, *_, info = solve(zeros, zeros, np.zeros(problem.n_eq), np.zeros(problem.n_ineq), x0)
  assert sc.Status(int(info.status)) == sc.Status.OK
  leaves = primal if isinstance(primal, tuple) else (primal,)
  return dict(zip(layout.var_names, (np.asarray(v) for v in leaves), strict=True)), float(info.objective)


@pytest.mark.method("opt.piqp")
def test_a_discrete_ocp_solves_the_same_sparse_and_condensed() -> None:
  problem = ocp.DiscreteOCP(
    step=double_integrator,
    N=10,
    stage_cost=ocp.Quadratic(np.eye(2), 0.1 * np.eye(1)),
    terminal_cost=ocp.Quadratic(10.0 * np.eye(2)),
    u_bounds=(-1.0, 1.0),
    x_bounds=([-5.0, -0.6], [5.0, 0.6]),
  )
  sparse, sparse_layout = ocp.to_problem(problem)
  condensed, condensed_layout = ocp.to_problem(problem, "condensed")
  assert ocp.to_problem(problem) is ocp.to_problem(problem)  # built once per form
  assert sparse_layout.var_names == ("xs", "us") and condensed_layout.var_names == ("us",)
  s, cost_s = _solve(sparse, sparse_layout, sc.opt.PIQP(sparse=True, options={"eps_abs": 1e-10, "eps_rel": 1e-10}), X0)
  c, cost_c = _solve(condensed, condensed_layout, sc.opt.PIQP(options={"eps_abs": 1e-10, "eps_rel": 1e-10}), X0)
  np.testing.assert_allclose(c["us"], s["us"], atol=1e-7)
  np.testing.assert_allclose(cost_c, cost_s, rtol=1e-8)
  assert condensed_layout.states is not None
  np.testing.assert_allclose(condensed_layout.states(X0, c["us"]), s["xs"], atol=1e-7)
  assert np.abs(s["xs"].reshape(11, 2)[1:, 1]).max() <= 0.6 + 1e-7  # the state bound holds after the start


@pytest.mark.method("opt.ipopt")
def test_a_continuous_ocp_transcribed_three_ways() -> None:
  continuous = ocp.ContinuousOCP(spring, T=1.0, stage_cost=effort, u_bounds=(-2.0, 2.0), name="spring_ocp")
  costs = {}
  for label, transcription in (
    ("shooting", ocp.MultipleShooting(si.RK4(steps=4))),
    ("collocation", ocp.Collocation(3)),
    ("pseudospectral", ocp.Pseudospectral(4)),
  ):
    discrete = ocp.transcribe(continuous, transcription, N=10)
    assert discrete.dt == pytest.approx(0.1) and discrete.N == 10 and discrete.name == "spring_ocp"
    problem, layout = ocp.to_problem(discrete)
    _, costs[label] = _solve(problem, layout, sc.opt.IPOPT(options={"tol": 1e-10}), X0)
  # The same problem, integrated three ways: the costs differ by the discretizations' errors.
  assert costs["collocation"] == pytest.approx(costs["shooting"], rel=1e-3)
  assert costs["pseudospectral"] == pytest.approx(costs["shooting"], rel=2e-2)


def test_the_stage_structure_and_the_layout() -> None:
  discrete = ocp.DiscreteOCP(step=double_integrator, N=4, stage_cost=effort, dt=0.1)
  assert discrete.stage == ocp.StageStructure(nx=2, nu=1, nw=0, n_dynamics=2)
  np.testing.assert_allclose(discrete.times, [0.0, 0.1, 0.2, 0.3, 0.4])
  _, layout = ocp.to_problem(discrete)
  assert layout.var_sizes == (10, 4) and layout.var_blocks == (2, 1)
  assert layout.multiplier_blocks() == ([(2, 8, 2)], [])  # the dynamics after x0's pin, one block of 2 per stage

  collocated = ocp.transcribe(ocp.ContinuousOCP(spring, T=0.4, stage_cost=effort), ocp.Collocation(3), N=4)
  interval = collocated.interval
  assert collocated.stage == ocp.StageStructure(nx=2, nu=1, nw=interval.n_internal, n_dynamics=interval.n_residual)
  assert interval.n_internal == 4 and interval.n_residual == 6  # two internal states; three collocation conditions of two
  _, layout = ocp.to_problem(collocated)
  assert layout.var_names == ("xs", "us", "zs") and layout.var_blocks == (2, 1, 4)
  assert layout.multiplier_blocks()[0] == [(2, 24, 6)]

  @sc.function(2, 1, output="g")
  def speed(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    return x[1:]

  soft = ocp.DiscreteOCP(step=double_integrator, N=4, constraints=[ocp.Path(speed, lo=-1.0, hi=1.0, soft=10.0)])
  _, layout = ocp.to_problem(soft)
  assert layout.var_names == ("xs", "us", "slack") and layout.var_blocks == (2, 1, 1)
  assert layout.multiplier_blocks()[1] == [(0, 4, 1), (4, 4, 1)]  # the upper and the lower rows, per stage


def test_what_an_ocp_refuses() -> None:
  continuous = ocp.ContinuousOCP(spring, T=1.0, stage_cost=effort)
  with pytest.raises(ValueError, match="T, the horizon's length, must be positive"):
    ocp.ContinuousOCP(spring, T=0.0)
  with pytest.raises(ValueError, match="cost is 'points' or 'integral'"):
    ocp.ContinuousOCP(spring, T=1.0, cost="trapezoid")  # ty: ignore[invalid-argument-type]
  with pytest.raises(ValueError, match="N must be a positive integer"):
    ocp.transcribe(continuous, N=0)
  with pytest.raises(TypeError, match="takes a ContinuousOCP"):
    ocp.transcribe(ocp.DiscreteOCP(step=double_integrator, N=3), N=3)  # ty: ignore[invalid-argument-type]
  with pytest.raises(ValueError, match="form is 'sparse' or 'condensed'"):
    ocp.to_problem(ocp.DiscreteOCP(step=double_integrator, N=3), "dense")  # ty: ignore[invalid-argument-type]
  with pytest.raises(ValueError, match="the condensed form takes a discrete map"):
    ocp.to_problem(ocp.transcribe(continuous, ocp.Collocation(2), N=3), "condensed")
  with pytest.raises(ValueError, match="takes the state, then the control"):
    ocp.DiscreteOCP(step=sc.Function.from_exprs("one_input", [x := sc.sym("x", 2)], [x], ["x"], ["xnext"]), N=3)
  with pytest.raises(ValueError, match="Q must be 2x2"):
    ocp.DiscreteOCP(step=double_integrator, N=3, stage_cost=ocp.Quadratic(np.eye(1)))
  with pytest.raises(ValueError, match="varying names"):
    ocp.DiscreteOCP(step=double_integrator, N=3, varying=("r",))

  @sc.function(2, 1, 3, output="l")
  def clash(x: sc.Expr, u: sc.Expr, stiffness: sc.Expr) -> sc.Expr:
    return (x * x).sum()

  @sc.function(2, 1, (), output="xdot")
  def stiff(x: sc.Expr, u: sc.Expr, stiffness: sc.Expr) -> sc.Expr:
    return sc.stack([x[1], -stiffness * x[0] + u[0]])

  with pytest.raises(ValueError, match="parameter 'stiffness' is"):
    ocp.transcribe(ocp.ContinuousOCP(stiff, T=1.0, stage_cost=clash), N=3)

  @sc.function(2, 1, 2, output="xnext")
  def taken(x: sc.Expr, u: sc.Expr, x0: sc.Expr) -> sc.Expr:
    return x + x0

  with pytest.raises(ValueError, match="may not be named 'x0'"):
    ocp.DiscreteOCP(step=taken, N=3)
