"""Terminal ingredients: the LQR, the maximal positively invariant set against a brute-force closed
loop, ellipsoids, and MPC with them: u = Kx without constraints, recursive feasibility and decrease
with them."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import scaly as sc
from scaly import mpc
from scaly.opt.qp import NotQuadratic
from scaly.opt.external.graph import solver_descriptor

A = np.array([[1.0, 0.1], [0.0, 1.0]])
B = np.array([[0.005], [0.1]])
Q, R = np.eye(2), 0.1 * np.eye(1)
X = mpc.Polytope.box([-5, -2], [5, 2])
U = mpc.Polytope.box([-1], [1])
K, P = mpc.lqr(A, B, Q, R)
A_K = A + B @ K
PIQP = {"eps_abs": 1e-11, "eps_rel": 1e-11}


def test_lqr_solves_the_riccati_equation_and_stabilizes() -> None:
  riccati = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
  np.testing.assert_allclose(riccati, P, rtol=1e-10)
  assert np.max(np.abs(np.linalg.eigvals(A_K))) < 1 and np.allclose(P, P.T)
  # The cost to go is the sum of the stage costs along the closed loop.
  x, total = np.array([1.0, -0.5]), 0.0
  for _ in range(2000):
    u = K @ x
    total += x @ Q @ x + u @ R @ u
    x = A_K @ x
  assert total == pytest.approx(np.array([1.0, -0.5]) @ P @ np.array([1.0, -0.5]), rel=1e-10)


def test_lqr_refuses_what_it_cannot_stabilize() -> None:
  with pytest.raises((ValueError, np.linalg.LinAlgError)):
    mpc.lqr(np.diag([2.0, 0.5]), np.array([[0.0], [1.0]]), np.eye(2), np.eye(1))  # the unstable mode is not reachable


def test_the_maximal_invariant_set_is_exactly_the_safe_states() -> None:
  constraints = X.intersect(U.preimage(K))
  invariant = mpc.max_invariant_set(A_K, constraints)
  for row, bound in zip(invariant.H @ A_K, invariant.h, strict=True):  # invariance: the image of the set stays inside
    assert invariant.support(row) <= bound + 1e-9
  grid = np.stack(np.meshgrid(np.linspace(-5, 5, 41), np.linspace(-2, 2, 41)), axis=-1).reshape(-1, 2)
  safe = np.ones(len(grid), dtype=bool)
  states = grid.copy()
  for _ in range(400):  # a trajectory that ever leaves the constraints is not in the set
    safe &= constraints.contains(states, tol=1e-9)
    states = states @ A_K.T
  for i in range(invariant.h.size):  # every row bounds the set: without it the set would grow
    others = mpc.Polytope(np.delete(invariant.H, i, axis=0), np.delete(invariant.h, i))
    assert others.support(invariant.H[i]) > invariant.h[i] + 1e-9
  inside = invariant.contains(grid, tol=1e-9)
  margin = np.min(invariant.h[None, :] - grid @ invariant.H.T, axis=1)
  near = np.abs(margin) < 1e-6  # points on the boundary may go either way at a finite tolerance
  assert np.array_equal(inside[~near], safe[~near]) and inside.sum() > 50


def test_a_set_that_is_not_finitely_determined_raises() -> None:
  angle = np.sqrt(2.0) / 10  # a rotation by an irrational angle: its invariant subset of a box is a disk
  rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
  with pytest.raises(ValueError, match="not finitely determined within 20 steps"):
    mpc.max_invariant_set(rotation, mpc.Polytope.box([-1, -1], [1, 1]), max_iter=20)


def test_the_largest_lqr_ellipsoid_touches_the_polytope_and_is_invariant() -> None:
  constraints = X.intersect(U.preimage(K))
  ellipsoid = mpc.largest_ellipsoid(P, constraints)
  reach = np.sqrt(ellipsoid.alpha * np.einsum("ij,jk,ik->i", constraints.H, np.linalg.inv(P), constraints.H))
  assert np.all(reach <= constraints.h + 1e-12) and np.isclose(reach / constraints.h, 1.0).any()  # inside, and touching
  rng = np.random.default_rng(3)
  points = rng.normal(size=(500, 2))
  points *= np.sqrt(ellipsoid.alpha / np.einsum("ni,ij,nj->n", points, P, points))[:, None] * rng.uniform(0, 1, (500, 1))
  assert ellipsoid.contains(points).all() and ellipsoid.contains(points @ A_K.T).all()
  with pytest.raises(ValueError, match="origin must lie in the polytope's interior"):
    mpc.largest_ellipsoid(P, mpc.Polytope.box([0.5, -1], [1, 1]))
  wide = mpc.largest_ellipsoid(np.eye(2), mpc.Polytope.box([-3, -3], [3, 3]))
  assert wide.alpha == pytest.approx(9.0)  # the disk of radius 3: alpha is h squared over H P^-1 H'
  x = sc.sym("x", 2)
  _, (group,) = ellipsoid.constraints(x)
  assert group.hi is not None and group.hi.value == pytest.approx(ellipsoid.alpha) and group.expr.shape == (1,)


@pytest.mark.solver("piqp")
@pytest.mark.parametrize("condensed", [False, True])
@pytest.mark.parametrize("horizon", [1, 5, 20])
def test_unconstrained_mpc_with_the_lqr_terminal_cost_is_the_lqr(condensed: bool, horizon: int) -> None:
  ocp = mpc.OCP(
    step=mpc.linear(A, B),
    horizon=horizon,
    stage_cost=mpc.Quadratic(Q, R),
    terminal_cost=mpc.Quadratic(P),
    condensed=condensed,
    name=f"lqr_{horizon}_{int(condensed)}",
  )
  controller = mpc.MPC(ocp, "piqp", options=PIQP)
  for x in (np.array([1.3, -0.7]), np.array([-4.0, 2.5])):
    solution = controller.solve(x, guess=controller.initial_guess(x))
    np.testing.assert_allclose(solution.us[0], K @ x, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(solution.cost, x @ P @ x, rtol=1e-9)  # the optimal cost is the cost to go


def _constrained(name: str, horizon: int = 25, condensed: bool = False, terminal=None) -> mpc.OCP:
  return mpc.OCP(
    step=mpc.linear(A, B),
    horizon=horizon,
    stage_cost=mpc.Quadratic(Q, R),
    terminal_cost=mpc.Quadratic(P),
    u_bounds=(-1, 1),
    x_bounds=([-5, -2], [5, 2]),
    terminal=terminal if terminal is not None else mpc.max_invariant_set(A_K, X.intersect(U.preimage(K))),
    condensed=condensed,
    name=name,
  )


@pytest.mark.solver("piqp")
def test_the_condensed_form_solves_the_same_problem() -> None:
  x = np.array([2.5, 0.5])
  sparse = mpc.MPC(_constrained("dense_vs_sparse_s"), "piqp", options=PIQP).solve(x)
  dense = mpc.MPC(_constrained("dense_vs_sparse_c", condensed=True), "piqp", options=PIQP).solve(x)
  assert sparse.status.ok and dense.status.ok and dense.xs.shape == sparse.xs.shape
  np.testing.assert_allclose(dense.us, sparse.us, rtol=1e-7, atol=1e-8)
  np.testing.assert_allclose(dense.xs, sparse.xs, rtol=1e-7, atol=1e-8)
  assert np.abs(sparse.us).max() > 1 - 1e-6  # the input bound binds


@pytest.mark.solver("piqp")
def test_the_condensed_form_keeps_its_state_bounds() -> None:
  tight: dict[str, Any] = {"x_bounds": ([-5, -0.3], [5, 0.3])}
  x = np.array([1.0, 0.0])

  def solve(condensed: bool) -> mpc.Solution:
    ocp = mpc.OCP(
      step=mpc.linear(A, B),
      horizon=25,
      stage_cost=mpc.Quadratic(Q, R),
      terminal_cost=mpc.Quadratic(P),
      u_bounds=(-1, 1),
      condensed=condensed,
      name=f"tight_{int(condensed)}",
      **tight,
    )
    return mpc.MPC(ocp, "piqp", options=PIQP).solve(x)

  sparse, dense = solve(False), solve(True)
  assert np.abs(sparse.xs[:, 1]).max() == pytest.approx(0.3, abs=1e-6)  # the speed limit binds
  np.testing.assert_allclose(dense.xs, sparse.xs, rtol=1e-7, atol=1e-8)


@pytest.mark.solver("piqp")
def test_piqp_runs_its_sparse_backend_on_a_horizon() -> None:
  sparse = mpc.MPC(_constrained("backend_sparse", horizon=5), "piqp")
  dense = mpc.MPC(_constrained("backend_dense", horizon=5, condensed=True), "piqp")
  chosen = mpc.MPC(_constrained("backend_chosen", horizon=5), "piqp", options={"sparse": False})
  assert solver_descriptor(sparse.solver).sparse and not solver_descriptor(dense.solver).sparse and not solver_descriptor(chosen.solver).sparse


@pytest.mark.solver("piqp")
def test_recursive_feasibility_and_decrease_in_closed_loop() -> None:
  """With the maximal invariant set and the LQR cost as terminal ingredients, a state the MPC can
  solve from stays solvable, and the optimal cost falls by at least the stage cost at every step."""
  controller = mpc.MPC(_constrained("recursive"), "piqp", options=PIQP)
  rng = np.random.default_rng(5)
  starts = 0
  while starts < 4:
    x = rng.uniform([-5, -2], [5, 2])
    first = controller.solve(x, guess=controller.initial_guess(x))
    if not first.status.ok:
      continue  # outside the feasible set; draw another
    starts += 1
    cost = first.cost
    for _ in range(25):
      solution = controller.solve(x)
      assert solution.status.ok
      u = solution.us[0]
      x_next = A @ x + B @ u
      following = controller.solve(x_next)
      assert following.status.ok
      assert following.cost <= solution.cost - (x @ Q @ x + u @ R @ u) + 1e-7
      x, cost = x_next, following.cost
    assert cost < first.cost


@pytest.mark.solver("ipopt")
def test_an_ellipsoidal_terminal_set_goes_to_an_nlp_solver_not_a_qp_one() -> None:
  ellipsoid = mpc.largest_ellipsoid(P, X.intersect(U.preimage(K)))
  ocp = _constrained("ellipsoid", terminal=ellipsoid)
  solution = mpc.MPC(ocp, "ipopt", options={"tol": 1e-10}).solve(np.array([1.0, 0.0]))  # the ellipsoid is well inside the invariant set
  assert solution.status.ok and ellipsoid.contains(solution.xs[-1], tol=1e-6)
  with pytest.raises(NotQuadratic, match="terminal"):
    mpc.MPC(_constrained("ellipsoid_qp", terminal=ellipsoid), "piqp")


def test_linear_models_and_the_condensed_form_refuse_what_they_cannot_take() -> None:
  with pytest.raises(ValueError, match="A must be square"):
    mpc.linear(np.ones((2, 3)), np.ones((2, 1)))

  @sc.function(2, 1, output="xdot")
  def ode(x, u):
    return x + u

  with pytest.raises(ValueError, match="the condensed form takes a discrete map"):
    mpc.OCP(ode=ode, dt=0.1, horizon=3, transcription=sc.integrators.Collocation(2), condensed=True)
