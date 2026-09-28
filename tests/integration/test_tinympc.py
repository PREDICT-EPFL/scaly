"""The TinyMPC example (``examples/tinympc``): the generated ADMM against the NumPy port of the library.

The NumPy port (``problem.reference_solve``) follows the TinyMPC library's ``solve`` line by line;
the benchmark checks it against the library itself. Here the generated solver must take the same
iterations and return the same iterates on every path: box bounds on and off, affine dynamics,
input and state cones, stopping on the tolerance and on the iteration limit, warm starts across a
closed loop, and bounds as inputs or as constants.
"""

from __future__ import annotations

import itertools
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

import scaly as sc

TINYMPC = Path(__file__).resolve().parents[2] / "examples" / "tinympc"
# The example's modules import each other by name, so its directory goes on the path first.
sys.path.insert(0, str(TINYMPC))

import problem as tp  # noqa: E402  # ty: ignore[unresolved-import]
import problems as tps  # noqa: E402  # ty: ignore[unresolved-import]
import solver as ts  # noqa: E402  # ty: ignore[unresolved-import]


def _replay(s: tps.Scenario, steps: int, **kw) -> tuple[tps.Episode, tps.Episode, ts.Solver, tp.ReferenceSolver]:
  gen, ref = ts.Solver(s.problem, s.bounds, **kw), tp.ReferenceSolver(s.problem, s.bounds)
  return tps.closed_loop(s, gen, steps), tps.closed_loop(s, ref, steps), gen, ref


def _assert_same(a: tps.Episode, b: tps.Episode, gen: ts.Solver, ref: tp.ReferenceSolver, atol: float) -> None:
  np.testing.assert_array_equal(a.iterations, b.iterations)
  np.testing.assert_array_equal(a.solved, b.solved)
  np.testing.assert_allclose(a.u0, b.u0, rtol=0, atol=atol)
  state = ts.unpack_state(gen.problem, gen.state)
  for name, value in state.items():
    np.testing.assert_allclose(value, ref.state[name], rtol=0, atol=atol * max(1.0, np.abs(ref.state[name]).max()), err_msg=name)


@pytest.mark.parametrize(
  "scenario, steps, atol",
  [
    (lambda: tps.random_mpc(6, 3, 8), 12, 1e-11),
    (lambda: tps.safety_filter(4, 10), 12, 1e-11),
    (lambda: tps.rocket_landing(8), 12, 1e-9),
  ],
  ids=["random_mpc", "safety_filter", "rocket_landing"],
)
def test_generated_solver_replays_the_library_algorithm(scenario, steps: int, atol: float) -> None:
  a, b, gen, ref = _replay(scenario(), steps)
  _assert_same(a, b, gen, ref, atol)
  assert a.iterations.max() > 3  # the loop, not just its first test, is exercised


def test_state_cones_and_disabled_input_bounds() -> None:
  """A glide-slope cone on the position (state cone) next to the thrust cone, input box off."""
  s = tps.rocket_landing(6)
  # Distinct tolerances, and an input box that would bind if it were (wrongly) applied.
  settings = tp.Settings(1e-2, 3e-2, 300, en_state_bound=False, en_input_bound=False)
  s.bounds.u_min[:], s.bounds.u_max[:] = -1.0, 12.0
  p = s.problem
  s.problem = tp.Problem(p.A, p.B, p.Q, p.R, p.N, p.rho, f=p.f, settings=settings, state_cones=(tp.Cone(0, 3, 0.6),), input_cones=p.input_cones)
  a, b, gen, ref = _replay(s, 10)
  _assert_same(a, b, gen, ref, 1e-9)
  assert "x_min" not in gen.function.input_names and "u_min" not in gen.function.input_names


def test_iteration_limit_updates_the_previous_slacks() -> None:
  """Stopping on ``max_iter`` updates ``v``/``z``; stopping on the tolerance does not."""
  s = tps.safety_filter(4, 10)
  p = s.problem
  s.problem = tp.Problem(p.A, p.B, p.Q, p.R, p.N, p.rho, settings=tp.Settings(1e-2, 1e-2, 3))
  a, b, gen, ref = _replay(s, 6)
  assert not a.solved.any() and (a.iterations == 3).all()
  _assert_same(a, b, gen, ref, 1e-11)
  state = ts.unpack_state(gen.problem, gen.state)
  np.testing.assert_array_equal(state["v"], state["vnew"])


def test_bounds_as_constants_give_the_same_iterates() -> None:
  s = tps.safety_filter(4, 10)
  a, b, gen, ref = _replay(s, 8, fixed_bounds=True)
  _assert_same(a, b, gen, ref, 1e-11)
  assert not {"x_min", "x_max", "u_min", "u_max"} & set(gen.function.input_names)


def test_cone_projection_matches_the_reference_and_is_a_projection() -> None:
  """The library's ``project_soc`` projects onto ``|x| <= mu t`` in the metric that scales the
  axis by ``mu`` (the Euclidean projection when ``mu = 1``): Moreau's decomposition holds there."""
  rng = np.random.default_rng(4)
  pts = rng.normal(size=(200, 3)) * rng.choice([0.1, 1.0, 10.0], size=(200, 1))
  pts[:4] = [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.25, 0.0, 1.0]]  # axis, polar axis, apex, boundary
  mu = 0.25
  w = sc.sym("w", pts.size)
  fn = sc.Function.from_exprs("proj", [w], [ts._project_cones(w, len(pts), 3, (tp.Cone(0, 3, mu),))], ["w"], ["p"])
  got = np.asarray(fn(pts.reshape(-1))).reshape(pts.shape)
  want = np.array([tp.project_soc(s, mu) for s in pts])
  np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
  norm, axis = np.linalg.norm(got[:, :2], axis=1), got[:, 2]
  assert np.all(norm <= mu * axis + 1e-12)  # in the cone
  again = np.array([tp.project_soc(s, mu) for s in got])
  np.testing.assert_allclose(again, got, atol=1e-12)  # idempotent
  # Moreau in the scaled metric D = diag(1, 1, mu): D(s - P(s)) is in the polar of the standard
  # cone and orthogonal to D P(s).
  d = np.array([1.0, 1.0, mu])
  r, g = (pts - got) * d, got * d
  np.testing.assert_allclose(np.einsum("ij,ij->i", r, g), 0.0, atol=1e-9)
  assert np.all(np.linalg.norm(r[:, :2], axis=1) <= -r[:, 2] + 1e-9)


def test_cache_is_the_penalized_riccati_fixed_point() -> None:
  s = tps.rocket_landing(4)
  p, c = s.problem, tp.cache(s.problem)
  q1, r1 = np.diag(p.Q + p.rho), np.diag(p.R + p.rho)
  k = np.linalg.solve(r1 + p.B.T @ c.Pinf @ p.B, p.B.T @ c.Pinf @ p.A)
  np.testing.assert_allclose(c.Kinf, k, atol=1e-4)  # the library stops once K moves by less than 1e-5
  np.testing.assert_allclose(c.Quu_inv, np.linalg.inv(r1 + p.B.T @ c.Pinf @ p.B), rtol=1e-12)
  np.testing.assert_allclose(c.APf, (p.A - p.B @ c.Kinf).T @ c.Pinf @ p.fdyn, rtol=1e-12)
  np.testing.assert_allclose(c.Pinf, q1 + p.A.T @ c.Pinf @ (p.A - p.B @ c.Kinf), rtol=1e-3)


def test_random_problem_generator_reproduces_the_published_draws() -> None:
  """``prob_nx_10`` of the published benchmark was drawn with the generator's seed. ``A`` comes from
  an SVD, and the published generator's ``U S V`` (it multiplies by ``vh.T``) depends on the signs
  of the singular vectors, which LAPACK builds choose differently: some choice of signs for this
  machine's SVD of the seed's draw gives the published rows."""
  d = tps.random_mpc_data(10, 4, 10)
  np.testing.assert_allclose(np.diag(d["Q"])[:3], [6.96469186, 2.86139335, 2.26851454], atol=1e-8)
  rs = np.random.RandomState(123)  # the generator's draws, up to A's
  rs.uniform(0, 10, 10)
  rs.normal(size=(4, 199))
  uu, s, vh = np.linalg.svd(rs.uniform(-1, 1, (10, 10)))
  e = np.diag(s / s.max())
  published = [-0.01307376, 0.22461101, 0.25444081]
  signs = (np.array(bits) for bits in itertools.product((1.0, -1.0), repeat=10))
  assert any(np.allclose(((uu * d_) @ e @ (d_[:, None] * vh).T)[0, :3], published, atol=1e-8) for d_ in signs)
  assert np.abs(np.linalg.eigvals(d["A"])).max() <= 1.0 + 1e-12


def test_examples_run() -> None:
  out = runpy.run_path(str(TINYMPC / "random_mpc.py"), run_name="tinympc_random_mpc")["main"](steps=15, check_steps=5)
  assert out["reference_same_iterations"] and out["reference_u0_diff"] < 1e-10
  assert np.abs(out["episode"].u0).max() <= 3.0 + 1e-3
  out = runpy.run_path(str(TINYMPC / "safety_filter.py"), run_name="tinympc_safety_filter")["main"](steps=15, check_steps=5)
  assert out["reference_same_iterations"] and out["modified"].any()
  out = runpy.run_path(str(TINYMPC / "rocket_landing.py"), run_name="tinympc_rocket_landing")["main"](n=8, steps=40, check_steps=5)
  assert out["reference_same_iterations"] and out["reference_u0_diff"] < 1e-9
  assert out["episode"].x0[-1][2] < out["episode"].x0[0][2]  # descending
