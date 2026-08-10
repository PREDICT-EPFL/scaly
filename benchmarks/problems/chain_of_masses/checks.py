"""Formulation correctness gates for the chain-of-masses problem.

These are properties of *this benchmark problem* — its dimensions, the RK4 plant,
the equality-constraint Jacobian against both CasADi and a dense reference, and the
closed-loop episode's shapes. They run before any timing is recorded, via
``benchmarks/run.py smoke``.

They deliberately do **not** live in ``tests/``: per `AGENTS.md`, the pytest suite
covers Alloy's core and must not depend on a benchmark problem. The IR/AD/codegen
behaviours these touch have self-contained reproductions in
``tests/alloy/test_stage_transcription.py`` and ``tests/alloy/test_alloy_sparsity.py``,
so this problem can be retired without dropping compiler coverage.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from alloy.toolchain import solver_loadable
from benchmarks.problems.chain_of_masses import (
  NU,
  ChainParams,
  ca_chain_eq_jac,
  ca_chain_nlpsol,
  chain_eq_function,
  chain_eq_jac_dense_reference,
  chain_nlp,
  chain_step_fn,
  initial_state,
  n_dec,
  n_param,
  n_state,
  rk4_step_np,
  sample_inputs,
)
from benchmarks.problems.chain_of_masses.closed_loop import ClosedLoopConfig, extract_positions, plant_step, run_episode


def check_dims_and_rk4() -> None:
  """Generated step function reproduces the NumPy RK4 plant at the declared dimensions."""
  assert n_state(5) == 21, n_state(5)
  params = ChainParams()
  rng = np.random.default_rng(3)
  for n_masses in (3, 5):
    x = initial_state(n_masses) + rng.normal(scale=0.02, size=n_state(n_masses))
    u = rng.normal(scale=0.1, size=NU)
    actual = chain_step_fn(n_masses)(x, u, *[np.array([value]) for value in params.array()])
    np.testing.assert_allclose(actual, rk4_step_np(x, u, params), rtol=1e-10, atol=1e-10)


def check_eq_jacobian_matches_casadi_and_dense_reference() -> None:
  """Dense and sparse equality Jacobians agree with CasADi and with the dense reference."""
  for n_masses, horizon in ((3, 2), (5, 3)):
    fn = chain_eq_function(n_masses, horizon)
    dense = fn.factory(f"chain_dense_M{n_masses}_N{horizon}", ["z", "p"], ["jac:eq:z"])
    sparse = fn.factory(f"chain_sparse_M{n_masses}_N{horizon}", ["z", "p"], ["spjac:eq:z"])
    ca_dense = ca_chain_eq_jac(n_masses, horizon)
    zv, pv = sample_inputs(n_masses, horizon, seed=11)

    actual = np.asarray(dense(zv, pv))
    np.testing.assert_allclose(actual, np.asarray(ca_dense(zv, pv)), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(chain_eq_jac_dense_reference(n_masses, horizon, zv, pv), actual, rtol=1e-10, atol=1e-10)

    sparsity = sparse.output_sparsities[0]
    assert sparsity is not None
    compact = np.asarray(sparse(zv, pv)).reshape(-1)
    flat = np.asarray(sparsity.rows) * actual.shape[1] + np.asarray(sparsity.cols)
    np.testing.assert_allclose(compact, actual.ravel()[flat], rtol=1e-10, atol=1e-10)
    assert sparsity.nnz < actual.size, (sparsity.nnz, actual.size)


def check_nlp_objective_matches_casadi() -> None:
  """Alloy/IPOPT and CasADi/IPOPT reach the same optimal objective from the same start."""
  n_masses, horizon = 3, 10
  nx, nz = n_state(n_masses), n_state(n_masses) + NU
  zv, pv = sample_inputs(n_masses, horizon, seed=0)
  base = initial_state(n_masses)
  for i in range(horizon):
    zv[i * nz : i * nz + nx] = base
    zv[i * nz + nx : (i + 1) * nz] = 0.0
  zv[horizon * nz :] = base

  generated = chain_nlp(n_masses, horizon)
  alloy_out = generated(zv, np.zeros(nx * (horizon + 1)), np.zeros(0), np.zeros(n_dec(n_masses, horizon)), pv)
  assert generated.last_status is not None and generated.last_status.ok
  assert generated.last_stats is not None and generated.last_stats.iter > 0

  lb, ub = np.full(n_dec(n_masses, horizon), -np.inf), np.full(n_dec(n_masses, horizon), np.inf)
  for i in range(horizon):
    lb[i * nz + nx : (i + 1) * nz] = -1.0
    ub[i * nz + nx : (i + 1) * nz] = 1.0
  ca_solver = ca_chain_nlpsol(n_masses, horizon)
  ca_out = ca_solver(x0=zv, p=pv, lbg=0.0, ubg=0.0, lbx=lb, ubx=ub)
  assert ca_solver.stats()["success"]
  np.testing.assert_allclose(float(alloy_out["f"]), float(ca_out["f"]), rtol=1e-6)


def check_extract_positions() -> None:
  """Rendered positions carry the fixed anchor and reject a mis-shaped state."""
  state = initial_state(5)
  points = extract_positions(state, 5)

  assert points.shape == (5, 3), points.shape
  np.testing.assert_array_equal(points[0], np.zeros(3))
  np.testing.assert_allclose(points[:, 0], np.linspace(0.0, 7.0, 5))
  np.testing.assert_array_equal(points[:, 1:], np.zeros((5, 2)))
  try:
    extract_positions(state[:-1], 5)
  except ValueError as e:
    assert "expected state shape" in str(e), str(e)
  else:
    raise AssertionError("extract_positions accepted a mis-shaped state")


def check_plant_step_is_parameterized_rk4() -> None:
  """The closed-loop plant is exactly the parameterized fixed-step RK4, not a re-implementation."""
  state = initial_state(3)
  control = np.array([0.1, -0.2, 0.3])
  params = ChainParams(mass=0.04, spring_d=0.8, rest_len=0.05, gravity=-8.0, dt=0.01)

  actual = plant_step(state, control, params)
  np.testing.assert_allclose(actual, rk4_step_np(state, control, params), rtol=0.0, atol=0.0)
  assert actual.shape == state.shape
  assert np.all(np.isfinite(actual))


def check_episode_artifacts() -> None:
  """A one-step episode produces the declared shapes and finite telemetry."""
  config = ClosedLoopConfig(n_masses=3, horizon=1, steps=1, params=ChainParams(dt=0.05))
  episode = run_episode(config)

  assert episode.states.shape == (2, 9), episode.states.shape
  assert episode.controls.shape == (1, 3), episode.controls.shape
  assert episode.points.shape == (2, 3, 3), episode.points.shape
  assert episode.oracle_z.shape == (n_dec(3, 1),)
  assert episode.oracle_p.shape == (n_param(3),)
  assert len(episode.telemetry) == 1
  stats = episode.telemetry[0]
  assert stats.t_total >= 0.0 and stats.t_solver >= 0.0 and stats.t_fe >= 0.0 and stats.t_glue >= 0.0
  assert np.all(np.isfinite(episode.states))
  assert np.all(np.isfinite(episode.controls))
  assert np.all(np.isfinite(episode.points))


# name -> (check, requires an IPOPT-backed solve, requires CasADi)
CHECKS: dict[str, tuple[Callable[[], None], bool, bool]] = {
  "dims_and_rk4": (check_dims_and_rk4, False, False),
  "eq_jacobian": (check_eq_jacobian_matches_casadi_and_dense_reference, False, True),
  "nlp_objective": (check_nlp_objective_matches_casadi, True, True),
  "extract_positions": (check_extract_positions, False, False),
  "plant_step": (check_plant_step_is_parameterized_rk4, False, False),
  "episode_artifacts": (check_episode_artifacts, True, False),
}


def run_checks() -> Iterator[tuple[str, str]]:
  """Yield ``(name, outcome)`` for each gate; ``outcome`` is "ok", "skipped: ..." or raises."""
  have_ipopt = solver_loadable("ipopt")
  try:
    import casadi  # noqa: F401

    have_casadi = True
  except ImportError:
    have_casadi = False
  for name, (check, needs_ipopt, needs_casadi) in CHECKS.items():
    if needs_ipopt and not have_ipopt:
      yield name, "skipped: IPOPT plugin not loadable"
    elif needs_casadi and not have_casadi:
      yield name, "skipped: casadi not installed"
    else:
      check()
      yield name, "ok"
