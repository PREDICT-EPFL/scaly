"""Formulation correctness gates for the chain-of-masses problem.

These are properties of *this benchmark problem* — its dimensions, the RK4 plant,
the equality-constraint Jacobian against both CasADi and a dense reference, and the
closed-loop episode's shapes. They run before any timing is recorded, via
``benchmarks/run.py smoke``.

They deliberately do **not** live in ``tests/``: per `AGENTS.md`, the pytest suite
covers Alloy's core and must not depend on a benchmark problem. The IR/AD/codegen
behaviours these touch have self-contained reproductions in
``tests/integration/test_stage_transcription.py`` and ``tests/ad/test_sparsity.py``,
so this problem can be retired without dropping compiler coverage.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

import alloy as al
from alloy.solvers.paths import solver_loadable
from benchmarks.problems.chain import (
  END_REF,
  NU,
  ChainParams,
  ca_chain_eq_jac,
  ca_chain_nlpsol,
  chain_eq_function,
  chain_eq_jac_dense_reference,
  chain_nlp,
  chain_objective_fn,
  chain_step_fn,
  initial_state,
  n_dec,
  n_param,
  n_state,
  rk4_step_np,
  sample_inputs,
)
from benchmarks.problems.chain.closed_loop import ClosedLoopConfig, extract_positions, plant_step, run_episode


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
    dense = fn.factory(f"chain_dense_M{n_masses}_N{horizon}", ["z", "p"], [al.factory.Jac("eq", "z")])
    sparse = chain_nlp(n_masses, horizon).descriptor.jac
    assert isinstance(sparse, al.Function)
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
  assert generated.descriptor.hess is not None
  assert dict(generated.descriptor.options).get("hessian_approximation") != "limited-memory"
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


def check_one_reference_for_every_end_mass_term() -> None:
  """Stage and terminal costs track the same end-mass reference, and only that reference.

  This is the deliberate deviation from laopt, whose expanded stage and terminal terms imply two
  different targets (`x = 3.0` and `x = 0.75`), so it is worth pinning: put every end mass on
  ``END_REF`` at rest with zero control, and the objective's gradient must vanish in every
  end-mass block — a stage weight that disagreed with the terminal one would show up here.
  """
  n_masses, horizon = 5, 4
  nz = n_state(n_masses) + NU
  end = 3 * (n_masses - 2)
  grad = chain_objective_fn(n_masses, horizon).factory(f"chain_obj_grad_M{n_masses}_N{horizon}", ["z"], [al.factory.Grad("f", "z")])

  z = np.zeros(n_dec(n_masses, horizon))
  for stage in range(horizon + 1):
    z[stage * nz + end : stage * nz + end + 3] = END_REF
  actual = np.asarray(grad(z)).reshape(-1)

  np.testing.assert_allclose(actual, np.zeros_like(actual), rtol=0.0, atol=1e-12)
  # and the reference really is the only stationary point: moving one end mass off it costs
  for offset in (np.array([0.1, 0.0, 0.0]), np.array([0.0, -0.2, 0.0]), np.array([0.0, 0.0, 0.3])):
    moved = z.copy()
    moved[end : end + 3] += offset
    assert np.any(np.abs(np.asarray(grad(moved)).reshape(-1)) > 1e-9), offset


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
  assert episode.plans.shape == (1, 2, 3, 3), episode.plans.shape
  assert episode.oracle_z.shape == (n_dec(3, 1),)
  assert episode.oracle_p.shape == (n_param(3),)
  assert len(episode.telemetry) == 1
  stats = episode.telemetry[0]
  assert stats.t_total >= 0.0 and stats.t_solver >= 0.0 and stats.t_fe >= 0.0 and stats.t_glue >= 0.0
  assert np.all(np.isfinite(episode.states))
  assert np.all(np.isfinite(episode.controls))
  assert np.all(np.isfinite(episode.points))
  # the plan's first node is the measured state the solve was handed
  np.testing.assert_allclose(episode.plans[0, 0], episode.points[0], rtol=0.0, atol=1e-9)


def check_sqp_oracles_agree() -> None:
  """The same SQP produces the same smoke episode from Alloy and CasADi C oracles."""
  config = ClosedLoopConfig.smoke()
  alloy_run = run_episode(config, solver="sqp", oracle="alloy")
  casadi_run = run_episode(config, solver="sqp", oracle="casadi")
  np.testing.assert_allclose(alloy_run.controls, casadi_run.controls, rtol=1e-8, atol=1e-8)
  np.testing.assert_allclose(alloy_run.states, casadi_run.states, rtol=1e-9, atol=5e-10)
  for run in (alloy_run, casadi_run):
    for stats in run.telemetry:
      assert stats.status.value <= 1 and stats.t_qp > 0.0
      np.testing.assert_allclose(stats.t_total, stats.t_fe + stats.t_solver + stats.t_qp + stats.t_globalization + stats.t_glue, rtol=1e-10)


def check_sqp_matches_ipopt() -> None:
  """SQP and IPOPT drive the same smoke episode to the same per-step solutions.

  Different algorithms are allowed small differences, not major ones. Measured healthy-step
  differences at these settings (exact Lagrangian Hessian, tol 1e-6): control <= 3e-6,
  plan <= 3e-6, objective <= 2.6e-7 on obj ~223, both eq violations <= 6e-13; the control and
  plan tolerances sit ~100x above, the objective and violation ones far higher. On divergence
  the message carries the first diverging step with both solvers' status and constraint
  violation — the signal Phase 9 robustness work consumes.
  """
  config = ClosedLoopConfig.smoke()
  ipopt_run = run_episode(config, solver="ipopt", oracle="alloy")
  sqp_run = run_episode(config, solver="sqp", oracle="alloy")
  assert len(ipopt_run.controls) == len(sqp_run.controls) == config.steps
  eq_fn = chain_eq_function(config.n_masses, config.horizon)
  for k in range(config.steps):
    stats_i, stats_s = ipopt_run.telemetry[k], sqp_run.telemetry[k]
    assert stats_i.status.value <= 1, f"step {k}: ipopt reported {stats_i.status.name}"
    assert stats_s.status.value <= 1, f"step {k}: sqp reported {stats_s.status.name}"
    assert stats_s.iter == 0 or stats_s.alpha > 0.0, f"step {k}: sqp accepted no step (iter={stats_s.iter})"
    du = float(np.max(np.abs(ipopt_run.controls[k] - sqp_run.controls[k])))
    dplan = float(np.max(np.abs(ipopt_run.plans[k] - sqp_run.plans[k])))
    obj_i, obj_s = ipopt_run.telemetry[k].obj, sqp_run.telemetry[k].obj
    viol_i, viol_s = (float(np.max(np.abs(np.asarray(eq_fn(run.oracle_inputs[k]["z"], run.oracle_inputs[k]["p"]))))) for run in (ipopt_run, sqp_run))
    assert du <= 3e-4 and dplan <= 3e-4 and abs(obj_i - obj_s) <= 1e-6 * (1.0 + abs(obj_i)) and max(viol_i, viol_s) <= 1e-6, (
      f"SQP first diverges from IPOPT at step {k}: |du|={du:.3e} |dplan|={dplan:.3e} |dobj|={abs(obj_i - obj_s):.3e}; "
      f"ipopt: status={ipopt_run.telemetry[k].status.name} obj={obj_i:.6e} eq_violation={viol_i:.3e}; "
      f"sqp: status={sqp_run.telemetry[k].status.name} obj={obj_s:.6e} eq_violation={viol_s:.3e}"
    )


def check_recorded_scene() -> None:
  """The runner feeds the 3D scene builders: this problem's end-mass reference once, and on every
  state that has one, the applied control plus the open-loop plan behind it.

  The builders are generic geometry covered by ``tests/viz/test_recording.py``;
  what belongs here is that the chain runner calls them with the chain's own data.
  """
  import tempfile
  from pathlib import Path

  from benchmarks.harness import recording
  from benchmarks.harness.closed_loop import run_chain

  record_chain, record_references = recording.ChainRecorder.record_chain, recording.ChainRecorder.record_chain_references
  record_plan = recording.ChainRecorder.record_chain_plan
  controls: list[list[float] | None] = []
  references: list[dict[str, tuple[float, ...]]] = []
  plans: list[tuple[int, int]] = []

  def spy_chain(self, points, *, control=None, _seen=controls):  # type: ignore[no-untyped-def]
    _seen.append(None if control is None else list(control.applied))
    return record_chain(self, points, control=control)

  def spy_references(self, refs, _seen=references):  # type: ignore[no-untyped-def]
    _seen.append({name: tuple(position) for name, position in refs.items()})
    return record_references(self, refs)

  def spy_plan(self, plan, _seen=plans):  # type: ignore[no-untyped-def]
    logged = record_plan(self, plan)
    _seen.append((len(logged.nodes), logged.n_masses))
    return logged

  recording.ChainRecorder.record_chain, recording.ChainRecorder.record_chain_references = spy_chain, spy_references
  recording.ChainRecorder.record_chain_plan = spy_plan
  try:
    with tempfile.TemporaryDirectory() as directory:
      output = run_chain(smoke=True, out_dir=Path(directory), cli_args=[])
      assert (output / "episode.mcap").stat().st_size > 0, "empty MCAP"
  finally:
    recording.ChainRecorder.record_chain, recording.ChainRecorder.record_chain_references = record_chain, record_references
    recording.ChainRecorder.record_chain_plan = record_plan

  assert references == [{"end-mass reference": END_REF}], references
  config = ClosedLoopConfig.smoke()
  assert len(controls) == config.steps + 1, len(controls)
  assert controls[-1] is None, "the final state has no control to draw"
  assert all(control is not None and len(control) == NU for control in controls[:-1]), controls
  # one open-loop plan per applied control, each covering every node of the horizon it solved
  assert plans == [(config.horizon + 1, config.n_masses)] * config.steps, plans


# name -> (check, requires an IPOPT-backed solve, requires CasADi)
CHECKS: dict[str, tuple[Callable[[], None], bool, bool]] = {
  "dims_and_rk4": (check_dims_and_rk4, False, False),
  "eq_jacobian": (check_eq_jacobian_matches_casadi_and_dense_reference, False, True),
  "nlp_objective": (check_nlp_objective_matches_casadi, True, True),
  "one_reference": (check_one_reference_for_every_end_mass_term, False, False),
  "extract_positions": (check_extract_positions, False, False),
  "plant_step": (check_plant_step_is_parameterized_rk4, False, False),
  "episode_artifacts": (check_episode_artifacts, True, False),
  "sqp_matches_ipopt": (check_sqp_matches_ipopt, True, False),
  "sqp_oracles_agree": (check_sqp_oracles_agree, True, True),
  "recorded_scene": (check_recorded_scene, True, False),
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
