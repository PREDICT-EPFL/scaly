"""Formulation correctness gates for the race-car problem.

These are properties of *this benchmark problem* — the vendored track data, the
spline reference generator, the physical constants, the transcription's parameter
layout, and the agreement between the Scaly and CasADi implementations of the
same NLP. They run before any timing is recorded, via
``benchmarks/run.py smoke``.

They deliberately do **not** live in ``tests/``: per `AGENTS.md`, the pytest suite
covers Scaly's core and must not depend on a benchmark problem. Where one of these
checks also pins an IR/AD/codegen behaviour, a minimal self-contained reproduction
of that behaviour lives in ``tests/integration/test_stage_transcription.py`` instead, so
this problem can be retired or reshaped without dropping compiler coverage.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterator
import io
from pathlib import Path

import numpy as np

import scaly as sc
from scaly.solvers.paths import solver_loadable, solver_paths
from scaly.solvers.graph import solver_descriptor
from benchmarks.harness import problem_stats, solve_problem
from benchmarks.problems.race_cars import (
  CAR_LENGTH,
  CAR_WIDTH,
  DELTA_MAX,
  NX,
  NZ,
  T_MAX,
  RaceCarParams,
  continuous_dynamics_np,
  n_param,
  race_car_constraint_jac_dense_reference,
  race_car_eq_function,
  rk4_step_np,
)
from benchmarks.problems.race_cars.closed_loop import (
  EpisodeConfig,
  _reference_guess,
  _race_car_nlp,
  build_solver,
  run_episode,
  steady_throttle,
)
from benchmarks.problems.race_cars.reference import MotionPlanner, fit_spline
from benchmarks.problems.race_cars.tracks import CONE_COLORS, load_track, track_names

CANONICAL_TRACK = "fsds_competition_1"
# minimal_tracking_nmpc reports this lap length for the canonical track through its OSQP spline fit
CANONICAL_LAP_LENGTH = 340.21


def check_track_data() -> None:
  """Every vendored track is a closed circuit with both cone walls and a plausible corridor."""
  names = track_names()
  assert CANONICAL_TRACK in names, f"canonical track missing; have {names}"
  for name in names:
    track = load_track(name)
    assert track.center_line.ndim == 2 and track.center_line.shape[1] == 2, name
    assert set(track.cones) == set(CONE_COLORS), name
    assert len(track.cones["blue"]) > 10 and len(track.cones["yellow"]) > 10, name
    spacing = np.linalg.norm(np.diff(track.center_line, axis=0), axis=1)
    closing = np.linalg.norm(track.center_line[0] - track.center_line[-1])
    # closed, and the first/last waypoints are distinct, which is what the spline fit assumes
    assert 0.1 < closing < 3.0 * spacing.max(), f"{name}: closing gap {closing:.3f} m"
    assert 1.5 < track.half_width < 2.0, f"{name}: half width {track.half_width:.3f} m"


def check_spline_fit_is_periodic() -> None:
  """The dependency-free KKT spline fit is closed to machine precision and tracks the waypoints."""
  center_line = load_track(CANONICAL_TRACK).center_line
  coeffs_x, coeffs_y = fit_spline(center_line)
  delta_s = np.concatenate([np.linalg.norm(np.diff(center_line, axis=0), axis=1), [np.linalg.norm(center_line[0] - center_line[-1])]])
  rho = delta_s / np.roll(delta_s, -1)
  for coeffs in (coeffs_x, coeffs_y):
    np.testing.assert_allclose(coeffs.sum(axis=1), np.roll(coeffs[:, 0], -1), atol=1e-9)
    np.testing.assert_allclose(coeffs[:, 1] + 2 * coeffs[:, 2] + 3 * coeffs[:, 3], rho * np.roll(coeffs[:, 1], -1), atol=1e-9)
    np.testing.assert_allclose(2 * coeffs[:, 2] + 6 * coeffs[:, 3], 2 * rho**2 * np.roll(coeffs[:, 2], -1), atol=1e-9)
  knots = np.stack([coeffs_x[:, 0], coeffs_y[:, 0]], axis=1)
  assert np.max(np.linalg.norm(knots - center_line, axis=1)) < 0.1


def check_planner() -> None:
  """Uniform arc-length sampling, the reference lap length, exact projection, forward horizons."""
  horizon, dt, v_ref = 40, 0.05, 5.0
  planner = MotionPlanner(load_track(CANONICAL_TRACK).center_line, horizon=horizon, dt=dt, v_ref=v_ref)
  assert abs(planner.lap_length - CANONICAL_LAP_LENGTH) < 0.05, f"lap length {planner.lap_length:.3f} m"
  assert abs(planner.lap_time - planner.lap_length / v_ref) < 1e-9
  samples = planner.center_path.shape[0]
  spacing = np.linalg.norm(np.diff(planner.center_path, axis=0), axis=1)
  np.testing.assert_allclose(spacing, planner.lap_length / samples, rtol=0.05)
  assert np.all(np.diff(planner.s_ref) > 0.0)
  # unwrapped heading, so the three laid-out laps advance by exactly three counter-clockwise turns
  assert abs((planner.phi_ref[-1] - planner.phi_ref[0]) - 6.0 * np.pi) < 0.1

  for k in range(0, samples, 41):
    expected = float(planner.s_ref[samples + k])
    x, y = float(planner.x_ref[samples + k]), float(planner.y_ref[samples + k])
    assert abs(planner.project(x, y, expected - 4.0) - expected) < 1e-6, f"projection at s={expected:.3f}"

  start = planner.center_path[0]
  s0, reference = planner.plan(float(start[0]), float(start[1]), float(planner.phi_ref[0]), 0.0)
  assert abs(s0) < 1e-6
  assert reference.shape == (horizon + 1, NX)
  assert np.all(reference[:, 3] == v_ref)
  travelled = np.sum(np.linalg.norm(np.diff(reference[:, :2], axis=0), axis=1))
  assert abs(travelled - horizon * dt * v_ref) < 0.01 * horizon * dt * v_ref
  assert np.all(np.abs(reference[:, 2] - planner.phi_ref[0]) < np.pi)


def check_plant_and_geometry() -> None:
  """The NumPy plant holds a steady speed at its feedforward throttle, and the body fits the corridor."""
  params = RaceCarParams(dt=0.02)
  x = np.array([1.0, -2.0, 0.4, 5.0])
  u = np.array([steady_throttle(5.0, params), 0.0])
  stepped = rk4_step_np(x, u, params)
  assert abs(stepped[3] - 5.0) < 1e-6, f"speed drifted to {stepped[3]:.6f}"
  np.testing.assert_allclose((stepped - x) / params.dt, continuous_dynamics_np(x, u, params), rtol=2e-2, atol=2e-2)
  half_step = rk4_step_np(x, u, RaceCarParams(dt=0.5 * params.dt))
  np.testing.assert_allclose(half_step - x, 0.5 * (stepped - x), rtol=5e-3, atol=2e-5)

  config = EpisodeConfig()
  assert CAR_LENGTH > RaceCarParams().wheelbase, "body must extend past the wheelbase"
  assert CAR_WIDTH < 2.0 * config.track_half_width, "body is wider than the corridor"
  assert config.track_half_width < load_track(config.track).half_width, "corridor bound exceeds the real track"


def check_transcription_parameter_layout() -> None:
  """Pins the tail order ``[wheelbase, dt, mass, c_m0, c_r0, c_r1, c_r2]`` of ``p``.

  A distinct value per entry means any permutation of the tail changes the residual, so this
  fails loudly if the symbolic layout and `RaceCarParams.array()` ever disagree.
  """
  horizon = 2
  params = RaceCarParams(wheelbase=2.9, dt=0.13, mass=0.77, c_m0=13.0, c_r0=0.21, c_r1=0.033, c_r2=0.0047)
  rng = np.random.default_rng(4)
  zv = rng.normal(size=NZ * (horizon + 1))
  pv = np.concatenate([rng.normal(size=NX * (horizon + 1)), params.array()])
  parts = [zv[:NX] - pv[:NX]]
  for i in range(horizon):
    zi, znext = zv[i * NZ : (i + 1) * NZ], zv[(i + 1) * NZ : (i + 2) * NZ]
    parts.append(rk4_step_np(zi[:NX], zi[NX:NZ], params) - znext[:NX])
  got = np.asarray(race_car_eq_function(horizon)((zv, pv))).reshape(-1)
  np.testing.assert_allclose(got, np.concatenate(parts), rtol=1e-12, atol=1e-12)


def check_default_constants() -> None:
  """Regression pin on the full-size Formula Student defaults; any constant edit changes these."""
  horizon = 1
  solver = _race_car_nlp(EpisodeConfig(horizon=horizon))
  fn, sparsity = solver_descriptor(solver).jac, solver_descriptor(solver).jac_sparsity
  assert isinstance(fn, sc.Function) and sparsity is not None
  rng = np.random.default_rng(0)
  zv = rng.normal(size=NZ * (horizon + 1))
  pv = np.zeros(n_param(horizon))
  pv[-RaceCarParams().array().size :] = RaceCarParams().array()
  expected = np.zeros((8, 12))
  expected[:4, :4] = np.eye(4)
  expected[4:, :10] = np.array(
    [
      [1.0, 0.0, -0.0029972578986962642, 0.02908579900413167, 1.6317700586218338e-05, -0.0014629269971133468, -1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0027872546854283233, 0.03130680634577367, 1.7564310241585775e-05, 0.0014397288376962168, 0.0, -1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.00978476836121362, 5.489543220680116e-06, 0.002576808134174416, 0.0, 0.0, -1.0, 0.0],
      [0.0, 0.0, 0.0, 0.6984639883035602, 0.0008885917405503103, 0.0021584335486474196, 0.0, 0.0, 0.0, -1.0],
    ]
  )
  actual = np.zeros(sparsity.shape)
  actual[np.asarray(sparsity.rows), np.asarray(sparsity.cols)] = np.asarray(fn((zv, pv))).reshape(-1)
  reference = race_car_constraint_jac_dense_reference(horizon, zv, pv)
  np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(actual[: expected.shape[0]], expected, rtol=1e-12, atol=1e-12)


def check_casadi_mirror_dimensions() -> None:
  """The CasADi mirror declares the same decision vector, rows, and bounds as the Scaly builder."""
  from benchmarks.problems.race_cars.casadi_nlp import build_casadi_race_car_nlp

  config = EpisodeConfig(horizon=4)
  pieces = build_casadi_race_car_nlp(config)
  assert pieces["z"].shape[0] == NZ * (config.horizon + 1)
  assert pieces["n_eq"] == NX * (config.horizon + 1) == pieces["h_eq"].shape[0]
  assert pieces["n_ineq"] == 2 * config.horizon == pieces["g_ineq"].shape[0]
  np.testing.assert_array_equal(pieces["x_lb"][NX : NX + 2], [-T_MAX, -DELTA_MAX])
  np.testing.assert_array_equal(pieces["x_ub"][3::NZ], np.full(config.horizon + 1, config.max_speed))


def check_mapped_cost_matches_casadi() -> None:
  """Initial, running and terminal costs and their derivatives match the unrolled mirror."""
  import casadi as ca

  from benchmarks.problems.race_cars.casadi_nlp import build_casadi_race_car_nlp

  rng = np.random.default_rng(19)
  for horizon in (1, 4):
    config = EpisodeConfig(horizon=horizon)
    descriptor = solver_descriptor(_race_car_nlp(config))
    pieces = build_casadi_race_car_nlp(config)
    z, p, cost = pieces["z"], pieces["p"], pieces["f"]
    reference = ca.Function("cost_reference", [z, p], [cost, ca.gradient(cost, z), ca.hessian(cost, z)[0]])
    zv = rng.normal(size=NZ * (horizon + 1))
    pv = np.concatenate([rng.normal(size=NX * (horizon + 1)), config.params.array()])
    expected = reference(zv, pv)
    np.testing.assert_allclose(descriptor.base((zv, pv))[0], np.asarray(expected[0]).reshape(-1), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(descriptor.grad((zv, pv)), np.asarray(expected[1]).reshape(-1), rtol=1e-12, atol=1e-12)
    sparsity = descriptor.hess_sparsity
    assert descriptor.hess is not None and sparsity is not None
    actual = np.zeros(sparsity.shape)
    actual[np.asarray(sparsity.rows), np.asarray(sparsity.cols)] = np.asarray(
      descriptor.hess(((zv, pv), (np.array(1.0), np.zeros(NX * (horizon + 1) + 2 * horizon))))
    ).reshape(-1)
    actual += np.tril(actual, -1).T
    np.testing.assert_allclose(actual, np.asarray(expected[2]), rtol=1e-12, atol=1e-12)


def check_exact_hessian_default() -> None:
  """The canonical solver asks every provider for exact Lagrangian Hessians."""
  solver = _race_car_nlp(EpisodeConfig.smoke())
  assert solver_descriptor(solver).hess is not None
  assert dict(solver_descriptor(solver).options).get("hessian_approximation") != "limited-memory"
  sqp = _race_car_nlp(EpisodeConfig.smoke(), solver="sqp")
  assert dict(solver_descriptor(sqp).options).get("hessian", "exact") == "exact"


def check_casadi_ipopt_is_compiled() -> None:
  """The timed CasADi column is generated C linked to Scaly's IPOPT."""
  config = EpisodeConfig.smoke()
  controller = build_solver(config, "ipopt", "casadi")
  expected = solver_paths(required=True).loads["ipopt"]
  assert controller.compiled and controller.expand and expected is not None
  assert controller.resolved_ipopt_library.read_bytes() == Path(expected).read_bytes()
  planner = MotionPlanner(load_track(config.track).center_line, horizon=config.horizon, dt=config.params.dt, v_ref=config.v_ref)
  start = planner.center_path[0]
  heading = float(planner.phi_ref[0])
  state = np.array(
    [start[0] - config.initial_lateral_offset * np.sin(heading), start[1] + config.initial_lateral_offset * np.cos(heading), heading, 0.0]
  )
  _, reference = planner.plan(float(state[0]), float(state[1]), float(state[2]), 0.0)
  stage_reference = reference.copy()
  stage_reference[0] = state
  z0 = _reference_guess(reference, config)
  solve_problem(
    controller,
    z0,
    np.zeros(NX * (config.horizon + 1)),
    np.zeros(2 * config.horizon),
    np.zeros(z0.size),
    np.concatenate([stage_reference.reshape(-1), config.params.array()]),
  )
  stats = problem_stats(controller)
  assert stats is not None and stats.n_eval_h > 0


def check_harvested_sqp_globalizations() -> None:
  """Both oracle providers solve the harvested hard-QP race failure."""
  encoded = (Path(__file__).parent / "data" / "step_198.npz.b64").read_text()
  with np.load(io.BytesIO(base64.b64decode(encoded))) as stored:
    inputs = {name: stored[name] for name in ("z0", "lam_eq0", "lam_ineq0", "lam_box0", "p")}
  for oracle in ("scaly", "casadi"):
    for name, options, expected_iter in (
      ("filter", {}, 3),
      ("l1-watchdog-five", {"globalization": "l1", "watchdog": 5, "hessian": "objective"}, 4),
    ):
      solver = build_solver(EpisodeConfig(), "sqp", oracle, sqp_options=options)
      out = solve_problem(solver, inputs["z0"], inputs["lam_eq0"], inputs["lam_ineq0"], inputs["lam_box0"], inputs["p"])
      stats = problem_stats(solver)
      assert stats is not None and stats.status.value == 0, f"sqp/{oracle}/{name}: {stats}"
      assert stats.iter == expected_iter and stats.primal_viol < 1e-7 and stats.alpha == 1.0, f"sqp/{oracle}/{name}: {stats}"
      np.testing.assert_allclose(out["f"], 6.6328204, rtol=1e-7)


def check_episode_artifacts() -> None:
  """A short episode returns consistently shaped, finite, in-bounds, monotone-progress data."""
  config = EpisodeConfig.smoke()
  result = run_episode(config)
  steps = len(result.controls)
  assert 0 < steps <= config.max_steps
  assert result.states.shape == (steps + 1, NX)
  assert result.references.shape == result.states.shape
  assert result.arc_length.shape == (steps + 1,)
  assert result.predictions.shape == (steps, config.horizon + 1, NX)
  assert result.reference_horizons.shape == result.predictions.shape
  assert result.oracle_z.shape == (NZ * (config.horizon + 1),)
  assert result.oracle_p.shape == (n_param(config.horizon),)
  assert len(result.telemetry) == steps
  for array in (result.states, result.references, result.controls, result.predictions, result.reference_horizons):
    assert np.all(np.isfinite(array))
  assert np.all(np.diff(result.arc_length) > -1e-3)
  assert np.all(np.abs(result.controls[:, 0]) <= T_MAX + 1e-9)
  assert np.all(np.abs(result.controls[:, 1]) <= DELTA_MAX + 1e-9)
  for item in result.telemetry:
    assert item.stats.iter >= 0 and item.stats.t_solver >= 0.0 and item.stats.t_fe >= 0.0 and item.stats.t_glue >= 0.0


def check_recorded_scene_and_artifacts() -> None:
  """Both IPOPT oracle providers write the full artifact set and feed the scene builders.

  The scene builders themselves are covered by ``tests/viz/test_recording.py`` (they
  are generic geometry helpers with no problem coupling); what needs checking here is that this
  problem's runner calls them, once for the track and once per step for the two horizons.
  """
  import tempfile

  from benchmarks.harness import recording
  from benchmarks.harness.closed_loop import run_race_cars

  record_track, record_horizons = recording.RaceCarRecorder.record_track, recording.RaceCarRecorder.record_horizons
  for oracle in ("scaly", "casadi"):
    tracks: list[tuple[tuple[int, ...], dict[str, int]]] = []
    horizons: list[list[str]] = []

    def spy_track(self, center_line, cones, _seen=tracks):
      _seen.append((np.asarray(center_line).shape, {name: len(positions) for name, positions in cones.items()}))
      return record_track(self, center_line, cones)

    def spy_horizons(self, paths, _seen=horizons):
      logged = record_horizons(self, paths)
      _seen.append([path.path_id for path in logged])
      return logged

    recording.RaceCarRecorder.record_track, recording.RaceCarRecorder.record_horizons = spy_track, spy_horizons
    try:
      with tempfile.TemporaryDirectory() as directory:
        output = run_race_cars(smoke=True, out_dir=Path(directory), cli_args=[], solver="ipopt", oracle=oracle)
        for name in ("episode.mcap", "rollout.npz", "config.json", "summary.json", "provenance.json", "metadata.json"):
          assert (output / name).is_file(), f"{oracle}: {name} missing"
        assert (output / "episode.mcap").stat().st_size > 0, f"{oracle}: empty MCAP"
        with np.load(output / "rollout.npz") as rollout:
          steps = rollout["control"].shape[0]
          assert rollout["prediction"].shape == rollout["reference_horizon"].shape
          center_line = rollout["center_line"]
        with np.load(output / "representative_fe_inputs.npz") as inputs:
          assert set(inputs.files) == {"p", "z"}
          assert all(np.all(np.isfinite(inputs[name])) for name in inputs.files)
    finally:
      recording.RaceCarRecorder.record_track, recording.RaceCarRecorder.record_horizons = record_track, record_horizons
    assert len(tracks) == 1, f"{oracle}: track logged {len(tracks)} times, expected once"
    shape, cone_counts = tracks[0]
    assert shape == center_line.shape
    assert cone_counts["blue"] > 10 and cone_counts["yellow"] > 10 and cone_counts["big_orange"] > 0
    assert horizons == [["reference", "prediction"]] * steps, f"{oracle}: horizons {horizons}"


def check_oracles_agree() -> None:
  """The core gate for the two-oracle comparison.

  Both oracle providers build the same NLP and hand it to the same IPOPT, so a closed-loop episode must
  come out identical to solver tolerance. Divergence means one of the two symbolic builders has
  drifted from the other, and any timing comparison between them would be meaningless.
  """
  config = EpisodeConfig(horizon=8, max_steps=6)
  scaly_run = run_episode(config, solver="ipopt", oracle="scaly")
  casadi_run = run_episode(config, solver="ipopt", oracle="casadi")
  np.testing.assert_allclose(scaly_run.controls, casadi_run.controls, rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(scaly_run.states, casadi_run.states, rtol=1e-8, atol=1e-8)
  np.testing.assert_allclose(scaly_run.predictions, casadi_run.predictions, rtol=1e-6, atol=1e-6)
  scaly_iters = [item.stats.iter for item in scaly_run.telemetry]
  casadi_iters = [item.stats.iter for item in casadi_run.telemetry]
  assert scaly_iters == casadi_iters, f"iteration counts diverged: {scaly_iters} vs {casadi_iters}"
  for run in (scaly_run, casadi_run):
    for item in run.telemetry:
      assert item.stats.status.value <= 1, f"solve reported {item.stats.status.name}"
      assert item.stats.t_fe > 0.0 and item.stats.n_eval_jac_g > 0, "provider reported no oracle work"


def check_sqp_matches_ipopt() -> None:
  """SQP and IPOPT drive the same episode to per-step solutions that differ only mildly.

  The N=3 smoke point checks the independent solver trajectories without conflating the
  canonical column's fixed N=40 nominal seed. Measured healthy-step differences are control
  <= 7.5e-7, plan <= 5.5e-8, objective <= 9.4e-5 absolute, and violation <= 1.7e-10; the
  tolerances retain the wider envelope established for nonlinear solver agreement. On
  divergence the message carries the first diverging step with both solvers' status and
  constraint violation. The alpha guard rejects a solver handing back the warm start with no
  accepted step.
  """
  config = EpisodeConfig.smoke()
  ipopt_run = run_episode(config, solver="ipopt", oracle="scaly")
  sqp_run = run_episode(config, solver="sqp", oracle="scaly")
  assert len(ipopt_run.controls) == len(sqp_run.controls)
  for k in range(len(ipopt_run.controls)):
    tel_i, tel_s = ipopt_run.telemetry[k], sqp_run.telemetry[k]
    assert tel_i.stats.status.value <= 1, f"step {k}: ipopt reported {tel_i.stats.status.name}"
    assert tel_s.stats.status.value <= 1, f"step {k}: sqp reported {tel_s.stats.status.name}"
    assert tel_s.stats.iter == 0 or tel_s.stats.alpha > 0.0, f"step {k}: sqp accepted no step (iter={tel_s.stats.iter})"
    du = float(np.max(np.abs(ipopt_run.controls[k] - sqp_run.controls[k])))
    dplan = float(np.max(np.abs(ipopt_run.predictions[k] - sqp_run.predictions[k])))
    dobj = abs(tel_i.stats.obj - tel_s.stats.obj)
    viol_i = max(tel_i.eq_violation, tel_i.ineq_violation, tel_i.bound_violation)
    viol_s = max(tel_s.eq_violation, tel_s.ineq_violation, tel_s.bound_violation)
    assert du <= 5e-3 and dplan <= 5e-3 and dobj <= 5e-4 * (1.0 + abs(tel_i.stats.obj)) and max(viol_i, viol_s) <= 2e-5, (
      f"SQP first diverges from IPOPT at step {k}: |du|={du:.3e} |dplan|={dplan:.3e} |dobj|={dobj:.3e}; "
      f"ipopt: status={tel_i.stats.status.name} obj={tel_i.stats.obj:.6e} violation={viol_i:.3e}; "
      f"sqp: status={tel_s.stats.status.name} obj={tel_s.stats.obj:.6e} violation={viol_s:.3e}"
    )


def check_sqp_oracles_agree() -> None:
  """One SQP configuration follows the same smoke trajectory with either C oracle provider."""
  scaly_run = run_episode(smoke=True, solver="sqp", oracle="scaly")
  casadi_run = run_episode(smoke=True, solver="sqp", oracle="casadi")
  np.testing.assert_allclose(scaly_run.controls, casadi_run.controls, rtol=1e-8, atol=1e-8)
  np.testing.assert_allclose(scaly_run.states, casadi_run.states, rtol=1e-10, atol=1e-10)
  for run in (scaly_run, casadi_run):
    for item in run.telemetry:
      stats = item.stats
      assert stats.status.value <= 1 and stats.t_qp > 0.0 and stats.n_eval_h > 0
      np.testing.assert_allclose(stats.t_total, stats.t_fe + stats.t_solver + stats.t_qp + stats.t_globalization + stats.t_glue, rtol=1e-10)


def check_failure_closes_incremental_mcap_and_writes_partial_artifacts() -> None:
  """A failed episode closes the MCAP and retains the completed steps."""
  import json
  import tempfile
  from unittest.mock import patch

  from benchmarks.harness.closed_loop import run_race_cars
  from scaly.solvers import SCALY_SOLVER_STATS_VERSION, ScalySolveStatus, SolverStats
  from benchmarks.problems.race_cars import NX, NU
  from benchmarks.problems.race_cars import closed_loop
  from benchmarks.problems.race_cars.closed_loop import StepRecord, StepTelemetry

  def stats(status: ScalySolveStatus, native: int) -> SolverStats:
    return SolverStats(SCALY_SOLVER_STATS_VERSION, status, native, 2, 1.0, 0.01, 0.002, 0.0, 0.006, 0.001, 0.001, 2, 2, 2, 2, 2)

  def fail(config, *, solver, oracle, record_step, **_kwargs):
    reference = np.zeros((config.horizon + 1, NX))
    prediction = reference.copy()
    good_stats = stats(ScalySolveStatus.OK, 1)
    telemetry = StepTelemetry(good_stats, 0.1, 0, 0.0, 0.0, 0.0)
    record_step(
      StepRecord(
        0,
        np.zeros(NX),
        np.zeros(NX),
        np.zeros(NU),
        prediction,
        reference,
        {"z": np.zeros(1), "p": np.zeros(1)},
        good_stats,
        telemetry,
      )
    )
    failed_stats = stats(ScalySolveStatus.MAX_ITER, -1)
    record_step(StepRecord(1, np.ones(NX), np.ones(NX), None, None, reference, {"z": np.ones(1), "p": np.ones(1)}, failed_stats, None))
    raise RuntimeError(f"synthetic {solver}+{oracle} failure")

  with tempfile.TemporaryDirectory() as directory, patch.object(closed_loop, "run_episode", fail):
    tmp_path = Path(directory)
    try:
      run_race_cars(smoke=True, out_dir=tmp_path, cli_args=["closed-loop", "--problem", "race_cars", "--smoke"])
    except RuntimeError as error:
      assert "synthetic ipopt+scaly failure" in str(error), str(error)
    else:
      raise AssertionError("the failed episode did not raise")

    output = tmp_path / "race_cars" / "ipopt+scaly"
    data = (output / "episode.mcap").read_bytes()
    assert data.startswith(b"\x89MCAP0\r\n") and data.endswith(b"\x89MCAP0\r\n")
    with np.load(output / "rollout.npz") as rollout:
      assert rollout["state"].shape == (2, NX)
      assert rollout["control"].shape == (1, NU)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["failed_step"] == 1 and summary["status"] == "MAX_ITER" and summary["native_status"] == -1


# name -> (check, requires an IPOPT-backed solve, requires CasADi)
CHECKS: dict[str, tuple[Callable[[], None], bool, bool]] = {
  "failure_artifacts": (check_failure_closes_incremental_mcap_and_writes_partial_artifacts, False, False),
  "track_data": (check_track_data, False, False),
  "spline_fit": (check_spline_fit_is_periodic, False, False),
  "planner": (check_planner, False, False),
  "plant_and_geometry": (check_plant_and_geometry, False, False),
  "parameter_layout": (check_transcription_parameter_layout, False, False),
  "default_constants": (check_default_constants, False, False),
  "casadi_mirror": (check_casadi_mirror_dimensions, False, True),
  "mapped_cost": (check_mapped_cost_matches_casadi, False, True),
  "exact_hessian_default": (check_exact_hessian_default, False, False),
  "casadi_ipopt_compiled": (check_casadi_ipopt_is_compiled, True, True),
  "episode_artifacts": (check_episode_artifacts, True, False),
  "recorded_scene": (check_recorded_scene_and_artifacts, True, True),
  "oracles_agree": (check_oracles_agree, True, True),
  "sqp_matches_ipopt": (check_sqp_matches_ipopt, True, False),
  "sqp_oracles_agree": (check_sqp_oracles_agree, True, True),
  "harvested_sqp_globalizations": (check_harvested_sqp_globalizations, True, True),
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
