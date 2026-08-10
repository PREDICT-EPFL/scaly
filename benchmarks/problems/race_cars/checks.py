"""Formulation correctness gates for the race-car problem.

These are properties of *this benchmark problem* — the vendored track data, the
spline reference generator, the physical constants, the transcription's parameter
layout, and the agreement between the Alloy and CasADi implementations of the
same NLP. They run before any timing is recorded, via
``benchmarks/run.py smoke``.

They deliberately do **not** live in ``tests/``: per `AGENTS.md`, the pytest suite
covers Alloy's core and must not depend on a benchmark problem. Where one of these
checks also pins an IR/AD/codegen behaviour, a minimal self-contained reproduction
of that behaviour lives in ``tests/alloy/test_stage_transcription.py`` instead, so
this problem can be retired or reshaped without dropping compiler coverage.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

from alloy.toolchain import solver_loadable
from benchmarks.problems.race_cars import (
  CAR_LENGTH,
  CAR_WIDTH,
  DELTA_MAX,
  NX,
  NZ,
  T_MAX,
  RaceCarParams,
  n_param,
  race_car_eq_function,
)
from benchmarks.problems.race_cars.closed_loop import (
  EpisodeConfig,
  continuous_dynamics_np,
  rk4_step_np,
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
  got = np.asarray(race_car_eq_function(horizon)(zv, pv)).reshape(-1)
  np.testing.assert_allclose(got, np.concatenate(parts), rtol=1e-12, atol=1e-12)


def check_default_constants() -> None:
  """Regression pin on the full-size Formula Student defaults; any constant edit changes these."""
  horizon = 1
  fn = race_car_eq_function(horizon).factory("race_car_default_params_jac", ["z", "p"], ["jac:eq:z"])
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
  np.testing.assert_allclose(fn(zv, pv), expected, rtol=1e-12, atol=1e-12)


def check_casadi_mirror_dimensions() -> None:
  """The CasADi mirror declares the same decision vector, rows, and bounds as the Alloy builder."""
  from benchmarks.problems.race_cars.casadi_nlp import build_casadi_race_car_nlp

  config = EpisodeConfig(horizon=4)
  pieces = build_casadi_race_car_nlp(config)
  assert pieces["z"].shape[0] == NZ * (config.horizon + 1)
  assert pieces["n_eq"] == NX * (config.horizon + 1) == pieces["h_eq"].shape[0]
  assert pieces["n_ineq"] == 2 * config.horizon == pieces["g_ineq"].shape[0]
  np.testing.assert_array_equal(pieces["x_lb"][NX : NX + 2], [-T_MAX, -DELTA_MAX])
  np.testing.assert_array_equal(pieces["x_ub"][3::NZ], np.full(config.horizon + 1, config.max_speed))


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
  """Both backends write the full artifact set, and the runner really feeds the scene builders.

  The scene builders themselves are covered by ``tests/alloy/test_benchmark_recording.py`` (they
  are generic geometry helpers with no problem coupling); what needs checking here is that this
  problem's runner calls them, once for the track and once per step for the two horizons.
  """
  import tempfile

  from benchmarks.harness import recording
  from benchmarks.harness.closed_loop import run_race_cars

  record_track, record_horizons = recording.Recorder.record_track, recording.Recorder.record_horizons
  for backend in ("alloy", "casadi"):
    tracks: list[tuple[tuple[int, ...], dict[str, int]]] = []
    horizons: list[list[str]] = []

    def spy_track(self, center_line, cones, _seen=tracks):
      _seen.append((np.asarray(center_line).shape, {name: len(positions) for name, positions in cones.items()}))
      return record_track(self, center_line, cones)

    def spy_horizons(self, paths, _seen=horizons):
      logged = record_horizons(self, paths)
      _seen.append([path.path_id for path in logged])
      return logged

    recording.Recorder.record_track, recording.Recorder.record_horizons = spy_track, spy_horizons
    try:
      with tempfile.TemporaryDirectory() as directory:
        output = run_race_cars(smoke=True, out_dir=Path(directory), cli_args=[], backend=backend)
        for name in ("episode.mcap", "rollout.npz", "config.json", "summary.json", "provenance.json", "metadata.json"):
          assert (output / name).is_file(), f"{backend}: {name} missing"
        assert (output / "episode.mcap").stat().st_size > 0, f"{backend}: empty MCAP"
        with np.load(output / "rollout.npz") as rollout:
          steps = rollout["control"].shape[0]
          assert rollout["prediction"].shape == rollout["reference_horizon"].shape
          center_line = rollout["center_line"]
        with np.load(output / "representative_fe_inputs.npz") as inputs:
          assert set(inputs.files) == {"p", "z"}
          assert all(np.all(np.isfinite(inputs[name])) for name in inputs.files)
    finally:
      recording.Recorder.record_track, recording.Recorder.record_horizons = record_track, record_horizons
    assert len(tracks) == 1, f"{backend}: track logged {len(tracks)} times, expected once"
    shape, cone_counts = tracks[0]
    assert shape == center_line.shape
    assert cone_counts["blue"] > 10 and cone_counts["yellow"] > 10 and cone_counts["big_orange"] > 0
    assert horizons == [["reference", "prediction"]] * steps, f"{backend}: horizons {horizons}"


def check_backends_agree() -> None:
  """The core gate for the two-oracle comparison.

  Both backends build the same NLP and hand it to the same IPOPT, so a closed-loop episode must
  come out identical to solver tolerance. Divergence means one of the two symbolic builders has
  drifted from the other, and any timing comparison between them would be meaningless.
  """
  config = EpisodeConfig(horizon=8, max_steps=6)
  alloy_run = run_episode(config, backend="alloy")
  casadi_run = run_episode(config, backend="casadi")
  np.testing.assert_allclose(alloy_run.controls, casadi_run.controls, rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(alloy_run.states, casadi_run.states, rtol=1e-8, atol=1e-8)
  np.testing.assert_allclose(alloy_run.predictions, casadi_run.predictions, rtol=1e-6, atol=1e-6)
  alloy_iters = [item.stats.iter for item in alloy_run.telemetry]
  casadi_iters = [item.stats.iter for item in casadi_run.telemetry]
  assert alloy_iters == casadi_iters, f"iteration counts diverged: {alloy_iters} vs {casadi_iters}"
  for run in (alloy_run, casadi_run):
    for item in run.telemetry:
      assert item.stats.status.value <= 1, f"solve reported {item.stats.status.name}"
      assert item.stats.t_fe > 0.0 and item.stats.n_eval_jac_g > 0, "backend reported no oracle work"


# name -> (check, requires an IPOPT-backed solve, requires CasADi)
CHECKS: dict[str, tuple[Callable[[], None], bool, bool]] = {
  "track_data": (check_track_data, False, False),
  "spline_fit": (check_spline_fit_is_periodic, False, False),
  "planner": (check_planner, False, False),
  "plant_and_geometry": (check_plant_and_geometry, False, False),
  "parameter_layout": (check_transcription_parameter_layout, False, False),
  "default_constants": (check_default_constants, False, False),
  "casadi_mirror": (check_casadi_mirror_dimensions, False, True),
  "episode_artifacts": (check_episode_artifacts, True, False),
  "recorded_scene": (check_recorded_scene_and_artifacts, True, True),
  "backends_agree": (check_backends_agree, True, True),
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
