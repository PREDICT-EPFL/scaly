from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmarks.harness import gbench
from benchmarks.harness.provenance import collect
from benchmarks.harness.recording import (
  CarShape,
  ControlState,
  HorizonPath,
  PlanarVehicleState,
  PointState3D,
  Recorder,
  RunMetadata,
  ScalarTelemetry,
  write_result_artifacts,
)

ROOT = Path(__file__).resolve().parents[2]


def _finite(value: float) -> float | None:
  return float(value) if np.isfinite(value) else None


def _lateral_error(state: np.ndarray, reference: np.ndarray) -> float:
  """Signed offset from the reference point, across the reference heading."""
  dx, dy = state[0] - reference[0], state[1] - reference[1]
  return float(-np.sin(reference[2]) * dx + np.cos(reference[2]) * dy)


def _provenance(cli_args: list[str]) -> dict[str, object]:
  return collect(ROOT, gbench.compiler(), cli_args)


def run_chain(*, smoke: bool, out_dir: Path, cli_args: list[str]) -> Path:
  from benchmarks.problems.chain_of_masses import HORIZON, NU, n_dec, n_state
  from benchmarks.problems.chain_of_masses.closed_loop import ClosedLoopConfig, plant_step, run_episode

  config = ClosedLoopConfig.smoke() if smoke else ClosedLoopConfig.canonical()
  episode = run_episode(config)
  output = out_dir / "chain"
  output.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output / "rollout.npz", state=episode.states, control=episode.controls, points=episode.points)
  chain_center = tuple(np.mean(episode.points[0], axis=0).tolist())
  with Recorder(output / "episode.mcap", allow_overwrite=True, scene_center=chain_center) as recorder:
    recorder.record_metadata(
      RunMetadata(
        run_id=f"chain-M{config.n_masses}-N{config.horizon}",
        problem="chain_of_masses",
        backend="alloy-ipopt",
        seed=0,
        dt=config.params.dt,
        config={"n_masses": config.n_masses, "horizon": config.horizon, "steps": config.steps},
      )
    )
    for step, points in enumerate(episode.points):
      time_s = step * config.params.dt
      recorder.record_chain(
        [
          PointState3D(step=step, time_s=time_s, point_id=str(i), x=float(point[0]), y=float(point[1]), z=float(point[2]))
          for i, point in enumerate(points)
        ]
      )
      if step >= len(episode.telemetry):
        continue
      recorder.record_control([ControlState(step=step, time_s=time_s, entity_id="tip", applied=episode.controls[step].tolist())])
      stats = episode.telemetry[step]
      recorder.record_telemetry(
        ScalarTelemetry(
          step=step,
          time_s=time_s,
          success=stats.status.value <= 1,
          solver_time_ms=stats.t_total * 1000.0,
          fe_time_ms=stats.t_fe * 1000.0,
          objective=_finite(stats.obj),
          scalars={"iterations": float(stats.iter), "qp_or_solver_time_ms": stats.t_solver * 1000.0, "glue_time_ms": stats.t_glue * 1000.0},
        )
      )
  representative_step = len(episode.controls) // 2
  benchmark_z = np.empty(n_dec(config.n_masses, HORIZON), dtype=np.float64)
  nx, nz = n_state(config.n_masses), n_state(config.n_masses) + NU
  state = episode.states[representative_step].copy()
  control = episode.controls[representative_step].copy()
  benchmark_p = np.concatenate([state, config.params.array()])
  for stage in range(HORIZON):
    benchmark_z[stage * nz : stage * nz + nx] = state
    benchmark_z[stage * nz + nx : (stage + 1) * nz] = control
    state = plant_step(state, control, config.params)
  benchmark_z[HORIZON * nz :] = state
  summary = {
    "problem": "chain_of_masses",
    "n_masses": config.n_masses,
    "horizon": config.horizon,
    "steps": config.steps,
    "mean_solver_ms": float(np.mean([stats.t_total for stats in episode.telemetry]) * 1000.0),
    "max_control": float(np.max(np.abs(episode.controls))),
  }
  write_result_artifacts(
    output,
    config=asdict(config),
    summary=summary,
    provenance=_provenance(cli_args),
    fe_inputs=[{"z": benchmark_z, "p": benchmark_p}] * len(episode.oracle_inputs),
    successful_steps=range(len(episode.oracle_inputs)),
  )
  return output


def run_race_cars(*, smoke: bool, out_dir: Path, cli_args: list[str], backend: str = "alloy") -> Path:
  from benchmarks.problems.race_cars import CAR_HEIGHT, CAR_LENGTH, CAR_WIDTH, DELTA_MAX, T_MAX, WHEELBASE
  from benchmarks.problems.race_cars.closed_loop import EpisodeConfig, run_episode

  config = EpisodeConfig.smoke() if smoke else EpisodeConfig()
  episode = run_episode(config, backend=backend)
  steps = len(episode.controls)
  output = out_dir / "race_cars" / backend
  output.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    output / "rollout.npz",
    state=episode.states,
    reference=episode.references,
    control=episode.controls,
    arc_length=episode.arc_length,
    laps=episode.laps,
    prediction=episode.predictions,
    reference_horizon=episode.reference_horizons,
    center_line=episode.center_path,
  )
  shape = CarShape(
    length=CAR_LENGTH,
    width=CAR_WIDTH,
    height=CAR_HEIGHT,
    max_throttle=T_MAX,
    max_steer=DELTA_MAX,
    wheelbase=WHEELBASE,
  )
  centroid = np.mean(episode.center_path, axis=0)
  with Recorder(
    output / "episode.mcap", allow_overwrite=True, scene_center=(float(centroid[0]), float(centroid[1]), 0.0), car_shape=shape
  ) as recorder:
    recorder.record_metadata(
      RunMetadata(
        run_id=f"race_cars-{backend}-{config.track}-N{config.horizon}",
        problem="race_cars",
        backend=f"{backend}-ipopt",
        seed=0,
        dt=config.params.dt,
        config={"track": config.track, "horizon": config.horizon, "steps": steps, "v_ref": config.v_ref, "lap_length": episode.lap_length},
      )
    )
    recorder.record_track(episode.center_path, episode.track.cones)
    for step, (state, reference) in enumerate(zip(episode.states, episode.references, strict=True)):
      time_s = step * config.params.dt
      controls = [ControlState(step=step, time_s=time_s, entity_id="vehicle", applied=episode.controls[step].tolist())] if step < steps else []
      # Only the plant gets a body: the reference pose is already the head of the reference horizon,
      # and drawing it as a second car reads as a second vehicle on the track.
      recorder.record_planar(
        [PlanarVehicleState(step=step, time_s=time_s, vehicle_id="vehicle", x=float(state[0]), y=float(state[1]), yaw=float(state[2]))],
        controls=controls,
      )
      if step >= steps:
        continue
      recorder.record_control(controls)
      recorder.record_horizons(
        [
          HorizonPath(
            step=step,
            time_s=time_s,
            path_id=path_id,
            x=horizon[:, 0].tolist(),
            y=horizon[:, 1].tolist(),
            yaw=horizon[:, 2].tolist(),
          )
          for path_id, horizon in (("reference", episode.reference_horizons[step]), ("prediction", episode.predictions[step]))
        ]
      )
      item = episode.telemetry[step]
      stats = item.stats
      recorder.record_telemetry(
        ScalarTelemetry(
          step=step,
          time_s=time_s,
          success=stats.status.value <= 1,
          solver_time_ms=stats.t_total * 1000.0,
          fe_time_ms=stats.t_fe * 1000.0,
          objective=_finite(stats.obj),
          scalars={
            "iterations": float(stats.iter),
            "qp_or_solver_time_ms": stats.t_solver * 1000.0,
            "glue_time_ms": stats.t_glue * 1000.0,
            "arc_length": item.arc_length,
            "laps": float(item.laps),
            "speed": float(state[3]),
            "lateral_error": _lateral_error(state, reference),
          },
        )
      )
  lateral = np.array([_lateral_error(state, reference) for state, reference in zip(episode.states, episode.references, strict=True)])
  summary = {
    "problem": "race_cars",
    "backend": backend,
    "track": config.track,
    "horizon": config.horizon,
    "steps": steps,
    "lap_length_m": episode.lap_length,
    "lap_time_s": steps * config.params.dt,
    "laps": int(episode.laps[-1]),
    "mean_solver_ms": float(np.mean([item.stats.t_total for item in episode.telemetry]) * 1000.0),
    # the FE share is the point of the two-backend comparison, so keep it in the headline summary
    "mean_fe_ms": float(np.mean([item.stats.t_fe for item in episode.telemetry]) * 1000.0),
    "mean_iterations": float(np.mean([item.stats.iter for item in episode.telemetry])),
    "rms_lateral_error": float(np.sqrt(np.mean(lateral**2))),
    "max_abs_lateral_error": float(np.max(np.abs(lateral))),
  }
  write_result_artifacts(
    output,
    config=asdict(config),
    summary=summary,
    provenance=_provenance(cli_args),
    fe_inputs=episode.oracle_inputs,
    successful_steps=range(len(episode.oracle_inputs)),
  )
  return output


def run_bumpercars(*, smoke: bool, backend: str, out_dir: Path, cli_args: list[str]) -> Path:
  from benchmarks.problems.bumpercars_filter.common import ClosedLoopConfig, FilterConfig, load_ct_full_weights, sample_initial_states
  from benchmarks.problems.bumpercars_filter.run_closed_loop import plot_outputs, run_one

  loop = ClosedLoopConfig(ncars=2, horizon=2) if smoke else ClosedLoopConfig()
  filt = FilterConfig(ipopt_max_iter=40) if smoke else FilterConfig()
  weights = load_ct_full_weights()
  initial = sample_initial_states(loop)
  result = run_one(backend, initial, loop, filt, weights, out_dir, False)
  plot_outputs({backend: result}, loop, out_dir, False)
  return out_dir / backend


def run(problem: str, *, smoke: bool, backend: str, out_dir: Path, cli_args: list[str]) -> Path:
  if problem == "bumpercars":
    return run_bumpercars(smoke=smoke, backend=backend, out_dir=out_dir / problem, cli_args=cli_args)
  if problem == "race_cars":
    return run_race_cars(smoke=smoke, out_dir=out_dir, cli_args=cli_args, backend=backend)
  return run_chain(smoke=smoke, out_dir=out_dir, cli_args=cli_args)
