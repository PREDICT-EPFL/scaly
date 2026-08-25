from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from benchmarks.harness import ROOT, gbench, solver_oracle_name
from benchmarks.harness.provenance import collect
from benchmarks.harness.recording import (
  CarShape,
  ChainPlan,
  ChainRecorder,
  ControlState,
  FurutaPlan,
  FurutaShape,
  FurutaState,
  HorizonPath,
  NpmpcRecorder,
  PlanarVehicleState,
  PointState3D,
  RaceCarRecorder,
  RunMetadata,
  ScalarTelemetry,
  write_result_artifacts,
)


def _finite(value: float) -> float | None:
  return float(value) if np.isfinite(value) else None


def _lateral_error(state: np.ndarray, reference: np.ndarray) -> float:
  """Signed offset from the reference point, across the reference heading."""
  dx, dy = state[0] - reference[0], state[1] - reference[1]
  return float(-np.sin(reference[2]) * dx + np.cos(reference[2]) * dy)


def _provenance(cli_args: list[str]) -> dict[str, object]:
  return collect(ROOT, gbench.compiler(), cli_args)


def run_chain(*, smoke: bool, out_dir: Path, cli_args: list[str], solver: str = "ipopt", oracle: str = "alloy") -> Path:
  from benchmarks.problems.chain import END_REF, HORIZON, NU, n_dec, n_state
  from benchmarks.problems.chain.closed_loop import ClosedLoopConfig, plant_step, run_episode

  config = ClosedLoopConfig.smoke() if smoke else ClosedLoopConfig.canonical()
  episode = run_episode(config, solver=solver, oracle=oracle)
  name = solver_oracle_name(solver, oracle)
  output = out_dir / "chain" / name
  output.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output / "rollout.npz", state=episode.states, control=episode.controls, points=episode.points, plan=episode.plans)
  chain_center = tuple(np.mean(episode.points[0], axis=0).tolist())
  with ChainRecorder(output / "episode.mcap", allow_overwrite=True, scene_center=chain_center) as recorder:
    recorder.record_metadata(
      RunMetadata(
        run_id=f"chain-M{config.n_masses}-N{config.horizon}",
        problem="chain",
        solver=solver,
        oracle=oracle,
        seed=0,
        dt=config.params.dt,
        config={"n_masses": config.n_masses, "horizon": config.horizon, "steps": config.steps},
      )
    )
    recorder.record_chain_references({"end-mass reference": END_REF})
    for step, points in enumerate(episode.points):
      time_s = step * config.params.dt
      applied = (
        ControlState(step=step, time_s=time_s, entity_id="tip", applied=episode.controls[step].tolist()) if step < len(episode.telemetry) else None
      )
      recorder.record_chain(
        [
          PointState3D(step=step, time_s=time_s, point_id=str(i), x=float(point[0]), y=float(point[1]), z=float(point[2]))
          for i, point in enumerate(points)
        ],
        control=applied,
      )
      if applied is None:
        continue
      recorder.record_control([applied])
      recorder.record_chain_plan(
        ChainPlan(
          step=step,
          time_s=time_s,
          n_masses=config.n_masses,
          nodes=[node.reshape(-1).tolist() for node in episode.plans[step]],
        )
      )
      stats = episode.telemetry[step]
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
            "qp_or_solver_time_ms": (stats.t_solver + stats.t_qp) * 1000.0,
            "globalization_time_ms": stats.t_globalization * 1000.0,
            "glue_time_ms": stats.t_glue * 1000.0,
          },
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
    "problem": "chain",
    "solver": solver,
    "oracle": oracle,
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


def run_race_cars(*, smoke: bool, out_dir: Path, cli_args: list[str], solver: str = "ipopt", oracle: str = "alloy") -> Path:
  from benchmarks.problems.race_cars import CAR_HEIGHT, CAR_LENGTH, CAR_WIDTH, DELTA_MAX, T_MAX, WHEELBASE
  from benchmarks.problems.race_cars.closed_loop import EpisodeConfig, StepRecord, run_episode
  from benchmarks.problems.race_cars.reference import MotionPlanner
  from benchmarks.problems.race_cars.tracks import load_track

  config = EpisodeConfig.smoke() if smoke else EpisodeConfig()
  name = solver_oracle_name(solver, oracle)
  output = out_dir / "race_cars" / name
  output.mkdir(parents=True, exist_ok=True)
  shape = CarShape(
    length=CAR_LENGTH,
    width=CAR_WIDTH,
    height=CAR_HEIGHT,
    max_throttle=T_MAX,
    max_steer=DELTA_MAX,
    wheelbase=WHEELBASE,
  )
  track = load_track(config.track)
  planner = MotionPlanner(track.center_line, horizon=config.horizon, dt=config.params.dt, v_ref=config.v_ref)
  centroid = np.mean(planner.center_path, axis=0)
  records: list[StepRecord] = []
  episode = None
  failure: RuntimeError | None = None
  with RaceCarRecorder(
    output / "episode.mcap", allow_overwrite=True, scene_center=(float(centroid[0]), float(centroid[1]), 0.0), car_shape=shape
  ) as recorder:
    recorder.record_metadata(
      RunMetadata(
        run_id=f"race_cars-{name}-{config.track}-N{config.horizon}",
        problem="race_cars",
        solver=solver,
        oracle=oracle,
        seed=0,
        dt=config.params.dt,
        config={
          "track": config.track,
          "horizon": config.horizon,
          "max_steps": config.max_steps,
          "v_ref": config.v_ref,
          "lap_length": planner.lap_length,
        },
      )
    )
    recorder.record_track(planner.center_path, track.cones)

    def record(item: StepRecord) -> None:
      records.append(item)
      time_s = item.step * config.params.dt
      controls = [ControlState(step=item.step, time_s=time_s, entity_id="vehicle", applied=item.control.tolist())] if item.control is not None else []
      recorder.record_planar(
        [
          PlanarVehicleState(
            step=item.step,
            time_s=time_s,
            vehicle_id="vehicle",
            x=float(item.state[0]),
            y=float(item.state[1]),
            yaw=float(item.state[2]),
          )
        ],
        controls=controls,
      )
      if controls:
        recorder.record_control(controls)
      if item.prediction is not None:
        recorder.record_horizons(
          [
            HorizonPath(
              step=item.step,
              time_s=time_s,
              path_id=path_id,
              x=horizon[:, 0].tolist(),
              y=horizon[:, 1].tolist(),
              yaw=horizon[:, 2].tolist(),
            )
            for path_id, horizon in (("reference", item.reference_horizon), ("prediction", item.prediction))
          ]
        )
      if item.stats is not None:
        stats = item.stats
        scalars = {
          "iterations": float(stats.iter),
          "native_status": float(stats.native_status),
          "qp_or_solver_time_ms": (stats.t_solver + stats.t_qp) * 1000.0,
          "globalization_time_ms": stats.t_globalization * 1000.0,
          "glue_time_ms": stats.t_glue * 1000.0,
          "speed": float(item.state[3]),
          "lateral_error": _lateral_error(item.state, item.reference),
        }
        if item.telemetry is not None:
          scalars.update(arc_length=item.telemetry.arc_length, laps=float(item.telemetry.laps))
        recorder.record_telemetry(
          ScalarTelemetry(
            step=item.step,
            time_s=time_s,
            success=item.control is not None and stats.status.value <= 1,
            solver_time_ms=stats.t_total * 1000.0,
            fe_time_ms=stats.t_fe * 1000.0,
            objective=_finite(stats.obj),
            scalars=scalars,
          )
        )

    try:
      episode = run_episode(config, solver=solver, oracle=oracle, record_step=record)
    except RuntimeError as exc:
      if not records:
        raise
      failure = exc

  if episode is None:
    successful = [item for item in records if item.control is not None and item.telemetry is not None]
    terminal = records[-1]
    states = np.asarray([item.state for item in successful] + [terminal.state])
    references = np.asarray([item.reference for item in successful] + [terminal.reference])
    controls = np.asarray([item.control for item in successful])
    predictions = np.asarray([item.prediction for item in successful])
    reference_horizons = np.asarray([item.reference_horizon for item in successful])
    successful_telemetry = [item.telemetry for item in successful if item.telemetry is not None]
    arc_length = np.asarray([0.0, *(item.arc_length for item in successful_telemetry)])
    laps = np.asarray([0, *(item.laps for item in successful_telemetry)], dtype=np.int64)
    np.savez_compressed(
      output / "rollout.npz",
      state=states,
      reference=references,
      control=controls,
      arc_length=arc_length,
      laps=laps,
      prediction=predictions,
      reference_horizon=reference_horizons,
      center_line=planner.center_path,
    )
    failed_stats = terminal.stats
    if terminal.oracle_input is not None:
      np.savez_compressed(output / "failing_solver_inputs.npz", **terminal.oracle_input)
    summary = {
      "problem": "race_cars",
      "solver": solver,
      "oracle": oracle,
      "failed_step": terminal.step,
      "status": "missing" if failed_stats is None else failed_stats.status.name,
      "native_status": None if failed_stats is None else failed_stats.native_status,
      "completed_steps": len(successful),
    }
    fe_inputs = [item.oracle_input for item in successful if item.oracle_input is not None]
    if fe_inputs:
      write_result_artifacts(
        output,
        config=asdict(config),
        summary=summary,
        provenance=_provenance(cli_args),
        fe_inputs=fe_inputs,
        successful_steps=range(len(fe_inputs)),
      )
    else:
      for name, document in (("config", asdict(config)), ("summary", summary), ("provenance", _provenance(cli_args))):
        (output / f"{name}.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    assert failure is not None
    raise failure

  steps = len(episode.controls)
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
  lateral = np.array([_lateral_error(state, reference) for state, reference in zip(episode.states, episode.references, strict=True)])
  summary = {
    "problem": "race_cars",
    "solver": solver,
    "oracle": oracle,
    "track": config.track,
    "horizon": config.horizon,
    "steps": steps,
    "lap_length_m": episode.lap_length,
    "lap_time_s": steps * config.params.dt,
    "laps": int(episode.laps[-1]),
    "mean_solver_ms": float(np.mean([item.stats.t_total for item in episode.telemetry]) * 1000.0),
    # the FE share is the point of the two-oracle comparison, so keep it in the headline summary
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


def run_npmpc(*, smoke: bool, out_dir: Path, cli_args: list[str], solver: str = "ipopt", oracle: str = "alloy") -> Path:
  from benchmarks.problems.npmpc import PHI_LIMIT, PLANT_SUBSTEPS, TORQUE_LIMIT
  from benchmarks.problems.npmpc.closed_loop import EpisodeConfig, run_episode, settling_step, upright_error

  config = EpisodeConfig.smoke() if smoke else EpisodeConfig()
  episode = run_episode(config, solver=solver, oracle=oracle)
  name = solver_oracle_name(solver, oracle)
  output = out_dir / "npmpc" / name
  output.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    output / "rollout.npz",
    state=episode.states,
    control=episode.controls,
    prediction=episode.predictions,
    plan=episode.plans,
    slack=episode.slacks,
    terminal_weight=episode.terminal_weight,
  )
  shape = FurutaShape(
    arm_length=config.plant.l_r,
    pendulum_length=config.plant.l_p,
    # a torque on its bound draws an arrow half the arm's length
    torque_scale=0.5 * config.plant.l_r / TORQUE_LIMIT,
    arm_limit=PHI_LIMIT,
  )
  with NpmpcRecorder(output / "episode.mcap", allow_overwrite=True, shape=shape) as recorder:
    recorder.record_metadata(
      RunMetadata(
        run_id=f"npmpc-{name}-N{config.horizon}",
        problem="npmpc",
        solver=solver,
        oracle=oracle,
        seed=0,
        dt=config.dt,
        config={"horizon": config.horizon, "steps": config.steps, "decoder": list(config.decoder.hidden), "substeps": config.substeps},
      )
    )
    recorder.record_furuta_static()
    for step, state in enumerate(episode.states):
      time_s = step * config.dt
      torque = float(episode.controls[step, 0]) if step < config.steps else 0.0
      recorder.record_furuta(FurutaState(step=step, time_s=time_s, theta=float(state[0]), phi=float(state[1]), torque=torque))
      if step >= config.steps:
        continue
      control = ControlState(step=step, time_s=time_s, entity_id="arm", applied=[torque])
      recorder.record_control([control])
      recorder.record_furuta_plan(
        FurutaPlan(
          step=step,
          time_s=time_s,
          theta=episode.predictions[step, :, 0].tolist(),
          phi=episode.predictions[step, :, 1].tolist(),
        )
      )
      stats = episode.telemetry[step]
      recorder.record_telemetry(
        ScalarTelemetry(
          step=step,
          time_s=time_s,
          success=stats.status.value <= 1,
          solver_time_ms=stats.t_total * 1000.0,
          fe_time_ms=stats.t_fe * 1000.0,
          objective=_finite(stats.obj),
          # the arm angle's distance from the bound the single slack softens, so an active bound shows
          constraint_margin=float(PHI_LIMIT - abs(state[1])),
          scalars={
            "iterations": float(stats.iter),
            "upright_error_deg": float(np.rad2deg(upright_error(state))),
            "arm_angle": float(state[1]),
            "pendulum_rate": float(state[2]),
            "arm_rate": float(state[3]),
            "slack": float(episode.slacks[step]),
            "qp_or_solver_time_ms": (stats.t_solver + stats.t_qp) * 1000.0,
            "globalization_time_ms": stats.t_globalization * 1000.0,
            "glue_time_ms": stats.t_glue * 1000.0,
          },
        )
      )
  settled = settling_step(episode.states)
  summary = {
    "problem": "npmpc",
    "solver": solver,
    "oracle": oracle,
    "horizon": config.horizon,
    "steps": config.steps,
    "decoder": "x".join(str(width) for width in config.decoder.hidden),
    "plant_substeps": PLANT_SUBSTEPS,
    "settling_step": settled,
    "settling_time_s": None if settled is None else settled * config.dt,
    "final_upright_error_deg": float(np.rad2deg(upright_error(episode.states[-1]))),
    "mean_solver_ms": float(np.mean([stats.t_total for stats in episode.telemetry]) * 1000.0),
    # the reason this problem is in the suite: with 65 variables the FE share dominates the solve
    "mean_fe_ms": float(np.mean([stats.t_fe for stats in episode.telemetry]) * 1000.0),
    "mean_iterations": float(np.mean([stats.iter for stats in episode.telemetry])),
    "max_abs_control": float(np.max(np.abs(episode.controls))),
    "max_slack": float(np.max(episode.slacks)),
  }
  write_result_artifacts(
    output,
    config=asdict(config),
    summary=summary,
    provenance=_provenance(cli_args),
    fe_inputs=[{"z": item["z"], "p": item["p"]} for item in episode.oracle_inputs],
    successful_steps=range(len(episode.oracle_inputs)),
  )
  return output


def run_unbumpercars(*, smoke: bool, solver: str, oracle: str | None, out_dir: Path, cli_args: list[str]) -> Path:
  from benchmarks.problems.unbumpercars.common import ClosedLoopConfig, FilterConfig, load_dt_mlp_weights, sample_initial_states
  from benchmarks.problems.unbumpercars.run_closed_loop import plot_outputs, run_one

  loop = ClosedLoopConfig(ncars=2, steps=2) if smoke else ClosedLoopConfig()
  filt = FilterConfig(ipopt_max_iter=40) if smoke else FilterConfig()
  # filt.model defaults to "dt", so the filter reads the discrete MLP's weights
  weights = load_dt_mlp_weights()
  initial = sample_initial_states(loop)
  name = solver_oracle_name(solver, oracle)
  result = run_one(solver, oracle, initial, loop, filt, weights, out_dir, False)
  plot_outputs({name: result}, loop, out_dir, False)
  return out_dir / name


def run(problem: str, *, smoke: bool, solver: str, oracle: str | None, out_dir: Path, cli_args: list[str]) -> Path:
  if problem == "unbumpercars":
    return run_unbumpercars(smoke=smoke, solver=solver, oracle=oracle, out_dir=out_dir / problem, cli_args=cli_args)
  if oracle is None:
    raise ValueError(f"{problem} requires an oracle")
  if problem == "race_cars":
    return run_race_cars(smoke=smoke, out_dir=out_dir, cli_args=cli_args, solver=solver, oracle=oracle)
  if problem == "npmpc":
    return run_npmpc(smoke=smoke, out_dir=out_dir, cli_args=cli_args, solver=solver, oracle=oracle)
  return run_chain(smoke=smoke, out_dir=out_dir, cli_args=cli_args, solver=solver, oracle=oracle)
