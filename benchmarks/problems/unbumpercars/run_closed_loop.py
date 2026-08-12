#!/usr/bin/env python3
"""Closed-loop plant + safety-filter runner.

By default both the plant (`--plant`) and the filter's one-step prediction (`--filter-model`)
run the natively discrete `MLPModel`; either can be switched to the RK4 map of the
continuous-time model. Mismatched pairings are what the problem README's mismatch study was
measured on and remain available for that reason.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmarks.harness import CLOSED_LOOP_RESULTS, gbench
from benchmarks.harness.provenance import collect
from benchmarks.harness.recording import (
  CarShape,
  ControlState,
  PlanarVehicleState,
  UnbumpercarsRecorder,
  RunMetadata,
  ScalarTelemetry,
  layout_path,
  write_result_artifacts,
)

from .common import (
  ClosedLoopConfig,
  FilterConfig,
  dt_mlp_step_np,
  load_ct_full_weights,
  load_dt_mlp_weights,
  min_pair_distance,
  normalize_angle,
  rk4_step_np,
  sample_initial_states,
  write_json,
)
from .filters import AlloyDTCBFSafetyFilter, CasadiDTCBFSafetyFilter, FilterStats, OpenLoopFilter, SafetyFilter

PROBLEM = "unbumpercars"
DEFAULT_OUT_DIR = CLOSED_LOOP_RESULTS


def result_dir(out_dir: Path) -> Path:
  return out_dir / PROBLEM


# Plant models, keyed by ClosedLoopConfig.plant. Both take (state, u, dt, weights, physics).
PLANT_MODELS = {"dt": (load_dt_mlp_weights, dt_mlp_step_np), "ct": (load_ct_full_weights, rk4_step_np)}


class Simulator:
  def __init__(self, initial_state: np.ndarray, cfg: ClosedLoopConfig):
    self.states = np.asarray(initial_state, dtype=np.float64).copy()
    self.ncars = cfg.ncars
    self.dt = cfg.dt
    self.physics = cfg.physics
    load_weights, self.plant_step = PLANT_MODELS[cfg.plant]
    self.plant_weights = load_weights()
    self.target_center = cfg.target_center
    self.nominal_speed = cfg.nominal_speed

  def desired_inputs(self) -> np.ndarray:
    out = np.zeros((self.ncars, 2), dtype=np.float64)
    center = np.array([(self.physics.x_min + self.physics.x_max) / 2.0, (self.physics.y_min + self.physics.y_max) / 2.0])
    for i, state in enumerate(self.states):
      out[i, 0] = self.nominal_speed
      if self.target_center:
        angle = np.arctan2(center[1] - state[1], center[0] - state[0])
        out[i, 1] = np.clip(float(normalize_angle(angle - state[2])) / self.physics.max_delta, -1.0, 1.0)
    return np.clip(out, -1.0, 1.0)

  def step(self, safe_inputs: np.ndarray) -> np.ndarray:
    self.states = np.stack(
      [self.plant_step(state, u, self.dt, self.plant_weights, self.physics) for state, u in zip(self.states, safe_inputs, strict=True)]
    )
    return self.states.copy()


def make_filter(kind: str, loop_cfg: ClosedLoopConfig, filt_cfg: FilterConfig, weights):
  if kind == "casadi":
    return CasadiDTCBFSafetyFilter(loop_cfg, filt_cfg, weights)
  if kind == "alloy":
    return AlloyDTCBFSafetyFilter(loop_cfg, filt_cfg, weights)
  if kind == "open":
    return OpenLoopFilter()
  raise ValueError(f"unknown filter kind {kind!r}")


def run_one(kind: str, initial_state: np.ndarray, loop_cfg: ClosedLoopConfig, filt_cfg: FilterConfig, weights, out_dir: Path, dump_alloy_c: bool):
  safety_filter: SafetyFilter = make_filter(kind, loop_cfg, filt_cfg, weights)
  sim = Simulator(initial_state, loop_cfg)
  impl_dir = out_dir / kind
  impl_dir.mkdir(parents=True, exist_ok=True)
  if dump_alloy_c and isinstance(safety_filter, AlloyDTCBFSafetyFilter):
    safety_filter.dump_c(impl_dir / "alloy_c")
  states = sim.states
  state_traj = np.zeros((loop_cfg.steps + 1, loop_cfg.ncars, states.shape[1]), dtype=np.float64)
  desired_traj = np.zeros((loop_cfg.steps, loop_cfg.ncars, 2), dtype=np.float64)
  input_traj = np.zeros((loop_cfg.steps, loop_cfg.ncars, 2), dtype=np.float64)
  fe_inputs = []
  min_dist = float("inf")
  collisions = 0
  state_traj[0] = states
  for t in range(loop_cfg.steps):
    desired = sim.desired_inputs()
    safe = safety_filter.compute_safe_input(states, desired, t)
    desired_traj[t] = desired
    input_traj[t] = safe
    z = getattr(safety_filter, "last_z", None)
    fe_inputs.append(
      {
        "z": np.asarray(z, dtype=np.float64) if z is not None else np.concatenate([safe.reshape(-1), np.zeros(loop_cfg.n_slack)]),
        "bar_x": states.reshape(-1),
        "u_des": desired.reshape(-1),
        "pw": weights.packed,
        "physics": loop_cfg.physics.array(),
        "dt": np.array([loop_cfg.dt]),
      }
    )
    states = sim.step(safe)
    state_traj[t + 1] = states
    step_dist = min_pair_distance(states)
    min_dist = min(min_dist, step_dist)
    collisions += step_dist < loop_cfg.collision_radius

  np.savez_compressed(impl_dir / "rollout.npz", state=state_traj, desired=desired_traj, applied=input_traj)
  write_stats_csv(impl_dir / "stats.csv", safety_filter.stats_history)
  summary = {
    "kind": kind,
    "min_pair_distance": min_dist,
    "collision_steps": collisions,
    "max_consecutive_failures": max_consecutive_failures(safety_filter.stats_history),
    "avg_solver_ms": float(np.mean([s.solver_ms for s in safety_filter.stats_history])),
    "p95_solver_ms": float(np.percentile([s.solver_ms for s in safety_filter.stats_history], 95)),
    "avg_tracking_cost": float(np.mean([s.tracking_cost for s in safety_filter.stats_history])),
    "success_rate": float(np.mean([s.success for s in safety_filter.stats_history])),
    "last_stats": safety_filter.stats_history[-1] if safety_filter.stats_history else None,
  }
  write_json(impl_dir / "summary.json", summary)
  _write_foxglove(kind, loop_cfg, filt_cfg, state_traj, desired_traj, input_traj, safety_filter.stats_history, impl_dir)
  provenance = collect(Path(__file__).resolve().parents[3], gbench.compiler(), sys.argv[1:])
  write_result_artifacts(
    impl_dir,
    config={"loop": asdict(loop_cfg), "filter": asdict(filt_cfg), "weights": str(weights.path), "plant_weights": str(sim.plant_weights.path)},
    summary={key: value for key, value in summary.items() if key != "last_stats"},
    provenance=provenance,
    fe_inputs=fe_inputs,
    successful_steps=[stats.step for stats in safety_filter.stats_history if stats.success],
  )
  return safety_filter, state_traj, desired_traj, input_traj, min_dist, collisions


def max_consecutive_failures(stats: list[FilterStats]) -> int:
  longest = current = 0
  for item in stats:
    current = 0 if item.success else current + 1
    longest = max(longest, current)
  return longest


def _finite(value: float) -> float | None:
  return float(value) if np.isfinite(value) else None


def _write_foxglove(
  kind: str,
  loop_cfg: ClosedLoopConfig,
  filt_cfg: FilterConfig,
  states: np.ndarray,
  desired: np.ndarray,
  applied: np.ndarray,
  stats: list[FilterStats],
  impl_dir: Path,
) -> None:
  physics = loop_cfg.physics
  center = ((physics.x_min + physics.x_max) / 2.0, (physics.y_min + physics.y_max) / 2.0, 0.0)
  # Body: the lf + lr wheelbase plus ~0.3 m of overhang, narrow enough that the pair
  # barrier's keep-out disc circumscribes it. The state position is the centre of
  # gravity, which sits (lf - lr) / 2 behind the geometric centre.
  shape = CarShape(
    length=physics.lf + physics.lr + 0.33,
    width=0.9,
    height=0.5,
    center_offset=0.5 * (physics.lf - physics.lr),
    safety_radius=0.5 * loop_cfg.collision_radius,
    keep_out_radius=0.5 * loop_cfg.safety_radius,
    max_steer=physics.max_delta,
    arrow_length=loop_cfg.safety_radius,
  )
  with UnbumpercarsRecorder(impl_dir / "episode.mcap", allow_overwrite=True, scene_center=center, car_shape=shape) as recorder:
    recorder.record_arena(
      (physics.x_min, physics.x_max, physics.y_min, physics.y_max), margin=loop_cfg.wall_margin if loop_cfg.arena_avoidance else 0.0
    )
    recorder.record_metadata(
      RunMetadata(
        run_id=f"unbumpercars-{kind}-{loop_cfg.seed}",
        problem="unbumpercars",
        backend=kind,
        seed=loop_cfg.seed,
        dt=loop_cfg.dt,
        config={"ncars": loop_cfg.ncars, "steps": loop_cfg.steps, "exact_hessian": not filt_cfg.limited_memory_hessian},
      )
    )
    for step, frame in enumerate(states):
      time_s = step * loop_cfg.dt
      controls = (
        [
          ControlState(
            step=step,
            time_s=time_s,
            entity_id=str(i),
            desired=desired[step, i].tolist(),
            applied=applied[step, i].tolist(),
          )
          for i in range(loop_cfg.ncars)
        ]
        if step < len(stats)
        else []
      )
      recorder.record_planar(
        [
          PlanarVehicleState(step=step, time_s=time_s, vehicle_id=str(i), x=float(state[0]), y=float(state[1]), yaw=float(state[2]))
          for i, state in enumerate(frame)
        ],
        controls=controls,
      )
      if not controls:
        continue
      recorder.record_control(controls)
      item = stats[step]
      recorder.record_telemetry(
        ScalarTelemetry(
          step=step,
          time_s=time_s,
          success=item.success,
          solver_time_ms=max(0.0, item.solver_ms),
          fe_time_ms=max(0.0, item.eval_ms.get("fe_total", 0.0)),
          objective=_finite(item.objective),
          constraint_margin=_finite(item.min_g),
          scalars={
            "iterations": float(item.iterations or 0),
            "glue_time_ms": max(0.0, item.eval_ms.get("glue", 0.0)),
            "slack_l1": item.slack_l1,
            "slack_max": float(item.extra.get("max_slack", 0.0)),
            "tracking_cost": item.tracking_cost,
            "min_pair_distance": min_pair_distance(frame),
          },
        )
      )


def write_stats_csv(path: Path, stats: list[FilterStats]) -> None:
  labels = sorted({k for s in stats for k in s.eval_ms} | {f"n_{k}" for s in stats for k in s.eval_counts})
  fields = [
    "step",
    "impl",
    "success",
    "status",
    "solver_ms",
    "iterations",
    "objective",
    "min_g",
    "slack_l1",
    "tracking_cost",
    *labels,
  ]
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for s in stats:
      row = {
        "step": s.step,
        "impl": s.impl,
        "success": s.success,
        "status": s.status,
        "solver_ms": s.solver_ms,
        "iterations": "" if s.iterations is None else s.iterations,
        "objective": s.objective,
        "min_g": s.min_g,
        "slack_l1": s.slack_l1,
        "tracking_cost": s.tracking_cost,
      }
      row.update(s.eval_ms)
      row.update({f"n_{k}": v for k, v in s.eval_counts.items()})
      writer.writerow(row)


def plot_outputs(results: dict[str, tuple], loop_cfg: ClosedLoopConfig, out_dir: Path, show: bool) -> None:
  import matplotlib

  if not show:
    matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.patches import Circle

  colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(loop_cfg.ncars, 2)))
  for kind, (_, state_traj, _, _, min_dist, collisions) in results.items():
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(f"{kind}: neural DT plant + HCBF safety filter\nmin pair distance = {min_dist:.3f} m, {collisions} colliding steps")
    ax.set_xlim(loop_cfg.physics.x_min, loop_cfg.physics.x_max)
    ax.set_ylim(loop_cfg.physics.y_min, loop_cfg.physics.y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    for i in range(loop_cfg.ncars):
      xy = state_traj[:, i, :2]
      ax.plot(xy[:, 0], xy[:, 1], color=colors[i], lw=1.5, label=f"car {i}")
      ax.plot(xy[0, 0], xy[0, 1], marker="o", color=colors[i], ms=5)
      ax.plot(xy[-1, 0], xy[-1, 1], marker="x", color=colors[i], ms=7)
      ax.add_patch(Circle((xy[-1, 0], xy[-1, 1]), loop_cfg.collision_radius / 2.0, color=colors[i], alpha=0.08))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / kind / "trajectories.png", dpi=160)

    stats = results[kind][0].stats_history
    if stats:
      fig, axs = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
      t = np.array([s.step for s in stats])
      axs[0].plot(t, [s.solver_ms for s in stats], lw=1.2)
      axs[0].set_ylabel("solve [ms]")
      iterations = np.array([np.nan if s.iterations is None else s.iterations for s in stats], dtype=np.float64)
      axs[1].plot(t, iterations, lw=1.2)
      axs[1].set_ylabel("iters")
      axs[2].plot(t, [s.slack_l1 for s in stats], lw=1.2)
      axs[2].set_ylabel("L1 slack")
      eval_keys = sorted({k for s in stats for k in s.eval_ms})
      for key in eval_keys:
        axs[3].plot(t, [s.eval_ms.get(key, np.nan) for s in stats], lw=1.0, label=key)
      axs[3].set_ylabel("eval [ms]")
      axs[3].set_xlabel("step")
      if eval_keys:
        axs[3].legend(ncols=4, fontsize=8)
      for ax in axs:
        ax.grid(True, alpha=0.25)
      fig.suptitle(f"{kind} instrumentation")
      fig.tight_layout()
      fig.savefig(out_dir / kind / "performance.png", dpi=160)
  if show:
    plt.show()


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(
    description="Closed-loop centralized DTCBF safety filter: the natively discrete MLP as both the plant and the filter's model by default."
  )
  p.add_argument("--filter", choices=["casadi", "alloy", "both", "open"], default="casadi")
  p.add_argument(
    "--plant",
    choices=list(PLANT_MODELS),
    default="dt",
    help="Model the plant steps with: dt = the natively discrete MLPModel, ct = the RK4 map of the continuous-time model.",
  )
  p.add_argument(
    "--filter-model",
    choices=list(PLANT_MODELS),
    default="dt",
    help="Model the filter predicts one step ahead with. dt uses the discrete MLP, smoothed; see common.dt_mlp_step_smooth_np.",
  )
  p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="closed-loop artifact root; the problem and backend are appended")
  p.add_argument("--ncars", type=int, default=8)
  p.add_argument("--steps", type=int, default=200)
  p.add_argument("--dt", type=float, default=0.1)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--safety-radius", type=float, default=2.28)
  p.add_argument("--collision-radius", type=float, default=1.9)
  p.add_argument("--pair-gamma", type=float, default=0.35)
  p.add_argument("--wall-gamma", type=float, default=0.35)
  p.add_argument("--nominal-speed", type=float, default=0.55)
  p.add_argument("--no-walls", action="store_true")
  p.add_argument("--ipopt-max-iter", type=int, default=300)
  p.add_argument("--ipopt-tol", type=float, default=1e-6)
  p.add_argument(
    "--limited-memory-hessian",
    action="store_true",
    help="Use IPOPT's limited-memory Hessian approximation on both backends instead of exact Lagrangian Hessians.",
  )
  p.add_argument("--no-casadi-expand", action="store_true", help="Disable CasADi MX-to-SX expansion before constructing the NLP solver.")
  p.add_argument("--eval-repeats", type=int, default=1)
  p.add_argument("--dump-alloy-c", action="store_true")
  p.add_argument("--show", action="store_true")
  return p.parse_args()


def main() -> None:
  args = parse_args()
  out_dir = result_dir(args.out_dir)
  weights = load_dt_mlp_weights() if args.filter_model == "dt" else load_ct_full_weights()
  loop_cfg = ClosedLoopConfig(
    ncars=args.ncars,
    steps=args.steps,
    dt=args.dt,
    seed=args.seed,
    plant=args.plant,
    safety_radius=args.safety_radius,
    collision_radius=args.collision_radius,
    pair_gamma=args.pair_gamma,
    wall_gamma=args.wall_gamma,
    arena_avoidance=not args.no_walls,
    nominal_speed=args.nominal_speed,
  )
  filt_cfg = FilterConfig(
    model=args.filter_model,
    ipopt_max_iter=args.ipopt_max_iter,
    ipopt_tol=args.ipopt_tol,
    eval_repeats=max(1, args.eval_repeats),
    limited_memory_hessian=args.limited_memory_hessian,
    casadi_expand=not args.no_casadi_expand,
  )
  out_dir.mkdir(parents=True, exist_ok=True)
  initial = sample_initial_states(loop_cfg)
  write_json(
    out_dir / "config.json",
    {"loop": asdict(loop_cfg), "filter": asdict(filt_cfg), "weights": str(weights.path), "initial_state": initial},
  )
  kinds = ["casadi", "alloy"] if args.filter == "both" else [args.filter]
  results = {}
  for kind in kinds:
    print(
      f"[run] {kind}  ncars={loop_cfg.ncars} steps={loop_cfg.steps} walls={loop_cfg.arena_avoidance}"
      f" plant={loop_cfg.plant} filter_model={filt_cfg.model}"
    )
    result = run_one(kind, initial, loop_cfg, filt_cfg, weights, out_dir, args.dump_alloy_c)
    results[kind] = result
    controller, min_dist, collisions = result[0], result[4], result[5]
    stats = controller.stats_history
    avg_solve = float(np.mean([s.solver_ms for s in stats])) if stats else float("nan")
    p95_solve = float(np.percentile([s.solver_ms for s in stats], 95)) if stats else float("nan")
    avg_track = float(np.mean([s.tracking_cost for s in stats])) if stats else float("nan")
    print(
      f"  min_pair_distance={min_dist:.3f}m  collision_steps={collisions}  max_consec_fail={max_consecutive_failures(stats)}"
      f"  avg_solve={avg_solve:.2f}ms  p95_solve={p95_solve:.2f}ms  avg_tracking={avg_track:.4g}"
    )
    if stats:
      last = stats[-1]
      print(f"  last status={last.status} success={last.success} iter={last.iterations} min_g={last.min_g:.3e} slack_l1={last.slack_l1:.3e}")
      if last.eval_ms:
        print("  eval ms: " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(last.eval_ms.items())))
      if last.extra:
        print(f"  extra: {last.extra}")
  plot_outputs(results, loop_cfg, out_dir, args.show)
  print(f"[done] outputs in {out_dir}")
  print(f"[foxglove] layout {layout_path('unbumpercars')}, episodes in {out_dir}/<filter>/episode.mcap")


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    sys.exit(130)
