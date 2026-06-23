#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from common import (
  DEFAULT_MODEL_PATH,
  ClosedLoopConfig,
  FilterConfig,
  desired_inputs_to_center,
  load_ct_full_weights,
  min_pair_distance,
  rk4_step_np,
  sample_initial_states,
  write_json,
)
from filters import AlloyDTCBFSafetyFilter, CasadiDTCBFSafetyFilter, FilterStats, OpenLoopFilter, SafetyFilter


def make_filter(kind: str, loop_cfg: ClosedLoopConfig, filt_cfg: FilterConfig, weights):
  if kind == "casadi":
    return CasadiDTCBFSafetyFilter(loop_cfg, filt_cfg, weights)
  if kind == "alloy":
    return AlloyDTCBFSafetyFilter(loop_cfg, filt_cfg, weights)
  if kind == "open":
    return OpenLoopFilter()
  raise ValueError(f"unknown filter kind {kind!r}")


def run_one(kind: str, initial_state: np.ndarray, loop_cfg: ClosedLoopConfig, filt_cfg: FilterConfig, weights, out_dir: Path, dump_alloy_c: bool):
  controller: SafetyFilter = make_filter(kind, loop_cfg, filt_cfg, weights)
  impl_dir = out_dir / kind
  impl_dir.mkdir(parents=True, exist_ok=True)
  if dump_alloy_c and isinstance(controller, AlloyDTCBFSafetyFilter):
    controller.dump_c(impl_dir / "alloy_c")
  states = initial_state.copy()
  state_traj = np.zeros((loop_cfg.horizon + 1, loop_cfg.ncars, states.shape[1]), dtype=np.float64)
  desired_traj = np.zeros((loop_cfg.horizon, loop_cfg.ncars, 2), dtype=np.float64)
  input_traj = np.zeros((loop_cfg.horizon, loop_cfg.ncars, 2), dtype=np.float64)
  min_dist = float("inf")
  state_traj[0] = states
  for t in range(loop_cfg.horizon):
    desired = desired_inputs_to_center(states, loop_cfg)
    safe = controller.compute_safe_input(states, desired, t)
    desired_traj[t] = desired
    input_traj[t] = safe
    next_states = np.zeros_like(states)
    for i in range(loop_cfg.ncars):
      next_states[i] = rk4_step_np(states[i], safe[i], loop_cfg.dt, weights, loop_cfg.physics)
    states = next_states
    state_traj[t + 1] = states
    min_dist = min(min_dist, min_pair_distance(states))

  np.savez_compressed(impl_dir / "rollout.npz", state=state_traj, desired=desired_traj, applied=input_traj)
  write_stats_csv(impl_dir / "stats.csv", controller.stats_history)
  write_json(
    impl_dir / "summary.json",
    {
      "kind": kind,
      "min_pair_distance": min_dist,
      "avg_solver_ms": float(np.mean([s.solver_ms for s in controller.stats_history])),
      "p95_solver_ms": float(np.percentile([s.solver_ms for s in controller.stats_history], 95)),
      "avg_tracking_cost": float(np.mean([s.tracking_cost for s in controller.stats_history])),
      "success_rate": float(np.mean([s.success for s in controller.stats_history])),
      "last_stats": controller.stats_history[-1] if controller.stats_history else None,
    },
  )
  return controller, state_traj, desired_traj, input_traj, min_dist


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
    "slack",
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
        "slack": s.slack,
        "tracking_cost": s.tracking_cost,
      }
      row.update(s.eval_ms)
      row.update({f"n_{k}": v for k, v in s.eval_counts.items()})
      writer.writerow(row)


def plot_outputs(
  results: dict[str, tuple[SafetyFilter, np.ndarray, np.ndarray, np.ndarray, float]], loop_cfg: ClosedLoopConfig, out_dir: Path, show: bool
) -> None:
  import matplotlib

  if not show:
    matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.patches import Circle

  colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(loop_cfg.ncars, 2)))
  for kind, (_, state_traj, _, _, min_dist) in results.items():
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title(f"{kind}: CTFull RK4 simulator + DTCBF safety filter\nmin pair distance = {min_dist:.3f} m")
    ax.set_xlim(loop_cfg.physics.x_min, loop_cfg.physics.x_max)
    ax.set_ylim(loop_cfg.physics.y_min, loop_cfg.physics.y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    for i in range(loop_cfg.ncars):
      xy = state_traj[:, i, :2]
      ax.plot(xy[:, 0], xy[:, 1], color=colors[i], lw=1.5, label=f"car {i}")
      ax.plot(xy[0, 0], xy[0, 1], marker="o", color=colors[i], ms=5)
      ax.plot(xy[-1, 0], xy[-1, 1], marker="x", color=colors[i], ms=7)
      ax.add_patch(Circle((xy[-1, 0], xy[-1, 1]), loop_cfg.safety_radius / 2.0, color=colors[i], alpha=0.08))
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
      axs[2].plot(t, [s.slack for s in stats], lw=1.2)
      axs[2].set_ylabel("slack")
      eval_keys = sorted({k for s in stats for k in s.eval_ms})
      for key in eval_keys:
        axs[3].plot(t, [s.eval_ms.get(key, np.nan) for s in stats], lw=1.0, label=key)
      axs[3].set_ylabel("eval [ms]")
      axs[3].set_xlabel("step")
      axs[3].legend(ncols=4, fontsize=8)
      for ax in axs:
        ax.grid(True, alpha=0.25)
      fig.suptitle(f"{kind} instrumentation")
      fig.tight_layout()
      fig.savefig(out_dir / kind / "performance.png", dpi=160)
  if show:
    plt.show()


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Closed-loop centralized DTCBF safety filter using the bumper_car_simulator CTFull xlarge model.")
  p.add_argument("--filter", choices=["casadi", "alloy", "both", "open"], default="casadi")
  p.add_argument("--weights", type=Path, default=DEFAULT_MODEL_PATH)
  p.add_argument("--out-dir", type=Path, default=Path("examples/ct_dt_cbf_filter/out"))
  p.add_argument("--ncars", type=int, default=4)
  p.add_argument("--horizon", type=int, default=80)
  p.add_argument("--dt", type=float, default=0.1)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--safety-radius", type=float, default=1.9)
  p.add_argument("--pair-gamma", type=float, default=0.35)
  p.add_argument("--wall-gamma", type=float, default=0.35)
  p.add_argument("--nominal-speed", type=float, default=0.55)
  p.add_argument("--no-walls", action="store_true")
  p.add_argument("--ipopt-max-iter", type=int, default=300)
  p.add_argument("--ipopt-tol", type=float, default=1e-6)
  p.add_argument(
    "--exact-hessian",
    action="store_true",
    help="Use/evaluate exact Hessians in CasADi. Alloy stays limited-memory because reverse AD through MAP Hessians is not available here.",
  )
  p.add_argument("--eval-repeats", type=int, default=1)
  p.add_argument("--dump-alloy-c", action="store_true")
  p.add_argument("--show", action="store_true")
  return p.parse_args()


def main() -> None:
  args = parse_args()
  weights = load_ct_full_weights(args.weights)
  loop_cfg = ClosedLoopConfig(
    ncars=args.ncars,
    horizon=args.horizon,
    dt=args.dt,
    seed=args.seed,
    safety_radius=args.safety_radius,
    pair_gamma=args.pair_gamma,
    wall_gamma=args.wall_gamma,
    arena_avoidance=not args.no_walls,
    nominal_speed=args.nominal_speed,
  )
  filt_cfg = FilterConfig(
    ipopt_max_iter=args.ipopt_max_iter,
    ipopt_tol=args.ipopt_tol,
    eval_repeats=max(1, args.eval_repeats),
    limited_memory_hessian=not args.exact_hessian,
  )
  args.out_dir.mkdir(parents=True, exist_ok=True)
  initial = sample_initial_states(loop_cfg)
  write_json(
    args.out_dir / "config.json",
    {"loop": asdict(loop_cfg), "filter": asdict(filt_cfg), "weights": str(weights.path), "initial_state": initial},
  )
  kinds = ["casadi", "alloy"] if args.filter == "both" else [args.filter]
  results = {}
  for kind in kinds:
    print(f"[run] {kind}  ncars={loop_cfg.ncars} horizon={loop_cfg.horizon} walls={loop_cfg.arena_avoidance}")
    controller, state, desired, applied, min_dist = run_one(kind, initial, loop_cfg, filt_cfg, weights, args.out_dir, args.dump_alloy_c)
    results[kind] = (controller, state, desired, applied, min_dist)
    stats = controller.stats_history
    avg_solve = float(np.mean([s.solver_ms for s in stats])) if stats else float("nan")
    p95_solve = float(np.percentile([s.solver_ms for s in stats], 95)) if stats else float("nan")
    avg_track = float(np.mean([s.tracking_cost for s in stats])) if stats else float("nan")
    print(f"  min_pair_distance={min_dist:.3f}m  avg_solve={avg_solve:.2f}ms  p95_solve={p95_solve:.2f}ms  avg_tracking={avg_track:.4g}")
    if stats:
      last = stats[-1]
      print(f"  last status={last.status} success={last.success} iter={last.iterations} min_g={last.min_g:.3e} slack={last.slack:.3e}")
      if last.eval_ms:
        print("  eval ms: " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(last.eval_ms.items())))
      if last.extra:
        print(f"  extra: {last.extra}")
  plot_outputs(results, loop_cfg, args.out_dir, args.show)
  print(f"[done] outputs in {args.out_dir}")


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    sys.exit(130)
