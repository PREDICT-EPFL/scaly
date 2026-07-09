"""Metric computation from episode logs.

Three families of metrics are produced:

* **solve quality** - per-step optimizer behaviour (suboptimality vs a
  high-accuracy reference, KKT residual, constraint violation, success rate);
* **control quality** - closed-loop performance (cost, tracking RMSE, terminal
  error, settling time, control effort, input rate, constraint violations);
* **timing / real-time** - solve-time distribution and deadline-miss rate
  (fraction of steps whose solve time exceeds the sampling period ``dt``).

Build/codegen metrics (sizes, generation/compile times) are attached
separately from :class:`fastbench.solvers.base.BuildInfo`.
"""
from __future__ import annotations

from typing import List

import numpy as np

from fastbench.core.simulator import EpisodeLog


def _percentile(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def closed_loop_cost(problem, log: EpisodeLog) -> float:
    """Sum of stage costs evaluated numerically along the realised trajectory."""
    import casadi as ca
    nx, nu = problem.meta.nx, problem.meta.nu
    xs, us = ca.SX.sym("x", nx), ca.SX.sym("u", nu)
    xr, ur = ca.SX.sym("xr", nx), ca.SX.sym("ur", nu)
    lfun = ca.Function("l", [xs, us, xr, ur],
                       [problem.stage_cost(xs, us, xr, ur)])
    total = 0.0
    for k in range(len(log.U)):
        xref, uref = problem.reference_traj(k)
        total += float(lfun(log.X[k], log.U[k], xref, uref))
    return total


def tracking_rmse(log: EpisodeLog) -> float:
    err = log.X - log.Xref
    return float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))


def terminal_error(log: EpisodeLog) -> float:
    return float(np.linalg.norm(log.X[-1] - log.Xref[-1]))


def settling_time(problem, log: EpisodeLog, frac: float = 0.05) -> float:
    """Time [s] after which the state stays within a scaled band of the goal."""
    goal = log.Xref[-1]
    tol = frac * (np.abs(goal) + 1.0)
    inside = np.all(np.abs(log.X - goal) <= tol, axis=1)
    n = len(inside)
    last_outside = 0
    for i in range(n):
        if not inside[i]:
            last_outside = i
    if inside[-1] is np.False_ or not inside[-1]:
        return float("nan")           # never settled
    return last_outside * problem.meta.dt


def control_effort(log: EpisodeLog) -> float:
    return float(np.sum(log.U ** 2))


def input_rate(log: EpisodeLog) -> float:
    if len(log.U) < 2:
        return 0.0
    return float(np.sum(np.diff(log.U, axis=0) ** 2))


def state_constraint_violation(problem, log: EpisodeLog) -> float:
    lbx, ubx = problem.bounds()[0], problem.bounds()[1]
    below = np.maximum(lbx - log.X, 0.0)
    above = np.maximum(log.X - ubx, 0.0)
    return float(np.max(np.concatenate([below.flatten(), above.flatten(), [0.0]])))


def timing_summary(stats) -> dict:
    t = np.array([s.solve_time_s for s in stats if np.isfinite(s.solve_time_s)])
    iters = np.array([s.iterations for s in stats if s.iterations >= 0])
    return {
        "solve_time_min_s": float(np.min(t)) if len(t) else float("nan"),
        "solve_time_median_s": _percentile(t, 50),
        "solve_time_mean_s": float(np.mean(t)) if len(t) else float("nan"),
        "solve_time_p95_s": _percentile(t, 95),
        "solve_time_p99_s": _percentile(t, 99),
        "solve_time_max_s": float(np.max(t)) if len(t) else float("nan"),
        "iter_median": float(np.median(iters)) if len(iters) else float("nan"),
        "iter_max": int(np.max(iters)) if len(iters) else -1,
    }


def episode_metrics(problem, log: EpisodeLog, ref_cost: float = None) -> dict:
    stats = log.stats
    n = max(len(stats), 1)
    n_success = sum(1 for s in stats if s.success)
    dt = problem.meta.dt
    deadline_misses = sum(1 for s in stats
                          if np.isfinite(s.solve_time_s) and s.solve_time_s > dt)
    kkt = np.array([s.kkt_residual for s in stats if np.isfinite(s.kkt_residual)])
    cviol = np.array([s.constraint_violation for s in stats
                      if np.isfinite(s.constraint_violation)])

    cl_cost = closed_loop_cost(problem, log)
    m = {
        # control quality
        "closed_loop_cost": cl_cost,
        "tracking_rmse": tracking_rmse(log),
        "terminal_error": terminal_error(log),
        "settling_time_s": settling_time(problem, log),
        "control_effort": control_effort(log),
        "input_rate": input_rate(log),
        "state_constraint_violation": state_constraint_violation(problem, log),
        "episode_success": bool(problem.episode_success(log.X, log.U)),
        # solve quality
        "step_success_rate": n_success / n,
        "deadline_miss_rate": deadline_misses / n,
        "kkt_residual_max": float(np.max(kkt)) if len(kkt) else float("nan"),
        "ocp_constraint_violation_max": float(np.max(cviol)) if len(cviol) else float("nan"),
    }
    if ref_cost is not None and np.isfinite(ref_cost) and ref_cost != 0:
        m["suboptimality"] = (cl_cost - ref_cost) / abs(ref_cost)
    m.update(timing_summary(stats))
    return m


def aggregate_episodes(per_episode: List[dict]) -> dict:
    """Mean/std/worst across Monte-Carlo episodes."""
    if not per_episode:
        return {}
    keys = per_episode[0].keys()
    out = {"n_episodes": len(per_episode)}
    for k in keys:
        vals = np.array([e[k] for e in per_episode], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            out[f"{k}_mean"] = float("nan")
            continue
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
        out[f"{k}_worst"] = float(np.max(vals))
    # success rates as plain fractions
    out["episode_success_rate"] = float(np.mean(
        [1.0 if e["episode_success"] else 0.0 for e in per_episode]))
    return out
