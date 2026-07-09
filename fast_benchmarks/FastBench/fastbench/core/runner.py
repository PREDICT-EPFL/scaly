"""Benchmark orchestration: build, run Monte-Carlo episodes, collect metrics.

Fairness model: an episode is a pair ``(x0_i, plant_seed_i)``.  The same list
of episodes is used for every solver, and the plant's disturbance RNG is seeded
identically, so each solver sees the *same* initial conditions and the *same*
realised disturbances.  A high-accuracy IPOPT pass computes a per-episode
reference closed-loop cost used to score suboptimality.
"""
from __future__ import annotations

import json
import os
import traceback
from dataclasses import asdict
from typing import List, Optional

import numpy as np

from fastbench.core.env import capture_environment
from fastbench.core.metrics import aggregate_episodes, closed_loop_cost, episode_metrics
from fastbench.core.registry import get_problem, get_solver
from fastbench.core.simulator import run_closed_loop


def _episodes(problem, n_episodes: int, seed: int):
    rng = np.random.default_rng(seed)
    x0s = problem.x0_samples(n_episodes, rng)
    seeds = [int(seed * 100003 + i) for i in range(len(x0s))]
    return list(zip(x0s, seeds))


def _reference_costs(problem, episodes) -> List[float]:
    """Per-episode high-accuracy closed-loop cost (IPOPT, tight tol)."""
    try:
        ref = get_solver("casadi_ipopt_ref")
        ref.build(problem)
    except Exception:
        return [float("nan")] * len(episodes)
    costs = []
    for x0, sd in episodes:
        try:
            log = run_closed_loop(problem, ref, x0, np.random.default_rng(sd))
            costs.append(closed_loop_cost(problem, log))
        except Exception:
            costs.append(float("nan"))
    return costs


def run_benchmark(problems: List[str], solvers: List[str], n_episodes: int = 5,
                  seed: int = 0, outdir: str = "results",
                  ref_costs: bool = True) -> dict:
    os.makedirs(outdir, exist_ok=True)
    env = capture_environment()
    records = []
    path = os.path.join(outdir, "results.json")

    def checkpoint():
        out = {"env": env,
               "config": {"problems": problems, "solvers": solvers,
                          "n_episodes": n_episodes, "seed": seed},
               "records": records}
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2, default=_json_default)

    for pname in problems:
        problem = get_problem(pname)
        episodes = _episodes(problem, n_episodes, seed)
        refc = _reference_costs(problem, episodes) if ref_costs else [None] * len(episodes)

        for sname in solvers:
            rec = {"problem": pname, "solver": sname,
                   "problem_meta": _meta_dict(problem)}
            adapter = get_solver(sname)
            if not adapter.available():
                rec["status"] = "skipped"; rec["reason"] = "solver unavailable"
                records.append(rec); checkpoint()
                print(f"[skip] {pname}/{sname}: unavailable")
                continue
            if not adapter.supports(problem):
                rec["status"] = "skipped"; rec["reason"] = "unsupported (needs LTI/QP or path constraints)"
                records.append(rec); checkpoint()
                print(f"[skip] {pname}/{sname}: unsupported")
                continue
            try:
                build = adapter.build(problem)
            except Exception as e:
                rec["status"] = "error"; rec["reason"] = f"build: {e}"
                rec["traceback"] = traceback.format_exc()
                records.append(rec); checkpoint()
                print(f"[err ] {pname}/{sname}: build failed: {e}")
                continue

            per = []
            for (x0, sd), rc in zip(episodes, refc):
                try:
                    log = run_closed_loop(problem, adapter, x0,
                                          np.random.default_rng(sd))
                    per.append(episode_metrics(problem, log, rc))
                except Exception as e:
                    print(f"[err ] {pname}/{sname}: episode failed: {e}")
            if not per:
                rec["status"] = "error"; rec["reason"] = "all episodes failed"
                records.append(rec); checkpoint(); continue

            rec["status"] = "ok"
            rec["build"] = asdict(build)
            rec["metrics"] = aggregate_episodes(per)
            records.append(rec); checkpoint()
            mt = rec["metrics"]
            print(f"[ ok ] {pname}/{sname}: "
                  f"success={mt.get('episode_success_rate', float('nan')):.2f} "
                  f"t_med={mt.get('solve_time_median_s_mean', float('nan'))*1e3:.2f}ms "
                  f"cost={mt.get('closed_loop_cost_mean', float('nan')):.3g}")

    out = {"env": env, "config": {"problems": problems, "solvers": solvers,
                                  "n_episodes": n_episodes, "seed": seed},
           "records": records}
    path = os.path.join(outdir, "results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=_json_default)
    print(f"\nWrote {path} ({len(records)} records)")
    return out


def _meta_dict(problem) -> dict:
    m = problem.meta
    return {"nx": m.nx, "nu": m.nu, "dt": m.dt, "N": m.N, "n_sim": m.n_sim,
            "is_lti": m.is_lti, "problem_class": m.problem_class,
            "description": m.description}


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
