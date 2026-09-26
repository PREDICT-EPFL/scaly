"""A predictive safety filter (TinyMPC, CDC 2024 benchmarks): a double integrator kept in its box.

A PD controller chases a sinusoid of amplitude 2 in each of ``nx / 2`` axes, which a box
``|x| <= 1.5`` forbids. At each step the controller's inputs over the horizon (from a rollout of
the nominal model) are the reference of an MPC problem with ``R = 100``, ``Q = 0``: the filter
returns the admissible input sequence closest to them, and its first input is applied. The solver
is the TinyMPC ADMM Scaly generated for the problem (``solver.py``); the first steps are checked
against the NumPy port of the library's algorithm.

    uv run examples/tinympc/safety_filter.py [nx N]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from problem import ReferenceSolver
from problems import closed_loop, safety_filter
from solver import Solver


def main(nx: int = 4, n: int = 20, steps: int | None = None, check_steps: int = 10) -> dict:
  s = safety_filter(nx, n)
  episode = closed_loop(s, Solver(s.problem, s.bounds), steps)
  check = closed_loop(s, ReferenceSolver(s.problem, s.bounds), min(check_steps, len(episode.x0)))
  k = len(check.x0)
  modified = np.abs(episode.u0 - episode.uref[:, 0]).max(axis=1) > 1e-6
  return {
    "scenario": s,
    "episode": episode,
    "modified": modified,
    "reference_u0_diff": float(np.abs(check.u0 - episode.u0[:k]).max()),
    "reference_same_iterations": bool(np.array_equal(check.iterations, episode.iterations[:k])),
  }


if __name__ == "__main__":
  nx, n = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (4, 20)
  out = main(nx, n)
  ep = out["episode"]
  print(f"{out['scenario'].name}: {len(ep.x0)} steps, {ep.iterations.mean():.1f} ADMM iterations per step (max {ep.iterations.max()})")
  print(f"the filter changed the PD input at {out['modified'].sum()} of {len(ep.x0)} steps")
  print(f"largest |x| {np.abs(ep.x0).max():.3f} (box 1.5, tolerance 1e-2), largest |u| {np.abs(ep.u0).max():.3f} (bound 2)")
  print(f"against the NumPy reference: same iterations {out['reference_same_iterations']}, largest input difference {out['reference_u0_diff']:.1e}")
