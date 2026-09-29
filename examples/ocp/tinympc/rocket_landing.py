# /// script
# requires-python = ">=3.12"
# dependencies = ["scaly"]
#
# [tool.ty.environment]
# extra-paths = ["."]  # the modules beside this script, which it imports
# ///
"""Rocket soft landing with a thrust cone (Conic-TinyMPC): the second-order-cone example.

A point-mass rocket under gravity follows a straight reference down to the origin. Its thrust is
bounded in a box and in the cone ``|(u_x, u_y)| <= 0.25 u_z``, which ADMM handles by projecting a
slack copy of the inputs onto the cone. The solver is the TinyMPC ADMM Scaly generated for the
problem (``solver.py``); the first steps are checked against the NumPy port of the library's
algorithm, which it matches to rounding. (The library itself computes part of the cone projection
in single precision, so it agrees with both to about 1e-6; see ``benchmark/``.)

    uv run python examples/ocp/tinympc/rocket_landing.py [N]
"""

from __future__ import annotations

import sys

import numpy as np


from problem import ReferenceSolver
from problems import closed_loop, rocket_landing
from solver import Solver


def main(n: int = 32, steps: int | None = None, check_steps: int = 10) -> dict:
  s = rocket_landing(n)
  episode = closed_loop(s, Solver(s.problem, s.bounds), steps)
  check = closed_loop(s, ReferenceSolver(s.problem, s.bounds), min(check_steps, len(episode.x0)))
  k = len(check.x0)
  u = episode.u0
  x_final = s.plant(len(u) - 1, episode.x0[-1], u[-1])
  return {
    "scenario": s,
    "episode": episode,
    "x_final": x_final,
    "cone_violation": np.linalg.norm(u[:, :2], axis=1) - 0.25 * u[:, 2],
    "reference_u0_diff": float(np.abs(check.u0 - episode.u0[:k]).max()),
    "reference_same_iterations": bool(np.array_equal(check.iterations, episode.iterations[:k])),
  }


if __name__ == "__main__":
  n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
  out = main(n)
  ep = out["episode"]
  print(f"{out['scenario'].name}: {len(ep.x0)} steps, {ep.iterations.mean():.1f} ADMM iterations per step (max {ep.iterations.max()})")
  print(f"final position {np.round(out['x_final'][:3], 3).tolist()} m, velocity {np.round(out['x_final'][3:], 3).tolist()} m/s")
  print(f"largest thrust-cone violation of the applied input {out['cone_violation'].max():.2e} (ADMM tolerance 1e-2)")
  print(f"against the NumPy reference: same iterations {out['reference_same_iterations']}, largest input difference {out['reference_u0_diff']:.1e}")
