"""Random QP-MPC (TinyMPC, ICRA 2024): tracking a random trajectory with bounded inputs.

A random stable and controllable ``(A, B)`` with ``nx`` states and ``nu`` inputs tracks a feasible
random trajectory under ``|u| <= 3``, with a small noise on the state, for 190 steps. Every step
calls the TinyMPC solver Scaly generated for this problem (``solver.py``), warm-started from the
previous step. The first steps are also solved by the NumPy port of the library's algorithm
(``problem.reference_solve``), which must take the same iterations and return the same input.

    uv run examples/tinympc/random_mpc.py [nx nu N]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from problem import ReferenceSolver
from problems import closed_loop, random_mpc
from solver import Solver


def main(nx: int = 10, nu: int = 4, n: int = 10, steps: int | None = None, check_steps: int = 10) -> dict:
  s = random_mpc(nx, nu, n)
  episode = closed_loop(s, Solver(s.problem, s.bounds), steps)
  check = closed_loop(s, ReferenceSolver(s.problem, s.bounds), min(check_steps, len(episode.x0)))
  k = len(check.x0)
  xbar = np.array([s.references(i, episode.x0[i])[0][0] for i in range(len(episode.x0))])
  return {
    "scenario": s,
    "episode": episode,
    "tracking_error": np.linalg.norm(episode.x0 - xbar, axis=1),
    "reference_u0_diff": float(np.abs(check.u0 - episode.u0[:k]).max()),
    "reference_same_iterations": bool(np.array_equal(check.iterations, episode.iterations[:k])),
  }


if __name__ == "__main__":
  nx, nu, n = (int(a) for a in sys.argv[1:4]) if len(sys.argv) > 3 else (10, 4, 10)
  out = main(nx, nu, n)
  ep = out["episode"]
  print(
    f"{out['scenario'].name}: {len(ep.x0)} steps, {ep.iterations.mean():.1f} ADMM iterations per step (max {ep.iterations.max()}), all converged: {ep.solved.all()}"
  )
  print(f"largest |u| applied {np.abs(ep.u0).max():.3f} (bound 3), tracking error mean {out['tracking_error'].mean():.3f}")
  print(f"against the NumPy reference: same iterations {out['reference_same_iterations']}, largest input difference {out['reference_u0_diff']:.1e}")
