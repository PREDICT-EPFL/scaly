"""M1: an OCP built by `scaly.mpc` against the same problem written by hand, solved by IPOPT.

    uv run internal/notes/perf_2026_09_27_mpc/bench_ocp.py [--rounds 15]

The cart-pole of `tests/mpc/test_ocp.py` (the formulation of `examples/nmpc_cartpole.py`: N = 20,
two RK4 substeps, soft track limits), from X0 = (0.3, 0.4, 0, 0) and the same cold start each time.
Per variant: IPOPT's iterations and its own total time (`solver_stats().t_total`, the fastest over
interleaved rounds), the wall time of the Python call, and the time to build and compile the
solver. Variants: the hand-written problem; `scaly.mpc` with the same transcription; and, for
context, `scaly.mpc` with Radau collocation (degree 3) and Radau pseudospectral segments (4 nodes).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--rounds", type=int, default=15)
  args = parser.parse_args()
  import scaly as sc
  from scaly import integrators as si
  from scaly import mpc
  from tests.mpc.test_ocp import N, NX, X0, hand_written, library

  options = {"tol": 1e-8}
  builds: dict[str, float] = {}
  t0 = time.perf_counter()
  hand = sc.solver(hand_written(), "ipopt", name="bench_hand", options=options)
  guess = (np.tile(X0, N + 1), np.zeros(N), np.zeros(N))
  hand_args = (guess, tuple(np.zeros(g.size) for g in guess), np.zeros((N + 1) * NX), np.zeros(2 * N), X0)
  hand(*hand_args)
  builds["hand"] = time.perf_counter() - t0
  runners = {"hand": (lambda: hand(*hand_args), hand)}
  for label, transcription in (("library", None), ("library_collocation3", si.Collocation(3)), ("library_pseudospectral4", si.Pseudospectral(4))):
    t0 = time.perf_counter()
    controller = mpc.MPC(library(transcription, name=f"bench_{label}"), "ipopt", options=options)
    start = controller.initial_guess(X0)
    controller.solve(X0, guess=start)
    builds[label] = time.perf_counter() - t0
    runners[label] = (lambda c=controller, g=start: c.solve(X0, guess=g), controller.solver)
  best: dict[str, list[float]] = {k: [np.inf, np.inf] for k in runners}
  iterations: dict[str, int] = {}
  for _ in range(args.rounds):
    for label, (run, fn) in runners.items():
      t0 = time.perf_counter()
      run()
      wall = time.perf_counter() - t0
      stats = fn.solver_stats()
      iterations[label] = stats.iter
      best[label] = [min(best[label][0], stats.t_total), min(best[label][1], wall)]
  print("| variant | IPOPT iterations | IPOPT t_total ms | call wall ms | vs hand | build + compile s |")
  print("| --- | --- | --- | --- | --- | --- |")
  for label, (t_total, wall) in best.items():
    print(f"| {label} | {iterations[label]} | {t_total * 1e3:.3f} | {wall * 1e3:.3f} | {t_total / best['hand'][0]:.3f} | {builds[label]:.2f} |")


if __name__ == "__main__":
  main()
