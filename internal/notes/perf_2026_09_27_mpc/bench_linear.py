"""M2: linear MPC through PIQP, sparse against condensed, and a hand-written sparse QP.

    uv run internal/notes/perf_2026_09_27_mpc/bench_linear.py [--rounds 15]

A chain of masses and springs (nx = 4 or 12: 2 or 6 masses), ZOH-discretized at 0.1 s, one force
per mass; stage cost x'x + 0.1 u'u, the LQR terminal cost, |u| <= 1, |x| <= 3. Per (nx, N):
PIQP's iterations and its own time (`t_total`, fastest of interleaved rounds from one cold start),
the Python call, the problem's size, and the build. `hand` is the same QP written with `sc.problem`
and matrix products, no `scaly.mpc`, on PIQP's sparse backend as `sparse` is; `condensed` is dense. Also: the maximal invariant set's computation, offline.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def chain(masses: int) -> tuple[np.ndarray, np.ndarray]:
  from scaly import integrators as si

  n = 2 * masses
  a = np.zeros((n, n))
  a[:masses, masses:] = np.eye(masses)
  stiffness = 2 * np.eye(masses) - np.eye(masses, k=1) - np.eye(masses, k=-1)
  a[masses:, :masses] = -stiffness
  a[masses:, masses:] = -0.1 * np.eye(masses)
  b = np.vstack([np.zeros((masses, masses)), np.eye(masses)])
  return si.zoh(a, b, 0.1)


def hand_problem(a: np.ndarray, b: np.ndarray, p: np.ndarray, horizon: int, name: str):
  import scaly as sc

  nx, nu = b.shape

  @sc.problem(vars=sc.G(sc.L("xs", (horizon + 1) * nx), sc.L("us", horizon * nu)), params=sc.L("x0", nx), name=name)
  def qp(v, x0):
    xs, us = v
    x2, u2 = xs.reshape((horizon + 1, nx)), us.reshape((horizon, nu))
    dynamics = (x2[1:] - x2[:-1] @ sc.const(a.T) - u2 @ sc.const(b.T)).reshape((horizon * nx,))
    cost = sc.sumsqr(xs[: horizon * nx]) + 0.1 * sc.sumsqr(us) + xs[horizon * nx :] @ (sc.const(p) @ xs[horizon * nx :])
    return sc.ProblemSpec(
      minimize=cost,
      eq=(xs[:nx] - x0, dynamics),
      lb=(sc.const(np.concatenate([np.full(nx, -np.inf), np.full(horizon * nx, -3.0)])), sc.const(np.full(horizon * nu, -1.0))),
      ub=(sc.const(np.concatenate([np.full(nx, np.inf), np.full(horizon * nx, 3.0)])), sc.const(np.full(horizon * nu, 1.0))),
    )

  return qp


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--rounds", type=int, default=15)
  args = parser.parse_args()
  import scaly as sc
  from scaly import mpc

  options = {"eps_abs": 1e-8, "eps_rel": 1e-8}
  rows = []
  for masses in (2, 6):
    a, b = chain(masses)
    nx, nu = b.shape
    k, p = mpc.lqr(a, b, np.eye(nx), 0.1 * np.eye(nu))
    x0 = np.tile([1.5, -1.0], nx // 2)[:nx] * np.linspace(1, 0.5, nx)
    t0 = time.perf_counter()
    invariant = mpc.max_invariant_set(a + b @ k, mpc.Polytope.box(-3 * np.ones(nx), 3 * np.ones(nx)).intersect(mpc.Polytope.box(-np.ones(nu), np.ones(nu)).preimage(k)))
    print(f"nx = {nx}: maximal invariant set, {invariant.H.shape[0]} rows, in {time.perf_counter() - t0:.2f} s")
    for horizon in (10, 50):
      runners = {}
      t0 = time.perf_counter()
      hand = sc.solver(hand_problem(a, b, p, horizon, f"hand_{nx}_{horizon}"), "piqp", options={**options, "sparse": True})
      hand_args = ((np.zeros((horizon + 1) * nx), np.zeros(horizon * nu)), (np.zeros((horizon + 1) * nx), np.zeros(horizon * nu)), np.zeros((horizon + 1) * nx), np.zeros(0), x0)
      hand(*hand_args)
      runners["hand"] = (lambda h=hand, g=hand_args: h(*g), hand, time.perf_counter() - t0, (horizon + 1) * nx + horizon * nu)
      for label, condensed in (("sparse", False), ("condensed", True)):
        t0 = time.perf_counter()
        ocp = mpc.OCP(step=mpc.linear(a, b, name=f"chain{nx}"), horizon=horizon, stage_cost=mpc.Quadratic(np.eye(nx), 0.1 * np.eye(nu)), terminal_cost=mpc.Quadratic(p), u_bounds=(-1, 1), x_bounds=(-3, 3), condensed=condensed, name=f"{label}_{nx}_{horizon}")
        controller = mpc.MPC(ocp, "piqp", options=options)
        start = controller.initial_guess(x0)
        controller.solve(x0, guess=start)
        runners[label] = (lambda c=controller, g=start: c.solve(x0, guess=g), controller.solver, time.perf_counter() - t0, ocp.layout.n_vars)
      best = {key: np.inf for key in runners}
      walls = {key: np.inf for key in runners}
      iterations = {}
      for _ in range(args.rounds):
        for key, (run, fn, _, _) in runners.items():
          t0 = time.perf_counter()
          run()
          walls[key] = min(walls[key], time.perf_counter() - t0)
          stats = fn.solver_stats()
          best[key], iterations[key] = min(best[key], stats.t_total), stats.iter
      for key, (_, _, build, n_vars) in runners.items():
        rows.append((nx, horizon, key, n_vars, iterations[key], best[key] * 1e6, walls[key] * 1e6, best[key] / best["hand"], build))
  print("| nx | N | variant | variables | PIQP iterations | PIQP t_total us | call us | vs hand | build + compile s |")
  print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
  for nx, horizon, key, n_vars, iters, t, wall, ratio, build in rows:
    print(f"| {nx} | {horizon} | {key} | {n_vars} | {iters} | {t:.1f} | {wall:.1f} | {ratio:.2f} | {build:.2f} |")


if __name__ == "__main__":
  main()
