"""The oscillating-masses MPC of qoco-benchmarks (the QOCO paper's QP family), for Scaly's solvers.

qoco-benchmarks (`problems/oscillating_masses.py`, pinned by `baseline/setup.sh`) builds, for a horizon
`T`: 4 masses between springs (8 states, a force on each mass, `dt = 0.25`, exact discretisation),
`Q = diag(U(0, 10)^8)`, `R = diag(U(0, 10)^4)`, a start `x0 = clip(N(0, 1)^8, -1.8, 1.8)`, and

    minimize   sum_{k<T} x_k' Q x_k + u_k' R u_k + x_T' Q x_T
    subject to x_0 = x0,  x_{k+1} = A x_k + B u_k,  |x_k| <= 2 (k < T),  |u_k| <= 5.

`instances` replays its random draws in its order (seed 123, then per horizon and instance: `Q`, `R`,
`x0`), so instance `(T, i)` here is the one the benchmark solves. Scaly's problem takes `Q`, `R` and
`x0` as parameters, which is what the generated QOCO solver updates between solves too.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

import scaly as sc

N_MASSES, DT, U_MAX, X_MAX = 4, 0.25, 5.0, 2.0
NX, NU = 2 * N_MASSES, N_MASSES
HORIZONS = [8, 20, 32, 44, 56, 76, 96, 116, 136, 156]  # run_problems/run_oscillating_masses.py


def dynamics() -> tuple[np.ndarray, np.ndarray]:
  band = -2 * np.eye(N_MASSES) + np.eye(N_MASSES, k=1) + np.eye(N_MASSES, k=-1)
  ac = np.block([[np.zeros((N_MASSES, N_MASSES)), np.eye(N_MASSES)], [band, np.zeros((N_MASSES, N_MASSES))]])
  bc = np.vstack([np.zeros((N_MASSES, N_MASSES)), np.eye(N_MASSES)])
  a = sla.expm(ac * DT)
  return a, np.linalg.inv(ac) @ (a - np.eye(NX)) @ bc


def instances(horizons: list[int] = HORIZONS, per_horizon: int = 1) -> dict[tuple[int, int], dict[str, np.ndarray]]:
  """`(T, i) -> {q, r, x0}` in the benchmark's draw order."""
  rng = np.random.RandomState(123)
  xlim = X_MAX * np.ones(NX)
  out = {}
  for t in horizons:
    for i in range(per_horizon):
      q = rng.uniform(0, 10, NX)
      r = rng.uniform(0, 10, NU)
      x0 = np.clip(rng.randn(NX), -0.9 * xlim, 0.9 * xlim)
      out[(t, i)] = {"q": q, "r": r, "x0": x0}
  return out


def problem(horizon: int) -> sc.opt.NLP:
  a, b = dynamics()

  @sc.opt.problem(
    vars=sc.G(sc.L("u", (horizon, NU)), sc.L("x", (horizon + 1, NX))),
    params=sc.G(sc.L("q", NX), sc.L("r", NU), sc.L("x0", NX)),
    name=f"masses_T{horizon}",
  )
  def masses(variables, params):
    u, x = variables
    q, r, x0 = params
    cost = (x * x * q.reshape((1, NX))).sum() + (u * u * r.reshape((1, NU))).sum()
    dyn = (x[1:] - x[:-1] @ sc.const(a.T) - u @ sc.const(b.T)).vec()
    x_bound = np.full((horizon + 1, NX), X_MAX)
    x_bound[horizon] = np.inf  # the terminal state is not bounded
    return sc.opt.ProblemSpec(
      minimize=cost,
      eq=(x[0] - x0, dyn),
      lb=(sc.const(np.full((horizon, NU), -U_MAX)), sc.const(-x_bound)),
      ub=(sc.const(np.full((horizon, NU), U_MAX)), sc.const(x_bound)),
    )

  return masses


def objective(horizon: int, data: dict[str, np.ndarray], u: np.ndarray, x: np.ndarray) -> float:
  return float((x * x * data["q"]).sum() + (u * u * data["r"]).sum())
