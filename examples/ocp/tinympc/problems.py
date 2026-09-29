"""The three problem families of the TinyMPC microcontroller benchmarks, as closed-loop scenarios.

Ported from https://github.com/RoboticExplorationLab/mcu-solver-benchmarks (MIT):

- ``random_mpc``: the ICRA 2024 random QP-MPC (``ICRA_benchmarks/qp_mpc_problem/gen_mpc_problem.py``).
  A random stable, controllable ``(A, B)`` tracks a random feasible trajectory with ``|u| <= 3``;
  the sweeps vary the state dimension, the input dimension and the horizon.
- ``safety_filter``: the CDC 2024 QP safety filter (``safety_filter/safety_filter.ipynb`` and
  ``tinympc_f/tinympc_teensy/src/tiny_main.cpp``). A double integrator in ``nx / 2`` axes keeps
  ``|x| <= 1.5`` and ``|u| <= 2`` while staying as close as possible (``R = 100``, ``Q = 0``) to the
  inputs of a PD controller chasing a sinusoid; the sweeps vary the dimension and the horizon.
- ``rocket_landing``: the Conic-TinyMPC rocket soft landing (``rocket_landing/gen_rocket.py`` and
  ``tinympc/tinympc_teensy/src/rocket_landing_mpc.cpp``): 3-D point mass under gravity, thrust in a
  box and in the cone ``|(u_x, u_y)| <= 0.25 u_z``; the sweep varies the horizon.

Each function returns a ``Scenario``: the ``Problem``, the bounds, the initial state, a function
giving the references at step ``k`` from the current state, and the plant. The data, settings and
closed-loop simulations are those of the benchmark repository; where its code relies on C
``rand()`` (the noise), NumPy draws with a fixed seed replace it. The TinyMPC library of today
(``tiny_setup``) is what the settings are fed to, so the penalty ``rho`` is folded into the cache
exactly as ``problem.cache`` does.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from problem import Bounds, Cone, Problem, Settings


@dataclass(eq=False)
class Scenario:
  name: str
  problem: Problem
  bounds: Bounds
  x_init: np.ndarray
  steps: int
  references: Callable[[int, np.ndarray], tuple[np.ndarray, np.ndarray]]  # (k, x_k) -> (xref (N, nx), uref (N-1, nu))
  plant: Callable[[int, np.ndarray, np.ndarray], np.ndarray]  # (k, x_k, u_k) -> x_{k+1}


def _boxes(n: int, nx: int, nu: int, x_lo, x_hi, u_lo, u_hi) -> Bounds:
  return Bounds(
    np.tile(np.broadcast_to(np.asarray(x_lo, float), (nx,)), (n, 1)),
    np.tile(np.broadcast_to(np.asarray(x_hi, float), (nx,)), (n, 1)),
    np.tile(np.broadcast_to(np.asarray(u_lo, float), (nu,)), (n - 1, 1)),
    np.tile(np.broadcast_to(np.asarray(u_hi, float), (nu,)), (n - 1, 1)),
  )


# --- random QP-MPC --------------------------------------------------------------------------------

RANDOM_MPC_SWEEPS = {
  "nx": [(nx, 4, 10) for nx in (4, 8, 12, 16, 20, 24, 28, 32)],
  "nu": [(10, nu, 10) for nu in (4, 8, 12, 16, 20, 24, 28, 32)],
  "N": [(10, 4, n) for n in (4, 8, 12, 16, 30, 40, 50)],
}
"""The ``(nx, nu, N)`` of the published problem folders (``random_problems/prob_{nx,nu,Nh}_*``)."""


def random_mpc_data(nx: int, nu: int, n: int, nsim: int = 200) -> dict[str, np.ndarray]:
  """``generate_data`` of ``gen_mpc_problem.py``, draw for draw (legacy ``np.random.seed(123)``)."""
  rs = np.random.RandomState(123)
  q = np.diag(rs.uniform(0, 10, nx))
  r = np.diag(0.1 * np.ones(nu))
  u = rs.normal(size=(nu, nsim - 1))
  while True:
    a = rs.uniform(low=-1, high=1, size=(nx, nx))
    uu, s, vh = np.linalg.svd(a)
    e = np.zeros((uu.shape[0], vh.shape[0]))
    np.fill_diagonal(e, s / np.max(s))
    a = uu @ e @ vh.T  # sic: the published generator uses vh.T
    b = rs.uniform(low=-1, high=1, size=(nx, nu))
    ctrb = np.hstack([np.linalg.matrix_power(a, k) @ b for k in range(nx)])
    if np.linalg.matrix_rank(ctrb) == nx:
      break
  xbar = np.zeros((nsim, nx))
  for k in range(nsim - 1):
    xbar[k + 1] = a @ xbar[k] + b @ u[:, k]
  return {"A": a, "B": b, "Q": q, "R": r, "x_bar": xbar}


def random_mpc(nx: int = 10, nu: int = 4, n: int = 10, nsim: int = 200, seed: int = 0, data: str | Path | None = None) -> Scenario:
  """Tracking of ``x_bar`` with ``|u| <= 3`` (the state bounds, 1e4, never bind), ``rho = 0.1``,
  tolerances 1e-4 and at most 4000 iterations as in ``teensy_tinympc_benchmark``. The closed loop
  runs ``nsim - N`` steps from ``x = 0`` with a noise of 0.01 on the state.

  ``data`` names a published ``rand_prob_osqp_params.npz`` to use instead of the generator: only
  some of the published folders were drawn with the generator's seed (``prob_nx_4``, ``prob_nx_10``
  are; ``prob_nx_32`` is not), so the benchmark reads the published matrices."""
  if data is None:
    d = random_mpc_data(nx, nu, n, nsim)
  else:
    with np.load(data) as z:
      d = {k: np.array(z[k]) for k in ("A", "B", "Q", "R", "x_bar")}
    if d["B"].shape != (nx, nu) or d["x_bar"].shape != (nsim, nx):
      raise ValueError(f"{data} holds nx, nu = {d['B'].shape}, not {(nx, nu)}")
  p = Problem(d["A"], d["B"], np.diag(d["Q"]).copy(), np.diag(d["R"]).copy(), n, 0.1, settings=Settings(1e-4, 1e-4, 4000))
  noise = np.random.default_rng(seed).uniform(-0.01, 0.01, (nsim, nx))
  xbar = d["x_bar"]

  def references(k: int, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return xbar[k : k + n].copy(), np.zeros((n - 1, nu))

  def plant(k: int, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return d["A"] @ x + d["B"] @ u + noise[k]

  return Scenario(f"random_mpc_nx{nx}_nu{nu}_N{n}", p, _boxes(n, nx, nu, -1e4, 1e4, -3.0, 3.0), np.zeros(nx), nsim - n, references, plant)


# --- predictive safety filter ---------------------------------------------------------------------

SAFETY_FILTER_SWEEPS = {
  "nx": [(nx, 10) for nx in (2, 4, 8, 12, 16, 20, 24, 28, 32)],
  "N": [(10, n) for n in (4, 8, 10, 16, 22, 30, 32, 40, 50, 64, 100)],
}
"""The ``(nx, N)`` of the published sheets ("states (horizon=10)", "horizon (states = 10)")."""


def double_integrator(nx: int, h: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
  m = nx // 2
  a = np.block([[np.eye(m), h * np.eye(m)], [np.zeros((m, m)), np.eye(m)]])
  b = np.block([[0.5 * h * h * np.eye(m)], [h * np.eye(m)]])
  return a, b


def safety_filter(nx: int = 4, n: int = 20, ntotal: int = 201, seed: int = 1) -> Scenario:
  """``R = 100``, ``Q = 0``, ``rho = 100``, ``|x| <= 1.5``, ``|u| <= 2``, tolerances 1e-2, at most
  500 iterations. At each step a PD controller (``kp = 7``, ``kd = 3``) rolls the nominal model out
  over the horizon towards ``2 sin(0.05 (step + k))`` in every position; its inputs are the
  reference the filter stays close to, and the filter's first input is applied."""
  a, b = double_integrator(nx)
  nu, npos, dt = nx // 2, nx // 2, 0.05
  kp, kd = 7.0, 3.0
  p = Problem(a, b, np.zeros(nx), 1e2 * np.ones(nu), n, 1e2, settings=Settings(1e-2, 1e-2, 500))

  def references(step: int, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uref = np.zeros((n - 1, nu))
    xh = x.copy()
    for k in range(n - 1):
      target = 2.0 * np.sin(dt * (step + k))
      uref[k] = kp * (target - xh[:npos]) + kd * (0.0 - xh[npos:])
      xh = a @ xh + b @ uref[k]
    return np.zeros((n, nx)), uref

  def plant(k: int, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return a @ x + b @ u

  x_init = np.random.default_rng(seed).uniform(-1.0, 1.0, nx) * 0.1
  return Scenario(f"safety_filter_nx{nx}_N{n}", p, _boxes(n, nx, nu, -1.5, 1.5, -2.0, 2.0), x_init, ntotal - n - 1, references, plant)


# --- rocket soft landing (second-order cone) -------------------------------------------------------

ROCKET_LANDING_SWEEP = [2, 4, 8, 16, 32, 64, 128, 192, 256]
"""The horizons of the published sheets ("horizon 2" ... "horizon 256")."""


def rocket_dynamics(dt: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """``gen_rocket.py``: a unit-mass point in 3-D, zero-order hold at 20 Hz, gravity 9.81."""
  a = np.eye(6)
  a[0:3, 3:6] = dt * np.eye(3)
  b = np.vstack([0.5 * dt * dt * np.eye(3), dt * np.eye(3)])
  f = np.array([0.0, 0.0, -0.5 * 9.81 * dt * dt, 0.0, 0.0, -9.81 * dt])
  return a, b, f


def rocket_landing(n: int = 32, ntotal: int = 301, seed: int = 1) -> Scenario:
  """``Q = 100 I``, ``R = I``, ``rho = 1``; thrust in ``[-10, 105]`` and in the cone of half-opening
  ``atan 0.25`` about ``u_z``; no state constraints; tolerances 1e-2, at most 500 iterations. The
  reference runs linearly from ``xinit`` to the origin over ``ntotal`` steps; the rocket starts at
  ``1.1 xinit`` and every step adds a uniform noise of 0.01."""
  a, b, f = rocket_dynamics()
  settings = Settings(1e-2, 1e-2, 500, en_state_bound=False, en_input_bound=True)
  p = Problem(a, b, 100.0 * np.ones(6), 1.0 * np.ones(3), n, 1.0, f=f, settings=settings, input_cones=(Cone(0, 3, 0.25),))
  xinit = np.array([4.0, 2.0, 20.0, -3.0, 2.0, -4.5])
  xg = np.zeros(6)
  noise = np.random.default_rng(seed).uniform(-1.0, 1.0, (ntotal, 6)) * 0.01

  def references(k: int, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(n)[:, None] + k
    xref = np.where(i >= ntotal, xg, xinit + (xg - xinit) * i / ntotal)
    uref = np.tile([0.0, 0.0, 10.0], (n - 1, 1))
    return xref, uref

  def plant(k: int, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return a @ x + b @ u + f + noise[k]

  bounds = _boxes(n, 6, 3, -np.inf, np.inf, -10.0, 105.0)
  return Scenario(f"rocket_landing_N{n}", p, bounds, 1.1 * xinit, ntotal, references, plant)


# --- closed loop ----------------------------------------------------------------------------------


@dataclass(eq=False)
class Episode:
  """What a closed loop fed to the solver and got back, step by step: enough to replay the same
  sequence of warm-started solves in another implementation."""

  x0: np.ndarray  # (steps, nx)
  xref: np.ndarray  # (steps, N, nx)
  uref: np.ndarray  # (steps, N-1, nu)
  u0: np.ndarray  # (steps, nu)
  iterations: np.ndarray  # (steps,)
  solved: np.ndarray  # (steps,)


Solve = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, int, bool]]
"""``(x0, xref, uref) -> (u0, iterations, solved)`` for a solver that keeps its own warm-start state."""


def closed_loop(s: Scenario, solve: Solve, steps: int | None = None) -> Episode:
  steps = s.steps if steps is None else steps
  n, nx, nu = s.problem.N, s.problem.nx, s.problem.nu
  ep = Episode(
    np.zeros((steps, nx)), np.zeros((steps, n, nx)), np.zeros((steps, n - 1, nu)), np.zeros((steps, nu)), np.zeros(steps, int), np.zeros(steps, bool)
  )
  x = s.x_init.copy()
  for k in range(steps):
    xref, uref = s.references(k, x)
    u0, it, ok = solve(x, xref, uref)
    ep.x0[k], ep.xref[k], ep.uref[k], ep.u0[k], ep.iterations[k], ep.solved[k] = x, xref, uref, u0, it, ok
    x = s.plant(k, x, u0)
  return ep
