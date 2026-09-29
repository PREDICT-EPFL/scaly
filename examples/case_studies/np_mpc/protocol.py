"""What every controller in the NP-MPC study runs against: the plant, the context experiment and the
lockstep closed loop.

NumPy only, so the upstream's pinned environment (PyTorch 2.5.1, CasADi 3.7.0; `baseline/upstream.py`)
and Scaly's (`run_scaly.py`) import the same code, and a difference between two controllers' runs is
the controllers', never the harness's.

The upstream runs its plant in a separate process that integrates on wall-clock time steps (its
`QubeSimulator`, nominally 4 kHz) while the controller sleeps out each 20 ms period; no two runs
agree. Here the same plant, the analytic Furuta model integrated by explicit midpoint at the same
nominal 4 kHz, advances in lockstep with the controller. `lockstep` reproduces the upstream's
controller loop (`MPCController.run_controller` over `RuntimeBase`) step for step, including its delay
compensation: at step k the controller measures x_k, predicts one period ahead with its own model
under the input already being applied, solves from that prediction, and its first input acts over
the period after next. That is what the real-time loop does when every solve fits in its period.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

DT = 0.02  # the controller's period and the model's step (model/furuta_mpc.json)
HORIZON = 12
NX, NU = 4, 1
PLANT_SUBSTEPS = 80  # QubeSimulator's 4 kHz over one 20 ms period
GRAVITY = 9.81
SIM_STEPS = 100  # the released episodes and the paper's metrics run 100 steps (the config says 50)
CONTEXT_STEPS = 100  # model/furuta_experiment.json: experiment_length and n_context
CONTEXT_TORQUE = 0.02  # its "random" experiment: torques uniform in [-0.02, 0.02] N m
X_HANGING = np.array([np.pi, 0.0, 0.0, 0.0])
GUESS_TARGET = np.array([2.0 * np.pi, 0.0, 0.0, 0.0])  # the cold start's straight line ends upright
THETA_TOL_DEG = 10.0  # settling tolerance of the upstream's compute_metrics

# The three pendulums of the paper's adaptation study, [l_p, m_p, l_r, m_r] (model/furuta_experiment.json).
SYSTEMS = {
  "sys1": np.array([0.12094932405160891, 0.0231214108959178, 0.08593512757392434, 0.022501803023034478]),
  "sys2": np.array([0.1290918838099598, 0.016075376621634278, 0.08442631816614132, 0.026250053557759324]),
  "sys3": np.array([0.1377502795417797, 0.00880773464519363, 0.08949330135206067, 0.027303962257072]),
}

# model/furuta_mpc.json
COST = {
  "x": np.array([2.5, 1.5, 0.005, 0.01]),
  "x_diff": np.array([0.0, 0.0, 0.05, 0.125]),
  "x_end": np.array([1.0, 10.0, 0.1, 0.1]),
  "u": np.array([1.0]),
}
PHI_LIMIT = 2.0  # hard_bound.x[1], softened
SLACK_LIMIT = 5.0  # slack_bound.x[1]
TORQUE_LIMIT = 0.05  # hard_bound.u
X0_BAND = 1e-3  # MPCController: x[0] within x0 +- 1e-3
SLACK_WEIGHT = 1000.0  # MPCBase.slack_weight


def furuta_ode(x: np.ndarray, u: float, p: np.ndarray) -> np.ndarray:
  """The upstream's `FurutaDynamics`: x = (theta, phi, theta_dot, phi_dot), theta = 0 upright, u the arm
  torque, p = (l_p, m_p, l_r, m_r)."""
  theta, _, theta_dot, phi_dot = x
  lp, mp, lr, mr = p
  j_r = mr * lr**2 / 3 + mp * lr**2
  j_p = mp * lp**2 / 3
  s, c, s2 = np.sin(theta), np.cos(theta), np.sin(2 * theta)
  b00, b01, b11 = j_p, -mp * lr * lp / 2 * c, j_r + j_p * s * s
  a0 = j_p * s2 / 2 * phi_dot**2 + mp * lp * GRAVITY / 2 * s
  a1 = -j_p * s2 * phi_dot * theta_dot - mp * lr * lp / 2 * s * theta_dot**2 + u
  det = b00 * b11 - b01 * b01
  return np.array([theta_dot, phi_dot, (b11 * a0 - b01 * a1) / det, (-b01 * a0 + b00 * a1) / det])


def midpoint_step(x: np.ndarray, u: float, dt: float, p: np.ndarray) -> np.ndarray:
  """One explicit midpoint step (the upstream's `MidpointIntegration.step`)."""
  k1 = furuta_ode(x, u, p)
  return x + dt * furuta_ode(x + dt / 2 * k1, u, p)


def plant_step(x: np.ndarray, u: float, p: np.ndarray, dt: float = DT, substeps: int = PLANT_SUBSTEPS) -> np.ndarray:
  """The plant over one control period with the input held: `substeps` midpoint steps."""
  for _ in range(substeps):
    x = midpoint_step(x, u, dt / substeps, p)
  return x


def context_inputs(seed: int, n: int = CONTEXT_STEPS, magnitude: float = CONTEXT_TORQUE) -> np.ndarray:
  """The context experiment's torques, uniform in [-magnitude, magnitude] (the upstream's `random`
  experiment draws them from torch's generator; this study draws them from NumPy's so both
  environments see the same numbers)."""
  return np.random.default_rng(seed).uniform(-magnitude, magnitude, n)


def context_run(p: np.ndarray, inputs: np.ndarray, x_start: np.ndarray = X_HANGING) -> np.ndarray:
  """The recorded states of the context experiment, `(len(inputs) + 1, 4)`."""
  states = [np.asarray(x_start, dtype=np.float64)]
  for u in inputs:
    states.append(plant_step(states[-1], float(u), p))
  return np.array(states)


def context_pairs(states: np.ndarray, inputs: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
  """The first `n` steps as the encoder's context (the upstream's `FurutaTraining._build_context`):
  x = (sin theta, cos theta, theta_dot, phi_dot, u), y = the velocity change over the step."""
  x = np.column_stack([np.sin(states[:n, 0]), np.cos(states[:n, 0]), states[:n, 2:4], inputs[:n]])
  return x, states[1 : n + 1, 2:4] - states[:n, 2:4]


def cold_guess(x0: np.ndarray, horizon: int = HORIZON) -> tuple[np.ndarray, np.ndarray]:
  """`MPCController.warm_start`'s initial guess: states on a straight line from x0 to upright, zero torque."""
  s = np.linspace(0.0, 1.0, horizon + 1)[:, None]
  return x0 + s * (GUESS_TARGET - x0), np.zeros(horizon)


def realized_cost(states: np.ndarray, first_inputs: np.ndarray) -> float:
  """The upstream's closed-loop cost (`FurutaRuntime.compute_metrics`): stage plus inter-stage cost over
  consecutive measured states, with each step's planned first input; no terminal term."""
  total = 0.0
  for k in range(len(states) - 1):
    x, dx = states[k], states[k + 1] - states[k]
    lifted = np.array([2.0 * (1.0 - np.cos(x[0])), x[1] ** 2, x[2] ** 2, x[3] ** 2])
    total += COST["x"] @ lifted + COST["u"][0] * first_inputs[k] ** 2 + COST["x_diff"] @ (dx * dx)
  return float(total)


def settle_step(states: np.ndarray, tol_deg: float = THETA_TOL_DEG) -> int | None:
  """The first step after which theta stays within `tol_deg` of upright; None if it never settles."""
  err = np.abs((states[:, 0] + np.pi) % (2.0 * np.pi) - np.pi)
  within = err < np.deg2rad(tol_deg)
  if not within.any() or not within[-1]:
    return None
  outside = np.flatnonzero(~within)
  return int(outside[-1] + 1) if outside.size else 0


def shift(u: np.ndarray) -> np.ndarray:
  """The upstream's input warm start: drop the first input, repeat the last."""
  return np.r_[u[1:], u[-1:]]


Solve = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]
Rollout = Callable[[np.ndarray, np.ndarray], np.ndarray]


def lockstep(
  solve: Solve,
  rollout: Rollout,
  p_plant: np.ndarray,
  *,
  steps: int = SIM_STEPS,
  x_start: np.ndarray = X_HANGING,
  delay: bool = True,
  warm_retries: int = 100,
) -> dict[str, Any]:
  """One closed-loop episode, the upstream's controller loop in lockstep with the plant.

  `solve(x0, x_guess, u_guess, s_guess)` solves the OCP once and returns at least `x` `(N+1, 4)`,
  `u` `(N,)`, `slack` `(4,)` and `converged`; everything else it returns (iterations, timings) is
  kept per step. `rollout(x, u_seq)` is the controller's own model, `(len(u_seq) + 1, 4)`.

  With `delay` (the upstream's loop): the first solve starts cold from the measured state and is
  repeated from its own iterate until it converges (`warm_start`); at every step k >= 1 the controller
  predicts x_{k+1} with its model under the input being applied, warm-starts from the shifted plan
  rolled out from that prediction, and the plan's first input acts over [t_{k+1}, t_{k+2}). Without
  `delay` it solves from the measurement and its input acts immediately.
  """
  x = np.asarray(x_start, dtype=np.float64)
  x_guess, u_guess = cold_guess(x)
  s_guess = np.zeros(NX)  # Opti keeps a variable's last set_initial; only warm_start's retries set the slack's
  for _ in range(warm_retries):
    first = solve(x, x_guess, u_guess, s_guess)
    if first["converged"]:
      break
    x_guess, u_guess, s_guess = first["x"], first["u"], first["slack"]
  plan_x, plan_u = first["x"], first["u"]
  applied = 0.0  # the torque acting over [t_k, t_{k+1})
  record: dict[str, list] = {
    "states": [x],
    "applied": [],
    "x0": [],
    "x_guess": [],
    "u_guess": [],
    "plan_x": [],
    "plan_u": [],
    "slack": [],
    "steps": [],
  }
  for k in range(steps):
    if k == 0:  # run_controller's first step starts from warm_start's solution as is
      x0, xg, ug = (plan_x[0] if delay else x), plan_x, plan_u
    else:
      ug = shift(plan_u)
      x0 = rollout(x, np.array([applied]))[1] if delay else x
      xg = rollout(x0, ug)
    out = solve(x0, xg, ug, s_guess)
    plan_x, plan_u = out["x"], out["u"]
    for key, value in (("x0", x0), ("x_guess", xg), ("u_guess", ug), ("plan_x", plan_x), ("plan_u", plan_u), ("slack", out["slack"])):
      record[key].append(np.asarray(value, dtype=np.float64))
    record["steps"].append({key: value for key, value in out.items() if key not in ("x", "u", "slack")})
    act = applied if delay else float(plan_u[0])
    x = plant_step(x, act, p_plant)
    record["states"].append(x)
    record["applied"].append(act)
    applied = float(plan_u[0])
  states = np.array(record["states"])
  first_inputs = np.array([u[0] for u in record["plan_u"]])
  settle = settle_step(states[:steps])
  return {
    **{key: np.array(value) for key, value in record.items() if key != "steps"},
    "steps": record["steps"],
    "warm_start": {key: value for key, value in first.items() if key not in ("x", "u", "slack")},
    "cost": realized_cost(states[:steps], first_inputs),
    "settle_step": settle,
    "settle_time": None if settle is None else settle * DT,
    "max_abs_phi": float(np.abs(states[:, 1]).max()),
  }


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
  """`fn()` and its wall time in seconds."""
  t0 = time.perf_counter()
  out = fn()
  return out, time.perf_counter() - t0


def instances(run: dict[str, Any]) -> list[dict[str, list]]:
  """A lockstep run's OCP instances (the initial state and the primal guess each solve started from),
  for the replay drivers."""
  return [
    {"x0": x0.tolist(), "x_guess": xg.tolist(), "u_guess": [[float(v)] for v in ug]}
    for x0, xg, ug in zip(run["x0"], run["x_guess"], run["u_guess"], strict=True)
  ]


def compact(run: dict[str, Any]) -> dict[str, Any]:
  """An episode without the per-step guesses and plans, for everything but the reference episode
  (whose OCPs are the replays' instances)."""
  return {key: value for key, value in run.items() if key not in ("x_guess", "u_guess", "plan_x")}


def jsonable(value: Any) -> Any:
  """NumPy arrays and scalars to plain JSON types, recursively."""
  if isinstance(value, dict):
    return {str(k): jsonable(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [jsonable(v) for v in value]
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  return value
