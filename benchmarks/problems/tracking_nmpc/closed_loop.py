"""Model-in-the-loop tracking NMPC using Alloy's generated IPOPT path.

The canonical episode uses a 30-step horizon, 260 plant steps, and the 50 ms
physical sample time from :class:`TrackingParams`.  ``run_episode(smoke=True)``
uses a 3-step horizon and 2 plant steps so plugin/JIT checks finish quickly.
The closed reference is a circle whose heading is kept unwrapped across laps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import alloy as al
from alloy.solvers import SolverStats
from benchmarks.problems.tracking_nmpc import N_PARAMS, NU, NX, NZ, TrackingParams, tracking_eq_function


@dataclass(frozen=True)
class ReferencePath:
  radius: float = 3.0
  center_x: float = 0.0
  center_y: float = 0.0
  speed: float = 1.5


@dataclass(frozen=True)
class EpisodeConfig:
  horizon: int = 30
  steps: int = 260
  params: TrackingParams = TrackingParams()
  path: ReferencePath = ReferencePath()
  q_position: float = 20.0
  q_heading: float = 4.0
  q_speed: float = 2.0
  r_throttle: float = 0.2
  r_steering: float = 0.3
  terminal_scale: float = 4.0
  max_speed: float = 4.0
  max_steering: float = 0.7
  ipopt_tol: float = 1e-6
  ipopt_max_iter: int = 80

  @classmethod
  def smoke(cls) -> EpisodeConfig:
    return cls(horizon=3, steps=2, ipopt_tol=1e-5, ipopt_max_iter=40)


@dataclass(frozen=True)
class StepTelemetry:
  stats: SolverStats
  progress: float
  laps: int

  @property
  def solver_time(self) -> float:
    return self.stats.t_solver

  @property
  def function_evaluation_time(self) -> float:
    return self.stats.t_fe

  @property
  def glue_time(self) -> float:
    return self.stats.t_glue


@dataclass(frozen=True)
class EpisodeResult:
  states: np.ndarray
  references: np.ndarray
  controls: np.ndarray
  progress: np.ndarray
  laps: np.ndarray
  telemetry: tuple[StepTelemetry, ...]
  oracle_z: np.ndarray
  oracle_p: np.ndarray
  oracle_inputs: tuple[dict[str, np.ndarray], ...]


def reference_state(phase: float | np.ndarray, path: ReferencePath = ReferencePath()) -> np.ndarray:
  """Return ``[x, y, unwrapped heading, speed]`` on the circular path."""
  phase = np.asarray(phase, dtype=np.float64)
  return np.stack(
    [
      path.center_x + path.radius * np.cos(phase),
      path.center_y + path.radius * np.sin(phase),
      phase + np.pi / 2.0,
      np.full_like(phase, path.speed),
    ],
    axis=-1,
  )


def reference_control(path: ReferencePath = ReferencePath(), params: TrackingParams = TrackingParams()) -> np.ndarray:
  """Steady-state input corresponding to ``reference_state``."""
  beta = np.arctan(0.5 * params.wheelbase / path.radius)
  steering = 2.0 * beta
  vx = path.speed * np.cos(beta)
  resistance = (params.c_r0 + params.c_r1 * vx + params.c_r2 * vx * vx) * np.tanh(10.0 * vx)
  return np.array([resistance / params.c_m0, steering], dtype=np.float64)


def continuous_dynamics_np(x: np.ndarray, u: np.ndarray, params: TrackingParams = TrackingParams()) -> np.ndarray:
  x, u = np.asarray(x, dtype=np.float64), np.asarray(u, dtype=np.float64)
  if x.shape != (NX,) or u.shape != (NU,):
    raise ValueError(f"expected x/u shapes {(NX,)} / {(NU,)}, got {x.shape} / {u.shape}")
  beta = 0.5 * u[1]
  vx = x[3] * np.cos(beta)
  resistance = (params.c_r0 + params.c_r1 * vx + params.c_r2 * vx * vx) * np.tanh(10.0 * vx)
  return np.array(
    [
      x[3] * np.cos(x[2] + beta),
      x[3] * np.sin(x[2] + beta),
      x[3] * np.sin(beta) / (0.5 * params.wheelbase),
      (params.c_m0 * u[0] - resistance) / params.mass,
    ],
    dtype=np.float64,
  )


def rk4_step_np(x: np.ndarray, u: np.ndarray, params: TrackingParams = TrackingParams()) -> np.ndarray:
  """Advance the NumPy plant by the physical parameter's fixed sample time."""
  x, u, h = np.asarray(x, dtype=np.float64), np.asarray(u, dtype=np.float64), params.dt
  k1 = continuous_dynamics_np(x, u, params)
  k2 = continuous_dynamics_np(x + 0.5 * h * k1, u, params)
  k3 = continuous_dynamics_np(x + 0.5 * h * k2, u, params)
  k4 = continuous_dynamics_np(x + h * k3, u, params)
  return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def project_progress(x: np.ndarray, previous: float, path: ReferencePath = ReferencePath()) -> float:
  """Project position onto the circle and unwrap it nearest ``previous``."""
  raw = np.arctan2(float(x[1]) - path.center_y, float(x[0]) - path.center_x)
  return float(raw + 2.0 * np.pi * np.round((previous - raw) / (2.0 * np.pi)))


def _tracking_nlp(config: EpisodeConfig) -> al.SolverFunction:
  n = config.horizon
  z = al.sym("z", NZ * (n + 1))
  p = al.sym("p", NX * (n + 1) + N_PARAMS, diff=False)
  cost = al.const(0.0)
  for i in range(n + 1):
    zi, ref = z[i * NZ : (i + 1) * NZ], p[i * NX : (i + 1) * NX]
    scale = config.terminal_scale if i == n else 1.0
    cost = cost + scale * (
      config.q_position * al.sumsqr(zi[:2] - ref[:2])
      + config.q_heading * al.sumsqr(zi[2:3] - ref[2:3])
      + config.q_speed * al.sumsqr(zi[3:4] - ref[3:4])
      + config.r_throttle * al.sumsqr(zi[NX : NX + 1])
      + config.r_steering * al.sumsqr(zi[NX + 1 : NZ])
    )
  eq = tracking_eq_function(n).call([z, p])[0]
  lb, ub = np.full(z.size, -np.inf), np.full(z.size, np.inf)
  for i in range(n + 1):
    lb[i * NZ + 3], ub[i * NZ + 3] = 0.0, config.max_speed
    lb[i * NZ + NX : (i + 1) * NZ] = [-1.0, -config.max_steering]
    ub[i * NZ + NX : (i + 1) * NZ] = [1.0, config.max_steering]
  return al.nlp(
    x=z,
    p=p,
    f=cost,
    h_eq=eq,
    x_lb=lb,
    x_ub=ub,
    solver="ipopt",
    name=f"tracking_closed_loop_N{n}",
    options={
      "print_level": 0,
      "sb": "yes",
      "tol": config.ipopt_tol,
      "max_iter": config.ipopt_max_iter,
      "warm_start_init_point": "yes",
    },
  )


def _initial_guess(x: np.ndarray, control: np.ndarray, config: EpisodeConfig) -> np.ndarray:
  stages = np.empty((config.horizon + 1, NZ), dtype=np.float64)
  state = x.copy()
  for i in range(config.horizon + 1):
    stages[i, :NX], stages[i, NX:] = state, control
    if i < config.horizon:
      state = rk4_step_np(state, control, config.params)
  return stages.reshape(-1)


def _shift_blocks(values: np.ndarray, width: int) -> np.ndarray:
  blocks = np.asarray(values, dtype=np.float64).reshape(-1, width)
  return np.concatenate([blocks[1:], blocks[-1:]]).reshape(-1)


def run_episode(config: EpisodeConfig | None = None, *, smoke: bool = False) -> EpisodeResult:
  """Run a deterministic receding-horizon episode and return harvestable data."""
  config = EpisodeConfig.smoke() if config is None and smoke else (config or EpisodeConfig())
  if config.horizon < 1 or config.steps < 1:
    raise ValueError("horizon and steps must be positive")
  solver = _tracking_nlp(config)
  feedforward = reference_control(config.path, config.params)
  state = reference_state(0.0, config.path).copy()
  state[1] += 0.12
  state[3] *= 0.9
  progress = project_progress(state, 0.0, config.path)
  start_progress = progress

  states = np.empty((config.steps + 1, NX), dtype=np.float64)
  references = np.empty_like(states)
  controls = np.empty((config.steps, NU), dtype=np.float64)
  progresses = np.empty(config.steps + 1, dtype=np.float64)
  laps = np.empty(config.steps + 1, dtype=np.int64)
  states[0], references[0], progresses[0], laps[0] = state, reference_state(progress, config.path), progress, 0

  z0 = _initial_guess(state, feedforward, config)
  lam_eq0 = np.zeros(NX * (config.horizon + 1))
  lam_box0 = np.zeros(z0.size)
  telemetry: list[StepTelemetry] = []
  oracle_inputs: list[dict[str, np.ndarray]] = []

  for step in range(config.steps):
    phases = progress + np.arange(config.horizon + 1) * config.params.dt * config.path.speed / config.path.radius
    stage_reference = reference_state(phases, config.path)
    stage_reference[0] = state
    p = np.concatenate([stage_reference.reshape(-1), config.params.array()])
    oracle_inputs.append({"z": z0.copy(), "p": p.copy()})
    out = solver(z0, lam_eq0, np.zeros(0), lam_box0, p)
    stats = solver.last_stats
    if stats is None or solver.last_status is None or not solver.last_status.ok or not np.all(np.isfinite(out["x"])):
      status = "missing" if stats is None else stats.status.name
      raise RuntimeError(f"tracking NMPC solve failed at step {step}: {status}")
    solution = np.asarray(out["x"], dtype=np.float64).reshape(config.horizon + 1, NZ)
    control = np.clip(solution[0, NX:], [-1.0, -config.max_steering], [1.0, config.max_steering])
    state = rk4_step_np(state, control, config.params)
    progress = project_progress(state, progress, config.path)
    lap = max(0, int(np.floor((progress - start_progress) / (2.0 * np.pi))))

    controls[step], states[step + 1] = control, state
    references[step + 1] = reference_state(progress, config.path)
    progresses[step + 1], laps[step + 1] = progress, lap
    telemetry.append(StepTelemetry(stats, progress, lap))

    shifted = np.concatenate([solution[1:], solution[-1:]]).copy()
    shifted[0, :NX] = state
    shifted[-1, :NX] = rk4_step_np(shifted[-2, :NX], shifted[-2, NX:], config.params)
    z0 = shifted.reshape(-1)
    lam_eq0 = _shift_blocks(out["lam_eq"], NX)
    lam_box0 = _shift_blocks(out["lam_box"], NZ)
    if not (np.all(np.isfinite(z0)) and np.all(np.isfinite(lam_eq0)) and np.all(np.isfinite(lam_box0))):
      z0, lam_eq0, lam_box0 = _initial_guess(state, feedforward, config), np.zeros_like(lam_eq0), np.zeros_like(lam_box0)

  representative = oracle_inputs[len(oracle_inputs) // 2]
  return EpisodeResult(
    states,
    references,
    controls,
    progresses,
    laps,
    tuple(telemetry),
    representative["z"],
    representative["p"],
    tuple(oracle_inputs),
  )
