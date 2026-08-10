"""Model-in-the-loop race-car NMPC over Alloy's generated IPOPT path or CasADi's.

The canonical episode drives one lap of ``fsds_competition_1`` with a 40-step
horizon and the 50 ms sample time from :class:`RaceCarParams`, following the
center-line reference produced by :class:`MotionPlanner`.
``run_episode(smoke=True)`` uses a 3-step horizon and 2 plant steps so
plugin/JIT checks finish quickly.

The formulation is the one from ``minimal_tracking_nmpc/nmpc.py``: position error
split into longitudinal and lateral components in the reference frame so the two
directions can be weighted independently, a steady-state throttle feedforward in
the control cost, and a corridor constraint keeping the car's front corners
within the track.

``backend="casadi"`` swaps in :mod:`.casadi_nlp`, which builds the identical
problem through CasADi and solves it with the same IPOPT and the same options.
Everything else — plant, planner, warm starts, lap logic — is shared, so the two
columns differ only in who supplies the oracles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import alloy as al
from alloy.solvers import SolverStats
from benchmarks.problems.race_cars import (
  CAR_LENGTH,
  CAR_WIDTH,
  DELTA_MAX,
  NU,
  NX,
  NZ,
  T_MAX,
  RaceCarParams,
  n_param,
  race_car_eq_function,
)
from benchmarks.problems.race_cars.reference import MotionPlanner
from benchmarks.problems.race_cars.tracks import Track, load_track


@dataclass(frozen=True)
class EpisodeConfig:
  track: str = "fsds_competition_1"
  horizon: int = 40
  max_steps: int = 1500
  params: RaceCarParams = RaceCarParams()
  v_ref: float = 5.0
  max_speed: float = 5.0
  initial_lateral_offset: float = 0.5
  # weights from minimal_tracking_nmpc/nmpc.py
  q_lon: float = 10.0
  q_lat: float = 20.0
  q_phi: float = 50.0
  q_v: float = 20.0
  r_throttle: float = 1e-3
  r_steering: float = 2.0
  q_lon_f: float = 1000.0
  q_lat_f: float = 1000.0
  q_phi_f: float = 500.0
  q_v_f: float = 1000.0
  # the FSDS tracks report 1.67-1.76 m per side; 1.5 m leaves margin for the 1.5 m-wide body
  track_half_width: float = 1.5
  ipopt_tol: float = 1e-6
  ipopt_max_iter: int = 80

  @classmethod
  def smoke(cls) -> EpisodeConfig:
    return cls(horizon=3, max_steps=2, ipopt_tol=1e-5, ipopt_max_iter=40)


@dataclass(frozen=True)
class StepTelemetry:
  stats: SolverStats
  arc_length: float
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
  arc_length: np.ndarray
  laps: np.ndarray
  telemetry: tuple[StepTelemetry, ...]
  predictions: np.ndarray
  reference_horizons: np.ndarray
  oracle_z: np.ndarray
  oracle_p: np.ndarray
  oracle_inputs: tuple[dict[str, np.ndarray], ...]
  track: Track
  center_path: np.ndarray
  lap_length: float


def steady_throttle(v: float, params: RaceCarParams = RaceCarParams()) -> float:
  """Throttle holding ``v`` against rolling and aerodynamic resistance."""
  return float(np.tanh(10.0 * v) * (params.c_r0 + params.c_r1 * v + params.c_r2 * v * v) / params.c_m0)


def continuous_dynamics_np(x: np.ndarray, u: np.ndarray, params: RaceCarParams = RaceCarParams()) -> np.ndarray:
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


def rk4_step_np(x: np.ndarray, u: np.ndarray, params: RaceCarParams = RaceCarParams()) -> np.ndarray:
  """Advance the NumPy plant by the physical parameter's fixed sample time."""
  x, u, h = np.asarray(x, dtype=np.float64), np.asarray(u, dtype=np.float64), params.dt
  k1 = continuous_dynamics_np(x, u, params)
  k2 = continuous_dynamics_np(x + 0.5 * h * k1, u, params)
  k3 = continuous_dynamics_np(x + 0.5 * h * k2, u, params)
  k4 = continuous_dynamics_np(x + h * k3, u, params)
  return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _race_car_nlp(config: EpisodeConfig) -> al.SolverFunction:
  n = config.horizon
  z = al.sym("z", NZ * (n + 1))
  p = al.sym("p", n_param(n), diff=False)
  params = p[NX * (n + 1) :]
  c_m0, c_r0, c_r1, c_r2 = params[3], params[4], params[5], params[6]
  half_length, half_width = 0.5 * CAR_LENGTH, 0.5 * CAR_WIDTH

  # every cost term is a weighted square, so the cost is one flat dot(weights, residuals**2).
  # accumulating it as `cost = cost + ...` instead builds a 250-deep expression chain and the
  # recursive fusion pass overflows Python's stack at this horizon.
  weights: list[float] = []
  residuals = []
  corridor = []
  for i in range(n + 1):
    zi, ref = z[i * NZ : (i + 1) * NZ], p[i * NX : (i + 1) * NX]
    v_ref = ref[3]
    # parameter-only feedforward, so it contributes no derivative work
    throttle_ref = (10.0 * v_ref).tanh() * (c_r0 + c_r1 * v_ref + c_r2 * v_ref * v_ref) / c_m0
    weights.extend([config.r_throttle, config.r_steering])
    residuals.extend([zi[NX] - throttle_ref, zi[NX + 1]])
    if i == 0:
      # x_0 is pinned to the measurement by the initial-value equality, so its state cost is constant
      continue
    cos_ref, sin_ref = ref[2].cos(), ref[2].sin()
    dx, dy = zi[0] - ref[0], zi[1] - ref[1]
    e_lon = cos_ref * dx + sin_ref * dy
    e_lat = -sin_ref * dx + cos_ref * dy
    d_phi, d_v = zi[2] - ref[2], zi[3] - v_ref
    if i == n:
      weights.extend([config.q_lon_f, config.q_lat_f, config.q_phi_f, config.q_v_f])
    else:
      weights.extend([config.q_lon, config.q_lat, config.q_phi, config.q_v])
    residuals.extend([e_lon, e_lat, d_phi, d_v])
    # lateral reach of the two front corners; the rear pair only differs by the sign of the sin term
    reach = e_lat + half_length * d_phi.sin()
    corridor.extend([reach + half_width * d_phi.cos(), reach - half_width * d_phi.cos()])
  cost = al.dot(al.const(np.array(weights)), al.stack(residuals) ** 2)

  eq = race_car_eq_function(n).call([z, p])[0]
  lb, ub = np.full(z.size, -np.inf), np.full(z.size, np.inf)
  for i in range(n + 1):
    lb[i * NZ + 3], ub[i * NZ + 3] = 0.0, config.max_speed
    lb[i * NZ + NX : (i + 1) * NZ] = [-T_MAX, -DELTA_MAX]
    ub[i * NZ + NX : (i + 1) * NZ] = [T_MAX, DELTA_MAX]
  return al.nlp(
    x=z,
    p=p,
    f=cost,
    h_eq=eq,
    g_ineq=al.stack(corridor),
    l_ineq=np.full(len(corridor), -config.track_half_width),
    u_ineq=np.full(len(corridor), config.track_half_width),
    x_lb=lb,
    x_ub=ub,
    solver="ipopt",
    name=f"race_car_closed_loop_N{n}",
    options={
      "print_level": 0,
      "sb": "yes",
      "tol": config.ipopt_tol,
      "max_iter": config.ipopt_max_iter,
      "warm_start_init_point": "yes",
    },
  )


def _reference_guess(reference: np.ndarray, config: EpisodeConfig) -> np.ndarray:
  """Guess the reference states with their steady-state throttle and no steering."""
  stages = np.empty((config.horizon + 1, NZ), dtype=np.float64)
  stages[:, :NX] = reference
  stages[:, NX] = [steady_throttle(float(v), config.params) for v in reference[:, 3]]
  stages[:, NX + 1] = 0.0
  return stages.reshape(-1)


def _shift_blocks(values: np.ndarray, width: int) -> np.ndarray:
  blocks = np.asarray(values, dtype=np.float64).reshape(-1, width)
  return np.concatenate([blocks[1:], blocks[-1:]]).reshape(-1)


def build_solver(config: EpisodeConfig, backend: str = "alloy"):
  """The controller for `backend`; both present the same call signature and statistics."""
  if backend == "alloy":
    return _race_car_nlp(config)
  if backend == "casadi":
    from benchmarks.problems.race_cars.casadi_nlp import CasadiRaceCarSolver

    return CasadiRaceCarSolver(config)
  raise ValueError(f"unknown backend {backend!r}; expected 'alloy' or 'casadi'")


def run_episode(config: EpisodeConfig | None = None, *, smoke: bool = False, backend: str = "alloy") -> EpisodeResult:
  """Drive one lap of the configured track and return harvestable data."""
  config = EpisodeConfig.smoke() if config is None and smoke else (config or EpisodeConfig())
  if config.horizon < 1 or config.max_steps < 1:
    raise ValueError("horizon and max_steps must be positive")
  track = load_track(config.track)
  planner = MotionPlanner(track.center_line, horizon=config.horizon, dt=config.params.dt, v_ref=config.v_ref)
  solver = build_solver(config, backend)

  start = planner.center_path[0]
  start_heading = float(planner.phi_ref[0])
  offset = config.initial_lateral_offset
  state = np.array([start[0] - offset * np.sin(start_heading), start[1] + offset * np.cos(start_heading), start_heading, 0.0])
  s, reference = planner.plan(float(state[0]), float(state[1]), float(state[2]), 0.0)
  s_start = s

  states = [state]
  arc_lengths = [s]
  laps = [0]
  reference_poses = []
  reference_horizons: list[np.ndarray] = []
  predictions: list[np.ndarray] = []
  controls: list[np.ndarray] = []
  telemetry: list[StepTelemetry] = []
  oracle_inputs: list[dict[str, np.ndarray]] = []

  z0 = _reference_guess(reference, config)
  lam_eq0 = np.zeros(NX * (config.horizon + 1))
  lam_ineq0 = np.zeros(2 * config.horizon)
  lam_box0 = np.zeros(z0.size)
  control_lb = np.array([-T_MAX, -DELTA_MAX])
  control_ub = np.array([T_MAX, DELTA_MAX])

  for step in range(config.max_steps):
    stage_reference = reference.copy()
    stage_reference[0] = state  # p[:NX] doubles as the initial-value constraint
    p = np.concatenate([stage_reference.reshape(-1), config.params.array()])
    oracle_inputs.append({"z": z0.copy(), "p": p.copy()})
    out = solver(z0, lam_eq0, lam_ineq0, lam_box0, p)
    stats = solver.last_stats
    if stats is None or solver.last_status is None or not solver.last_status.ok or not np.all(np.isfinite(out["x"])):
      status = "missing" if stats is None else stats.status.name
      raise RuntimeError(f"race-car NMPC solve failed at step {step} on the {backend} backend: {status}")
    solution = np.asarray(out["x"], dtype=np.float64).reshape(config.horizon + 1, NZ)
    control = np.clip(solution[0, NX:], control_lb, control_ub)

    reference_poses.append(reference[0])
    reference_horizons.append(reference)
    predictions.append(solution[:, :NX].copy())
    controls.append(control)

    state = rk4_step_np(state, control, config.params)
    s, reference = planner.plan(float(state[0]), float(state[1]), float(state[2]), s)
    lap = max(0, int((s - s_start) // planner.lap_length))
    states.append(state)
    arc_lengths.append(s)
    laps.append(lap)
    telemetry.append(StepTelemetry(stats, s - s_start, lap))

    shifted = np.concatenate([solution[1:], solution[-1:]]).copy()
    shifted[0, :NX] = state
    shifted[-1, :NX] = rk4_step_np(shifted[-2, :NX], shifted[-2, NX:], config.params)
    z0 = shifted.reshape(-1)
    lam_eq0 = _shift_blocks(out["lam_eq"], NX)
    lam_ineq0 = _shift_blocks(out["lam_ineq"], 2)
    lam_box0 = _shift_blocks(out["lam_box"], NZ)
    if not all(np.all(np.isfinite(value)) for value in (z0, lam_eq0, lam_ineq0, lam_box0)):
      z0 = _reference_guess(reference, config)
      lam_eq0, lam_ineq0, lam_box0 = np.zeros_like(lam_eq0), np.zeros_like(lam_ineq0), np.zeros_like(lam_box0)
    if s - s_start >= planner.lap_length:
      break

  reference_poses.append(reference[0])
  representative = oracle_inputs[len(oracle_inputs) // 2]
  return EpisodeResult(
    states=np.asarray(states),
    references=np.asarray(reference_poses),
    controls=np.asarray(controls),
    arc_length=np.asarray(arc_lengths) - s_start,
    laps=np.asarray(laps, dtype=np.int64),
    telemetry=tuple(telemetry),
    predictions=np.asarray(predictions),
    reference_horizons=np.asarray(reference_horizons),
    oracle_z=representative["z"],
    oracle_p=representative["p"],
    oracle_inputs=tuple(oracle_inputs),
    track=track,
    center_path=planner.center_path,
    lap_length=planner.lap_length,
  )
