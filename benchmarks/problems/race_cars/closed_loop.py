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

The solver and oracle provider are selected independently. CasADi or Alloy can
supply the generated oracles while IPOPT or Alloy's SQP implementation drives
the solve. Everything else — plant, planner, warm starts, and lap logic — is
shared.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass
import io
from pathlib import Path

import numpy as np

from benchmarks.harness.timing import SolveTiming

import alloy as al
from alloy.ir.expr import substitute
from alloy.solvers import SolverStats
from benchmarks.harness import problem_stats, solve_problem
from benchmarks.problems.race_cars import (
  CAR_LENGTH,
  CAR_WIDTH,
  DELTA_MAX,
  N_PARAMS,
  NX,
  NZ,
  T_MAX,
  RaceCarParams,
  _race_car_eq_vmap_expr,
  n_param,
  rk4_step_np,
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
  # warm-started SQP steps converge in <= 5 iterations on the canonical lap (mean 2.74);
  # a small budget keeps a divergence from burning 80 iterations before it is reported
  sqp_max_iter: int = 8

  @classmethod
  def smoke(cls) -> EpisodeConfig:
    return cls(horizon=3, max_steps=2, ipopt_tol=1e-5, ipopt_max_iter=80)


@dataclass(frozen=True)
class StepTelemetry:
  stats: SolverStats
  arc_length: float
  laps: int
  # constraint violations of the accepted solution, for the SQP-versus-IPOPT divergence report
  eq_violation: float
  ineq_violation: float
  bound_violation: float

  @property
  def solver_time(self) -> float:
    return self.stats.t_solver

  @property
  def qp_time(self) -> float:
    return self.stats.t_qp

  @property
  def globalization_time(self) -> float:
    return self.stats.t_globalization

  @property
  def function_evaluation_time(self) -> float:
    return self.stats.t_fe

  @property
  def glue_time(self) -> float:
    return self.stats.t_glue


@dataclass(frozen=True)
class StepRecord:
  step: int
  state: np.ndarray
  reference: np.ndarray
  control: np.ndarray | None
  prediction: np.ndarray | None
  reference_horizon: np.ndarray
  oracle_input: dict[str, np.ndarray] | None
  stats: SolverStats | None
  telemetry: StepTelemetry | None


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

  timing: dict[str, object]


def steady_throttle(v: float, params: RaceCarParams = RaceCarParams()) -> float:
  """Throttle holding ``v`` against rolling and aerodynamic resistance."""
  return float(np.tanh(10.0 * v) * (params.c_r0 + params.c_r1 * v + params.c_r2 * v * v) / params.c_m0)


@al.function(al.G(al.L("z", NZ), al.L("ref", NX)), al.L("corridor", ...), name="race_car_corridor_stage")
def _corridor_stage(inputs):  # type: ignore[no-untyped-def]
  z, ref = inputs
  cos_ref, sin_ref = ref[2].cos(), ref[2].sin()
  dx, dy = z[0] - ref[0], z[1] - ref[1]
  e_lat = -sin_ref * dx + cos_ref * dy
  d_phi = z[2] - ref[2]
  reach = e_lat + 0.5 * CAR_LENGTH * d_phi.sin()
  half_width = 0.5 * CAR_WIDTH * d_phi.cos()
  return al.stack([reach + half_width, reach - half_width])


@al.function(al.G(al.L("z", NZ), al.L("ref", NX), al.L("params", N_PARAMS)), al.L("residuals", ...), name="race_car_cost_stage")
def _cost_stage(inputs):
  z, ref, params = inputs
  c_m0, c_r0, c_r1, c_r2 = params[3], params[4], params[5], params[6]
  v_ref = ref[3]
  throttle_ref = (10.0 * v_ref).tanh() * (c_r0 + c_r1 * v_ref + c_r2 * v_ref * v_ref) / c_m0
  cos_ref, sin_ref = ref[2].cos(), ref[2].sin()
  dx, dy = z[0] - ref[0], z[1] - ref[1]
  return al.stack([z[NX] - throttle_ref, z[NX + 1], cos_ref * dx + sin_ref * dy, -sin_ref * dx + cos_ref * dy, z[2] - ref[2], z[3] - v_ref])


def race_car_lag_hess_dense_reference(config: EpisodeConfig, z: np.ndarray, p: np.ndarray, lam_f: float, lam_g: np.ndarray) -> np.ndarray:
  """Dense NumPy Hessian of ``lam_f * f + dot(lam_g, [h_eq; g_ineq])`` for ``_race_car_nlp``.

  Every nonlinear term touches one stage's ``z`` block only: the stage cost, the RK4 step into the
  next stage, and the corridor rows. The couplings to the next state are linear, so the Hessian is
  block diagonal and each block comes from one exact hyper-dual evaluation per column.
  """
  from benchmarks.harness.hyperdual import lagrangian_hessian_np

  n = config.horizon
  z, p, lam_g = np.asarray(z, dtype=np.float64), np.asarray(p, dtype=np.float64), np.asarray(lam_g, dtype=np.float64)
  if z.shape != (NZ * (n + 1),) or p.shape != (n_param(n),) or lam_g.shape != (NX * (n + 1) + 2 * n,):
    raise ValueError(f"invalid z/p/lam_g shapes {z.shape} / {p.shape} / {lam_g.shape}")
  params = RaceCarParams(*p[-N_PARAMS:])
  n_eq = NX * (n + 1)
  dense = np.zeros((z.size, z.size), dtype=np.float64)
  for i in range(n + 1):
    ref = p[NX * i : NX * (i + 1)]
    throttle_ref = steady_throttle(float(ref[3]), params)
    cos_ref, sin_ref = float(np.cos(ref[2])), float(np.sin(ref[2]))
    stage_weights = [config.r_throttle, config.r_steering]
    if 0 < i < n:
      stage_weights += [config.q_lon, config.q_lat, config.q_phi, config.q_v]
    elif i == n:
      stage_weights += [config.q_lon_f, config.q_lat_f, config.q_phi_f, config.q_v_f]

    def stage(zi, i=i, ref=ref, throttle_ref=throttle_ref, cos_ref=cos_ref, sin_ref=sin_ref, stage_weights=stage_weights):
      residuals = [zi[NX] - throttle_ref, zi[NX + 1]]
      outputs = []
      if i > 0:
        dx, dy = zi[0] - ref[0], zi[1] - ref[1]
        e_lat = -sin_ref * dx + cos_ref * dy
        d_phi = zi[2] - ref[2]
        residuals += [cos_ref * dx + sin_ref * dy, e_lat, d_phi, zi[3] - ref[3]]
        reach = e_lat + 0.5 * CAR_LENGTH * np.sin(d_phi)
        half_width = 0.5 * CAR_WIDTH * np.cos(d_phi)
        outputs += [reach + half_width, reach - half_width]
      cost = sum(weight * residual * residual for weight, residual in zip(stage_weights, residuals, strict=True))
      step = list(rk4_step_np(zi[:NX], zi[NX:], params)) if i < n else []
      return [cost, *step, *outputs]

    multipliers = [lam_f]
    if i < n:
      multipliers += list(lam_g[NX * (i + 1) : NX * (i + 2)])
    if i > 0:
      multipliers += list(lam_g[n_eq + 2 * (i - 1) : n_eq + 2 * i])
    block = slice(NZ * i, NZ * (i + 1))
    dense[block, block] = lagrangian_hessian_np(stage, z[block], np.array(multipliers))
  return dense


def _race_car_nlp(config: EpisodeConfig, *, solver: str = "ipopt", sqp_options: dict[str, str | int | float] | None = None) -> al.Function:
  n = config.horizon
  z = al.sym("z", NZ * (n + 1))
  p = al.sym("p", n_param(n), diff=False)
  weights = np.tile([config.r_throttle, config.r_steering, config.q_lon, config.q_lat, config.q_phi, config.q_v], (n + 1, 1))
  weights[-1, 2:] = [config.q_lon_f, config.q_lat_f, config.q_phi_f, config.q_v_f]
  weights[0, 2:] = 0.0
  residuals = al.vmap(
    _cost_stage,
    length=n + 1,
    inputs={"z": (z, 0, NZ), "ref": (p, 0, NX), "params": (p, NX * (n + 1), 0)},
  )
  corridor = al.vmap(
    _corridor_stage,
    length=n,
    inputs={"z": (z, NZ, NZ), "ref": (p, NX, NX)},
  )
  cost = al.dot(al.const(weights.reshape(-1)), residuals**2)

  eq = _race_car_eq_vmap_expr(z, p, n)
  lb, ub = np.full(z.size, -np.inf), np.full(z.size, np.inf)
  for i in range(n + 1):
    lb[i * NZ + 3], ub[i * NZ + 3] = 0.0, config.max_speed
    lb[i * NZ + NX : (i + 1) * NZ] = [-T_MAX, -DELTA_MAX]
    ub[i * NZ + NX : (i + 1) * NZ] = [T_MAX, DELTA_MAX]
  problem_name = f"race_car_closed_loop_N{n}"

  @al.problem(vars=al.L("z", z.type), params=al.L("p", p.type), name=problem_name)
  def problem(new_z, new_p):
    replacements = {z: new_z, p: new_p}
    return al.ProblemSpec(
      minimize=substitute(cost, replacements),
      eq=(substitute(eq, replacements),),
      ineq=(
        al.bounded(
          substitute(corridor, replacements),
          lo=al.const(np.full(2 * n, -config.track_half_width)),
          hi=al.const(np.full(2 * n, config.track_half_width)),
          name="corridor",
        ),
      ),
      lb=al.const(lb),
      ub=al.const(ub),
    )

  return al.solver(
    problem,
    solver,
    name=f"{problem_name}_{solver}",
    options=(
      {"tol": config.ipopt_tol, "max_iter": config.sqp_max_iter, **(sqp_options or {})}
      if solver == "sqp"
      else {
        "print_level": 0,
        "sb": "yes",
        "tol": config.ipopt_tol,
        "max_iter": config.ipopt_max_iter,
        "warm_start_init_point": "yes",
      }
    ),
  )


def _reference_guess(reference: np.ndarray, config: EpisodeConfig) -> np.ndarray:
  """Guess the reference states with their steady-state throttle and no steering."""
  stages = np.empty((config.horizon + 1, NZ), dtype=np.float64)
  stages[:, :NX] = reference
  stages[:, NX] = [steady_throttle(float(v), config.params) for v in reference[:, 3]]
  stages[:, NX + 1] = 0.0
  return stages.reshape(-1)


def _feasible_guess(state: np.ndarray, reference: np.ndarray, config: EpisodeConfig) -> np.ndarray:
  stages = np.empty((config.horizon + 1, NZ), dtype=np.float64)
  predicted = state.copy()
  for stage in range(config.horizon + 1):
    heading_error = (reference[stage, 2] - predicted[2] + np.pi) % (2.0 * np.pi) - np.pi
    dx, dy = predicted[0] - reference[stage, 0], predicted[1] - reference[stage, 1]
    lateral_error = -np.sin(reference[stage, 2]) * dx + np.cos(reference[stage, 2]) * dy
    control = np.array([T_MAX, np.clip(heading_error - 0.5 * lateral_error, -DELTA_MAX, DELTA_MAX)])
    stages[stage, :NX], stages[stage, NX:] = predicted, control
    if stage < config.horizon:
      predicted = rk4_step_np(predicted, control, config.params)
  return stages.reshape(-1)


def _shift_blocks(values: np.ndarray, width: int) -> np.ndarray:
  blocks = np.asarray(values, dtype=np.float64).reshape(-1, width)
  return np.concatenate([blocks[1:], blocks[-1:]]).reshape(-1)


def _canonical_nominal(config: EpisodeConfig) -> np.ndarray | None:
  comparable = EpisodeConfig(max_steps=config.max_steps, ipopt_tol=config.ipopt_tol, ipopt_max_iter=config.ipopt_max_iter)
  if config != comparable:
    return None
  encoded = (Path(__file__).parent / "data" / "nominal_N40.npz.b64").read_text()
  with np.load(io.BytesIO(base64.b64decode(encoded))) as stored:
    nominal = np.asarray(stored["z"], dtype=np.float64)
  expected = (NZ * (config.horizon + 1),)
  if nominal.shape != expected or not np.all(np.isfinite(nominal)):
    raise ValueError(f"canonical race nominal has shape {nominal.shape}, expected finite {expected}")
  return nominal


def build_solver(
  config: EpisodeConfig,
  solver: str = "ipopt",
  oracle: str = "alloy",
  *,
  sqp_options: dict[str, str | int | float] | None = None,
):
  """Build the selected solver with either provider's generated oracles."""
  if (solver, oracle) == ("ipopt", "alloy"):
    return _race_car_nlp(config)
  if (solver, oracle) == ("ipopt", "casadi"):
    from benchmarks.problems.race_cars.casadi_nlp import CasadiRaceCarSolver

    return CasadiRaceCarSolver(config)
  if (solver, oracle) == ("sqp", "alloy"):
    return _race_car_nlp(config, solver="sqp", sqp_options=sqp_options)
  if (solver, oracle) == ("sqp", "casadi"):
    from benchmarks.problems.race_cars.casadi_nlp import build_casadi_race_car_sqp

    return build_casadi_race_car_sqp(config, sqp_options=sqp_options)
  raise ValueError(f"unsupported race-car solver/oracle pair {solver!r}/{oracle!r}")


def run_episode(
  config: EpisodeConfig | None = None,
  *,
  smoke: bool = False,
  solver: str = "ipopt",
  oracle: str = "alloy",
  sqp_options: dict[str, str | int | float] | None = None,
  record_step: Callable[[StepRecord], None] | None = None,
) -> EpisodeResult:
  """Drive one lap of the configured track and return harvestable data."""
  config = EpisodeConfig.smoke() if config is None and smoke else (config or EpisodeConfig())
  if config.horizon < 1 or config.max_steps < 1:
    raise ValueError("horizon and max_steps must be positive")
  track = load_track(config.track)
  planner = MotionPlanner(track.center_line, horizon=config.horizon, dt=config.params.dt, v_ref=config.v_ref)
  timing = SolveTiming()
  controller = build_solver(config, solver, oracle, sqp_options=sqp_options)
  timing.prepared(controller)

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

  initial_reference = reference.copy()
  initial_reference[0] = state
  nominal = _canonical_nominal(config) if solver == "sqp" else None
  z0 = (
    nominal.copy()
    if nominal is not None
    else (_feasible_guess(state, initial_reference, config) if solver == "sqp" else _reference_guess(reference, config))
  )
  lam_eq0 = np.zeros(NX * (config.horizon + 1))
  lam_ineq0 = np.zeros(2 * config.horizon)
  lam_box0 = np.zeros(z0.size)
  control_lb = np.array([-T_MAX, -DELTA_MAX])
  control_ub = np.array([T_MAX, DELTA_MAX])

  for step in range(config.max_steps):
    stage_reference = reference.copy()
    stage_reference[0] = state  # p[:NX] doubles as the initial-value constraint
    p = np.concatenate([stage_reference.reshape(-1), config.params.array()])
    timing.start_step()
    out = solve_problem(controller, z0, lam_eq0, lam_ineq0, lam_box0, p)
    timing.end_step()
    stats = problem_stats(controller)
    status = None if stats is None else stats.to_solver_status()
    if stats is None or status is None or not status.ok or not np.all(np.isfinite(out["x"])):
      status = "missing" if stats is None else stats.status.name
      residual = ""
      if "h_eq" in out and "g_ineq" in out:
        eq = float(np.max(np.abs(out["h_eq"]), initial=0.0))
        ineq = float(np.max(np.maximum(0.0, np.abs(out["g_ineq"]) - config.track_half_width), initial=0.0))
        residual = f", eq={eq:.3g}, ineq={ineq:.3g}"
      if record_step is not None:
        record_step(
          StepRecord(
            step,
            state.copy(),
            reference[0].copy(),
            None,
            None,
            reference.copy(),
            {
              "z": z0.copy(),
              "lam_eq": lam_eq0.copy(),
              "lam_ineq": lam_ineq0.copy(),
              "lam_box": lam_box0.copy(),
              "p": p.copy(),
            },
            stats,
            None,
          )
        )
      native = "missing" if stats is None else str(stats.native_status)
      raise RuntimeError(f"race-car NMPC solve failed at step {step} with {solver}/{oracle}: {status} (native {native}){residual}")
    oracle_inputs.append({"z": z0.copy(), "p": p.copy()})
    solution = np.asarray(out["x"], dtype=np.float64).reshape(config.horizon + 1, NZ)
    eq_violation = float(np.max(np.abs(out["h_eq"]), initial=0.0))
    ineq_violation = float(np.max(np.maximum(0.0, np.abs(out["g_ineq"]) - config.track_half_width), initial=0.0))
    bound_violation = max(
      float(np.max(np.maximum(0.0, -solution[:, 3]), initial=0.0)),
      float(np.max(np.maximum(0.0, solution[:, 3] - config.max_speed), initial=0.0)),
      float(np.max(np.maximum(0.0, np.abs(solution[:, NX]) - T_MAX), initial=0.0)),
      float(np.max(np.maximum(0.0, np.abs(solution[:, NX + 1]) - DELTA_MAX), initial=0.0)),
    )
    if max(eq_violation, ineq_violation, bound_violation) > max(config.ipopt_tol, 1e-5):
      failed = StepTelemetry(stats, s - s_start, laps[-1], eq_violation, ineq_violation, bound_violation)
      if record_step is not None:
        record_step(
          StepRecord(
            step,
            state.copy(),
            reference[0].copy(),
            None,
            solution[:, :NX].copy(),
            reference.copy(),
            {
              "z": z0.copy(),
              "lam_eq": lam_eq0.copy(),
              "lam_ineq": lam_ineq0.copy(),
              "lam_box": lam_box0.copy(),
              "p": p.copy(),
            },
            stats,
            failed,
          )
        )
      raise RuntimeError(
        f"race-car NMPC returned infeasible {solver}/{oracle} solution at step {step}: "
        f"eq={eq_violation:.3g}, ineq={ineq_violation:.3g}, bound={bound_violation:.3g}"
      )
    control = np.clip(solution[0, NX:], control_lb, control_ub)

    applied_state, applied_reference, applied_horizon = state.copy(), reference[0].copy(), reference.copy()
    reference_poses.append(applied_reference)
    reference_horizons.append(applied_horizon)
    predictions.append(solution[:, :NX].copy())
    controls.append(control)

    state = rk4_step_np(state, control, config.params)
    s, reference = planner.plan(float(state[0]), float(state[1]), float(state[2]), s)
    lap = max(0, int((s - s_start) // planner.lap_length))
    states.append(state)
    arc_lengths.append(s)
    laps.append(lap)
    step_telemetry = StepTelemetry(stats, s - s_start, lap, eq_violation, ineq_violation, bound_violation)
    telemetry.append(step_telemetry)
    if record_step is not None:
      record_step(
        StepRecord(
          step,
          applied_state,
          applied_reference,
          control.copy(),
          solution[:, :NX].copy(),
          applied_horizon,
          oracle_inputs[-1],
          stats,
          step_telemetry,
        )
      )

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
  if record_step is not None:
    record_step(StepRecord(len(controls), state.copy(), reference[0].copy(), None, None, reference.copy(), None, None, None))
  representative = oracle_inputs[len(oracle_inputs) // 2]
  return EpisodeResult(
    states=np.asarray(states),
    references=np.asarray(reference_poses),
    controls=np.asarray(controls),
    arc_length=np.asarray(arc_lengths) - s_start,
    laps=np.asarray(laps, dtype=np.int64),
    telemetry=tuple(telemetry),
    timing=timing.summary(),
    predictions=np.asarray(predictions),
    reference_horizons=np.asarray(reference_horizons),
    oracle_z=representative["z"],
    oracle_p=representative["p"],
    oracle_inputs=tuple(oracle_inputs),
    track=track,
    center_path=planner.center_path,
    lap_length=planner.lap_length,
  )
