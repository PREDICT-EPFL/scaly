"""Model-in-the-loop neural-process MPC: the learned model controls the analytic Furuta pendulum.

The canonical episode is the paper's own: system 3, a 12-step horizon at ``dt = 0.02``, 100 control
steps starting from hanging (``theta = pi``), which is long enough to swing the pendulum up and
hold it. The controller's model is the conditional neural process; the plant is the analytic
rigid-body pendulum the process approximates, integrated at a fixed substep, so an episode is
reproducible in a way the reference implementation's wall-clock 4 kHz plant process is not.

Warm starting follows the reference implementation: the previous solution's controls shift by one
stage with the last repeated, and rolling those through the learned model from the measurement
gives a state guess that satisfies the dynamics exactly. Multipliers are reset each step rather
than shifted -- with the primal trajectory warm the solves take about ten IPOPT iterations from
zero duals.

Their delay compensation is deliberately *not* reproduced, and this is the one departure that
changes behaviour rather than only reproducibility. Their controller pins the first horizon node to
the measurement pushed one step forward through the learned model, because their plant keeps moving
in another process while the solve runs. This loop is synchronous: the control the solve returns
acts over the very next interval, so pinning the first node one step ahead does not remove a delay,
it adds one. On this pendulum one 20 ms interval at the torque limit is worth about 10 rad/s, and
the difference is not subtle -- with the compensation kept the episode never swings up at all and
the pendulum spins through six revolutions; without it, it settles upright at step 10 against their
step 14. See the problem README.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bench.harness.timing import SolveTiming

from scaly.opt import SolverStats
from bench.harness import problem_stats, solve_problem
from bench.problems.npmpc import (
  DT,
  HORIZON,
  NU,
  NX,
  PLANT_SUBSTEPS,
  TORQUE_LIMIT,
  CostWeights,
  Decoder,
  FurutaParams,
  constraint_counts,
  load_decoder_weights,
  n_dec,
  npmpc_nlp,
  pack_params,
  pack_nlp_params,
  plant_step,
  step_np,
  terminal_P,
)

# The pendulum's upright pose, reached by rotating the hanging start forward rather than back --
# their warm start interpolates towards `2 pi`, which is what picks the swing-up direction.
UPRIGHT = (2.0 * np.pi, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
  """Everything that fixes an episode. The defaults are the paper's closed-loop experiment."""

  horizon: int = HORIZON
  steps: int = 100
  dt: float = DT
  decoder: Decoder = Decoder()
  plant: FurutaParams = FurutaParams()
  substeps: int = PLANT_SUBSTEPS
  weights: CostWeights = CostWeights()
  x_start: tuple[float, float, float, float] = (float(np.pi), 0.0, 0.0, 0.0)
  ipopt_tol: float = 1e-6
  ipopt_max_iter: int = 50
  # Measured on the canonical episode: mean 4.67, worst 22 at step 3 where the pendulum is swinging
  # fastest, and 6-8 once it is holding. A budget just under twice the worst case reports a
  # divergence instead of burning the full count on it.
  sqp_max_iter: int = 40
  sqp_tol: float = 1e-6

  def __post_init__(self) -> None:
    if self.horizon < 1 or self.steps < 1 or self.substeps < 1:
      raise ValueError(f"horizon, steps and substeps must be positive, got {self.horizon}, {self.steps}, {self.substeps}")
    if self.dt <= 0.0:
      raise ValueError(f"dt must be positive, got {self.dt}")

  @classmethod
  def smoke(cls) -> EpisodeConfig:
    """A short toolchain check: a four-step horizon and two plant steps."""
    return cls(horizon=4, steps=2)


@dataclass(frozen=True, slots=True)
class EpisodeResult:
  config: EpisodeConfig
  #: measured plant states, ``(steps + 1, NX)``
  states: np.ndarray
  #: applied torques, ``(steps, NU)``
  controls: np.ndarray
  #: the solved horizon behind every applied control, ``(steps, horizon + 1, NX)``
  predictions: np.ndarray
  #: the rest of the solved control sequence, ``(steps, horizon, NU)``
  plans: np.ndarray
  #: the arm-angle slack each solve settled on, ``(steps,)``
  slacks: np.ndarray
  telemetry: tuple[SolverStats, ...]
  oracle_inputs: tuple[dict[str, np.ndarray], ...]
  terminal_weight: np.ndarray = field(repr=False)
  timing: dict[str, object]


def build_solver(config: EpisodeConfig, solver: str = "ipopt", oracle: str = "scaly"):
  """Build the selected solver with either provider's generated oracles."""
  if oracle == "scaly":
    options = (
      {"tol": config.sqp_tol, "max_iter": config.sqp_max_iter}
      if solver == "sqp"
      else {"print_level": 0, "sb": "yes", "tol": config.ipopt_tol, "max_iter": config.ipopt_max_iter, "warm_start_init_point": "yes"}
    )
    return npmpc_nlp(config.horizon, config.decoder, solver=solver, options=options)
  if oracle == "casadi":
    from bench.problems.npmpc.casadi_nlp import build_casadi_npmpc

    return build_casadi_npmpc(config, solver=solver)
  raise ValueError(f"unsupported npmpc solver/oracle pair {solver!r}/{oracle!r}")


def initial_guess(x_start: np.ndarray, config: EpisodeConfig) -> np.ndarray:
  """The reference implementation's cold start: interpolate from the measurement to upright, no torque."""
  ramp = np.linspace(0.0, 1.0, config.horizon + 1)[:, None]
  states = np.asarray(x_start, dtype=np.float64) + ramp * (np.array(UPRIGHT) - np.asarray(x_start, dtype=np.float64))
  return np.concatenate([states.reshape(-1), np.zeros(config.horizon * NU + 1)])


def shifted_guess(solution: np.ndarray, measured: np.ndarray, config: EpisodeConfig, pw: np.ndarray) -> np.ndarray:
  """Shift the previous solution one stage and re-roll it from the measurement.

  The controls shift one stage with the last repeated, and the states come from rolling those
  controls through the learned model starting at the measurement, so the guess satisfies the
  dynamics exactly. The slack carries over.
  """
  offset = NX * (config.horizon + 1)
  controls = solution[offset : offset + NU * config.horizon].reshape(config.horizon, NU)
  shifted = np.concatenate([controls[1:], controls[-1:]])
  states = [np.asarray(measured, dtype=np.float64)]
  for stage in range(config.horizon):
    states.append(step_np(config.decoder, pw, states[-1], shifted[stage], config.dt))
  return np.concatenate([np.asarray(states).reshape(-1), shifted.reshape(-1), solution[-1:]])


def run_episode(
  config: EpisodeConfig | None = None,
  *,
  smoke: bool = False,
  solver: str = "ipopt",
  oracle: str = "scaly",
  weights: np.ndarray | None = None,
) -> EpisodeResult:
  """Run one deterministic episode and return everything the recorder and the gates consume."""
  config = config if config is not None else (EpisodeConfig.smoke() if smoke else EpisodeConfig())
  pw = pack_params(config.decoder, load_decoder_weights(config.decoder) if weights is None else weights)
  P = terminal_P(config.decoder, pw, config.weights, config.dt)
  timing = SolveTiming()
  controller = build_solver(config, solver, oracle)
  timing.prepared(controller)
  (n_eq, n_ineq), nz = constraint_counts(config.horizon), n_dec(config.horizon)

  state = np.array(config.x_start, dtype=np.float64)
  guess = initial_guess(state, config)
  states = np.empty((config.steps + 1, NX))
  controls = np.empty((config.steps, NU))
  predictions = np.empty((config.steps, config.horizon + 1, NX))
  plans = np.empty((config.steps, config.horizon, NU))
  slacks = np.empty(config.steps)
  states[0] = state
  telemetry: list[SolverStats] = []
  oracle_inputs: list[dict[str, np.ndarray]] = []

  for step in range(config.steps):
    p = pack_nlp_params(config.decoder, state, pw, P, weights=config.weights, dt=config.dt)
    timing.start_step()
    out = solve_problem(controller, guess, np.zeros(n_eq), np.zeros(n_ineq), np.zeros(nz), p)
    timing.end_step()
    stats = problem_stats(controller)
    status = None if stats is None else stats.to_solver_status()
    if stats is None or status is None:
      raise RuntimeError(f"{solver}/{oracle} did not return solve status and statistics")
    solution = np.asarray(out["x"], dtype=np.float64).reshape(-1)
    if not status.ok or not np.all(np.isfinite(solution)):
      native = "missing" if stats is None else str(stats.native_status)
      raise RuntimeError(f"npmpc solve failed at step {step} with {solver}/{oracle}: {stats.status.name} (native {native})")
    oracle_inputs.append({"z": solution.copy(), "p": p})
    telemetry.append(stats)
    offset = NX * (config.horizon + 1)
    predictions[step] = solution[:offset].reshape(config.horizon + 1, NX)
    plans[step] = solution[offset : offset + NU * config.horizon].reshape(config.horizon, NU)
    slacks[step] = solution[-1]

    control = np.clip(solution[offset : offset + NU], -TORQUE_LIMIT, TORQUE_LIMIT)
    state = plant_step(state, control, config.dt, config.plant, config.substeps)
    if not np.all(np.isfinite(state)):
      raise RuntimeError(f"the analytic plant returned a non-finite state at step {step}")
    controls[step], states[step + 1] = control, state
    guess = shifted_guess(solution, state, config, pw)

  return EpisodeResult(
    config=config,
    states=states,
    controls=controls,
    predictions=predictions,
    plans=plans,
    slacks=slacks,
    telemetry=tuple(telemetry),
    timing=timing.summary(),
    oracle_inputs=tuple(oracle_inputs),
    terminal_weight=P,
  )


def upright_error(states: np.ndarray) -> np.ndarray:
  """Pendulum angle measured from the nearest upright pose, which is what settling is judged on."""
  theta = np.asarray(states, dtype=np.float64)[..., 0]
  return np.abs((theta + np.pi) % (2.0 * np.pi) - np.pi)


def settling_step(states: np.ndarray, tolerance_deg: float = 10.0) -> int | None:
  """First step after which the pendulum stays within `tolerance_deg` of upright, or None.

  The reference implementation's own convergence metric, reproduced so the two runs are judged the
  same way: it reports step 14 (0.28 s) for the released episode at the default tolerance.
  """
  within = upright_error(states) < np.deg2rad(tolerance_deg)
  if not within.size or not within[-1]:
    return None
  violations = np.flatnonzero(~within)
  return int(violations[-1] + 1) if violations.size else 0


def horizon_eq_violation(result: EpisodeResult, step: int) -> float:
  """Largest dynamics residual of the solution accepted at `step`, for divergence reports."""
  from bench.problems.npmpc import npmpc_eq_function

  eq = npmpc_eq_function(result.config.horizon, result.config.decoder)
  item = result.oracle_inputs[step]
  return float(np.max(np.abs(np.asarray(eq((item["z"], item["p"]))))))
