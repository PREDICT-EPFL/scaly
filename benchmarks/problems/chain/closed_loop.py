"""Receding-horizon chain-of-masses benchmark runner.

The canonical B3 point is five masses, a 12-interval controller horizon, and
90 closed-loop steps (eighteen simulated seconds with the default ``dt=0.2``) —
long enough for the chain to settle onto ``END_REF``, with the end mass within
0.06 of it, moving under 3 mm per step, and ``|u|`` under 0.02 by the end.
``ClosedLoopConfig.smoke()`` reduces this to two intervals and two steps for a
short toolchain check.  The plant deliberately uses the independent NumPy RK4
implementation while the controller uses the selected generated solver path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmarks.harness.timing import SolveTiming

from alloy import SolverStats
from benchmarks.harness import problem_stats, solve_problem
from benchmarks.problems.chain import (
  NU,
  ChainParams,
  chain_nlp,
  initial_state,
  n_dec,
  n_state,
  rk4_step_np,
)


@dataclass(frozen=True, slots=True)
class ClosedLoopConfig:
  n_masses: int = 5
  horizon: int = 12
  steps: int = 90
  params: ChainParams = ChainParams()

  def __post_init__(self) -> None:
    if self.n_masses < 3:
      raise ValueError(f"n_masses must be at least 3, got {self.n_masses}")
    if self.horizon < 1:
      raise ValueError(f"horizon must be positive, got {self.horizon}")
    if self.steps < 1:
      raise ValueError(f"steps must be positive, got {self.steps}")
    if self.params.dt <= 0.0:
      raise ValueError(f"dt must be positive, got {self.params.dt}")

  @classmethod
  def canonical(cls, *, params: ChainParams = ChainParams()) -> ClosedLoopConfig:
    return cls(n_masses=5, horizon=12, steps=90, params=params)

  @classmethod
  def smoke(cls, *, params: ChainParams = ChainParams()) -> ClosedLoopConfig:
    return cls(n_masses=3, horizon=2, steps=2, params=params)


@dataclass(frozen=True, slots=True)
class ClosedLoopEpisode:
  config: ClosedLoopConfig
  states: np.ndarray
  controls: np.ndarray
  telemetry: tuple[SolverStats, ...]
  points: np.ndarray
  #: ``(steps, horizon + 1, n_masses, 3)`` — the open-loop plan behind every applied control,
  #: i.e. the chain the controller predicted at each node of the horizon it solved.
  plans: np.ndarray
  oracle_z: np.ndarray
  oracle_p: np.ndarray
  oracle_inputs: tuple[dict[str, np.ndarray], ...]
  timing: dict[str, object]


def extract_positions(state: np.ndarray, n_masses: int) -> np.ndarray:
  """Return the fixed anchor and all moving mass positions as ``(M, 3)``."""
  state = np.asarray(state, dtype=np.float64)
  nx = n_state(n_masses)
  if state.shape != (nx,):
    raise ValueError(f"expected state shape {(nx,)}, got {state.shape}")
  return np.concatenate([np.zeros(3), state[: 3 * (n_masses - 1)]]).reshape(n_masses, 3)


def plant_step(state: np.ndarray, control: np.ndarray, params: ChainParams = ChainParams()) -> np.ndarray:
  """Advance the independent, fixed-step NumPy plant by one sample."""
  state = np.asarray(state, dtype=np.float64)
  control = np.asarray(control, dtype=np.float64)
  return rk4_step_np(state, control, params)


def _rollout_guess(state: np.ndarray, config: ClosedLoopConfig) -> np.ndarray:
  nx, nz = n_state(config.n_masses), n_state(config.n_masses) + NU
  guess = np.empty(n_dec(config.n_masses, config.horizon), dtype=np.float64)
  predicted = state.copy()
  zero_control = np.zeros(NU)
  for stage in range(config.horizon):
    guess[stage * nz : stage * nz + nx] = predicted
    guess[stage * nz + nx : (stage + 1) * nz] = zero_control
    predicted = plant_step(predicted, zero_control, config.params)
  guess[config.horizon * nz :] = predicted
  return guess


def _shift_primal(solution: np.ndarray, measured: np.ndarray, config: ClosedLoopConfig) -> np.ndarray:
  """Shift state/control stages and extend the tail with the NumPy model."""
  nx, nz, horizon = n_state(config.n_masses), n_state(config.n_masses) + NU, config.horizon
  states = np.empty((horizon + 1, nx), dtype=np.float64)
  controls = np.empty((horizon, NU), dtype=np.float64)
  for stage in range(horizon):
    states[stage] = solution[stage * nz : stage * nz + nx]
    controls[stage] = solution[stage * nz + nx : (stage + 1) * nz]
  states[-1] = solution[horizon * nz :]

  shifted_states = np.empty_like(states)
  shifted_states[:-1] = states[1:]
  shifted_states[0] = measured
  shifted_controls = np.empty_like(controls)
  if horizon > 1:
    shifted_controls[:-1] = controls[1:]
  shifted_controls[-1] = controls[-1]
  shifted_states[-1] = plant_step(shifted_states[-2], shifted_controls[-1], config.params)

  shifted = np.empty_like(solution)
  for stage in range(horizon):
    shifted[stage * nz : stage * nz + nx] = shifted_states[stage]
    shifted[stage * nz + nx : (stage + 1) * nz] = shifted_controls[stage]
  shifted[horizon * nz :] = shifted_states[-1]
  return shifted


def run_episode(
  config: ClosedLoopConfig | None = None,
  *,
  smoke: bool = False,
  solver: str = "ipopt",
  oracle: str = "alloy",
) -> ClosedLoopEpisode:
  """Run one deterministic model-in-the-loop episode.

  ``config`` selects all physical and sizing parameters.  Passing ``smoke``
  without a config selects the very short smoke point; explicit configuration
  always takes precedence.  The primal trajectory is shifted between solves.
  Equality multipliers are reset because their initial-condition/transition
  roles change after a receding-horizon shift, so reusing them is not safe.
  """
  config = config if config is not None else (ClosedLoopConfig.smoke() if smoke else ClosedLoopConfig.canonical())
  nx, nz = n_state(config.n_masses), n_state(config.n_masses) + NU
  timing = SolveTiming()
  if (solver, oracle) == ("ipopt", "alloy"):
    controller = chain_nlp(config.n_masses, config.horizon)
  elif (solver, oracle) == ("sqp", "alloy"):
    controller = chain_nlp(config.n_masses, config.horizon, solver="sqp")
  elif (solver, oracle) == ("sqp", "casadi"):
    from benchmarks.problems.chain import ca_chain_sqp

    controller = ca_chain_sqp(config.n_masses, config.horizon)
  else:
    raise ValueError(f"unsupported chain solver/oracle pair {solver!r}/{oracle!r}")
  timing.prepared(controller)
  state = initial_state(config.n_masses)
  guess = _rollout_guess(state, config)
  lam_eq0 = np.zeros(nx * (config.horizon + 1))
  lam_ineq0 = np.zeros(0)
  lam_box0 = np.zeros(guess.size)

  states = np.empty((config.steps + 1, nx), dtype=np.float64)
  controls = np.empty((config.steps, NU), dtype=np.float64)
  points = np.empty((config.steps + 1, config.n_masses, 3), dtype=np.float64)
  plans = np.empty((config.steps, config.horizon + 1, config.n_masses, 3), dtype=np.float64)
  states[0], points[0] = state, extract_positions(state, config.n_masses)
  telemetry: list[SolverStats] = []
  oracle_inputs: list[dict[str, np.ndarray]] = []

  for step in range(config.steps):
    p = np.concatenate([state, config.params.array()])
    timing.start_step()
    out = solve_problem(controller, guess, lam_eq0, lam_ineq0, lam_box0, p)
    timing.end_step()
    stats = problem_stats(controller)
    status = None if stats is None else stats.to_solver_status()
    if stats is None or status is None:
      raise RuntimeError(f"{solver}/{oracle} did not return solve status and statistics")
    if not status.ok:
      raise RuntimeError(f"chain NMPC solve failed at step {step} with {solver}/{oracle}: {stats.status.name}")
    solution = np.asarray(out["x"], dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(solution)):
      raise RuntimeError(f"{solver}/{oracle} returned a non-finite trajectory at step {step}")
    oracle_inputs.append({"z": solution.copy(), "p": p.copy()})
    telemetry.append(stats)
    for stage in range(config.horizon + 1):
      plans[step, stage] = extract_positions(solution[stage * nz : stage * nz + nx], config.n_masses)

    control = np.clip(solution[nx:nz], -1.0, 1.0)
    state = plant_step(state, control, config.params)
    if not np.all(np.isfinite(state)):
      raise RuntimeError(f"NumPy plant returned a non-finite state at step {step}")
    controls[step], states[step + 1] = control, state
    points[step + 1] = extract_positions(state, config.n_masses)
    guess = _shift_primal(solution, state, config)

  representative = oracle_inputs[len(oracle_inputs) // 2]
  return ClosedLoopEpisode(
    config,
    states,
    controls,
    tuple(telemetry),
    points,
    plans,
    representative["z"],
    representative["p"],
    tuple(oracle_inputs),
    timing.summary(),
  )
