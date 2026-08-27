"""Per-iteration oracle cost against decoder width, measured on real solves.

The paper this problem comes from shrank its decoder to two layers of 32 to meet a 20 ms sampling
time, so the question worth measuring is how much wider the network can be before the solve stops
fitting. Only the shipped width has a trained checkpoint, and an untrained decoder cannot produce a
meaningful trajectory, so the quantity measured here is deliberately narrower than a closed-loop
column: **time per solver iteration**, which does not depend on the trajectory being sensible.

Run it with

```bash
uv run python -m benchmarks.problems.npmpc.width_study
```

The numbers it prints are the ones quoted in `README.md`, and the real-time claim there anchors on
the trained closed-loop measurement and scales it by the ratios in this table. Two caveats travel
with that: the anchor is trained and these rows are not, and the iteration count a *trained* wide
decoder would need is unknown. The README states both.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import sys

import numpy as np

from benchmarks.harness import problem_stats, solve_problem
from benchmarks.problems.npmpc import (
  CostWeights,
  Decoder,
  constraint_counts,
  n_dec,
  npmpc_nlp,
  pack_params,
  plant_step,
  random_decoder_weights,
)
from benchmarks.problems.npmpc.closed_loop import EpisodeConfig, initial_guess, shifted_guess

DEFAULT_WIDTHS = (32, 64, 128, 256)
# The cold start costs several times a warm one, so it is excluded; what is left is the warm-started
# per-iteration cost the real-time budget is spent on.
DEFAULT_STEPS = 6
# Untrained weights are arbitrary, and at some widths a given draw sends the trajectory somewhere the
# solver will not follow within a few steps. The seed is therefore part of the cell rather than a
# constant: the first draw that completes every warm solve is the one measured, and it is reported.
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True, slots=True)
class WidthCell:
  """One measured cell: warm-started solves at one decoder width through one solver."""

  width: int
  solver: str
  seed: int
  #: warm steps that contributed, excluding the cold start
  steps: int
  iterations: float
  total_ms_per_iteration: float
  fe_ms_per_iteration: float
  note: str = ""

  @property
  def fe_share(self) -> float:
    return self.fe_ms_per_iteration / self.total_ms_per_iteration


def measure(width: int, solver: str = "sqp", *, steps: int = DEFAULT_STEPS, seed: int = 0) -> WidthCell:
  """Warm-start a few solves at one decoder width and return the per-iteration cost.

  The weights are untrained, so the terminal weight is the plain state weight rather than a Riccati
  solve: an untrained model has no reason to be stabilizable upright, and the terminal matrix is a
  constant either way, so this cannot move a timing. A failed solve ends the cell early and is
  reported rather than raised, because the interesting failure at these widths is the trajectory
  wandering off, not the timing.
  """
  decoder = Decoder((width, width))
  config = replace(EpisodeConfig(), decoder=decoder)
  pw = pack_params(decoder, random_decoder_weights(decoder, seed))
  controller = npmpc_nlp(
    np.diag(CostWeights().x_end),
    config.horizon,
    decoder,
    weights=config.weights,
    dt=config.dt,
    solver=solver,
    options=(
      {"tol": config.sqp_tol, "max_iter": config.sqp_max_iter}
      if solver == "sqp"
      else {"print_level": 0, "sb": "yes", "tol": config.ipopt_tol, "max_iter": config.ipopt_max_iter, "warm_start_init_point": "yes"}
    ),
  )
  n_eq, n_ineq = constraint_counts(config.horizon)
  zeros = (np.zeros(n_eq), np.zeros(n_ineq), np.zeros(n_dec(config.horizon)))

  state = np.array(config.x_start, dtype=np.float64)
  guess = initial_guess(state, config)
  offset = 4 * (config.horizon + 1)
  totals, evaluations, iterations, note = [], [], [], ""
  for step in range(steps + 1):
    out = solve_problem(controller, guess, *zeros, np.concatenate([state, pw]))
    stats = problem_stats(controller)
    status = None if stats is None else stats.to_solver_status()
    if stats is None or status is None or not status.ok:
      note = f"stopped at step {step}: {'missing stats' if stats is None else stats.status.name}"
      break
    if step:  # the cold start is not a warm solve
      totals.append(stats.t_total)
      evaluations.append(stats.t_fe)
      iterations.append(stats.iter)
    solution = np.asarray(out["x"], dtype=np.float64).reshape(-1)
    state = plant_step(state, np.clip(solution[offset : offset + 1], -0.05, 0.05), config.dt, config.plant, config.substeps)
    guess = shifted_guess(solution, state, config, pw)

  if not iterations or not sum(iterations):
    return WidthCell(width, solver, seed, len(iterations), 0.0, float("nan"), float("nan"), note or "no warm solve completed")
  return WidthCell(
    width=width,
    solver=solver,
    seed=seed,
    steps=len(iterations),
    iterations=float(np.mean(iterations)),
    total_ms_per_iteration=1e3 * float(np.sum(totals)) / float(np.sum(iterations)),
    fe_ms_per_iteration=1e3 * float(np.sum(evaluations)) / float(np.sum(iterations)),
    note=note,
  )


def best_cell(width: int, solver: str, *, steps: int = DEFAULT_STEPS, seeds: tuple[int, ...] = DEFAULT_SEEDS) -> WidthCell:
  """The first weight draw that completes every warm solve, or the one that got furthest."""
  best = None
  for seed in seeds:
    cell = measure(width, solver, steps=steps, seed=seed)
    if cell.steps == steps and np.isfinite(cell.total_ms_per_iteration):
      return cell
    if best is None or cell.steps > best.steps:
      best = cell
  assert best is not None
  return best


def main(widths: tuple[int, ...] = DEFAULT_WIDTHS, *, steps: int = DEFAULT_STEPS) -> int:
  print(f"{'W':>5} {'solver':>6} {'seed':>5} {'steps':>6} {'iters':>6} {'total/iter':>12} {'fe/iter':>10} {'fe share':>9}  note")
  incomplete = 0
  for width in widths:
    for solver in ("ipopt", "sqp"):
      cell = best_cell(width, solver, steps=steps)
      if cell.steps < steps:
        incomplete += 1
      if not np.isfinite(cell.total_ms_per_iteration):
        print(f"{cell.width:5d} {cell.solver:>6} {cell.seed:5d} {cell.steps:6d} {'':>6} {'':>12} {'':>10} {'':>9}  {cell.note}")
        continue
      print(
        f"{cell.width:5d} {cell.solver:>6} {cell.seed:5d} {cell.steps:6d} {cell.iterations:6.2f} "
        f"{cell.total_ms_per_iteration:11.3f}ms {cell.fe_ms_per_iteration:9.3f}ms {100 * cell.fe_share:8.0f}%  {cell.note}"
      )
  # an incomplete cell still carries a usable per-iteration cost, but say so rather than let the
  # table read as if every width had held for the whole run
  return 1 if incomplete else 0


if __name__ == "__main__":
  sys.exit(main())
