"""Benchmark generation, correctness, and sweep helpers."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results"
CLOSED_LOOP_RESULTS = RESULTS / "closed-loop"
SMOKE_RESULTS = RESULTS / "smoke"
SWEEP_RESULTS = RESULTS / "sweep"


def closed_loop_results_root(*, smoke: bool, out_dir: Path | None = None) -> Path:
  if out_dir is not None:
    return out_dir
  return SMOKE_RESULTS / "closed-loop" if smoke else CLOSED_LOOP_RESULTS


def solver_oracle_name(solver: str, oracle: str | None) -> str:
  if solver == "none":
    return solver
  if oracle is None:
    raise ValueError(f"solver {solver!r} requires an oracle")
  return f"{solver}+{oracle}"


def solve_problem(solver, x0, lam_eq, lam_ineq, lam_box, params):
  """Run either a typed Alloy solver Function or the benchmark CasADi adapter."""
  import numpy as np
  import alloy as al

  if not isinstance(solver, al.Function):
    return solver(x0, lam_eq, lam_ineq, lam_box, params)

  x, lam_box, lam_eq, lam_ineq = solver.numerical_call((x0, lam_box, lam_eq, lam_ineq, params))
  descriptor = solver.descriptor
  base = descriptor.base
  if isinstance(base, al.Function):
    values = base.numerical_call((np.asarray(x).reshape(-1), params))
  else:
    evaluator = getattr(solver, "_benchmark_base", None)
    if evaluator is None:
      raise TypeError(f"solver {solver.name!r} has no numerical benchmark oracle")
    values = evaluator(np.asarray(x).reshape(-1), params)
  if isinstance(values, tuple):
    cost, constraints = values
  else:
    cost, constraints = values, np.zeros(0)
  constraints = np.asarray(constraints).reshape(-1)
  return {
    "x": x,
    "f": np.asarray(cost).reshape(()),
    "h_eq": constraints[: descriptor.n_eq],
    "g_ineq": constraints[descriptor.n_eq :],
    "lam_eq": lam_eq,
    "lam_ineq": lam_ineq,
    "lam_box": lam_box,
  }


def problem_stats(solver):
  """Return stats from either a typed Alloy Function or the CasADi adapter."""
  import alloy as al

  return solver.solver_stats() if isinstance(solver, al.Function) else solver.last_stats
