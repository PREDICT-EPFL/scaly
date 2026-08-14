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
