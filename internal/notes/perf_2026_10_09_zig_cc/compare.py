"""Compare the gcc and zig cc arms of the closed-loop measurement for #82, cell by cell."""

import csv
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results/zig-cc-2026-10-09")
step_fields = ("fe_time_ms", "solver_time_ms", "iterations")


def episode(path: Path) -> dict[str, float]:
  with (path / "telemetry.csv").open() as fp:
    steps = list(csv.DictReader(fp))[1:]
  out = {f: statistics.mean(float(s[f]) for s in steps) for f in step_fields}
  out["build_ms"] = json.loads((path / "summary.json").read_text())["build_ms"]
  return out


def cell(arm: str, problem: str, solver: str) -> dict[str, tuple[float, float]]:
  runs = [episode(p) for p in sorted((root / arm).glob(f"repeat_*/{problem}/{solver}+scaly"))]
  assert len(runs) == 5, (arm, problem, solver, len(runs))
  stats = {}
  for f in runs[0]:
    values = [r[f] for r in runs]
    stats[f] = (statistics.mean(values), statistics.stdev(values) / statistics.mean(values))
  return stats


print("| Problem | Solver | Mean per step after the first | gcc 13.3 (CV) | zig cc 0.16 (CV) | zig / gcc |")
print("|---|---|---|---:|---:|---:|")
for problem in ("race_cars", "unbumpercars", "npmpc"):
  for solver in ("ipopt", "sqp"):
    gcc, zig = cell("gcc", problem, solver), cell("zig", problem, solver)
    for f in gcc:
      (g, gcv), (z, zcv) = gcc[f], zig[f]
      print(f"| {problem} | {solver} | {f} | {g:.4g} ({gcv:.1%}) | {z:.4g} ({zcv:.1%}) | {z / g:.3f} |")
