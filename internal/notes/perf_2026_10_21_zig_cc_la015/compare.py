"""Compare the gcc, zig cc and clang 20 arms of the 2026-10-21 la015 closed-loop measurement for #82."""

import csv
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results/zig017-2026-10-21")
arms = ("gcc", "zig", "clang20")


def episode(arm: str, problem: str, solver: str) -> dict[str, float]:
  (path,) = (root / arm).glob(f"repeat_*/{problem}/{solver}+scaly")
  with (path / "telemetry.csv").open() as fp:
    steps = list(csv.DictReader(fp))[1:]
  out = {f: statistics.mean(float(s[f]) for s in steps) for f in ("fe_time_ms", "solver_time_ms", "iterations")}
  out["build_ms"] = json.loads((path / "summary.json").read_text())["build_ms"]
  return out


print("| Problem | Solver | Mean per step after the first | gcc 13.3 | zig cc 0.17 | clang 20.1 | zig / gcc | zig / clang 20 |")
print("|---|---|---|---:|---:|---:|---:|---:|")
for problem in ("race_cars", "unbumpercars", "npmpc"):
  for solver in ("ipopt", "sqp"):
    gcc, zig, clang = (episode(arm, problem, solver) for arm in arms)
    for f in gcc:
      print(f"| {problem} | {solver} | {f} | {gcc[f]:.4g} | {zig[f]:.4g} | {clang[f]:.4g} | {zig[f] / gcc[f]:.3f} | {zig[f] / clang[f]:.3f} |")
