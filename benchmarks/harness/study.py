"""Run the frozen headline grids into one directory: every kernel sweep, every closed loop, then the report."""

from __future__ import annotations

import json
import subprocess
import sys

from benchmarks.harness import ROOT, gbench
from benchmarks.harness.provenance import collect
from benchmarks.harness.report import report

# The exact sparse Lagrangian Hessian grids the results pages quote. Chain is internal validation.
SWEEP_GRID = {
  "race_cars": [1, 5, 10, 25, 40, 50, 100, 200, 500],
  "npmpc": [6, 12, 25, 50, 100, 200],
  "unbumpercars": [2, 4, 8, 16, 32],
  "chain": [3, 5, 9],
}
PROBLEMS = tuple(SWEEP_GRID)
PARTS = ("sweep", "closed-loop")


def run_study(args, cli_args: list[str]) -> bool:
  out = args.out_dir.resolve()
  if out.exists() and any(out.iterdir()):
    raise SystemExit(f"study requires an unused output directory: {out}")
  out.mkdir(parents=True, exist_ok=True)
  run_py = str(ROOT / "benchmarks" / "run.py")
  common = ["--repetitions", str(args.repetitions), "--order-seed", str(args.order_seed)]
  if args.headline:
    common += ["--headline", "--boost", args.boost]
  commands = []
  for problem in args.problems:
    if "sweep" in args.only:
      sizes = ",".join(str(size) for size in SWEEP_GRID[problem])
      target = out / "sweep" / problem / f"{problem}.csv"
      commands.append(
        [sys.executable, run_py, "sweep", "--workloads", problem, "--sizes", sizes, "--benchmark-min-time", "0.5s", *common, "--out", str(target)]
      )
    if "closed-loop" in args.only:
      target = out / "closed-loop" / problem
      commands.append(
        [sys.executable, run_py, "closed-loop", "--problem", problem, "--solver", "sqp", "--oracle", "both", *common, "--out-dir", str(target)]
      )
  record = []
  for command in commands:
    print("$ " + " ".join(command[1:]), flush=True)
    status = subprocess.run(command).returncode
    record.append({"command": command[1:], "returncode": status})
  manifest = {**collect(ROOT, gbench.compiler(), cli_args), "commands": record}
  (out / "study.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  print(f"report written to {report(out)}")
  return all(item["returncode"] == 0 for item in record)
