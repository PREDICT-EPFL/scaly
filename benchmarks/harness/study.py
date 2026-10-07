"""Run the frozen headline grids into one directory: every kernel sweep, every closed loop, then the report."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from benchmarks.harness import CLOSED_LOOP_PAIRS, ROOT, gbench
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
  measuring = Path.home() / ".scaly-measuring"
  if measuring.exists():
    raise SystemExit(f"measurement already owns this machine ({measuring}):\n{measuring.read_text()}")
  out = args.out_dir.resolve()
  if out.exists() and any(out.iterdir()) and not args.overwrite:
    raise SystemExit(f"study requires an unused output directory: {out} (or pass --overwrite)")
  out.mkdir(parents=True, exist_ok=True)
  run_py = str(ROOT / "benchmarks" / "run.py")
  common = ["--repetitions", str(args.repetitions), "--order-seed", str(args.order_seed)]
  if args.headline:
    common += ["--headline", "--boost", args.boost]
  commands = []
  for problem in args.problem:
    if "sweep" in args.only:
      sizes = ",".join(str(size) for size in SWEEP_GRID[problem])
      target = out / "sweep" / problem / f"{problem}.csv"
      commands.append(
        [sys.executable, run_py, "sweep", "--workloads", problem, "--sizes", sizes, "--benchmark-min-time", "0.5s", *common, "--out", str(target)]
      )
    if "closed-loop" in args.only:
      overwrite = ["--overwrite"] if args.overwrite else []
      for solver in ("ipopt", "sqp"):
        target = out / "closed-loop" / problem / solver
        oracles = ",".join(oracle for s, oracle in CLOSED_LOOP_PAIRS[problem] if s == solver and oracle)
        commands.append(
          [
            sys.executable,
            run_py,
            "closed-loop",
            "--problem",
            problem,
            "--solver",
            solver,
            "--oracle",
            oracles,
            *common,
            *overwrite,
            "--out-dir",
            str(target),
          ]
        )
  record = []
  manifest = {**collect(ROOT, gbench.compiler(), cli_args), "commands": record}
  for command in commands:
    print("$ " + " ".join(command[1:]), flush=True)
    started = datetime.now(timezone.utc)
    tick = time.monotonic()
    status = subprocess.run(command).returncode
    record.append({"command": command[1:], "returncode": status, "started": started.isoformat(), "seconds": round(time.monotonic() - tick, 1)})
    (out / "study.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  (out / "study.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  print(f"report written to {report(out)}")
  return all(item["returncode"] == 0 for item in record)
