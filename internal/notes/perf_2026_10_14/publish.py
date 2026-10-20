"""Audit the October 14 study and write the data files the benchmark pages read."""

# ruff: noqa: E402 -- direct execution must add the repository root before package imports

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from benchmarks.harness.report import COUNTS, _episode

WORKLOADS = ("race_cars", "npmpc", "unbumpercars", "chain")
STEPS = {"chain": 90, "npmpc": 100, "race_cars": 1367, "unbumpercars": 200}
STATUSES = {("ok", "ok"), ("timeout", "skipped"), ("skipped_after_failure", "skipped")}


def read_csv(path):
  with path.open() as stream:
    return list(csv.DictReader(stream))


def write_csv(path, rows):
  with path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def main(study):
  manifest = json.loads((study / "study.json").read_text())
  assert not manifest["git_dirty"] and len(manifest["commands"]) == 12 and all(c["returncode"] == 0 for c in manifest["commands"])
  assert manifest["vector_libm"] == "glibc" and manifest["cpu_settings"]["boost_enabled"] is False
  assert all(p["scaling_governor"] == "performance" for p in manifest["cpu_settings"]["policies"].values())
  raw, sweep = [], []
  for workload in WORKLOADS:
    rows = read_csv(study / "sweep" / workload / f"{workload}.csv")
    assert all((r["compile_status"], r["runtime_status"]) in STATUSES for r in rows)
    assert all(r["runtime_status"] == "ok" for r in rows if r["backend"] == "scaly")
    raw += rows
    sweep += read_csv(study / "sweep" / workload / f"{workload}.summary.csv")
  summary = json.loads((study / "closed-loop" / "closed_loop.summary.json").read_text())
  assert len(summary["runs"]) == len(summary["groups"]) == 15 and len(summary["oracle_comparisons"]) == 7
  assert all(r["steps"] == r["successful_steps"] == STEPS[r["problem"]] for r in summary["runs"])
  pairs = []
  for pair in summary["oracle_comparisons"]:
    assert pair["same_steps"] and all(pair[f"{key}_mismatches"] == 0 for key in COUNTS)
    for metric in ("solver_time_ms", "fe_time_ms"):
      ratios = np.asarray(pair[f"{metric}_speedups"])
      p10, median, p90 = np.percentile(ratios, [10, 50, 90])
      key = f"{pair['problem']}:{pair['solver']}:{metric}"
      pairs.append(
        dict(problem=pair["problem"], solver=pair["solver"], metric=metric, key=key, steps=len(ratios), median=median, p10=p10, p90=p90)
        | dict(iteration_mismatches=pair["iterations_mismatches"])
      )
  steps = []
  for telemetry in sorted(study.glob("closed-loop/*/*/repeat_*/*/*/telemetry.csv")):
    run, rows = _episode(telemetry)
    for row in rows:
      values = {key: f"{value:.6g}" if isinstance(value, float) else value for key, value in row.items()}
      steps.append(dict(problem=run["problem"], solver=run["solver"], oracle=run["oracle"]) | values)
  target = ROOT / "docs/assets/benchmarks"
  write_csv(target / "sweep.summary.csv", sweep)
  write_csv(target / "sweep.csv", raw)
  (target / "closed_loop.summary.csv").write_text((study / "closed-loop/closed_loop.summary.csv").read_text())
  write_csv(target / "closed_loop_speedup.summary.csv", pairs)
  write_csv(target / "closed_loop.steps.csv", steps)
  print(f"Audit passed: {len(raw)} sweep attempts, {len(steps)} closed-loop steps. Data written to {target}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("study", type=Path)
  main(parser.parse_args().study.resolve())
