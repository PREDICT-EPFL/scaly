"""Summarize and compare the saved Windows Zig SQP episodes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
RESULTS = ROOT / "benchmarks/results/windows-zig-sqp/canonical"
PROBLEMS = ("chain", "npmpc", "race_cars", "unbumpercars")
COUNTS = ("n_eval_f", "n_eval_g", "n_eval_grad_f", "n_eval_jac_g", "n_eval_h")


def main() -> None:
  rows = []
  checks = {}
  paired = []
  for problem in PROBLEMS:
    saved = {}
    for oracle in ("scaly", "casadi"):
      directory = RESULTS / problem / "repeat_1" / problem / f"sqp+{oracle}"
      summary = json.loads((directory / "summary.json").read_text())
      config = json.loads((directory / "config.json").read_text())
      with (directory / "telemetry.csv").open(newline="") as stream:
        telemetry = list(csv.DictReader(stream))
      total = np.array([float(item["solver_time_ms"]) for item in telemetry])
      fe = np.array([float(item["fe_time_ms"]) for item in telemetry])
      row = {
        "problem": problem,
        "oracle": oracle,
        "steps": len(telemetry),
        "successful_steps": sum(item["success"] == "True" for item in telemetry),
        "solver_mean_ms": float(np.mean(total)),
        "solver_median_ms": float(np.median(total)),
        "solver_p95_ms": float(np.quantile(total, 0.95)),
        "fe_mean_ms": float(np.mean(fe)),
        "steady_wall_ms": summary["steady_step_ms"],
        "build_ms": summary["build_ms"],
        "first_solve_ms": summary["first_solve_ms"],
        "iterations_mean": float(np.mean([float(item["iterations"]) for item in telemetry])),
      }
      assert row["steps"] and row["successful_steps"] == row["steps"], row
      rows.append(row)
      saved[oracle] = (directory, config, telemetry, total, fe)
    left, right = saved["scaly"], saved["casadi"]
    assert left[1] == right[1], f"configuration mismatch: {problem}"
    assert len(left[2]) == len(right[2]), f"episode length mismatch: {problem}"
    iterations = sum(a["iterations"] != b["iterations"] for a, b in zip(left[2], right[2], strict=True))
    counts_available = all(key in left[2][0] and key in right[2][0] for key in COUNTS)
    count_mismatches = sum(any(a[key] != b[key] for key in COUNTS) for a, b in zip(left[2], right[2], strict=True)) if counts_available else None
    differences = {}
    with np.load(left[0] / "rollout.npz") as a, np.load(right[0] / "rollout.npz") as b:
      assert set(a.files) == set(b.files)
      for key in a.files:
        assert a[key].shape == b[key].shape, (problem, key)
        assert np.all(np.isfinite(a[key])) and np.all(np.isfinite(b[key])), (problem, key)
        differences[key] = float(np.max(np.abs(a[key] - b[key]))) if a[key].size else 0.0
    checks[problem] = {
      "configurations_equal": True,
      "steps_per_oracle": len(left[2]),
      "all_steps_successful": True,
      "iteration_mismatches": iterations,
      "oracle_count_mismatches": count_mismatches,
      "rollout_max_abs_differences": differences,
    }
    ratio = right[3] / left[3]
    fe_ratio = right[4] / left[4]
    paired.append(
      {
        "problem": problem,
        "steps": len(ratio),
        "solver_mean_ratio_casadi_over_scaly": float(np.mean(right[3]) / np.mean(left[3])),
        "solver_paired_ratio_median": float(np.median(ratio)),
        "solver_paired_ratio_p10": float(np.quantile(ratio, 0.1)),
        "solver_paired_ratio_p90": float(np.quantile(ratio, 0.9)),
        "fe_mean_ratio_casadi_over_scaly": float(np.mean(right[4]) / np.mean(left[4])),
        "fe_paired_ratio_median": float(np.median(fe_ratio)),
        "iteration_mismatches": iterations,
        "oracle_count_mismatches": count_mismatches,
      }
    )
  for name, values in (("summary.csv", rows), ("paired.csv", paired)):
    with (OUTPUT / name).open("w", newline="") as stream:
      writer = csv.DictWriter(stream, fieldnames=list(values[0]))
      writer.writeheader()
      writer.writerows(values)
  (OUTPUT / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
  for row in rows:
    print(f"{row['problem']:14} {row['oracle']:6} {row['steps']:4} steps: total={row['solver_mean_ms']:.6f} ms, FE={row['fe_mean_ms']:.6f} ms")
  print(json.dumps(checks, indent=2))


if __name__ == "__main__":
  main()
