"""Check the closed-loop report's per-step statistics on a synthetic single-process study."""

import csv
import json

import numpy as np
import pytest

from benchmarks.harness.report import closed_loop_summary, closed_loop_tables


def _episode(root, oracle, solver_ms, fe_ms):
  path = root / "repeat_1" / "race_cars" / f"sqp+{oracle}"
  path.mkdir(parents=True)
  with (path / "telemetry.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(("step", "success", "solver_time_ms", "fe_time_ms", "iterations", "globalization_time_ms", "glue_time_ms", "qp_time_ms"))
    for step, (total, fe) in enumerate(zip(solver_ms, fe_ms, strict=True)):
      writer.writerow((step, True, total, fe, 3, 0.0, 0.0, total - fe))
  (path / "summary.json").write_text(json.dumps({}))
  np.savez(path / "rollout.npz", state=np.zeros((4, 2)), control=np.zeros((3, 1)))


def test_one_process_reports_paired_per_step_speedups(tmp_path):
  _episode(tmp_path, "scaly", [1.0, 2.0, 4.0], [0.5, 1.0, 1.0])
  _episode(tmp_path, "casadi", [2.0, 3.0, 4.0], [1.0, 2.0, 3.0])
  summary = closed_loop_summary(tmp_path)
  (comparison,) = summary["oracle_comparisons"]
  assert comparison["solver_time_ms_speedups"] == [2.0, 1.5, 1.0]
  assert comparison["fe_time_ms_speedups"] == [2.0, 2.0, 3.0]
  scaly = next(g for g in summary["groups"] if g["oracle"] == "scaly")
  assert scaly["processes"] == 1 and scaly["solver_time_ms"]["cv"] is None
  assert scaly["step_solver_time_ms"]["median"] == 2.0
  assert scaly["step_solver_time_ms"]["stdev"] == pytest.approx(np.std([1.0, 2.0, 4.0], ddof=1))
  assert "| race_cars | sqp | 1 | 1.50× (1.10–1.90) | 2.00× (2.00–2.80) |" in closed_loop_tables(summary)
