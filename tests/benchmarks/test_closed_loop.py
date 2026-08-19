from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.harness.closed_loop import run_chain, run_race_cars

pytestmark = pytest.mark.solver("ipopt")


@pytest.mark.parametrize("runner,problem", [(run_chain, "chain"), (run_race_cars, "race_cars")])
def test_solver_backed_smoke_episode_writes_replay_and_harvest_artifacts(tmp_path: Path, runner, problem: str) -> None:
  output = runner(smoke=True, out_dir=tmp_path, cli_args=["closed-loop", "--problem", problem, "--smoke"])
  assert output.name == "ipopt+alloy"

  for name in (
    "episode.mcap",
    "rollout.npz",
    "config.json",
    "summary.json",
    "provenance.json",
    "metadata.json",
    "representative_fe_inputs.npz",
  ):
    assert (output / name).is_file(), name
  assert (output / "episode.mcap").stat().st_size > 0
  metadata = json.loads((output / "metadata.json").read_text())
  assert metadata["representative_step"] in metadata["successful_steps"]
  summary = json.loads((output / "summary.json").read_text())
  assert summary["solver"] == "ipopt" and summary["oracle"] == "alloy"
  with np.load(output / "representative_fe_inputs.npz") as inputs:
    assert set(inputs.files) == {"p", "z"}
    assert all(np.all(np.isfinite(inputs[name])) for name in inputs.files)


def test_race_failure_closes_incremental_mcap_and_writes_partial_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  from alloy.solvers import ALLOY_SOLVER_STATS_VERSION, AlloySolveStatus, SolverStats
  from benchmarks.problems.race_cars import NX, NU
  from benchmarks.problems.race_cars import closed_loop
  from benchmarks.problems.race_cars.closed_loop import StepRecord, StepTelemetry

  def stats(status: AlloySolveStatus, native: int) -> SolverStats:
    return SolverStats(ALLOY_SOLVER_STATS_VERSION, status, native, 2, 1.0, 0.01, 0.002, 0.0, 0.006, 0.001, 0.001, 2, 2, 2, 2, 2)

  def fail(config, *, solver, oracle, record_step, **_kwargs):
    reference = np.zeros((config.horizon + 1, NX))
    prediction = reference.copy()
    good_stats = stats(AlloySolveStatus.OK, 1)
    telemetry = StepTelemetry(good_stats, 0.1, 0, 0.0, 0.0, 0.0)
    record_step(
      StepRecord(
        0,
        np.zeros(NX),
        np.zeros(NX),
        np.zeros(NU),
        prediction,
        reference,
        {"z": np.zeros(1), "p": np.zeros(1)},
        good_stats,
        telemetry,
      )
    )
    failed_stats = stats(AlloySolveStatus.MAX_ITER, -1)
    record_step(StepRecord(1, np.ones(NX), np.ones(NX), None, None, reference, {"z": np.ones(1), "p": np.ones(1)}, failed_stats, None))
    raise RuntimeError(f"synthetic {solver}+{oracle} failure")

  monkeypatch.setattr(closed_loop, "run_episode", fail)
  with pytest.raises(RuntimeError, match=r"synthetic ipopt\+alloy failure"):
    run_race_cars(smoke=True, out_dir=tmp_path, cli_args=["closed-loop", "--problem", "race_cars", "--smoke"])

  output = tmp_path / "race_cars" / "ipopt+alloy"
  data = (output / "episode.mcap").read_bytes()
  assert data.startswith(b"\x89MCAP0\r\n") and data.endswith(b"\x89MCAP0\r\n")
  with np.load(output / "rollout.npz") as rollout:
    assert rollout["state"].shape == (2, NX)
    assert rollout["control"].shape == (1, NU)
  summary = json.loads((output / "summary.json").read_text())
  assert summary["failed_step"] == 1 and summary["status"] == "MAX_ITER" and summary["native_status"] == -1
