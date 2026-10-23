from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pytest

from benchmarks.harness.closed_loop import run_chain, run_race_cars


class SmokeRunner(Protocol):
  def __call__(self, *, smoke: bool, out_dir: Path, cli_args: list[str], solver: str = "ipopt") -> Path: ...


if TYPE_CHECKING:
  runner: SmokeRunner = run_chain
  runner(smkoe=True, out_dir=Path("."), cli_args=[])  # ty: ignore[unknown-argument, missing-argument]
  runner = run_race_cars


@pytest.mark.parametrize("runner,problem", [(run_chain, "chain"), (run_race_cars, "race_cars")])
@pytest.mark.parametrize("solver", [pytest.param("ipopt", marks=pytest.mark.solver("ipopt")), pytest.param("sqp", marks=pytest.mark.solver("sqp"))])
def test_solver_backed_smoke_episode_writes_replay_and_harvest_artifacts(tmp_path: Path, runner: SmokeRunner, problem: str, solver: str) -> None:
  output = runner(smoke=True, out_dir=tmp_path, cli_args=["closed-loop", "--problem", problem, "--smoke"], solver=solver)
  assert output.name == f"{solver}+scaly"

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
  assert summary["solver"] == solver and summary["oracle"] == "scaly"
  with np.load(output / "representative_fe_inputs.npz") as inputs:
    assert set(inputs.files) == {"p", "z"}
    assert all(np.all(np.isfinite(inputs[name])) for name in inputs.files)
