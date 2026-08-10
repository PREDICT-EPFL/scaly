from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from alloy.toolchain import solver_diagnostic, solver_loadable
from benchmarks.harness.closed_loop import run_chain, run_tracking

pytestmark = pytest.mark.skipif(not solver_loadable("ipopt"), reason=solver_diagnostic("ipopt"))


@pytest.mark.parametrize("runner,problem", [(run_chain, "chain"), (run_tracking, "tracking")])
def test_solver_backed_smoke_episode_writes_replay_and_harvest_artifacts(tmp_path: Path, runner, problem: str) -> None:
  output = runner(smoke=True, out_dir=tmp_path, cli_args=["closed-loop", "--problem", problem, "--smoke"])

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
  with np.load(output / "representative_fe_inputs.npz") as inputs:
    assert set(inputs.files) == {"p", "z"}
    assert all(np.all(np.isfinite(inputs[name])) for name in inputs.files)
