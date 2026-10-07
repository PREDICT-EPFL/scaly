"""Check study progress persistence and the machine's measurement reservation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.harness import study


@pytest.fixture
def study_args(tmp_path, monkeypatch):
  monkeypatch.setattr(Path, "home", lambda: tmp_path)
  monkeypatch.setattr(study, "collect", lambda *args: {"fixture": True})
  monkeypatch.setattr(study.gbench, "compiler", lambda: "cc")
  monkeypatch.setattr(study, "report", lambda out: out / "report.md")
  return SimpleNamespace(
    out_dir=tmp_path / "out", overwrite=False, repetitions=1, order_seed=0, headline=False, problem=["chain", "race_cars"], only=["sweep"]
  )


@pytest.mark.parametrize("status", [0, 1])
def test_progress_is_written_between_commands(study_args, monkeypatch, status):
  calls = []
  manifest_path = study_args.out_dir / "study.json"

  def run(command):
    if calls:
      manifest = json.loads(manifest_path.read_text())
      assert manifest["fixture"] is True
      assert len(manifest["commands"]) == 1
      assert manifest["commands"][0]["command"] == calls[0][1:]
      assert manifest["commands"][0]["returncode"] == status
    calls.append(command)
    return SimpleNamespace(returncode=status)

  monkeypatch.setattr(study.subprocess, "run", run)
  assert study.run_study(study_args, ["study"]) is (status == 0)
  commands = json.loads(manifest_path.read_text())["commands"]
  assert len(commands) == len(calls) == 2
  assert all(set(item) == {"command", "returncode", "started", "seconds"} for item in commands)


def test_measurement_reservation_refuses_to_start(study_args, monkeypatch):
  reservation = Path.home() / ".scaly-measuring"
  reservation.write_text("owner: another study\n")
  monkeypatch.setattr(study.subprocess, "run", lambda *args: pytest.fail("reserved machine started a command"))
  with pytest.raises(SystemExit, match="owner: another study"):
    study.run_study(study_args, ["study"])
  assert reservation.read_text() == "owner: another study\n"
  assert not study_args.out_dir.exists()
