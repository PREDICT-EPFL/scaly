"""Release reuse requires completed checks and the exact surviving artifacts they tested."""

import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import zipfile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "release.py"
SHA = "a" * 40
PACKAGES = {"scaly": "0.1.0", "scaly-piqp": "0.1.0"}
DISTS = ["dist-pure", "dist-linux-x86_64", "dist-linux-arm64", "dist-macos-arm64", "dist-windows-x86_64"]


@pytest.fixture
def release():
  spec = importlib.util.spec_from_file_location("release_workflow", SCRIPT)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class GitHub:
  def __init__(self):
    self.runs = {"release.yml": [], "nightly.yml": []}
    self.artifacts = {}
    self.archives = {}
    self.calls = []

  def add(self, workflow, run_id, kind=None, **overrides):
    run = {
      "id": run_id,
      "head_sha": SHA,
      "head_branch": "main",
      "event": "workflow_dispatch",
      "status": "completed",
      "conclusion": "success",
      "run_attempt": 1,
      "created_at": f"2026-11-03T00:00:{run_id:02d}Z",
    } | overrides
    self.runs[workflow].insert(0, run)
    self.artifacts[run_id] = []
    if kind:
      names = ["release-dist"] if kind == "testpypi" else DISTS
      artifacts = [{"id": run_id * 100 + i, "name": name, "expired": False} for i, name in enumerate(names)]
      proof = {"id": run_id * 100 + 90, "name": f"{kind}-validated", "expired": False}
      self.artifacts[run_id] = artifacts + [proof]
      record = {
        "sha": SHA,
        "run_id": run_id,
        "run_attempt": 1,
        "kind": kind,
        "artifacts": [a["id"] for a in artifacts],
        "packages": PACKAGES if kind == "testpypi" else {},
      }
      self.set_record(proof["id"], record)
    return run

  def set_record(self, artifact_id, record):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
      archive.writestr("validation.json", json.dumps(record))
    self.archives[artifact_id] = data.getvalue()

  def gh(self, *args):
    self.calls.append(args)
    path = args[-1]
    if "/workflows/" in path:
      workflow = path.split("/workflows/")[1].split("/")[0]
      runs = self.runs[workflow]
      return json.dumps([{"workflow_runs": runs[i : i + 100]} for i in range(0, len(runs), 100)]).encode()
    if "/runs/" in path:
      run_id = int(path.split("/runs/")[1].split("/")[0])
      return json.dumps([{"artifacts": self.artifacts[run_id]}]).encode()
    return self.archives[int(path.split("/artifacts/")[1].split("/")[0])]


def select(release, monkeypatch, github):
  monkeypatch.setattr(release, "gh", github.gh)
  return release.select_source("PREDICT-EPFL/scaly", SHA, 99, PACKAGES)


def test_no_previous_validation_runs_the_full_suite(release, monkeypatch):
  assert select(release, monkeypatch, GitHub()) == {
    "artifact-run": "99",
    "artifact-ids": "",
    "reused": "false",
    "testpypi-verified": "false",
  }


def test_verified_testpypi_files_take_priority_over_a_newer_nightly(release, monkeypatch):
  github = GitHub()
  github.add("release.yml", 1, "testpypi")
  github.add("nightly.yml", 2, "nightly")
  assert select(release, monkeypatch, github) == {
    "artifact-run": "1",
    "artifact-ids": "100",
    "reused": "true",
    "testpypi-verified": "true",
  }


def test_a_full_nightly_reuses_every_platform_and_still_needs_testpypi(release, monkeypatch):
  github = GitHub()
  github.add("nightly.yml", 1, "nightly", event="schedule")
  assert select(release, monkeypatch, github) == {
    "artifact-run": "1",
    "artifact-ids": "100,101,102,103,104",
    "reused": "true",
    "testpypi-verified": "false",
  }


@pytest.mark.parametrize(
  "overrides",
  [
    {"head_sha": "b" * 40},
    {"head_branch": "feature"},
    {"event": "pull_request"},
    {"status": "in_progress"},
    {"conclusion": "failure"},
  ],
)
def test_an_ineligible_nightly_cannot_supply_release_files(release, monkeypatch, overrides):
  github = GitHub()
  github.add("nightly.yml", 1, "nightly", **overrides)
  assert select(release, monkeypatch, github)["reused"] == "false"


def test_a_green_nightly_that_skipped_its_tests_does_not_qualify(release, monkeypatch):
  github = GitHub()
  github.add("nightly.yml", 1)
  assert select(release, monkeypatch, github)["reused"] == "false"


@pytest.mark.parametrize("damage", ["expired-files", "missing-files", "expired-record", "broken-record", "wrong-packages"])
def test_unusable_verified_files_fall_back_to_the_full_nightly(release, monkeypatch, damage):
  github = GitHub()
  github.add("release.yml", 2, "testpypi")
  github.add("nightly.yml", 1, "nightly")
  if damage == "expired-files":
    github.artifacts[2][0]["expired"] = True
  elif damage == "missing-files":
    github.artifacts[2].pop(0)
  elif damage == "expired-record":
    github.artifacts[2][-1]["expired"] = True
  elif damage == "broken-record":
    github.archives[290] = b"not a zip archive"
  else:
    with zipfile.ZipFile(io.BytesIO(github.archives[290])) as archive:
      record = json.loads(archive.read("validation.json"))
    record["packages"]["scaly"] = "0.0.1"
    github.set_record(290, record)
  result = select(release, monkeypatch, github)
  assert result["artifact-run"] == "1"
  assert result["testpypi-verified"] == "false"


def test_a_later_publication_failure_preserves_completed_testpypi_verification(release, monkeypatch):
  github = GitHub()
  github.add("release.yml", 1, "testpypi", conclusion="failure")
  assert select(release, monkeypatch, github)["testpypi-verified"] == "true"


def test_a_release_can_reuse_full_validation_before_testpypi_completed(release, monkeypatch):
  github = GitHub()
  github.add("release.yml", 1, "nightly", conclusion="failure")
  result = select(release, monkeypatch, github)
  assert result["artifact-run"] == "1"
  assert result["testpypi-verified"] == "false"


@pytest.mark.parametrize("kind", ["nightly", "testpypi"])
def test_record_binds_validation_to_current_artifact_ids(release, monkeypatch, kind):
  github = GitHub()
  github.add("release.yml", 1, kind)
  monkeypatch.setattr(release, "gh", github.gh)
  record = release.record_validation("PREDICT-EPFL/scaly", SHA, 1, 1, kind, PACKAGES)
  assert record == {
    "sha": SHA,
    "run_id": 1,
    "run_attempt": 1,
    "kind": kind,
    "artifacts": [100] if kind == "testpypi" else list(range(100, 105)),
    "packages": PACKAGES if kind == "testpypi" else {},
  }
  github.artifacts[1][0]["expired"] = True
  with pytest.raises(ValueError, match="artifacts"):
    release.record_validation("PREDICT-EPFL/scaly", SHA, 1, 1, kind, PACKAGES)


def test_selection_cli_writes_workflow_outputs_and_summary(release, monkeypatch, tmp_path):
  github = GitHub()
  github.add("release.yml", 1, "testpypi")
  monkeypatch.setattr(release, "gh", github.gh)
  for name, value in {
    "GITHUB_REPOSITORY": "PREDICT-EPFL/scaly",
    "GITHUB_SHA": SHA,
    "GITHUB_RUN_ID": "99",
    "GITHUB_OUTPUT": str(tmp_path / "output"),
    "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
    "PACKAGES": json.dumps([{"name": name, "version": version} for name, version in PACKAGES.items()]),
  }.items():
    monkeypatch.setenv(name, value)
  monkeypatch.setattr(sys, "argv", [str(SCRIPT), "select"])
  release.main()
  outputs = dict(line.split("=", 1) for line in Path(os.environ["GITHUB_OUTPUT"]).read_text().splitlines())
  assert outputs["artifact-run"] == "1" and outputs["testpypi-verified"] == "true"
  assert "https://github.com/PREDICT-EPFL/scaly/actions/runs/1" in (tmp_path / "summary").read_text()


@pytest.mark.parametrize(
  "change",
  [
    {"sha": "b" * 40},
    {"run_id": 9},
    {"run_attempt": 2},
    {"kind": "testpypi"},
    {"artifacts": [100, 100, 100, 100, 100]},
  ],
)
def test_a_record_must_describe_the_current_run_and_complete_artifact_set(release, monkeypatch, change):
  github = GitHub()
  github.add("nightly.yml", 1, "nightly")
  with zipfile.ZipFile(io.BytesIO(github.archives[190])) as archive:
    record = json.loads(archive.read("validation.json"))
  github.set_record(190, record | change)
  assert select(release, monkeypatch, github)["reused"] == "false"


def test_record_cli_writes_the_validation_file(release, monkeypatch, tmp_path):
  github = GitHub()
  github.add("nightly.yml", 1, "nightly")
  monkeypatch.setattr(release, "gh", github.gh)
  monkeypatch.chdir(tmp_path)
  for name, value in {"GITHUB_REPOSITORY": "PREDICT-EPFL/scaly", "GITHUB_SHA": SHA, "GITHUB_RUN_ID": "1", "GITHUB_RUN_ATTEMPT": "1"}.items():
    monkeypatch.setenv(name, value)
  monkeypatch.setattr(sys, "argv", [str(SCRIPT), "record", "nightly"])
  release.main()
  record = json.loads(Path("validation.json").read_text())
  assert record["sha"] == SHA and record["artifacts"] == list(range(100, 105))


def test_github_queries_use_the_cli_and_paginate(release, monkeypatch):
  calls = []

  def command(args):
    calls.append(args)
    return b'[{"items": [1]}, {"items": [2]}]'

  monkeypatch.setattr(release.subprocess, "check_output", command)
  assert release.listing("repos/PREDICT-EPFL/scaly/example", "items") == [1, 2]
  assert calls == [["gh", "api", "--paginate", "--slurp", "repos/PREDICT-EPFL/scaly/example"]]


@pytest.mark.parametrize("damage", [None, "different-bytes"])
def test_testpypi_checks_the_exact_bytes_of_every_release_file(release, monkeypatch, tmp_path, damage):
  import hashlib

  wheel = tmp_path / "scaly-0.1.0-py3-none-any.whl"
  wheel.write_bytes(b"the selected wheel")
  sdist = tmp_path / "scaly-0.1.0.tar.gz"
  sdist.write_bytes(b"the selected sdist")
  urls = [{"filename": p.name, "digests": {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}} for p in (wheel, sdist)]
  if damage == "different-bytes":
    urls[0]["digests"]["sha256"] = "0" * 64

  def urlopen(url, timeout):
    assert url == "https://test.pypi.org/pypi/scaly/0.1.0/json" and timeout == 30
    return io.BytesIO(json.dumps({"urls": urls}).encode())

  monkeypatch.setattr(release.urllib.request, "urlopen", urlopen)
  if damage:
    with pytest.raises(ValueError, match="TestPyPI.*differs"):
      release.check_testpypi({"scaly": "0.1.0"}, tmp_path)
  else:
    release.check_testpypi({"scaly": "0.1.0"}, tmp_path)
