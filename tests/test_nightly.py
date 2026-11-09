"""Nightly Python coverage follows package metadata, and issue labels follow failed jobs."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "nightly.py"


def test_python_matrix_follows_classifiers(tmp_path):
  (tmp_path / "pyproject.toml").write_text(
    '[project]\nclassifiers = ["Programming Language :: Python :: 3", '
    '"Programming Language :: Python :: 3.12", "Programming Language :: Python :: 3.15", "Programming Language :: C"]\n'
  )
  result = subprocess.run([sys.executable, str(SCRIPT), "python"], cwd=tmp_path, check=True, capture_output=True, text=True)
  assert json.loads(result.stdout) == [
    {"platform": "linux-x86_64", "runner": "ubuntu-latest", "python": "3.12", "label": "3.12"},
    {"platform": "linux-x86_64", "runner": "ubuntu-latest", "python": "3.15", "label": "3.15"},
  ]


@pytest.mark.parametrize(
  "failed, benchmark",
  [
    ("toolchain / ubuntu-24.04 clang-18\npython / 3.14\n", False),
    ("benchmarks / linux-arm64\npython / 3.12\n", True),
  ],
)
def test_failure_labels_follow_the_benchmark_group(failed, benchmark):
  result = subprocess.run([sys.executable, str(SCRIPT), "labels"], input=failed, check=True, capture_output=True, text=True)
  labels = set(result.stdout.strip().split(","))
  assert labels == {"source:maintainer", "kind:bug", "area:tooling", "ci:nightly-failing"} | ({"area:benchmarks"} if benchmark else set())
