"""The ``method`` marker: a built-in method is available, a missing one or a malformed name is not,
with the reason the skip reports; a missing method skips its test, and under
``SCALY_REQUIRE_METHODS=1`` fails the run instead."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scaly.testing.plugin import REQUIRE, method_available


def test_what_is_available() -> None:
  assert method_available("opt.ipm") == (True, "")
  assert method_available("ocp.ilqr") == (True, "")
  ok, reason = method_available("opt.nonexistent")
  assert not ok and reason == "method opt.nonexistent is not installed"
  ok, reason = method_available("nowhere.thing")
  assert not ok and reason == "no method domain scaly.nowhere"
  ok, reason = method_available("ipm")
  assert not ok and "domain.name" in reason


def _run(tmp_path: Path, require: bool) -> subprocess.CompletedProcess[str]:
  (tmp_path / "test_needs.py").write_text(
    'import pytest\n\n\n@pytest.mark.method("opt.nonexistent")\ndef test_needs():\n  pass\n\n\n@pytest.mark.method("opt.ipm")\ndef test_has():\n  pass\n'
  )
  env = {**os.environ, REQUIRE: "1" if require else "0"}
  argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:xdist", "-rs", "test_needs.py"]
  return subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, check=False)


@pytest.mark.parametrize("require", [False, True])
def test_a_missing_method_skips_or_fails_the_run(tmp_path: Path, require: bool) -> None:
  proc = _run(tmp_path, require)
  if require:
    assert proc.returncode == pytest.ExitCode.USAGE_ERROR, proc.stdout + proc.stderr
    assert f"{REQUIRE}=1 but not available: opt.nonexistent" in proc.stderr
  else:
    assert proc.returncode == pytest.ExitCode.OK, proc.stdout + proc.stderr
    assert "1 passed, 1 skipped" in proc.stdout and "method opt.nonexistent is not installed" in proc.stdout
