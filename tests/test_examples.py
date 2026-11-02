"""Run the standalone examples and the README's opening Python example."""

import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOLVERS = {"multiple_shooting.py": "ipopt", "sparse_constant_mpc.py": "piqp", "README.md": "ipopt"}
EXAMPLES = [*sorted((ROOT / "examples").glob("*.py")), ROOT / "README.md"]


# TEMPORARY (PR #251): before the timeout, list the compiler processes on Windows, to tell a busy zig from a blocked one.
_LIST_COMPILERS = """
import subprocess as _sp, sys as _sys, threading as _th
def _list():
  ps = "Get-Process | Where-Object { $_.ProcessName -match '^(zig|clang|lld|ld)' } | Format-Table Id,CPU,StartTime,ProcessName -AutoSize | Out-String -Width 200; Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(zig|clang|lld|ld)' } | Format-List ProcessId,ParentProcessId,CommandLine | Out-String -Width 4000"
  print(_sp.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout, file=_sys.stderr, flush=True)
if _sys.platform == "win32":
  _t = _th.Timer(100, _list)
  _t.daemon = True
  _t.start()
"""


@pytest.mark.parametrize(
  "path",
  [pytest.param(path, id=path.name, marks=[pytest.mark.solver(SOLVERS[path.name])] if path.name in SOLVERS else []) for path in EXAMPLES],
)
def test_example(path: Path, tmp_path: Path) -> None:
  source = path.read_text()
  if path.name == "README.md":
    fence = re.search(r"^```python\s*\n(.*?)^```", source, re.MULTILINE | re.DOTALL)
    assert fence is not None, "README.md needs a Python example"
    source = fence[1]
  script = tmp_path / (path.stem + ".py")
  # Shortly before the timeout, print where every thread is, so a hang shows its place.
  script.write_text(f"import faulthandler\nfaulthandler.dump_traceback_later(110, exit=True)\n{_LIST_COMPILERS}\n{source}")
  env = {**os.environ, "SCALY_VIZ_DIR": str(tmp_path / "viz")}
  try:
    result = subprocess.run([sys.executable, "-u", str(script)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
  except subprocess.TimeoutExpired as exc:
    pytest.fail(f"{path.name} timed out:\n{exc.stdout}\n{exc.stderr}")
  assert result.returncode == 0, f"{path.name}:\n{result.stdout}\n{result.stderr}"
