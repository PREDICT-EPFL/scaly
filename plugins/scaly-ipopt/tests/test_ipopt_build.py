"""OpenBLAS's archive writers run one at a time, so parallel sub-makes cannot corrupt its static library."""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def test_archive_writers_share_one_lock(tmp_path):
  writer = tmp_path / "writer.py"
  writer.write_text(
    "from pathlib import Path\nimport sys,time\nroot=Path(sys.argv[1])\n"
    "guard=root/'writing'\nwith guard.open('x'):\n time.sleep(0.05)\n"
    " with (root/'calls').open('a') as out: out.write('done\\n')\nguard.unlink()\n"
  )
  command = [
    sys.executable,
    str(Path(__file__).parents[1] / "native_build.py"),
    str(tmp_path / "archive.lock"),
    sys.executable,
    str(writer),
    str(tmp_path),
  ]
  with ThreadPoolExecutor(max_workers=6) as pool:
    results = list(pool.map(lambda _: subprocess.run(command, capture_output=True, text=True), range(6)))
  assert all(result.returncode == 0 for result in results), [result.stderr for result in results]
  assert (tmp_path / "calls").read_text().splitlines() == ["done"] * 6
