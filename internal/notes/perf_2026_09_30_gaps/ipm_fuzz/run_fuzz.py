"""C-226 (Tier 7): ``fuzz.py`` over every structure of ``bases.txt`` on one source tree, six at a time.

  uv run python internal/notes/perf_2026_09_30_gaps/ipm_fuzz/run_fuzz.py <dir holding scaly/> <out dir>

Each structure's solver is compiled once and run on its value sets (the base problem, then scalings
of variables, rows and cost, perturbations, a zeroed or scaled quadratic term, shifted right-hand
sides, pinched boxes). ``cmpfuzz.py <out of one tree> <out of another>`` compares two trees'
results: status, iteration count and a hash of the solution per instance. ``repro_worse.py BASE K``
runs one instance with its trace. ``results/c226_fuzz.txt`` holds the comparisons of the tree
before C-226 with C-226 as first written and as it stands.
"""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def main() -> None:
  tree, out = sys.argv[1], sys.argv[2]
  cells = [line.split() for line in (HERE / "bases.txt").read_text().splitlines()]

  def run(cell: list[str]) -> str:
    base, count, name = cell
    target = f"{out}/{name}"
    if os.path.exists(target):
      return f"{base} cached"
    env = {**os.environ, "PYTHONPATH": tree, "SCALY_CACHE_DIR": f"{out}/cache"}
    done = subprocess.run(
      ["uv", "run", "--no-sync", "python", "-W", "ignore", str(HERE / "fuzz.py"), base, count, target], capture_output=True, text=True, env=env, cwd=ROOT
    )
    return f"{base} " + ("FAILED " + done.stderr.strip().splitlines()[-1][-200:] if done.returncode else "ok")

  with ThreadPoolExecutor(6) as pool:
    for line in pool.map(run, cells):
      print(line, flush=True)


if __name__ == "__main__":
  main()
