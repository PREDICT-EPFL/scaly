"""Temporary diagnostic for #252: what a restored zig cache rebuilds on Windows. Removed before review."""

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
import scaly as sc  # noqa: E402
from benchmarks.harness.gbench import compiler  # noqa: E402

root = Path(os.environ["ZIG_GLOBAL_CACHE_DIR"])


def snapshot() -> dict[str, float]:
  return {str(p.relative_to(root)): p.stat().st_mtime for p in root.rglob("*") if p.is_file()} if root.exists() else {}


def report(label: str, start: float, before: dict[str, float]) -> None:
  elapsed = time.perf_counter() - start
  after = snapshot()
  new = [name for name in after if name not in before]
  changed = [name for name in after if name in before and after[name] != before[name]]
  print(f"== {label}: {elapsed:.1f} s, {len(new)} new files, {len(changed)} rewritten")
  for name in sorted(new):
    if Path(name).parts[0] == "o":
      print("  new output:", name)
    elif Path(name).parts[0] == "h":
      strings = [m.decode() for m in re.findall(rb"[\x20-\x7e]{6,}", (root / name).read_bytes())]
      print(f"  new manifest {name}: {strings[:2]}")


files = snapshot()
print(f"{root}: {len(files)} files before")
before, start = snapshot(), time.perf_counter()
print(sc.function(sc.arg("x", 2), outputs=sc.arg("y"), name="probe")(lambda x: x.sum())(np.ones(2)))
report("JIT call", start, before)
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
  Path(tmp, "warm.cpp").write_text('#include <string>\nint main() { return std::string("warm").size() == 4 ? 0 : 1; }\n')
  before, start = snapshot(), time.perf_counter()
  subprocess.run([*compiler(), "warm.cpp", "-o", "warm.exe", "-lshlwapi"], cwd=tmp, check=True)
  report("C++ link", start, before)
