"""Time the first call of a small JIT function in fresh processes on Windows, with a cold and a warm zig cache.

Adapted from `internal/notes/perf_2026_10_21_zig_cc_macos/first_compile.py`, without the system compiler row.
"""

import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

runs = 5


def child() -> None:
  import numpy as np

  import scaly as sc

  @sc.function(sc.arg("x", 2), sc.arg("u", 1), outputs=sc.arg("xnext"))
  def model(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    return x + 0.1 * sc.concat([x[1:], u])

  start = time.perf_counter()
  model(np.array([1.0, 2.0]), np.array([0.5]))
  print(f"{time.perf_counter() - start:.6f}")


def first_call(env: dict[str, str]) -> float:
  with tempfile.TemporaryDirectory() as scaly_cache:
    full = {**os.environ, **env, "SCALY_CACHE_DIR": scaly_cache}
    out = subprocess.run([sys.executable, __file__, "child"], env=full, check=True, capture_output=True, text=True)
  return float(out.stdout.split()[-1])


def main() -> None:
  base = {k: v for k, v in os.environ.items() if k != "SCALY_CC"}
  os.environ.clear()
  os.environ.update(base)
  results = {}
  cold = []
  for _ in range(runs):
    with tempfile.TemporaryDirectory() as zig_cache:
      cold.append(first_call({"ZIG_GLOBAL_CACHE_DIR": zig_cache}))
  results["zig cc, empty zig cache"] = cold
  with tempfile.TemporaryDirectory() as zig_cache:
    first_call({"ZIG_GLOBAL_CACHE_DIR": zig_cache})
    results["zig cc, warm zig cache"] = [first_call({"ZIG_GLOBAL_CACHE_DIR": zig_cache}) for _ in range(runs)]
  print("| First call | Mean (s) | Min (s) | Max (s) |")
  print("|---|---:|---:|---:|")
  for name, values in results.items():
    print(f"| {name} | {statistics.mean(values):.3f} | {min(values):.3f} | {max(values):.3f} |")


if __name__ == "__main__":
  child() if sys.argv[1:] == ["child"] else main()
