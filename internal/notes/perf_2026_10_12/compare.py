"""Compare a study's Scaly timings, code size and compile time against the published September 23 summary files."""

import csv
import io
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# The September 23 summary files as published, before this study replaced them.
BASELINE = "1c955a07"


def rows(path):
  with open(path) as f:
    return list(csv.DictReader(f))


def baseline(name):
  text = subprocess.run(["git", "show", f"{BASELINE}:docs/assets/benchmarks/{name}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
  return list(csv.DictReader(io.StringIO(text)))


def pct(new, old):
  return 100 * (new / old - 1)


def cv(value):
  return f"{100 * float(value):.2f}" if value else ""


def main(study):
  old = {(r["workload"], r["size"]): r for r in baseline("sweep.summary.csv") if r["backend"] == "scaly"}
  print("| Problem | Size | Sep 23, µs | CV, % | Now, µs | CV, % | Change, % | Executable bytes, % | Kernel compile, % |")
  print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
  for path in sorted(study.glob("sweep/*/*.summary.csv")):
    for r in rows(path):
      if r["backend"] != "scaly" or not r["runtime_ns_mean"]:
        continue
      o = old.get((r["workload"], r["size"]))
      if o is None or not o["runtime_ns_mean"]:
        continue
      a, b = float(o["runtime_ns_mean"]) / 1e3, float(r["runtime_ns_mean"]) / 1e3
      size = pct(float(r["executable_bytes_mean"]), float(o["executable_bytes_mean"]))
      compile_ = pct(float(r["kernel_compile_ms_mean"]), float(o["kernel_compile_ms_mean"]))
      print(
        f"| {r['workload']} | {r['size']} | {a:.3f} | {cv(o['runtime_ns_cv'])} | {b:.3f} | {cv(r['runtime_ns_cv'])} | {pct(b, a):+.1f} "
        f"| {size:+.1f} | {compile_:+.1f} |"
      )

  old = {(r["problem"], r["solver"], r["oracle"]): r for r in baseline("closed_loop.summary.csv")}
  print()
  print("| Problem | Solver | Provider | Sep 23 total, ms | Now, ms | CV, % | Change, % | Sep 23 FE, ms | Now, ms | Change, % |")
  print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
  for r in rows(study / "closed-loop" / "closed_loop.summary.csv"):
    o = old.get((r["problem"], r["solver"], r["oracle"]))
    if o is None:
      continue
    t0, t1 = float(o["solver_time_ms_mean"]), float(r["solver_time_ms_mean"])
    f0, f1 = float(o["fe_time_ms_mean"]), float(r["fe_time_ms_mean"])
    print(
      f"| {r['problem']} | {r['solver']} | {r['oracle']} | {t0:.3f} | {t1:.3f} | {cv(r['solver_time_ms_cv'])} | {pct(t1, t0):+.1f} "
      f"| {f0:.3f} | {f1:.3f} | {pct(f1, f0):+.1f} |"
    )


if __name__ == "__main__":
  main(Path(sys.argv[1]).resolve())
