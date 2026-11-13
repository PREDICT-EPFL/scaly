"""Repeat noisy saved kernels and unchanged controls without rebuilding or replacing earlier data."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess


def settings():
  root = Path("/sys/devices/system/cpu")
  governors = [p.read_text().strip() for p in (root / "cpufreq").glob("policy*/scaling_governor")]
  assert governors and set(governors) == {"performance"}
  assert (root / "cpufreq/boost").read_text().strip() == "0"


def main(root):
  root = root.resolve()
  target = root / "retiming-noise-check"
  target.mkdir(exist_ok=False)
  with (root / "retiming.csv").open() as fp:
    original = list(csv.DictReader(fp))
  grouped = defaultdict(list)
  for row in original:
    grouped[row["workload"], row["size"], row["backend"], row["side"]].append(float(row["runtime_ns"]))
  selected = {key[:3] for key, values in grouped.items() if key[2] != "scaly" or statistics.stdev(values) / statistics.mean(values) > 0.05}
  print(f"Retiming {len(selected)} cells: all unchanged controls and kernels with over 5% timing variation.", flush=True)
  measuring = Path.home() / ".scaly-measuring"
  with measuring.open("x") as lock:
    lock.write(f"post-run-2026-11-12 noise check pid:{os.getpid()} out:{target}\n")
  try:
    with (root / "retiming-noise-check.csv").open("w") as fp:
      writer = csv.DictWriter(fp, fieldnames=list(original[0]))
      writer.writeheader()
      for workload, size, backend in sorted(selected):
        category, builds = ("sweep", 3) if backend == "scaly" else ("controls", 1)
        axis = "M" if workload.startswith("chain") else "W" if workload.startswith("npmpc_decoder") else "C" if workload == "unbumpercars" else "N"
        for build in range(1, builds + 1):
          for repeat in range(5, 9):
            for side in ("before", "after") if repeat % 2 else ("after", "before"):
              settings()
              directory = root / side / category / workload / f"repeat_{build}" / workload / f"{backend}_{axis}{size}"
              command = [str(directory / "benchmark"), "--benchmark_min_time=0.5s", "--benchmark_format=json"]
              result = subprocess.run(command, cwd=directory, capture_output=True, text=True, check=True, timeout=60)
              stem = target / f"{workload}_{size}_{backend}_{build}_{repeat}_{side}"
              stem.with_suffix(".json").write_text(result.stdout)
              stem.with_suffix(".log").write_text(result.stderr)
              assert "correctness ok" in result.stdout + result.stderr
              sample = json.loads(result.stdout[result.stdout.index("{") :])["benchmarks"][0]
              assert sample["time_unit"] == "ns"
              writer.writerow(
                dict(workload=workload, size=size, backend=backend, build=build, repetition=repeat, side=side, runtime_ns=sample["real_time"])
              )
              fp.flush()
              settings()
        print(f"Completed {workload} {size} {backend}", flush=True)
    (target / "COMPLETE").write_text(datetime.now(timezone.utc).isoformat() + "\n")
  finally:
    measuring.unlink()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("root", type=Path)
  main(parser.parse_args().root)
