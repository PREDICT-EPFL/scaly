"""Retime the retained binaries of two unchanged cells without changing the published study."""

# ruff: noqa: E402 -- direct execution must add the repository root before package imports

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmarks.harness.provenance import require_headline_settings
from internal.notes.perf_2026_10_14.publish import write_csv


def main(studies: list[Path], out: Path) -> None:
  assert not (Path.home() / ".scaly-measuring").exists()
  require_headline_settings(False)
  out.mkdir(parents=True, exist_ok=False)
  rows = []
  for study in studies:
    for cell in ("casadi_mx_N25", "scaly_N200"):
      directory = study.resolve() / "sweep/npmpc/repeat_1/npmpc" / cell
      for repetition in range(1, 4):
        result = subprocess.run(
          [str(directory / "benchmark"), "--benchmark_min_time=0.5s", "--benchmark_format=json"],
          cwd=directory,
          check=True,
          capture_output=True,
          text=True,
          timeout=60,
        )
        name = f"{study.name}-{cell}-{repetition}"
        (out / f"{name}.json").write_text(result.stdout)
        (out / f"{name}.log").write_text(result.stderr)
        assert "correctness ok" in result.stdout + result.stderr
        values = json.loads(result.stdout[result.stdout.index("{") :])["benchmarks"]
        assert len(values) == 1 and values[0]["time_unit"] == "ns"
        rows.append(dict(study=study.name, cell=cell, repetition=repetition, runtime_ns=values[0]["real_time"]))
  require_headline_settings(False)
  write_csv(out / "summary.csv", rows)
  for row in rows:
    print(row)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("studies", nargs=2, type=Path)
  parser.add_argument("--out", required=True, type=Path)
  args = parser.parse_args()
  main(args.studies, args.out.resolve())
