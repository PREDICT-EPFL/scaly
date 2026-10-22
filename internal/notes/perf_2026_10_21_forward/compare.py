"""Summarize retained native measurements, including repeats and rejected candidates."""

import csv
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
WORKLOADS = ("chain_jac", "npmpc_jac", "chain", "unbumpercars")
EXCLUDED = {"candidate7-repeat-3"}


def samples(phase: str, workload: str) -> list[dict[str, str]]:
  rows = []
  for directory in sorted((ROOT / "raw").glob(f"{phase}*")):
    if directory.name in EXCLUDED or directory.name != phase and not directory.name.startswith(f"{phase}-repeat-"):
      continue
    path = directory / f"{workload}.csv"
    if not path.exists():
      continue
    data = list(csv.DictReader(path.open()))
    assert len(data) == 1, path
    row = data[0]
    assert row["compile_status"] == "ok" and row["runtime_status"] == "ok", path
    rows.append(row)
  return rows


def metric(rows: list[dict[str, str]], key: str, scale: float = 1) -> str:
  values = [float(row[key]) * scale for row in rows]
  avg = mean(values)
  cv = 100 * stdev(values) / avg if len(values) > 1 else 0
  return f"{avg:.3f} ({cv:.1f}%)"


def main() -> None:
  lines = [
    "| Workload | Revision | Processes | C bytes | Generation ms (CV) | Native µs (CV) | Workspace doubles | Runtime vs main |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
  ]
  for workload in WORKLOADS:
    control = samples("main", workload)
    for phase in ("main", "rebased", "fixed"):
      rows = samples(phase, workload)
      if not rows:
        continue
      assert len({row["source_bytes"] for row in rows}) == 1, (phase, workload)
      assert len({row["workspace"] for row in rows}) == 1, (phase, workload)
      delta = (
        "" if phase == "main" else f"{(mean(float(r['runtime_ns']) for r in rows) / mean(float(r['runtime_ns']) for r in control) - 1) * 100:+.1f}%"
      )
      lines.append(
        f"| {workload} | {phase} | {len(rows)} | {rows[0]['source_bytes']} | {metric(rows, 'codegen_ms')} | "
        f"{metric(rows, 'runtime_ns', 0.001)} | {rows[0]['workspace']} | {delta} |"
      )
  lines.extend(["", "| Candidate | chain_jac µs | npmpc_jac µs | chain µs | unbumpercars µs |", "|---|---:|---:|---:|---:|"])
  for phase in (f"candidate{i}" for i in range(1, 9)):
    cells = [samples(phase, workload) for workload in WORKLOADS]
    if all(cells):
      lines.append(f"| {phase} | " + " | ".join(metric(rows, "runtime_ns", 0.001) for rows in cells) + " |")
  result = "\n".join(lines) + "\n"
  (ROOT / "comparison.md").write_text(result)
  print(result)


if __name__ == "__main__":
  main()
