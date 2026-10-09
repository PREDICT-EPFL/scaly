"""Compare issue 19's retained native sweep records on la015."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path


def cells(revision: str) -> dict[tuple[str, int], list[dict[str, str]]]:
  result = {}
  for path in sorted((Path(__file__).parent / "data" / revision).glob("sweep/*/*.csv")):
    if path.name.endswith(".summary.csv"):
      continue
    with path.open() as stream:
      rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    assert all(row["compile_status"] == "ok" and row["runtime_status"] == "ok" for row in rows)
    result[rows[0]["workload"], int(rows[0]["size"])] = rows
  return result


def mean_cv(rows: list[dict[str, str]], field: str) -> tuple[float, float]:
  values = [float(row[field]) for row in rows]
  mean = statistics.mean(values)
  return mean, statistics.stdev(values) / mean if mean else 0.0


def main() -> None:
  before, after = cells("before"), cells("after")
  assert before.keys() == after.keys()
  lines = [
    "| Kernel | C bytes before → after | Generation ms before → after (CV) | Native µs before → after (CV) | Workspace doubles before → after |",
    "| --- | ---: | ---: | ---: | ---: |",
  ]
  for key in before:
    a, b = before[key], after[key]
    source = [mean_cv(rows, "source_bytes")[0] for rows in (a, b)]
    generation = [mean_cv(rows, "codegen_ms") for rows in (a, b)]
    native = [mean_cv(rows, "runtime_ns") for rows in (a, b)]
    workspace = [mean_cv(rows, "workspace")[0] for rows in (a, b)]
    gen_text = " → ".join(f"{mean:.1f} ({cv:.1%})" for mean, cv in generation)
    native_text = " → ".join(f"{mean / 1000:.3f} ({cv:.1%})" for mean, cv in native)
    lines.append(
      f"| {key[0]} {key[1]} | {source[0]:,.0f} → {source[1]:,.0f} | {gen_text} | {native_text} | {workspace[0]:,.0f} → {workspace[1]:,.0f} |"
    )
  (Path(__file__).parent / "comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
  main()
