"""Classify final stored derivative output writes by their actual optimized addressing."""

import json
from pathlib import Path

from study import benchmark_functions, program_counts, render_c_module


def main():
  rows = []
  for family in ("chain", "race_cars", "npmpc", "unbumpercars"):
    for name, fun, pattern in benchmark_functions(family):
      print(name, flush=True)
      module = render_c_module(fun, lanes=1, dialect="c", cpu="generic")
      counts = program_counts(module.program, coverage=True)
      assert "count_error" not in counts, counts
      coverage = counts["output_coverage"]
      assert sum(sum(v.values()) for v in coverage.values()) == pattern.nnz
      assert all(v["unknown"] == 0 for v in coverage.values()), coverage
      rows.append({"name": name, "coverage": coverage})
  Path(__file__).with_name("coverage.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
  main()
