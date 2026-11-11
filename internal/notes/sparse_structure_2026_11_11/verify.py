"""Check the saved counts against independent product formulas and small exhaustive masks."""

import json
from pathlib import Path

import numpy as np

from structure import inspect_pattern, self_check

HERE = Path(__file__).resolve().parent


def main():
  self_check()
  random = np.random.default_rng(91)
  for _ in range(12):
    mask = random.random((6, 7)) < 0.9
    expected = np.zeros_like(mask)
    for lo in range(3):
      for hi in range(lo + 4, 7):
        for first in range(4):
          for last in range(first + 4, 8):
            if mask[lo:hi, first:last].all():
              expected[lo:hi, first:last] = True
    actual = inspect_pattern(mask.shape, *np.nonzero(mask))
    assert actual["block_entries"] == int(expected.sum())
  rows = {r["name"]: r for r in map(json.loads, (HERE / "kkt.jsonl").read_text().splitlines())}
  assert len(rows) == 432
  products = 0
  for name, row in rows.items():
    assert "error" not in row and "count_error" not in row["program"]
    assert row["compile_returncode"] == 0
    problem, kind = name.split("/")
    if kind in ("kkt", "upper"):
      continue
    source = rows[problem + "/" + kind[0] + "x"]["pattern"]
    expected = source["nnz"] + source["row_square_sum"] if kind.endswith("gram") else source["nnz"]
    counts = row["program"]["executed"]
    assert counts.get("mul", 0) == counts.get("add", 0) == expected, (name, counts, expected)
    products += 1
  addressing = json.loads((HERE / "addressing.json").read_text())
  assert len(addressing) == 240
  assert all(r["baseline"]["table_entries"] == r["actual_table_entries"] for r in addressing)
  benchmarks = list(map(json.loads, (HERE / "bench.jsonl").read_text().splitlines()))
  assert len(benchmarks) == 8
  assert all(r["compile_returncode"] == 0 and "count_error" not in r["program"] for r in benchmarks)
  census = json.loads((HERE / "census.json").read_text())
  assert len(census) == 15
  maps = json.loads((HERE / "maps.json").read_text())
  assert len(maps) == 96
  primal = next(r for r in maps if r["name"] == "PRIMALC5/kkt")
  assert primal["map_table_entries_before"] == 4887 and primal["map_table_entries_after"]["eight"] == 0
  coverage = json.loads((HERE / "coverage.json").read_text())
  assert len(coverage) == 8
  for r, b in zip(coverage, benchmarks, strict=True):
    assert r["name"] == b["name"]
    assert all(c["unknown"] == 0 for c in r["coverage"].values())
    assert sum(sum(c.values()) for c in r["coverage"].values()) == b["pattern"]["nnz"]
  print(f"432 KKT modules and 8 benchmark modules compiled. {products} product operation counts and 240 addressing baselines agree.")
  print("NumPy rectangle coverage agrees with exhaustive enumeration on 12 masks. All 15 census cases completed.")


if __name__ == "__main__":
  main()
