"""Audit the paired lowering-fusion measurements and print their comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

CELLS = (("chain", 5), ("race_cars", 5), ("npmpc_jac", 100), ("npmpc", 100))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("study", type=Path)
  args = parser.parse_args()
  rows = json.loads((args.study / "results.json").read_text())
  manifest = json.loads((args.study / "manifest.json").read_text())
  print("| Kernel | C lines | C bytes | Generation, ms | Native runtime, us | Runtime change | Runtime CV, % | Generation CV, % |")
  print("|---|---:|---:|---:|---:|---:|---:|---:|")
  for workload, size in CELLS:
    metrics = {}
    samples = None
    for revision in ("main", "pr"):
      group = [r for r in rows if not r.get("excluded") and (r["workload"], r["size"], r["revision"]) == (workload, size, revision)]
      assert len(group) == manifest["repetitions"] == 3
      for row in group:
        assert row["compile_status"] == row["runtime_status"] == "ok"
        path = row.get("cell_path", f"repeat_{row['repetition']}/{revision}/{workload}_{size}")
        artifact = args.study / path
        provenance = json.loads((artifact / "provenance.json").read_text())
        assert not provenance["git_dirty"]
        assert provenance["compiler_revision"] == manifest["base" if revision == "main" else "head"]
        assert provenance["cpu_settings"]["boost_enabled"] is False
        assert provenance["cpu_settings"]["affinity"] == [2]
        assert all(p["scaling_governor"] == "performance" for p in provenance["cpu_settings"]["policies"].values())
        assert provenance["vector_libm"] == "glibc"
        assert samples is None or samples == provenance["sample_hashes"]
        samples = provenance["sample_hashes"]
        assert "correctness ok:" in (artifact / "benchmark.json").read_text()
      metrics[revision] = {key: [float(r[key]) for r in group] for key in ("source_lines", "source_bytes", "workspace", "codegen_ms", "runtime_ns")}
      for key in ("source_lines", "source_bytes", "workspace"):
        assert len(set(metrics[revision][key])) == 1

    def pair(key: str, scale: float = 1, digits: int = 0) -> str:
      return " → ".join(f"{statistics.median(metrics[r][key]) / scale:,.{digits}f}" for r in ("main", "pr"))

    def cv_pair(key: str) -> str:
      return " / ".join(f"{100 * statistics.stdev(metrics[r][key]) / statistics.mean(metrics[r][key]):.2f}" for r in ("main", "pr"))

    ratio = statistics.median(metrics["pr"]["runtime_ns"]) / statistics.median(metrics["main"]["runtime_ns"])
    if workload.startswith("npmpc"):
      assert ratio <= 1.05, f"{workload} exceeds the roadmap's 5% runtime limit"
    print(
      f"| {workload} {size} | {pair('source_lines')} | {pair('source_bytes')} | {pair('codegen_ms', digits=1)} | "
      f"{pair('runtime_ns', scale=1000, digits=3)} | {(ratio - 1) * 100:+.2f}% | {cv_pair('runtime_ns')} | {cv_pair('codegen_ms')} |"
    )
  excluded = sum(bool(row.get("excluded")) for row in rows)
  print(f"\nAccepted {len(rows) - excluded} rows; retained {excluded} excluded rows. One further timing attempt failed the quiet-machine check.")


if __name__ == "__main__":
  main()
