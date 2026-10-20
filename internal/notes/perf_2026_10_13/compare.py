"""Audit the post-179 study, compare the October 12 summaries, and prepare its published data."""

import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASELINE = "d9755969"
GRID = {"race_cars": (9, 5), "npmpc": (6, 8), "unbumpercars": (5, 6), "chain": (3, 5)}


def read_csv(path):
  with path.open() as stream:
    return list(csv.DictReader(stream))


def baseline(name):
  result = subprocess.run(["git", "show", f"{BASELINE}:docs/assets/benchmarks/{name}"], cwd=ROOT, capture_output=True, text=True, check=True)
  return list(csv.DictReader(io.StringIO(result.stdout)))


def write_csv(path, rows):
  with path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def change(new, old):
  return 100 * (float(new) / float(old) - 1)


def main(study, publish):
  manifest = json.loads((study / "study.json").read_text())
  assert manifest["git_commit"].startswith("38bd5896") and not manifest["git_dirty"]
  assert len(manifest["commands"]) == 12 and all(c["returncode"] == 0 for c in manifest["commands"])
  assert manifest["vector_libm"] == "glibc" and manifest["cpu_settings"]["boost_enabled"] is False
  assert all(p["scaling_governor"] == "performance" for p in manifest["cpu_settings"]["policies"].values())
  raw = []
  sweep = []
  for problem, (sizes, backends) in GRID.items():
    rows = read_csv(study / "sweep" / problem / f"{problem}.csv")
    assert len(rows) == sizes * backends
    assert len({(r["size"], r["backend"]) for r in rows}) == len(rows)
    assert all(r["repetition"] == "1" for r in rows)
    assert all(
      (r["compile_status"], r["runtime_status"]) in {("ok", "ok"), ("timeout", "skipped"), ("skipped_after_failure", "skipped")} for r in rows
    )
    assert all(r["runtime_status"] == "ok" for r in rows if r["backend"] == "scaly")
    raw.extend(rows)
    sweep.extend(read_csv(study / "sweep" / problem / f"{problem}.summary.csv"))
  counts = Counter(r["compile_status"] for r in raw)
  assert len(raw) == 138 and sum(r["backend"] == "scaly" for r in raw) == 23
  summary = json.loads((study / "closed-loop" / "closed_loop.summary.json").read_text())
  assert len(summary["runs"]) == len(summary["groups"]) == 15
  assert len(summary["oracle_comparisons"]) == 7
  assert all(g["all_steps_successful"] and g["processes"] == 1 for g in summary["groups"])
  expected_steps = {"chain": 90, "npmpc": 100, "race_cars": 1367, "unbumpercars": 200}
  assert all(r["steps"] == r["successful_steps"] == expected_steps[r["problem"]] for r in summary["runs"])
  pairs = []
  for pair in summary["oracle_comparisons"]:
    assert pair["same_steps"]
    assert all(pair[f"{key}_mismatches"] == 0 for key in ("n_eval_f", "n_eval_g", "n_eval_grad_f", "n_eval_jac_g", "n_eval_h"))
    for metric in ("solver_time_ms", "fe_time_ms"):
      ratios = np.asarray(pair[f"{metric}_speedups"])
      assert len(ratios) == expected_steps[pair["problem"]] and np.all(np.isfinite(ratios) & (ratios > 0))
      p10, median, p90 = np.percentile(ratios, [10, 50, 90])
      pairs.append(
        dict(
          problem=pair["problem"],
          solver=pair["solver"],
          metric=metric,
          key=f"{pair['problem']}:{pair['solver']}:{metric}",
          steps=len(ratios),
          median=median,
          p10=p10,
          p90=p90,
          iteration_mismatches=pair["iterations_mismatches"],
        )
      )
  lines = [
    "# October 13 study audit and comparison",
    "",
    f"Revision: `{manifest['git_commit']}`. Duration: {sum(c['seconds'] for c in manifest['commands']) / 60:.1f} minutes.",
    "",
    f"138 sweep attempts: {counts['ok']} timed and correctness-gated, {counts['timeout']} compile timeouts, {counts['skipped_after_failure']} skipped after failure. No other failures. All 23 Scaly cells timed. All 15 episodes and 6,938 solver steps succeeded.",
    "",
    "The October 12 study had 690 attempts: 515 timed, 17 timeouts and 158 skipped after failure. Five processes versus one means the raw counts are not directly comparable. The partially completed mapped-SX cells near the compile limit account for the changed coverage.",
    "",
    "## Scaly Hessians against October 12",
    "",
    "| Problem | Size | October 12, µs | Previous CV, % | October 13, µs | Runtime change, % | Executable bytes change, % | Kernel compile change, % |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
  ]
  old = {(r["workload"], r["size"], r["backend"]): r for r in baseline("sweep.summary.csv")}
  for row in sweep:
    if row["backend"] != "scaly":
      continue
    before = old[(row["workload"], row["size"], "scaly")]
    lines.append(
      f"| {row['workload']} | {row['size']} | {float(before['runtime_ns_mean']) / 1000:.3f} | {float(before['runtime_ns_cv']) * 100:.2f} | {float(row['runtime_ns_mean']) / 1000:.3f} | {change(row['runtime_ns_mean'], before['runtime_ns_mean']):+.1f} | {change(row['executable_bytes_mean'], before['executable_bytes_mean']):+.1f} | {change(row['kernel_compile_ms_mean'], before['kernel_compile_ms_mean']):+.1f} |"
    )
  lines += [
    "",
    "## Closed-loop means against October 12",
    "",
    "| Problem | Solver | Provider | October 12 total, ms | October 13 total, ms | Total change, % | October 12 FE, ms | October 13 FE, ms | FE change, % |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|",
  ]
  old_loop = {(r["problem"], r["solver"], r["oracle"]): r for r in baseline("closed_loop.summary.csv")}
  for group in summary["groups"]:
    before = old_loop[(group["problem"], group["solver"], group["oracle"])]
    total, fe = group["solver_time_ms"]["mean"], group["fe_time_ms"]["mean"]
    lines.append(
      f"| {group['problem']} | {group['solver']} | {group['oracle']} | {float(before['solver_time_ms_mean']):.3f} | {total:.3f} | {change(total, before['solver_time_ms_mean']):+.1f} | {float(before['fe_time_ms_mean']):.3f} | {fe:.3f} | {change(fe, before['fe_time_ms_mean']):+.1f} |"
    )
  lines += ["", "## Fastest completed CasADi Hessians", "", "| Problem | Size | Encoding | CasADi / Scaly |", "|---|---:|---|---:|"]
  for problem in GRID:
    sizes = sorted({int(r["size"]) for r in sweep if r["workload"] == problem})
    for size in sizes:
      cell = [r for r in sweep if r["workload"] == problem and int(r["size"]) == size and r["runtime_ns_mean"]]
      scaly = next(r for r in cell if r["backend"] == "scaly")
      casadi = [r for r in cell if r["backend"].startswith("casadi_")]
      if casadi:
        fastest = min(casadi, key=lambda r: float(r["runtime_ns_mean"]))
        lines.append(f"| {problem} | {size} | {fastest['backend']} | {float(fastest['runtime_ns_mean']) / float(scaly['runtime_ns_mean']):.3f} |")
  lines += ["", "## Current full report", "", (study / "report.md").read_text()]
  (Path(__file__).parent / "analysis.md").write_text("\n".join(lines) + "\n")
  if publish:
    target = ROOT / "docs/assets/benchmarks"
    write_csv(target / "sweep.summary.csv", sweep)
    (target / "closed_loop.summary.csv").write_text((study / "closed-loop/closed_loop.summary.csv").read_text())
    write_csv(target / "closed_loop_speedup.summary.csv", pairs)
  print(f"Audit passed: {dict(counts)}, 15 successful episodes, seven paired comparisons. Analysis: {Path(__file__).parent / 'analysis.md'}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("study", type=Path)
  parser.add_argument("--publish", action="store_true")
  args = parser.parse_args()
  main(args.study.resolve(), args.publish)
