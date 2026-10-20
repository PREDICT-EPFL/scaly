"""Audit the October 20 study, publish its data and compare every cell with October 15."""

# ruff: noqa: E402 -- direct execution must add the repository root before package imports

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from benchmarks.harness import CLOSED_LOOP_PAIRS
from benchmarks.harness.report import COUNTS, _episode
from benchmarks.harness.study import SWEEP_GRID
from benchmarks.harness.sweep import DEFAULT_BACKENDS
from internal.notes.perf_2026_10_14.publish import read_csv, write_csv

STEPS = {"chain": 90, "npmpc": 100, "race_cars": 1367, "unbumpercars": 200}
SAMPLE_MS = {"chain": 200, "npmpc": 20, "race_cars": 50, "unbumpercars": 100}
CANONICAL = {"chain": 5, "npmpc": 12, "race_cars": 50, "unbumpercars": 8}
STATUSES = {("ok", "ok"), ("timeout", "skipped"), ("skipped_after_failure", "skipped")}


def main(study: Path, previous: Path) -> None:
  manifest = json.loads((study / "study.json").read_text())
  assert len(manifest["commands"]) == 12 and all(c["returncode"] == 0 for c in manifest["commands"])
  provenance = [manifest, *(json.loads(p.read_text()) for p in study.rglob("*provenance*.json"))]
  for record in provenance:
    assert record["git_commit"] == manifest["git_commit"] and not record["git_dirty"]
    assert record["vector_libm"] == "glibc" and record["cpu_settings"]["boost_enabled"] is False
    assert all(p["scaling_governor"] == "performance" for p in record["cpu_settings"]["policies"].values())
  raw, sweep, old = [], [], []
  lines = ["# October 20 study audit", "", f"Duration: {sum(c['seconds'] for c in manifest['commands']) / 60:.1f} minutes.", ""]
  for workload, sizes in SWEEP_GRID.items():
    rows = read_csv(study / "sweep" / workload / f"{workload}.csv")
    expected = {(str(size), backend) for size in sizes for backend in DEFAULT_BACKENDS[workload]}
    assert len(rows) == len(expected) and {(r["size"], r["backend"]) for r in rows} == expected
    assert all(r["repetition"] == "1" and (r["compile_status"], r["runtime_status"]) in STATUSES for r in rows)
    assert all(r["runtime_status"] == "ok" for r in rows if r["backend"] == "scaly")
    raw += rows
    sweep += read_csv(study / "sweep" / workload / f"{workload}.summary.csv")
    old += read_csv(previous / "sweep" / workload / f"{workload}.csv")
    lines.append(f"- {workload}: {dict(Counter(r['compile_status'] for r in rows))}")
  assert len(raw) == 106
  lines += ["", f"October 15 outcomes: {dict(Counter(r['compile_status'] for r in old))}."]
  assert all(r["runtime_ns_cv"] == "" and r["attempts"] == "1" for r in sweep)
  summary = json.loads((study / "closed-loop/closed_loop.summary.json").read_text())
  expected_runs = {(problem, solver, oracle) for problem, pairs in CLOSED_LOOP_PAIRS.items() for solver, oracle in pairs if oracle}
  assert len(summary["runs"]) == len(summary["groups"]) == len(expected_runs) == 15
  assert {(r["problem"], r["solver"], r["oracle"]) for r in summary["runs"]} == expected_runs
  assert all(r["repetition"] == 1 and r["steps"] == r["successful_steps"] == STEPS[r["problem"]] for r in summary["runs"])
  pairs = []
  assert len(summary["oracle_comparisons"]) == 7
  for pair in summary["oracle_comparisons"]:
    assert pair["same_steps"] and all(pair[f"{key}_mismatches"] == 0 for key in COUNTS)
    assert pair["iterations_mismatches"] == (2 if (pair["problem"], pair["solver"]) == ("unbumpercars", "sqp") else 0)
    control = "applied" if pair["problem"] == "unbumpercars" else "control"
    assert pair["state_max_abs_diff"] < 1e-6 and pair[f"{control}_max_abs_diff"] < 1e-6
    for metric in ("solver_time_ms", "fe_time_ms"):
      ratios = np.asarray(pair[f"{metric}_speedups"])
      assert len(ratios) == STEPS[pair["problem"]] and np.all(np.isfinite(ratios) & (ratios > 0))
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
  steps, timing = [], []
  for telemetry in sorted(study.glob("closed-loop/*/*/repeat_*/*/*/telemetry.csv")):
    run, rows = _episode(telemetry)
    assert [row["step"] for row in rows] == list(range(len(rows))) and all(row["success"] for row in rows)
    if run["problem"] == "race_cars":
      config = json.loads((telemetry.parent / "config.json").read_text())
      result = json.loads((telemetry.parent / "summary.json").read_text())
      assert config["horizon"] == result["horizon"] == 50 and result["laps"] == 1
      assert np.isclose(result["lap_time_s"], len(rows) * config["params"]["dt"])
    values = np.asarray([row["solver_time_ms"] for row in rows])
    timing.append(
      dict(
        problem=run["problem"],
        solver=run["solver"],
        oracle=run["oracle"],
        median=np.median(values),
        maximum=values.max(),
        over=int(np.sum(values > SAMPLE_MS[run["problem"]])),
        sample_fraction=values.max() / SAMPLE_MS[run["problem"]],
        iterations=run["iterations"],
      )
    )
    for row in rows:
      values = {key: f"{value:.6g}" if isinstance(value, float) else value for key, value in row.items()}
      steps.append(dict(problem=run["problem"], solver=run["solver"], oracle=run["oracle"]) | values)
  target = ROOT / "docs/assets/benchmarks"
  write_csv(target / "sweep.summary.csv", sweep)
  write_csv(target / "sweep.csv", raw)
  (target / "closed_loop.summary.csv").write_text((study / "closed-loop/closed_loop.summary.csv").read_text())
  write_csv(target / "closed_loop_speedup.summary.csv", pairs)
  write_csv(target / "closed_loop.steps.csv", steps)
  lines += [
    "",
    f"All {len(steps)} solver steps succeeded. All timed kernels passed the symbolic and compiled-C gates.",
    "",
    "## Paired speedups",
    "",
    "| Problem | Solver | Metric | Median | P10 | P90 |",
    "|---|---|---|---:|---:|---:|",
  ]
  lines += [f"| {r['problem']} | {r['solver']} | {r['metric']} | {r['median']:.4f} | {r['p10']:.4f} | {r['p90']:.4f} |" for r in pairs]
  lines += [
    "",
    "## Controller steps",
    "",
    "| Problem | Solver | Provider | Median, ms | Maximum, ms | Over sampling time | Mean iterations |",
    "|---|---|---|---:|---:|---:|---:|",
  ]
  lines += [
    f"| {r['problem']} | {r['solver']} | {r['oracle']} | {r['median']:.4f} | {r['maximum']:.4f} | {r['over']} | {r['iterations']:.3f} |"
    for r in timing
  ]
  lines += [
    "",
    "## Sweep cells since October 15",
    "",
    "Ratios are October 20 divided by October 15. One process gives no noise estimate.",
    "",
    "| Problem | Size | Backend | Runtime ratio | Kernel compile ratio | Source byte change | Object byte change |",
    "|---|---:|---|---:|---:|---:|---:|",
  ]
  old_cells = {(r["workload"], r["size"], r["backend"]): r for r in old if r["runtime_status"] == "ok"}
  changes = []
  for row in raw:
    before = old_cells.get((row["workload"], row["size"], row["backend"]))
    if before is None or row["runtime_status"] != "ok":
      continue
    ratio = float(row["runtime_ns"]) / float(before["runtime_ns"])
    changes.append(ratio)
    lines.append(
      f"| {row['workload']} | {row['size']} | {row['backend']} | {ratio:.4f} | "
      f"{float(row['kernel_compile_ms']) / float(before['kernel_compile_ms']):.4f} | "
      f"{int(row['source_bytes']) - int(before['source_bytes'])} | {int(row['executable_bytes']) - int(before['executable_bytes'])} |"
    )
  lines += ["", f"Runtime-ratio P10/median/P90: {np.percentile(changes, [10, 50, 90]).round(4).tolist()}.", "", "## Figure and prose values", ""]
  for workload, sizes in SWEEP_GRID.items():
    lines += [
      f"### {workload}",
      "",
      "| Size | Best CasADi | Runtime speedup | Scaly build, s | Best CasADi build, s | Scaly object, bytes |",
      "|---:|---|---:|---:|---:|---:|",
    ]
    for size in sizes:
      cell = [r for r in raw if r["workload"] == workload and int(r["size"]) == size and r["runtime_status"] == "ok"]
      scaly = next(r for r in cell if r["backend"] == "scaly")
      casadi = [r for r in cell if r["backend"] != "scaly"]
      best = min(casadi, key=lambda r: float(r["runtime_ns"])) if casadi else None
      build = lambda r: (float(r["codegen_ms"]) + float(r["kernel_compile_ms"])) / 1000
      speedup = f"{float(best['runtime_ns']) / float(scaly['runtime_ns']):.3f}" if best else ""
      casadi_build = f"{min(map(build, casadi)):.3f}" if casadi else ""
      lines.append(f"| {size} | {best['backend'] if best else ''} | {speedup} | {build(scaly):.3f} | {casadi_build} | {scaly['executable_bytes']} |")
    lines.append("")
  lines += [
    "## Build stages at controller sizes",
    "",
    "| Problem | Backend | Build, ms | Render, ms | Compile, ms | Render share, % | Compile share, % |",
    "|---|---|---:|---:|---:|---:|---:|",
  ]
  for row in raw:
    if row["runtime_status"] != "ok" or int(row["size"]) != CANONICAL[row["workload"]]:
      continue
    build, render, compile_time = (float(row[key]) for key in ("build_ms", "render_ms", "kernel_compile_ms"))
    total = build + render + compile_time
    lines.append(
      f"| {row['workload']} | {row['backend']} | {build:.1f} | {render:.1f} | {compile_time:.1f} | {100 * render / total:.2f} | {100 * compile_time / total:.2f} |"
    )
  lines += [
    "",
    "## Closed-loop changes since October 15",
    "",
    "| Problem | Solver | Provider | Total-time ratio | Function-evaluation ratio |",
    "|---|---|---|---:|---:|",
  ]
  old_summary = json.loads((previous / "closed-loop/closed_loop.summary.json").read_text())
  old_groups = {(r["problem"], r["solver"], r["oracle"]): r for r in old_summary["groups"]}
  for group in summary["groups"]:
    before = old_groups[(group["problem"], group["solver"], group["oracle"])]
    lines.append(
      f"| {group['problem']} | {group['solver']} | {group['oracle']} | "
      f"{group['solver_time_ms']['mean'] / before['solver_time_ms']['mean']:.4f} | "
      f"{group['fe_time_ms']['mean'] / before['fe_time_ms']['mean']:.4f} |"
    )
  lines += ["", "## Controller build and function evaluation", ""]
  for group in summary["groups"]:
    key = f"{group['problem']} {group['solver']} {group['oracle']}"
    lines.append(
      f"- {key}: build {group['build_ms']['mean'] / 1000:.4f} s, function-evaluation share "
      f"{100 * group['fe_time_ms']['mean'] / group['solver_time_ms']['mean']:.3f}%."
    )
  lines += ["", "## Race-car endpoint build shares", ""]
  for row in raw:
    if row["workload"] != "race_cars" or row["backend"] not in {"scaly", "casadi_sx"} or row["size"] not in {"5", "400"}:
      continue
    total = float(row["codegen_ms"]) + float(row["kernel_compile_ms"])
    lines.append(
      f"- {row['backend']} N={row['size']}: render {100 * float(row['render_ms']) / total:.2f}%, compile {100 * float(row['kernel_compile_ms']) / total:.2f}%."
    )
  (Path(__file__).parent / "analysis.md").write_text("\n".join(lines) + "\n")
  print(f"Audit passed: {len(raw)} sweep attempts, {dict(Counter(r['compile_status'] for r in raw))}, {len(steps)} successful solver steps.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("study", type=Path)
  parser.add_argument("previous", type=Path)
  args = parser.parse_args()
  main(args.study.resolve(), args.previous.resolve())
