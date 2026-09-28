"""Render the sweep and closed-loop tables the results pages quote from a study directory."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics

import numpy as np

BACKEND_ORDER = (
  "scaly",
  "casadi_sx",
  "casadi_mx",
  "casadi_call_mx",
  "casadi_map_sx",
  "casadi_mx_gemm",
  "casadi_mx_gemm_classic",
  "casadi_mx_gemm_blasfeo",
)
COUNTS = ("n_eval_f", "n_eval_g", "n_eval_grad_f", "n_eval_jac_g", "n_eval_h")
METRICS = ("solver_time_ms", "fe_time_ms", "qp_time_ms", "globalization_time_ms", "glue_time_ms", "iterations")
SWEEP_HEADER = (
  "| Size | Backend | Successful processes | Mean, µs | CV, % | Executable bytes | Workspace, doubles | Kernel compile mean, ms |\n"
  "|---:|---|---:|---:|---:|---:|---:|---:|"
)


def sweep_table(raw_csv: Path) -> str:
  """One Markdown row per (size, backend) cell, with failure statuses in place of a runtime."""
  rows = list(csv.DictReader(raw_csv.open()))
  summary = list(csv.DictReader(raw_csv.with_suffix(".summary.csv").open()))
  cells = defaultdict(list)
  for row in rows:
    cells[(int(row["size"]), row["backend"])].append(row)
  lines = [SWEEP_HEADER]
  for s in sorted(summary, key=lambda r: (int(r["size"]), BACKEND_ORDER.index(r["backend"]))):
    group = cells[(int(s["size"]), s["backend"])]
    successful = int(s["successful"])
    sizes = []
    for field in ("executable_bytes", "workspace"):
      values = sorted({int(r[field]) for r in group if r[field]})
      sizes.append("" if not values else str(values[0]) if len(values) == 1 else f"{values[0]}–{values[-1]}")
    if successful:
      runtime = f"{float(s['runtime_ns_mean']) / 1000:.3f}"
    else:
      statuses = Counter((r["compile_status"], r["runtime_status"]) for r in group)
      runtime = "; ".join(a if b == "skipped" else f"{a}/{b}" for a, b in statuses)
    cv = f"{float(s['runtime_ns_cv']) * 100:.2f}" if s["runtime_ns_cv"] else ""
    compile_time = f"{float(s['kernel_compile_ms_mean']):.1f}" if s["kernel_compile_ms_mean"] else ""
    lines.append(f"| {s['size']} | `{s['backend']}` | {successful}/{s['attempts']} | {runtime} | {cv} | {sizes[0]} | {sizes[1]} | {compile_time} |")
  return "\n".join(lines) + "\n"


def _episode(telemetry: Path) -> tuple[dict, list[dict]]:
  path = telemetry.parent
  solver, oracle = path.name.split("+")
  rows = []
  for record in csv.DictReader(telemetry.open()):
    row = {"step": int(record["step"]), "success": record["success"] == "True"}
    row.update({k: float(record[k]) for k in ("solver_time_ms", "fe_time_ms", "iterations", "globalization_time_ms", "glue_time_ms")})
    row["qp_time_ms"] = float(record.get("qp_time_ms") or record["qp_or_solver_time_ms"])
    row.update({k: int(float(record[k])) for k in COUNTS if record.get(k)})
    rows.append(row)
  summary = json.loads((path / "summary.json").read_text())
  run = dict(
    problem=path.parent.name,
    solver=solver,
    oracle=oracle,
    repetition=int(path.parents[1].name.removeprefix("repeat_")),
    steps=len(rows),
    successful_steps=sum(r["success"] for r in rows),
  )
  run.update({k: statistics.mean(r[k] for r in rows) for k in METRICS})
  run.update({k: sum(r[k] for r in rows) for k in COUNTS if all(k in r for r in rows)})
  run.update({k: summary.get(k) for k in ("build_ms", "first_solve_ms", "steady_step_ms")})
  return run, rows


def closed_loop_summary(root: Path) -> dict:
  """Per-process means, per-provider dispersion, and the equal-work comparison of the two oracle providers."""
  episodes = {}
  runs = []
  for telemetry in sorted(root.glob("**/repeat_*/*/*+*/telemetry.csv")):
    run, rows = _episode(telemetry)
    runs.append(run)
    episodes[(run["problem"], run["solver"], run["repetition"], run["oracle"])] = (telemetry.parent, rows)
  comparisons = []
  for problem, solver, repetition in sorted({key[:3] for key in episodes}):
    if any((problem, solver, repetition, oracle) not in episodes for oracle in ("scaly", "casadi")):
      continue
    a, ar = episodes[(problem, solver, repetition, "scaly")]
    b, br = episodes[(problem, solver, repetition, "casadi")]
    row: dict = dict(problem=problem, solver=solver, repetition=repetition, same_steps=len(ar) == len(br))
    if len(ar) == len(br):
      for key in ("iterations", *COUNTS):
        row[f"{key}_mismatches"] = sum(x.get(key) != y.get(key) for x, y in zip(ar, br, strict=True))
    with np.load(a / "rollout.npz") as av, np.load(b / "rollout.npz") as bv:
      for key in ("state", "applied" if problem == "unbumpercars" else "control"):
        same = av[key].shape == bv[key].shape
        row[f"{key}_max_abs_diff"] = float(np.max(np.abs(av[key] - bv[key]))) if same else None
    comparisons.append(row)
  groups = []
  for problem, solver, oracle in sorted({(r["problem"], r["solver"], r["oracle"]) for r in runs}):
    selected = [r for r in runs if (r["problem"], r["solver"], r["oracle"]) == (problem, solver, oracle)]
    group: dict = dict(problem=problem, solver=solver, oracle=oracle, processes=len(selected))
    group["all_steps_successful"] = all(r["steps"] == r["successful_steps"] for r in selected)
    group["steps"] = selected[0]["steps"]
    for key in (*METRICS, "build_ms", "first_solve_ms", "steady_step_ms"):
      values = [r[key] for r in selected if r.get(key) is not None]
      mean = statistics.mean(values) if values else None
      sd = statistics.stdev(values) if len(values) > 1 else None
      group[key] = dict(mean=mean, stdev=sd, cv=sd / mean if sd is not None and mean else None)
    groups.append(group)
  return dict(runs=runs, groups=groups, oracle_comparisons=comparisons)


def _flat(group: dict) -> dict:
  row = {}
  for key, value in group.items():
    row.update({f"{key}_{stat}": x for stat, x in value.items()} if isinstance(value, dict) else {key: value})
  return row


def closed_loop_tables(summary: dict) -> str:
  """The timing table and the episode-agreement table on the results index."""
  lines = [
    "| Problem | Solver | Provider | Total, ms | Total CV, % | Function evaluation, ms | QP, ms | Globalization, ms | Glue, ms | Steps | Processes |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for g in summary["groups"]:
    total = g["solver_time_ms"]
    cv = f"{total['cv'] * 100:.2f}" if total["cv"] is not None else ""
    cells = " | ".join(f"{g[k]['mean']:.3f}" for k in ("fe_time_ms", "qp_time_ms", "globalization_time_ms", "glue_time_ms"))
    lines.append(f"| {g['problem']} | {g['solver']} | {g['oracle']} | {total['mean']:.3f} | {cv} | {cells} | {g['steps']} | {g['processes']} |")
  if summary["oracle_comparisons"]:
    lines += [
      "",
      "| Problem | Solver | Repetitions | Per-step iteration mismatches | Per-step oracle-count mismatches | Maximum state difference | Maximum control difference |",
      "|---|---|---:|---:|---:|---:|---:|",
    ]
    by_problem = defaultdict(list)
    for row in summary["oracle_comparisons"]:
      by_problem[(row["problem"], row["solver"])].append(row)
    for (problem, solver), rows in sorted(by_problem.items()):
      iterations = sum(r.get("iterations_mismatches", 0) for r in rows)
      oracles = sum(r.get(f"{k}_mismatches", 0) for r in rows for k in COUNTS)
      control_key = "applied_max_abs_diff" if problem == "unbumpercars" else "control_max_abs_diff"
      state = max((r["state_max_abs_diff"] or 0.0) for r in rows)
      control = max((r[control_key] or 0.0) for r in rows)
      lines.append(f"| {problem} | {solver} | {len(rows)} | {iterations} | {oracles} | {state:.3g} | {control:.3g} |")
  return "\n".join(lines) + "\n"


def report(study_dir: Path) -> Path:
  """Write every table a study directory supports and return the combined `report.md`."""
  sections = []
  for raw_csv in sorted(study_dir.glob("sweep/*/*.csv")):
    if raw_csv.name.endswith(".summary.csv") or not raw_csv.with_suffix(".summary.csv").exists():
      continue
    table = sweep_table(raw_csv)
    (raw_csv.parent / "table.md").write_text(table)
    rows = list(csv.DictReader(raw_csv.open()))
    statuses = Counter((r["compile_status"], r["runtime_status"]) for r in rows)
    outcome = ", ".join(f"{count} {a if b == 'skipped' else b}" for (a, b), count in sorted(statuses.items()))
    sections.append(f"## Sweep: {raw_csv.parent.name}\n\n{len(rows)} rows: {outcome}.\n\n{table}")
  closed_loop = study_dir / "closed-loop"
  if closed_loop.is_dir():
    summary = closed_loop_summary(closed_loop)
    if summary["runs"]:
      (closed_loop / "closed_loop.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
      groups = [_flat(group) for group in summary["groups"]]
      with (closed_loop / "closed_loop.summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
      sections.append(f"## Closed loop\n\n{closed_loop_tables(summary)}")
  out = study_dir / "report.md"
  out.write_text(f"# Study report: {study_dir.name}\n\n" + "\n".join(sections))
  return out
