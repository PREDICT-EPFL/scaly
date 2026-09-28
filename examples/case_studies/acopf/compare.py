"""The AC-OPF study's sweep: Scaly, JuMP and ExaModels on PGLib-OPF cases, all through Ipopt 3.14.19 and MUMPS.

    examples/case_studies/acopf/baseline/setup.sh
    uv run examples/case_studies/acopf/compare.py --out examples/case_studies/acopf/results/sweep.json
    uv run examples/case_studies/acopf/compare.py --out examples/case_studies/acopf/results/sweep.json --report

For every case and tolerance, three fresh processes behind the quiet-machine gate:

- `jump`: rosetta-opf's `jump.jl`, unmodified;
- `examodels`: rosetta-opf's `examodels.jl` with `baseline/examodels_v0.12.patch` (ExaModels 0.12's names
  for the same calls), in a copy of rosetta-opf's directory;
- `scaly`: `run_scaly.py` on the case as `baseline/export_case.jl` writes it (PowerModels' processing).

Every process solves the case twice and the second solve is recorded (the Julia processes first solve
rosetta-opf's warm-up case). Ipopt's options come from one `ipopt.opt` for the Julia side and the same
values passed to Scaly's solver: the tolerance, `linear_solver mumps`, `print_timing_statistics yes`.
All three are timed by Ipopt's own timing statistics, parsed from each process's output the same way:
the whole algorithm, function evaluations (every derivative and value callback) and the linear solver.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "baseline"
TP = BASE / "third_party"
sys.path.insert(0, str(HERE.parent))
from _common import wait_for_quiet  # noqa: E402

CASES = ["case14_ieee", "case118_ieee", "case300_ieee", "case1354_pegase", "case2869_pegase", "case9241_pegase", "case13659_pegase"]
STACKS = ("jump", "examodels", "scaly")
TIMING = re.compile(r"^\s*(\w[\w ]*?)\.{2,}:\s+[\d.]+ \(sys:\s+[\d.]+ wall:\s+([\d.]+)\)", re.M)


def julia() -> tuple[list[str], dict]:
  env = {}
  for line in (BASE / "env.sh").read_text().splitlines():
    if line.startswith("export "):
      key, value = line[len("export ") :].split("=", 1)
      env[key] = value.strip('"')
  return [env["JULIA"], f"--project={env['OPF_ENV']}"], {**os.environ, "JULIA_DEPOT_PATH": env["JULIA_DEPOT_PATH"]}


def parse_ipopt(block: str) -> dict:
  """Iterations, objective, exit status and every timing statistic's wall time from one Ipopt run's output."""
  out = {"timing": {name.strip(): float(wall) for name, wall in TIMING.findall(block)}}
  m = re.search(r"Number of Iterations\.*:\s+(\d+)", block)
  out["iterations"] = int(m.group(1)) if m else None
  m = re.search(r"^Objective\.*:\s+\S+\s+(\S+)", block, re.M)
  out["objective"] = float(m.group(1)) if m else None
  m = re.search(r"^EXIT: (.*)$", block, re.M)
  out["exit"] = m.group(1).strip() if m else None
  return out


def last_solve(stdout: str) -> str:
  return stdout.rsplit("==== SOLVE ====", 1)[-1]


def run_case(case: str, tol: float, work: Path) -> list[dict]:
  jl, jenv = julia()
  m_file = TP / "pglib-opf" / f"pglib_opf_{case}.m"
  exported = work / f"{case}.json"
  if not exported.exists():
    subprocess.run([*jl, str(BASE / "export_case.jl"), str(m_file), str(exported)], env=jenv, check=True, capture_output=True)
  run_dir = work / f"run_tol{tol:g}"
  run_dir.mkdir(exist_ok=True)
  (run_dir / "ipopt.opt").write_text(f"tol {tol:g}\nlinear_solver mumps\nprint_timing_statistics yes\n")
  rosetta = work / "rosetta-opf"
  if not rosetta.exists():
    shutil.copytree(TP / "rosetta-opf", rosetta, ignore=shutil.ignore_patterns(".git"))
    subprocess.run(["patch", "-p1", "-s", "-i", str(BASE / "examodels_v0.12.patch")], cwd=rosetta, check=True)
  rows = []
  for stack in STACKS:
    if stack == "scaly":
      out = run_dir / f"scaly_{case}.json"
      cmd = ["uv", "run", str(HERE / "run_scaly.py"), "--case", str(exported), "--tol", str(tol), "--out", str(out)]
      env = dict(os.environ)
    else:
      out = run_dir / f"{stack}_{case}.json"
      cmd = [*jl, str(BASE / "run_rosetta.jl"), str(rosetta / f"{stack}.jl"), str(m_file), str(out)]
      env = jenv
    load = wait_for_quiet()
    t0 = time.perf_counter()
    done = subprocess.run(cmd, cwd=run_dir, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if done.returncode:
      raise RuntimeError(f"{stack} {case} failed:\n{done.stdout[-2000:]}\n{done.stderr[-3000:]}")
    row = {"case": case, "tol": tol, "stack": stack, "process_wall": wall, "load": load, **parse_ipopt(last_solve(done.stdout)), "own": json.loads(out.read_text())}
    rows.append(row)
    t = row["timing"]
    print(
      f"{case:18s} tol={tol:g} {stack:9s} iterations {row['iterations']}, objective {row['objective']:.6f}, Ipopt {t.get('OverallAlgorithm', float('nan')):.3f} s, "
      f"function evaluations {t.get('Function Evaluations', float('nan')):.3f} s, process {wall:.1f} s",
      flush=True,
    )
  return rows


def table(rows: list[dict], tol: float) -> str:
  lines = [
    "| Case | Buses | Stack | Iterations | Objective | Ipopt total, s | Function evaluations, s | Linear solver, s | First solve, s |",
    "|" + "---|" * 9,
  ]
  for r in [r for r in rows if r["tol"] == tol]:
    t = r["timing"]
    first = r["own"]["first_call"] + (r["own"].get("t_build", 0.0) + r["own"].get("t_import", 0.0) if r["stack"] == "scaly" else r["own"]["t_warmup_process"])
    lin = t.get("PDSystemSolverTotal", float("nan"))
    lines.append(
      f"| {r['case']} | {r['case'].split('_')[0][4:]} | {r['stack']} | {r['iterations']} | {r['objective']:.6f} | {t['OverallAlgorithm']:.3f} | "
      f"{t['Function Evaluations']:.3f} | {lin:.3f} | {first:.1f} |"
    )
  return "\n".join(lines)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--cases", nargs="+", default=CASES)
  ap.add_argument("--tols", type=float, nargs="+", default=[1e-8, 1e-4])
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--report", action="store_true")
  args = ap.parse_args()
  rows = json.loads(args.out.read_text())["rows"] if args.out.exists() else []
  if not args.report:
    work = Path(tempfile.mkdtemp(prefix="acopf-"))
    done = {(r["case"], r["tol"]) for r in rows}
    for tol in args.tols:
      for case in args.cases:
        if (case, tol) in done:
          continue
        rows += run_case(case, tol, work)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"rows": rows}, indent=1))
  for tol in sorted({r["tol"] for r in rows}):
    print(f"\ntol {tol:g}\n" + table(rows, tol))


if __name__ == "__main__":
  main()
