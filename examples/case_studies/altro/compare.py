"""Run the ALTRO study's solvers in fresh processes and tabulate them.

    examples/case_studies/altro/baseline/setup.sh
    uv run examples/case_studies/altro/compare.py --out examples/case_studies/altro/results/solve.json

Per problem (parallel park, cartpole), each algorithm on each side: the augmented-Lagrangian iLQR alone
(`altro_al`, `scaly_al`) and full ALTRO, the augmented Lagrangian to 1e-4 and then the projected Newton
phase (`altro_pn`, `scaly_pn`); Altro.jl 0.5 in Julia, Scaly's in generated C (`sc.ocp.ALTRO`, following
Altro.jl's source). Each runs in `--processes` fresh processes, each
held until the machine is quiet (`_common.wait_for_quiet`); the table keeps the fastest. A Julia process
times cold solves with BenchmarkTools (50 samples); a Scaly process times the generated entry point from
C (`--repeats` calls). Agreement is the largest difference of each trajectory from the same algorithm's
in Altro.jl.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import wait_for_quiet  # noqa: E402

PROBLEMS = ("parallel_park", "cartpole")
VARIANTS = ("altro_al", "scaly_al", "altro_pn", "scaly_pn")


def julia() -> list[str]:
  """The Julia command `baseline/setup.sh` prepared: its binary, depot and project, from `baseline/env.sh`."""
  env = {}
  for line in (HERE / "baseline" / "env.sh").read_text().splitlines():
    if line.startswith("export "):
      key, value = line[len("export ") :].split("=", 1)
      env[key] = value.strip('"')
  os.environ["JULIA_DEPOT_PATH"] = env["JULIA_DEPOT_PATH"]
  return [env["JULIA"], f"--project={env['ALTRO_ENV']}"]


def run(variant: str, problem: str, tolerance: float, repeats: int, out: Path) -> dict:
  if variant.startswith("scaly"):
    cmd = ["uv", "run", str(HERE / "run_scaly.py"), "--problem", problem, "--tolerance", str(tolerance), "--repeats", str(repeats), "--out", str(out)]
    cmd += ["--projected-newton"] if variant == "scaly_pn" else []
  else:
    cmd = [*julia(), str(HERE / "baseline" / "run_altro.jl"), problem, str(variant == "altro_pn").lower(), str(tolerance), str(out)]
  load = wait_for_quiet()
  t0 = time.perf_counter()
  done = subprocess.run(cmd, capture_output=True, text=True)
  wall = time.perf_counter() - t0
  if done.returncode:
    raise RuntimeError(f"{variant} {problem} failed:\n{done.stderr[-3000:]}")
  row = json.loads(out.read_text())
  row.update(variant=variant, process_wall=wall, load=load)
  return row


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--problems", nargs="+", default=list(PROBLEMS))
  ap.add_argument("--tolerance", type=float, default=1e-6)
  ap.add_argument("--processes", type=int, default=3)
  ap.add_argument("--repeats", type=int, default=200)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--report", action="store_true", help="tabulate an existing --out without running anything")
  args = ap.parse_args()
  if args.report:
    print(table(json.loads(args.out.read_text())["rows"]))
    return
  work = Path(tempfile.mkdtemp(prefix="altro-compare-"))
  rows = []
  for problem in args.problems:
    best: dict[str, dict] = {}
    for p in range(args.processes):
      for variant in VARIANTS:
        row = run(variant, problem, args.tolerance, args.repeats, work / f"{variant}_{problem}_{p}.json")
        if variant not in best or row["time_min"] < best[variant]["time_min"]:
          best[variant] = row
        print(
          problem,
          variant,
          p,
          f"{1e3 * row['time_min']:.3f} ms",
          row["iterations"],
          f"{row['cost']:.8f}",
          f"{row['max_violation']:.2e}",
          file=sys.stderr,
          flush=True,
        )
    for variant in VARIANTS:
      row, ref = best[variant], best["altro_" + variant.split("_")[1]]
      # Altro.jl's trajectory holds a control at the last knot too, which no constraint or cost reaches.
      row["dx_vs_altro"] = float(np.abs(np.asarray(row["X"]) - np.asarray(ref["X"])).max())
      row["du_vs_altro"] = float(np.abs(np.asarray(row["U"]) - np.asarray(ref["U"])[: len(row["U"])]).max())
      row["objective"] = objective(problem, row["X"], row["U"])
      row["dobjective_rel_vs_altro"] = abs(row["objective"] - objective(problem, ref["X"], ref["U"])) / abs(row["objective"])
      rows.append(row)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps({"tolerance": args.tolerance, "processes": args.processes, "rows": rows}, indent=1))
  print(table(rows))


def objective(problem: str, X: list, U: list) -> float:
  """The problem's cost at a trajectory, the same formula for both sides (Altro.jl's `cost(solver)` after
  a solve is its own evaluation, which this recomputes from the returned trajectory)."""
  sys.path.insert(0, str(HERE))
  from scaly_impl import PROBLEMS

  ocp = PROBLEMS[problem]()
  K = ocp.n_knots - 1
  X, U = np.asarray(X)[: K + 1], np.asarray(U)[:K]
  e = X - ocp.xf
  return float(ocp.dt * 0.5 * ((e[:K] ** 2 @ ocp.q).sum() + (U**2 @ ocp.r).sum()) + 0.5 * (e[K] ** 2 @ ocp.qf))


def first_solve(r: dict) -> float:
  """Seconds from a fresh process to the first solution, less interpreter start-up: Julia's package load,
  problem setup and compiling first solve; Scaly's import, build, C generation and compile."""
  return r["t_load"] + r["t_setup"] + r["t_first_solve"] if "t_load" in r else r["t_import"] + r["t_build"] + r["t_generate"] + r["t_compile"]


def table(rows: list[dict]) -> str:
  head = "| Problem | Algorithm | Implementation | Iterations | Objective | Max violation | Max \\|Δx\\| vs Altro.jl | Solve (best) | Speed-up | First solve |"
  lines = [head, "|" + "---|" * 10]
  ref = {(r["problem"], r["variant"].split("_")[1]): r for r in rows if r["variant"].startswith("altro")}
  for r in rows:
    impl, alg = r["variant"].split("_")
    base = ref[(r["problem"], alg)]
    dx = float(np.abs(np.asarray(r["X"]) - np.asarray(base["X"])).max())
    lines.append(
      f"| {r['problem']} | {'AL-iLQR' if alg == 'al' else 'ALTRO'} | {'Altro.jl' if impl == 'altro' else 'Scaly, generated C'} | {r['iterations']} | "
      f"{objective(r['problem'], r['X'], r['U']):.10f} | {r['max_violation']:.1e} | {dx:.0e} | {1e3 * r['time_min']:.3f} ms | "
      f"{base['time_min'] / r['time_min']:.1f}x | {first_solve(r):.1f} s |"
    )
  return "\n".join(lines)


if __name__ == "__main__":
  main()
