"""The SCvx study's comparison: OpenSCvx and Scaly on the 6-DoF landing, end to end, in fresh processes.

    examples/case_studies/scvx/baseline/setup.sh
    uv run examples/case_studies/scvx/compare.py --out examples/case_studies/scvx/results/compare.json
    uv run examples/case_studies/scvx/compare.py --out examples/case_studies/scvx/results/compare.json --report

Each round runs, behind the quiet-machine gate: OpenSCvx with an empty compilation cache (cold), OpenSCvx
again with that cache (warm), Scaly with an empty JIT cache (cold) and Scaly again with it (warm). The
table keeps each column's fastest round; `parts` sets Scaly's solve, and its discretization and QP solve
timed alone from C, beside OpenSCvx's per-iteration split recorded in `results/reference_6dof.json`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "baseline"
sys.path.insert(0, str(HERE.parent))
from _common import wait_for_quiet  # noqa: E402


def env() -> dict[str, str]:
  out = {}
  for line in (BASE / "env.sh").read_text().splitlines():
    if line.startswith("export "):
      k, v = line[len("export ") :].split("=", 1)
      out[k] = v.strip('"')
  return out


def run(cmd: list[str], out: Path) -> dict:
  load = wait_for_quiet()
  t0 = time.perf_counter()
  subprocess.run(cmd, check=True, capture_output=True)
  wall = time.perf_counter() - t0
  return {**json.loads(out.read_text()), "process_wall": wall, "load": load}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--rounds", type=int, default=5)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--report", action="store_true")
  args = ap.parse_args()
  if not args.report:
    e = env()
    work = Path(tempfile.mkdtemp(prefix="scvx-compare-"))
    rows = []
    for r in range(args.rounds):
      osx_cache, scaly_cache = work / f"openscvx_cache_{r}", work / f"scaly_cache_{r}"
      osx_cache.mkdir()
      scaly_cache.mkdir()
      for start in ("cold", "warm"):
        out = work / f"openscvx_{r}_{start}.json"
        rows.append(
          {
            "stack": "openscvx",
            "start": start,
            "round": r,
            **run([e["OPENSCVX_PYTHON"], str(BASE / "run_openscvx.py"), e["OPENSCVX"], str(osx_cache), str(out)], out),
          }
        )
      for start in ("cold", "warm"):
        out = work / f"scaly_{r}_{start}.json"
        rows.append(
          {
            "stack": "scaly",
            "start": start,
            "round": r,
            **run(["uv", "run", str(HERE / "run_scaly.py"), "--cache", str(scaly_cache), "--out", str(out)], out),
          }
        )
      print(f"round {r} done", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows}, indent=1))
  rows = json.loads(args.out.read_text())["rows"]
  print(table(rows))
  print()
  print(parts(rows))


def end_to_end(r: dict) -> float:
  """From the imports to the first solution, inside the process."""
  if r["stack"] == "openscvx":
    return r["t_import"] + r["t_problem"] + r["t_initialize"] + r["t_solve"]
  return r["t_import"] + r["t_build"] + r["t_first_call"]


def table(rows: list[dict]) -> str:
  """The fastest round per column. OpenSCvx's iteration counter includes the warm-up iteration its
  `initialize()` runs, so its PTR iterations are one fewer; the process wall of Scaly's runs includes the
  study's own C timing (a second library and the harness), so it is not in the table."""
  lines = [
    "| Stack | Start | Imports, s | Build, s | Compile and first solve, s | End to end, s | Solve, ms | PTR iterations | Final mass |",
    "|" + "---|" * 9,
  ]
  for stack in ("openscvx", "scaly"):
    for start in ("cold", "warm"):
      rs = [r for r in rows if r["stack"] == stack and r["start"] == start]
      r = min(rs, key=end_to_end)
      if stack == "openscvx":
        build, first, solve, iters = r["t_problem"], r["t_initialize"], 1e3 * min(x["t_solve"] for x in rs), r["iterations"] - 1
      else:
        build, first, solve, iters = r["t_build"], r["t_first_call"], 1e3 * min(x["solve_c_s"] for x in rs), r["iterations"]
      lines.append(
        f"| {stack} | {start} | {r['t_import']:.2f} | {build:.2f} | {first:.2f} | {end_to_end(r):.2f} | {solve:.2f} | {iters} | {r['final_mass']:.7f} |"
      )
  return "\n".join(lines)


def parts(rows: list[dict]) -> str:
  """Where one PTR iteration's time goes, per side, in milliseconds. OpenSCvx's split is its own
  instrumented run (`reference_6dof.json`: the whole step, CVXPY's solve call, QOCO inside it), its
  iterations 2 to 5; Scaly's is the best from C over every round, the QP at OpenSCvx's first iteration."""
  ref = json.loads((HERE / "results" / "reference_6dof.json").read_text())["iterations"][1:]
  step, cvx, qoco = (min(it[k] for it in ref) * 1e3 for k in ("ptr_step_wall_s", "cvxpy_solve_wall_s", "qoco_solve_time_s"))
  sc = [r for r in rows if r["stack"] == "scaly"]
  disc, qp, solve = (min(r[k] for r in sc) * 1e3 for k in ("discretize_c_s", "qp_c_s", "solve_c_s"))
  iters = sc[0]["iterations"]
  return "\n".join(
    [
      "| Per PTR iteration, ms | OpenSCvx | Scaly |",
      "|---|---:|---:|",
      f"| whole iteration | {step:.2f} | {solve / iters:.3f} |",
      f"| QP solve | {qoco:.3f} (QOCO; CVXPY's call {cvx:.2f}) | {qp:.3f} |",
      f"| discretization | two per iteration, and the rest: {step - cvx:.2f} | {disc:.3f} |",
    ]
  )


if __name__ == "__main__":
  main()
