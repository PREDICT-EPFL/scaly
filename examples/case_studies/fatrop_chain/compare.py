"""Run every variant of the Fatrop chain study in fresh processes and tabulate them.

    examples/case_studies/fatrop_chain/baseline/setup.sh
    uv run --with rockit-meco==0.6.7 examples/case_studies/fatrop_chain/compare.py --out examples/case_studies/fatrop_chain/results/solve.json

Variants per chain (2D, 3D): CasADi's Fatrop and IPOPT with the oracles in CasADi's virtual machine
(`vm`, the paper's fallback) and compiled (`jit`, the flags Scaly's JIT uses); Scaly's IPOPT (the
drop-in: same options, MUMPS on both sides, but Scaly's IPOPT 3.14.19 build against CasADi's 3.14.11),
Fatrop with Scaly's oracles (the drop-in with the same libfatrop), and Scaly's generated SQP. Each
variant runs in `--processes` fresh processes (one for the `jit` ones, whose compile takes minutes),
each with an empty JIT cache and the compiler logged through `baseline/cc_timed.sh`; the table keeps
the fastest process. Agreement is the largest difference of the solution from CasADi's IPOPT.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import wait_for_quiet  # noqa: E402

VARIANTS = [
  ("casadi", "fatrop", "vm"),
  ("casadi", "ipopt", "vm"),
  ("casadi", "fatrop", "jit"),
  ("casadi", "ipopt", "jit"),
  ("scaly", "fatrop", "scaly"),
  ("scaly", "ipopt", "scaly"),
  ("scaly", "sqp", "scaly"),
]


def run(side: str, solver: str, mode: str, dim: int, repeats: int) -> dict:
  env = {**os.environ, "OMP_NUM_THREADS": "1", "SCALY_CC": str(HERE / "baseline" / "cc_timed.sh")}
  env.pop("SCALY_CACHE_DIR", None)
  if side == "casadi":
    cmd = [sys.executable, str(HERE / "baseline" / "run_casadi.py"), "--dim", str(dim), "--solver", solver, "--mode", mode, "--repeats", str(repeats)]
  else:
    cmd = [sys.executable, str(HERE / "run_scaly.py"), "--dim", str(dim), "--solver", solver, "--repeats", str(repeats)]
  load = wait_for_quiet()
  proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
  if proc.returncode:
    return {"side": side, "solver": solver, "mode": mode, "dim": dim, "failed": proc.stderr[-800:], "load": load}
  return {"side": side, **json.loads(proc.stdout.strip().splitlines()[-1]), "load": load}


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--dims", type=int, nargs="+", default=[2, 3])
  ap.add_argument("--processes", type=int, default=5)
  ap.add_argument("--repeats", type=int, default=30)
  ap.add_argument("--only", nargs="*", help="side:solver:mode triples to run")
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()
  results = json.loads(args.out.read_text()) if args.out.exists() else {}
  for dim in args.dims:
    for side, solver, mode in VARIANTS:
      key = f"{dim}d:{side}:{solver}:{mode}"
      if args.only and f"{side}:{solver}:{mode}" not in args.only:
        continue
      runs = [run(side, solver, mode, dim, args.repeats) for _ in range(1 if mode == "jit" else args.processes)]
      ok = [r for r in runs if "failed" not in r]
      best = min(ok, key=lambda r: r["wall_min"]) if ok else runs[0]
      best["processes"] = len(runs)
      best["wall_min_per_process"] = [r["wall_min"] for r in ok]
      results[key] = best
      args.out.parent.mkdir(parents=True, exist_ok=True)
      args.out.write_text(json.dumps(results, indent=1))
      print(key, {k: best.get(k) for k in ("iterations", "wall_min", "fe_at_min", "t_setup", "t_build", "compile_wall", "failed")}, flush=True)
  for dim in args.dims:
    ref = np.array(results[f"{dim}d:casadi:ipopt:vm"]["x"])
    for key, r in results.items():
      if key.startswith(f"{dim}d:") and "x" in r:
        r["agreement"] = float(np.max(np.abs(np.array(r["x"]) - ref)))
  args.out.write_text(json.dumps(results, indent=1))
  print(table(results, args.dims))


def table(results: dict, dims: list[int]) -> str:
  rows = [
    "| Chain | Solver | Oracles | Iterations | Solve, ms | Function evaluation, ms | Rest, ms | Setup, s | Compile, s | Peak compiler memory, GB | Agreement |",
    "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  names = {"vm": "CasADi VM", "jit": "CasADi C (JIT)", "scaly": "Scaly C"}
  for dim in dims:
    for side, solver, mode in VARIANTS:
      r = results.get(f"{dim}d:{side}:{solver}:{mode}")
      if r is None:
        continue
      label = {"fatrop": "Fatrop 1.1.8", "ipopt": "IPOPT", "sqp": "Scaly SQP + PIQP"}[solver]
      if solver == "ipopt":
        label += " 3.14.19 (Scaly's)" if side == "scaly" else " 3.14.11 (CasADi's)"
      if "failed" in r:
        rows.append(f"| {dim}D | {label} | {names[mode]} | failed | | | | | | | |")
        continue
      setup = r.get("t_setup", r.get("t_model", 0) + r.get("t_build", 0) + r.get("t_first", 0))
      rows.append(
        f"| {dim}D | {label} | {names[mode]} | {r['iterations']} | {1e3 * r['wall_min']:.2f} | {1e3 * r['fe_at_min']:.2f} | {1e3 * (r['wall_min'] - r['fe_at_min']):.2f} "
        f"| {setup:.1f} | {r['compile_wall']:.1f} | {r['compile_peak_rss_bytes'] / 1e9:.2f} | {r.get('agreement', float('nan')):.0e} |"
      )
  return "\n".join(rows)


if __name__ == "__main__":
  main()
