"""The Scaly side of the AC-OPF study: one case, built, compiled and solved twice by Scaly's IPOPT.

    uv run examples/case_studies/acopf/run_scaly.py --case case.json --tol 1e-8 --out result.json

Reads the case as `baseline/export_case.jl` writes it, builds `scaly_impl.problem`, and solves it from
rosetta-opf's starting point with the same IPOPT options the Julia side reads from `ipopt.opt` (the
tolerance, MUMPS, timing statistics). The first call compiles; the second is the one recorded, as
the Julia side records its second solve. IPOPT's output goes to stdout, where compare.py parses its
timing statistics exactly as it does the Julia side's.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--case", type=Path, required=True)
  ap.add_argument("--tol", type=float, default=1e-8)
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()
  os.environ.setdefault("SCALY_CACHE_DIR", tempfile.mkdtemp(prefix="acopf-scaly-"))
  t0 = time.perf_counter()
  import scaly as sc

  import scaly_impl as si

  t_import = time.perf_counter() - t0
  t0 = time.perf_counter()
  c = si.load_case(args.case)
  t_load = time.perf_counter() - t0
  t0 = time.perf_counter()
  prob = si.problem(c)
  options = {"tol": args.tol, "linear_solver": "mumps", "print_timing_statistics": "yes", "print_level": 5}
  solve = sc.solver(prob, "ipopt", name=f"acopf_{c.name}", options=options)
  t_build = time.perf_counter() - t0
  x0 = si.start(c)
  zeros = tuple(np.zeros_like(a) for a in x0)
  runs = []
  for _ in range(2):
    print("==== SOLVE ====", flush=True)
    t0 = time.perf_counter()
    (va, vm, pg, qg, p, q), *_ = solve(x0, zeros, np.zeros(prob.n_eq), np.zeros(prob.n_ineq), ())
    wall = time.perf_counter() - t0
    st = solve.solver_stats()
    runs.append({"wall": wall, "iterations": int(st.iter), "objective": float(st.obj), "status": st.to_solver_status().name, "t_total": st.t_total, "t_fe": st.t_fe})
    sys.stdout.flush()
  row = {
    "case": c.name,
    "tol": args.tol,
    "n_var": int(sum(a.size for a in x0)),
    "n_eq": int(prob.n_eq),
    "n_ineq": int(prob.n_ineq),
    "t_import": t_import,
    "t_load": t_load,
    "t_build": t_build,
    "first_call": runs[0]["wall"],
    **runs[1],
    "vm": vm.tolist(),
    "pg": pg.tolist(),
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(row))


if __name__ == "__main__":
  main()
