"""The Scaly side of the Fatrop chain study, one variant per process. Prints one JSON line.

    uv run examples/case_studies/fatrop_chain/run_scaly.py --dim 2 --solver ipopt

`--solver ipopt` is the drop-in against CasADi's IPOPT (same options, MUMPS); `fatrop` is the drop-in
against CasADi's Fatrop plugin (the same libfatrop, Scaly's oracles through `fatrop_scaly.c`);
`sqp` is the full stack, Scaly's generated SQP with PIQP subproblems. Setup is everything from the
first Scaly import to the first result: modelling, derivatives, C generation and compilation. With
`SCALY_CC` pointing at `baseline/cc_timed.sh` the compiler's wall time and peak memory are logged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--dim", type=int, choices=(2, 3), required=True)
  ap.add_argument("--solver", choices=("ipopt", "sqp", "fatrop"), required=True)
  ap.add_argument("--no-masses", type=int, default=6)
  ap.add_argument("--repeats", type=int, default=20)
  args = ap.parse_args()
  work = Path(tempfile.mkdtemp(prefix="fatrop-chain-scaly-"))
  os.environ.setdefault("SCALY_CACHE_DIR", str(work / "cache"))
  log = work / "cc.log"
  os.environ["CC_TIMED_LOG"] = str(log)
  import numpy as np

  t0 = time.perf_counter()
  import scaly as sc

  sys.path.insert(0, str(HERE))
  from scaly_impl import IPOPT_OPTIONS, build, initial_guess, initial_state, sizes

  dim, nm = args.dim, args.no_masses
  nx, nu = sizes(dim, nm)
  x0 = initial_state(dim, nm)
  if args.solver == "fatrop":
    from fatrop_dropin import build as build_fatrop
    from fatrop_dropin import from_fatrop, solver, to_fatrop

    b = {"horizon": 25}
    lib, info = build_fatrop(dim, work / "fatrop", nm)
    solve_fatrop = solver(lib, dim, nm, options={"tol": 1e-8})
    w0 = to_fatrop(initial_guess(dim, nm, 25), nx, nu, 25)

    def run():
      w, out = solve_fatrop(x0, w0)
      return from_fatrop(w, nx, nu, 25), out["iterations"], out["wall"], out["t_oracle"]

  else:
    b = build(dim, nm)
    p = b["problem"]
    options = IPOPT_OPTIONS if args.solver == "ipopt" else {"tol": 1e-8, "max_iter": 100}
    solve = sc.solver(p, args.solver, name=f"fatrop_chain{dim}d_M{nm}_{args.solver}", options=options)
    z0 = initial_guess(dim, nm, b["horizon"])
    zeros = (np.zeros(b["n_var"]), np.zeros(p.n_eq), np.zeros(p.n_ineq))

    def run():
      z, *_ = solve(z0, *zeros, x0)
      st = solve.solver_stats()
      return np.asarray(z), st.iter, st.t_total, st.t_fe

  z, iters, wall, fe = run()
  t_setup = time.perf_counter() - t0
  walls, fes = [], []
  for _ in range(args.repeats):
    z, iters, wall, fe = run()
    walls.append(wall)
    fes.append(fe)
  compiles = [line.split() for line in log.read_text().splitlines()] if log.exists() else []
  i = int(np.argmin(walls))
  print(
    json.dumps(
      {
        "dim": dim,
        "solver": args.solver,
        "mode": "scaly",
        "iterations": int(iters),
        "t_setup": t_setup,
        "wall_min": walls[i],
        "wall_median": float(np.median(walls)),
        "fe_at_min": fes[i],
        "fe_median": float(np.median(fes)),
        "compile_wall": sum(float(e) - float(s) for s, e, _, _ in compiles),
        "compile_peak_rss_bytes": max((int(r) for _, _, r, _ in compiles), default=0),
        "compile_failures": sum(int(c) != 0 for *_, c in compiles),
        "x": z.tolist(),
      }
    )
  )


if __name__ == "__main__":
  main()
