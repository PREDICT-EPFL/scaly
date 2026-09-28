"""The CasADi side of the Fatrop chain study: the authors' rockit problem, solved by CasADi's own nlpsol.

    uv run --with rockit-meco==0.6.7 examples/case_studies/fatrop_chain/baseline/run_casadi.py --dim 2 --solver fatrop --mode vm

rockit builds the transcription exactly as `fatrop_benchmarks` does (MultipleShooting, N = 25, one RK4
step per interval); the NLP it hands to Opti is then given to `nlpsol` directly, so the timer covers the
solver call and not Opti's Python bookkeeping. `--mode vm` evaluates the oracles in CasADi's virtual
machine (expanded to SX), which is what the paper fell back to when compiling the chain ran out of
memory; `--mode jit` compiles them with the flags Scaly's JIT uses, through `cc_timed.sh`, which logs
each compiler run's wall time and peak memory. Prints one JSON line.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FE_KEYS = ("t_wall_nlp_f", "t_wall_nlp_g", "t_wall_nlp_grad", "t_wall_nlp_grad_f", "t_wall_nlp_jac_g", "t_wall_nlp_hess_l")
IPOPT = {
  "tol": 1e-8,
  "gamma_theta": 1e-12,
  "mu_init": 1e2,
  "kappa_d": 1e-5,
  "min_refinement_steps": 0,
  "residual_ratio_max": 1e-6,
  "linear_solver": "mumps",
}


def build_nlp(dim: int, no_masses: int, horizon: int):
  import casadi as ca
  import numpy as np
  import rockit

  sys.path.insert(0, str(HERE / "third_party" / "fatrop_benchmarks"))
  from hanging_chain.problem_specificationMPC import HangingChainMPC

  prob = HangingChainMPC(T={2: 4.0, 3: 2.0}[dim], no_masses=no_masses, dim=dim)
  prob.ocp.method(rockit.MultipleShooting(N=horizon))
  prob.ocp.solver("ipopt", {"expand": True, "print_time": False, "ipopt.print_level": 0, "ipopt.sb": "yes", "ipopt.max_iter": 0})
  try:
    prob.ocp.solve()  # max_iter 0: only makes rockit lay out the NLP
  except RuntimeError:
    pass
  opti = prob.ocp._method.opti
  x0 = np.array(opti.debug.value(opti.x, opti.initial())).ravel()
  lbg, ubg = (np.array(ca.evalf(b)).ravel() for b in (opti.lbg, opti.ubg))
  return {"x": opti.x, "f": opti.f, "g": opti.g}, x0, lbg, ubg


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--dim", type=int, choices=(2, 3), required=True)
  ap.add_argument("--solver", choices=("fatrop", "ipopt"), required=True)
  ap.add_argument("--mode", choices=("vm", "jit"), required=True)
  ap.add_argument("--no-masses", type=int, default=6)
  ap.add_argument("--horizon", type=int, default=25)
  ap.add_argument("--repeats", type=int, default=20)
  args = ap.parse_args()

  import casadi as ca
  import numpy as np

  t0 = time.perf_counter()
  nlp, x_init, lbg, ubg = build_nlp(args.dim, args.no_masses, args.horizon)
  t_model = time.perf_counter() - t0
  opts: dict = {"expand": True, "print_time": False}
  if args.solver == "fatrop":
    equality = [bool(lo == hi) for lo, hi in zip(lbg, ubg, strict=True)]  # what Opti passes for rockit
    opts |= {"structure_detection": "auto", "equality": equality, "fatrop": {"print_level": 0, "tol": 1e-8}}
  else:
    opts |= {"ipopt": {"print_level": 0, "sb": "yes", **IPOPT}}
  work = Path(tempfile.mkdtemp(prefix="fatrop-chain-casadi-"))
  log = work / "cc.log"
  if args.mode == "jit":
    native = "-mcpu=native" if platform.machine().lower() in {"arm64", "aarch64"} else "-march=native"
    os.environ["CC_TIMED_LOG"] = str(log)
    opts |= {"jit": True, "compiler": "shell", "jit_options": {"compiler": str(HERE / "cc_timed.sh"), "flags": ["-O2", native, "-fno-math-errno"]}}
  os.chdir(work)
  t0 = time.perf_counter()
  solver = ca.nlpsol("chain", args.solver, nlp, opts)
  t_build = time.perf_counter() - t0
  t0 = time.perf_counter()
  sol = solver(x0=x_init, lbg=lbg, ubg=ubg)
  t_first = time.perf_counter() - t0
  walls, fes = [], []
  for _ in range(args.repeats):
    t0 = time.perf_counter()
    sol = solver(x0=x_init, lbg=lbg, ubg=ubg)
    walls.append(time.perf_counter() - t0)
    st = solver.stats()
    fes.append(sum(st.get(k, 0.0) for k in FE_KEYS))
  st = solver.stats()
  compiles = [line.split() for line in log.read_text().splitlines()] if log.exists() else []
  record = {
    "dim": args.dim,
    "solver": args.solver,
    "mode": args.mode,
    "iterations": int(st.get("iter_count", -1)),
    "status": str(st.get("return_status", st.get("success"))),
    "t_model": t_model,
    "t_build": t_build,
    "t_first": t_first,
    "wall_min": min(walls),
    "wall_median": float(np.median(walls)),
    "fe_at_min": fes[int(np.argmin(walls))],
    "fe_median": float(np.median(fes)),
    "fe_split": {k: st.get(k, 0.0) for k in FE_KEYS},
    "compile_wall": sum(float(e) - float(s) for s, e, _, _ in compiles),
    "compile_peak_rss_bytes": max((int(r) for _, _, r, _ in compiles), default=0),
    "compile_failures": sum(int(c) != 0 for *_, c in compiles),
    "x": np.array(sol["x"]).ravel().tolist(),
  }
  print(json.dumps(record))


if __name__ == "__main__":
  main()
