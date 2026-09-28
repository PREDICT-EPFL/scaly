"""The QOCO side of the embedded-QP study: QOCO, QOCOGEN and Clarabel on the oscillating-masses MPC.

    uv run --with qoco --with qocogen --with clarabel --with cvxpy --with pandas \\
      examples/case_studies/embedded_qp/baseline/run_qoco.py --horizon 20 --out results/qoco_T20.json

The CVXPY problem, its conversion to QOCO's standard form and the timed C harness are qoco-benchmarks'
own (`baseline/third_party/qoco-benchmarks`, pinned by `setup.sh`), fed the instance `problem.py`
replays. As in the benchmark, QOCOGEN is generated from the horizon's first instance and every instance
is solved by that code with its data rewritten (`update_data`, inside the timer, then
`qoco_custom_solve`, the minimum over `--runs`). Two differences from the benchmark, both for the
comparison with Scaly: the generated sources are compiled with the flags Scaly's JIT uses (not CMake's
`-O3 -march=native`, which Apple clang rejects on arm64), and the generation and compile times and the
machine-code size are recorded. QOCO and Clarabel are timed as the benchmark times them (their own
solve timers, at tolerance 1e-7), and OSQP 1.x at 1e-7 and at 1e-3 (its own solve timer,
warm starting off).
"""

from __future__ import annotations

import argparse
import json
import platform
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
QB = HERE / "third_party" / "qoco-benchmarks"
sys.path.insert(0, str(QB))
sys.path.insert(0, str(HERE.parent))
NATIVE = "-mcpu=native" if platform.machine().lower() in {"arm64", "aarch64"} else "-march=native"
CFLAGS = ["-O2", NATIVE, "-fno-math-errno"]


def cvxpy_problem(horizon: int, data: dict):
  import cvxpy as cp

  from problem import NU, NX, U_MAX, X_MAX, dynamics

  a, b = dynamics()
  q, r = np.diag(data["q"]), np.diag(data["r"])
  u, x = cp.Variable((NU, horizon)), cp.Variable((NX, horizon + 1))
  xlim, ulim = X_MAX * np.ones(NX), U_MAX * np.ones(NU)
  obj, con = 0, [x[:, 0] == data["x0"]]
  for k in range(horizon):  # the benchmark's own formulation, term for term
    obj += cp.quad_form(x[:, k], q) + cp.quad_form(u[:, k], r)
    con += [x[:, k + 1] == a @ x[:, k] + b @ u[:, k]]
    con += [-xlim <= x[:, k], x[:, k] <= xlim]
    con += [-ulim <= u[:, k], u[:, k] <= ulim]
  obj += cp.quad_form(x[:, horizon], q)
  return cp.Problem(cp.Minimize(obj), con)


def text_bytes(path: Path) -> int:
  out = subprocess.run(["size", "-m", str(path)] if sys.platform == "darwin" else ["size", "-A", str(path)], capture_output=True, text=True).stdout
  for line in out.splitlines():
    parts = line.replace(":", " ").split()
    for tag in ("__text", ".text"):
      if tag in parts:
        return int(next(p for p in parts[parts.index(tag) + 1 :] if p.isdigit()))
  return 0


def qocogen_build(horizon: int, data0: dict, work: Path) -> tuple[Path, dict]:
  import qocogen

  from solvers.cvxpy_to_qoco import convert

  prob = cvxpy_problem(horizon, data0)
  n, m, p, P, c, A, b, G, h, n_orthant, nsoc, q = convert(prob)
  work.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  qocogen.generate_solver(n, m, p, P, c, A, b, G, h, n_orthant, nsoc, q, str(work), "qoco_custom")
  t_gen = time.perf_counter() - t0
  solver = work / "qoco_custom"
  sources = sorted(p for p in solver.glob("*.c") if p.name != "runtest.c")
  return solver, {"t_generate": t_gen, "source_bytes": sum(s.stat().st_size for s in sources), "sources": [str(s) for s in sources]}


def qocogen_run(solver: Path, sources: list[str], prob, runs: int) -> dict:
  """The benchmark's runtest (update_data + solve, minimum over runs), compiled with Scaly's flags."""
  from solvers.cvxpy_to_qoco import convert
  from solvers.run_generated_solver import create_qoco_runtest

  _, _, _, P, c, A, b, G, h, _, _, _ = convert(prob)
  runtest = solver / "runtest.c"
  runtest.unlink(missing_ok=True)
  create_qoco_runtest(str(solver), runs, P, A, G, c, b, h)
  exe = solver / "runtest"
  t0 = time.perf_counter()
  subprocess.run(["cc", *CFLAGS, "-o", str(exe), *sources, str(runtest), "-lm"], check=True, capture_output=True, text=True)
  t_cc = time.perf_counter() - t0
  subprocess.run([str(exe)], check=True, capture_output=True, cwd=solver)
  with open(solver / "result.bin", "rb") as f:
    solved = struct.unpack("B", f.read(1))[0]
    iters = struct.unpack("I", f.read(4))[0]
    obj = struct.unpack("d", f.read(8))[0]
    runtime = struct.unpack("d", f.read(8))[0]
  return {
    "status": "solved" if solved == 1 else f"status {solved}",
    "iterations": iters,
    "obj": obj,
    "solve_s": runtime,
    "t_compile": t_cc,
    "text_bytes": text_bytes(exe),
  }


def osqp_solve(prob, tol: float, runs: int) -> dict:
  """OSQP 1.x on CVXPY's own OSQP form of the problem, warm starting off so every solve starts cold,
  polishing off; its own solve timer, the minimum over `runs`."""
  import cvxpy as cp
  import osqp

  from scipy import sparse

  data, _, _ = prob.get_problem_data(cp.OSQP)  # min 1/2 x'Px + q'x, A x = b, F x <= G: OSQP's l <= [A; F] x <= u
  def csc32(m):
    m = sparse.csc_matrix(m)
    m.indices, m.indptr = m.indices.astype(np.int32), m.indptr.astype(np.int32)
    return m

  a = csc32(sparse.vstack([data["A"], data["F"]]))
  lo = np.concatenate([data["b"], np.full(data["G"].size, -np.inf)])
  hi = np.concatenate([data["b"], data["G"]])
  m = osqp.OSQP()
  m.setup(P=csc32(data["P"]), q=data["q"], A=a, l=lo, u=hi, eps_abs=tol, eps_rel=tol, polishing=False, warm_starting=False, verbose=False, max_iter=200000)
  best, res = np.inf, None
  for _ in range(runs):
    res = m.solve()
    best = min(best, res.info.solve_time)
  return {"status": res.info.status, "iters": int(res.info.iter), "solve_time": best, "obj": float(res.info.obj_val)}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--horizon", type=int, required=True)
  ap.add_argument("--instances", type=int, default=5)
  ap.add_argument("--runs", type=int, default=100)
  ap.add_argument("--no-qocogen", action="store_true")
  ap.add_argument("--work", type=Path, required=True)
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()
  from problem import HORIZONS, instances

  import types

  sys.modules.setdefault("gurobipy", types.ModuleType("gurobipy"))  # imported at the top of solvers.py, used only by gurobi_solve
  from solvers.solvers import clarabel_solve, qoco_solve

  horizons = sorted(set(HORIZONS) | {args.horizon})
  every = instances(horizons, args.instances)
  data = {i: every[(args.horizon, i)] for i in range(args.instances)}
  rows = []
  build = None
  if not args.no_qocogen:
    solver, build = qocogen_build(args.horizon, data[0], args.work / f"T{args.horizon}")
  for i, d in data.items():
    prob = cvxpy_problem(args.horizon, d)
    row = {"horizon": args.horizon, "instance": i}
    row["qoco"] = qoco_solve(prob, 1e-7, args.runs)
    row["clarabel"] = clarabel_solve(prob, 1e-7, args.runs)
    row["osqp"] = osqp_solve(prob, 1e-7, args.runs)
    row["osqp_1e-3"] = osqp_solve(prob, 1e-3, args.runs)
    if build is not None:
      row["qocogen"] = qocogen_run(solver, build["sources"], prob, args.runs)
    rows.append(row)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps({"build": build, "rows": rows}, default=float, indent=1))


if __name__ == "__main__":
  main()
