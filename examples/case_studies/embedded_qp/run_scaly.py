"""The Scaly side of the embedded-QP study: the generated PIQP and the PIQP library, one horizon per process.

    uv run examples/case_studies/embedded_qp/run_scaly.py --horizon 20 --solver scaly_sparse --out results/scaly_T20_scaly_sparse.json

Solvers: `scaly_sparse` is PIQP's algorithm generated for this problem's KKT pattern,
`sc.opt.solver(problem, sc.opt.IPM(sparse=True))`; `piqp_sparse` is the vendored PIQP 0.6.2 library behind
`sc.opt.solver(problem, sc.opt.PIQP(sparse=True))`. Both at `eps_abs = eps_rel = 1e-7`, the
benchmark's tolerance, and both take a warm start (which neither uses) and the parameters, and return
the solution, the multipliers and an `Info`. Timing is `examples/opt/qp_solvers`' harness: the generated
entry point called from C (`time_entry.c`) on the instance's parameters, which include the data the
QOCO timer rewrites (`Q`, `R`, `x0`); the minimum over `--repeats` calls.
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
sys.path.insert(0, str(HERE.parents[1] / "opt" / "qp_solvers"))  # compare.py's timing harness
TOL = 1e-7


def build(problem, solver: str):
  import scaly as sc

  options = {"eps_abs": TOL, "eps_rel": TOL}
  method = sc.opt.IPM(sparse=True, options=options) if solver == "scaly_sparse" else sc.opt.PIQP(sparse=True, options=options)
  return sc.opt.solver(problem, method, name=f"{problem.name}_{solver}")


def main() -> None:
  import scaly as sc

  ap = argparse.ArgumentParser()
  ap.add_argument("--horizon", type=int, required=True)
  ap.add_argument("--solver", choices=("scaly_sparse", "piqp_sparse"), required=True)
  ap.add_argument("--instances", type=int, default=5)
  ap.add_argument("--repeats", type=int, default=200)
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()
  work = Path(tempfile.mkdtemp(prefix="embedded-qp-"))
  os.environ.setdefault("SCALY_CACHE_DIR", str(work / "cache"))
  from scaly.codegen import render_c_module
  from scaly.codegen.abi import c_ident
  from scaly.codegen.jit import compile_flags

  import compare as qp  # examples/opt/qp_solvers/compare.py: sh, object_size, time_driver
  from problem import HORIZONS, instances, objective, problem

  t0 = time.perf_counter()
  p = problem(args.horizon)
  fun = build(p, args.solver)
  t_build = time.perf_counter() - t0
  t0 = time.perf_counter()
  module = render_c_module(fun)
  t_generate = time.perf_counter() - t0
  src, lib = work / f"{c_ident(fun.name)}.c", work / f"lib{c_ident(fun.name)}.so"
  src.write_text(module.body)
  t0 = time.perf_counter()
  qp.sh([os.environ.get("SCALY_CC", "cc"), *compile_flags(), "-fPIC", "-shared", str(src), *module.link_flags, "-lm", "-o", str(lib)])
  t_compile = time.perf_counter() - t0
  row = {
    "horizon": args.horizon,
    "solver": args.solver,
    "t_build": t_build,
    "t_generate": t_generate,
    "t_compile": t_compile,
    "c_bytes": len(module.body.encode()),
    "object_bytes": qp.object_size(lib),
    "text_bytes": text_bytes(lib),
    "instances": [],
  }
  every = instances(sorted(set(HORIZONS) | {args.horizon}), args.instances)
  zeros = p.vars.unflatten(tuple(np.zeros(s) for s in p.vars.shapes))
  for i in range(args.instances):
    d = every[(args.horizon, i)]
    args_ = (zeros, zeros, np.zeros(p.n_eq), np.zeros(p.n_ineq), (d["q"], d["r"], d["x0"]))
    (u, xs), *_, info = fun(*args_)
    status, iters = sc.Status(int(info.status)), int(info.iter)
    flat = [np.asarray(a, float) for a in fun.input_tree.flatten_numerical(args_, "solver inputs")]
    sizes_out = [int(e.size) for e in fun.outputs]
    blob = np.array([len(flat), len(sizes_out), module.workspace_size, *[a.size for a in flat], *sizes_out], dtype="<i8").tobytes()
    blob += b"".join(np.ascontiguousarray(a, dtype="<f8").tobytes() for a in flat)
    (work / "inputs.bin").write_bytes(blob)
    best, median = qp.sh(
      [str(qp.time_driver()), str(lib), c_ident(fun.name), str(work / "inputs.bin"), str(args.repeats), str(work / "outputs.bin")]
    ).split()
    row["instances"].append(
      {
        "instance": i,
        "status": "solved" if status == sc.Status.OK else status.name,
        "iterations": iters,
        "obj": objective(args.horizon, d, np.asarray(u), np.asarray(xs)),
        "solve_s": float(best) * 1e-9,
        "solve_median_s": float(median) * 1e-9,
      }
    )
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(row, indent=1))


def text_bytes(path: Path) -> int:
  import subprocess

  out = subprocess.run(["size", "-m", str(path)] if sys.platform == "darwin" else ["size", "-A", str(path)], capture_output=True, text=True).stdout
  for line in out.splitlines():
    parts = line.replace(":", " ").split()
    for tag in ("__text", ".text"):
      if tag in parts:
        return int(next(q for q in parts[parts.index(tag) + 1 :] if q.isdigit()))
  return 0


if __name__ == "__main__":
  main()
