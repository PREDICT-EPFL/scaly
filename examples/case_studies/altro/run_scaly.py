"""The Scaly side of the ALTRO study: build the AL-iLQR for one problem, generate and compile it, time it in C.

    uv run examples/case_studies/altro/run_scaly.py --problem cartpole [--projected-newton] --out results/scaly_cartpole.json

The solve is timed by `examples/opt/qp_solvers/time_entry.c`, which calls the generated entry point with no
Python in the loop, `--repeats` times on the same initial state: every call is a cold solve from the
problem's initial controls, as each Altro.jl sample is. The C is compiled with the flags Scaly's JIT
uses (`-O2 -mcpu=native -fno-math-errno`), and building, generating and compiling are timed apart.
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
sys.path.insert(0, str(HERE.parents[1] / "opt" / "qp_solvers"))


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--problem", choices=("parallel_park", "cartpole"), required=True)
  ap.add_argument("--tolerance", type=float, default=1e-6)
  ap.add_argument("--projected-newton", action="store_true", help="ALTRO: the augmented Lagrangian to 1e-4, then the projection")
  ap.add_argument("--repeats", type=int, default=200)
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()
  work = Path(tempfile.mkdtemp(prefix="altro-"))
  os.environ.setdefault("SCALY_CACHE_DIR", str(work / "cache"))
  t0 = time.perf_counter()
  import scaly  # noqa: F401  (timed: the counterpart of Julia's package load)

  t_import = time.perf_counter() - t0
  from scaly.codegen import render_c_module
  from scaly.codegen.abi import c_ident
  from scaly.codegen.jit import compile_flags

  import compare as qp  # examples/opt/qp_solvers/compare.py: sh, object_size, time_driver
  from scaly_impl import build

  t0 = time.perf_counter()
  ocp, fun = build(args.problem, args.tolerance, args.projected_newton)
  t_build = time.perf_counter() - t0
  t0 = time.perf_counter()
  module = render_c_module(fun)
  t_generate = time.perf_counter() - t0
  src, lib = work / f"{c_ident(fun.name)}.c", work / f"lib{c_ident(fun.name)}.so"
  src.write_text(module.body)
  t0 = time.perf_counter()
  qp.sh([os.environ.get("SCALY_CC", "cc"), *compile_flags(), "-fPIC", "-shared", str(src), *module.link_flags, "-lm", "-o", str(lib)])
  t_compile = time.perf_counter() - t0

  us, xs, cost, outer, inner, violation, projections, projection_steps = fun(ocp.x0)
  sizes_out = [int(e.size) for e in fun.outputs]
  blob = np.array([1, len(sizes_out), module.workspace_size, ocp.nx, *sizes_out], dtype="<i8").tobytes()
  blob += np.ascontiguousarray(ocp.x0, dtype="<f8").tobytes()
  (work / "inputs.bin").write_bytes(blob)
  best, median = qp.sh(
    [str(qp.time_driver()), str(lib), c_ident(fun.name), str(work / "inputs.bin"), str(args.repeats), str(work / "outputs.bin")]
  ).split()
  c_out = np.frombuffer((work / "outputs.bin").read_bytes(), dtype="<f8")
  if not np.array_equal(c_out[: us.size], np.asarray(us).reshape(-1)):
    raise RuntimeError("the C-timed solve returned different controls from the Python call")
  row = {
    "problem": args.problem,
    "projected_newton": args.projected_newton,
    "constraint_tolerance": args.tolerance,
    "iterations": int(inner) + int(projections),  # Altro.jl's count: iLQR iterations and projections
    "iterations_ilqr": int(inner),
    "iterations_outer": int(outer),
    "projections": int(projections),
    "projection_steps": int(projection_steps),
    "cost": float(cost),
    "max_violation": float(violation),
    "time_min": float(best) * 1e-9,
    "time_median": float(median) * 1e-9,
    "t_import": t_import,
    "t_build": t_build,
    "t_generate": t_generate,
    "t_compile": t_compile,
    "c_bytes": len(module.body.encode()),
    "object_bytes": qp.object_size(lib),
    "workspace_doubles": module.workspace_size,
    "X": np.asarray(xs).tolist(),
    "U": np.asarray(us).tolist(),
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(row, indent=1))
  print(json.dumps({k: v for k, v in row.items() if k not in ("X", "U")}))


if __name__ == "__main__":
  main()
