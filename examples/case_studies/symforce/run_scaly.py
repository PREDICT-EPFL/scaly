"""The Scaly side of the SymForce study: one problem size, built, compiled and timed from C.

    uv run examples/case_studies/symforce/run_scaly.py --measurements gen/measurements.cc [--symforce symforce.json] --out result.json

Reads the problem from the `measurements.cc` SymForce's C++ reads (so both sides solve the same numbers),
builds three Functions, and compiles each with the flags Scaly's JIT uses:

- `solve`: SymForce's Levenberg-Marquardt from the identity poses to its stopping test (`lm_function`);
- `iterate`: one call of the same with at most one iteration, at SymForce's solution, which is what
  SymForce's benchmark times (`Optimize(values, 1)` on values its earlier calls have converged);
- `linearize`: error, gradient and Gauss-Newton Hessian at the identity poses (`Relinearize`).

Each is timed by `fatrop_chain/time_kernel.c` (batches of calls of at least 1 ms, the best batch).
With `--symforce`, the solve's per-iteration trace is compared with SymForce's.
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
sys.path.insert(0, str(HERE))


def time_c(
  lib: Path, symbol: str, inputs: list[np.ndarray], outputs: list[int], w: int, work: Path, repeats: int = 400
) -> tuple[float, float, np.ndarray]:
  harness = work / "time_kernel"
  if not harness.exists():
    subprocess.run(["cc", "-O2", "-o", str(harness), str(HERE.parent / "fatrop_chain" / "time_kernel.c")], check=True)
  blob = np.array([len(inputs), len(outputs), max(w, 1), 0, len(inputs), len(outputs), *(x.size for x in inputs), *outputs], dtype=np.int64).tobytes()
  blob += b"".join(np.ascontiguousarray(x, dtype=np.float64).tobytes() for x in inputs)
  (work / "in.bin").write_bytes(blob)
  out = subprocess.run(
    [str(harness), str(lib), symbol, str(work / "in.bin"), str(repeats), str(work / "out.bin")], capture_output=True, text=True, check=True
  )
  best, median = (float(v) * 1e-9 for v in out.stdout.split())
  return best, median, np.frombuffer((work / "out.bin").read_bytes(), dtype=np.float64)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--measurements", type=Path, required=True)
  ap.add_argument("--symforce", type=Path, help="bench_symforce's JSON, for the iterate comparison and the solution")
  ap.add_argument("--per-factor", action="store_true", help="one vmap body per matching factor, not one per pose")
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()
  work = Path(tempfile.mkdtemp(prefix="symforce-scaly-"))
  os.environ.setdefault("SCALY_CACHE_DIR", str(work / "cache"))
  from scaly.codegen import render_c_module
  from scaly.codegen.abi import c_ident
  from scaly.codegen.jit import compile_flags

  import scaly_impl as si

  p = si.parse_measurements(args.measurements)
  n = p.n_poses
  X0 = si.identity_poses(n)
  grouped = not args.per_factor
  row: dict = {"n_poses": n, "n_landmarks": p.n_landmarks, "n_factors": n * p.n_landmarks + n - 1, "grouped": grouped}
  t0 = time.perf_counter()
  tag = f"loc{n}{'' if grouped else '_pf'}"
  fns = {
    "solve": si.lm_function(p, name=tag, grouped=grouped),
    "iterate": si.lm_function(p, max_iterations=1, name=f"{tag}_one", grouped=grouped),
    "linearize": si.linearize_function(p, name=tag, grouped=grouped),
  }
  row["t_build"] = time.perf_counter() - t0
  solution = None
  for key, fn in fns.items():
    t0 = time.perf_counter()
    module = render_c_module(fn)
    t_generate = time.perf_counter() - t0
    src, lib = work / f"{key}.c", work / f"lib{key}.so"
    src.write_text(module.body)
    t0 = time.perf_counter()
    subprocess.run(
      [os.environ.get("SCALY_CC", "cc"), *compile_flags(), "-fPIC", "-shared", str(src), *module.link_flags, "-lm", "-o", str(lib)], check=True
    )
    t_compile = time.perf_counter() - t0
    x = X0 if key != "iterate" else solution
    outputs = [int(np.prod(e.shape)) if e.shape else 1 for e in fn.outputs]
    best, median, out = time_c(lib, c_ident(fn.name), [x], outputs, module.workspace_size, work)
    row[key] = {
      "best_us": best * 1e6,
      "median_us": median * 1e6,
      "t_generate": t_generate,
      "t_compile": t_compile,
      "c_bytes": len(module.body.encode()),
      "workspace_doubles": module.workspace_size,
    }
    if key == "solve":
      values = fn(X0)
      solution = np.asarray(values[0])
      K = values[3].size
      it = int(values[2])
      row["solve"].update(
        iterations=it,
        final_error=float(values[1]),
        trace=[{"lambda": float(values[4][k]), "new_error": float(values[3][k]), "accepted": bool(values[5][k])} for k in range(min(it, K))],
        poses=solution.tolist(),
      )
      if not np.allclose(out[: solution.size], solution, rtol=0, atol=1e-12):
        raise RuntimeError("the C-timed solve returned different poses from the Python call")
  if args.symforce:
    ref = json.loads(args.symforce.read_text())["fixed"]
    trace = row["solve"]["trace"]
    ref_trace = ref["trace"][1:]
    row["vs_symforce"] = {
      "iterations_equal": row["solve"]["iterations"] == ref["iterations"],
      "lambda_equal": all(abs(a["lambda"] - b["lambda"]) <= 1e-12 * b["lambda"] for a, b in zip(trace, ref_trace, strict=False)),
      "accepted_equal": [t["accepted"] for t in trace] == [t["accepted"] for t in ref_trace],
      "max_rel_error_diff": max(abs(a["new_error"] - b["new_error"]) / abs(b["new_error"]) for a, b in zip(trace, ref_trace, strict=False)),
      "max_pose_diff": float(np.abs(solution - np.array(ref["poses"])).max()),
    }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(row, indent=1))
  print(
    json.dumps(
      {k: v for k, v in row.items() if k not in ("solve",)} | {"solve": {k: v for k, v in row["solve"].items() if k not in ("trace", "poses")}}
    )
  )


if __name__ == "__main__":
  main()
