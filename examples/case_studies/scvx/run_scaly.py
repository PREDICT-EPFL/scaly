"""The Scaly side of the SCvx study: the whole PTR as one Function, from a fresh process.

    uv run examples/case_studies/scvx/run_scaly.py --cache <dir> --out result.json

The user's path is timed as OpenSCvx's is: the import of Scaly, building the Function (`ptr_function`,
which builds the generated QP solver too), and the first call, which renders the C, compiles it and
solves (with `<dir>` as `SCALY_CACHE_DIR`: empty for a cold start; reused, the compiled library is
loaded instead). Then the solve alone, through the Python call (best of 50) and from C
(`fatrop_chain/time_kernel.c`, the best 1 ms batch), the solve's two parts from C (one discretization and
one QP solve, at OpenSCvx's first iteration), and the answer against OpenSCvx's recorded one.
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

HERE = Path(__file__).resolve().parent


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--cache", type=Path, required=True)
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()
  os.environ["SCALY_CACHE_DIR"] = str(args.cache)
  t0 = time.perf_counter()
  import numpy as np

  import scaly  # noqa: F401

  t_import = time.perf_counter() - t0
  sys.path.insert(0, str(HERE))
  import scaly_impl as si

  ref = json.loads((HERE / "results" / "reference_6dof.json").read_text())
  d = si.Data(ref)
  X0, U0 = np.ravel(d.X0), np.ravel(d.U0)
  t0 = time.perf_counter()
  ptr = si.ptr_function(d)
  t_build = time.perf_counter() - t0
  t0 = time.perf_counter()
  X, U, it, jtr, jvc = ptr(X0, U0)
  t_first = time.perf_counter() - t0
  samples = []
  for _ in range(50):
    t = time.perf_counter()
    ptr(X0, U0)
    samples.append(time.perf_counter() - t)
  row = {
    "t_import": t_import,
    "t_build": t_build,
    "t_first_call": t_first,
    "solve_python_s": min(samples),
    "iterations": int(it),
    "J_tr": np.asarray(jtr)[: int(it)].tolist(),
    "J_vc": np.asarray(jvc)[: int(it)].tolist(),
    "final_mass": float(np.asarray(X)[-1, 0]),
    "t_f": float(np.asarray(X)[-1, 14]),
    "max_dX_vs_openscvx": float(np.abs(np.asarray(X) - np.array(ref["final"]["X"])).max()),
    "max_dU_vs_openscvx": float(np.abs(np.asarray(U) - np.array(ref["final"]["U"])).max()),
    "X": np.asarray(X).tolist(),
    "U": np.asarray(U).tolist(),
  }
  # The solve from C, and its two parts at OpenSCvx's first iteration: one discretization, one QP solve.
  work = Path(tempfile.mkdtemp(prefix="scvx-"))
  harness = work / "time_kernel"
  subprocess.run(["cc", "-O2", "-o", str(harness), str(HERE.parent / "fatrop_chain" / "time_kernel.c")], check=True)
  best, median, c_bytes, ws = time_from_c(ptr, [X0, U0], work, harness)
  row.update(solve_c_s=best, solve_c_median_s=median, c_bytes=c_bytes, workspace_doubles=ws)
  disc = si.discretize_function()
  xp, A, B, C = (np.asarray(a) for a in disc(X0, U0))
  row["discretize_c_s"] = time_from_c(disc, [X0, U0], work, harness)[0]
  row["qp_c_s"] = time_from_c(si.qp_function(d), [*si.qp_warm_start(d), xp, A, B, C, X0, U0], work, harness)[0]
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(row))
  print(json.dumps({k: v for k, v in row.items() if k not in ("X", "U")}))


def time_from_c(fn, args: list, work: Path, harness: Path) -> tuple[float, float, int, int]:
  """Best and median of `fn` from C (`fatrop_chain/time_kernel.c`, 1 ms batches), on a library rendered
  and compiled here for the purpose; and the C's size and workspace."""
  import numpy as np

  from scaly.codegen import render_c_module
  from scaly.codegen.abi import c_ident
  from scaly.codegen.jit import compile_flags

  module = render_c_module(fn)
  src, lib = work / f"{fn.name}.c", work / f"lib{fn.name}.so"
  src.write_text(module.body)
  subprocess.run(
    [os.environ.get("SCALY_CC", "cc"), *compile_flags(), "-fPIC", "-shared", str(src), *module.link_flags, "-lm", "-o", str(lib)], check=True
  )
  args = [np.ravel(np.asarray(a, dtype=np.float64)) for a in args]
  outputs = [int(np.prod(e.shape)) if e.shape else 1 for e in fn.outputs]
  header = [len(args), len(outputs), max(module.workspace_size, 1), 0, len(args), len(outputs), *(a.size for a in args), *outputs]
  blob = np.array(header, dtype=np.int64).tobytes() + b"".join(a.astype("<f8").tobytes() for a in args)
  (work / "in.bin").write_bytes(blob)
  out = subprocess.run(
    [str(harness), str(lib), c_ident(fn.name), str(work / "in.bin"), "200", str(work / "out.bin")], capture_output=True, text=True, check=True
  )
  best, median = (float(v) * 1e-9 for v in out.stdout.split())
  return best, median, len(module.body.encode()), module.workspace_size


if __name__ == "__main__":
  main()
