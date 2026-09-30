"""Time small dense kernels of any shape, rendered two or more ways, from C.

    uv run internal/notes/perf_2026_09_30_gaps/shapes.py --op trsm --shapes 8,1 16,1 32,1 --modes straight,loops
    uv run internal/notes/perf_2026_09_30_gaps/shapes.py --op gemm --shapes 12,12,12 96,96,96 --modes auto --base <worktree>/src

Ops and their shapes: ``gemm m,k,n`` (``a @ b``), ``potrf n`` (``cholesky``), ``trsm n,r`` (lower
``solve_triangular`` with ``r`` right-hand sides, a vector for ``r = 0``), ``cho_solve n`` (the
factor and both triangular solves of a vector, as the dense IPM step does). Modes as in
``kernels.py``: ``auto``, ``straight`` (every factorization straight-line), ``loops`` (none, and a
product kept in loops). ``--base`` renders the modes from another checkout's ``src`` as well, each
tagged ``base:<mode>``. Each variant is compiled with the JIT's flags and timed by
``../perf_2026_09_30_codegen/time_entry.c``: rounds interleaved across variants, the fastest sample
kept, after a 200 ms warm-up. Prints one row per shape, times over the first variant's, and the
largest difference from the first variant's outputs.
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
ROOT = HERE.parents[2]
DRIVER_SRC = HERE.parent / "perf_2026_09_30_codegen" / "time_entry.c"
BUILD = HERE / "build"

_RENDER = r"""
import json, sys
import numpy as np
import scaly as sc
from scaly.codegen import render_c_module
from scaly.linalg import cho_solve, cholesky, solve_triangular
op, dims, mode, out = sys.argv[1], [int(v) for v in sys.argv[2].split(",")], sys.argv[3], sys.argv[4]
unroll = {"straight": 1000, "loops": 0}.get(mode)
with sc.options(**({} if unroll is None else {"linalg": {"dense_unroll": unroll}})):
  if op == "gemm":
    m, k, n = dims
    a, b = sc.sym("a", (m, k)), sc.sym("b", (k, n))
    ins, res = [a, b], a @ b
    res = res.block() if mode == "loops" else res
  elif op == "potrf":
    a = sc.sym("a", (dims[0], dims[0]))
    ins, res = [a], cholesky(a)
  elif op == "trsm":
    n, r = dims
    a, b = sc.sym("a", (n, n)), sc.sym("b", (n, r) if r else (n,))
    ins, res = [a, b], solve_triangular(a, b, lower=True)
  elif op == "cho_solve":
    n = dims[0]
    a, b = sc.sym("a", (n, n)), sc.sym("b", n)
    ins, res = [a, b], cho_solve(cholesky(a), b)
  else:
    raise SystemExit(op)
name = "sh_" + op + "_" + "_".join(map(str, dims))
fn = sc.Function.from_exprs(name, ins, [res], [f"i{j}" for j in range(len(ins))], ["o"])
module = render_c_module(fn)
open(out, "w").write(module.body)
print(json.dumps({"symbol": name, "workspace": int(module.workspace_size), "inputs": [list(x.shape) for x in ins], "output": int(fn.outputs[0].size)}))
"""


def driver() -> Path:
  exe = BUILD / "time_entry"
  if not exe.exists() or exe.stat().st_mtime < DRIVER_SRC.stat().st_mtime:
    BUILD.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cc", "-O2", str(DRIVER_SRC), "-o", str(exe)], check=True)
  return exe


def inputs(op: str, shapes: list[list[int]], rng: np.random.Generator) -> list[np.ndarray]:
  if op == "gemm":
    return [rng.standard_normal(s) for s in shapes]
  n = shapes[0][0]
  m = rng.standard_normal((n, n))
  if op in ("potrf", "cho_solve"):
    spd = m @ m.T + n * np.eye(n)
    return [spd] + [rng.standard_normal(s) for s in shapes[1:]]
  low = np.tril(m) / n + np.diag(1.0 + 0.5 * rng.random(n))
  return [low, rng.standard_normal(shapes[1])]


def blob(flat: list[np.ndarray], out_size: int, workspace: int) -> bytes:
  head = np.array([len(flat), 1, workspace, *(a.size for a in flat), out_size], dtype="<i8").tobytes()
  return head + b"".join(np.ascontiguousarray(a, dtype="<f8").tobytes() for a in flat)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--op", required=True, choices=("gemm", "potrf", "trsm", "cho_solve"))
  parser.add_argument("--shapes", nargs="+", required=True)
  parser.add_argument("--modes", default="auto")
  parser.add_argument("--base", type=Path, default=None)
  parser.add_argument("--rounds", type=int, default=7)
  parser.add_argument("--budget", type=float, default=0.05, help="seconds per sample")
  parser.add_argument("--tag", default=None)
  args = parser.parse_args()
  sys.path.insert(0, str(ROOT / "src"))
  from scaly.codegen.jit import compile_flags

  flags = list(compile_flags())
  exe = driver()
  modes = args.modes.split(",")
  variants = [(None, m, m) for m in modes] + ([(args.base, m, f"base:{m}") for m in modes] if args.base else [])
  rows = []
  for shape in args.shapes:
    dims = [int(v) for v in shape.split(",")]
    with tempfile.TemporaryDirectory() as tmp:
      d = Path(tmp)
      cells = []
      for src, mode, label in variants:
        env = {**os.environ, **({"PYTHONPATH": str(src)} if src else {})}
        c = d / f"{label.replace(':', '_')}.c"
        meta = json.loads(subprocess.run([sys.executable, "-c", _RENDER, args.op, shape, mode, str(c)], env=env, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
        lib = c.with_suffix(".so")
        subprocess.run(["cc", *flags, "-fPIC", "-shared", str(c), "-lm", "-o", str(lib)], check=True)
        cells.append((label, lib, meta))
      meta0 = cells[0][2]
      data = inputs(args.op, meta0["inputs"], np.random.default_rng(7))
      bin_in = d / "inputs.bin"
      bin_in.write_bytes(blob([np.ravel(a) for a in data], meta0["output"], max(m["workspace"] for _, _, m in cells)))
      # the batch: enough calls to fill the budget, from one calibration call
      best = {label: float("inf") for label, _, _ in cells}
      outs = {}
      t = time.perf_counter()
      while time.perf_counter() - t < 0.2:
        subprocess.run([str(exe), str(cells[0][1]), meta0["symbol"], str(bin_in), "20", "/dev/null"], capture_output=True, check=True)
      batches = {}
      for label, lib, meta in cells:
        probe = subprocess.run([str(exe), str(lib), meta["symbol"], str(bin_in), "5", "/dev/null", "10"], capture_output=True, text=True, check=True)
        per = float(probe.stdout.split()[0])
        batches[label] = max(1, int(args.budget * 1e9 / max(per, 1.0) / 20))
      for _ in range(args.rounds):
        for label, lib, meta in cells:
          o = d / f"{label.replace(':', '_')}.out"
          r = subprocess.run([str(exe), str(lib), meta["symbol"], str(bin_in), "20", str(o), str(batches[label])], capture_output=True, text=True, check=True)
          best[label] = min(best[label], float(r.stdout.split()[0]))
          outs[label] = np.fromfile(o)
    first = cells[0][0]
    row = {"shape": shape, **{label: best[label] for label in best}, "diff": max(float(np.abs(outs[label] - outs[first]).max()) for label in outs)}
    rows.append(row)
    ratios = "  ".join(f"{label} {best[label]:9.1f} ns ({best[label] / best[first]:4.2f})" for label in best)
    print(f"{args.op:9} {shape:>12}  {ratios}  |diff| {row['diff']:.1e}", flush=True)
  if args.tag:
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"shapes_{args.tag}.json").write_text(json.dumps({"op": args.op, "flags": flags, "rows": rows}, indent=1))


if __name__ == "__main__":
  main()
