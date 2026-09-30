"""Scaly's dense kernels against a base checkout's and against BLASFEO, natively on the Mac.

    uv run internal/notes/perf_2026_09_30_gaps/kernels.py --base <worktree>/src [--ops gemm,potrf,trsm] [--ns 16,64,128]
    uv run internal/notes/perf_2026_09_30_gaps/kernels.py --base-mode straight --new-mode loops --ns 8,12,16

Per op and n: the kernel rendered from this checkout (``new``) and from ``--base`` (``base``, a
worktree's ``src``), each compiled with the JIT's flags into a shared library, then
``../perf_2026_09_30_blasfeo/bench.c`` (its ``jit`` slot takes ``base``, its ``o3`` slot ``new``)
built with clang against the vendored BLASFEO (``ARMV8A_APPLE_M1``, panel-major) and run: batches of
at least 20 ms per sample, samples interleaved across variants, the fastest of 9 kept, after a
200 ms warm-up. ``--base-mode`` and ``--new-mode`` render either side another way: ``straight``
(every factorization straight-line, ``dense_unroll`` 1 000), ``loops`` (none straight-line, and the
product kept in loops with ``.block()``), or ``auto``, the default. Without ``--base``, both sides
come from this checkout. Output: one table row per kernel, and ``results/kernels_<tag>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BLASFEO = ROOT / "plugins/scaly-piqp/third_party/blasfeo_install/arm64"
BENCH_C = HERE.parent / "perf_2026_09_30_blasfeo" / "bench.c"
BUILD = HERE / "build"

_RENDER = r"""
import json, sys
import scaly as sc
from scaly.codegen import render_c_module
from scaly.linalg import cholesky, solve_triangular
op, m, k, n, out, mode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
a = sc.sym("a", (m, k))
b = sc.sym("b", (k, n))
unroll = {"straight": 1000, "loops": 0}.get(mode)
with sc.options(**({} if unroll is None else {"linalg": {"dense_unroll": unroll}})):
  if op == "gemm":
    ins, names, res = [a, b], ["a", "b"], (a @ b).block() if mode == "loops" else a @ b
  elif op == "potrf":
    ins, names, res = [a], ["a"], cholesky(a)
  elif op == "trsm":
    ins, names, res = [a, b], ["a", "b"], solve_triangular(a, b, lower=True)
  else:
    raise SystemExit(op)
symbol = f"bf_{op}_{n}" if m == k == n else f"kn_{op}_{m}_{k}_{n}"
fn = sc.Function.from_exprs(symbol, ins, [res], names, ["o"])
module = render_c_module(fn)
open(out, "w").write(module.body)
print(json.dumps({"symbol": symbol, "workspace": int(module.workspace_size), "c_bytes": len(module.body)}))
"""


def _flags() -> list[str]:
  sys.path.insert(0, str(ROOT / "src"))
  from scaly.codegen.jit import compile_flags

  return list(compile_flags())


def render(src: Path | None, op: str, m: int, k: int, n: int, out: Path, mode: str = "auto") -> dict:
  env = {**os.environ}
  if src is not None:
    env["PYTHONPATH"] = str(src)
  done = subprocess.run([sys.executable, "-c", _RENDER, op, str(m), str(k), str(n), str(out), mode], env=env, capture_output=True, text=True, check=True)
  return json.loads(done.stdout.strip().splitlines()[-1])


def compile_so(c: Path, flags: list[str]) -> Path:
  lib = c.with_suffix(".so")
  subprocess.run(["cc", *flags, "-fPIC", "-shared", str(c), "-lm", "-o", str(lib)], check=True)
  return lib


def build_bench() -> Path:
  exe = BUILD / "bench_blasfeo"
  if not exe.exists() or exe.stat().st_mtime < BENCH_C.stat().st_mtime:
    BUILD.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cc", "-O2", "-DBENCH_NO_BLAS_API", f"-I{BLASFEO / 'include'}", str(BENCH_C), str(BLASFEO / "lib" / "libblasfeo.a"), "-lm", "-o", str(exe)], check=True)
  return exe


def warm_up() -> None:
  """Spin the core for 200 ms so its clock has ramped before the first sample."""
  import time

  t = time.perf_counter()
  x = 0.0
  while time.perf_counter() - t < 0.2:
    x += 1.0


def square(args, flags: list[str]) -> list[dict]:
  exe = build_bench()
  rows = []
  for op in args.ops.split(","):
    for n in (int(v) for v in args.ns.split(",")):
      with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        meta = {}
        for tag, src, mode in (("base", args.base, args.base_mode), ("new", None, args.new_mode)):
          meta[tag] = render(src, op, n, n, n, d / f"{tag}.c", mode)
        libs = {tag: compile_so(d / f"{tag}.c", flags) for tag in ("base", "new")}
        warm_up()
        ws = max(meta["base"]["workspace"], meta["new"]["workspace"])
        out = subprocess.run([str(exe), op, str(n), str(libs["base"]), str(libs["new"]), meta["new"]["symbol"], str(ws), "20", "9"], capture_output=True, text=True, check=True).stdout
      times = {}
      for line in out.splitlines():
        if line.startswith("#"):
          continue
        _, name, _, ns, gf, diff = line.split()
        times[name] = {"ns": float(ns), "gflops": float(gf), "diff": float(diff)}
      best_bf = min(v["ns"] for name, v in times.items() if name.startswith("bf_") and "blas" not in name and "lapack" not in name and "llnn" not in name)
      row = {"op": op, "n": n, "base_ns": times["scaly_jit"]["ns"], "new_ns": times["scaly_o3"]["ns"], "blasfeo_ns": best_bf,
             "new_over_base": times["scaly_o3"]["ns"] / times["scaly_jit"]["ns"], "base_over_bf": times["scaly_jit"]["ns"] / best_bf,
             "new_over_bf": times["scaly_o3"]["ns"] / best_bf, "new_gflops": times["scaly_o3"]["gflops"], "diff_new_base": times["scaly_o3"]["diff"],
             "c_bytes_base": meta["base"]["c_bytes"], "c_bytes_new": meta["new"]["c_bytes"]}
      rows.append(row)
      print(f"{op:5} n={n:4d}  base {row['base_ns']:10.1f} ns  new {row['new_ns']:10.1f} ns  new/base {row['new_over_base']:5.2f}  "
            f"base/BF {row['base_over_bf']:5.2f}  new/BF {row['new_over_bf']:5.2f}  {row['new_gflops']:5.1f} GF/s  |new-base| {row['diff_new_base']:.1e}", flush=True)
  return rows


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--base", type=Path, default=None, help="the base checkout's src directory (default: this one)")
  parser.add_argument("--base-mode", default="auto", choices=("auto", "straight", "loops"))
  parser.add_argument("--new-mode", default="auto", choices=("auto", "straight", "loops"))
  parser.add_argument("--ops", default="gemm,potrf,trsm")
  parser.add_argument("--ns", default="8,12,16,24,32,48,64,96,128")
  parser.add_argument("--tag", default="run")
  args = parser.parse_args()
  flags = _flags()
  rows = square(args, flags)
  (HERE / "results").mkdir(exist_ok=True)
  (HERE / "results" / f"kernels_{args.tag}.json").write_text(json.dumps({"flags": flags, "rows": rows}, indent=1))


if __name__ == "__main__":
  main()
