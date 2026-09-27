"""I2: the generated dense LU solve against LAPACK's dgesv, both timed in C.

    uv run internal/notes/perf_2026_09_27_integrators/bench_lu.py [--rounds 7]

Per order n: `linalg.solve(a, b, assume="gen")` as a scaly Function, compiled with the JIT's flags,
and a shim with the same universal entry that copies the matrix (dgesv overwrites it) and calls
LAPACK `dgesv_` from Apple's Accelerate, which is handed the matrix in column-major order so neither
side transposes. Both run through `time_entry.c`, rounds interleaved, fastest sample kept, and the
two solutions are compared first. Also reported: the unrolled or looped form, and the C lines.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUILD = HERE / "build" / "lu"
SIZES = (4, 8, 12, 24, 40)
SHIM = r"""
#include <string.h>
extern void dgesv_(int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
int lapack_solve(const double **arg, double **res, int *iw, double *w, int mem) {
  int n = N, one = 1, info, ipiv[N];
  memcpy(w, arg[0], sizeof(double) * N * N);
  memcpy(res[0], arg[1], sizeof(double) * N);
  dgesv_(&n, &one, w, &n, ipiv, res[0], &n, &info);
  (void)iw; (void)mem;
  return info;
}
"""


def blob(flat: list[np.ndarray], out_sizes: list[int], workspace: int) -> bytes:
  head = np.array([len(flat), len(out_sizes), workspace, *(a.size for a in flat), *out_sizes], dtype="<i8").tobytes()
  return head + b"".join(np.ascontiguousarray(a, dtype="<f8").tobytes() for a in flat)


def build(n: int) -> dict:
  import scaly as sc
  from scaly import linalg
  from scaly.codegen import render_c_module
  from scaly.codegen.jit import compile_flags
  from scaly.codegen.toolchain import find_c_compiler

  rng = np.random.default_rng(n)
  a, b = rng.standard_normal((n, n)), rng.standard_normal(n)
  a[0, 0] = 0.0

  @sc.function((n, n), n, output="x", name=f"gen_solve{n}")
  def gen(m, v):
    return linalg.solve(m, v, assume="gen")

  cc, flags = find_c_compiler().cc, compile_flags()
  scaly_dir, lapack_dir = BUILD / f"scaly_{n}", BUILD / f"lapack_{n}"
  for d in (scaly_dir, lapack_dir):
    d.mkdir(parents=True, exist_ok=True)
  module = render_c_module(gen.concrete)
  (scaly_dir / "f.c").write_text(module.body)
  subprocess.run([cc, *flags, "-fPIC", "-shared", str(scaly_dir / "f.c"), "-lm", "-o", str(scaly_dir / "lib.so")], check=True)
  (scaly_dir / "inputs.bin").write_bytes(blob([a, b], [n], int(module.workspace_size)))
  (lapack_dir / "f.c").write_text(SHIM)
  subprocess.run([cc, *flags, f"-DN={n}", "-fPIC", "-shared", str(lapack_dir / "f.c"), "-framework", "Accelerate", "-o", str(lapack_dir / "lib.so")], check=True)
  (lapack_dir / "inputs.bin").write_bytes(blob([a.T, b], [n], n * n))
  lines = sum(1 for line in module.body.splitlines() if line.strip())
  return {"expected": np.linalg.solve(a, b), "unrolled": n <= sc.get_options().dense_unroll, "c_lines": lines}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--rounds", type=int, default=7)
  args = parser.parse_args()
  sys.path.insert(0, str(HERE))
  from bench_explicit import driver

  metas = {n: build(n) for n in SIZES}
  best: dict[tuple[str, int], float] = {}
  for _ in range(args.rounds):
    for n in SIZES:
      inner = max(1, 20000 // (n * n))
      for side, symbol in (("scaly", f"gen_solve{n}"), ("lapack", "lapack_solve")):
        cell = BUILD / f"{side}_{n}"
        cmd = [str(driver()), str(cell / "lib.so"), symbol, str(cell / "inputs.bin"), "300", str(cell / "outputs.bin"), str(inner)]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        x = np.fromfile(cell / "outputs.bin", dtype="<f8")
        assert np.allclose(x, metas[n]["expected"], rtol=1e-9, atol=1e-10), (side, n)
        best[side, n] = min(best.get((side, n), np.inf), float(out.stdout.split()[0]))
  print("| n | form | scaly ns | LAPACK dgesv ns | scaly/LAPACK | C lines |")
  print("| --- | --- | --- | --- | --- | --- |")
  for n in SIZES:
    s, lp = best["scaly", n], best["lapack", n]
    print(f"| {n} | {'unrolled' if metas[n]['unrolled'] else 'loops'} | {s:.1f} | {lp:.1f} | {s / lp:.2f} | {metas[n]['c_lines']} |")


if __name__ == "__main__":
  sys.path.insert(0, str(ROOT))
  main()
