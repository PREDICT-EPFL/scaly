"""The LDL^T microbenchmark of the QOCO paper (Appendix B, Table 3), with Scaly's generated factorization added.

    uv run --with qocogen examples/case_studies/embedded_qp/ldl_bench.py --out examples/case_studies/embedded_qp/results/ldl.json

The matrix is the paper's LQR KKT system, rebuilt from its description because the script is not
public: 6 states, 3 inputs, horizon T, random `A` and `B`, `Q = R = I`, primal order all states then
all inputs, `K = [P H'; H -eps I]` with the dynamics rows `[A -I ... B]` followed by the initial-state
row `[I 0 ...]` (order `15 T - 3`). Three factorizations of it, numeric factorization only (the
paper times that, with `perf`, on a Ryzen 7950X3D):

  qocogen  QOCOGEN's generated, fully unrolled `ldl()` for this problem (the equality-constrained QP
           whose KKT matrix this is), timed from C after `load_data`
  qdldl    QOCO's vendored QDLDL, `QDLDL_factor` on the same matrix permuted by the AMD ordering
           QOCOGEN uses (qdldl-python's), elimination tree computed outside the timer
  scaly    `sc.linalg.SparseLDL` of the same matrix, generated and compiled: with Scaly's own ordering
           choice and its default schedule (straight-line below `sparse_unroll`, loops over tables
           above), with QOCOGEN's AMD permutation, and forced straight-line (`schedule="unroll"`, the
           counterpart of QOCOGEN's code) up to `--unroll-limit` multiply-adds

All compiled with `-O2 -mcpu=native -fno-math-errno`; each timing is the best 1 ms batch of calls over
200 batches, and the best of three runs of the timing binary.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
NATIVE = "-mcpu=native" if platform.machine().lower() in {"arm64", "aarch64"} else "-march=native"
CFLAGS = ["-O2", NATIVE, "-fno-math-errno"]
QDLDL = HERE / "baseline" / "third_party" / "qoco" / "lib" / "qdldl"
NX, NU, EPS = 6, 3, 1e-7
HORIZONS = [5, 15, 50, 75, 100]
PROCESSES = 3  # each timing binary runs this many times; the best is kept
PAPER = {5: (2.103, 0.435), 15: (7.507, 1.777), 50: (27.077, 6.117), 75: (38.848, 8.485), 100: (49.577, 12.894)}  # µs, qdldl / custom


def lqr(horizon: int, seed: int = 0) -> tuple[sparse.csc_array, sparse.csc_array]:
  """`P` and `H` of the paper's LQR QP."""
  rng = np.random.default_rng(seed)
  a, b = rng.standard_normal((NX, NX)), rng.standard_normal((NX, NU))
  n = NX * horizon + NU * (horizon - 1)
  rows = []
  for k in range(horizon - 1):
    row = sparse.lil_array((NX, n))
    row[:, k * NX : (k + 1) * NX] = a
    row[:, (k + 1) * NX : (k + 2) * NX] = -np.eye(NX)
    row[:, NX * horizon + k * NU : NX * horizon + (k + 1) * NU] = b
    rows.append(row)
  first = sparse.lil_array((NX, n))
  first[:, :NX] = np.eye(NX)
  rows.append(first)
  return sparse.csc_array(sparse.identity(n)), sparse.csc_array(sparse.vstack(rows))


def kkt_upper(p: sparse.csc_array, h: sparse.csc_array) -> sparse.csc_array:
  k = sparse.block_array([[p, h.T], [h, -EPS * sparse.identity(h.shape[0])]], format="csc")
  return sparse.csc_array(sparse.triu(k, format="csc"))


def text_bytes(path: Path) -> int:
  out = subprocess.run(["size", "-m", str(path)] if sys.platform == "darwin" else ["size", "-A", str(path)], capture_output=True, text=True).stdout
  for line in out.splitlines():
    parts = line.replace(":", " ").split()
    for tag in ("__text", ".text"):
      if tag in parts:
        return int(next(q for q in parts[parts.index(tag) + 1 :] if q.isdigit()))
  return 0


LOOP = r"""
static double now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return 1e9 * t.tv_sec + t.tv_nsec; }
#define TIME_BEST(stmt, out) do { \
  for (int w_ = 0; w_ < 10; ++w_) { stmt; } \
  double t0_ = now_ns(); for (int w_ = 0; w_ < 64; ++w_) { stmt; } \
  double per_ = (now_ns() - t0_) / 64.0; long batch_ = per_ > 1e6 ? 1 : (long)(1e6 / (per_ > 1 ? per_ : 1)) + 1; \
  out = 1e300; \
  for (int r_ = 0; r_ < 200; ++r_) { double t1_ = now_ns(); for (long b_ = 0; b_ < batch_; ++b_) { stmt; } \
    double v_ = (now_ns() - t1_) / batch_; if (v_ < out) out = v_; } \
} while (0)
"""


def run_qocogen(horizon: int, p, h, work: Path, budget: float) -> dict:
  import qocogen

  n, neq = p.shape[0], h.shape[0]
  t0 = time.perf_counter()
  work.mkdir(parents=True, exist_ok=True)
  qocogen.generate_solver(n, 0, neq, p, np.zeros(n), h, np.zeros(neq), None, None, 0, 0, [], str(work), "qoco_custom")
  t_gen = time.perf_counter() - t0
  solver = work / "qoco_custom"
  harness = solver / "time_ldl.c"
  harness.write_text(
    '#include <stdio.h>\n#include <time.h>\n#include "qoco_custom.h"\n#include "ldl.h"\n'
    + LOOP
    + "int main(void) { static Workspace work; set_default_settings(&work); load_data(&work); double best;\n"
    + "  TIME_BEST(ldl(&work), best); printf(\"%.3f\\n\", best); return 0; }\n"
  )
  sources = [str(s) for s in sorted(solver.glob("*.c")) if s.name not in ("runtest.c", "time_ldl.c")]
  exe = solver / "time_ldl"
  t0 = time.perf_counter()
  subprocess.run(["cc", *CFLAGS, "-o", str(exe), *sources, str(harness), "-lm"], check=True, capture_output=True, timeout=budget)
  t_cc = time.perf_counter() - t0
  best = min(float(subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout) for _ in range(PROCESSES))
  ldl_src = solver / "ldl.c"
  return {"t_generate": t_gen, "t_compile": t_cc, "factor_ns": best, "ldl_source_bytes": ldl_src.stat().st_size, "text_bytes": text_bytes(exe)}


def amd_perm(p, h) -> np.ndarray:
  """The AMD ordering QOCOGEN takes from qdldl-python (`codegen.py`: `qdldl.Solver(K).factors()`)."""
  import qdldl

  n, neq = p.shape[0], h.shape[0]
  k = sparse.block_array([[p + sparse.identity(n), h.T], [h, -sparse.identity(neq)]], format="csc")
  _, _, perm = qdldl.Solver(k).factors()
  return np.asarray(perm, dtype=np.int64)


def run_qdldl(k_up: sparse.csc_array, perm: np.ndarray, n_primal: int, work: Path) -> dict:
  kp = sparse.csc_array(sparse.triu(sparse.csc_array((k_up + sparse.triu(k_up, k=1).T))[perm][:, perm], format="csc"))
  kp.sort_indices()
  nn = kp.shape[0]
  (work / "qdldl_types.h").write_text(
    "#ifndef QDLDL_TYPES_H\n#define QDLDL_TYPES_H\n#include <limits.h>\ntypedef int QDLDL_int;\ntypedef double QDLDL_float;\n"
    "typedef unsigned char QDLDL_bool;\n#define QDLDL_INT_MAX INT_MAX\n#endif\n"
  )
  (work / "qdldl_version.h").write_text('#define QDLDL_VERSION "qoco"\n')

  def arr(name, a, ctype):
    return f"static const {ctype} {name}[{max(len(a), 1)}] = {{{', '.join(repr(float(v)) if ctype == 'double' else str(int(v)) for v in a) or '0'}}};\n"

  perm_inv = np.empty_like(perm)
  perm_inv[perm] = np.arange(perm.size)
  src = (
    '#include <stdio.h>\n#include <stdlib.h>\n#include <time.h>\n#include "qdldl.h"\n'
    + LOOP
    + f"#define N {nn}\n"
    + arr("Ap", kp.indptr, "int")
    + arr("Ai", kp.indices, "int")
    + arr("Ax", kp.data, "double")
    + arr("perm", perm, "int")
    + "int main(void) { int *Lnz = malloc(N * sizeof(int)), *etree = malloc(N * sizeof(int)), *iw = malloc(3 * N * sizeof(int));\n"
    + "  int sumLnz = QDLDL_etree(N, Ap, Ai, iw, Lnz, etree);\n"
    + "  int *Lp = malloc((N + 1) * sizeof(int)), *Li = malloc(sumLnz * sizeof(int)); double *Lx = malloc(sumLnz * sizeof(double));\n"
    + "  double *D = malloc(N * sizeof(double)), *Dinv = malloc(N * sizeof(double)), *fw = malloc(N * sizeof(double));\n"
    + "  unsigned char *bw = malloc(N);\n"
    + f"  double best; TIME_BEST(QDLDL_factor(N, Ap, Ai, Ax, Lp, Li, Lx, D, Dinv, Lnz, etree, bw, iw, fw, (int*)perm, {n_primal}, 1e-11), best);\n"
    + '  printf("%.3f %d\\n", best, sumLnz); return 0; }\n'
  )
  (work / "time_qdldl.c").write_text(src)
  exe = work / "time_qdldl"
  subprocess.run(["cc", *CFLAGS, f"-I{work}", f"-I{QDLDL / 'include'}", "-o", str(exe), str(work / "time_qdldl.c"), str(QDLDL / "src" / "qdldl.c")], check=True, capture_output=True)
  runs = [subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.split() for _ in range(PROCESSES)]
  return {"factor_ns": min(float(r[0]) for r in runs), "nnz_l": int(runs[0][1])}


def run_scaly(k_up: sparse.csc_array, perm: np.ndarray | None, work: Path, schedule: str = "auto", repeats: int = 200) -> dict:
  import scaly as sc
  from scaly.codegen import write_module
  from scaly.linalg import SparseLDL
  from scaly.linalg.symbolic import analyze

  pattern = k_up.copy()
  pattern.data[:] = 1.0
  rows, cols = sparse.coo_array(k_up).coords
  t0 = time.perf_counter()
  sym = analyze(k_up.shape, np.asarray(rows), np.asarray(cols), "auto", perm=perm)

  @sc.function(sc.S(pattern.toarray() != 0), name="kkt_ldl")
  def factor(k):
    return SparseLDL(k, symbolic=sym, schedule=schedule, name="kkt").values

  module = write_module(factor, work / "scaly")
  t_gen = time.perf_counter() - t0
  lib = work / "scaly_ldl.so"
  t0 = time.perf_counter()
  subprocess.run(["cc", *CFLAGS, "-shared", "-fPIC", "-o", str(lib), str(work / "scaly" / module.source_name), "-lm"], check=True, capture_output=True)
  t_cc = time.perf_counter() - t0
  sys.path.insert(0, str(HERE.parent / "fatrop_chain"))
  from sweep import time_call

  values = sparse.csc_array(k_up).data  # CSC order, as the sparse input's C signature takes it
  best = min(time_call(lib, "kkt_ldl", [values], [int(factor.outputs[0].size)], (max(module.workspace_size, 1), 0, 0, 0), repeats, work)[0] for _ in range(PROCESSES))
  return {
    "factor_ns": best * 1e9,
    "nnz_l": int(sym.nnz_l),
    "work": int(sym.update_lanes + sym.nnz_l),
    "t_generate": t_gen,
    "t_compile": t_cc,
    "source_bytes": (work / "scaly" / module.source_name).stat().st_size,
    "text_bytes": text_bytes(lib),
  }


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
  ap.add_argument("--budget", type=float, default=600.0, help="seconds allowed for QOCOGEN's compile")
  ap.add_argument("--unroll-limit", type=int, default=60000, help="largest factorization (multiply-adds) generated straight-line")
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()
  sys.path.insert(0, str(HERE.parent))
  from _common import wait_for_quiet

  rows = json.loads(args.out.read_text()) if args.out.exists() else []
  done = {r["horizon"] for r in rows}
  for horizon in args.horizons:
    if horizon in done:
      continue
    p, h = lqr(horizon)
    k_up = kkt_upper(p, h)
    perm = amd_perm(p, h)
    row = {"horizon": horizon, "order": int(k_up.shape[0]), "nnz_kkt_upper": int(k_up.nnz), "paper_us": PAPER.get(horizon)}
    with tempfile.TemporaryDirectory(prefix="ldl-bench-") as tmp:
      work = Path(tmp)
      os.environ["SCALY_CACHE_DIR"] = str(work / "cache")
      row["load"] = wait_for_quiet()
      row["qdldl"] = run_qdldl(k_up, perm, p.shape[0], work)
      row["scaly_own"] = run_scaly(k_up, None, work / "own")
      row["scaly_amd"] = run_scaly(k_up, perm, work / "amd")
      if row["scaly_own"]["work"] <= args.unroll_limit:
        row["scaly_unroll"] = run_scaly(k_up, None, work / "unroll", schedule="unroll")
      try:
        row["qocogen"] = run_qocogen(horizon, p, h, work / "qocogen", args.budget)
      except subprocess.TimeoutExpired:
        row["qocogen"] = {"failed": f"compile exceeded {args.budget:.0f} s"}
    rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))
    brief = {k: (v.get("factor_ns") if isinstance(v, dict) else v) for k, v in row.items()}
    print(horizon, brief, flush=True)


if __name__ == "__main__":
  main()
