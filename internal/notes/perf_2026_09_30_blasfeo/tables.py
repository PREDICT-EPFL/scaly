"""Markdown tables from results/raw_*.txt, results/peak.txt and build/*/meta.json.

    python3 internal/notes/perf_2026_09_30_blasfeo/tables.py > internal/notes/perf_2026_09_30_blasfeo/results/tables.md
"""

import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOMINAL = 16 * 4.05  # 4 FMA pipes x 2 doubles x 2 flops x 4.05 GHz
peak = dict(line.split()[:2] for line in (HERE / "results/peak.txt").read_text().splitlines() if line and not line.startswith("("))
MEAS = float(peak["fma_gflops"])
rows = defaultdict(dict)
for f in sorted((HERE / "results").glob("raw_*.txt")):
  for line in f.read_text().splitlines():
    if line.startswith("#") or not line.strip():
      continue
    op, var, n, ns, gf, diff = line.split()
    rows[(op, int(n))][var] = (float(ns), float(gf), float(diff))

NATIVE = {"gemm": ("bf_dgemm_nn", "bf_dgemm_nt"), "potrf": ("bf_dpotrf_l",), "trsm": ("bf_dtrsm_llnn", "bf_dtrsm_rltn")}
COLMAJ = {"gemm": "bf_blas_dgemm", "potrf": "bf_lapack_dpotrf", "trsm": "bf_blas_dtrsm"}
print(f"Measured FMA peak {MEAS:.1f} GFLOP/s at {float(peak['clock_ghz']):.2f} GHz (nominal {NOMINAL:.1f} at 4.05 GHz). "
      "Ratio = Scaly (JIT flags) ns / BLASFEO ns; > 1 means Scaly is slower.\n")
for op in ("gemm", "potrf", "trsm"):
  nat = NATIVE[op]
  print(f"### {op}\n")
  head = ["n", "Scaly JIT ns", "GF/s", "%peak", "Scaly -O3 ns", *[f"{v} ns" for v in nat], "BF best GF/s", "%peak", f"{COLMAJ[op]} ns", "ratio vs best native", "ratio vs col-major", "max|diff|"]
  print("| " + " | ".join(head) + " |")
  print("|" + "---|" * len(head))
  for (o, n), r in sorted(rows.items()):
    if o != op:
      continue
    sj, so = r["scaly_jit"], r["scaly_o3"]
    best = min((r[v] for v in nat), key=lambda x: x[0])
    cm = r[COLMAJ[op]]
    diff = max(x[2] for k, x in r.items() if k != "scaly_jit")
    cells = [n, f"{sj[0]:.1f}", f"{sj[1]:.1f}", f"{100 * sj[1] / MEAS:.0f}", f"{so[0]:.1f}", *[f"{r[v][0]:.1f}" for v in nat],
             f"{best[1]:.1f}", f"{100 * best[1] / MEAS:.0f}", f"{cm[0]:.1f}", f"{sj[0] / best[0]:.2f}", f"{sj[0] / cm[0]:.2f}", f"{diff:.1e}"]
    print("| " + " | ".join(map(str, cells)) + " |")
  print()

print("### Generation and compile (per kernel)\n")
print("| op | n | gen ms | cc JIT ms | cc -O3 ms | C bytes | C lines | .text JIT B | .text -O3 B |")
print("|---|---|---|---|---|---|---|---|---|")
metas = [json.loads(p.read_text()) for p in (HERE / "build").glob("*/meta.json")]
for m in sorted(metas, key=lambda m: (["gemm", "potrf", "trsm"].index(m["op"]), m["n"])):
  print(f"| {m['op']} | {m['n']} | {1e3 * (m['build_s'] + m['generate_s']):.1f} | {1e3 * m['jit_compile_s']:.0f} | {1e3 * m['o3_compile_s']:.0f} | "
        f"{m['c_bytes']} | {m['c_lines']} | {m['jit_text_bytes']} | {m['o3_text_bytes']} |")
print(f"\nJIT flags: `{' '.join(metas[0]['jit_flags'])}`; second column `{' '.join(metas[0]['o3_flags'])}`; compiler `{metas[0]['cc']}`.")
