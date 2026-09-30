# Remaining speed gaps, 2026-09-30

Measurements behind section 2 of `../perf_gaps_proposal_2026_09_30.html` (the chain's stage Hessian
and the instruction cache). Run from the repository root:

| File | What it does |
| --- | --- |
| `chain_probe.py M OUTDIR` | builds the chain benchmark's Lagrangian Hessian (lower triangle, horizon 40) with `nlp_oracles`, prints its colouring width and nnz, writes the generated C to `OUTDIR/chain_M.c` and prints per-procedure line, loop, division and multiplication counts |
| `chain_time.py M` | times the same Hessian through the Function call, fastest of 7 × 50 calls |
| `kernels.py --base <worktree>/src` | the implementation plan's kernel gate (`../perf_gaps_plan_2026_09_30.html`): dense `a @ b`, `cholesky` and lower `solve_triangular` of order n, rendered from this checkout and from a base worktree, compiled with the JIT's flags, and timed natively against the vendored BLASFEO (`ARMV8A_APPLE_M1`, panel-major) with `../perf_2026_09_30_blasfeo/bench.c`; writes `results/kernels_<tag>.json` |

Section 2's numbers come from the Linux VM on the M3 Max (gcc 11.4, `-O2 -mcpu=native -fno-math-errno`);
machine-code sizes are from `nm -S` on `cc -O2 -mcpu=native -fno-math-errno -c chain_M.c`. Repeat on the
Mac with Apple clang before acting on them (step A1's first experiment).
