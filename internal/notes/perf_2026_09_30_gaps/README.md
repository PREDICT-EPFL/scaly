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
| `shapes.py --op OP --shapes ... --modes ...` | small dense kernels of any shape (products, factors, solves, `cho_solve`) rendered two or more ways (`auto`, `straight`, `loops`, or another checkout's), timed from C |
| `chain_cliff.py [M ...]` | A1's measurement, natively: the chain's stage Hessian at M = 3 … 9, the stage body's machine code and the time per 1 000 multiplications at `-O2` and `-Os`; `results/chain_cliff.json` |
| `chain_groups.py [M ...]` | C-211 (Tier 3): the same Hessian with its seeds in groups, per body budget (the host's, one body, all, a quarter and 16 KiB): the groups made, the work the mapped bodies generate and the time from C; `results/chain_groups.json` |

`results/` holds what the plan cites: the native kernel sweeps against BLASFEO (`kernels_*.json`),
the straight-line and loop comparisons (`shapes_*.json`), the corpus timings of C-205 and C-206
(`corpus_*.json`), the dense IPM's before and after C-205 (`ipm_dense_c205.json`), A1's cliff
(`chain_cliff.json`), the groups that take it away (`chain_groups.json`) and the benchmark sweep of the chain against CasADi SX after them (`sweep_chain_t3.*`), and Tier 4's: the register-tile prototype, rendered from a scratch copy of the lowering (`kernels_t4_proto_4x8.json`, `shapes_t4_proto_small.json`); GEMM after C-212 and after C-213 (`kernels_t4_c212.json`, `kernels_t4_c213.json`), the tile shapes tried (`kernels_t4_tile_*.json`), tiles without panels (`kernels_t4_nopanel.json`), the factorizations after Tier 4 (`kernels_t4_fact.json`); small and large products against Tier 3 (`shapes_t4_small_final.json`), expanded against looped narrow products (`shapes_t4_c206.json`), panels against tiles in place (`shapes_t4_panel_rows.json`, `shapes_t4r_panel_l2.json`); the corpus (`corpus_c212.json`, `corpus_c213.json`) and the Riccati rule (`riccati_rules_t4.txt`) and A7's profile of the dense IPM (`a7_profile_dense_ipm.txt`, made with
`../perf_2026_09_27_ipm_speed/prof.py`).
