# Case study E3: embedded convex MPC QPs

The QP benchmarks of QOCO/QOCOGEN (Chari and Açıkmeşe, Math. Prog. Comp. 2026), which is the closest
prior work to Scaly's generated PIQP: a generated interior-point solver specialized to one KKT pattern.
Two parts, both with Scaly added: QOCO's LDLᵀ microbenchmark (Appendix B, Table 3), and its
oscillating-masses MPC family, solved by every solver across horizons, including past where QOCOGEN
stops generating.

```bash
examples/case_studies/embedded_qp/baseline/setup.sh   # qoco-benchmarks and qoco pinned
uv run --with qocogen --with qdldl examples/case_studies/embedded_qp/ldl_bench.py --out examples/case_studies/embedded_qp/results/ldl.json
uv run --with qoco --with qocogen --with clarabel --with cvxpy --with pandas --with osqp \
  examples/case_studies/embedded_qp/compare.py --out examples/case_studies/embedded_qp/results/grid.json
uv run --with qoco --with qocogen --with cvxpy --with pandas examples/case_studies/embedded_qp/embedded_size.py --out examples/case_studies/embedded_qp/results/embedded.json
uv run --with jupyterlab --with qdldl jupyter lab examples/case_studies/embedded_qp/embedded_qp.ipynb
```

`embedded_qp.ipynb` explains the study, solves an instance with the generated PIQP live, shows the KKT
pattern and its factor, times Scaly's two factorization schedules, and plots the recorded tables. Its
`RUN_BASELINES` flag reruns the baselines.

## What changed from the plan

- CVXPYgen's quadcopter MPC is not reproducible: its script and its numbers (A, B, weights, limits)
  were never published, and the paper reports times only in a figure. qoco-benchmarks' oscillating
  masses, the QOCO paper's QP family, is public and fully specified, so it is the family here.
- CVXPYgen itself is left out. Importing it imports `pdaqp`, whose `juliacall` downloads Julia and
  writes `~/.julia` on first import. OSQP and Clarabel are timed as libraries.
- QOCO's Appendix B script is not public either. The LQR KKT matrix is rebuilt from the paper's
  description, with random `A`, `B` of the stated sizes and an unstated `eps`, here 1e-7.
- `sc.opt.solver`'s QP build-time growth (CS-6) did not need fixing first: the generated PIQP builds in
  under 2 s up to horizon 32 (the grid records every build).

## Files

| File | What it is |
| --- | --- |
| `baseline/setup.sh` | pins qoco-benchmarks (`d7e00f5`) and qoco (`0198625`, for its vendored QDLDL) |
| `baseline/run_qoco.py` | QOCO, Clarabel, OSQP at 1e-7 and 1e-3, and QOCOGEN, through qoco-benchmarks' own problem conversion and C timing harness, with QOCOGEN compiled with Scaly's flags |
| `problem.py` | the family as a Scaly `sc.opt.problem` with `Q`, `R`, `x0` as parameters, and the benchmark's random instances replayed in its draw order |
| `run_scaly.py` | the generated PIQP (`examples/qp_solvers/generated_piqp.py`, Tier 4's `backend="scaly"`) and the PIQP library, timed from C |
| `compare.py` | every solver, every horizon, fresh processes, the quiet-machine gate |
| `ldl_bench.py` | the LDLᵀ microbenchmark: QOCOGEN's `ldl()`, QDLDL, Scaly's `SparseLDL` looped and straight-line |
| `embedded_size.py` | the embedded leg without a board: flash and RAM for a Cortex-M7 build |
| `embedded_qp.ipynb` | the study as a notebook |

`tests/integration/test_case_study_embedded_qp.py` checks the generated PIQP on a QP whose Hessian
weights are parameters against the PIQP library, iteration for iteration.

## Results

Apple M3 Max, Apple clang 21 (Homebrew clang 21 for the Cortex-M7 build), `-O2 -mcpu=native
-fno-math-errno` for every solver. Not the reference machine. Every C timing is the best 1 ms batch
of calls, the best of three processes. The paper's numbers are from a Ryzen 9 7950X3D, context only.

### The LDLᵀ microbenchmark

| T | Order | nnz(L) | QDLDL, µs | QOCOGEN, µs | Scaly looped, µs | Scaly looped (AMD), µs | Scaly unrolled, µs | Paper QDLDL / QOCOGEN, µs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 72 | 396 | 1.99 | 0.36 | 0.98 | 0.98 | 0.34 | 2.103 / 0.435 |
| 15 | 222 | 1,446 | 7.01 | 1.48 | 3.58 | 3.36 | 1.25 | 7.507 / 1.777 |
| 50 | 747 | 5,121 | 24.73 | 12.83 | 13.60 | 12.99 | 16.22 | 27.077 / 6.117 |
| 75 | 1122 | 7,746 | 38.08 | 20.35 | 21.09 | 20.26 | 29.11 | 38.848 / 8.485 |
| 100 | 1497 | 10,371 | 51.95 | 28.24 | 28.32 | 27.76 | 39.35 | 49.577 / 12.894 |

| T | QOCOGEN generate + compile, s | QOCOGEN `ldl.c`, kB | QOCOGEN machine code (whole solver), kB | Scaly looped generate + compile, s | Scaly looped source, kB | Scaly looped machine code, kB | Scaly unrolled generate + compile, s | Scaly unrolled machine code, kB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.1 + 0.9 | 143 | 66 | 0.01 + 0.07 | 20 | 2.1 | 0.8 + 0.5 | 22 |
| 15 | 0.7 + 5.5 | 534 | 184 | 0.02 + 0.08 | 51 | 2.1 | 3.4 + 2.6 | 82 |
| 50 | 9.0 + 44.5 | 1936 | 619 | 0.03 + 0.09 | 167 | 2.1 | 12.6 + 11.6 | 402 |
| 75 | 20.1 + 73.1 | 2945 | 929 | 0.04 + 0.09 | 253 | 2.0 | 19.9 + 20.5 | 672 |
| 100 | 35.5 + 140.7 | 3975 | 1240 | 0.05 + 0.11 | 345 | 2.0 | 26.7 + 30.2 | 952 |

- QDLDL reproduces the paper's times to within 10% at every horizon. QOCOGEN is 17 to 20% faster
  than the paper's at T = 5 and 15 but 2.1 to 2.4x slower from T = 50. Its straight-line code grows to 1.2 MB of
  machine code, which the paper's 7950X3D with 96 MB of cache and `-O3` may serve better than this
  machine at `-O2`.
- Scaly's straight-line factorization (`schedule="unroll"`) beats QOCOGEN at T = 5 and 15 (0.34
  against 0.36 µs, 1.25 against 1.48 µs). Its looped factorization, the default above
  `sparse_unroll`, matches QOCOGEN from T = 50 (27.8 against 28.2 µs at T = 100). It does that with
  2 kB of machine code and a 0.1 s build, where QOCOGEN's whole solver needs 1.2 MB and 176 s.
- Both Scaly factorizations beat QDLDL throughout: looped 1.8 to 2.0x, straight-line 5.9x at T = 5.
- Straight-line code stops paying past T = 15: from T = 50 Scaly's unrolled factorization is slower
  than its own loops, and the looped one is what the default schedule picks.

### The oscillating-masses grid

qoco-benchmarks' oscillating masses (8 states, 4 inputs; box limits on states and inputs; `Q`, `R`, `x0`
drawn per instance), five instances per horizon, every solver at 1e-7 (OSQP also at 1e-3, the
benchmark's second setting). QOCOGEN is regenerated and recompiled for each horizon, until one
generation plus compile exceeds the budget. Median solve time over the five instances, µs:

| Horizon | Variables | QOCOGEN | Scaly PIQP (generated) | PIQP library | QOCO | Clarabel | OSQP 1e-7 | OSQP 1e-3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 104 | 225 | **126** | 163 | 211 | 614 | 380 | 129 |
| 20 | 248 | 487 | **329** | 438 | 504 | 1419 | 1009 | 343 |
| 32 | 392 | 756 | **537** | 711 | 814 | 2294 | 1627 | 553 |
| 44 | 536 | 1015 | **710** | 901 | 1121 | 3099 | 1533 | 1501 |
| 56 | 680 | 1627 | **984** | 1226 | 1693 | 4526 | 2961 | 1958 |
| 76 | 920 | – | **1344** | 1711 | 1980 | 5329 | 2637 | 2619 |
| 96 | 1160 | – | **1713** | 2126 | 2488 | 6503 | 6662 | 3328 |
| 116 | 1400 | – | **2070** | 2579 | 2958 | 7796 | 7853 | 3920 |
| 136 | 1640 | – | **2430** | 3057 | 3511 | 9162 | 9205 | 4589 |
| 156 | 1880 | – | **2873** | 3542 | 4689 | 12054 | 8093 | 2789 |
| 200 | 2408 | – | **3636** | 4622 | 5755 | 13371 | 6825 | 6785 |

| Horizon | QOCOGEN generate + compile, s | QOCOGEN machine code, kB | Scaly build + generate + compile, s | Scaly machine code, kB | Scaly C source, kB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 4 + 11 | 293 | 0.5 + 0.8 + 0.6 | 55 | 334 |
| 20 | 27 + 51 | 729 | 0.5 + 1.0 + 0.7 | 59 | 650 |
| 32 | 67 + 89 | 1167 | 0.6 + 1.1 + 0.5 | 46 | 857 |
| 44 | 139 + 226 | 1606 | 0.7 + 1.3 + 0.6 | 49 | 1133 |
| 56 | 237 + 216 | 2041 | 0.8 + 1.5 + 0.6 | 49 | 1416 |
| 76 | over 2400 in all, stopped | – | 1.0 + 1.9 + 0.6 | 49 | 1898 |
| 200 | – | – | 3.8 + 0.8 + 0.8 | 54 | 5161 |

- **Scaly's generated PIQP is the fastest solver at every horizon**: 1.4 to 1.8x QOCOGEN where
  QOCOGEN exists (T = 8 to 56), 1.2 to 1.3x the PIQP library (the same algorithm, the same iteration
  counts: 6 to 8), 1.4 to 1.7x QOCO and 3.7 to 4.9x Clarabel. Only OSQP at its loose 1e-3 tolerance
  comes close (level at T = 8, 3% faster at T = 156), with objectives off by up to 1e-3.
- **QOCOGEN stops at T = 56.** At T = 76 its generation and compile together ran past 40 minutes. At
  T = 56 it took 237 s to generate and 216 s to compile, into 2.0 MB of machine code. Scaly builds,
  generates and compiles the T = 200 solver in 5.4 s, into 54 kB of machine code: its loops keep the
  code the same size at every horizon, where QOCOGEN writes the factorization out entry by entry.
- **Agreement.** Every solver at 1e-7 agrees with Clarabel's objective to 2e-7 or better on every
  instance (the interior-point solvers to 2e-9); OSQP at 1e-3 to 1e-4 to 1e-3. QOCOGEN takes 5 to 7
  iterations, PIQP (library and generated) 6 to 8.
- Scaly's C source grows with the horizon (0.3 to 5.2 MB), because the KKT pattern's tables are
  written out as data; the machine code does not.

### The embedded leg: sizes for a Cortex-M7

No board is attached, so this is what can be said without one: each generated solver built with clang
for `thumbv7em-none-eabihf -mcpu=cortex-m7 -Os`, its flash (`.text` + `.rodata`; QOCOGEN's iteration
printers, which `DISABLE_PRINTING` leaves in, subtracted) and the workspace it needs at run time
(Scaly: the `w` array its header declares; QOCOGEN: `sizeof(Workspace)`). Neither has static RAM.

| Horizon | Scaly flash, kB (code + tables) | Scaly workspace, kB | QOCOGEN flash, kB | QOCOGEN workspace, kB |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 198 (66 + 132) | 233 | 602 | 96 |
| 20 | 411 (81 + 330) | 911 | 1568 | 235 |
| 32 | 585 (56 + 528) | 1942 | 2541 | 373 |
| 56 | 981 (56 + 924) | 4990 | 4503 | 651 |

- **Flash: Scaly's is a fifth to a third of QOCOGEN's.** Its code stays at 56 to 81 kB; what grows is
  the KKT pattern's tables, linearly. QOCOGEN writes the factorization out as straight-line code and
  passes 2 MB at T = 32, the flash of an STM32H7.
- **RAM: Scaly's workspace is 2.4 to 7.7x QOCOGEN's, and grows quadratically.** At T = 8 it already
  exceeds an STM32F4's 192 kB. The generated PIQP zero-fills a dense `n x n` array every solve to build
  `P` from the parameters `Q`, `R`, then reads its nonzeros back (CS-12 in `internal/todo.md`); the rest
  of the workspace is linear in `n`. Extracting `P` sparse would put Scaly's RAM below QOCOGEN's.
- Against an STM32F4 (1 MB flash, 192 kB RAM): at T = 8 QOCOGEN fits and Scaly does not, on RAM alone;
  at T = 20 neither does (QOCOGEN's flash is 1.6 MB, Scaly's RAM 0.9 MB). Without the dense array,
  about 1.6 n^2 doubles, Scaly's workspace would be an estimated 95 and 125 kB, and it would fit at both.
