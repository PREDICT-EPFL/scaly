# Scaly dense kernels against BLASFEO, 2026-09-30

Scaly-generated dense `a @ b`, `cholesky` and lower `solve_triangular` (n right-hand sides) against
BLASFEO 0.1.4.x (vendored `plugins/scaly-piqp/third_party/blasfeo`, rebuilt for Linux), double
precision, n = 4 … 128. Measured in the Linux VM (aarch64, 4 vCPUs, gcc 11.4) on the M3 Max; indicative,
not reference-machine results, and nothing under `docs/` cites them. Full tables: `results/tables.md`.

## How it was run (from the repository root)

| File | What it does |
| --- | --- |
| `gen.py [n ...]` | builds each kernel as a `Function` over n x n row-major inputs, renders its C (`render_c_module`), compiles it with the JIT's `compile_flags()` (`-O2 -ftree-vectorize -mcpu=native -fno-math-errno` for gcc 11) and with `-O3 -mcpu=native -fno-math-errno`; writes `build/<op>_<n>/{kernel.c,jit.so,o3.so,meta.json}` (generation and compile time, C size, `.text` size) |
| `bench.c` | the C harness: `dlopen`s both Scaly libraries, packs the same data into BLASFEO `blasfeo_dmat`s, checks every variant against Scaly (and Scaly against a naive residual), then times batches of >= 20 ms per sample, 9 samples interleaved across variants, minimum kept |
| `run.sh [ops] [ns]` | builds `bench.c` against `$BLASFEO` (default `$HOME/blasfeo_build`) and runs every built (op, n); output in `results/raw_*.txt` |
| `peak.c` | achievable FMA throughput (24 independent `fmla` chains, inline asm) and the core clock (dependent 1-cycle `add` chain); `results/peak.txt` |
| `tables.py` | `results/tables.md` from the above |

The venv was not `uv run`: the checkout's `.venv` is the Mac's, and `uv run` in the VM would have
replaced it. Instead `uv venv $HOME/venv --python 3.12` + `numpy scipy`, and
`PYTHONPATH=src $HOME/venv/bin/python internal/notes/perf_2026_09_30_blasfeo/gen.py`.
BLASFEO: copied to `$HOME/blasfeo_build`, `make static_library -j4 TARGET=ARMV8A_APPLE_M1
LA=HIGH_PERFORMANCE MF=PANELMAJ` (OS_LINUX auto-detected; default `-O2 -march=armv8-a+crc+crypto+simd`,
`BLAS_API=1` so `blasfeo_blas_dgemm`/`dtrsm` and `blasfeo_lapack_dpotrf` exist; no Fortran `dgemm_`).
6 s build.

## Machine

Measured: 16.0 flops/cycle (4 FMA pipes x 2 doubles), clock 3.56–3.60 GHz in the VM, so the achievable
peak is 57–58 GFLOP/s (nominal 64.8 at 4.05 GHz). "%peak" below is of the measured 58.2.

## Results (ns per call, Scaly at the JIT's flags; ratio = Scaly / best BLASFEO panel-major; > 1 Scaly slower)

| n | gemm Scaly | GF/s | BLASFEO nn/nt | GF/s | ratio | potrf Scaly | GF/s | dpotrf_l | GF/s | ratio | trsm Scaly | GF/s | dtrsm_rltn | GF/s | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 3.7 | 34.6 | 10.2 | 12.6 | 0.36 | 5.3 | 4.0 | 19.5 | 1.1 | 0.27 | 4.1 | 15.6 | 12.1 | 5.3 | 0.34 |
| 8 | 28.3 | 36.2 | 36.1 | 28.4 | 0.78 | 26.3 | 6.5 | 46.0 | 3.7 | 0.57 | 18.5 | 27.6 | 32.9 | 15.6 | 0.56 |
| 12 | 115 | 30.1 | 97.6 | 35.4 | 1.18 | 139 | 4.2 | 91.1 | 6.3 | 1.52 | 151 | 11.4 | 82.8 | 20.9 | 1.82 |
| 16 | 226 | 36.2 | 195 | 42.0 | 1.16 | 249 | 5.5 | 122 | 11.2 | 2.03 | 301 | 13.6 | 150 | 27.2 | 2.00 |
| 24 | 915 | 30.2 | 590 | 46.8 | 1.55 | 623 | 7.4 | 292 | 15.8 | 2.13 | 779 | 17.7 | 402 | 34.3 | 1.94 |
| 32 | 1870 | 35.0 | 1339 | 49.0 | 1.40 | 1250 | 8.7 | 522 | 20.9 | 2.40 | 1698 | 19.3 | 850 | 38.5 | 2.00 |
| 48 | 6497 | 34.0 | 4254 | 52.0 | 1.53 | 3547 | 10.4 | 1264 | 29.2 | 2.81 | 4884 | 22.6 | 2547 | 43.4 | 1.92 |
| 64 | 15714 | 33.4 | 9947 | 52.7 | 1.58 | 7330 | 11.9 | 2518 | 34.7 | 2.91 | 10832 | 24.2 | 5700 | 46.0 | 1.90 |
| 96 | 137543 | 12.9 | 32954 | 53.7 | 4.17 | 20650 | 14.3 | 7262 | 40.6 | 2.84 | 34249 | 25.8 | 18477 | 47.9 | 1.85 |
| 128 | 339103 | 12.4 | 78797 | 53.2 | 4.30 | 43857 | 15.9 | 16386 | 42.7 | 2.68 | 79344 | 26.4 | 41849 | 50.1 | 1.90 |

Flops: gemm 2n^3, potrf n^3/3, trsm n^3. BLASFEO's column-major API (packing on the fly): `blasfeo_blas_dgemm` is within 1–6 % of the panel-major
gemm from n = 16 up; `blasfeo_lapack_dpotrf` is 1.8x slower than `dpotrf_l` at n = 16, falling to 1.14x at
n = 128; `blasfeo_blas_dtrsm` is 1.25–1.75x slower than `dtrsm_rltn` from n = 16; at n <= 12 they are 1.1–3.5x slower;
columns in `results/tables.md`. Max |diff| to
BLASFEO <= 2.8e-16 (gemm, trsm) and 1.8e-15 (potrf, lower triangle); Scaly `-O3` output is bitwise equal
to `-O2`.

## Observations

- **n <= 8 (straight-line code):** Scaly is 1.3–3.7x faster than BLASFEO on all three; BLASFEO's
  per-call dispatch and edge (`_vs_`) kernels dominate at these sizes.
- **Unroll threshold:** `cholesky`/`solve_triangular` unroll only to `linalg.dense_unroll = 8`; at n = 12
  both fall to loops and lose 1.5–2.4x in GFLOP/s versus n = 8 (potrf 6.5 -> 4.2, trsm 27.6 -> 11.4 GF/s).
  gemm unrolls through n = 12 (41 KB of C, 163 ms generation + 314 ms compile, the only slow build).
- **gemm 16–64:** the loop kernel keeps one 1 x 16 row of C in registers, broadcasting `a[i,k]` against 8
  loads of B per k: load-bound near 2/3 of peak, measured 30–36 GF/s (52–62 %) against BLASFEO's
  42–53 (72–91 %). Ratio 1.2–1.6.
- **gemm n >= 96:** codegen switches to a plain k-r-j triple loop accumulating in `res[]` in memory,
  12–13 GF/s, 4.2–4.3x slower than BLASFEO.
- **potrf:** 4 x 4-tiled scalar dot products, 4–16 GF/s, 2–2.9x slower than `blasfeo_dpotrf_l` from
  n = 16 up.
- **trsm:** a row-oriented loop, 11–26 GF/s, a steady ~1.9–2x slower than `blasfeo_dtrsm_rltn`, but
  on par with `blasfeo_dtrsm_llnn` at n = 24–32, which on this target uses BLASFEO's *generic C*
  `kernel_dtrsm_nn_ll_inv_4x4` (there is no armv8a asm llnn kernel).
- **-O3 vs JIT flags:** no systematic difference (within ±5 %) except trsm n = 24–32, where -O3 is
  ~20 % faster.

## Caveats

- Throughput timing: consecutive calls use the same inputs and are independent, so out-of-order
  execution overlaps them; at n <= 8 (4–50 ns per call) a latency-bound caller would see more.
  Identical protocol for every variant.
- BLASFEO kernels selected by `TARGET_ARMV8A_APPLE_M1` (which also defines `ARMV8A_ARM_CORTEX_A57`):
  armv8a asm `kernel_dgemm_nt/nn_8x4_lib4`/`4x4`, `kernel_dpotrf_nt_l_8x4`/`4x4`,
  `kernel_dtrsm_nt_rl_inv_8x4`/`4x4`; `dtrsm_llnn` uses the generic C 4x4 kernel. `LA=HIGH_PERFORMANCE`,
  `MF=PANELMAJ`, default `-O2`, `MACRO_LEVEL=1`.
- The trsm variants reset BLASFEO's cached inverse diagonal (`sA.use_dA = 0`) every call, i.e. a fresh
  factor each time as for Scaly. The in-place column-major `lapack_dpotrf` and `blas_dtrsm` restore their
  input with a memcpy per call, timed alone and subtracted. BLASFEO panel-major data is packed once,
  outside the timing (the "data already in BLASFEO format" case).
- VM (4 vCPUs, host scheduler), clock 3.6 GHz measured rather than the 4.05 GHz single-core boost;
  a first 3-sample run of n = 4, 8, 16 agreed with the final one within ~2 %.
- `.text` sizes (`size -A`, in `results/tables.md`) include ~250 B of shared-object boilerplate (an
  empty `.so` is 248 B): Scaly kernels are 0.2–8.7 KB of code (gemm n = 12 unrolled the largest).
- Scaly generation (graph build + `render_c_module`) is 1–60 ms per kernel except gemm n = 12
  (163 ms); gcc 11 compile 29–330 ms.
