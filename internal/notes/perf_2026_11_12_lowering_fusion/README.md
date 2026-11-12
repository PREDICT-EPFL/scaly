# Lowering fusion measurements, November 12

PR #452 implements layer 1 of #69. Compare the pushed compiler with main at `3b89797b`, using the exact revisions and source-tree hashes recorded in the raw manifest. The benchmark harness, problem definitions, solver plugins and dependency lockfile are identical between them. Main is imported from a source archive; the PR uses its clean, pushed checkout. Later merge-preparation changes tighten tests and retain this evidence without changing the measured compiler.

The neural-process model predictive control Jacobian is 0.72% slower and its Lagrangian Hessian is 4.01% slower, measured by the ratio of process medians. Both pass the roadmap's 5% runtime limit. Three fresh processes per revision and kernel provide the following medians. Coefficients of variation (CVs) use the sample standard deviation divided by the mean; each pair lists main / PR. Generation includes symbolic derivative construction, lowering and C rendering, excluding native compilation.

| Kernel | C lines | C bytes | Generation, ms | Native runtime, us | Runtime change | Runtime CV, % | Generation CV, % |
|---|---:|---:|---:|---:|---:|---:|---:|
| chain 5 | 7,864 → 7,877 | 790,960 → 788,491 | 10,417.0 → 9,046.8 | 162.927 → 164.375 | +0.89% | 0.56 / 0.13 | 1.51 / 0.38 |
| race_cars 5 | 1,146 → 1,148 | 71,416 → 71,628 | 983.2 → 835.7 | 0.973 → 0.961 | -1.25% | 0.97 / 0.32 | 5.28 / 4.30 |
| npmpc_jac 100 | 1,102 → 1,108 | 67,203 → 67,439 | 842.7 → 861.8 | 70.996 → 71.510 | +0.72% | 0.95 / 0.73 | 0.72 / 4.04 |
| npmpc 100 | 1,582 → 1,503 | 137,012 → 134,429 | 1,220.4 → 1,186.3 | 122.819 → 127.740 | +4.01% | 0.44 / 0.80 | 4.14 / 1.86 |

Caller workspace is 109,944 → 66,744 doubles for chain M=5. The race-car Hessian at N=5 and neural MPC Jacobian at N=100 both use zero on main and the PR. Their zero-workspace assertions are regression guards. The neural MPC Hessian at N=100 uses 19,744 doubles on both revisions. Source line counts alone do not describe storage or runtime: the chain source gains 13 lines while its workspace falls by 43,200 doubles.

## Controls and retained attempts

Use the [benchmark protocol](../benchmark_protocol.md) on la015, an AMD Ryzen 9 7940HS with 16 logical CPUs. The governor is `performance`, boost is disabled, and each cell is pinned to CPU 2. The kernel compiler is Clang 20.1.8, compiling C with `-O3 -march=native -fno-math-errno -fveclib=libmvec`; only the Google Benchmark wrapper is C++. Vector math uses glibc 2.39. Python is 3.14.3. Environment is `SCALY_VECTOR_LIBM=glibc SCALY_CC=gcc CC=gcc CXX=clang++ SCALY_JIT_KEY=verify`. Each process has an empty Scaly cache and deterministic synthetic samples, without harvested inputs. Google Benchmark calls the native symbol directly for a minimum of 0.5 seconds, excluding Python dispatch.

The study ran under a named tmux session, held `~/.scaly-measuring` and both suite-slot locks, and checked CPU policy before and after every cell. The standard smoke passed all 13 checks, followed by a separate smallest-cell probe. Revision order alternated between rounds and the workload order rotated. Every accepted cell passed the symbolic reference check and the compiled-C reference check, and retained input and expected-output hashes match between revisions and repetitions.

There were 30 timing attempts. Five completed rows are excluded because foreign focused tests overlapped the first and third chain pairs or the last race-car pair. One further race-car attempt failed the quiet-machine check after native execution. The six replacement attempts ran when no foreign tests or builds were observed. The final table has 24 accepted rows, three per revision per kernel, with no correctness failures, compilation failures, timeouts or skipped cells. All attempted artifacts remain in place; exclusions are explicit in `results.json`. Accepted runtime coefficients of variation are below 1%; generation coefficients range from 0.38% to 5.28%. Small generation differences should be read with that spread.

Raw artifacts are under `benchmarks/results/flush69/measurements-2026-11-12/`. They include the revision manifest, results, per-cell provenance, generated C, native compiler commands, samples, correctness records and Google Benchmark output. The adjacent `measurement.log` and `measurement-rerun.log` retain the initial run and recovery. The launch driver is retained at `benchmarks/results/flush69/measurement_driver.py`.

From the worktree root, the original command was:

```sh
uv run benchmarks/results/flush69/measurement_driver.py run \
  --base 3b89797b --repetitions 3 \
  --out benchmarks/results/flush69/measurements-2026-11-12
```

The recovery command used the same output directory and driver with mode `rerun`. Regenerate and audit the table with the tracked analysis script:

```sh
uv run internal/notes/perf_2026_11_12_lowering_fusion/analyze.py \
  benchmarks/results/flush69/measurements-2026-11-12
```

No published benchmark page changes. These measurements cover the four specified sparse derivative kernels, rather than closed-loop solver performance.
