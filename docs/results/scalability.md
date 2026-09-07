# Alloy scalability sweep

The per-cell measurements behind [the results overview](index.md), which is the page to read first
if you want the summary rather than the tables.

The initial 2026-09-05 Hessian pilot and its range extensions below are the current kernel
baseline. Other measurement sections retain earlier results and their limitations. Do not
combine those older timings with the current baseline.

Runs `benchmarks/run.py sweep` over a fixed cell grid for each workload, capturing per-cell codegen / compile / runtime / source-size metrics. Each cell compiles its selected backend into a separate Google Benchmark binary. The binary scatters the compact result into a dense matrix and compares it with an independent reference before it records a timing.

The unsuffixed workloads measure the exact sparse Lagrangian Hessian from the solver descriptor.
The `_jac` workloads retain the constraint Jacobian rows for the long paper and do not run by default.

Skip rules applied automatically:

- per-cell compile timeout (default 180 s);
- max generated source size (default 50 MiB) — skip without compiling;
- after a backend hits any of the above at one size, larger sizes for that backend are skipped immediately, because both generated source size and compile cost are monotonically increasing in the iteration count.

CSV with the raw cell data: `benchmarks/results/sweep/scalability.csv`. Each cell's generated code, samples, binary, and logs live beside it under `benchmarks/results/sweep/repeat_<n>/<workload>/<backend>_<axis><size>/`.

`coloring_width` is an Alloy-owned construction metric, not a cross-backend comparison. It counts
the compressed tangent directions that Alloy executes. Structured Jacobian rows add the independently
executed per-formal or local batches, while Hessian rows use the global star-color count. CasADi
leaves the field blank because its generated code exposes no internal derivative count.

The current race-car formulation maps the stage dynamics and cost. Unbumpercars maps car
dynamics and pair constraints, while its per-car wall rows remain unrolled. Earlier unbumpercars
measurements below predate the pair-constraint port.

## Initial frozen-protocol pilot, 2026-09-05

The [reference machine and measurement protocol](fairness.md#the-reference-machine) define the
comparison. The reference machine used `performance` on every CPU policy, boost disabled, and unrestricted
affinity across CPUs 0 through 15. Each cell ran in five fresh processes, with backend order
rotated from seed 0 and a minimum Google Benchmark duration of 0.5 seconds. Compilation and
timing ran sequentially. The existing Quanser broker and development server remained running.

These are exact sparse Lagrangian Hessian kernel measurements with synthetic inputs. No harvested
closed-loop inputs were present. They do not measure closed-loop solve time. The compiler and
benchmark implementation were unchanged during measurement. Provenance reports a dirty tree because of local documentation
edits and untracked files, including the temporary task plan.

Raw CSVs, summaries, provenance, generated source, samples, binaries, and compile logs are local,
gitignored artifacts under `benchmarks/results/pilot/2026-09-05/`. This is an initial baseline,
not the immutable publication archive. CasADi 3.8 transformation was enabled. The default limits
were 300 seconds for code generation, 180 seconds for compilation, and 50 MiB of generated C source.

The 245 recorded rows contain 210 successful timings, 20 compile-budget failures, and 15 skips
after a smaller-size failure. No correctness check failed. Every timed combination has all five
successful processes. The runtime coefficient of variation (CV) ranges from 0.07% to 2.17%.

### Race-car Hessian

Reproduce with:

```bash
uv run benchmarks/run.py sweep --workloads race_cars --sizes 10,40,100 --repetitions 5 --order-seed 0 --benchmark-min-time 0.5s --headline --boost off --out benchmarks/results/pilot/2026-09-05/race_cars/race.csv
```

All 70 completed timings passed the dense-reference correctness check. Unrolled MX at N=100
hit the 180-second generated-kernel compile timeout in all five attempts.

Runtime is the process mean in microseconds. CV is the sample coefficient of variation across
processes. Executable source excludes static metadata. Workspace counts caller-supplied doubles,
not total memory or stack storage. Compilation is the generated kernel only, excluding the wrapper
and linking.

| N | Backend | Mean, µs | CV, % | Executable bytes | Workspace, doubles | Kernel compile mean, ms |
|---:|---|---:|---:|---:|---:|---:|
| 10 | `alloy` | 6.705 | 0.21 | 49591 | 0 | 297.7 |
| 10 | `casadi_sx` | 4.263 | 0.66 | 209958 | 119 | 693.8 |
| 10 | `casadi_mx` | 5.928 | 2.14 | 879492 | 2065 | 4034.0 |
| 10 | `casadi_call_mx` | 14.267 | 1.04 | 269955 | 3494 | 1590.4 |
| 10 | `casadi_map_sx` | 15.202 | 1.07 | 142699 | 11334 | 1061.1 |
| 40 | `alloy` | 26.500 | 0.45 | 49659 | 3840 | 306.4 |
| 40 | `casadi_sx` | 17.035 | 0.07 | 823845 | 119 | 2978.1 |
| 40 | `casadi_mx` | 23.261 | 1.01 | 3799659 | 7975 | 57645.3 |
| 40 | `casadi_call_mx` | 56.940 | 0.82 | 398502 | 11414 | 2238.7 |
| 40 | `casadi_map_sx` | 59.902 | 0.67 | 273406 | 43854 | 924.3 |
| 100 | `alloy` | 65.553 | 0.52 | 49713 | 16872 | 339.9 |
| 100 | `casadi_sx` | 43.910 | 0.78 | 2052051 | 119 | 9697.0 |
| 100 | `casadi_mx` | timeout |  | 9614050 | 19795 | >180000 |
| 100 | `casadi_call_mx` | 142.841 | 1.25 | 679651 | 27134 | 7117.0 |
| 100 | `casadi_map_sx` | 151.281 | 0.34 | 543857 | 108894 | 2042.2 |

SX is fastest at all three sizes. Alloy takes 1.49 to 1.57 times as long as SX, missing the
20% runtime margin. Against retained mapped SX, Alloy is about 2.3 times faster and
uses less executable source and caller workspace. Alloy executable source changes from 49,591
to 49,713 bytes as N grows from 10 to 100. SX caller workspace stays at 119 doubles, so these
measurements do not establish a workspace advantage over the fastest CasADi encoding.

### Unbumpercars Hessian

Reproduce with:

```bash
uv run benchmarks/run.py sweep --workloads unbumpercars --sizes 2,4,8 --repetitions 5 --order-seed 0 --benchmark-min-time 0.5s --headline --boost off --out benchmarks/results/pilot/2026-09-05/unbumpercars/unbumpercars.csv
```

The columns use the same units and definitions as the race-car table.

| C | Backend | Mean, µs | CV, % | Executable bytes | Workspace, doubles | Kernel compile mean, ms |
|---:|---|---:|---:|---:|---:|---:|
| 2 | `alloy` | 847.714 | 0.46 | 145442 | 34304 | 856.5 |
| 2 | `casadi_sx` | timeout |  | 26778898 | 37182 | >180000 |
| 2 | `casadi_mx` | 890.748 | 0.98 | 110035 | 110087 | 1071.6 |
| 2 | `casadi_mx_gemm` | 1003.622 | 0.57 | 124566 | 112315 | 830.8 |
| 2 | `casadi_mx_gemm_classic` | 1006.313 | 0.35 | 124574 | 112315 | 829.6 |
| 2 | `casadi_mx_gemm_blasfeo` | 1004.488 | 1.01 | 124574 | 112315 | 831.1 |
| 4 | `alloy` | 1708.219 | 0.22 | 190260 | 34304 | 1576.2 |
| 4 | `casadi_sx` | skipped_after_failure |  |  |  |  |
| 4 | `casadi_mx` | 2820.784 | 0.94 | 361999 | 112213 | 4474.7 |
| 4 | `casadi_mx_gemm` | 3499.961 | 0.75 | 629111 | 120257 | 3174.3 |
| 4 | `casadi_mx_gemm_classic` | 3502.710 | 0.45 | 629119 | 120257 | 3178.5 |
| 4 | `casadi_mx_gemm_blasfeo` | 3475.076 | 0.65 | 629119 | 120257 | 3181.4 |
| 8 | `alloy` | 3485.192 | 0.38 | 278329 | 51200 | 3994.0 |
| 8 | `casadi_sx` | skipped_after_failure |  |  |  |  |
| 8 | `casadi_mx` | 9982.943 | 1.10 | 1327412 | 116049 | 39891.0 |
| 8 | `casadi_mx_gemm` | 13058.710 | 0.66 | 4569620 | 136111 | 29820.3 |
| 8 | `casadi_mx_gemm_classic` | 13068.264 | 0.80 | 4569628 | 136111 | 29695.5 |
| 8 | `casadi_mx_gemm_blasfeo` | 13060.631 | 0.49 | 4569628 | 136111 | 29716.6 |

All 75 completed timings passed correctness. Literal SX exceeded the 180-second kernel compile
limit at C=2 in all five processes. The runner skipped C=4 and C=8 for SX after each failure.
Plain MX is the fastest completed CasADi encoding at every size. Alloy is 1.05, 1.65, and 2.86
times faster at C=2, 4, and 8. Alloy executable source grows
from 145,442 to 278,329 bytes between C=2 and C=8, while MX grows from 110,035 to 1,327,412 bytes.
Pair constraints are mapped, but per-car wall rows remain unrolled in `filters.py`. These data
support improved source-size scaling over MX, not fixed total executable source. No pre-port
measurement under this protocol was run, so the table alone does not isolate the effect of the port.

### Neural-process model predictive control Hessian

Reproduce with:

```bash
uv run benchmarks/run.py sweep --workloads npmpc --sizes 6,12 --repetitions 5 --order-seed 0 --benchmark-min-time 0.5s --headline --boost off --out benchmarks/results/pilot/2026-09-05/npmpc/npmpc.csv
```

The columns use the same units and definitions as the race-car table.

| N | Backend | Mean, µs | CV, % | Executable bytes | Workspace, doubles | Kernel compile mean, ms |
|---:|---|---:|---:|---:|---:|---:|
| 6 | `alloy` | 41.330 | 0.10 | 32175 | 1024 | 535.9 |
| 6 | `casadi_sx` | 78.975 | 1.06 | 7564280 | 15690 | 96243.4 |
| 6 | `casadi_mx` | 25.227 | 1.63 | 146908 | 7502 | 5789.9 |
| 6 | `casadi_call_mx` | 154.818 | 0.25 | 211807 | 46693 | 8013.1 |
| 6 | `casadi_map_sx` | timeout |  | 4900634 | 392119 |  |
| 6 | `casadi_mx_gemm` | 50.541 | 0.65 | 113631 | 8579 | 1609.0 |
| 6 | `casadi_mx_gemm_classic` | 50.804 | 0.96 | 113639 | 8579 | 1609.3 |
| 6 | `casadi_mx_gemm_blasfeo` | 50.554 | 0.48 | 113639 | 8579 | 1590.0 |
| 12 | `alloy` | 83.458 | 1.47 | 32208 | 1024 | 534.9 |
| 12 | `casadi_sx` | timeout |  | 15056568 | 30274 | >180000 |
| 12 | `casadi_mx` | 52.672 | 2.17 | 267896 | 10356 | 14747.1 |
| 12 | `casadi_call_mx` | 300.577 | 0.37 | 233070 | 56263 | 8222.2 |
| 12 | `casadi_map_sx` | skipped_after_failure |  |  |  |  |
| 12 | `casadi_mx_gemm` | 99.608 | 0.94 | 180446 | 12882 | 2752.3 |
| 12 | `casadi_mx_gemm_classic` | 99.574 | 0.76 | 180454 | 12882 | 2733.8 |
| 12 | `casadi_mx_gemm_blasfeo` | 100.174 | 0.92 | 180454 | 12882 | 2750.1 |

All 65 completed timings passed correctness. Mapped SX at N=6 exhausted the 180-second compile
budget in every process, four during kernel compilation and one during wrapper compilation.
Its N=12 cells were skipped after those failures. Literal SX compiled at N=6 but exceeded the
kernel compile limit at N=12 in all five processes.

Plain MX is fastest at both horizons. Alloy takes 1.64 and 1.58 times as long at N=6 and N=12,
but uses less executable source and caller workspace. Its executable source changes from 32,175
to 32,208 bytes, and caller workspace stays at 1,024 doubles. MX grows from 146,908 to 267,896
executable bytes and from 7,502 to 10,356 caller-workspace doubles. These two horizons establish
an initial runtime-versus-size comparison, not full-range scaling.

## Extended Hessian sweeps, 2026-09-06

Race-car and neural-process runs extend the initial pilot under the same compiler, CPU controls,
five-process protocol, and synthetic-input policy. Unbumpercars repeats its full grid under the
corrected sampling policy described below. The original pilot remains above. Combined CSVs retain
the pilot and extension rows for the stage workloads, and the fresh full grid for unbumpercars.

Artifacts are local and gitignored under `benchmarks/results/followup/2026-09-05/`:
`sweep/<problem>/` holds each new run, and `combined/<problem>.csv` and `.summary.csv` hold the
pilot and extension rows pooled into the full primary grid. The pooling was a one-off; a future
run measures each full grid in one invocation through `run.py study`, and `run.py report` renders
every table below from the study directory. The directory uses the study's start date; runs
continued after midnight in Europe/Zurich.

The table units match the pilot. A timeout has no runtime estimate. Compilation limits are the
same 180-second budget, and larger cells are skipped after a smaller-size failure within each
invocation. The extension starts a new invocation, so it attempts N=200 MX even though the
pilot already recorded a failure at N=100.

### Race-car extension

```bash
uv run benchmarks/run.py sweep --workloads race_cars --sizes 1,5,25,50,200,500 --repetitions 5 --order-seed 0 --benchmark-min-time 0.5s --headline --boost off --out benchmarks/results/followup/2026-09-05/sweep/race_cars/race_cars.csv
```

The extension records 140 successful timings, five MX compilation timeouts at N=200, and five
resulting MX skips at N=500. Combined with the pilot, the full grid has 225 rows: 210 successful
timings, ten timeouts, and five skips. No correctness check failed. Every timed cell has five
successful processes; runtime CV over the combined grid ranges from 0.07% to 2.39%.

| Size | Backend | Successful processes | Mean, µs | CV, % | Executable bytes | Workspace, doubles | Kernel compile mean, ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `alloy` | 5/5 | 0.747 | 1.40 | 49306 | 0 | 276.3 |
| 1 | `casadi_sx` | 5/5 | 0.460 | 0.35 | 25755 | 108 | 183.2 |
| 1 | `casadi_mx` | 5/5 | 0.519 | 2.39 | 95400 | 304 | 366.2 |
| 1 | `casadi_call_mx` | 5/5 | 1.382 | 0.98 | 228004 | 1067 | 784.4 |
| 1 | `casadi_map_sx` | 5/5 | 1.217 | 0.26 | 94438 | 826 | 381.6 |
| 5 | `alloy` | 5/5 | 3.418 | 0.70 | 49570 | 0 | 285.6 |
| 5 | `casadi_sx` | 5/5 | 2.162 | 0.75 | 107717 | 119 | 401.8 |
| 5 | `casadi_mx` | 5/5 | 2.953 | 0.95 | 441760 | 1085 | 1555.0 |
| 5 | `casadi_call_mx` | 5/5 | 7.146 | 1.19 | 248703 | 2244 | 1247.4 |
| 5 | `casadi_map_sx` | 5/5 | 7.652 | 0.67 | 120288 | 5914 | 926.9 |
| 25 | `alloy` | 5/5 | 16.624 | 0.93 | 49659 | 2400 | 301.8 |
| 25 | `casadi_sx` | 5/5 | 10.622 | 0.21 | 516870 | 119 | 1774.4 |
| 25 | `casadi_mx` | 5/5 | 14.515 | 1.47 | 2360004 | 5020 | 22377.9 |
| 25 | `casadi_call_mx` | 5/5 | 35.254 | 1.03 | 334019 | 7484 | 1852.4 |
| 25 | `casadi_map_sx` | 5/5 | 37.302 | 0.32 | 206150 | 27594 | 936.3 |
| 50 | `alloy` | 5/5 | 32.606 | 0.14 | 49689 | 8472 | 313.1 |
| 50 | `casadi_sx` | 5/5 | 21.329 | 0.08 | 1028495 | 119 | 3862.8 |
| 50 | `casadi_mx` | 5/5 | 29.314 | 0.99 | 4762326 | 9945 | 95956.4 |
| 50 | `casadi_call_mx` | 5/5 | 70.756 | 0.40 | 441709 | 14034 | 3232.0 |
| 50 | `casadi_map_sx` | 5/5 | 75.131 | 0.93 | 318176 | 54694 | 1508.1 |
| 200 | `alloy` | 5/5 | 129.489 | 0.25 | 49753 | 36084 | 380.7 |
| 200 | `casadi_sx` | 5/5 | 90.008 | 2.01 | 4100194 | 119 | 26807.9 |
| 200 | `casadi_mx` | 0/5 | timeout |  | 20493309 | 39495 |  |
| 200 | `casadi_call_mx` | 5/5 | 283.605 | 0.42 | 1143240 | 53874 | 23221.7 |
| 200 | `casadi_map_sx` | 5/5 | 307.094 | 0.74 | 1030885 | 217294 | 5995.0 |
| 500 | `alloy` | 5/5 | 329.479 | 0.20 | 49765 | 90084 | 521.9 |
| 500 | `casadi_sx` | 5/5 | 227.813 | 2.35 | 10246853 | 119 | 153075.8 |
| 500 | `casadi_mx` | 0/5 | skipped_after_failure |  |  |  |  |
| 500 | `casadi_call_mx` | 5/5 | 725.616 | 0.49 | 2552822 | 134274 | 148584.2 |
| 500 | `casadi_map_sx` | 5/5 | 786.129 | 0.78 | 2535806 | 542494 | 30216.1 |

SX is fastest at every sampled horizon from N=1 to N=500. Alloy takes 1.44–1.62 times SX
runtime, so the 20% runtime target remains unmet across the extended range. At N=500 the means
are 329.479 µs for Alloy and 227.813 µs for SX. Their process CVs are 0.20% and 2.35%.

Alloy executable source grows from 49,306 bytes at N=1 to 49,765 at N=500. SX grows from
25,755 to 10,246,853 bytes. At N=500, however, Alloy uses 90,084 caller-workspace doubles
against SX's 119. The runtime and workspace comparison therefore differs from the source-size
comparison. Against retained mapped SX at N=500, Alloy is 2.39 times faster and uses less
executable source and caller workspace.

### Neural-process extension

```bash
uv run benchmarks/run.py sweep --workloads npmpc --sizes 25,50,100,200 --repetitions 5 --order-seed 0 --benchmark-min-time 0.5s --headline --boost off --out benchmarks/results/followup/2026-09-05/sweep/npmpc/npmpc.csv
```

The extension records 90 successful timings, 30 compilation timeouts, and 40 skips after failures.
Combined with the pilot, the N=6–200 grid has 240 rows: 155 successful timings, 40 timeouts,
and 45 skips. No correctness check failed. Every timed cell has five successful processes;
runtime CV over the combined grid ranges from 0.10% to 2.17%.

| Size | Backend | Successful processes | Mean, µs | CV, % | Executable bytes | Workspace, doubles | Kernel compile mean, ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 25 | `alloy` | 5/5 | 173.354 | 0.39 | 32252 | 3724 | 545.0 |
| 25 | `casadi_sx` | 0/5 | timeout |  | 31293142 | 61622 |  |
| 25 | `casadi_mx` | 5/5 | 113.075 | 1.57 | 530986 | 16763 | 49545.2 |
| 25 | `casadi_call_mx` | 5/5 | 611.673 | 0.57 | 272875 | 76952 | 9447.6 |
| 25 | `casadi_map_sx` | 0/5 | timeout |  | 7834848 | 1601822 |  |
| 25 | `casadi_mx_gemm` | 5/5 | 213.102 | 1.85 | 326890 | 22332 | 6268.9 |
| 25 | `casadi_mx_gemm_classic` | 5/5 | 211.544 | 0.62 | 326898 | 22332 | 6287.3 |
| 25 | `casadi_mx_gemm_blasfeo` | 5/5 | 213.932 | 1.07 | 326898 | 22332 | 6235.2 |
| 50 | `alloy` | 5/5 | 344.498 | 0.21 | 32319 | 17134 | 569.0 |
| 50 | `casadi_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 50 | `casadi_mx` | 0/5 | timeout |  | 1038918 | 28973 |  |
| 50 | `casadi_call_mx` | 5/5 | 1244.162 | 1.35 | 350742 | 116487 | 10269.4 |
| 50 | `casadi_map_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 50 | `casadi_mx_gemm` | 5/5 | 422.725 | 0.92 | 610767 | 40458 | 17902.8 |
| 50 | `casadi_mx_gemm_classic` | 5/5 | 421.391 | 0.52 | 610775 | 40458 | 17976.8 |
| 50 | `casadi_mx_gemm_blasfeo` | 5/5 | 424.026 | 0.73 | 610775 | 40458 | 18007.1 |
| 100 | `alloy` | 5/5 | 689.953 | 0.71 | 32344 | 33034 | 592.1 |
| 100 | `casadi_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 100 | `casadi_mx` | 0/5 | skipped_after_failure |  |  |  |  |
| 100 | `casadi_call_mx` | 5/5 | 2443.511 | 0.24 | 508705 | 195857 | 15991.6 |
| 100 | `casadi_map_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 100 | `casadi_mx_gemm` | 5/5 | 859.817 | 1.18 | 1180957 | 76728 | 68284.5 |
| 100 | `casadi_mx_gemm_classic` | 5/5 | 854.254 | 0.56 | 1180965 | 76728 | 69272.1 |
| 100 | `casadi_mx_gemm_blasfeo` | 5/5 | 859.453 | 1.38 | 1180965 | 76728 | 70411.5 |
| 200 | `alloy` | 5/5 | 1378.616 | 0.51 | 32369 | 64834 | 673.3 |
| 200 | `casadi_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 200 | `casadi_mx` | 0/5 | skipped_after_failure |  |  |  |  |
| 200 | `casadi_call_mx` | 5/5 | 4910.648 | 1.19 | 850524 | 354397 | 37432.8 |
| 200 | `casadi_map_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 200 | `casadi_mx_gemm` | 0/5 | timeout |  | 2424113 | 149268 |  |
| 200 | `casadi_mx_gemm_classic` | 0/5 | timeout |  | 2424121 | 149268 |  |
| 200 | `casadi_mx_gemm_blasfeo` | 0/5 | timeout |  | 2424121 | 149268 |  |

Both SX variants exceed the kernel compile limit at N=25 in all five processes. Plain MX
compiles at N=25 but times out at N=50. All three matrix-product variants compile through
N=100 and time out at N=200. These limits change which CasADi encoding supplies the fastest
completed timing at each horizon.

Alloy takes 1.53 times plain MX runtime at N=25. At N=50 and N=100 it is 1.22 and 1.24 times
faster than the fastest completed CasADi variant, classic matrix-product MX. At N=200 it is
3.56 times faster than call-node MX, the only CasADi encoding that completes under the budget.
This comparison is conditional on the recorded compile limit; it does not establish the runtime
of an encoding that failed to compile.

Alloy executable source grows from 32,175 bytes at N=6 to 32,369 at N=200. Caller workspace
does not stay fixed: it grows from 1,024 to 64,834 doubles. At N=200, call-node MX uses
850,524 executable bytes and 354,397 caller-workspace doubles. The large-horizon measurements
support runtime, source-size, and caller-workspace advantages over the best completed encoding.

### Unbumpercars full-grid rerun

The original collision-free synthetic placement cannot fit C=16 and C=32 in the default arena.
The interrupted extension is retained under `diagnostic-unbumpercars-placement/`; its
`correctness_fail` rows report input-construction errors, not derivative mismatches.

This rerun uses seed 42 to sample states inside the same arena without rejecting collisions.
The full C=2,4,8,16,32 grid was rerun, so no collision-free-input pilot timings are pooled into
this table. All encodings receive the same synthetic inputs at each size. Closed-loop placement
keeps its collision-free default. The new C=16/32 input gate and all existing unbumpercars problem
gates passed before timing; restoring the old placement policy makes the new gate fail.
The [measurement protocol](fairness.md#the-measurement-protocol) explains this distinction.

```bash
uv run benchmarks/run.py sweep --workloads unbumpercars --sizes 2,4,8,16,32 --repetitions 5 --order-seed 0 --benchmark-min-time 0.5s --headline --boost off --out benchmarks/results/followup/2026-09-05/sweep/unbumpercars/unbumpercars.csv
```

The 150 rows contain 80 successful timings, 25 compilation timeouts,
40 skips after smaller-size failures, and 5 source-limit skips. No correctness check failed in this rerun.
Every timed cell has five successful processes. Runtime CV ranges from 0.28% to 1.38%.
`skipped_size` means the generated C exceeds the frozen 50 MiB source cap.

| Size | Backend | Successful processes | Mean, µs | CV, % | Executable bytes | Workspace, doubles | Kernel compile mean, ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | `alloy` | 5/5 | 845.802 | 0.28 | 145442 | 34304 | 858.5 |
| 2 | `casadi_sx` | 0/5 | timeout |  | 26778898 | 37182 |  |
| 2 | `casadi_mx` | 5/5 | 899.868 | 0.95 | 110035 | 110087 | 1052.6 |
| 2 | `casadi_mx_gemm` | 5/5 | 1007.041 | 0.56 | 124566 | 112315 | 828.7 |
| 2 | `casadi_mx_gemm_classic` | 5/5 | 1004.844 | 0.66 | 124574 | 112315 | 836.1 |
| 2 | `casadi_mx_gemm_blasfeo` | 5/5 | 1007.824 | 0.51 | 124574 | 112315 | 831.0 |
| 4 | `alloy` | 5/5 | 1715.191 | 0.30 | 190260 | 34304 | 1582.1 |
| 4 | `casadi_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 4 | `casadi_mx` | 5/5 | 2800.886 | 1.00 | 361999 | 112213 | 4518.5 |
| 4 | `casadi_mx_gemm` | 5/5 | 3523.587 | 1.38 | 629111 | 120257 | 3213.1 |
| 4 | `casadi_mx_gemm_classic` | 5/5 | 3500.195 | 0.40 | 629119 | 120257 | 3204.6 |
| 4 | `casadi_mx_gemm_blasfeo` | 5/5 | 3513.676 | 0.47 | 629119 | 120257 | 3203.3 |
| 8 | `alloy` | 5/5 | 3478.775 | 0.53 | 278329 | 51200 | 3975.3 |
| 8 | `casadi_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 8 | `casadi_mx` | 5/5 | 9949.845 | 1.15 | 1327412 | 116049 | 39969.4 |
| 8 | `casadi_mx_gemm` | 5/5 | 13079.191 | 0.61 | 4569620 | 136111 | 29718.6 |
| 8 | `casadi_mx_gemm_classic` | 5/5 | 13011.433 | 0.75 | 4569628 | 136111 | 29643.5 |
| 8 | `casadi_mx_gemm_blasfeo` | 5/5 | 13070.838 | 0.70 | 4569628 | 136111 | 29767.3 |
| 16 | `alloy` | 5/5 | 7319.301 | 0.54 | 456440 | 256704 | 14573.5 |
| 16 | `casadi_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 16 | `casadi_mx` | 0/5 | timeout |  | 5399074 | 125987 |  |
| 16 | `casadi_mx_gemm` | 0/5 | timeout |  | 35514289 | 171611 |  |
| 16 | `casadi_mx_gemm_classic` | 0/5 | timeout |  | 35514297 | 171611 |  |
| 16 | `casadi_mx_gemm_blasfeo` | 0/5 | timeout |  | 35514297 | 171611 |  |
| 32 | `alloy` | 0/5 | skipped_size |  | 835338 | 1756208 |  |
| 32 | `casadi_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 32 | `casadi_mx` | 0/5 | skipped_after_failure |  |  |  |  |
| 32 | `casadi_mx_gemm` | 0/5 | skipped_after_failure |  |  |  |  |
| 32 | `casadi_mx_gemm_classic` | 0/5 | skipped_after_failure |  |  |  |  |
| 32 | `casadi_mx_gemm_blasfeo` | 0/5 | skipped_after_failure |  |  |  |  |

Plain MX is the fastest completed CasADi encoding at C=2,4,8. Alloy is 1.06, 1.63, and
2.86 times faster at those sizes. At C=16, Alloy completes in 7.319 ms on average; all
tested CasADi encodings time out or are skipped after smaller-size failures. There is no C=16
CasADi runtime estimate to divide by. No encoding produces a timing at C=32 under these limits.

#### Executable source and metadata

Alloy's executable source is much smaller than its complete generated artifact. The source cap
checks the C file before compilation; artifact bytes also include the generated header.

| C | Executable bytes | Static metadata bytes | Total artifact bytes | C source bytes | Caller workspace, doubles |
|---:|---:|---:|---:|---:|---:|
| 2 | 145442 | 16222 | 161664 | 159106 | 34304 |
| 4 | 190260 | 78641 | 268901 | 265685 | 34304 |
| 8 | 278329 | 631332 | 909661 | 903661 | 51200 |
| 16 | 456440 | 5868795 | 6325235 | 6307930 | 256704 |
| 32 | 835338 | 53938863 | 54774201 | 54708980 | 1756208 |

At C=32, the C file is 54,708,980 bytes (52.2 MiB), above the 50 MiB cap. Its complete artifact
contains 835,338 executable bytes and 53,938,863 static-metadata bytes. This is a generated-source
limit, not a measured compile failure or runtime. Reporting only executable bytes would hide the
reason that this cell supplies no timing. The mapped pair rows do not make the total artifact
fixed-size, and the remaining unrolled wall rows also contribute executable-source growth.

### Chain Hessian, 2026-09-06

Internal validation: chain stays outside the short paper. These exact sparse Lagrangian Hessian
measurements use M=3,5,9 at a fixed horizon N=40 under the same protocol as the other extensions,
with seed 0 and five fresh processes per cell. They compare backends on the current Hessian; they
are not a before/after measurement of a compiler change.

```bash
uv run benchmarks/run.py sweep --workloads chain --sizes 3,5,9 --repetitions 5 --order-seed 0 --benchmark-min-time 0.5s --headline --boost off --out benchmarks/results/followup/2026-09-05/sweep/chain/chain.csv
```

The 75 rows contain 48 successful timings, 15 compile-budget failures, and 12 skips after
failures. No correctness check failed. SX at M=5 completed in only three of five processes;
its timing and dispersion use those three successes. Every other timed cell has five successes.
Runtime CV ranges from 0.34% to 4.18%, with the largest CV on mapped SX at M=9.

| Size | Backend | Successful processes | Mean, µs | CV, % | Executable bytes | Workspace, doubles | Kernel compile mean, ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3 | `alloy` | 5/5 | 468.743 | 0.34 | 172350 | 266616 | 1609.9 |
| 3 | `casadi_sx` | 5/5 | 28.782 | 1.45 | 3330455 | 188 | 23568.0 |
| 3 | `casadi_mx` | 0/5 | timeout |  | 9410547 | 23678 |  |
| 3 | `casadi_call_mx` | 5/5 | 191.841 | 1.50 | 602188 | 27321 | 11700.7 |
| 3 | `casadi_map_sx` | 5/5 | 136.518 | 0.94 | 556793 | 94736 | 1844.4 |
| 5 | `alloy` | 5/5 | 2599.884 | 1.54 | 334535 | 1075248 | 4944.5 |
| 5 | `casadi_sx` | 3/5 | 95.470 | 0.48 | 10330465 | 335 | 174408.2 |
| 5 | `casadi_mx` | 0/5 | skipped_after_failure |  |  |  |  |
| 5 | `casadi_call_mx` | 5/5 | 972.449 | 1.19 | 2097970 | 100698 | 88149.7 |
| 5 | `casadi_map_sx` | 5/5 | 708.640 | 1.27 | 2561051 | 349353 | 11583.2 |
| 9 | `alloy` | 5/5 | 12433.306 | 0.56 | 550679 | 3778740 | 14929.8 |
| 9 | `casadi_sx` | 0/5 | timeout; skipped_after_failure |  | 24350782 | 897 |  |
| 9 | `casadi_mx` | 0/5 | skipped_after_failure |  |  |  |  |
| 9 | `casadi_call_mx` | 0/5 | timeout |  | 6664454 | 280960 |  |
| 9 | `casadi_map_sx` | 5/5 | 2804.120 | 4.18 | 9650809 | 999844 | 69531.8 |

Alloy is 3.43, 3.67, and 4.43 times slower than retained mapped SX at M=3,5,9. Against unrolled
SX it is 16.29 times slower at M=3 and 27.23 times slower at M=5, with the latter comparison
conditional on SX's three completed builds. The historical equality-Jacobian loss of 1.5–2.4
times does not describe this Hessian result. Alloy emits less executable source than the completed
CasADi encodings but uses more caller workspace: at M=9, 550,679 executable bytes and 3,778,740
workspace doubles against mapped SX's 9,650,809 bytes and 999,844 doubles. The coloring width
grows with M (12, 24, 42), so the per-stage Hessian pays more forward-over-reverse sweeps as the
stage block widens.

The [chain closed-loop run](index.md#canonical-closed-loop-sqp-runs-2026-09-05) measures a different
setup: N=12, all requested oracles together, and each problem's existing SQP oracle construction.
Its lower Alloy function-evaluation time does not overturn this N=40 kernel comparison.

## Why a C++ harness (and not the Google Benchmark Python bindings)

Every cell in the sweep codegens C, compiles a focused Google Benchmark binary, and calls the generated symbol from a tight C++ loop (`for (auto _ : state) fn(arg, res, ...)`). There is **zero Python in the timed region** — this is on purpose. The numbers above are a *codegen-quality* comparison: how fast is the generated C, with both backends measured identically.

The obvious simplification is to drop the per-cell C++ compile and instead drive [the google-benchmark Python bindings](https://pypi.org/project/google-benchmark/) (`@gb.register` + `while state:`) over `Function.__call__` and CasADi's `caf(DM)`. We measured whether that's viable:

- **The binding's own loop floor is negligible.** An empty `while state:` body benchmarks at ~14 ns/iter; a trivial Python call at ~24 ns. So the bindings do *not* add meaningful overhead on their own — whatever you call inside the loop is what you measure.
- **Per-call dispatch is the catch, and it is not symmetric between backends.** Anchoring against the race-car C-level numbers above (mean over the loop, M-series, default args):

  | N | Alloy C kernel | Alloy `cf.run()` (Python) | CasADi SX C kernel | CasADi `caf(DM)` (Python) |
  |---:|---:|---:|---:|---:|
  | 5 | 0.53 µs | 3.9 µs | 0.48 µs | 5.2 µs |
  | 100 | 10.3 µs | 15.1 µs | 9.3 µs | **79.5 µs** |

  Alloy's ctypes dispatch (`jit.py::CompiledFunction.run`: `np.asarray` inputs, allocate outputs + workspace, build pointer arrays, one FFI call, reshape) is a roughly **fixed ~3.3 µs floor** — its *relative* weight shrinks as the kernel grows (+47 % at N=100). CasADi's `caf(DM)` overhead instead **scales with nnz**, because it materializes a sparse `DM` return each call (≈ +70 µs at N=100). `caf(np, np)` is worse still (~120 µs at N=100) due to input conversion.

**Conclusion — use the right harness per question:**

- **Codegen-quality / scalability-vs-CasADi (these tables): keep the C++ harness.** A Python-level benchmark would report Alloy ~5× faster than CasADi SX at N=100 (15 vs 80 µs) when the generated code is actually within ~10 % (10.3 vs 9.3 µs). The 8× distortion is pure binding overhead, so the C++ harness is load-bearing here, not overhead-paranoia.
- **Alloy's own end-to-end Python latency, dispatch budgeting, and per-commit regression tracking: the Google Benchmark Python bindings are a great fit** — no per-cell compile, no 180 s timeouts, no source-size caps, and they measure the *realistic* cost paid when Alloy runs inside a Python solver loop. As a bonus they surface a genuinely favorable (and true) axis the C-only tables hide: Alloy's end-to-end Python dispatch is far lighter than CasADi's (15 vs 80 µs at N=100). The ~3.3 µs `cf.run()` floor is itself worth optimizing (preallocate workspace/outputs, cache the ctypes pointer arrays).

A minimal worked example used to live in `test_tracking_workload.py::test_tracking_eq_jac_python_gbench` (opt-in via `ALLOY_GBENCH=1`): it reproduced the equality-Jacobian cell through the Google Benchmark Python bindings, checked the result against the CasADi dense reference outside the timed loop, and recorded the dispatch time via `record_property`. It was dropped when the suite moved in-repo in `8b1dd3f`.

## Race-car equality Jacobian (`spjac:eq:z`) — historical

4-state, 2-control bicycle with slip-angle β=δ/2 and `tanh` rolling-resistance term, RK4 over the horizon. Decision vector size `(N+1)·6`, output size `(N+1)·4`. Dynamics:

```python
phi, v = x[2], x[3]
beta = 0.5 * delta
vx = v * cos(beta)
[v*cos(phi+beta), v*sin(phi+beta), v*sin(beta)/lr,
 (C_M0*throttle - (C_R0 + C_R1*vx + C_R2*vx*vx) * tanh(10*vx)) / M]
```

The Alloy fixture wrapped the interstage residual in a stage `Function` and assembled the equality
vector via `al.vmap(eq_interstage, length=N, ...)`. The current
`RaceCarConstraintJac` cell instead takes the full equality-plus-corridor Jacobian from the solver
descriptor. The table in this section predates that correction and remains an equality-only result
until the reference machine reruns the sweep.

Runtime (µs, mean from Google Benchmark `cpu_time`):

| N | Alloy | CasADi SX | CasADi MX |
|---:|---:|---:|---:|
| 1 | 0.09 | 0.09 | 0.13 |
| 5 | 0.53 | 0.48 | 0.74 |
| 10 | 1.05 | 0.91 | 1.50 |
| 25 | 2.53 | 2.28 | 3.75 |
| 50 | 5.24 | 4.70 | 8.39 |
| 100 | 10.27 | 9.28 | 18.72 |
| 200 | 21.20 | 18.44 | (compile >180 s — skipped) |
| 500 | 54.46 | 45.76 | (skipped_after_failure at N=200) |

Source size (KB):

| N | Alloy | CasADi SX | CasADi MX |
|---:|---:|---:|---:|
| 1 | 19.3 | 14.1 | 42.2 |
| 5 | 17.8 | 50.0 | 190.8 |
| 10 | 17.9 | 95.0 | 374.6 |
| 25 | 19.5 | 229.8 | 985.1 |
| 50 | 22.1 | 455.1 | 1996.0 |
| 100 | 28.3 | 907.0 | 4019.8 |
| 200 | 40.6 | 1810.8 | 8676.0 (gen, compile timed out) |
| 500 | 78.0 | 4528.9 | (skipped) |

Alloy source LOC stays essentially constant — 596 at N=1, then 481-482 from N=10 through N=500 — because the entire interstage Jacobian is one `for it` loop calling one const-seed JVP callee, and the byte count grows only with the constant `idx[]` gather/scatter tables (data, not code).

Workspace (doubles allocated by the generated function, reported via `SZ_W`):

| backend | workspace |
|---|---:|
| Alloy | 0 at N ≤ 25, then ≈ 63·N from N ≥ 50 (e.g. 31 504 at N=500) — spilled slots only |
| CasADi SX | 77 (constant) |
| CasADi MX | scales linearly: 199 at N=1 → 30 788 at N=200 |

Reading:

- Alloy and CasADi SX are within ~10 % of each other on runtime across the whole range, with Alloy slightly ahead at very small N and SX slightly ahead from N≈25 upward (per-iteration callee dispatch costs less than SX's scalar tape on the smallest cells but the gap closes as N grows).
- Alloy beats CasADi MX by ≈ 2× at every N where MX still compiles; MX times out at N=200 already.
- Alloy source at N=500 is 78 KB, **58× smaller than CasADi SX** (4.5 MB). LOC at N=500 is 482 — the same as at N=10.
- The per-cell codegen time at N=500 is 248 ms for Alloy vs 481 ms for CasADi SX (and >10 min for older variants of the Alloy path). That's the Python-AD-construction win from routing the JVP through one per-formal small graph instead of through the global unrolled tape.

## Neural-process MPC on the Furuta pendulum (`npmpc`)

A conditional-neural-process decoder — `9 → 32 → 32 → 2`, sigmoid, weights and latent code read out
of the parameter tail — evaluated at **every node of a prediction horizon**. This is the one workload
in the suite where a dense matmul sits inside the VMAP stage body, so it is the one that separates
loop-preserving lowering from scalar expansion most sharply. Formulation, vendored data and
closed-loop numbers live with the problem, in `benchmarks/problems/npmpc/README.md`; this section is
the sweep.

Both axes are gated per cell against a dense reference before any timing is recorded, and both
backends compile at `-O3` with the same compiler.

### Equality Jacobian (`spjac:eq:z`), horizon axis — historical

| N | alloy lines | SX lines | MX lines | alloy ns | SX ns | MX ns | SX/alloy | MX/alloy |
|---|---|---|---|---|---|---|---|---|
| 6 | **417** | 80 111 | 2 471 | 17 087 | 26 552 | 15 299 | 1.55 | 0.90 |
| 12 | **417** | 158 646 | 4 294 | 34 291 | 66 382 | 28 610 | 1.94 | 0.83 |
| 25 | **417** | 328 807 | 8 243 | 71 850 | 213 096 | 64 810 | 2.97 | 0.90 |
| 50 | **417** | 656 038 | 15 837 | 143 016 | 637 475 | 158 745 | 4.46 | 1.11 |
| 100 | **417** | 1 310 500 | 31 024 | 287 899 | *compile > 900 s* | 288 986 | — | 1.00 |
| 200 | **417** | *skipped* | 61 400 | 553 644 | — | *compile > 900 s* | — | — |

### Exact Lagrangian Hessian (`sphess:gamma:z`), horizon axis

| N | alloy lines | SX lines | MX lines | alloy ns | SX ns | MX ns | alloy compile | MX compile |
|---|---|---|---|---|---|---|---|---|
| 6 | **1041** | 293 452 | 4 924 | 31 969 | 62 620 | 26 512 | 0.8 s | 5 s |
| 12 | **1041** | 584 683 | 8 834 | 64 358 | 136 167 | 52 956 | 0.8 s | 13 s |
| 25 | **1041** | 1 215 891 | 17 355 | 134 614 | 273 704 | 116 385 | 0.8 s | 47 s |
| 50 | **1041** | *53 MB > cap* | 33 688 | 273 575 | — | 237 982 | 0.8 s | **227 s** |
| 100 | **1041** | *skipped* | 66 358 | 542 518 | — | *compile > 900 s* | 0.9 s | — |
| 200 | **1041** | *skipped* | — | 1 067 826 | — | — | 0.9 s | — |

### Decoder-width axis at N = 12

Untrained weights at every width, including 32, so the axis stays homogeneous: kernel timing and
generated code size depend on the graph's shape rather than on the numbers in it. These are code-size
and timing cells only, never accuracy cells.

Equality Jacobian — historical:

| W | alloy lines | SX lines | MX lines | alloy ns | SX ns | MX ns | MX/alloy |
|---|---|---|---|---|---|---|---|
| 16 | **417** | 45 496 | 4 156 | 10 054 | 9 078 | 8 457 | 0.84 |
| 32 | **417** | 158 646 | 4 294 | 33 201 | 52 663 | 27 023 | 0.81 |
| 64 | **417** | 590 768 | 4 762 | 145 176 | *compile > 900 s* | 126 174 | 0.87 |
| 128 | **417** | *skipped* | 6 466 | 597 719 | — | 815 594 | **1.36** |
| 256 | **417** | *skipped* | 12 946 | 2 890 248 | — | 7 539 505 | **2.61** |

Exact Lagrangian Hessian:

| W | alloy lines | SX lines | MX lines | alloy ns | SX ns | MX ns | MX/alloy |
|---|---|---|---|---|---|---|---|
| 16 | **1043** | 160 550 | 8 696 | 20 981 | 30 382 | 16 736 | 0.80 |
| 32 | **1041** | 584 612 | 8 834 | 65 064 | 131 291 | 52 576 | 0.81 |
| 64 | **1041** | *55 MB > cap* | 9 302 | 271 564 | — | 240 323 | 0.88 |
| 128 | **1041** | *skipped* | 11 006 | 1 284 817 | — | 1 382 204 | **1.08** |
| 256 | **1041** | *skipped* | 17 486 | 7 186 310 | — | 12 468 110 | **1.73** |

Reading, in descending order of confidence:

- **The code-size and compile-time result is unambiguous and large.** Alloy's source is 417 lines for
  the Jacobian at *every* point on *both* axes, and 1041 for the Hessian at every point but W = 16
  (1043 there), because the weights are read
  out of the parameter tail rather than baked in as literals and the stage body uses VMAP rather than
  unrolled. SX reaches 1.31 million lines at N = 100 — a factor of 3142 — and stops being compilable
  at all: clang exceeds a 15-minute budget there, and MX joins it at N = 200 for the Jacobian and
  N = 100 for the Hessian, where it already needs 227 s at N = 50. Alloy compiles the N = 200 Hessian
  in 0.94 s.
- **Against SX the runtime advantage is real**, and on the Jacobian it grows with the horizon: 1.55×
  at N = 6 to 4.46× at N = 50, after which SX drops out. On the Hessian it is flat instead, 1.96–2.12×
  over N = 6…25. The one cell where SX is ahead is the narrowest decoder, W = 16, at 0.90.
- **Against MX the horizon axis is a tie** — 0.82–1.11 with no trend — and the *width* axis is where
  alloy pulls ahead. Alloy's runtime grows 3.3×, 4.4×, 4.1×, 4.8× per doubling, which is the quadratic
  cost a matmul-dominated kernel should have. MX grows 891× over a 16× width increase against a ~256×
  quadratic expectation, so it crosses from 1.2× faster than alloy at the shipped width to 2.6×
  *slower* at W = 256. MX's source barely grows, because it keeps the matmuls as operations, so the
  blowup is in what its Jacobian does at runtime rather than in code size.

Alloy is *not* scalar-expanding these matmuls: the generated C contains real loop nests. The small
fixed handicap at narrow decoders and short horizons (0.80–0.90 across W = 16–64 and N = 6–25; by
N = 50 and N = 100 on the Jacobian axis it is gone, at 1.11 and 1.00) is ours, and the likeliest
cause is the AD mode — Alloy's `sparse_jacobian` colours columns only, and the per-stage block here is wider
than it is tall, which is the regime where a row-coloured or reverse sweep needs fewer passes. That
is a hypothesis with supporting structure, not a measured cause.

## Discrete-time HCBF safety filter (`unbumpercars`)

The Phase 5 driving workload: a centralized one-step CBF filter over `C` cars, with
`C(C-1)/2` hyperbolic pair rows plus four order-1 velocity wall rows per car, one L1 slack
per row, and a neural vehicle model inside the constraint. The formulation and its provenance live with the problem, in
`benchmarks/problems/unbumpercars/README.md`; this section is only the numbers.

> **Measurement date:** the tables below were recorded before the 2026-08-12 migration from
> position wall rows to order-1 velocity wall rows. They remain the latest backend-scaling
> measurements, but absolute oracle and solve times describe the older wall graph and need to
> be regenerated before being quoted for the current formulation.

**Two model choices matter for cost.** The default, `--filter-model dt`, predicts with the
natively discrete MLP; `--filter-model ct` predicts with an RK4 map of a
continuous-time network (`6 → 64 → 64 → 3`, SiLU, 4,803 weights, four evaluations per step for
the RK4 stages). The discrete MLP is `6 → 256 → 128 → 3`, smoothed ReLU, 35,075 weights, and
**one** evaluation per step because it is already a one-step map — 7.3x the weights but only
1.86x the multiply-accumulates.

Two different things are measured below, and they do not say the same thing: one isolated
kernel, and the whole oracle inside a real IPOPT loop.

### Isolated exact Lagrangian Hessian (`sphess:gamma:z:z`) — current

The current C=2/4/8 sweep uses the canonical discrete-time model and evaluates
the compact structurally sparse exact Lagrangian Hessian. Each cell checks
the reconstructed dense matrix against a CasADi MX reference before timing.
At C=8 it consumes the complete canonical closed-loop handoff: primal,
objective factor, constraint multipliers, state, desired input, model weights,
physics, and time step.

Measured 2026-08-20 on the current formulation (AMD Ryzen 9 7940HS, clang 20 at `-O3`,
single-threaded, both backends through the same Google Benchmark harness). CasADi SX is not run: it
was already past a 180 s compile budget at C=4 on the easier Jacobian kernel.

| C | backend | runtime | source | lines | workspace | compile |
|---:|---|---:|---:|---:|---:|---:|
| 2 | alloy | **654 µs** | 83.9 KB | 2 294 | 34 304 | 1.0 s |
| 2 | casadi_mx | 760 µs | 288.0 KB | 8 937 | 214 251 | 1.4 s |
| 4 | alloy | **1 315 µs** | 164.6 KB | 4 015 | 34 304 | 1.8 s |
| 4 | casadi_mx | 2 333 µs | 619.3 KB | 19 851 | 356 642 | 4.3 s |
| 8 | alloy | **2 681 µs** | 471.8 KB | 10 559 | 36 736 | 7.8 s |
| 8 | casadi_mx | 8 135 µs | 2 005.6 KB | 62 466 | 641 811 | 33.6 s |

**This is the suite's cleanest compiled-against-compiled oracle win, and it grows with the car
count**: 1.16× at C=2, 1.77× at C=4, **3.03× at C=8**, against 3.4–4.3× less source, 4–17× less
workspace and 1.4–4.3× less compile time. Alloy's workspace is essentially flat (34 304 → 36 736
doubles over a 4× car count) because the per-car neural dynamics stay a loop; CasADi MX's triples.

Alloy's advantage growing in C while its own workspace does not is the VMAP Hessian doing what it
is for. Note that alloy's source still grows here, because the `C(C-1)/2` pair rows are built by a
Python loop rather than `al.vmap` — the one place in the suite where *we* write the code-size growth
that the code-size claim argues against (tracked internally).

This kernel is also the right anchor for reading the problem's closed-loop numbers. At C=8 the closed
loop is 93% function evaluation, and CasADi's interpreted MX oracle set costs 126 ms per solve
against alloy's 50 ms — so the ~2.6× there and the 3.03× here are the same effect, and compiling
CasADi's oracles would close only part of it. The [fairness audit](fairness.md#unbumpercars-c8-40-steps-exact-lagrangian-hessian) has the
per-configuration closed-loop table.

### Isolated constraint Jacobian (`jac:g:z`) — historical

Historical sweep cells, Apple M-series, `-O3`, single-threaded. These cells pin the continuous-time model
(`FilterConfig(model="ct")`) rather than following the default, so they stay comparable with the
numbers recorded before the default changed:

| C | backend | runtime | source | lines | compile |
|---:|---|---:|---:|---:|---:|
| 2 | alloy | 61.0 µs | 59 KB | 849 | 0.6 s |
| 2 | casadi_sx | 64.3 µs | 5.4 MB | 236,847 | 144 s |
| 2 | casadi_mx | 65.1 µs | 185 KB | 5,619 | 2.8 s |
| 4 | alloy | 128.8 µs | 81 KB | 1,407 | 0.7 s |
| 4 | casadi_sx | — | 11.8 MB | 470,243 | **>180 s, timeout** |
| 4 | casadi_mx | 131.4 µs | 460 KB | 14,487 | 6.4 s |
| 8 | alloy | 263.4 µs | 170 KB | 3,459 | 2.6 s |
| 8 | casadi_mx | 263.3 µs | 1.4 MB | 45,312 | 18.4 s |

On this kernel **Alloy and CasADi MX tie on runtime** at every size, to within 1.5%. The win is
in the artefacts around it: 3–8x smaller C, 4–7x faster to compile, and zero workspace against
MX's 55k–179k doubles. CasADi SX is not viable here at all — 5.4 MB and 144 s to compile at
`C=2`, and past the 180 s budget by `C=4`, so the sweep short-circuits the larger cells.

### Per-solve, inside IPOPT

The kernel above is not what a solve actually calls: the solve wants `f`, `g`, `grad_f`, a
*sparse* `jac_g` and an exact sparse Lagrangian Hessian, several times per iteration. Mean per
solve over a 60-step episode, plant matched to the filter's model, exact Hessians:

| filter model | C | Alloy | CasADi | Alloy speedup | IPOPT iters (both) |
|---|---:|---:|---:|---:|---:|
| ct | 2 | 3.13 ms | 8.65 ms | 2.8x | 5.5 |
| ct | 4 | 8.09 ms | 25.31 ms | 3.1x | 7.8 |
| ct | 8 | 21.69 ms | 84.26 ms | 3.9x | 10.6 |
| dt | 2 | 3.78 ms | 17.13 ms | 4.5x | 5.9 |
| dt | 4 | 10.63 ms | 63.91 ms | 6.0x | 9.2 |
| dt | 8 | **32.02 ms** | **289.10 ms** | **9.0x** | 14.0 |

Iteration counts are identical between the two backends in every cell, so IPOPT walks the same
path and only the oracle provider differs.

Reading:

- **Alloy's advantage grows along both axes** — with car count (2.8x → 3.9x for `ct`, 4.5x → 9.0x
  for `dt`) and with network size (3.9x → 9.0x at `C=8`). The bigger the constraint graph, the
  more the oracle provider matters.
- **Function evaluation is where the solve lives**: 30.0 of Alloy's 32.0 ms and 279.5 of CasADi's
  289.1 ms at the largest cell. So the gap is essentially all oracle, not solver.
- **The gap is not in the dense Jacobian**, which ties above. It is in the exact Lagrangian
  Hessian and the call path: Alloy's generated C wrapper calls the kernels directly, where CasADi
  re-enters its own machinery per call. Measured per-call at `C=8` on the `ct` model, CasADi's
  `hess_lag` alone is 3.7 ms against 0.2–0.8 ms for its other oracle outputs.
- **The heavier network costs Alloy 1.5x and CasADi 3.4x** per solve at `C=8` (21.7 → 32.0 ms
  against 84.3 → 289.1 ms), for 7.3x the weights.
- **Superlinear in `C` for both**, as expected — the pair rows grow as `C(C-1)/2` and the
  iteration count grows too: Alloy ≈2.6x (`ct`) and ≈2.8x (`dt`) per doubling of `C`, CasADi
  ≈3.1x and ≈3.9x.
- **Build cost** is the one place Alloy pays: 5.0 s against CasADi's 2.2 s at `C=8` on `ct`, and
  3.8 s against 4.3 s on `dt`. It is a once-per-configuration cost, and the `.so` is cached.

One caveat on the `dt`-versus-`ct` rows: each is measured against *its own* plant, so they are
two coherent configurations rather than a controlled A/B. Against the faithful (discrete) plant,
which is the default, the `dt` filter is both safer and *faster* than the `ct` one — 31.7 ms
against 37.8 ms — because the mismatched filter needs 19.3 iterations to the matched filter's
14.1. The problem README has that comparison.

### How to reproduce

```bash
# Current isolated exact-Hessian cells
uv run python benchmarks/run.py sweep --workloads unbumpercars --out /tmp/sweep.csv

# Per-solve (the second table), one cell per invocation
uv run python -m benchmarks.problems.unbumpercars.run_closed_loop \
  --solver ipopt --oracle both --filter-model dt --ncars 8 --steps 60
```

Canonical unified-runner episodes write under
`benchmarks/results/closed-loop/unbumpercars/<solver>+<oracle>/`; unified `--smoke`
episodes use `benchmarks/results/smoke/closed-loop/unbumpercars/<solver>+<oracle>/` so
they cannot replace the canonical handoff. The direct problem module writes to
the canonical location unless given another output directory.

## Unbumpercars inequality Jacobian (`spjac:ineq:u`) — historical

> **The workload measured here no longer exists.** Its implementation and
> the retired `test_unbumpercars_workload.py` were removed in `1b03820`
> (2026-08-10): the fixture was the only thing exercising gather-fed and chained VMAPs,
> and its two numeric tests had been silently skipping because the
> `model_kinematic_mlp.pth` checkpoint is not in the repo — so it read as coverage
> without being any. The pattern moved to
> `tests/integration/test_vmap.py::test_gather_fed_chained_vmaps_spjac_and_sphess_match_dense`,
> which runs unconditionally on small artificial cases.
>
> The current workload reuses the canonical `unbumpercars` ID at
> `benchmarks/problems/unbumpercars/`. It keeps `al.vmap` for
> the per-car neural dynamics but builds its `C(C-1)/2` pair rows with an unrolled Python
> loop, so the constant-LOC property below does **not** hold for it: its `spjac:g:z`
> kernel goes 845 → 1403 → 3455 lines for `C = 2 → 4 → 8`. Porting it back onto the
> gather-fed shape is tracked internally; the numbers below are what that
> port is expected to recover, and are kept for that reason.

Official-size MLP (`256 → 128 → 3` with the example `model_kinematic_mlp.pth` weights), RK4 pose update per car, pairwise C3BF + per-car wall residuals, slack column. Decision vector size `2C + 1`, constraint count `C(C-1)/2 + 4C` (quadratic in `C`).

`unbumpercars_ineq_function` was built as three `ExprOp.VMAP` nodes:

1. `dynamics_fn` mapped over `C` packed states / `C` packed inputs (with `pw` broadcast).
2. `pair_c3bf_fn` mapped over `C(C-1)/2` `(i,j)` pairs, fed by `al.gather` from two constant index tables — one per side of the pair, each built as `concatenate([arange(NSTATE) + k * NSTATE for k in bodies])` over the strict upper triangle, so a gather produces exactly the contiguous `NSTATE` block per iteration that a VMAP wants. The same two tables gather both the parameter states and the first VMAP's output, which is what makes it a VMAP → gather → VMAP chain.
3. `wall_residuals_fn` mapped over `C` cars.

CasADi SX is dropped past C=2 because at C=2 it already takes >180 s to compile a 12 MB source file; the sweep records that and short-circuits larger C for SX.

Runtime (µs):

| C | Alloy | CasADi SX | CasADi MX | Alloy speedup over MX |
|---:|---:|---:|---:|---:|
| 2 | 96.1 | (compile >180 s — skipped) | 263.4 | 2.74× |
| 4 | 197.5 | (skipped after C=2) | 518.1 | 2.62× |
| 8 | 396.9 | (skipped) | 1058.6 | 2.67× |
| 16 | 875.1 | (skipped) | 2207.1 | 2.52× |
| 32 | 2254.4 | (skipped) | 4237.5 | 1.88× |

Source size (KB):

| C | Alloy | CasADi MX |
|---:|---:|---:|
| 2 | 47.2 | 211.8 |
| 4 | 53.6 | 326.4 |
| 8 | 91.5 | 739.3 |
| 16 | 338.1 | 2495.1 |
| 32 | 2406.1 | 9883.2 |

Alloy source LOC ranges from **1601 to 1889 across C=2..32** — the only thing that grows with `C` is the constant gather/scatter index tables.

Workspace (doubles):

| backend | C=2 | C=4 | C=8 | C=16 | C=32 |
|---|---:|---:|---:|---:|---:|
| Alloy | 0 | 0 | 6 664 | 110 616 | 1 092 752 |
| CasADi MX | 141 500 | 212 674 | 355 264 | 641 476 | 1 218 028 |

Reading:

- Alloy beats CasADi MX by a consistent **~2.5-2.7×** through C=16, narrowing to **1.88×** at C=32 — see the jump discussion below. Alloy's colored sparse Jacobian shares the per-car dynamics callee across colors and applies it inside a `for` loop, while MX clones the per-iteration graph through every JVP step (workspace and source both scale linearly in `C`).
- **Both Alloy and MX now compile at C=32**: Alloy at 1.6 s codegen + 1.1 s compile (source 2.4 MB); MX at 1.5 s codegen + 100 s compile (source 9.9 MB). The old unrolled Alloy path needed 18.5 MB of source at C=32 and timed out at compile.
- The 1.09 M-double Alloy workspace at C=32 lives in `w[]`; the wrapper allocates it `static` so the inner benchmark loop never goes through `malloc`. Without the spill threshold this would be ~8.7 MB of stack arrays and segfault under the 8 MB subprocess default `ulimit -s`.

### Why does runtime jump between C=16 and C=32?

Both backends slow down per-nnz between these two sizes:

| transition | Alloy ratio | MX ratio | nnz ratio |
|---|---:|---:|---:|
| C=4 → 8 | 2.01× | 2.04× | 3.03× |
| C=8 → 16 | 2.20× | 2.08× | 3.36× |
| C=16 → 32 | 2.58× | 1.92× | 3.62× |

So both backends are sub-linear in `nnz`, but the slope flattens noticeably at C=32. The bench runs on an Apple M-series chip with two CPU tiers (perf cores: L1-D 128 KB, L1-I 192 KB, L2 16 MB shared by 6 cores; efficiency cores: L1-D 64 KB, L1-I 128 KB, L2 4 MB shared by 4 cores), and `Run on` reports the smaller efficiency-core caches. Working-set sizes are:

| C | alloy workspace | alloy source | combined | vs L2 (eff 4 MB) |
|---:|---:|---:|---:|---:|
| 8 | 52 KB | 91 KB | 0.14 MB | fits |
| 16 | 885 KB | 338 KB | 1.22 MB | fits |
| 32 | 8.74 MB | 2.41 MB | 11.2 MB | spills (eff), fits (perf) |

So between C=16 and C=32 the combined code+data footprint goes from 1.2 MB to 11 MB — that's the first cell that exceeds the 4 MB efficiency-core L2. The macOS scheduler can place the bench thread on either core type, and on efficiency cores (or under perf-core L2 sharing with other threads) we start eating L2 misses. MX experiences a milder version of the same effect — its workspace was already past 4 MB at C=16 (5.1 MB) so the C=16→32 step doesn't cross a new boundary on the data side, only on the code side (2.5 MB → 9.9 MB).

Treat this as a hardware-locality story rather than a backend ceiling: the alloy code at C=32 is still 4× smaller than MX (2.4 MB vs 9.9 MB) and ~2× faster, just not the constant 2.7× we see at smaller `C`.

## Comparison with `tracking-nmpc-benchmarks` worktree

Tracking, N=50:

| metric | experiment2 (anvil, N=50, simple 4-state bicycle) | this sweep (alloy, N=50, slip-angle + tanh drag) |
|---|---:|---:|
| dense `jac:eq:z` | 72.1 µs | not measured (only spjac path benchmarked) |
| colored `spjac:eq:z` | 5.06 µs | 5.24 µs |
| `spjac_unroll` (CONST basis) | 103 µs (~87 s codegen) | n/a — alloy does not generate this path |
| `multistage` (per-block Jac) | **1.53 µs**, O(1) source | not yet — see plan |
| CasADi SX | 1.34 µs (experiment1, simple dynamics) | 4.70 µs (this sweep, fancier dynamics) |
| CasADi MX | 5.83 µs (experiment1) | 8.39 µs (this sweep) |

The 3-4× absolute-runtime gap between this sweep's SX/MX column and experiment1's matching column is from the heavier dynamics in our fixture (β-slip + tanh drag + division by `lr`), not from a backend regression — confirmed by inspecting both `bicycle_cont` implementations side by side.

For unbumpercars vs. `experiment3` of the worktree:

| C | alloy (this sweep) | anvil_ineq_jac (worktree, same colored-sparse approach) | CasADi MX (this sweep) | CasADi MX (worktree) |
|---:|---:|---:|---:|---:|
| 2 | 96.1 µs | 322 µs | 263 µs | 189 µs |
| 4 | 197.5 µs | 603 µs | 518 µs | 333 µs |
| 8 | 396.9 µs | 1417 µs | 1059 µs | 618 µs |
| 16 | 875.1 µs | 3978 µs | 2207 µs | 1179 µs |
| 32 | 2254.4 µs | 12273 µs | 4237 µs | 2329 µs |

Alloy beats the worktree's `anvil_ineq_jac` (which uses the same conceptual colored-sparse approach) by ~3.3-5.4× across the C=2..32 range, mostly thanks to the VMAP structure (loop-shaped per-car dynamics + pair C3BF), the sparse-constant matvec, scalar/vector inlining, the gather peephole, and the workspace spill that lets C=32 compile at all. CasADi MX's apparent disadvantage vs. its own worktree numbers is partly the heavier MLP fixture (official `256→128→3` weights vs. the worktree's simpler reduced MLP).

## Comparison with CasADi `Function.map(N, "serial")` (tracking)

For curiosity we ran the same workload through CasADi's own loop-preserving primitive: the interstage residual is wrapped in a stage `Function`, mapped over `N` via `.map(N, "serial")`, then `casadi.jacobian` is applied to the assembled equality vector.

| N   | CasADi SX unrolled  | CasADi SX with `.map`  | CasADi MX with `.map`        |
| --- | ------------------- | ---------------------- | ---------------------------- |
| 10  | 1030 ns / 96.97 KB  | 1026 ns / 96.88 KB     | 2113 ns / 109.1 KB, sz_w=4205 |
| 50  | 4574 ns / 466 KB    | 4534 ns / 466 KB       | 10190 ns / 196 KB, sz_w=20405 |
| 100 | 8951 ns / 928 KB    | 8946 ns / 928 KB       | 20475 ns / 317 KB, sz_w=40331 |
| 200 | 17830 ns / 1.85 MB  | 17879 ns / 1.85 MB     | 41339 ns / 560 KB, sz_w=80506 |

Reading:

- **CasADi SX with `.map` is byte-for-byte indistinguishable from fully unrolling** — SX flattens to scalars at codegen, so the `.map` abstraction does not survive past graph construction.
- **CasADi MX with `.map`** does keep a loop shape and is the most source-efficient CasADi configuration in the table, but `sz_w` grows linearly (4205 → 80506 doubles at N=200) and runtime is ~2× slower than SX/unrolled at every N. The MX evaluator's per-iteration workspace plumbing eats the win.
- **alloy** at N=200 is **45× less source than SX and 13× less than MX with `.map`**, runs at ~21 µs (close to SX, ~2× faster than MX `.map`), and uses 12 604 doubles of workspace — about 6× less than MX `.map`.

## How to reproduce

```bash
# Every headline grid on this page and the closed loops on the results index, into one directory
uv run benchmarks/run.py study --out-dir benchmarks/results/followup/<date>

# Only the kernel sweeps, or only one problem's sweep
uv run benchmarks/run.py study --out-dir benchmarks/results/followup/<date> --only sweep
uv run benchmarks/run.py study --out-dir benchmarks/results/followup/<date> --only sweep --problems race_cars

# Re-render the Markdown tables of an existing study directory
uv run benchmarks/run.py report benchmarks/results/followup/<date>

# Full exact-Hessian sweep with default cells
uv run benchmarks/run.py sweep --out benchmarks/results/sweep/scalability.csv

# Long-paper race-car Jacobian row
uv run benchmarks/run.py sweep --workloads race_cars_jac --out /tmp/race_cars-jac.csv

# Custom horizons / car counts / per-cell compile timeout
uv run benchmarks/run.py sweep \
    --workloads race_cars \
    --sizes 1,10,50,200 \
    --compile-timeout 60 \
    --out /tmp/quick.csv
```

Cells that hit the size cap or the per-cell compile timeout get `skipped_size` or `timeout` in
`compile_status`. A backend that does not apply to a workload gets `not_applicable`. After a backend
times out or exceeds the size cap, larger cells get `skipped_after_failure`. Runtime errors and parse
failures appear in `runtime_status`.

Race-car N=1000 used to appear in this table; it is dropped from the default cell grid because the bench-time dense reference (single-seed JVP × 6006 columns through the unrolled fixture) is the bottleneck rather than alloy itself — supply `--workloads race_cars --sizes 1000` to add it back when you're willing to wait several minutes.

## Continuous-time CBF safety filter — historical

> **The fixture measured here no longer exists.** Both
> the retired `test_safety_filter_workload.py` and
> `benchmarks/alloy_safety_filter_benchmark.py` are gone, so nothing below can be
> regenerated. It is kept because the input-affine-versus-fully-nonlinear reading still
> explains why the live workload is shaped the way it is: the dense `jac:ineq:u` blow-up on
> the nonlinear variant (4.4 ms at N=8 against an 89 µs forward) is exactly why
> `unbumpercars` calls `spjac`/`sphess` rather than dense factories. The successor is
> [Discrete-time HCBF safety filter](#discrete-time-hcbf-safety-filter-unbumpercars)
> above, whose per-solve numbers supersede these per-kernel ones. Note also that the
> Lagrangian-Hessian limitation this section records as blocking has since been closed —
> `sphess` through `ExprOp.VMAP` works and is what the live workload uses.

Fixture: the retired `test_safety_filter_workload.py` built the two variants of the retired
continuous-time HOCBF design study (both the fixture and that design are gone; see
`benchmarks/problems/unbumpercars/README.md` for the filter that exists):

- **Input-affine** — per-car velocity net `f_nn(x) + g_nn(x)·u` with a shared MLP body (`7 → 256 → 128`, SiLU) and two heads (drift 128→3, control 128→6). The constraint vector is the HOCBF residual `ḧ_ij + (γ1+γ2)·ḣ_ij + γ1·γ2·h_ij + s` over all `N(N-1)/2` pairs plus 4 wall residuals per car, with the slack term `s` shared. Cost is `Σ (u_i − u_des_i)^T Q (u_i − u_des_i) + M·s²`.
- **Fully nonlinear** — same shape but the velocity block is a single `f_nn(x, u)` MLP (`9 → 256 → 128 → 3`); the rest of the chain (pose kinematics, slack, HOCBF combination, cost) is unchanged.

The driver `benchmarks/alloy_safety_filter_benchmark.py` derives, for each `(ncars, variant)` cell, five single-output Alloy `Function`s — forward `ineq`, forward `cost`, dense `jac:ineq:u`, sparse `spjac:ineq:u`, and `grad:cost:u`. The sparse Lagrangian Hessian (`sphess:gamma:u:u`) is also requested but currently fails the `ExprOp.VMAP` reverse-mode path in `alloy.ad.reverse._local_vjp`, so it is caught and skipped per cell rather than working around the IR limitation here.

Each Function is rendered to C, compared against the Python interpreter on a deterministic input vector, and timed by Google Benchmark on the universal ABI entry point.

Runtime (Apple M-series, `-O3`, single-threaded; ns/call from `cpu_time`):

| Cell             | ineq   | cost  | jac\_u    | spjac\_u | grad\_cost\_u |
|------------------|-------:|------:|----------:|---------:|--------------:|
| affine N=2       | 22 µs  | 1.5 ns | 22 µs    | 22 µs    | 1.2 ns        |
| affine N=4       | 44 µs  | 1.9 ns | 44 µs    | 44 µs    | 1.5 ns        |
| affine N=8       | 89 µs  | 2.7 ns | 90 µs    | 88 µs    | 2.3 ns        |
| nonlin N=2       | 22 µs  | 1.5 ns | **288 µs** | 64 µs  | 1.3 ns        |
| nonlin N=4       | 44 µs  | 1.9 ns | **1.19 ms** | 129 µs | 1.5 ns       |
| nonlin N=8       | 89 µs  | 2.7 ns | **4.40 ms** | 267 µs | 2.3 ns       |

Source size / codegen (bytes, lines, NNZ for sparse Jacobian; codegen ms is Python-side):

| Cell             | ineq           | jac\_u           | spjac\_u (nnz)         |
|------------------|----------------|------------------|-----------------------:|
| affine N=2       | 16 KB / 559 L  | 27 KB / 875 L    | 12 KB / 422 L (20)     |
| affine N=4       | 28 KB / 1024 L | 96 KB / 3034 L   | 27 KB / 1120 L (56)    |
| affine N=8       | 69 KB / 2569 L | 493 KB / 14370 L | 133 KB / 5455 L (176)  |
| nonlin N=2       | 15 KB / 516 L  | 27 KB / 889 L    | 16 KB / 546 L (20)     |
| nonlin N=4       | 26 KB / 951 L  | 88 KB / 2817 L   | 32 KB / 1237 L (56)    |
| nonlin N=8       | 66 KB / 2436 L | 452 KB / 13257 L | 136 KB / 5464 L (176)  |

Reading:

- **Forward `ineq` is linear in ncars** and identical between variants at the same size — both go through one per-car MLP forward, which is what dominates (~10 µs/car at this network sizing). Pair count grows as N(N-1)/2 but adds negligible work; the floor is the network.
- **Cost and `grad:cost:u` are essentially free** (single-digit ns). The quadratic cost touches no MLP and only `O(N)` doubles.
- **Affine `jac:ineq:u` runs in the same envelope as the forward** — expected, because the constraint is linear in u and the Jacobian rows `b_ij^i = 2·Δπ^T·(∂κ_π/∂v)·g_nn(x_i)` just reuse the per-car NN outputs. The QP path is essentially "one forward and you have A".
- **Nonlinear `jac:ineq:u` is the obvious hotspot** — dense Jacobian seeds u (size `2·ncars`) through the MLP, which is roughly `2·ncars` forward passes; that is exactly the ~50× scaling we see at N=8 (`4.4 ms` vs `89 µs`).
- **`spjac:ineq:u` recovers most of that loss** for the nonlinear case (`267 µs` at N=8, ~3× the forward instead of ~50×) because Alloy's column coloring reduces the seed count to the number of structurally distinct columns. This is the right object for an NLP solver loop to call per IPOPT iteration.
- The Lagrangian Hessian wrt u would be the other per-iteration object for the NLP path; it is the most natural next target once `ExprOp.VMAP` is added to the reverse-mode AD (`alloy/ad/reverse.py::_local_vjp`).

How these were reproduced, at the time. **None of these commands work now** —
`alloy_safety_filter_benchmark.py` was removed along with the formulation it measured, and is
recorded here only so the numbers above can be read in context:

```bash
# Default sweep: ncars=2,4,8 over both variants
uv run python benchmarks/alloy_safety_filter_benchmark.py --ncars 2 4 8 --clean

# Just one variant or one size
uv run python benchmarks/alloy_safety_filter_benchmark.py --variant affine --ncars 8

# Code + source size stats only (skip compile + run)
uv run python benchmarks/alloy_safety_filter_benchmark.py --ncars 2 4 8 --stats-only
```
