# Scaly scalability sweep

These are all exact sparse Lagrangian Hessian cells from the study that started and finished on 2026-09-23. The [results overview](index.md) summarizes the canonical points, and the
[fairness audit](fairness.md) defines the comparison.

All cells use the libmvec policy from the [fairness audit](fairness.md). The sweep compiler is
Clang 20.1.8 with `-O3 -march=native -fno-math-errno -fveclib=libmvec`, and both providers link
`-lmvec -lm`.

Each cell compiles one generated C kernel into a separate Google Benchmark binary. Before timing,
the binary scatters the compact Hessian into a dense matrix and compares it with an independent
reference. A failed check produces no timing.

Each timing aggregates five fresh processes. Mean is the mean of their Google Benchmark timings,
and CV is the sample coefficient of variation. Executable bytes exclude static metadata. Workspace
counts caller-owned doubles. Kernel compile time includes compilation of the generated kernel, but
not the benchmark wrapper or linker.

The harness applies a 180-second compile limit and a 50 MiB generated-source limit. After a backend
fails at one size, the harness skips larger sizes for that backend because source size and compile
cost increase with the problem axis. The backend names are:

- `scaly`: Scaly's sparse generated C.
- `casadi_sx`: an unrolled CasADi SX graph.
- `casadi_mx`: a CasADi MX graph.
- `casadi_call_mx`: MX with first-class calls.
- `casadi_map_sx`: serial `Function.map` over SX.
- `casadi_mx_gemm*`: MX matrix products with the default, classic, or BLASFEO lowering.

## Hanging chain of masses (`chain`)

75 rows: 50 ok, 22 skipped_after_failure, 3 timeout.

| Size | Backend | Successful processes | Mean, µs | CV, % | Executable bytes | Workspace, doubles | Kernel compile mean, ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3 | `scaly` | 5/5 | 21.810 | 1.02 | 701062 | 26028 | 2748.0 |
| 3 | `casadi_sx` | 5/5 | 26.052 | 0.26 | 3330455 | 188 | 19046.2 |
| 3 | `casadi_mx` | 0/5 | timeout; skipped_after_failure |  | 9410547 | 23678 |  |
| 3 | `casadi_call_mx` | 5/5 | 180.579 | 1.31 | 602188 | 27321 | 12958.6 |
| 3 | `casadi_map_sx` | 5/5 | 123.867 | 0.86 | 556793 | 94736 | 4620.4 |
| 5 | `scaly` | 5/5 | 100.037 | 0.36 | 1544268 | 0 | 4081.6 |
| 5 | `casadi_sx` | 5/5 | 82.822 | 1.00 | 10330465 | 335 | 93339.7 |
| 5 | `casadi_mx` | 0/5 | skipped_after_failure |  |  |  |  |
| 5 | `casadi_call_mx` | 5/5 | 839.189 | 1.70 | 2097970 | 100698 | 79400.1 |
| 5 | `casadi_map_sx` | 5/5 | 630.884 | 0.82 | 2561051 | 349353 | 10358.3 |
| 9 | `scaly` | 5/5 | 282.649 | 0.06 | 3842302 | 0 | 18748.6 |
| 9 | `casadi_sx` | 0/5 | timeout; skipped_after_failure |  | 24350782 | 897 |  |
| 9 | `casadi_mx` | 0/5 | skipped_after_failure |  |  |  |  |
| 9 | `casadi_call_mx` | 0/5 | timeout; skipped_after_failure |  | 6664454 | 280960 |  |
| 9 | `casadi_map_sx` | 5/5 | 2473.311 | 1.19 | 9650809 | 999844 | 59732.8 |

## Neural-process model predictive control (`npmpc`)

240 rows: 175 ok, 59 skipped_after_failure, 6 timeout.

| Size | Backend | Successful processes | Mean, µs | CV, % | Executable bytes | Workspace, doubles | Kernel compile mean, ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 6 | `scaly` | 5/5 | 12.982 | 0.68 | 774894 | 0 | 997.0 |
| 6 | `casadi_sx` | 5/5 | 71.763 | 0.65 | 7564935 | 15913 | 104085.3 |
| 6 | `casadi_mx` | 5/5 | 19.488 | 0.37 | 151448 | 7570 | 5129.3 |
| 6 | `casadi_call_mx` | 5/5 | 147.167 | 1.47 | 232343 | 46861 | 6814.4 |
| 6 | `casadi_map_sx` | 5/5 | 199.968 | 1.31 | 4261664 | 341213 | 135897.2 |
| 6 | `casadi_mx_gemm` | 5/5 | 44.391 | 0.81 | 109496 | 8660 | 2442.9 |
| 6 | `casadi_mx_gemm_classic` | 5/5 | 44.228 | 0.72 | 109504 | 8660 | 2425.2 |
| 6 | `casadi_mx_gemm_blasfeo` | 5/5 | 44.588 | 1.16 | 109504 | 8660 | 2473.4 |
| 12 | `scaly` | 5/5 | 22.786 | 4.97 | 781549 | 0 | 1606.3 |
| 12 | `casadi_sx` | 0/5 | timeout; skipped_after_failure |  | 15063206 | 30463 |  |
| 12 | `casadi_mx` | 5/5 | 40.525 | 0.58 | 276488 | 9813 | 12782.8 |
| 12 | `casadi_call_mx` | 5/5 | 298.999 | 1.38 | 371542 | 68072 | 10393.8 |
| 12 | `casadi_map_sx` | 5/5 | 465.755 | 1.51 | 5788748 | 774841 | 178409.5 |
| 12 | `casadi_mx_gemm` | 5/5 | 100.771 | 0.68 | 189660 | 13031 | 5218.9 |
| 12 | `casadi_mx_gemm_classic` | 5/5 | 101.666 | 1.11 | 189668 | 13031 | 5250.0 |
| 12 | `casadi_mx_gemm_blasfeo` | 5/5 | 102.873 | 0.83 | 189668 | 13031 | 5130.8 |
| 25 | `scaly` | 5/5 | 45.339 | 0.87 | 779043 | 2700 | 1670.1 |
| 25 | `casadi_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 25 | `casadi_mx` | 5/5 | 91.355 | 7.91 | 545609 | 15595 | 41425.1 |
| 25 | `casadi_call_mx` | 5/5 | 617.398 | 0.84 | 424096 | 89139 | 11497.3 |
| 25 | `casadi_map_sx` | 5/5 | 998.692 | 0.94 | 7858327 | 1603408 | 175973.1 |
| 25 | `casadi_mx_gemm` | 5/5 | 212.648 | 1.14 | 343012 | 22648 | 8287.7 |
| 25 | `casadi_mx_gemm_classic` | 5/5 | 212.740 | 1.00 | 343020 | 22648 | 8278.6 |
| 25 | `casadi_mx_gemm_blasfeo` | 5/5 | 213.613 | 1.86 | 343020 | 22648 | 8299.6 |
| 50 | `scaly` | 5/5 | 93.701 | 0.80 | 781656 | 11670 | 1717.2 |
| 50 | `casadi_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 50 | `casadi_mx` | 5/5 | 169.963 | 0.09 | 1063583 | 25782 | 166891.6 |
| 50 | `casadi_call_mx` | 5/5 | 1234.066 | 0.25 | 520231 | 130185 | 13544.1 |
| 50 | `casadi_map_sx` | 0/5 | timeout; skipped_after_failure |  | 11844334 | 3196692 |  |
| 50 | `casadi_mx_gemm` | 5/5 | 431.233 | 0.33 | 638917 | 41169 | 25791.5 |
| 50 | `casadi_mx_gemm_classic` | 5/5 | 437.764 | 1.44 | 638925 | 41169 | 25877.9 |
| 50 | `casadi_mx_gemm_blasfeo` | 5/5 | 431.615 | 0.35 | 638925 | 41169 | 25947.2 |
| 100 | `scaly` | 5/5 | 143.361 | 0.77 | 788167 | 23270 | 1750.7 |
| 100 | `casadi_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 100 | `casadi_mx` | 0/5 | timeout; skipped_after_failure |  | 2191474 | 46937 |  |
| 100 | `casadi_call_mx` | 5/5 | 2462.045 | 0.34 | 712196 | 210580 | 21270.5 |
| 100 | `casadi_map_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 100 | `casadi_mx_gemm` | 5/5 | 882.141 | 0.67 | 1232183 | 77961 | 92099.4 |
| 100 | `casadi_mx_gemm_classic` | 5/5 | 883.475 | 1.19 | 1232191 | 77961 | 93788.5 |
| 100 | `casadi_mx_gemm_blasfeo` | 5/5 | 881.117 | 1.05 | 1232191 | 77961 | 92053.8 |
| 200 | `scaly` | 5/5 | 235.705 | 1.20 | 788359 | 51870 | 1061.1 |
| 200 | `casadi_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 200 | `casadi_mx` | 0/5 | skipped_after_failure |  |  |  |  |
| 200 | `casadi_call_mx` | 5/5 | 4977.875 | 1.13 | 1127544 | 371260 | 51694.3 |
| 200 | `casadi_map_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 200 | `casadi_mx_gemm` | 0/5 | timeout; skipped_after_failure |  | 2524675 | 151836 |  |
| 200 | `casadi_mx_gemm_classic` | 0/5 | timeout; skipped_after_failure |  | 2524683 | 151836 |  |
| 200 | `casadi_mx_gemm_blasfeo` | 0/5 | timeout; skipped_after_failure |  | 2524683 | 151836 |  |

## Race-car model predictive control (`race_cars`)

225 rows: 210 ok, 14 skipped_after_failure, 1 timeout.

| Size | Backend | Successful processes | Mean, µs | CV, % | Executable bytes | Workspace, doubles | Kernel compile mean, ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `scaly` | 5/5 | 0.360 | 0.83 | 22909 | 0 | 189.3 |
| 1 | `casadi_sx` | 5/5 | 0.400 | 0.60 | 25755 | 108 | 168.8 |
| 1 | `casadi_mx` | 5/5 | 0.506 | 0.19 | 95400 | 304 | 397.8 |
| 1 | `casadi_call_mx` | 5/5 | 0.953 | 0.87 | 228004 | 1067 | 792.8 |
| 1 | `casadi_map_sx` | 5/5 | 0.766 | 0.59 | 94438 | 826 | 398.9 |
| 5 | `scaly` | 5/5 | 0.861 | 0.78 | 119368 | 0 | 310.3 |
| 5 | `casadi_sx` | 5/5 | 1.936 | 0.57 | 107717 | 119 | 340.9 |
| 5 | `casadi_mx` | 5/5 | 2.806 | 0.71 | 441760 | 1085 | 1766.3 |
| 5 | `casadi_call_mx` | 5/5 | 5.045 | 0.85 | 248703 | 2244 | 1814.0 |
| 5 | `casadi_map_sx` | 5/5 | 5.312 | 0.61 | 120288 | 5914 | 1991.4 |
| 10 | `scaly` | 5/5 | 1.036 | 1.10 | 120390 | 0 | 326.7 |
| 10 | `casadi_sx` | 5/5 | 3.810 | 0.10 | 209958 | 119 | 576.8 |
| 10 | `casadi_mx` | 5/5 | 5.703 | 1.08 | 879492 | 2065 | 4316.3 |
| 10 | `casadi_call_mx` | 5/5 | 9.893 | 0.77 | 269955 | 3494 | 2549.4 |
| 10 | `casadi_map_sx` | 5/5 | 10.337 | 0.60 | 142699 | 11334 | 1952.4 |
| 25 | `scaly` | 5/5 | 2.015 | 2.34 | 120448 | 0 | 461.8 |
| 25 | `casadi_sx` | 5/5 | 9.640 | 0.62 | 516870 | 119 | 1325.8 |
| 25 | `casadi_mx` | 5/5 | 14.271 | 0.69 | 2360004 | 5020 | 23502.9 |
| 25 | `casadi_call_mx` | 5/5 | 25.095 | 2.32 | 334019 | 7484 | 2029.3 |
| 25 | `casadi_map_sx` | 5/5 | 25.751 | 0.87 | 206150 | 27594 | 1158.5 |
| 40 | `scaly` | 5/5 | 3.021 | 3.11 | 120448 | 0 | 459.7 |
| 40 | `casadi_sx` | 5/5 | 15.525 | 0.39 | 823845 | 119 | 2091.0 |
| 40 | `casadi_mx` | 5/5 | 23.047 | 0.64 | 3799659 | 7975 | 61333.5 |
| 40 | `casadi_call_mx` | 5/5 | 39.922 | 1.29 | 398502 | 11414 | 2568.5 |
| 40 | `casadi_map_sx` | 5/5 | 41.497 | 0.32 | 273406 | 43854 | 1274.9 |
| 50 | `scaly` | 5/5 | 3.495 | 1.50 | 120448 | 0 | 370.1 |
| 50 | `casadi_sx` | 5/5 | 19.405 | 0.26 | 1028495 | 119 | 2681.9 |
| 50 | `casadi_mx` | 5/5 | 28.989 | 0.83 | 4762326 | 9945 | 100443.8 |
| 50 | `casadi_call_mx` | 5/5 | 49.855 | 1.00 | 441709 | 14034 | 7373.2 |
| 50 | `casadi_map_sx` | 5/5 | 51.736 | 0.12 | 318176 | 54694 | 5457.8 |
| 100 | `scaly` | 5/5 | 6.940 | 5.66 | 121470 | 0 | 438.2 |
| 100 | `casadi_sx` | 5/5 | 39.596 | 1.38 | 2052051 | 119 | 6007.0 |
| 100 | `casadi_mx` | 0/5 | timeout; skipped_after_failure |  | 9614050 | 19795 |  |
| 100 | `casadi_call_mx` | 5/5 | 99.014 | 0.99 | 679651 | 27134 | 7534.7 |
| 100 | `casadi_map_sx` | 5/5 | 104.994 | 0.87 | 543857 | 108894 | 3368.6 |
| 200 | `scaly` | 5/5 | 12.502 | 1.69 | 121506 | 0 | 465.7 |
| 200 | `casadi_sx` | 5/5 | 81.206 | 0.39 | 4100194 | 119 | 14528.0 |
| 200 | `casadi_mx` | 0/5 | skipped_after_failure |  |  |  |  |
| 200 | `casadi_call_mx` | 5/5 | 199.731 | 0.92 | 1143240 | 53874 | 19814.4 |
| 200 | `casadi_map_sx` | 5/5 | 213.451 | 0.81 | 1030885 | 217294 | 6802.8 |
| 500 | `scaly` | 5/5 | 32.178 | 10.73 | 121528 | 0 | 459.1 |
| 500 | `casadi_sx` | 5/5 | 206.559 | 1.77 | 10246853 | 119 | 50453.0 |
| 500 | `casadi_mx` | 0/5 | skipped_after_failure |  |  |  |  |
| 500 | `casadi_call_mx` | 5/5 | 516.982 | 1.02 | 2552822 | 134274 | 119497.8 |
| 500 | `casadi_map_sx` | 5/5 | 546.787 | 0.63 | 2535806 | 542494 | 32146.4 |

## Discrete-time HCBF safety filter (`unbumpercars`)

150 rows: 85 ok, 60 skipped_after_failure, 5 timeout.

| Size | Backend | Successful processes | Mean, µs | CV, % | Executable bytes | Workspace, doubles | Kernel compile mean, ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | `scaly` | 5/5 | 185.986 | 0.74 | 1552718 | 122880 | 1586.8 |
| 2 | `casadi_sx` | 0/5 | timeout; skipped_after_failure |  | 26778898 | 37182 |  |
| 2 | `casadi_mx` | 5/5 | 962.702 | 1.00 | 110035 | 110087 | 2313.4 |
| 2 | `casadi_mx_gemm` | 5/5 | 1079.089 | 0.99 | 124566 | 112315 | 2277.8 |
| 2 | `casadi_mx_gemm_classic` | 5/5 | 1071.109 | 0.93 | 124574 | 112315 | 2251.6 |
| 2 | `casadi_mx_gemm_blasfeo` | 5/5 | 1075.932 | 0.94 | 124574 | 112315 | 2252.7 |
| 4 | `scaly` | 5/5 | 215.733 | 0.78 | 1899224 | 100352 | 3594.4 |
| 4 | `casadi_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 4 | `casadi_mx` | 5/5 | 3040.587 | 0.89 | 361999 | 112213 | 7524.7 |
| 4 | `casadi_mx_gemm` | 5/5 | 3756.188 | 0.56 | 629111 | 120257 | 5094.6 |
| 4 | `casadi_mx_gemm_classic` | 5/5 | 3803.001 | 1.91 | 629119 | 120257 | 5102.2 |
| 4 | `casadi_mx_gemm_blasfeo` | 5/5 | 3746.231 | 0.70 | 629119 | 120257 | 5070.8 |
| 8 | `scaly` | 5/5 | 484.377 | 0.60 | 2286594 | 117600 | 10377.4 |
| 8 | `casadi_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 8 | `casadi_mx` | 5/5 | 10747.810 | 1.03 | 1327412 | 116049 | 52872.1 |
| 8 | `casadi_mx_gemm` | 5/5 | 14052.266 | 1.17 | 4569620 | 136111 | 35142.3 |
| 8 | `casadi_mx_gemm_classic` | 5/5 | 14208.064 | 2.59 | 4569628 | 136111 | 35025.2 |
| 8 | `casadi_mx_gemm_blasfeo` | 5/5 | 14121.109 | 0.57 | 4569628 | 136111 | 35270.1 |
| 16 | `scaly` | 5/5 | 1422.562 | 1.67 | 3087153 | 243600 | 27255.9 |
| 16 | `casadi_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 16 | `casadi_mx` | 0/5 | timeout; skipped_after_failure |  | 5399074 | 125987 |  |
| 16 | `casadi_mx_gemm` | 0/5 | timeout; skipped_after_failure |  | 35514289 | 171611 |  |
| 16 | `casadi_mx_gemm_classic` | 0/5 | timeout; skipped_after_failure |  | 35514297 | 171611 |  |
| 16 | `casadi_mx_gemm_blasfeo` | 0/5 | timeout; skipped_after_failure |  | 35514297 | 171611 |  |
| 32 | `scaly` | 5/5 | 6245.988 | 0.51 | 4667865 | 1168752 | 40939.1 |
| 32 | `casadi_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 32 | `casadi_mx` | 0/5 | skipped_after_failure |  |  |  |  |
| 32 | `casadi_mx_gemm` | 0/5 | skipped_after_failure |  |  |  |  |
| 32 | `casadi_mx_gemm_classic` | 0/5 | skipped_after_failure |  |  |  |  |
| 32 | `casadi_mx_gemm_blasfeo` | 0/5 | skipped_after_failure |  |  |  |  |

## Reproduce the sweep

```bash
# Run every headline sweep and closed loop, then render the report.
SCALY_VECTOR_LIBM=glibc SCALY_CC=gcc CC=gcc CXX=clang++ uv run benchmarks/run.py study --out-dir benchmarks/results/<study-name>

# Run only the frozen sweep grids.
SCALY_VECTOR_LIBM=glibc SCALY_CC=gcc CC=gcc CXX=clang++ uv run benchmarks/run.py study --out-dir benchmarks/results/<study-name> --only sweep

# Render an existing study again.
uv run benchmarks/run.py report benchmarks/results/<study-name>
```

Raw comma-separated value files, generated code, samples, binaries, logs, and provenance live below
the selected study directory. The report command regenerates the tables on this page.
