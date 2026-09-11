# Alloy scalability sweep

These are all exact sparse Lagrangian Hessian cells from the study that started on 2026-09-10 and
finished on 2026-09-11. The [results overview](index.md) summarizes the canonical points, and the
[fairness audit](fairness.md) defines the comparison.

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

- `alloy`: Alloy's sparse generated C.
- `casadi_sx`: an unrolled CasADi SX graph.
- `casadi_mx`: a CasADi MX graph.
- `casadi_call_mx`: MX with first-class calls.
- `casadi_map_sx`: serial `Function.map` over SX.
- `casadi_mx_gemm*`: MX matrix products with the default, classic, or BLASFEO lowering.

## Hanging chain of masses (`chain`)

75 rows: 50 ok, 22 skipped_after_failure, 3 timeout.

| Size | Backend | Successful processes | Mean, µs | CV, % | Executable bytes | Workspace, doubles | Kernel compile mean, ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3 | `alloy` | 5/5 | 38.271 | 0.98 | 102918 | 26028 | 669.7 |
| 3 | `casadi_sx` | 5/5 | 25.979 | 0.16 | 3330455 | 188 | 19034.8 |
| 3 | `casadi_mx` | 0/5 | timeout; skipped_after_failure |  | 9410547 | 23678 |  |
| 3 | `casadi_call_mx` | 5/5 | 181.407 | 0.50 | 602188 | 27321 | 13029.2 |
| 3 | `casadi_map_sx` | 5/5 | 123.653 | 1.34 | 556793 | 94736 | 4635.8 |
| 5 | `alloy` | 5/5 | 202.900 | 0.90 | 393431 | 109944 | 2548.0 |
| 5 | `casadi_sx` | 5/5 | 82.632 | 0.69 | 10330465 | 335 | 92304.0 |
| 5 | `casadi_mx` | 0/5 | skipped_after_failure |  |  |  |  |
| 5 | `casadi_call_mx` | 5/5 | 825.058 | 0.18 | 2097970 | 100698 | 78595.1 |
| 5 | `casadi_map_sx` | 5/5 | 627.198 | 0.92 | 2561051 | 349353 | 10318.1 |
| 9 | `alloy` | 5/5 | 550.460 | 0.93 | 1044312 | 398916 | 9989.2 |
| 9 | `casadi_sx` | 0/5 | timeout; skipped_after_failure |  | 24350782 | 897 |  |
| 9 | `casadi_mx` | 0/5 | skipped_after_failure |  |  |  |  |
| 9 | `casadi_call_mx` | 0/5 | timeout; skipped_after_failure |  | 6664454 | 280960 |  |
| 9 | `casadi_map_sx` | 5/5 | 2455.255 | 1.63 | 9650809 | 999844 | 59340.9 |

## Neural-process model predictive control (`npmpc`)

240 rows: 169 ok, 64 skipped_after_failure, 7 timeout.

| Size | Backend | Successful processes | Mean, µs | CV, % | Executable bytes | Workspace, doubles | Kernel compile mean, ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 6 | `alloy` | 5/5 | 18.238 | 1.08 | 31757 | 0 | 611.2 |
| 6 | `casadi_sx` | 5/5 | 72.871 | 0.90 | 7564935 | 15913 | 102529.2 |
| 6 | `casadi_mx` | 5/5 | 20.859 | 1.58 | 151448 | 7570 | 5172.1 |
| 6 | `casadi_call_mx` | 5/5 | 149.281 | 1.06 | 232343 | 46861 | 6800.1 |
| 6 | `casadi_map_sx` | 5/5 | 198.880 | 1.33 | 4261664 | 341213 | 135464.6 |
| 6 | `casadi_mx_gemm` | 5/5 | 45.631 | 0.77 | 109496 | 8660 | 2498.1 |
| 6 | `casadi_mx_gemm_classic` | 5/5 | 45.906 | 2.09 | 109504 | 8660 | 2417.5 |
| 6 | `casadi_mx_gemm_blasfeo` | 5/5 | 45.739 | 1.36 | 109504 | 8660 | 2417.3 |
| 12 | `alloy` | 5/5 | 34.737 | 0.27 | 31783 | 0 | 609.4 |
| 12 | `casadi_sx` | 0/5 | timeout; skipped_after_failure |  | 15063206 | 30463 |  |
| 12 | `casadi_mx` | 5/5 | 44.184 | 2.22 | 276488 | 9813 | 12861.8 |
| 12 | `casadi_call_mx` | 5/5 | 298.150 | 0.19 | 371542 | 68072 | 10368.6 |
| 12 | `casadi_map_sx` | 2/5 | 462.489 | 1.06 | 5788748 | 774841 | 176811.9 |
| 12 | `casadi_mx_gemm` | 5/5 | 104.590 | 2.00 | 189660 | 13031 | 5253.9 |
| 12 | `casadi_mx_gemm_classic` | 5/5 | 105.657 | 1.80 | 189668 | 13031 | 5246.1 |
| 12 | `casadi_mx_gemm_blasfeo` | 5/5 | 103.946 | 0.88 | 189668 | 13031 | 5225.0 |
| 25 | `alloy` | 5/5 | 72.575 | 0.88 | 31755 | 2700 | 613.7 |
| 25 | `casadi_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 25 | `casadi_mx` | 5/5 | 94.092 | 0.54 | 545609 | 15595 | 41382.4 |
| 25 | `casadi_call_mx` | 5/5 | 624.902 | 0.81 | 424096 | 89139 | 11540.5 |
| 25 | `casadi_map_sx` | 2/5 | 1011.685 | 4.14 | 7858327 | 1603408 | 178523.9 |
| 25 | `casadi_mx_gemm` | 5/5 | 219.006 | 2.07 | 343012 | 22648 | 8327.9 |
| 25 | `casadi_mx_gemm_classic` | 5/5 | 217.633 | 0.99 | 343020 | 22648 | 8321.9 |
| 25 | `casadi_mx_gemm_blasfeo` | 5/5 | 218.387 | 0.84 | 343020 | 22648 | 8295.7 |
| 50 | `alloy` | 5/5 | 148.625 | 0.87 | 31867 | 9870 | 644.3 |
| 50 | `casadi_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 50 | `casadi_mx` | 5/5 | 186.418 | 1.55 | 1063583 | 25782 | 166608.9 |
| 50 | `casadi_call_mx` | 5/5 | 1260.787 | 0.51 | 520231 | 130185 | 13658.2 |
| 50 | `casadi_map_sx` | 0/5 | timeout; skipped_after_failure |  | 11844334 | 3196692 |  |
| 50 | `casadi_mx_gemm` | 5/5 | 444.467 | 0.99 | 638917 | 41169 | 26122.1 |
| 50 | `casadi_mx_gemm_classic` | 5/5 | 447.959 | 1.96 | 638925 | 41169 | 25910.4 |
| 50 | `casadi_mx_gemm_blasfeo` | 5/5 | 446.412 | 0.90 | 638925 | 41169 | 25768.6 |
| 100 | `alloy` | 5/5 | 297.443 | 0.94 | 31890 | 19670 | 663.7 |
| 100 | `casadi_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 100 | `casadi_mx` | 0/5 | timeout; skipped_after_failure |  | 2191474 | 46937 |  |
| 100 | `casadi_call_mx` | 5/5 | 2524.257 | 1.24 | 712196 | 210580 | 21310.3 |
| 100 | `casadi_map_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 100 | `casadi_mx_gemm` | 5/5 | 904.245 | 1.21 | 1232183 | 77961 | 91547.2 |
| 100 | `casadi_mx_gemm_classic` | 5/5 | 909.814 | 2.08 | 1232191 | 77961 | 92973.0 |
| 100 | `casadi_mx_gemm_blasfeo` | 5/5 | 904.997 | 0.81 | 1232191 | 77961 | 91622.0 |
| 200 | `alloy` | 5/5 | 595.213 | 1.17 | 31931 | 44670 | 723.1 |
| 200 | `casadi_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 200 | `casadi_mx` | 0/5 | skipped_after_failure |  |  |  |  |
| 200 | `casadi_call_mx` | 5/5 | 5037.273 | 1.59 | 1127544 | 371260 | 51396.7 |
| 200 | `casadi_map_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 200 | `casadi_mx_gemm` | 0/5 | timeout; skipped_after_failure |  | 2524675 | 151836 |  |
| 200 | `casadi_mx_gemm_classic` | 0/5 | timeout; skipped_after_failure |  | 2524683 | 151836 |  |
| 200 | `casadi_mx_gemm_blasfeo` | 0/5 | timeout; skipped_after_failure |  | 2524683 | 151836 |  |

## Race-car model predictive control (`race_cars`)

225 rows: 210 ok, 14 skipped_after_failure, 1 timeout.

| Size | Backend | Successful processes | Mean, µs | CV, % | Executable bytes | Workspace, doubles | Kernel compile mean, ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `alloy` | 5/5 | 0.567 | 0.70 | 28095 | 0 | 239.5 |
| 1 | `casadi_sx` | 5/5 | 0.412 | 0.29 | 25755 | 108 | 175.8 |
| 1 | `casadi_mx` | 5/5 | 0.510 | 0.78 | 95400 | 304 | 393.3 |
| 1 | `casadi_call_mx` | 5/5 | 0.927 | 0.88 | 228004 | 1067 | 783.5 |
| 1 | `casadi_map_sx` | 5/5 | 0.765 | 0.07 | 94438 | 826 | 398.1 |
| 5 | `alloy` | 5/5 | 2.625 | 0.31 | 23926 | 0 | 233.2 |
| 5 | `casadi_sx` | 5/5 | 1.987 | 0.32 | 107717 | 119 | 333.1 |
| 5 | `casadi_mx` | 5/5 | 2.728 | 1.81 | 441760 | 1085 | 1739.0 |
| 5 | `casadi_call_mx` | 5/5 | 4.935 | 0.97 | 248703 | 2244 | 1827.9 |
| 5 | `casadi_map_sx` | 5/5 | 5.332 | 0.98 | 120288 | 5914 | 1994.1 |
| 10 | `alloy` | 5/5 | 4.871 | 0.98 | 23946 | 0 | 238.8 |
| 10 | `casadi_sx` | 5/5 | 3.954 | 0.79 | 209958 | 119 | 557.5 |
| 10 | `casadi_mx` | 5/5 | 5.621 | 1.48 | 879492 | 2065 | 4251.2 |
| 10 | `casadi_call_mx` | 5/5 | 9.855 | 1.94 | 269955 | 3494 | 2526.8 |
| 10 | `casadi_map_sx` | 5/5 | 10.391 | 0.94 | 142699 | 11334 | 1956.9 |
| 25 | `alloy` | 5/5 | 12.338 | 0.60 | 24007 | 2400 | 297.9 |
| 25 | `casadi_sx` | 5/5 | 9.873 | 0.41 | 516870 | 119 | 1320.5 |
| 25 | `casadi_mx` | 5/5 | 14.015 | 1.02 | 2360004 | 5020 | 23498.3 |
| 25 | `casadi_call_mx` | 5/5 | 24.273 | 1.25 | 334019 | 7484 | 2022.3 |
| 25 | `casadi_map_sx` | 5/5 | 25.978 | 1.76 | 206150 | 27594 | 1171.7 |
| 40 | `alloy` | 5/5 | 19.601 | 0.94 | 24007 | 3840 | 267.7 |
| 40 | `casadi_sx` | 5/5 | 15.898 | 0.59 | 823845 | 119 | 2038.8 |
| 40 | `casadi_mx` | 5/5 | 22.663 | 0.41 | 3799659 | 7975 | 60500.8 |
| 40 | `casadi_call_mx` | 5/5 | 38.961 | 1.06 | 398502 | 11414 | 2551.0 |
| 40 | `casadi_map_sx` | 5/5 | 41.666 | 1.10 | 273406 | 43854 | 1288.2 |
| 50 | `alloy` | 5/5 | 23.499 | 0.18 | 24034 | 8472 | 270.2 |
| 50 | `casadi_sx` | 5/5 | 19.900 | 0.55 | 1028495 | 119 | 2597.8 |
| 50 | `casadi_mx` | 5/5 | 28.175 | 0.33 | 4762326 | 9945 | 97738.9 |
| 50 | `casadi_call_mx` | 5/5 | 48.883 | 1.49 | 441709 | 14034 | 7234.6 |
| 50 | `casadi_map_sx` | 5/5 | 51.813 | 0.15 | 318176 | 54694 | 5501.4 |
| 100 | `alloy` | 5/5 | 47.051 | 0.73 | 24057 | 16872 | 357.5 |
| 100 | `casadi_sx` | 5/5 | 40.369 | 0.37 | 2052051 | 119 | 5861.1 |
| 100 | `casadi_mx` | 0/5 | timeout; skipped_after_failure |  | 9614050 | 19795 |  |
| 100 | `casadi_call_mx` | 5/5 | 96.760 | 0.38 | 679651 | 27134 | 7547.3 |
| 100 | `casadi_map_sx` | 5/5 | 104.417 | 0.26 | 543857 | 108894 | 3371.9 |
| 200 | `alloy` | 5/5 | 89.722 | 0.46 | 24090 | 36084 | 300.4 |
| 200 | `casadi_sx` | 5/5 | 82.833 | 0.44 | 4100194 | 119 | 14113.7 |
| 200 | `casadi_mx` | 0/5 | skipped_after_failure |  |  |  |  |
| 200 | `casadi_call_mx` | 5/5 | 194.878 | 0.10 | 1143240 | 53874 | 19764.3 |
| 200 | `casadi_map_sx` | 5/5 | 214.172 | 0.95 | 1030885 | 217294 | 6712.7 |
| 500 | `alloy` | 5/5 | 228.589 | 0.97 | 24100 | 90084 | 337.0 |
| 500 | `casadi_sx` | 5/5 | 209.352 | 0.78 | 10246853 | 119 | 48849.8 |
| 500 | `casadi_mx` | 0/5 | skipped_after_failure |  |  |  |  |
| 500 | `casadi_call_mx` | 5/5 | 517.671 | 6.33 | 2552822 | 134274 | 119352.0 |
| 500 | `casadi_map_sx` | 5/5 | 550.675 | 1.54 | 2535806 | 542494 | 32139.8 |

## Discrete-time HCBF safety filter (`unbumpercars`)

150 rows: 85 ok, 60 skipped_after_failure, 5 timeout.

| Size | Backend | Successful processes | Mean, µs | CV, % | Executable bytes | Workspace, doubles | Kernel compile mean, ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | `alloy` | 5/5 | 207.925 | 1.02 | 100661 | 0 | 856.1 |
| 2 | `casadi_sx` | 0/5 | timeout; skipped_after_failure |  | 26778898 | 37182 |  |
| 2 | `casadi_mx` | 5/5 | 954.281 | 1.36 | 110035 | 110087 | 2283.5 |
| 2 | `casadi_mx_gemm` | 5/5 | 1076.646 | 0.62 | 124566 | 112315 | 2258.5 |
| 2 | `casadi_mx_gemm_classic` | 5/5 | 1076.588 | 1.33 | 124574 | 112315 | 2235.3 |
| 2 | `casadi_mx_gemm_blasfeo` | 5/5 | 1068.185 | 1.20 | 124574 | 112315 | 2254.4 |
| 4 | `alloy` | 5/5 | 414.715 | 0.18 | 135779 | 0 | 1165.2 |
| 4 | `casadi_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 4 | `casadi_mx` | 5/5 | 3061.284 | 2.99 | 361999 | 112213 | 7356.8 |
| 4 | `casadi_mx_gemm` | 5/5 | 3776.939 | 0.93 | 629111 | 120257 | 5073.3 |
| 4 | `casadi_mx_gemm_classic` | 5/5 | 3763.502 | 1.00 | 629119 | 120257 | 5020.2 |
| 4 | `casadi_mx_gemm_blasfeo` | 5/5 | 3758.067 | 0.85 | 629119 | 120257 | 5066.5 |
| 8 | `alloy` | 5/5 | 875.151 | 1.03 | 200043 | 20384 | 4526.1 |
| 8 | `casadi_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 8 | `casadi_mx` | 5/5 | 10959.742 | 3.29 | 1327412 | 116049 | 52601.6 |
| 8 | `casadi_mx_gemm` | 5/5 | 14204.947 | 0.38 | 4569620 | 136111 | 34535.3 |
| 8 | `casadi_mx_gemm_classic` | 5/5 | 14105.729 | 0.80 | 4569628 | 136111 | 34410.1 |
| 8 | `casadi_mx_gemm_blasfeo` | 5/5 | 14040.066 | 0.74 | 4569628 | 136111 | 34536.9 |
| 16 | `alloy` | 5/5 | 2089.176 | 1.00 | 334411 | 176848 | 19481.9 |
| 16 | `casadi_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 16 | `casadi_mx` | 0/5 | timeout; skipped_after_failure |  | 5399074 | 125987 |  |
| 16 | `casadi_mx_gemm` | 0/5 | timeout; skipped_after_failure |  | 35514289 | 171611 |  |
| 16 | `casadi_mx_gemm_classic` | 0/5 | timeout; skipped_after_failure |  | 35514297 | 171611 |  |
| 16 | `casadi_mx_gemm_blasfeo` | 0/5 | timeout; skipped_after_failure |  | 35514297 | 171611 |  |
| 32 | `alloy` | 5/5 | 6872.175 | 0.22 | 616253 | 1285536 | 18462.4 |
| 32 | `casadi_sx` | 0/5 | skipped_after_failure |  |  |  |  |
| 32 | `casadi_mx` | 0/5 | skipped_after_failure |  |  |  |  |
| 32 | `casadi_mx_gemm` | 0/5 | skipped_after_failure |  |  |  |  |
| 32 | `casadi_mx_gemm_classic` | 0/5 | skipped_after_failure |  |  |  |  |
| 32 | `casadi_mx_gemm_blasfeo` | 0/5 | skipped_after_failure |  |  |  |  |

## Reproduce the sweep

```bash
# Run every headline sweep and closed loop, then render the report.
uv run benchmarks/run.py study --out-dir benchmarks/results/<study-name>

# Run only the frozen sweep grids.
uv run benchmarks/run.py study --out-dir benchmarks/results/<study-name> --only sweep

# Render an existing study again.
uv run benchmarks/run.py report benchmarks/results/<study-name>
```

Raw comma-separated value files, generated code, samples, binaries, logs, and provenance live below
the selected study directory. The report command regenerates the tables on this page.
