# October 29 release study audit

This folder records the October 29 investigation of slowdowns since October 20 in unbumpercars, chain M=5 and M=9, and race cars N=400. PR #245 fixed unbumpercars. The chain and race-car slowdowns were left for a later release, with chain M=9 tracked in #214. The notes below preserve the investigation as recorded at the time.

Measured `a3cc544c4e952520c747b7f4c4cef59074303a61` in `benchmarks/results/study-2026-10-29-release-0.1.0`.
Finished 2026-10-29T10:26:47.799958+00:00, duration 75.02 minutes. All 12 commands returned zero. Audited 28 provenance records.

Both providers use Clang 20.1.8 here. October 20 used GCC 13.3.0 for controllers and Clang 20.1.8 for sweep kernels. Cross-study changes combine revision and toolchain changes and do not identify either cause.

Sweep outcomes: {'passed': 77, 'failed': 0, 'timed_out': 11, 'skipped': 18}. October 20: {'passed': 79, 'timed_out': 11, 'skipped': 16}.
All 77 timed cells have a saved compiled-C correctness marker, reference samples, a gate before timing and a successful timing record. The precompile sparsity/reference gate is established by the runner's control flow and retained samples, not a separate symbolic-value log.
Compiled-C gates use absolute and relative tolerances of 1e-9. Unbumpercars uses an independently constructed CasADi reference; the other problems use stage-based NumPy references.

There is one process per cell/provider. Runtime coefficients of variation are absent, not zero. October 12's repeated-process coefficients describe historical variation, not confidence intervals for this study. Large historical variation requires a measurement follow-up before a precise claim.

## Revision provenance

Revision counts across all provenance records: {'a3cc544c4e952520c747b7f4c4cef59074303a61': 9, 'eca417896b026c06b88264678da71c848861bc36': 19}.
The checkout was updated while the study ran. Later provenance differs from the launch manifest. The source, plugin, benchmark and lockfile trees are identical across those revisions. The difference is documentation and a Windows package classifier, so executable-source equivalence is established by a Git tree comparison. The study does not have one literal SHA in all provenance records.

| Revision | Changed files |
|---|---|
| eca417896b026c06b88264678da71c848861bc36 | README.md, docs/guide/codegen.md, docs/guide/env_vars.md, docs/guide/installation.md, docs/guide/solver_backends.md, pyproject.toml |

| Problem | Passed | Failed | Timed out | Skipped |
|---|---|---|---|---|
| race_cars | 32 | 0 | 1 | 2 |
| npmpc | 24 | 0 | 4 | 8 |
| unbumpercars | 11 | 0 | 3 | 6 |
| chain | 10 | 0 | 3 | 2 |

## Outcome changes against October 20

| Problem | Size | Backend | Previous | Current | Reason |
|---|---|---|---|---|---|
| npmpc | 25 | casadi_map_sx | passed | timed_out | compile > 180s (kernel_compile_ms) |
| npmpc | 50 | casadi_map_sx | passed | skipped | casadi_map_sx timeout at size=25; this size and larger won't fit |
| npmpc | 100 | casadi_map_sx | timed_out | skipped | casadi_map_sx timeout at size=25; this size and larger won't fit |

## Headline Hessians

Ratios divide the fastest completed CasADi encoding's time by Scaly's at the same size. Missing ratios mean no tested CasADi encoding completed under the budget.

| Problem | Size | Scaly, us | Best CasADi | CasADi, us | CasADi / Scaly |
|---|---|---|---|---|---|
| race_cars | 5 | 1.013 | casadi_sx | 1.927 | 1.903 |
| race_cars | 10 | 1.185 | casadi_sx | 3.804 | 3.210 |
| race_cars | 25 | 2.408 | casadi_sx | 9.666 | 4.014 |
| race_cars | 50 | 4.051 | casadi_sx | 19.506 | 4.816 |
| race_cars | 100 | 7.829 | casadi_sx | 39.307 | 5.020 |
| race_cars | 200 | 15.224 | casadi_sx | 80.808 | 5.308 |
| race_cars | 400 | 30.291 | casadi_sx | 162.125 | 5.352 |
| npmpc | 6 | 8.857 | casadi_mx | 20.805 | 2.349 |
| npmpc | 12 | 17.831 | casadi_mx | 42.358 | 2.376 |
| npmpc | 25 | 35.534 | casadi_mx | 85.095 | 2.395 |
| npmpc | 50 | 65.603 | casadi_mx | 176.813 | 2.695 |
| npmpc | 100 | 121.915 | casadi_mx_gemm | 886.546 | 7.272 |
| npmpc | 200 | 234.268 | casadi_call_mx | 4719.874 | 20.147 |
| unbumpercars | 2 | 224.802 | casadi_mx | 956.587 | 4.255 |
| unbumpercars | 4 | 284.711 | casadi_mx | 3049.815 | 10.712 |
| unbumpercars | 8 | 492.700 | casadi_mx | 11051.708 | 22.431 |
| unbumpercars | 16 | 1506.051 | None |  |  |
| unbumpercars | 32 | 5267.123 | None |  |  |
| chain | 3 | 21.177 | casadi_sx | 25.953 | 1.226 |
| chain | 5 | 96.304 | casadi_sx | 82.958 | 0.861 |
| chain | 9 | 4305.881 | casadi_map_sx | 2482.524 | 0.577 |

## Every matching timed sweep cell

All ratios are October 29 / October 20. Byte and hash checks read retained artifacts. Historical CV is the sample coefficient of variation across five October 12 processes, where that size exists.

| Problem | Size | Backend | Old, us | New, us | Ratio | Historical CV | Same source | Same object | Same samples |
|---|---|---|---|---|---|---|---|---|---|
| race_cars | 5 | casadi_mx | 2.836 | 2.848 | 1.0042 | 1.19% | True | True | True |
| race_cars | 5 | casadi_sx | 1.926 | 1.927 | 1.0005 | 0.53% | True | True | True |
| race_cars | 5 | scaly | 0.851 | 1.013 | 1.1896 | 1.12% | False | False | True |
| race_cars | 5 | casadi_map_sx | 5.217 | 5.210 | 0.9988 | 0.21% | True | True | True |
| race_cars | 5 | casadi_call_mx | 5.223 | 5.155 | 0.9871 | 1.03% | True | True | True |
| race_cars | 10 | casadi_mx | 5.633 | 5.637 | 1.0007 | 0.94% | True | True | True |
| race_cars | 10 | casadi_sx | 3.807 | 3.804 | 0.9991 | 0.16% | True | True | True |
| race_cars | 10 | scaly | 1.108 | 1.185 | 1.0691 | 1.90% | False | False | True |
| race_cars | 10 | casadi_map_sx | 10.224 | 10.206 | 0.9982 | 0.18% | True | True | True |
| race_cars | 10 | casadi_call_mx | 10.097 | 10.098 | 1.0002 | 1.65% | True | True | True |
| race_cars | 25 | casadi_mx | 14.200 | 14.176 | 0.9983 | 0.61% | True | True | True |
| race_cars | 25 | casadi_sx | 9.664 | 9.666 | 1.0002 | 0.02% | True | True | True |
| race_cars | 25 | scaly | 1.984 | 2.408 | 1.2138 | 1.01% | False | False | True |
| race_cars | 25 | casadi_map_sx | 25.378 | 25.458 | 1.0032 | 0.30% | True | True | True |
| race_cars | 25 | casadi_call_mx | 25.086 | 24.964 | 0.9951 | 0.19% | True | True | True |
| race_cars | 50 | casadi_mx | 28.793 | 28.729 | 0.9978 | 0.50% | True | True | True |
| race_cars | 50 | casadi_sx | 19.454 | 19.506 | 1.0027 | 0.16% | True | True | True |
| race_cars | 50 | scaly | 3.281 | 4.051 | 1.2346 | 2.47% | False | False | True |
| race_cars | 50 | casadi_map_sx | 51.618 | 51.638 | 1.0004 | 0.36% | True | True | True |
| race_cars | 50 | casadi_call_mx | 50.343 | 50.645 | 1.0060 | 0.33% | True | True | True |
| race_cars | 100 | casadi_sx | 39.566 | 39.307 | 0.9934 | 0.77% | True | True | True |
| race_cars | 100 | scaly | 6.380 | 7.829 | 1.2272 | 4.12% | False | False | True |
| race_cars | 100 | casadi_map_sx | 103.925 | 104.487 | 1.0054 | 0.33% | True | True | True |
| race_cars | 100 | casadi_call_mx | 105.252 | 100.087 | 0.9509 | 0.23% | True | True | True |
| race_cars | 200 | casadi_sx | 80.973 | 80.808 | 0.9980 | 1.41% | True | True | True |
| race_cars | 200 | scaly | 11.797 | 15.224 | 1.2905 | 2.11% | False | False | True |
| race_cars | 200 | casadi_map_sx | 212.771 | 212.097 | 0.9968 | 0.28% | True | True | True |
| race_cars | 200 | casadi_call_mx | 202.070 | 201.335 | 0.9964 | 0.18% | True | True | True |
| race_cars | 400 | casadi_sx | 162.621 | 162.125 | 0.9970 |  | True | True | True |
| race_cars | 400 | scaly | 22.951 | 30.291 | 1.3198 |  | False | False | True |
| race_cars | 400 | casadi_map_sx | 430.963 | 429.768 | 0.9972 |  | True | True | True |
| race_cars | 400 | casadi_call_mx | 414.047 | 414.554 | 1.0012 |  | True | True | True |
| npmpc | 6 | casadi_map_sx | 196.321 | 197.995 | 1.0085 | 2.48% | True | True | False |
| npmpc | 6 | casadi_mx | 20.574 | 20.805 | 1.0112 | 0.65% | True | True | False |
| npmpc | 6 | casadi_sx | 71.850 | 72.466 | 1.0086 | 2.06% | True | True | False |
| npmpc | 6 | scaly | 10.532 | 8.857 | 0.8410 | 1.39% | False | False | False |
| npmpc | 6 | casadi_mx_gemm | 43.821 | 44.453 | 1.0144 | 0.70% | True | True | False |
| npmpc | 6 | casadi_call_mx | 140.622 | 136.951 | 0.9739 | 0.95% | True | True | False |
| npmpc | 12 | casadi_map_sx | 468.723 | 459.792 | 0.9809 | 0.55% | True | True | False |
| npmpc | 12 | casadi_mx | 42.107 | 42.358 | 1.0060 | 0.95% | True | True | False |
| npmpc | 12 | scaly | 23.340 | 17.831 | 0.7639 | 0.46% | False | False | False |
| npmpc | 12 | casadi_mx_gemm | 101.358 | 101.523 | 1.0016 | 1.71% | True | True | False |
| npmpc | 12 | casadi_call_mx | 292.431 | 291.675 | 0.9974 | 0.45% | True | True | False |
| npmpc | 25 | casadi_mx | 84.819 | 85.095 | 1.0033 | 0.58% | True | True | False |
| npmpc | 25 | scaly | 48.643 | 35.534 | 0.7305 | 5.49% | False | False | False |
| npmpc | 25 | casadi_mx_gemm | 209.407 | 209.841 | 1.0021 | 0.47% | True | True | False |
| npmpc | 25 | casadi_call_mx | 583.093 | 572.824 | 0.9824 | 0.45% | True | True | False |
| npmpc | 50 | casadi_mx | 176.040 | 176.813 | 1.0044 | 0.74% | True | True | False |
| npmpc | 50 | scaly | 77.684 | 65.603 | 0.8445 | 2.97% | False | False | False |
| npmpc | 50 | casadi_mx_gemm | 431.216 | 454.572 | 1.0542 | 1.82% | True | True | False |
| npmpc | 50 | casadi_call_mx | 1159.206 | 1198.539 | 1.0339 | 0.78% | True | True | False |
| npmpc | 100 | scaly | 162.526 | 121.915 | 0.7501 | 0.35% | False | False | False |
| npmpc | 100 | casadi_mx_gemm | 899.670 | 886.546 | 0.9854 | 1.13% | True | True | False |
| npmpc | 100 | casadi_call_mx | 2322.052 | 2298.552 | 0.9899 | 0.42% | True | True | False |
| npmpc | 200 | scaly | 253.473 | 234.268 | 0.9242 | 0.88% | False | False | False |
| npmpc | 200 | casadi_call_mx | 4601.706 | 4719.874 | 1.0257 | 0.48% | True | True | False |
| unbumpercars | 2 | casadi_mx | 953.367 | 956.587 | 1.0034 | 1.31% | True | True | True |
| unbumpercars | 2 | scaly | 227.132 | 224.802 | 0.9897 | 0.22% | False | False | True |
| unbumpercars | 2 | casadi_mx_gemm | 1075.266 | 1087.143 | 1.0110 | 1.53% | True | True | True |
| unbumpercars | 4 | casadi_mx | 3069.227 | 3049.815 | 0.9937 | 0.67% | True | True | True |
| unbumpercars | 4 | scaly | 280.510 | 284.711 | 1.0150 | 0.12% | False | False | True |
| unbumpercars | 4 | casadi_mx_gemm | 3780.450 | 3749.696 | 0.9919 | 0.89% | True | True | True |
| unbumpercars | 8 | casadi_mx | 10860.516 | 11051.708 | 1.0176 | 3.07% | True | True | False |
| unbumpercars | 8 | scaly | 575.297 | 492.700 | 0.8564 | 0.46% | False | False | False |
| unbumpercars | 8 | casadi_mx_gemm | 14178.423 | 14307.648 | 1.0091 | 0.92% | True | True | False |
| unbumpercars | 16 | scaly | 1462.377 | 1506.051 | 1.0299 | 0.28% | False | False | True |
| unbumpercars | 32 | scaly | 5533.094 | 5267.123 | 0.9519 | 2.25% | False | False | True |
| chain | 3 | casadi_sx | 25.966 | 25.953 | 0.9995 | 0.25% | True | True | True |
| chain | 3 | scaly | 22.025 | 21.177 | 0.9615 | 0.24% | False | False | True |
| chain | 3 | casadi_map_sx | 121.500 | 122.963 | 1.0120 | 0.26% | True | True | True |
| chain | 3 | casadi_call_mx | 178.265 | 180.284 | 1.0113 | 0.35% | True | True | True |
| chain | 5 | casadi_sx | 82.285 | 82.958 | 1.0082 | 0.34% | True | True | True |
| chain | 5 | scaly | 100.177 | 96.304 | 0.9613 | 0.04% | False | False | True |
| chain | 5 | casadi_map_sx | 631.238 | 634.353 | 1.0049 | 1.03% | True | True | True |
| chain | 5 | casadi_call_mx | 811.145 | 815.364 | 1.0052 | 1.03% | True | True | True |
| chain | 9 | scaly | 282.288 | 4305.881 | 15.2535 | 0.08% | False | False | True |
| chain | 9 | casadi_map_sx | 2466.724 | 2482.524 | 1.0064 | 1.68% | True | True | True |

## Changed numerical samples

Byte differences alone need not change the workload. This table measures the numeric differences in every changed Scaly sample and reference file.

| Problem | Size | File | Changed values | Max absolute difference | All close at 1e-9 |
|---|---|---|---|---|---|
| npmpc | 6 | sample_p.bin | 16 | 1.18996e-09 | True |
| npmpc | 6 | expected_dense.bin | 16 | 1.60355e-09 | True |
| npmpc | 12 | sample_p.bin | 16 | 1.18996e-09 | True |
| npmpc | 12 | expected_dense.bin | 16 | 1.52568e-09 | True |
| npmpc | 25 | sample_p.bin | 16 | 1.18996e-09 | True |
| npmpc | 25 | expected_dense.bin | 16 | 1.99623e-09 | True |
| npmpc | 50 | sample_p.bin | 16 | 1.18996e-09 | True |
| npmpc | 50 | expected_dense.bin | 16 | 2.36093e-09 | True |
| npmpc | 100 | sample_p.bin | 16 | 1.18996e-09 | True |
| npmpc | 100 | expected_dense.bin | 16 | 9.42123e-10 | True |
| npmpc | 200 | sample_p.bin | 16 | 1.18996e-09 | True |
| npmpc | 200 | expected_dense.bin | 16 | 1.44655e-09 | True |
| unbumpercars | 8 | sample_u_s.bin | 76 | 0.998743 | False |
| unbumpercars | 8 | sample_u_des.bin | 8 | 1 | False |
| unbumpercars | 8 | sample_bar_x.bin | 56 | 4.81057 | False |
| unbumpercars | 8 | sample_lam_g.bin | 60 | 4.07164 | False |
| unbumpercars | 8 | expected_dense.bin | 256 | 2.90467 | False |

All current providers' normalized numerical inputs agree at absolute and relative tolerances of 1e-9 at each size. The changed unbumpercars C=8 cross-study samples prevent treating its sweep timing change as a fixed-input performance change.

## Historical retained-binary dispersion

These are October 15's existing three-process diagnostics, not new measurements. The wide NPMPC MX N=25 distribution is a measurement problem and does not establish an optimization or regression.

| Study | Cell | CV | Minimum, us | Maximum, us |
|---|---|---|---|---|
| study-2026-10-14 | casadi_mx_N25 | 9.41% | 84.658 | 99.317 |
| study-2026-10-14 | scaly_N200 | 0.87% | 254.108 | 258.562 |
| study-2026-10-15 | casadi_mx_N25 | 9.29% | 84.742 | 99.339 |
| study-2026-10-15 | scaly_N200 | 1.38% | 255.934 | 262.636 |

## Controller changes

Mean native time per step and function evaluation. These means are over steps, not process repetitions. The CSV also retains build times and oracle totals.

| Problem | Solver | Provider | Old total, ms | New total, ms | Ratio | Old FE, ms | New FE, ms | Ratio |
|---|---|---|---|---|---|---|---|---|
| chain | ipopt | scaly | 6.8479 | 6.9950 | 1.0215 | 0.3964 | 0.5182 | 1.3072 |
| chain | sqp | casadi | 6.3506 | 6.3084 | 0.9934 | 0.1604 | 0.1286 | 0.8016 |
| chain | sqp | scaly | 6.3453 | 6.4147 | 1.0109 | 0.1803 | 0.2220 | 1.2311 |
| npmpc | ipopt | casadi | 4.4627 | 4.1004 | 0.9188 | 1.7337 | 1.4173 | 0.8175 |
| npmpc | ipopt | scaly | 3.4988 | 3.3766 | 0.9651 | 0.9505 | 0.8008 | 0.8426 |
| npmpc | sqp | casadi | 1.8458 | 1.4893 | 0.8068 | 1.0270 | 0.6777 | 0.6599 |
| npmpc | sqp | scaly | 1.2631 | 1.1920 | 0.9437 | 0.4663 | 0.3949 | 0.8469 |
| race_cars | ipopt | casadi | 5.8281 | 5.9343 | 1.0182 | 0.4954 | 0.5051 | 1.0196 |
| race_cars | ipopt | scaly | 5.3824 | 5.4201 | 1.0070 | 0.1847 | 0.1778 | 0.9629 |
| race_cars | sqp | casadi | 3.3249 | 3.2537 | 0.9786 | 0.1560 | 0.1424 | 0.9128 |
| race_cars | sqp | scaly | 3.1936 | 3.1919 | 0.9995 | 0.0556 | 0.0477 | 0.8578 |
| unbumpercars | ipopt | casadi | 146.6987 | 170.4836 | 1.1621 | 142.1830 | 166.0358 | 1.1678 |
| unbumpercars | ipopt | scaly | 20.1200 | 30.3648 | 1.5092 | 16.2040 | 26.4409 | 1.6317 |
| unbumpercars | sqp | casadi | 87.5897 | 99.2273 | 1.1329 | 86.2305 | 97.9132 | 1.1355 |
| unbumpercars | sqp | scaly | 10.9436 | 17.2972 | 1.5806 | 9.6556 | 16.0368 | 1.6609 |

| Problem | Solver | Provider | Historical total CV | Historical FE CV | Iteration mismatches across dates | Oracle-counter mismatches across dates |
|---|---|---|---|---|---|---|
| chain | ipopt | scaly | 0.10% | 1.32% | 0 | 0 |
| chain | sqp | casadi | 0.51% | 0.14% | 0 | 0 |
| chain | sqp | scaly | 0.26% | 0.30% | 0 | 0 |
| npmpc | ipopt | casadi | 0.39% | 0.42% | 0 | 0 |
| npmpc | ipopt | scaly | 0.78% | 0.21% | 0 | 0 |
| npmpc | sqp | casadi | 0.27% | 0.15% | 0 | 0 |
| npmpc | sqp | scaly | 0.31% | 0.50% | 0 | 0 |
| race_cars | ipopt | casadi | 0.32% | 0.70% | 0 | 0 |
| race_cars | ipopt | scaly | 0.75% | 1.65% | 0 | 0 |
| race_cars | sqp | casadi | 0.64% | 1.20% | 0 | 0 |
| race_cars | sqp | scaly | 0.47% | 1.77% | 0 | 0 |
| unbumpercars | ipopt | casadi | 0.17% | 0.17% | 0 | 0 |
| unbumpercars | ipopt | scaly | 1.47% | 1.73% | 0 | 0 |
| unbumpercars | sqp | casadi | 1.30% | 1.31% | 2 | 12 |
| unbumpercars | sqp | scaly | 1.15% | 1.27% | 3 | 17 |

## Paired per-step CasADi / Scaly ratios

Medians and 10th/90th percentiles describe episode spread, not measurement uncertainty. Unbumpercars SQP differs in iterations on one step; its ratio is an observed episode result. Unbumpercars oracle counts are read from stats.csv because telemetry.csv omits them.

| Problem | Solver | Metric | October 20 median | October 29 median | P10 | P90 |
|---|---|---|---|---|---|---|
| chain | sqp | solver_time_ms | 1.0009 | 0.9830 | 0.9807 | 0.9855 |
| chain | sqp | fe_time_ms | 0.8809 | 0.5715 | 0.5576 | 0.5952 |
| npmpc | ipopt | solver_time_ms | 1.2659 | 1.2020 | 1.1937 | 1.2135 |
| npmpc | ipopt | fe_time_ms | 1.8134 | 1.7545 | 1.7237 | 1.7788 |
| npmpc | sqp | solver_time_ms | 1.4578 | 1.2444 | 1.2322 | 1.2626 |
| npmpc | sqp | fe_time_ms | 2.2063 | 1.6888 | 1.6566 | 1.7325 |
| race_cars | ipopt | solver_time_ms | 1.0831 | 1.0939 | 1.0830 | 1.1043 |
| race_cars | ipopt | fe_time_ms | 2.6888 | 2.8441 | 2.7295 | 2.9194 |
| race_cars | sqp | solver_time_ms | 1.0470 | 1.0255 | 0.9986 | 1.0505 |
| race_cars | sqp | fe_time_ms | 2.8126 | 3.0048 | 2.7993 | 3.1219 |
| unbumpercars | ipopt | solver_time_ms | 7.2849 | 5.6272 | 5.4968 | 5.7108 |
| unbumpercars | ipopt | fe_time_ms | 8.7813 | 6.3004 | 6.1541 | 6.3748 |
| unbumpercars | sqp | solver_time_ms | 7.9705 | 5.6267 | 5.2824 | 6.1060 |
| unbumpercars | sqp | fe_time_ms | 8.9310 | 5.9571 | 5.6120 | 6.5432 |

| Problem | Solver | Iteration mismatches | Oracle mismatches by counter | Affected steps | October 20 affected steps | Maximum state difference | Maximum control difference |
|---|---|---|---|---|---|---|---|
| chain | sqp | 0 | {'n_eval_f': 0, 'n_eval_g': 0, 'n_eval_grad_f': 0, 'n_eval_jac_g': 0, 'n_eval_h': 0} | [] | [] | 4.46423e-09 | 2.59639e-08 |
| npmpc | ipopt | 0 | {'n_eval_f': 0, 'n_eval_g': 0, 'n_eval_grad_f': 0, 'n_eval_jac_g': 0, 'n_eval_h': 0} | [] | [] | 5.9508e-13 | 3.23006e-15 |
| npmpc | sqp | 0 | {'n_eval_f': 0, 'n_eval_g': 0, 'n_eval_grad_f': 0, 'n_eval_jac_g': 0, 'n_eval_h': 0} | [] | [] | 6.48248e-10 | 3.21273e-12 |
| race_cars | ipopt | 0 | {'n_eval_f': 0, 'n_eval_g': 0, 'n_eval_grad_f': 0, 'n_eval_jac_g': 0, 'n_eval_h': 0} | [] | [] | 8.52651e-14 | 5.14433e-12 |
| race_cars | sqp | 0 | {'n_eval_f': 0, 'n_eval_g': 0, 'n_eval_grad_f': 0, 'n_eval_jac_g': 0, 'n_eval_h': 0} | [] | [] | 2.26574e-12 | 1.85784e-09 |
| unbumpercars | ipopt | 0 | {'n_eval_f': 0, 'n_eval_g': 0, 'n_eval_grad_f': 0, 'n_eval_jac_g': 0, 'n_eval_h': 0} | [] | [] | 4.61298e-12 | 3.19548e-12 |
| unbumpercars | sqp | 1 | {'n_eval_f': 3, 'n_eval_g': 3, 'n_eval_grad_f': 1, 'n_eval_jac_g': 1, 'n_eval_h': 1} | [30, 148, 198] | [30, 198] | 3.70111e-08 | 2.82225e-08 |

## Solver outcomes and deadlines

All 6,938 steps in 15 episodes succeeded. Deadline misses count solver time greater than the configured sampling period, not solver failures.

| Problem | Solver | Provider | Steps | Failed | Maximum, ms | Sampling, ms | Over sampling |
|---|---|---|---|---|---|---|---|
| chain | ipopt | scaly | 90 | 0 | 11.1065 | 200 | 0 |
| chain | sqp | casadi | 90 | 0 | 8.1173 | 200 | 0 |
| chain | sqp | scaly | 90 | 0 | 8.2223 | 200 | 0 |
| npmpc | ipopt | casadi | 100 | 0 | 10.4135 | 20 | 0 |
| npmpc | ipopt | scaly | 100 | 0 | 6.2739 | 20 | 0 |
| npmpc | sqp | casadi | 100 | 0 | 9.0659 | 20 | 0 |
| npmpc | sqp | scaly | 100 | 0 | 7.1880 | 20 | 0 |
| race_cars | ipopt | casadi | 1367 | 0 | 16.9036 | 50 | 0 |
| race_cars | ipopt | scaly | 1367 | 0 | 11.9482 | 50 | 0 |
| race_cars | sqp | casadi | 1367 | 0 | 9.7465 | 50 | 0 |
| race_cars | sqp | scaly | 1367 | 0 | 9.4058 | 50 | 0 |
| unbumpercars | ipopt | casadi | 200 | 0 | 347.8851 | 100 | 199 |
| unbumpercars | ipopt | scaly | 200 | 0 | 61.6659 | 100 | 0 |
| unbumpercars | sqp | casadi | 200 | 0 | 742.9807 | 100 | 55 |
| unbumpercars | sqp | scaly | 200 | 0 | 118.9148 | 100 | 2 |

Scaly deadline misses, unbumpercars sqp: [{'step': 92, 'ms': 118.9148249104619, 'previous_ms': 82.60104898363352}, {'step': 169, 'ms': 114.54449780285358, 'previous_ms': 67.33751762658358}].
