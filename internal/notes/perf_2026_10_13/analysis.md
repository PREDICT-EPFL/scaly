# October 13 study audit and comparison

Revision: `38bd589642896503807d2ef49fc4cea7205a9638`. Duration: 103.0 minutes.

138 sweep attempts: 103 timed and correctness-gated, 15 compile timeouts, 20 skipped after failure. No other failures. All 23 Scaly cells timed. All 15 episodes and 6,938 solver steps succeeded.

The October 12 study had 690 attempts: 515 timed, 17 timeouts and 158 skipped after failure. Five processes versus one means the raw counts are not directly comparable. The partially completed mapped-SX cells near the compile limit account for the changed coverage.

## Scaly Hessians against October 12

| Problem | Size | October 12, µs | Previous CV, % | October 13, µs | Runtime change, % | Executable bytes change, % | Kernel compile change, % |
|---|---:|---:|---:|---:|---:|---:|---:|
| race_cars | 1 | 0.357 | 0.10 | 0.358 | +0.1 | +0.0 | +0.1 |
| race_cars | 5 | 0.864 | 1.12 | 0.851 | -1.5 | -22.4 | +0.1 |
| race_cars | 10 | 1.030 | 1.90 | 1.090 | +5.8 | -22.7 | +3.4 |
| race_cars | 25 | 1.977 | 1.01 | 2.012 | +1.8 | -22.7 | +1.1 |
| race_cars | 40 | 3.002 | 1.80 | 2.810 | -6.4 | -22.7 | +0.0 |
| race_cars | 50 | 3.471 | 2.47 | 3.499 | +0.8 | -22.7 | +0.1 |
| race_cars | 100 | 6.918 | 4.12 | 6.378 | -7.8 | -23.0 | +0.8 |
| race_cars | 200 | 12.649 | 2.11 | 11.833 | -6.4 | -23.0 | +0.1 |
| race_cars | 500 | 31.402 | 4.77 | 28.727 | -8.5 | -23.0 | +3.6 |
| npmpc | 6 | 13.044 | 1.39 | 11.004 | -15.6 | -64.3 | -4.9 |
| npmpc | 12 | 22.342 | 0.46 | 25.103 | +12.4 | -64.5 | -3.1 |
| npmpc | 25 | 44.690 | 5.49 | 57.214 | +28.0 | -64.6 | -3.0 |
| npmpc | 50 | 90.509 | 2.97 | 77.436 | -14.4 | -64.5 | -2.9 |
| npmpc | 100 | 142.471 | 0.35 | 168.716 | +18.4 | -64.6 | -2.5 |
| npmpc | 200 | 245.132 | 0.88 | 256.101 | +4.5 | -64.6 | -5.3 |
| unbumpercars | 2 | 185.320 | 0.22 | 226.595 | +22.3 | -63.8 | -5.2 |
| unbumpercars | 4 | 212.036 | 0.12 | 280.702 | +32.4 | -59.6 | -2.8 |
| unbumpercars | 8 | 487.540 | 0.46 | 613.593 | +25.9 | -59.3 | -2.3 |
| unbumpercars | 16 | 1538.086 | 0.28 | 1798.427 | +16.9 | -59.0 | -3.0 |
| unbumpercars | 32 | 5167.377 | 2.25 | 5538.200 | +7.2 | -58.3 | -3.1 |
| chain | 3 | 21.672 | 0.24 | 21.731 | +0.3 | -39.7 | -0.8 |
| chain | 5 | 99.802 | 0.04 | 99.876 | +0.1 | -24.3 | +1.1 |
| chain | 9 | 282.448 | 0.08 | 283.303 | +0.3 | -22.4 | -0.6 |

## Closed-loop means against October 12

| Problem | Solver | Provider | October 12 total, ms | October 13 total, ms | Total change, % | October 12 FE, ms | October 13 FE, ms | FE change, % |
|---|---|---|---:|---:|---:|---:|---:|---:|
| chain | ipopt | scaly | 6.849 | 6.825 | -0.4 | 0.402 | 0.398 | -1.1 |
| chain | sqp | casadi | 14.424 | 6.388 | -55.7 | 8.121 | 0.160 | -98.0 |
| chain | sqp | scaly | 6.306 | 6.408 | +1.6 | 0.181 | 0.181 | +0.3 |
| npmpc | ipopt | casadi | 4.460 | 4.450 | -0.2 | 1.733 | 1.733 | +0.0 |
| npmpc | ipopt | scaly | 3.531 | 3.549 | +0.5 | 0.963 | 0.968 | +0.5 |
| npmpc | sqp | casadi | 1.851 | 1.843 | -0.4 | 1.027 | 1.026 | -0.1 |
| npmpc | sqp | scaly | 1.273 | 1.267 | -0.5 | 0.473 | 0.470 | -0.6 |
| race_cars | ipopt | casadi | 4.915 | 4.894 | -0.4 | 0.373 | 0.372 | -0.2 |
| race_cars | ipopt | scaly | 4.597 | 4.536 | -1.3 | 0.143 | 0.143 | +0.4 |
| race_cars | sqp | casadi | 2.673 | 2.569 | -3.9 | 0.220 | 0.122 | -44.7 |
| race_cars | sqp | scaly | 2.470 | 2.488 | +0.7 | 0.040 | 0.041 | +2.1 |
| unbumpercars | ipopt | casadi | 146.734 | 146.510 | -0.2 | 142.201 | 142.008 | -0.1 |
| unbumpercars | ipopt | scaly | 20.349 | 20.237 | -0.5 | 16.405 | 16.307 | -0.6 |
| unbumpercars | sqp | casadi | 107.007 | 87.364 | -18.4 | 105.636 | 86.024 | -18.6 |
| unbumpercars | sqp | scaly | 10.791 | 10.717 | -0.7 | 9.509 | 9.443 | -0.7 |

## Fastest completed CasADi Hessians

| Problem | Size | Encoding | CasADi / Scaly |
|---|---:|---|---:|
| race_cars | 1 | casadi_sx | 1.117 |
| race_cars | 5 | casadi_sx | 2.264 |
| race_cars | 10 | casadi_sx | 3.497 |
| race_cars | 25 | casadi_sx | 4.811 |
| race_cars | 40 | casadi_sx | 5.567 |
| race_cars | 50 | casadi_sx | 5.544 |
| race_cars | 100 | casadi_sx | 6.147 |
| race_cars | 200 | casadi_sx | 6.857 |
| race_cars | 500 | casadi_sx | 7.131 |
| npmpc | 6 | casadi_mx | 1.892 |
| npmpc | 12 | casadi_mx | 1.674 |
| npmpc | 25 | casadi_mx | 1.489 |
| npmpc | 50 | casadi_mx | 2.272 |
| npmpc | 100 | casadi_mx_gemm_classic | 5.180 |
| npmpc | 200 | casadi_call_mx | 18.147 |
| unbumpercars | 2 | casadi_mx | 4.208 |
| unbumpercars | 4 | casadi_mx | 10.802 |
| unbumpercars | 8 | casadi_mx | 17.803 |
| chain | 3 | casadi_sx | 1.191 |
| chain | 5 | casadi_sx | 0.824 |
| chain | 9 | casadi_map_sx | 8.623 |

## Current full report

# Study report: study-2026-10-13-post-179

## Sweep: chain

15 rows: 10 ok, 2 skipped_after_failure, 3 timeout.

| Size | Backend | Successful processes | Mean, µs | CV, % | Executable bytes | Workspace, doubles | Kernel compile mean, ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3 | `scaly` | 1/1 | 21.731 |  | 348851 | 26028 | 2715.0 |
| 3 | `casadi_sx` | 1/1 | 25.874 |  | 3330455 | 188 | 19105.1 |
| 3 | `casadi_mx` | 0/1 | timeout |  | 9410547 | 23678 |  |
| 3 | `casadi_call_mx` | 1/1 | 180.786 |  | 602188 | 27321 | 12979.6 |
| 3 | `casadi_map_sx` | 1/1 | 123.369 |  | 556793 | 94736 | 4608.3 |
| 5 | `scaly` | 1/1 | 99.876 |  | 967447 | 0 | 4066.3 |
| 5 | `casadi_sx` | 1/1 | 82.284 |  | 10330465 | 335 | 92166.7 |
| 5 | `casadi_mx` | 0/1 | skipped_after_failure |  |  |  |  |
| 5 | `casadi_call_mx` | 1/1 | 808.446 |  | 2097970 | 100698 | 78113.0 |
| 5 | `casadi_map_sx` | 1/1 | 626.626 |  | 2561051 | 349353 | 10395.4 |
| 9 | `scaly` | 1/1 | 283.303 |  | 2512514 | 0 | 18512.1 |
| 9 | `casadi_sx` | 0/1 | timeout |  | 24350782 | 897 |  |
| 9 | `casadi_mx` | 0/1 | skipped_after_failure |  |  |  |  |
| 9 | `casadi_call_mx` | 0/1 | timeout |  | 6664454 | 280960 |  |
| 9 | `casadi_map_sx` | 1/1 | 2442.952 |  | 9650809 | 999844 | 59597.7 |

## Sweep: npmpc

48 rows: 34 ok, 8 skipped_after_failure, 6 timeout.

| Size | Backend | Successful processes | Mean, µs | CV, % | Executable bytes | Workspace, doubles | Kernel compile mean, ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 6 | `scaly` | 1/1 | 11.004 |  | 194349 | 0 | 945.1 |
| 6 | `casadi_sx` | 1/1 | 72.221 |  | 7564935 | 15913 | 105446.5 |
| 6 | `casadi_mx` | 1/1 | 20.816 |  | 151448 | 7570 | 5126.0 |
| 6 | `casadi_call_mx` | 1/1 | 137.014 |  | 232343 | 46861 | 6729.9 |
| 6 | `casadi_map_sx` | 1/1 | 197.204 |  | 4261664 | 341213 | 138242.2 |
| 6 | `casadi_mx_gemm` | 1/1 | 43.869 |  | 109496 | 8660 | 2465.5 |
| 6 | `casadi_mx_gemm_classic` | 1/1 | 43.829 |  | 109504 | 8660 | 2449.8 |
| 6 | `casadi_mx_gemm_blasfeo` | 1/1 | 43.840 |  | 109504 | 8660 | 2415.1 |
| 12 | `scaly` | 1/1 | 25.103 |  | 194609 | 0 | 1566.5 |
| 12 | `casadi_sx` | 0/1 | timeout |  | 15063206 | 30463 |  |
| 12 | `casadi_mx` | 1/1 | 42.012 |  | 276488 | 9813 | 12698.3 |
| 12 | `casadi_call_mx` | 1/1 | 298.895 |  | 371542 | 68072 | 10306.0 |
| 12 | `casadi_map_sx` | 1/1 | 462.964 |  | 5788748 | 774841 | 177226.2 |
| 12 | `casadi_mx_gemm` | 1/1 | 102.465 |  | 189660 | 13031 | 5116.5 |
| 12 | `casadi_mx_gemm_classic` | 1/1 | 99.792 |  | 189668 | 13031 | 5290.4 |
| 12 | `casadi_mx_gemm_blasfeo` | 1/1 | 99.894 |  | 189668 | 13031 | 5289.0 |
| 25 | `scaly` | 1/1 | 57.214 |  | 193605 | 2700 | 1619.4 |
| 25 | `casadi_sx` | 0/1 | skipped_after_failure |  |  |  |  |
| 25 | `casadi_mx` | 1/1 | 85.193 |  | 545609 | 15595 | 41522.3 |
| 25 | `casadi_call_mx` | 1/1 | 584.994 |  | 424096 | 89139 | 11475.9 |
| 25 | `casadi_map_sx` | 0/1 | timeout |  | 7858327 | 1603408 |  |
| 25 | `casadi_mx_gemm` | 1/1 | 210.141 |  | 343012 | 22648 | 8254.1 |
| 25 | `casadi_mx_gemm_classic` | 1/1 | 209.343 |  | 343020 | 22648 | 8222.4 |
| 25 | `casadi_mx_gemm_blasfeo` | 1/1 | 212.991 |  | 343020 | 22648 | 8439.8 |
| 50 | `scaly` | 1/1 | 77.436 |  | 194712 | 11670 | 1668.7 |
| 50 | `casadi_sx` | 0/1 | skipped_after_failure |  |  |  |  |
| 50 | `casadi_mx` | 1/1 | 175.937 |  | 1063583 | 25782 | 162399.1 |
| 50 | `casadi_call_mx` | 1/1 | 1154.575 |  | 520231 | 130185 | 13704.2 |
| 50 | `casadi_map_sx` | 0/1 | skipped_after_failure |  |  |  |  |
| 50 | `casadi_mx_gemm` | 1/1 | 433.856 |  | 638917 | 41169 | 25917.3 |
| 50 | `casadi_mx_gemm_classic` | 1/1 | 431.725 |  | 638925 | 41169 | 26220.7 |
| 50 | `casadi_mx_gemm_blasfeo` | 1/1 | 431.391 |  | 638925 | 41169 | 25704.9 |
| 100 | `scaly` | 1/1 | 168.716 |  | 194842 | 23270 | 1712.6 |
| 100 | `casadi_sx` | 0/1 | skipped_after_failure |  |  |  |  |
| 100 | `casadi_mx` | 0/1 | timeout |  | 2191474 | 46937 |  |
| 100 | `casadi_call_mx` | 1/1 | 2299.706 |  | 712196 | 210580 | 21151.4 |
| 100 | `casadi_map_sx` | 0/1 | skipped_after_failure |  |  |  |  |
| 100 | `casadi_mx_gemm` | 1/1 | 882.691 |  | 1232183 | 77961 | 90572.0 |
| 100 | `casadi_mx_gemm_classic` | 1/1 | 873.946 |  | 1232191 | 77961 | 90815.2 |
| 100 | `casadi_mx_gemm_blasfeo` | 1/1 | 886.298 |  | 1232191 | 77961 | 92209.3 |
| 200 | `scaly` | 1/1 | 256.101 |  | 195018 | 51870 | 1003.8 |
| 200 | `casadi_sx` | 0/1 | skipped_after_failure |  |  |  |  |
| 200 | `casadi_mx` | 0/1 | skipped_after_failure |  |  |  |  |
| 200 | `casadi_call_mx` | 1/1 | 4647.427 |  | 1127544 | 371260 | 52059.2 |
| 200 | `casadi_map_sx` | 0/1 | skipped_after_failure |  |  |  |  |
| 200 | `casadi_mx_gemm` | 0/1 | timeout |  | 2524675 | 151836 |  |
| 200 | `casadi_mx_gemm_classic` | 0/1 | timeout |  | 2524683 | 151836 |  |
| 200 | `casadi_mx_gemm_blasfeo` | 0/1 | timeout |  | 2524683 | 151836 |  |

## Sweep: race_cars

45 rows: 42 ok, 2 skipped_after_failure, 1 timeout.

| Size | Backend | Successful processes | Mean, µs | CV, % | Executable bytes | Workspace, doubles | Kernel compile mean, ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `scaly` | 1/1 | 0.358 |  | 22909 | 0 | 189.8 |
| 1 | `casadi_sx` | 1/1 | 0.400 |  | 25755 | 108 | 168.4 |
| 1 | `casadi_mx` | 1/1 | 0.505 |  | 95400 | 304 | 389.1 |
| 1 | `casadi_call_mx` | 1/1 | 0.956 |  | 228004 | 1067 | 773.0 |
| 1 | `casadi_map_sx` | 1/1 | 0.780 |  | 94438 | 826 | 391.6 |
| 5 | `scaly` | 1/1 | 0.851 |  | 77929 | 0 | 306.5 |
| 5 | `casadi_sx` | 1/1 | 1.927 |  | 107717 | 119 | 334.4 |
| 5 | `casadi_mx` | 1/1 | 2.854 |  | 441760 | 1085 | 1725.5 |
| 5 | `casadi_call_mx` | 1/1 | 5.139 |  | 248703 | 2244 | 1796.0 |
| 5 | `casadi_map_sx` | 1/1 | 5.253 |  | 120288 | 5914 | 1972.2 |
| 10 | `scaly` | 1/1 | 1.090 |  | 77942 | 0 | 336.0 |
| 10 | `casadi_sx` | 1/1 | 3.810 |  | 209958 | 119 | 570.1 |
| 10 | `casadi_mx` | 1/1 | 5.611 |  | 879492 | 2065 | 4296.3 |
| 10 | `casadi_call_mx` | 1/1 | 10.112 |  | 269955 | 3494 | 2526.6 |
| 10 | `casadi_map_sx` | 1/1 | 10.216 |  | 142699 | 11334 | 1909.8 |
| 25 | `scaly` | 1/1 | 2.012 |  | 78000 | 0 | 466.4 |
| 25 | `casadi_sx` | 1/1 | 9.682 |  | 516870 | 119 | 1312.2 |
| 25 | `casadi_mx` | 1/1 | 14.320 |  | 2360004 | 5020 | 23224.2 |
| 25 | `casadi_call_mx` | 1/1 | 25.111 |  | 334019 | 7484 | 2032.9 |
| 25 | `casadi_map_sx` | 1/1 | 25.510 |  | 206150 | 27594 | 1153.0 |
| 40 | `scaly` | 1/1 | 2.810 |  | 78000 | 0 | 455.5 |
| 40 | `casadi_sx` | 1/1 | 15.645 |  | 823845 | 119 | 2065.8 |
| 40 | `casadi_mx` | 1/1 | 22.875 |  | 3799659 | 7975 | 61094.6 |
| 40 | `casadi_call_mx` | 1/1 | 40.188 |  | 398502 | 11414 | 2533.5 |
| 40 | `casadi_map_sx` | 1/1 | 41.264 |  | 273406 | 43854 | 1266.8 |
| 50 | `scaly` | 1/1 | 3.499 |  | 78000 | 0 | 367.9 |
| 50 | `casadi_sx` | 1/1 | 19.396 |  | 1028495 | 119 | 2677.5 |
| 50 | `casadi_mx` | 1/1 | 28.740 |  | 4762326 | 9945 | 98736.0 |
| 50 | `casadi_call_mx` | 1/1 | 50.330 |  | 441709 | 14034 | 7190.6 |
| 50 | `casadi_map_sx` | 1/1 | 51.546 |  | 318176 | 54694 | 5473.1 |
| 100 | `scaly` | 1/1 | 6.378 |  | 78013 | 0 | 442.0 |
| 100 | `casadi_sx` | 1/1 | 39.203 |  | 2052051 | 119 | 5977.2 |
| 100 | `casadi_mx` | 0/1 | timeout |  | 9614050 | 19795 |  |
| 100 | `casadi_call_mx` | 1/1 | 100.710 |  | 679651 | 27134 | 7506.3 |
| 100 | `casadi_map_sx` | 1/1 | 104.687 |  | 543857 | 108894 | 3302.4 |
| 200 | `scaly` | 1/1 | 11.833 |  | 78049 | 0 | 462.6 |
| 200 | `casadi_sx` | 1/1 | 81.144 |  | 4100194 | 119 | 14274.1 |
| 200 | `casadi_mx` | 0/1 | skipped_after_failure |  |  |  |  |
| 200 | `casadi_call_mx` | 1/1 | 201.589 |  | 1143240 | 53874 | 19508.2 |
| 200 | `casadi_map_sx` | 1/1 | 212.282 |  | 1030885 | 217294 | 6673.7 |
| 500 | `scaly` | 1/1 | 28.727 |  | 78071 | 0 | 471.5 |
| 500 | `casadi_sx` | 1/1 | 204.837 |  | 10246853 | 119 | 49853.4 |
| 500 | `casadi_mx` | 0/1 | skipped_after_failure |  |  |  |  |
| 500 | `casadi_call_mx` | 1/1 | 521.934 |  | 2552822 | 134274 | 116176.8 |
| 500 | `casadi_map_sx` | 1/1 | 543.431 |  | 2535806 | 542494 | 31983.4 |

## Sweep: unbumpercars

30 rows: 17 ok, 8 skipped_after_failure, 5 timeout.

| Size | Backend | Successful processes | Mean, µs | CV, % | Executable bytes | Workspace, doubles | Kernel compile mean, ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | `scaly` | 1/1 | 226.595 |  | 377583 | 122880 | 1428.2 |
| 2 | `casadi_sx` | 0/1 | timeout |  | 26778898 | 37182 |  |
| 2 | `casadi_mx` | 1/1 | 953.436 |  | 110035 | 110087 | 2344.9 |
| 2 | `casadi_mx_gemm` | 1/1 | 1085.343 |  | 124566 | 112315 | 2210.3 |
| 2 | `casadi_mx_gemm_classic` | 1/1 | 1060.293 |  | 124574 | 112315 | 2308.6 |
| 2 | `casadi_mx_gemm_blasfeo` | 1/1 | 1069.058 |  | 124574 | 112315 | 2210.7 |
| 4 | `scaly` | 1/1 | 280.702 |  | 474970 | 100352 | 2605.8 |
| 4 | `casadi_sx` | 0/1 | skipped_after_failure |  |  |  |  |
| 4 | `casadi_mx` | 1/1 | 3032.048 |  | 361999 | 112213 | 7401.6 |
| 4 | `casadi_mx_gemm` | 1/1 | 3749.838 |  | 629111 | 120257 | 5002.7 |
| 4 | `casadi_mx_gemm_classic` | 1/1 | 3793.565 |  | 629119 | 120257 | 5014.4 |
| 4 | `casadi_mx_gemm_blasfeo` | 1/1 | 3792.776 |  | 629119 | 120257 | 5013.4 |
| 8 | `scaly` | 1/1 | 613.593 |  | 484845 | 117600 | 3946.3 |
| 8 | `casadi_sx` | 0/1 | skipped_after_failure |  |  |  |  |
| 8 | `casadi_mx` | 1/1 | 10924.094 |  | 1327412 | 116049 | 52674.9 |
| 8 | `casadi_mx_gemm` | 1/1 | 14174.614 |  | 4569620 | 136111 | 35068.2 |
| 8 | `casadi_mx_gemm_classic` | 1/1 | 14212.641 |  | 4569628 | 136111 | 34866.4 |
| 8 | `casadi_mx_gemm_blasfeo` | 1/1 | 14161.778 |  | 4569628 | 136111 | 35084.6 |
| 16 | `scaly` | 1/1 | 1798.427 |  | 504840 | 243600 | 3140.8 |
| 16 | `casadi_sx` | 0/1 | skipped_after_failure |  |  |  |  |
| 16 | `casadi_mx` | 0/1 | timeout |  | 5399074 | 125987 |  |
| 16 | `casadi_mx_gemm` | 0/1 | timeout |  | 35514289 | 171611 |  |
| 16 | `casadi_mx_gemm_classic` | 0/1 | timeout |  | 35514297 | 171611 |  |
| 16 | `casadi_mx_gemm_blasfeo` | 0/1 | timeout |  | 35514297 | 171611 |  |
| 32 | `scaly` | 1/1 | 5538.200 |  | 542649 | 1168752 | 3537.1 |
| 32 | `casadi_sx` | 0/1 | skipped_after_failure |  |  |  |  |
| 32 | `casadi_mx` | 0/1 | skipped_after_failure |  |  |  |  |
| 32 | `casadi_mx_gemm` | 0/1 | skipped_after_failure |  |  |  |  |
| 32 | `casadi_mx_gemm_classic` | 0/1 | skipped_after_failure |  |  |  |  |
| 32 | `casadi_mx_gemm_blasfeo` | 0/1 | skipped_after_failure |  |  |  |  |

## Closed loop

| Problem | Solver | Provider | Total, ms | Step median, ms | Step SD, ms | Function evaluation, ms | QP, ms | Globalization, ms | Glue, ms | Steps | Processes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chain | ipopt | scaly | 6.825 | 6.539 | 1.077 | 0.398 | 6.148 | 0.000 | 0.278 | 90 | 1 |
| chain | sqp | casadi | 6.388 | 6.271 | 0.737 | 0.160 | 5.463 | 0.008 | 0.757 | 90 | 1 |
| chain | sqp | scaly | 6.408 | 6.328 | 0.678 | 0.181 | 5.497 | 0.008 | 0.722 | 90 | 1 |
| npmpc | ipopt | casadi | 4.450 | 4.362 | 0.814 | 1.733 | 2.538 | 0.000 | 0.179 | 100 | 1 |
| npmpc | ipopt | scaly | 3.549 | 3.465 | 0.573 | 0.968 | 2.328 | 0.000 | 0.252 | 100 | 1 |
| npmpc | sqp | casadi | 1.843 | 1.472 | 1.369 | 1.026 | 0.688 | 0.004 | 0.125 | 100 | 1 |
| npmpc | sqp | scaly | 1.267 | 1.013 | 0.935 | 0.470 | 0.677 | 0.004 | 0.116 | 100 | 1 |
| race_cars | ipopt | casadi | 4.894 | 4.855 | 0.428 | 0.372 | 4.316 | 0.000 | 0.206 | 1367 | 1 |
| race_cars | ipopt | scaly | 4.536 | 4.584 | 0.350 | 0.143 | 4.107 | 0.000 | 0.286 | 1367 | 1 |
| race_cars | sqp | casadi | 2.569 | 2.335 | 1.261 | 0.122 | 2.351 | 0.007 | 0.088 | 1367 | 1 |
| race_cars | sqp | scaly | 2.488 | 2.251 | 1.258 | 0.041 | 2.360 | 0.007 | 0.079 | 1367 | 1 |
| unbumpercars | ipopt | casadi | 146.510 | 139.130 | 37.809 | 142.008 | 0.000 | 0.000 | 0.274 | 200 | 1 |
| unbumpercars | ipopt | scaly | 20.237 | 19.098 | 5.192 | 16.307 | 0.000 | 0.000 | 0.257 | 200 | 1 |
| unbumpercars | sqp | casadi | 87.364 | 54.958 | 87.126 | 86.024 | 1.287 | 0.011 | 0.042 | 200 | 1 |
| unbumpercars | sqp | scaly | 10.717 | 6.751 | 10.533 | 9.443 | 1.229 | 0.010 | 0.034 | 200 | 1 |

| Problem | Solver | Repetitions | Per-step speedup, median (10th–90th percentile) | Per-step function-evaluation speedup | Per-step iteration mismatches | Per-step oracle-count mismatches | Maximum state difference | Maximum control difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| chain | sqp | 1 | 0.99× (0.99–0.99) | 0.87× (0.83–0.90) | 0 | 0 | 3.53e-09 | 1.24e-08 |
| npmpc | ipopt | 1 | 1.26× (1.22–1.27) | 1.79× (1.76–1.80) | 0 | 0 | 1.21e-12 | 6.09e-15 |
| npmpc | sqp | 1 | 1.45× (1.44–1.47) | 2.18× (2.15–2.23) | 0 | 0 | 3.02e-09 | 1.46e-11 |
| race_cars | ipopt | 1 | 1.08× (1.07–1.09) | 2.60× (2.49–2.68) | 0 | 0 | 5.68e-14 | 5.19e-12 |
| race_cars | sqp | 1 | 1.04× (1.02–1.05) | 2.96× (2.78–3.10) | 0 | 0 | 4.83e-12 | 4.17e-09 |
| unbumpercars | ipopt | 1 | 7.24× (7.08–7.37) | 8.71× (8.54–8.85) | 0 | 0 | 9.08e-12 | 6.62e-12 |
| unbumpercars | sqp | 1 | 8.10× (7.55–8.39) | 9.10× (8.44–9.40) | 3 | 0 | 6.09e-09 | 4.91e-09 |

