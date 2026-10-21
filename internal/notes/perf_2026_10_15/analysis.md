# October 15 study audit

Duration: 76.7 minutes.

- race_cars: {'ok': 32, 'timeout': 1, 'skipped_after_failure': 2}
- npmpc: {'ok': 24, 'timeout': 4, 'skipped_after_failure': 8}
- unbumpercars: {'ok': 11, 'timeout': 3, 'skipped_after_failure': 6}
- chain: {'timeout': 3, 'ok': 10, 'skipped_after_failure': 2}

October 14 outcomes: {'ok': 87, 'timeout': 11, 'skipped_after_failure': 18}.

All 6938 solver steps succeeded. All timed kernels passed the symbolic and compiled-C gates.

## Paired speedups

| Problem | Solver | Metric | Median | P10 | P90 |
|---|---|---|---:|---:|---:|
| chain | sqp | solver_time_ms | 1.0030 | 1.0012 | 1.0052 |
| chain | sqp | fe_time_ms | 0.8602 | 0.8433 | 0.8922 |
| npmpc | ipopt | solver_time_ms | 1.2451 | 1.1148 | 1.2642 |
| npmpc | ipopt | fe_time_ms | 1.7793 | 1.7424 | 1.8048 |
| npmpc | sqp | solver_time_ms | 1.4591 | 1.4394 | 1.4735 |
| npmpc | sqp | fe_time_ms | 2.2166 | 2.1671 | 2.2554 |
| race_cars | ipopt | solver_time_ms | 1.0830 | 1.0734 | 1.0928 |
| race_cars | ipopt | fe_time_ms | 2.7203 | 2.6268 | 2.7884 |
| race_cars | sqp | solver_time_ms | 1.0378 | 1.0122 | 1.0553 |
| race_cars | sqp | fe_time_ms | 3.1215 | 2.8886 | 3.2562 |
| unbumpercars | ipopt | solver_time_ms | 7.2826 | 7.1303 | 7.4394 |
| unbumpercars | ipopt | fe_time_ms | 8.7579 | 8.5886 | 8.9024 |
| unbumpercars | sqp | solver_time_ms | 8.0997 | 7.6775 | 8.3923 |
| unbumpercars | sqp | fe_time_ms | 9.1057 | 8.6590 | 9.4029 |

## Controller steps

| Problem | Solver | Provider | Median, ms | Maximum, ms | Over sampling time | Mean iterations |
|---|---|---|---:|---:|---:|---:|
| chain | ipopt | scaly | 6.5286 | 10.7901 | 0 | 7.400 |
| chain | sqp | casadi | 6.2301 | 8.1616 | 0 | 2.989 |
| chain | sqp | scaly | 6.2141 | 8.0767 | 0 | 2.989 |
| npmpc | ipopt | casadi | 4.3549 | 10.7349 | 0 | 10.070 |
| npmpc | ipopt | scaly | 3.6565 | 6.5501 | 0 | 10.070 |
| npmpc | sqp | casadi | 1.4824 | 11.1600 | 0 | 4.660 |
| npmpc | sqp | scaly | 1.0184 | 7.6152 | 0 | 4.660 |
| race_cars | ipopt | casadi | 5.8843 | 16.6895 | 0 | 10.778 |
| race_cars | ipopt | scaly | 5.4267 | 11.4298 | 0 | 10.778 |
| race_cars | sqp | casadi | 2.9226 | 9.7844 | 0 | 2.726 |
| race_cars | sqp | scaly | 2.8142 | 9.4176 | 0 | 2.726 |
| unbumpercars | ipopt | casadi | 139.4075 | 321.7916 | 186 | 13.715 |
| unbumpercars | ipopt | scaly | 19.0978 | 42.3252 | 0 | 13.715 |
| unbumpercars | sqp | casadi | 55.0315 | 661.7923 | 50 | 7.870 |
| unbumpercars | sqp | scaly | 6.7531 | 79.3265 | 0 | 7.875 |

## Unchanged sweep cells since October 14

Ratios are October 15 divided by October 14. One process gives no noise estimate.

| Problem | Size | Backend | Runtime ratio | Kernel compile ratio | Source byte change | Object byte change |
|---|---:|---|---:|---:|---:|---:|
| race_cars | 5 | casadi_mx | 1.0094 | 0.9936 | 0 | 0 |
| race_cars | 5 | casadi_sx | 1.0003 | 0.9943 | 0 | 0 |
| race_cars | 5 | scaly | 1.0001 | 0.9476 | 0 | 0 |
| race_cars | 5 | casadi_map_sx | 0.9926 | 0.9951 | 0 | 0 |
| race_cars | 5 | casadi_call_mx | 1.0005 | 0.9983 | 0 | 0 |
| race_cars | 10 | casadi_mx | 1.0425 | 0.9864 | 0 | 0 |
| race_cars | 10 | casadi_sx | 1.0001 | 0.9981 | 0 | 0 |
| race_cars | 10 | scaly | 1.0098 | 0.9937 | 0 | 0 |
| race_cars | 10 | casadi_map_sx | 1.0008 | 1.0249 | 0 | 0 |
| race_cars | 10 | casadi_call_mx | 0.9900 | 0.9998 | 0 | 0 |
| race_cars | 25 | casadi_mx | 0.9966 | 1.0018 | 0 | 0 |
| race_cars | 25 | casadi_sx | 0.9985 | 0.9961 | 0 | 0 |
| race_cars | 25 | scaly | 1.0267 | 0.9936 | 0 | 0 |
| race_cars | 25 | casadi_map_sx | 1.0027 | 0.9973 | 0 | 0 |
| race_cars | 25 | casadi_call_mx | 1.0007 | 1.0025 | 0 | 0 |
| race_cars | 50 | casadi_mx | 1.0000 | 0.9787 | 0 | 0 |
| race_cars | 50 | casadi_sx | 1.0028 | 1.0137 | 0 | 0 |
| race_cars | 50 | scaly | 0.9732 | 0.9857 | 0 | 0 |
| race_cars | 50 | casadi_map_sx | 0.9958 | 1.0309 | 0 | 0 |
| race_cars | 50 | casadi_call_mx | 0.9986 | 0.9616 | 0 | 0 |
| race_cars | 100 | casadi_sx | 0.9957 | 0.9800 | 0 | 0 |
| race_cars | 100 | scaly | 1.0040 | 1.0261 | 0 | 0 |
| race_cars | 100 | casadi_map_sx | 0.9971 | 0.9668 | 0 | 0 |
| race_cars | 100 | casadi_call_mx | 0.9978 | 1.0320 | 0 | 0 |
| race_cars | 200 | casadi_sx | 1.0045 | 0.9892 | 0 | 0 |
| race_cars | 200 | scaly | 1.0178 | 0.9997 | 0 | 0 |
| race_cars | 200 | casadi_map_sx | 1.0065 | 0.9870 | 0 | 0 |
| race_cars | 200 | casadi_call_mx | 1.0038 | 1.0143 | 0 | 0 |
| npmpc | 6 | casadi_map_sx | 0.9533 | 0.9890 | 0 | 0 |
| npmpc | 6 | casadi_mx | 1.0005 | 0.9698 | 0 | 0 |
| npmpc | 6 | casadi_sx | 0.9941 | 0.9484 | 0 | 0 |
| npmpc | 6 | scaly | 1.0087 | 0.9954 | 0 | 0 |
| npmpc | 6 | casadi_mx_gemm | 0.9991 | 0.9981 | 0 | 0 |
| npmpc | 6 | casadi_call_mx | 0.9827 | 0.9954 | 0 | 0 |
| npmpc | 12 | casadi_map_sx | 0.9639 | 0.9827 | 0 | 0 |
| npmpc | 12 | casadi_mx | 1.0093 | 1.0022 | 0 | 0 |
| npmpc | 12 | scaly | 1.0169 | 0.9950 | 0 | 0 |
| npmpc | 12 | casadi_mx_gemm | 1.0044 | 0.9915 | 0 | 0 |
| npmpc | 12 | casadi_call_mx | 1.0040 | 0.9960 | 0 | 0 |
| npmpc | 25 | casadi_mx | 1.1777 | 0.9688 | 0 | 0 |
| npmpc | 25 | scaly | 0.9955 | 0.9937 | 0 | 0 |
| npmpc | 25 | casadi_mx_gemm | 0.9838 | 0.9791 | 0 | 0 |
| npmpc | 25 | casadi_call_mx | 1.0021 | 0.9909 | 0 | 0 |
| npmpc | 50 | casadi_mx | 1.0038 | 1.0244 | 0 | 0 |
| npmpc | 50 | scaly | 0.9948 | 1.0032 | 0 | 0 |
| npmpc | 50 | casadi_mx_gemm | 0.9884 | 0.9976 | 0 | 0 |
| npmpc | 50 | casadi_call_mx | 0.9993 | 0.9748 | 0 | 0 |
| npmpc | 100 | scaly | 0.9928 | 1.0499 | 0 | 0 |
| npmpc | 100 | casadi_mx_gemm | 1.0190 | 1.0268 | 0 | 0 |
| npmpc | 100 | casadi_call_mx | 0.9921 | 1.0000 | 0 | 0 |
| npmpc | 200 | scaly | 0.9214 | 0.9943 | 0 | 0 |
| npmpc | 200 | casadi_call_mx | 0.9844 | 0.9789 | 0 | 0 |
| unbumpercars | 2 | casadi_mx | 1.0103 | 1.0042 | 0 | 0 |
| unbumpercars | 2 | scaly | 0.9958 | 1.0033 | 0 | 0 |
| unbumpercars | 2 | casadi_mx_gemm | 1.0063 | 0.9961 | 0 | 0 |
| unbumpercars | 4 | casadi_mx | 0.9874 | 0.9635 | 0 | 0 |
| unbumpercars | 4 | scaly | 1.0001 | 1.0026 | 0 | 0 |
| unbumpercars | 4 | casadi_mx_gemm | 0.9859 | 0.9939 | 0 | 0 |
| unbumpercars | 8 | casadi_mx | 1.0134 | 0.9857 | 0 | 0 |
| unbumpercars | 8 | scaly | 0.9942 | 0.9927 | 0 | 0 |
| unbumpercars | 8 | casadi_mx_gemm | 0.9942 | 0.9988 | 0 | 0 |
| unbumpercars | 16 | scaly | 0.9983 | 1.0129 | 0 | 0 |
| unbumpercars | 32 | scaly | 1.0070 | 0.9980 | 0 | 0 |
| chain | 3 | casadi_sx | 0.9528 | 0.9914 | 0 | 0 |
| chain | 3 | scaly | 1.0071 | 1.0004 | 0 | 0 |
| chain | 3 | casadi_map_sx | 1.0004 | 1.0033 | 0 | 0 |
| chain | 3 | casadi_call_mx | 0.9964 | 0.9935 | 0 | 0 |
| chain | 5 | casadi_sx | 1.0011 | 0.9855 | 0 | 0 |
| chain | 5 | scaly | 0.9996 | 0.9727 | 0 | 0 |
| chain | 5 | casadi_map_sx | 0.9772 | 0.9837 | 0 | 0 |
| chain | 5 | casadi_call_mx | 1.0022 | 1.0026 | 0 | 0 |
| chain | 9 | scaly | 1.0012 | 0.9883 | 0 | 0 |
| chain | 9 | casadi_map_sx | 0.9921 | 0.9959 | 0 | 0 |

Runtime-ratio P10/median/P90: [0.984, 1.0001, 1.0102].

## Figure and prose values

### race_cars

| Size | Best CasADi | Runtime speedup | Scaly build, s | Best CasADi build, s | Scaly object, bytes |
|---:|---|---:|---:|---:|---:|
| 5 | casadi_sx | 2.259 | 0.962 | 0.511 | 29415 |
| 10 | casadi_sx | 3.466 | 0.981 | 0.777 | 32559 |
| 25 | casadi_sx | 4.704 | 1.150 | 1.382 | 45575 |
| 50 | casadi_sx | 5.995 | 1.100 | 3.137 | 32247 |
| 100 | casadi_sx | 6.169 | 1.296 | 3.839 | 42751 |
| 200 | casadi_sx | 6.623 | 1.501 | 8.102 | 43399 |
| 400 | casadi_sx | 7.235 | 1.923 | 28.882 | 43399 |

### npmpc

| Size | Best CasADi | Runtime speedup | Scaly build, s | Best CasADi build, s | Scaly object, bytes |
|---:|---|---:|---:|---:|---:|
| 6 | casadi_mx | 1.863 | 2.159 | 2.778 | 83209 |
| 12 | casadi_mx | 1.687 | 2.778 | 5.503 | 148433 |
| 25 | casadi_mx | 1.885 | 2.826 | 8.607 | 175389 |
| 50 | casadi_mx | 2.268 | 2.891 | 14.144 | 271033 |
| 100 | casadi_mx_gemm | 5.022 | 3.049 | 22.576 | 428137 |
| 200 | casadi_call_mx | 17.929 | 2.283 | 54.887 | 694737 |

### unbumpercars

| Size | Best CasADi | Runtime speedup | Scaly build, s | Best CasADi build, s | Scaly object, bytes |
|---:|---|---:|---:|---:|---:|
| 2 | casadi_mx | 4.275 | 3.066 | 2.459 | 95791 |
| 4 | casadi_mx | 10.801 | 4.598 | 5.447 | 132617 |
| 8 | casadi_mx | 17.657 | 5.928 | 36.995 | 201589 |
| 16 |  |  | 5.349 |  | 262817 |
| 32 |  |  | 7.115 |  | 618929 |

### chain

| Size | Best CasADi | Runtime speedup | Scaly build, s | Best CasADi build, s | Scaly object, bytes |
|---:|---|---:|---:|---:|---:|
| 3 | casadi_sx | 1.178 | 4.146 | 4.872 | 92328 |
| 5 | casadi_sx | 0.824 | 16.973 | 10.672 | 323896 |
| 9 | casadi_map_sx | 8.664 | 58.106 | 60.600 | 907712 |

## Build stages at controller sizes

| Problem | Backend | Build, ms | Render, ms | Compile, ms | Render share, % | Compile share, % |
|---|---|---:|---:|---:|---:|---:|
| race_cars | casadi_mx | 1663.4 | 177.3 | 96800.6 | 0.18 | 98.13 |
| race_cars | casadi_sx | 583.0 | 43.9 | 2509.7 | 1.40 | 80.01 |
| race_cars | scaly | 215.8 | 608.1 | 276.0 | 55.29 | 25.09 |
| race_cars | casadi_map_sx | 396.2 | 13.9 | 5351.3 | 0.24 | 92.88 |
| race_cars | casadi_call_mx | 469.2 | 16.8 | 6924.5 | 0.23 | 93.44 |
| npmpc | casadi_map_sx | 949.2 | 217.1 | 173778.1 | 0.12 | 99.33 |
| npmpc | casadi_mx | 462.9 | 9.8 | 12693.3 | 0.07 | 96.41 |
| npmpc | scaly | 165.4 | 1154.4 | 1458.3 | 41.55 | 52.49 |
| npmpc | casadi_mx_gemm | 468.4 | 6.8 | 5027.5 | 0.12 | 91.36 |
| npmpc | casadi_call_mx | 516.2 | 12.8 | 10166.8 | 0.12 | 95.05 |
| unbumpercars | casadi_mx | 1004.0 | 50.7 | 53057.4 | 0.09 | 98.05 |
| unbumpercars | scaly | 717.2 | 1395.2 | 3816.0 | 23.53 | 64.37 |
| unbumpercars | casadi_mx_gemm | 2144.8 | 176.0 | 34674.6 | 0.48 | 93.73 |
| chain | casadi_sx | 4020.6 | 450.4 | 93144.5 | 0.46 | 95.42 |
| chain | scaly | 596.7 | 12510.5 | 3865.8 | 73.71 | 22.78 |
| chain | casadi_map_sx | 521.0 | 108.6 | 10042.5 | 1.02 | 94.10 |
| chain | casadi_call_mx | 754.4 | 77.8 | 78651.4 | 0.10 | 98.95 |

## Closed-loop changes since October 14

The race-car horizon changed, so its controller timings are not compared.

| Problem | Solver | Provider | Total-time ratio | Function-evaluation ratio |
|---|---|---|---:|---:|
| chain | ipopt | scaly | 0.9938 | 0.9914 |
| chain | sqp | casadi | 0.9974 | 0.9981 |
| chain | sqp | scaly | 0.9963 | 0.9926 |
| npmpc | ipopt | casadi | 1.0030 | 1.0003 |
| npmpc | ipopt | scaly | 1.0370 | 0.9981 |
| npmpc | sqp | casadi | 0.9992 | 1.0044 |
| npmpc | sqp | scaly | 1.0000 | 0.9973 |
| unbumpercars | ipopt | casadi | 1.0027 | 1.0034 |
| unbumpercars | ipopt | scaly | 0.9996 | 1.0007 |
| unbumpercars | sqp | casadi | 0.9957 | 0.9957 |
| unbumpercars | sqp | scaly | 1.0052 | 1.0054 |

## Controller build and function evaluation

- chain ipopt scaly: build 26.5558 s, function-evaluation share 5.826%.
- chain sqp casadi: build 27.7862 s, function-evaluation share 2.484%.
- chain sqp scaly: build 30.4735 s, function-evaluation share 2.867%.
- npmpc ipopt casadi: build 9.9811 s, function-evaluation share 38.808%.
- npmpc ipopt scaly: build 5.6323 s, function-evaluation share 26.278%.
- npmpc sqp casadi: build 7.9141 s, function-evaluation share 55.952%.
- npmpc sqp scaly: build 6.1083 s, function-evaluation share 36.961%.
- race_cars ipopt casadi: build 19.7106 s, function-evaluation share 8.492%.
- race_cars ipopt scaly: build 2.8063 s, function-evaluation share 3.386%.
- race_cars sqp casadi: build 9.5615 s, function-evaluation share 4.821%.
- race_cars sqp scaly: build 2.9813 s, function-evaluation share 1.598%.
- unbumpercars ipopt casadi: build 70.3951 s, function-evaluation share 96.926%.
- unbumpercars ipopt scaly: build 8.1863 s, function-evaluation share 80.571%.
- unbumpercars sqp casadi: build 60.6519 s, function-evaluation share 98.444%.
- unbumpercars sqp scaly: build 12.6137 s, function-evaluation share 88.011%.

## Race-car endpoint build shares

- casadi_sx N=5: render 0.96%, compile 47.72%.
- scaly N=5: render 55.68%, compile 22.75%.
- casadi_sx N=400: render 0.88%, compile 88.05%.
- scaly N=400: render 67.66%, compile 19.57%.

## Retained-binary diagnostic

These repetitions do not replace any published study cell.

| Study | Cell | Minimum, µs | Mean, µs | Maximum, µs | CV, % |
|---|---|---:|---:|---:|---:|
| study-2026-10-14 | casadi_mx_N25 | 84.658 | 89.583 | 99.317 | 9.41 |
| study-2026-10-14 | scaly_N200 | 254.108 | 256.319 | 258.562 | 0.87 |
| study-2026-10-15 | casadi_mx_N25 | 84.742 | 89.720 | 99.339 | 9.29 |
| study-2026-10-15 | scaly_N200 | 255.934 | 258.594 | 262.636 | 1.38 |
