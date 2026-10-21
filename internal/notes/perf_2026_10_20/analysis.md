# October 20 study audit

Duration: 82.9 minutes.

- race_cars: {'ok': 32, 'timeout': 1, 'skipped_after_failure': 2}
- npmpc: {'ok': 26, 'timeout': 4, 'skipped_after_failure': 6}
- unbumpercars: {'ok': 11, 'timeout': 3, 'skipped_after_failure': 6}
- chain: {'timeout': 3, 'ok': 10, 'skipped_after_failure': 2}

October 15 outcomes: {'ok': 77, 'timeout': 11, 'skipped_after_failure': 18}.

All 6938 solver steps succeeded. All timed kernels passed the symbolic and compiled-C gates.

## Paired speedups

| Problem | Solver | Metric | Median | P10 | P90 |
|---|---|---|---:|---:|---:|
| chain | sqp | solver_time_ms | 1.0009 | 0.9986 | 1.0028 |
| chain | sqp | fe_time_ms | 0.8809 | 0.8719 | 0.9063 |
| npmpc | ipopt | solver_time_ms | 1.2659 | 1.2552 | 1.2729 |
| npmpc | ipopt | fe_time_ms | 1.8134 | 1.7868 | 1.8303 |
| npmpc | sqp | solver_time_ms | 1.4578 | 1.4472 | 1.4758 |
| npmpc | sqp | fe_time_ms | 2.2063 | 2.1554 | 2.2425 |
| race_cars | ipopt | solver_time_ms | 1.0831 | 1.0705 | 1.0939 |
| race_cars | ipopt | fe_time_ms | 2.6888 | 2.5927 | 2.7555 |
| race_cars | sqp | solver_time_ms | 1.0470 | 1.0225 | 1.0670 |
| race_cars | sqp | fe_time_ms | 2.8126 | 2.6252 | 2.9243 |
| unbumpercars | ipopt | solver_time_ms | 7.2849 | 7.1478 | 7.4282 |
| unbumpercars | ipopt | fe_time_ms | 8.7813 | 8.6157 | 8.9176 |
| unbumpercars | sqp | solver_time_ms | 7.9705 | 7.6486 | 8.2212 |
| unbumpercars | sqp | fe_time_ms | 8.9310 | 8.5365 | 9.1953 |

## Controller steps

| Problem | Solver | Provider | Median, ms | Maximum, ms | Over sampling time | Mean iterations |
|---|---|---|---:|---:|---:|---:|
| chain | ipopt | scaly | 6.5609 | 10.7435 | 0 | 7.400 |
| chain | sqp | casadi | 6.2722 | 8.1740 | 0 | 2.989 |
| chain | sqp | scaly | 6.2657 | 8.2468 | 0 | 2.989 |
| npmpc | ipopt | casadi | 4.3697 | 10.4712 | 0 | 10.070 |
| npmpc | ipopt | scaly | 3.4500 | 6.5050 | 0 | 10.070 |
| npmpc | sqp | casadi | 1.4716 | 11.0945 | 0 | 4.660 |
| npmpc | sqp | scaly | 1.0109 | 7.5869 | 0 | 4.660 |
| race_cars | ipopt | casadi | 5.8884 | 16.4467 | 0 | 10.778 |
| race_cars | ipopt | scaly | 5.4389 | 11.4161 | 0 | 10.778 |
| race_cars | sqp | casadi | 2.9593 | 9.9379 | 0 | 2.726 |
| race_cars | sqp | scaly | 2.8208 | 9.3995 | 0 | 2.726 |
| unbumpercars | ipopt | casadi | 138.9275 | 297.6087 | 186 | 13.715 |
| unbumpercars | ipopt | scaly | 19.0717 | 42.1443 | 0 | 13.715 |
| unbumpercars | sqp | casadi | 55.2420 | 668.8163 | 48 | 7.870 |
| unbumpercars | sqp | scaly | 6.9033 | 82.6010 | 0 | 7.875 |

## Sweep cells since October 15

Ratios are October 20 divided by October 15. One process gives no noise estimate.

| Problem | Size | Backend | Runtime ratio | Kernel compile ratio | Source byte change | Object byte change |
|---|---:|---|---:|---:|---:|---:|
| race_cars | 5 | casadi_mx | 0.9907 | 0.9969 | 0 | 0 |
| race_cars | 5 | casadi_sx | 0.9994 | 0.9910 | 0 | 0 |
| race_cars | 5 | scaly | 0.9981 | 0.9936 | 0 | 0 |
| race_cars | 5 | casadi_map_sx | 0.9976 | 1.0173 | 0 | 0 |
| race_cars | 5 | casadi_call_mx | 1.0124 | 1.0038 | 0 | 0 |
| race_cars | 10 | casadi_mx | 0.9767 | 1.0021 | 0 | 0 |
| race_cars | 10 | casadi_sx | 0.9997 | 1.0071 | 0 | 0 |
| race_cars | 10 | scaly | 1.0086 | 0.9979 | 0 | 0 |
| race_cars | 10 | casadi_map_sx | 0.9991 | 1.0007 | 0 | 0 |
| race_cars | 10 | casadi_call_mx | 1.0039 | 0.9928 | 0 | 0 |
| race_cars | 25 | casadi_mx | 1.0015 | 1.0244 | 0 | 0 |
| race_cars | 25 | casadi_sx | 0.9986 | 0.9952 | 0 | 0 |
| race_cars | 25 | scaly | 0.9643 | 1.0135 | 0 | 0 |
| race_cars | 25 | casadi_map_sx | 0.9953 | 1.0154 | 0 | 0 |
| race_cars | 25 | casadi_call_mx | 1.0040 | 0.9818 | 0 | 0 |
| race_cars | 50 | casadi_mx | 1.0009 | 1.0049 | 0 | 0 |
| race_cars | 50 | casadi_sx | 0.9985 | 0.9732 | 0 | 0 |
| race_cars | 50 | scaly | 1.0095 | 1.0065 | 0 | 0 |
| race_cars | 50 | casadi_map_sx | 1.0006 | 1.0150 | 0 | 0 |
| race_cars | 50 | casadi_call_mx | 1.0024 | 1.0597 | 0 | 0 |
| race_cars | 100 | casadi_sx | 1.0063 | 1.0122 | 0 | 0 |
| race_cars | 100 | scaly | 1.0010 | 0.9701 | 0 | 0 |
| race_cars | 100 | casadi_map_sx | 1.0003 | 1.0261 | 0 | 0 |
| race_cars | 100 | casadi_call_mx | 1.0457 | 0.9568 | 0 | 0 |
| race_cars | 200 | casadi_sx | 0.9930 | 0.9967 | 0 | 0 |
| race_cars | 200 | scaly | 0.9582 | 1.0000 | 0 | 0 |
| race_cars | 200 | casadi_map_sx | 0.9945 | 0.9987 | 0 | 0 |
| race_cars | 200 | casadi_call_mx | 1.0047 | 0.9763 | 0 | 0 |
| race_cars | 400 | casadi_sx | 0.9803 | 0.9926 | 0 | 0 |
| race_cars | 400 | scaly | 1.0011 | 1.0050 | 0 | 0 |
| race_cars | 400 | casadi_map_sx | 1.0010 | 0.9933 | 0 | 0 |
| race_cars | 400 | casadi_call_mx | 1.0032 | 1.0173 | 0 | 0 |
| npmpc | 6 | casadi_map_sx | 0.9996 | 0.9995 | 0 | 0 |
| npmpc | 6 | casadi_mx | 0.9906 | 1.0016 | 0 | 0 |
| npmpc | 6 | casadi_sx | 1.0081 | 1.0442 | 0 | 0 |
| npmpc | 6 | scaly | 0.9445 | 1.0458 | 6462 | -18952 |
| npmpc | 6 | casadi_mx_gemm | 0.9997 | 0.9700 | 0 | 0 |
| npmpc | 6 | casadi_call_mx | 1.0249 | 1.0037 | 0 | 0 |
| npmpc | 12 | casadi_map_sx | 1.0042 | 0.9999 | 0 | 0 |
| npmpc | 12 | casadi_mx | 0.9896 | 0.9958 | 0 | 0 |
| npmpc | 12 | scaly | 0.9254 | 1.0217 | 793 | -35500 |
| npmpc | 12 | casadi_mx_gemm | 1.0105 | 0.9950 | 0 | 0 |
| npmpc | 12 | casadi_call_mx | 0.9956 | 1.0144 | 0 | 0 |
| npmpc | 25 | casadi_mx | 0.8492 | 1.0130 | 0 | 0 |
| npmpc | 25 | scaly | 0.9178 | 1.0024 | -13956 | -59704 |
| npmpc | 25 | casadi_mx_gemm | 1.0011 | 1.0096 | 0 | 0 |
| npmpc | 25 | casadi_call_mx | 1.0168 | 0.9944 | 0 | 0 |
| npmpc | 50 | casadi_mx | 0.9975 | 0.9579 | 0 | 0 |
| npmpc | 50 | scaly | 0.9982 | 1.0082 | -35300 | -148408 |
| npmpc | 50 | casadi_mx_gemm | 0.9988 | 0.9856 | 0 | 0 |
| npmpc | 50 | casadi_call_mx | 1.0007 | 1.0181 | 0 | 0 |
| npmpc | 100 | scaly | 0.9166 | 0.9502 | -84452 | -297284 |
| npmpc | 100 | casadi_mx_gemm | 1.0103 | 0.9627 | 0 | 0 |
| npmpc | 100 | casadi_call_mx | 1.0118 | 0.9908 | 0 | 0 |
| npmpc | 200 | scaly | 0.9869 | 0.9343 | -187696 | -597480 |
| npmpc | 200 | casadi_call_mx | 0.9993 | 1.0104 | 0 | 0 |
| unbumpercars | 2 | casadi_mx | 0.9869 | 1.0045 | 0 | 0 |
| unbumpercars | 2 | scaly | 1.0052 | 1.0031 | 2669 | -560 |
| unbumpercars | 2 | casadi_mx_gemm | 0.9998 | 1.0425 | 0 | 0 |
| unbumpercars | 4 | casadi_mx | 1.0081 | 1.0044 | 0 | 0 |
| unbumpercars | 4 | scaly | 0.9952 | 1.2798 | 1798 | 10416 |
| unbumpercars | 4 | casadi_mx_gemm | 1.0150 | 1.0060 | 0 | 0 |
| unbumpercars | 8 | casadi_mx | 1.0036 | 0.9823 | 0 | 0 |
| unbumpercars | 8 | scaly | 0.9386 | 0.6836 | -4188 | -50848 |
| unbumpercars | 8 | casadi_mx_gemm | 1.0010 | 0.9950 | 0 | 0 |
| unbumpercars | 16 | scaly | 0.8135 | 0.8632 | -18275 | -66760 |
| unbumpercars | 32 | scaly | 0.9918 | 0.8887 | -74655 | -235432 |
| chain | 3 | casadi_sx | 1.0022 | 0.9939 | 0 | 0 |
| chain | 3 | scaly | 1.0013 | 1.0026 | 0 | 0 |
| chain | 3 | casadi_map_sx | 0.9817 | 0.9975 | 0 | 0 |
| chain | 3 | casadi_call_mx | 0.9930 | 0.9992 | 0 | 0 |
| chain | 5 | casadi_sx | 1.0002 | 0.9944 | 0 | 0 |
| chain | 5 | scaly | 1.0033 | 1.0007 | 0 | 0 |
| chain | 5 | casadi_map_sx | 1.0098 | 1.0022 | 0 | 0 |
| chain | 5 | casadi_call_mx | 0.9992 | 0.9979 | 0 | 0 |
| chain | 9 | scaly | 0.9984 | 0.9946 | 0 | 0 |
| chain | 9 | casadi_map_sx | 1.0069 | 0.9985 | 0 | 0 |

Runtime-ratio P10/median/P90: [0.9619, 0.9997, 1.01].

## Figure and prose values

### race_cars

| Size | Best CasADi | Runtime speedup | Scaly build, s | Best CasADi build, s | Scaly object, bytes |
|---:|---|---:|---:|---:|---:|
| 5 | casadi_sx | 2.262 | 0.954 | 0.514 | 29415 |
| 10 | casadi_sx | 3.435 | 0.985 | 0.817 | 32559 |
| 25 | casadi_sx | 4.871 | 1.149 | 1.400 | 45575 |
| 50 | casadi_sx | 5.930 | 1.102 | 3.082 | 32247 |
| 100 | casadi_sx | 6.202 | 1.284 | 3.934 | 42751 |
| 200 | casadi_sx | 6.864 | 1.495 | 8.120 | 43399 |
| 400 | casadi_sx | 7.085 | 1.904 | 28.804 | 43399 |

### npmpc

| Size | Best CasADi | Runtime speedup | Scaly build, s | Best CasADi build, s | Scaly object, bytes |
|---:|---|---:|---:|---:|---:|
| 6 | casadi_mx | 1.954 | 2.191 | 2.721 | 64257 |
| 12 | casadi_mx | 1.804 | 2.814 | 5.464 | 112933 |
| 25 | casadi_mx | 1.744 | 2.828 | 8.669 | 115685 |
| 50 | casadi_mx | 2.266 | 2.900 | 14.412 | 122625 |
| 100 | casadi_mx_gemm | 5.536 | 2.951 | 22.413 | 130853 |
| 200 | casadi_call_mx | 18.155 | 2.231 | 55.407 | 97257 |

### unbumpercars

| Size | Best CasADi | Runtime speedup | Scaly build, s | Best CasADi build, s | Scaly object, bytes |
|---:|---|---:|---:|---:|---:|
| 2 | casadi_mx | 4.197 | 3.063 | 2.528 | 95231 |
| 4 | casadi_mx | 10.942 | 5.253 | 5.471 | 143033 |
| 8 | casadi_mx | 18.878 | 4.700 | 36.860 | 150741 |
| 16 |  |  | 4.862 |  | 196057 |
| 32 |  |  | 5.977 |  | 383497 |

### chain

| Size | Best CasADi | Runtime speedup | Scaly build, s | Best CasADi build, s | Scaly object, bytes |
|---:|---|---:|---:|---:|---:|
| 3 | casadi_sx | 1.179 | 4.142 | 4.891 | 92328 |
| 5 | casadi_sx | 0.821 | 16.353 | 10.718 | 323896 |
| 9 | casadi_map_sx | 8.738 | 58.005 | 60.488 | 907712 |

## Build stages at controller sizes

| Problem | Backend | Build, ms | Render, ms | Compile, ms | Render share, % | Compile share, % |
|---|---|---:|---:|---:|---:|---:|
| race_cars | casadi_mx | 1674.8 | 173.3 | 97279.0 | 0.17 | 98.14 |
| race_cars | casadi_sx | 594.3 | 45.0 | 2442.5 | 1.46 | 79.26 |
| race_cars | scaly | 215.9 | 608.4 | 277.8 | 55.20 | 25.21 |
| race_cars | casadi_map_sx | 394.0 | 14.3 | 5431.8 | 0.24 | 93.01 |
| race_cars | casadi_call_mx | 437.2 | 17.1 | 7338.2 | 0.22 | 94.17 |
| npmpc | casadi_map_sx | 937.1 | 215.3 | 173755.0 | 0.12 | 99.34 |
| npmpc | casadi_mx | 467.5 | 9.7 | 12639.4 | 0.07 | 96.36 |
| npmpc | scaly | 164.8 | 1159.4 | 1489.9 | 41.20 | 52.94 |
| npmpc | casadi_mx_gemm | 455.0 | 7.0 | 5002.3 | 0.13 | 91.55 |
| npmpc | casadi_call_mx | 529.9 | 12.9 | 10313.7 | 0.12 | 95.00 |
| unbumpercars | casadi_mx | 1015.7 | 50.6 | 52118.4 | 0.10 | 98.00 |
| unbumpercars | scaly | 701.5 | 1389.6 | 2608.6 | 29.57 | 55.51 |
| unbumpercars | casadi_mx_gemm | 2183.6 | 174.9 | 34501.6 | 0.47 | 93.60 |
| chain | casadi_sx | 3991.4 | 445.0 | 92619.3 | 0.46 | 95.43 |
| chain | scaly | 610.6 | 11873.9 | 3868.4 | 72.61 | 23.66 |
| chain | casadi_map_sx | 544.4 | 108.7 | 10064.6 | 1.01 | 93.91 |
| chain | casadi_call_mx | 741.7 | 78.2 | 78483.1 | 0.10 | 98.97 |

## Closed-loop changes since October 15

| Problem | Solver | Provider | Total-time ratio | Function-evaluation ratio |
|---|---|---|---:|---:|
| chain | ipopt | scaly | 1.0049 | 0.9985 |
| chain | sqp | casadi | 1.0064 | 1.0233 |
| chain | sqp | scaly | 1.0087 | 0.9999 |
| npmpc | ipopt | casadi | 1.0021 | 1.0032 |
| npmpc | ipopt | scaly | 0.9513 | 0.9834 |
| npmpc | sqp | casadi | 0.9938 | 0.9882 |
| npmpc | sqp | scaly | 0.9933 | 0.9921 |
| race_cars | ipopt | casadi | 1.0009 | 1.0018 |
| race_cars | ipopt | scaly | 1.0023 | 1.0156 |
| race_cars | sqp | casadi | 1.0126 | 0.9854 |
| race_cars | sqp | scaly | 1.0027 | 1.0915 |
| unbumpercars | ipopt | casadi | 0.9966 | 0.9965 |
| unbumpercars | ipopt | scaly | 0.9966 | 0.9962 |
| unbumpercars | sqp | casadi | 1.0015 | 1.0015 |
| unbumpercars | sqp | scaly | 1.0199 | 1.0225 |

## Controller build and function evaluation

- chain ipopt scaly: build 26.4459 s, function-evaluation share 5.789%.
- chain sqp casadi: build 27.8033 s, function-evaluation share 2.526%.
- chain sqp scaly: build 30.5252 s, function-evaluation share 2.842%.
- npmpc ipopt casadi: build 9.9853 s, function-evaluation share 38.848%.
- npmpc ipopt scaly: build 5.6948 s, function-evaluation share 27.166%.
- npmpc sqp casadi: build 7.8643 s, function-evaluation share 55.639%.
- npmpc sqp scaly: build 6.0353 s, function-evaluation share 36.914%.
- race_cars ipopt casadi: build 19.6922 s, function-evaluation share 8.500%.
- race_cars ipopt scaly: build 2.8155 s, function-evaluation share 3.431%.
- race_cars sqp casadi: build 9.6008 s, function-evaluation share 4.691%.
- race_cars sqp scaly: build 2.9985 s, function-evaluation share 1.740%.
- unbumpercars ipopt casadi: build 70.4043 s, function-evaluation share 96.922%.
- unbumpercars ipopt scaly: build 7.0426 s, function-evaluation share 80.537%.
- unbumpercars sqp casadi: build 61.0309 s, function-evaluation share 98.448%.
- unbumpercars sqp scaly: build 11.4618 s, function-evaluation share 88.231%.
