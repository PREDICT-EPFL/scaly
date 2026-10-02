# The stagewise backend's weights, and the three-way choice (C-247)

2 Oct 2026, Apple M3 Max, other sessions running. Microseconds an iteration, the fastest solve from C
over its iterations; `gen.py --variant t9fit --split` on 24 `mpc_<nx>_<nu>_<N>` problems, the dense
backend where it is worth timing; `stagewise_fit.py`. The pick is the model's with the stagewise
weights fitted without that problem.

```
stagewise weights: constant 0.000e+00, vectors 2.920e-02, entries 0.000e+00, cells 2.989e-03, pairs 9.718e-04, arrays 1.359e-03, products 4.000e-05, factor 7.000e-05, solve 0.000e+00
predicted/measured an iteration: median 0.94, range 0.70-1.36
leave-one-out: median 0.94, range 0.66-1.58
```

| problem | sparse | stagewise | dense | model's stagewise | pick | pick / fastest |
| --- | --- | --- | --- | --- | --- | --- |
| mpc_3_3_5 | 1.5 | 2.1 | 4.5 | 3.1 | sparse | 1.00 |
| mpc_2_1_10 | 1.6 | 1.8 | 3.8 | 2.9 | sparse | 1.00 |
| mpc_6_2_5 | 4.8 | 6.3 | 10.6 | 5.3 | sparse | 1.00 |
| mpc_4_2_10 | 5.9 | 7.0 | 16.5 | 6.2 | sparse | 1.00 |
| mpc_2_8_10 | 4.4 | 10.9 | 37.4 | 7.9 | sparse | 1.00 |
| mpc_2_1_40 | 11.1 | 12.8 | 50.8 | 8.4 | sparse | 1.00 |
| mpc_8_4_10 | 18.8 | 18.7 | 69.2 | 18.1 | sparse | 1.01 |
| mpc_6_2_20 | 21.3 | 21.1 | 109.4 | 18.7 | sparse | 1.01 |
| mpc_5_5_20 | 22.1 | 26.3 | 187.4 | 21.2 | sparse | 1.00 |
| mpc_8_2_20 | 34.6 | 30.2 | 188.7 | 28.3 | stagewise | 1.00 |
| mpc_4_2_40 | 25.5 | 34.6 | 252.6 | 23.0 | sparse | 1.00 |
| mpc_6_2_40 | 45.5 | 40.4 | 514.7 | 37.0 | sparse | 1.13 |
| mpc_12_4_20 | 76.3 | 68.5 | 591.1 | 63.2 | stagewise | 1.00 |
| mpc_10_2_30 | 77.5 | 63.3 | 808.4 | 59.2 | stagewise | 1.00 |
| mpc_16_8_10 | 65.5 | 67.9 | nan | 67.0 | stagewise | 1.04 |
| mpc_16_4_20 | 127.6 | 96.6 | nan | 102.2 | stagewise | 1.00 |
| mpc_20_4_20 | 196.7 | 137.8 | nan | 152.4 | stagewise | 1.00 |
| mpc_48_4_10 | 685.4 | 396.8 | nan | 476.1 | stagewise | 1.00 |
| mpc_12_4_40 | 154.7 | 136.2 | nan | 124.5 | stagewise | 1.00 |
| mpc_32_8_15 | 420.0 | 302.6 | nan | 338.0 | stagewise | 1.00 |
| mpc_26_2_25 | 403.2 | 282.6 | nan | 280.2 | stagewise | 1.00 |
| mpc_20_2_40 | 385.2 | 285.0 | nan | 263.1 | stagewise | 1.00 |
| mpc_27_6_30 | 557.9 | 426.5 | nan | 436.6 | stagewise | 1.00 |
| mpc_39_3_25 | 1075.6 | 635.1 | nan | 697.2 | stagewise | 1.00 |
the model (stagewise weights left one out) picks the fastest measured on 20/24; its worst pick is 1.13x the fastest
