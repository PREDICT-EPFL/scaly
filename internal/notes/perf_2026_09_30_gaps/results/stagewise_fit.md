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

## Refitted after the Tier 9 review

2 Oct 2026, the same machine, another session's benchmark suite running beside the timing: ratios
taken interleaved, seven rounds. The review proved the weights above wrong on a family they were not
fitted to: on multistage problems with inequality rows over each stage (`mpc_<nx>_<nu>_<N>_<r>`,
`scaly.testing.qp.mpc_qp(..., path_rows=r)`) and blocks of 5 or 6 slots, the default took the
stagewise backend where it is 1.17-1.33x the sparse backend's time, and on Hessians of dense blocks
coupled with their neighbours (`btri_<K>_<B>`) it put the stagewise backend at a third of its time.
`gen.py --variant t9rev --split` on 57 problems: the 24 above, 23 with path rows, 10 of coupled
blocks; the code is the review's (the symmetric product from sixteen terms).

```
stagewise weights: constant 0.000e+00, vectors 3.321e-02, entries 4.760e-03, cells 1.730e-04, pairs 1.414e-03, arrays 9.192e-04, products 4.000e-05, factor 7.000e-05, solve 0.000e+00
predicted/measured an iteration: median 0.97, range 0.77-1.50
leave-one-out: median 0.97, range 0.75-1.61
```

| problem | sparse | stagewise | dense | model's stagewise | pick | pick / fastest |
| --- | --- | --- | --- | --- | --- | --- |
| btri_10_4 | 2.1 | 2.4 | 5.6 | 2.6 | sparse | 1.00 |
| mpc_3_3_5 | 1.6 | 2.1 | 4.4 | 3.3 | sparse | 1.00 |
| mpc_2_1_10 | 1.7 | 1.9 | 3.8 | 3.1 | sparse | 1.00 |
| mpc_6_2_5 | 5.1 | 6.1 | 10.1 | 6.6 | sparse | 1.00 |
| btri_10_10 | 12.7 | 12.0 | 35.4 | 11.6 | stagewise | 1.00 |
| mpc_4_2_10 | 5.8 | 6.8 | 15.7 | 7.1 | sparse | 1.00 |
| btri_20_6 | 11.6 | 11.1 | 48.4 | 9.9 | sparse | 1.04 |
| btri_30_4 | 9.3 | 9.8 | 46.5 | 7.9 | sparse | 1.00 |
| mpc_2_8_10 | 4.3 | 10.5 | 36.0 | 8.2 | sparse | 1.00 |
| mpc_4_2_10_4 | 9.2 | 11.7 | 18.8 | 10.7 | sparse | 1.00 |
| btri_60_3 | 13.5 | 13.7 | 115.2 | 10.3 | sparse | 1.00 |
| mpc_2_1_40 | 11.5 | 12.4 | 50.6 | 11.1 | sparse | 1.00 |
| mpc_8_4_10 | 18.4 | 17.9 | 67.7 | 17.4 | sparse | 1.03 |
| btri_20_12 | 37.2 | 30.3 | 232.9 | 33.8 | stagewise | 1.00 |
| btri_30_8 | 25.9 | 24.8 | 228.5 | 24.3 | stagewise | 1.00 |
| btri_40_6 | 22.4 | 22.0 | 223.9 | 20.0 | sparse | 1.02 |
| mpc_8_4_10_4 | 24.8 | 22.1 | 74.8 | 21.8 | stagewise | 1.00 |
| mpc_4_1_20_4 | 16.6 | 20.2 | 43.9 | 17.9 | sparse | 1.00 |
| mpc_2_1_40_2 | 14.8 | 18.5 | 55.0 | 15.4 | sparse | 1.00 |
| mpc_4_2_20_4 | 19.3 | 24.0 | 60.6 | 21.0 | sparse | 1.00 |
| mpc_6_2_20 | 22.7 | 20.4 | 109.5 | 19.7 | sparse | 1.11 |
| btri_10_30 | 111.9 | 91.7 | 447.4 | 105.2 | stagewise | 1.00 |
| mpc_3_3_20_6 | 20.3 | 24.5 | 59.4 | 22.3 | sparse | 1.00 |
| mpc_5_5_20 | 21.9 | 25.3 | 179.2 | 20.8 | sparse | 1.00 |
| mpc_5_1_20_5 | 24.0 | 27.6 | 64.3 | 25.1 | sparse | 1.00 |
| mpc_4_2_20_8 | 25.2 | 31.4 | 65.8 | 28.1 | sparse | 1.00 |
| mpc_8_2_20 | 33.6 | 29.8 | 179.8 | 29.0 | stagewise | 1.00 |
| btri_20_20 | 96.5 | 85.1 | 892.8 | 91.1 | stagewise | 1.00 |
| mpc_4_2_40 | 26.0 | 33.1 | 241.1 | 27.2 | sparse | 1.00 |
| mpc_5_5_20_5 | 34.5 | 34.6 | 186.4 | 30.4 | stagewise | 1.00 |
| mpc_6_2_20_6 | 33.0 | 28.6 | 121.0 | 29.6 | stagewise | 1.00 |
| mpc_8_2_20_2 | 39.9 | 36.0 | 187.3 | 35.4 | stagewise | 1.00 |
| mpc_2_8_20_10 | 36.1 | 36.6 | 179.1 | 31.4 | stagewise | 1.01 |
| mpc_4_2_40_4 | 39.4 | 48.8 | 259.8 | 41.5 | sparse | 1.00 |
| mpc_6_2_40 | 45.4 | 40.5 | 516.0 | 38.8 | sparse | 1.12 |
| mpc_12_4_20 | 72.9 | 65.2 | 593.6 | 60.8 | stagewise | 1.00 |
| mpc_10_2_30 | 75.2 | 61.8 | 769.8 | 59.9 | stagewise | 1.00 |
| mpc_12_4_20_8 | 102.2 | 82.5 | 611.0 | 82.7 | stagewise | 1.00 |
| mpc_10_2_30_4 | 87.6 | 73.5 | 753.1 | 72.9 | stagewise | 1.00 |
| mpc_16_8_10 | 66.2 | 64.9 | nan | 60.1 | stagewise | 1.00 |
| mpc_16_4_20 | 122.1 | 102.7 | nan | 96.9 | stagewise | 1.00 |
| mpc_16_4_20_8 | 156.5 | 124.2 | nan | 123.4 | stagewise | 1.00 |
| mpc_20_4_20 | 187.8 | 146.0 | nan | 143.6 | stagewise | 1.00 |
| mpc_20_4_20_4 | 216.7 | 163.0 | nan | 159.1 | stagewise | 1.00 |
| mpc_6_2_40_12 | 89.0 | 70.3 | nan | 79.6 | stagewise | 1.00 |
| mpc_48_4_10 | 661.2 | 378.5 | nan | 443.7 | stagewise | 1.00 |
| mpc_12_4_40 | 147.9 | 130.1 | nan | 120.3 | stagewise | 1.00 |
| mpc_32_8_15 | 409.8 | 289.2 | nan | 310.8 | stagewise | 1.00 |
| mpc_12_4_40_6 | 203.1 | 156.7 | nan | 153.0 | stagewise | 1.00 |
| mpc_26_2_25 | 402.7 | 270.5 | nan | 270.7 | stagewise | 1.00 |
| mpc_26_2_25_4 | 375.9 | 292.9 | nan | 293.3 | stagewise | 1.00 |
| mpc_16_4_20_40 | 269.2 | 199.5 | nan | 236.5 | stagewise | 1.00 |
| mpc_20_2_40 | 367.8 | 272.2 | nan | 258.0 | stagewise | 1.00 |
| mpc_27_6_30 | 556.2 | 404.3 | nan | 409.2 | stagewise | 1.00 |
| mpc_27_6_30_6 | 655.2 | 453.7 | nan | 457.1 | stagewise | 1.00 |
| mpc_39_3_25 | 1055.1 | 604.2 | nan | 662.3 | stagewise | 1.00 |
| mpc_39_3_25_4 | 1135.6 | 650.2 | nan | 695.9 | stagewise | 1.00 |
the model (stagewise weights left one out) picks the fastest measured on 50/57; its worst pick is 1.12x the fastest

Every pick the model gets wrong is within 1.12x of the fastest, and five of the six keep the sparse
backend where the stagewise one is the faster. On the review's own timings (`rev9a_small`,
`rev9a_choice`) the eleven small-stage problems with path rows now take the sparse backend, and every
problem that takes the stagewise one is 0.69-0.97 of the sparse backend's time (`mpcpath_2_8_20_10`,
at 1.02, takes the sparse one). The 55 problems of
the IPM speed study take the same backends as before (the stagewise one for `mpc_12_4_20`,
`mpc_27_6_30` and `ex_mpc_N20`), and the race cars' generated SQP, 42 blocks of 6 slots, the sparse
one under both fits.
