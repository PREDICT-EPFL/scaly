# M1 (Tier 9): the generated IPM on stage-banded QPs, before a stage-structured backend

2 Oct 2026, Apple M3 Max, Apple clang 21, the JIT's flags; two review agents were running (load
average 5 to 8), so read ratios. `mpc_<nx>_<nu>_<N>` is `scaly.testing.qp.mpc_qp`: dense dynamics,
diagonal cost, boxes on every state and input. 26 and 39 states over 25 stages are the sizes of
E1's hanging chain in 2D and 3D, where Fatrop's linear algebra took 0.18 and 0.44 ms an iteration.

## The generated IPM today, against PIQP's dense and sparse backends

`perf_2026_09_27_ipm_speed/gen.py --variant m1 --split`, then `timing.py --variants m1 --rounds 3 --piqp`.

| problem | backend | n / p / m | iter | generated (us) | PIQP solve (us) | PIQP iter | generated / PIQP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mpc_4_2_10 | dense | 64 / 44 / 0 | 8 | 124.8 | 143.5 | 8 | 0.87 |
| mpc_6_2_40 | dense | 326 / 246 / 0 | 8 | 4126.8 | 4654.9 | 8 | 0.89 |
| mpc_12_4_20 | dense | 332 / 252 / 0 | 7 | 4153.7 | 4163.5 | 7 | 1.00 |
| mpc_26_2_25 | dense | 726 / 676 / 0 | 8 | 38379.1 | 40286.5 | 8 | 0.95 |
| mpc_27_6_30 | dense | 1017 / 837 / 0 | 8 | 92082.2 | 102537.0 | 8 | 0.90 |
| mpc_39_3_25 | dense | 1089 / 1014 / 0 | 7 | 106428.4 | 113092.2 | 7 | 0.94 |
| mpc_4_2_10 | sparse | 64 / 44 / 0 | 8 | 46.6 | 67.0 | 8 | 0.70 |
| mpc_6_2_40 | sparse | 326 / 246 / 0 | 8 | 363.4 | 499.9 | 8 | 0.73 |
| mpc_12_4_20 | sparse | 332 / 252 / 0 | 7 | 511.8 | 783.2 | 7 | 0.65 |
| mpc_26_2_25 | sparse | 726 / 676 / 0 | 8 | 3219.6 | 5259.4 | 8 | 0.61 |
| mpc_27_6_30 | sparse | 1017 / 837 / 0 | 8 | 4447.8 | 7462.5 | 8 | 0.60 |
| mpc_39_3_25 | sparse | 1089 / 1014 / 0 | 7 | 7377.4 | 12362.5 | 7 | 0.60 |

## PIQP with each of its KKT solvers (`m1_stagewise.py piqp`), variables in stage order

| problem | n / p | dense (us) | sparse_ldlt (us) | sparse_ldlt_cond (us) | multistage (us) | iterations | multistage, us an iteration |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mpc_4_2_10 | 64 / 44 | 143.1 | 64.8 | 66.7 | 62.4 | 8/8/8/8 | 7.8 |
| mpc_6_2_40 | 326 / 246 | 4441.2 | 507.0 | 462.5 | 330.2 | 8/8/8/8 | 41.3 |
| mpc_12_4_20 | 332 / 252 | 4186.7 | 815.2 | 789.2 | 329.3 | 7/7/7/7 | 47.0 |
| mpc_26_2_25 | 726 / 676 | 39045.3 | 5278.2 | 4323.2 | 1316.7 | 8/8/8/8 | 164.6 |
| mpc_27_6_30 | 1017 / 837 | 101173.4 | 7200.2 | 6840.5 | 1905.5 | 8/8/8/8 | 238.2 |
| mpc_39_3_25 | 1089 / 1014 | 111835.5 | 12205.0 | 10363.8 | 2437.9 | 7/7/7/7 | 348.3 |

The generated sparse backend over PIQP's multistage backend: 0.75, 1.10, 1.55, 2.45, 2.33, 3.03.
PIQP's multistage backend is at Fatrop's time an iteration at the chain's sizes (165 and 348 us
against 180 and 440), so it stands in for the Riccati solvers as the baseline.

## Multiply-adds of one factorization (`m1_stagewise.py counts`)

| problem | KKT order | nnz(L), sparse LDL' | its multiply-adds | dense Cholesky of the condensed | blocks, dense | blocks, zeros skipped |
| --- | --- | --- | --- | --- | --- | --- |
| mpc_4_2_10 | 108 | 452 | 1284 | 43690 | 2520 | 1560 |
| mpc_6_2_40 | 572 | 3934 | 16528 | 5774329 | 23880 | 16840 |
| mpc_12_4_20 | 584 | 7908 | 63236 | 6099061 | 95560 | 67400 |
| mpc_26_2_25 | 1402 | 42329 | 730783 | 63776196 | 640250 | 582850 |
| mpc_27_6_30 | 1854 | 58281 | 1035989 | 175311985 | 1257750 | 981540 |
| mpc_39_3_25 | 2103 | 95436 | 2455873 | 215244661 | 2160900 | 1967175 |

The scalar sparse LDL' of the whole KKT matrix does 0.8 to 1.25 of the block recursion's
multiply-adds. The ordering is already the recursion's: what differs is the rate they run at.

## A prototype of the block recursion on the generated kernels (`m1_stagewise.py blocks`)

Timed through the Python call (2 to 3 us of it in every figure).

| problem | block | dense Cholesky (us) | sparse LDL' of the KKT (us) | blocks, dense (us) | blocks, x-part (us) | its solve (us) | G multiply-adds a second, x-part | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mpc_4_2_10 | 6 | 9.1 | 3.0 | 3.4 | 3.4 | 3.1 | 0.5 | 2.4e-13 |
| mpc_6_2_40 | 8 | 344.2 | 13.7 | 8.7 | 7.6 | 6.5 | 2.2 | 2.8e-13 |
| mpc_12_4_20 | 16 | 367.3 | 29.2 | 20.0 | 16.4 | 8.3 | 4.1 | 4.4e-13 |
| mpc_26_2_25 | 28 | 3135.7 | 230.5 | 87.0 | 83.9 | 24.4 | 6.9 | 3.4e-13 |
| mpc_27_6_30 | 33 | 9327.8 | 321.5 | 168.5 | 148.1 | 38.7 | 6.6 | 1.1e-12 |
| mpc_39_3_25 | 42 | 10190.8 | 666.7 | 247.7 | 251.0 | 60.8 | 7.8 | 8.3e-13 |

The sparse LDL' runs at 3.2 to 3.7 G multiply-adds a second on the two chain sizes; the block
recursion, 0.36 to 0.56 of its time from blocks of 16, at 7 to 8 by the count of the last column.
The prototype's update is a full product where a symmetric one does half the work, and its
Cholesky and triangular solve are separate calls, so by the work it does it runs at about 11.

## Where a solve goes today (`perf_2026_09_27_ipm_speed/prof.py`, sparse backend)

| problem | us an iteration | factor | solve | step | other |
| --- | --- | --- | --- | --- | --- |
| mpc_26_2_25 | 429 | 64.8% | 17.5% | 13.7% | 4% |
| mpc_6_2_40 | 48 | 34.1% | 31.1% | 28.6% | 6% |

## What it predicts

At 26 states: 278 us of factorization become about 84, plus the blocks' assembly; three solves at
24 us against 75 today; the step's 59 stay. About 250 us an iteration against 429, 1.5 times PIQP's
multistage backend where it is 2.45 today. At 39 states about 600 against 1054, 1.7 times. Parity
at these sizes needs the factorization at the kernels' rate (K4): the diagonal block and the block
below factored as one panel, and a symmetric update. At blocks of 8 the two factorizations are
within 2x and the solve and the step are two thirds of an iteration, so the model has to choose.
