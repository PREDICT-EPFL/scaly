# A11 (Tier 9, C-245): the stagewise backend against the sparse one and PIQP's multistage

2 Oct 2026, Apple M3 Max, Apple clang 21, the JIT's flags; load average 2.6 to 4 from other
sessions. `mpc_<nx>_<nu>_<N>` is `scaly.testing.qp.mpc_qp`. Generated solvers:
`perf_2026_09_27_ipm_speed/gen.py --variant t9a11 --backends sparse,stagewise --split`, then
`timing.py --variants t9a11 --rounds 5`: the fastest solve from C. PIQP:
`m1_stagewise.py piqp`, its own solve timer, the fastest of 50, variables in stage order.

| problem | block | iterations | sparse (us) | stagewise (us) | stagewise / sparse | PIQP multistage (us) | stagewise / PIQP multistage | sparse / PIQP multistage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mpc_4_2_10 | 6 | 8 | 47.7 | 56.5 | 1.18 | 60.0 | 0.94 | 0.80 |
| mpc_8_2_20 | 10 | 8 | 282.5 | 250.8 | 0.89 | 209.4 | 1.20 | 1.35 |
| mpc_6_2_40 | 8 | 8 | 380.2 | 340.2 | 0.89 | 333.2 | 1.02 | 1.14 |
| mpc_12_4_20 | 16 | 7 | 535.4 | 477.9 | 0.89 | 333.7 | 1.43 | 1.60 |
| mpc_16_4_20 | 20 | 7 | 896.5 | 772.6 | 0.86 | 473.5 | 1.63 | 1.89 |
| mpc_26_2_25 | 28 | 8 | 3324.6 | 2361.1 | 0.71 | 1316.9 | 1.79 | 2.52 |
| mpc_27_6_30 | 33 | 8 | 4702.6 | 3551.1 | 0.76 | 2002.7 | 1.77 | 2.35 |
| mpc_39_3_25 | 42 | 7 | 7753.7 | 5073.2 | 0.65 | 2455.2 | 2.07 | 3.16 |

Every backend takes the same iterations on every problem, PIQP's.

## How it got there

The same six problems as each piece went in (three rounds, heavier load; ratios to the sparse
backend of that run).

| step | 4_2_10 | 6_2_40 | 12_4_20 | 26_2_25 | 27_6_30 | 39_3_25 |
| --- | --- | --- | --- | --- | --- | --- |
| the condensed matrix formed as a sparse one, then gathered into blocks | 1.15 | 1.24 | 1.58 | 1.66 | 2.06 | 1.93 |
| the stage rows' products as dense arrays, one a block | 1.16 | 1.08 | 1.15 | 0.97 | 1.10 | 0.92 |
| every term scattered straight into the block storage | 1.17 | 1.00 | 1.03 | 0.83 | 0.97 | 0.83 |
| rows with few entries in the dense columns kept out of the arrays | 1.19 | 0.97 | 0.98 | 0.76 | 0.86 | 0.71 |
| products with a vector through the arrays | 1.18 | 0.89 | 0.89 | 0.68 | 0.79 | 0.68 |

The first row is M1's prototype with its assembly counted: forming `A^T A` through index tables
costs as many multiply-adds as the factorization, at a tenth of the rate, and was 65% of a solve.

## Where a solve goes (`prof.py`, share of samples)

| problem | us an iteration | block factorization | assembly | the arrays' products | step | solve | solves' passes over the blocks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mpc_26_2_25 | 285 | 28.9% | 20.1% | 8.4% + 4.5% | 14.2% | 8.5% | 9.0% |
| mpc_39_3_25 | 725 | 37.0% | 15.7% | 13.2% + 3.4% | 10.6% | 8.0% | 5.9% |

## The stage step's three kernels (25 blocks in a `scan`, us; G multiply-adds a second)

| block, coupling | Cholesky | the block below | the update | Cholesky | the block below | the update |
| --- | --- | --- | --- | --- | --- | --- |
| 16, 12 | 7.4 | 9.5 | 2.2 | 2.3 | 3.0 | (too short to time) |
| 28, 26 | 25.1 | 36.9 | 26.8 | 3.6 | 6.4 | 19.0 |
| 33, 27 | 36.5 | 41.6 | 39.3 | 4.1 | 7.2 | 18.7 |
| 42, 39 | 65.1 | 101.9 | 87.0 | 4.7 | 7.8 | 19.8 |

The update is a full product at the product kernel's rate, where half of it is wanted. The
Cholesky and the triangular solve run at a quarter to a third of that rate: at these orders the
dot products are a few terms long, and the divisions and the square roots down the diagonal are a
chain the tiles do not shorten. That is K4.
