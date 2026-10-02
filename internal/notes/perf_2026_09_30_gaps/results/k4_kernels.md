# K4 (Tier 9, C-246): the stage step's kernels

2 Oct 2026, Apple M3 Max, Apple clang 21, the JIT's flags, other sessions running: ratios taken
interleaved. Kernel times are of 25 (or 50) calls in a `scan`, so the Python call does not count.

## A Cholesky with running sums and the diagonal's reciprocals, against the pairwise form

| order | pairwise (us) | G multiply-adds a second | running (us) | G multiply-adds a second | running / pairwise |
| --- | --- | --- | --- | --- | --- |
| 24 | 0.58 | 4.0 | 0.49 | 4.7 | 0.85 |
| 28 | 0.78 | 4.7 | 0.64 | 5.7 | 0.82 |
| 32 | 1.07 | 5.1 | 0.84 | 6.5 | 0.78 |
| 40 | 1.77 | 6.0 | 1.29 | 8.3 | 0.73 |
| 48 | 2.74 | 6.7 | 1.90 | 9.7 | 0.69 |
| 64 | 5.26 | 8.3 | 3.68 | 11.9 | 0.70 |

Running sums alone, still dividing each entry, gave 0.79 to 0.91 of the pairwise form's time from
order 24 and nothing under it: the quarters were not what held a small factorization back. A
count of a 28-row factorization's cycles: about 450 in the tiles' multiply-adds, 320 in starting
and storing the tiles, 100 copying a block's rows, 330 solving the rows under each diagonal block,
and 840 in the chain down the diagonal, a square root and a division a column. The reciprocals
take the divisions off every entry (about 400 at order 28) and leave one a column.

## The stage step's three kernels, 25 blocks in a scan (us)

| block, coupling | Cholesky | + the block below | + the update | all three, pairwise and full | ratio |
| --- | --- | --- | --- | --- | --- |
| 16, 12 | 7.4 | 12.4 | 16.7 | 16.8 | 0.99 |
| 28, 26 | 21.4 | 44.1 | 66.8 | 80.8 | 0.83 |
| 33, 27 | 31.9 | 65.6 | 96.4 | 119.0 | 0.81 |
| 42, 39 | 51.7 | 132.0 | 183.6 | 237.8 | 0.77 |

## A matrix times its own transpose, one triangle computed (us a product, 50 in a scan)

`a` is rows x columns; the symmetric lowering against the same Function with it off. Every result
is the same to the last bit on apple-m3, x86-64-v3 and generic.

| a | a a' | full | ratio | a' a | full | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 28 x 26 | 1.09 | 1.15 | 0.94 | 1.00 | 1.13 | 0.89 |
| 28 x 28 | 1.17 | 1.33 | 0.88 | 1.11 | 1.33 | 0.84 |
| 33 x 27 | 1.46 | 1.74 | 0.84 | 1.42 | 1.54 | 0.92 |
| 42 x 39 | 3.03 | 3.97 | 0.76 | 3.33 | 3.77 | 0.88 |
| 64 x 64 | 9.02 | 13.57 | 0.66 | 9.54 | 13.95 | 0.68 |
| 65 x 30 | 5.05 | 6.88 | 0.73 | 3.41 | 5.23 | 0.65 |
| 130 x 40 | 22.91 | 34.14 | 0.67 | 8.69 | 11.88 | 0.73 |

With two tiles of columns (16 and 20 rows) the tiles skipped were less work than the copy across
the diagonal, 1.03 to 1.07 of the full product's time: it is taken from three. A product expanded
into scalar code rounds by which terms its entries share, so its bits changed with the copy (4 x
5, 7 x 3, 9 x 4): it is taken only where the product is certain to stay in loops.

## The stagewise solver, with K4 against without (`timing.py --variants t9a11,t9k4 --rounds 5`)

| problem | block | iterations | A11 (us) | with K4 (us) | ratio |
| --- | --- | --- | --- | --- | --- |
| mpc_4_2_10 | 6 | 8 | 55.0 | 55.4 | 1.008 |
| mpc_8_2_20 | 10 | 8 | 244.8 | 240.2 | 0.981 |
| mpc_6_2_40 | 8 | 8 | 335.4 | 336.2 | 1.002 |
| mpc_12_4_20 | 16 | 7 | 468.8 | 469.2 | 1.001 |
| mpc_16_4_20 | 20 | 7 | 760.2 | 741.3 | 0.975 |
| mpc_26_2_25 | 28 | 8 | 2353.1 | 2229.4 | 0.947 |
| mpc_27_6_30 | 33 | 8 | 3647.4 | 3358.0 | 0.921 |
| mpc_39_3_25 | 42 | 7 | 4960.2 | 4353.0 | 0.878 |

Against the sparse backend that is 0.67, 0.71 and 0.56 at blocks of 28, 33 and 42, and against
PIQP's multistage backend 1.69, 1.68 and 1.77. After K4 a solve at 42 is 34% block factorization,
19% assembly, 12% the arrays' products, 11% the step and 15% the solves.
