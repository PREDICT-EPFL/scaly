# Benchmark results

These are the results of the complete study that started on 2026-09-10 and finished on
2026-09-11. The study measured exact sparse Lagrangian Hessians and canonical closed-loop
controllers against CasADi 3.8.0. The [scalability page](scalability.md) contains every sweep cell,
and the [fairness page](fairness.md) states what each comparison holds constant.

The study completed all 690 planned sweep attempts and 75 closed-loop episodes. Of the sweep
attempts, 514 produced timings, 16 reached the 180-second compilation limit, and 160 were skipped
after the same backend had failed at a smaller size. Every timed kernel passed its correctness
check.

## Hessian scalability

Each row below uses five fresh processes. The CasADi column is the fastest encoding that completed
at that size. Runtime is the process mean. The coefficient of variation (CV) is the sample CV across
the five process means.

| Problem | Size | Scaly, µs | Scaly CV, % | Best completed CasADi | CasADi, µs | Comparison |
|---|---:|---:|---:|---|---:|---:|
| Race-car | N=40 | 19.601 | 0.94 | SX | 15.898 | Scaly 1.23× slower |
| Race-car | N=500 | 228.589 | 0.97 | SX | 209.352 | Scaly 1.09× slower |
| Neural-process MPC | N=12 | 34.737 | 0.27 | MX | 44.184 | Scaly 1.27× faster |
| Neural-process MPC | N=200 | 595.213 | 1.17 | called MX | 5037.273 | Scaly 8.46× faster |
| Unbumpercars | C=8 | 875.151 | 1.03 | MX | 10959.742 | Scaly 12.52× faster |
| Unbumpercars | C=32 | 6872.175 | 0.22 | none completed | | only Scaly timed |
| Chain | M=5 | 202.900 | 0.90 | SX | 82.632 | Scaly 2.46× slower |
| Chain | M=9 | 550.460 | 0.93 | mapped SX | 2455.255 | Scaly 4.46× faster |

Race-car Scaly remains slower than SX throughout N=1 to N=500. The gap falls from 23% at the
canonical N=40 point to 9% at N=500.

Scaly is faster than the best completed CasADi encoding at every neural-process MPC size. Its
advantage grows after plain MX reaches the compilation limit at N=100. For unbumpercars, Scaly is
4.59×, 7.38×, and 12.52× faster than MX at C=2, C=4, and C=8. Only Scaly completes at C=16 and
C=32 under the compile budget.

The chain changes winner with size. SX is faster at M=3 and M=5, but it reaches the compilation
limit at M=9. Scaly is 4.46× faster than mapped SX, the only completed CasADi encoding at M=9.

## Closed-loop controllers

The closed-loop study uses the same solver implementation within each provider pair. The sequential
quadratic programming (SQP) pairs use Scaly SQP and PIQP. The IPOPT pairs use the same IPOPT 3.14.19
library. Both sides supply compiled C oracles. Times are native solver means per control step,
including the first step, across five fresh processes. Function evaluation includes all oracle work
requested by the solver.

| Problem | Solver | Provider | Total, ms | Total CV, % | Function evaluation, ms | QP, ms | Globalization, ms | Glue, ms | Steps |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| chain | IPOPT | Scaly | 7.158 | 1.42 | 0.686 | 6.189 | 0.000 | 0.283 | 90 |
| chain | SQP | Scaly | 6.410 | 0.92 | 0.282 | 5.399 | 0.008 | 0.721 | 90 |
| chain | SQP | CasADi | 14.377 | 0.38 | 8.118 | 5.474 | 0.008 | 0.776 | 90 |
| npmpc | IPOPT | Scaly | 3.557 | 0.17 | 1.006 | 2.305 | 0.000 | 0.246 | 100 |
| npmpc | IPOPT | CasADi | 4.481 | 0.53 | 1.742 | 2.559 | 0.000 | 0.179 | 100 |
| npmpc | SQP | Scaly | 1.295 | 0.48 | 0.503 | 0.670 | 0.004 | 0.118 | 100 |
| npmpc | SQP | CasADi | 1.864 | 1.66 | 1.044 | 0.694 | 0.004 | 0.122 | 100 |
| race_cars | IPOPT | Scaly | 4.934 | 0.49 | 0.484 | 4.165 | 0.000 | 0.285 | 1367 |
| race_cars | IPOPT | CasADi | 4.936 | 0.75 | 0.376 | 4.354 | 0.000 | 0.206 | 1367 |
| race_cars | SQP | Scaly | 2.551 | 0.25 | 0.134 | 2.327 | 0.007 | 0.083 | 1367 |
| race_cars | SQP | CasADi | 2.661 | 0.62 | 0.218 | 2.349 | 0.008 | 0.085 | 1367 |
| unbumpercars | IPOPT | Scaly | 20.375 | 1.15 | 16.419 | 0.000 | 0.000 | 0.260 | 200 |
| unbumpercars | IPOPT | CasADi | 146.324 | 0.55 | 141.860 | 0.000 | 0.000 | 0.242 | 200 |
| unbumpercars | SQP | Scaly | 10.786 | 0.55 | 9.513 | 1.232 | 0.010 | 0.031 | 200 |
| unbumpercars | SQP | CasADi | 106.186 | 0.43 | 104.822 | 1.310 | 0.011 | 0.043 | 200 |

Scaly reduces total SQP time by 4% for race-car, 31% for neural-process MPC, 55% for chain, and
90% for unbumpercars. With IPOPT, the race-car totals are equal within measurement dispersion.
Scaly reduces the neural-process MPC total by 21% and the unbumpercars total by 86%. Chain does not
provide an IPOPT/CasADi runner.

## Episode agreement

The comparisons below use every recorded control step. Mismatch counts sum across all five episode
pairs.

| Problem | Solver | Repetitions | Per-step iteration mismatches | Per-step oracle-count mismatches | Maximum state difference | Maximum control difference |
|---|---|---:|---:|---:|---:|---:|
| chain | SQP | 5 | 0 | 0 | 3.48e-09 | 1.29e-08 |
| npmpc | IPOPT | 5 | 0 | 0 | 4.66e-13 | 1.76e-15 |
| npmpc | SQP | 5 | 0 | 0 | 7.95e-10 | 3.93e-12 |
| race_cars | IPOPT | 5 | 0 | 0 | 1.35e-13 | 5.02e-12 |
| race_cars | SQP | 5 | 0 | 0 | 4.68e-12 | 3.39e-09 |
| unbumpercars | IPOPT | 5 | 0 | 0 | 2.99e-12 | 2.95e-12 |
| unbumpercars | SQP | 5 | 10 | 0 | 2.62e-07 | 2.02e-07 |

Every IPOPT pair matches per-step iterations and oracle-call counts. The chain, neural-process MPC,
and race-car SQP pairs also match. Unbumpercars has ten SQP iteration mismatches across 1,000 paired
steps, so its SQP total is an observed closed-loop result rather than a strict equal-work
comparison.

## Reproduce the study

```bash
uv run benchmarks/run.py study --out-dir benchmarks/results/<study-name>
uv run benchmarks/run.py report benchmarks/results/<study-name>
```

The study command runs the frozen grids with five fresh processes, order seed 0, the
`performance` CPU governor, boost disabled, a 180-second compile limit, and a 50 MiB source limit.
The report command regenerates the Markdown tables from the saved artifacts.
