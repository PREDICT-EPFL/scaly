# Benchmark results

The complete study on 2026-09-23 measured exact sparse Lagrangian Hessians and canonical closed-loop
controllers against CasADi 3.8.0. Race-car N=200 takes 12.502 µs, 6.50× faster than the best
completed CasADi encoding.

The study completed all 690 planned sweep attempts and 75 closed-loop episodes. Of the sweep
attempts, 520 produced timings, 15 reached the 180-second compilation limit, and 155 were skipped
after the same backend had failed at a smaller size. Every timed kernel passed its correctness
check. All Scaly cells completed five processes.

Every current result below comes from one libmvec study. Both providers receive the supported
compiler flags and link libraries under the [fairness policy](fairness.md). The sweep uses
Clang 20.1.8, and the closed loops use GCC 13.3.0. The [scalability page](scalability.md) contains
every sweep cell, including failures, compile times, executable sizes, and workspace requirements.

## Hessian scalability

Each row below uses five fresh processes. The CasADi column is the fastest encoding that completed
at that size. Runtime is the process mean. The coefficient of variation (CV) is the sample CV across
the five process means.

| Problem | Size | Scaly, µs | Scaly CV, % | Best completed CasADi | CasADi, µs | Comparison |
|---|---:|---:|---:|---|---:|---:|
| Race-car | N=40 | 3.021 | 3.11 | SX | 15.525 | Scaly 5.14× faster |
| Race-car | N=200 | 12.502 | 1.69 | SX | 81.206 | Scaly 6.50× faster |
| Race-car | N=500 | 32.178 | 10.73 | SX | 206.559 | Scaly 6.42× faster |
| Neural-process MPC | N=12 | 22.786 | 4.97 | MX | 40.525 | Scaly 1.78× faster |
| Neural-process MPC | N=200 | 235.705 | 1.20 | called MX | 4977.875 | Scaly 21.12× faster |
| Unbumpercars | C=8 | 484.377 | 0.60 | MX | 10747.810 | Scaly 22.19× faster |
| Unbumpercars | C=32 | 6245.988 | 0.51 | none completed | | only Scaly timed |
| Chain | M=5 | 100.037 | 0.36 | SX | 82.822 | Scaly 1.21× slower |
| Chain | M=9 | 282.649 | 0.06 | mapped SX | 2473.311 | Scaly 8.75× faster |

Scaly is faster than the best completed CasADi encoding at every race-car, neural-process model
predictive control (MPC), and unbumpercars size with a completed comparison. Only Scaly completes
unbumpercars C=16 and C=32 under the compile budget. Race-car N=500 has a 10.73% CV, so its mean is
less stable than the N=200 result.

The chain changes winner with size. Scaly is 1.19× faster than SX at M=3, while SX remains faster at
M=5. At M=9, SX reaches the compilation limit and Scaly is 8.75× faster than mapped SX.

## Closed-loop controllers

The closed-loop study uses the same solver implementation within each provider pair. The sequential
quadratic programming (SQP) pairs use Scaly SQP and PIQP. The IPOPT pairs use the same IPOPT 3.14.19
library. Both sides supply compiled C oracles. Times are native solver means per control step,
including the first step, across five fresh processes. Function evaluation includes all oracle work
requested by the solver. QP denotes the quadratic-program solve.

| Problem | Solver | Provider | Total, ms | Total CV, % | Function evaluation, ms | QP, ms | Globalization, ms | Glue, ms | Steps | Processes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chain | ipopt | scaly | 6.776 | 0.21 | 0.326 | 6.170 | 0.000 | 0.281 | 90 | 5 |
| chain | sqp | casadi | 14.425 | 0.28 | 8.159 | 5.487 | 0.008 | 0.771 | 90 | 5 |
| chain | sqp | scaly | 6.267 | 0.54 | 0.153 | 5.368 | 0.008 | 0.738 | 90 | 5 |
| npmpc | ipopt | casadi | 4.482 | 0.45 | 1.737 | 2.564 | 0.000 | 0.181 | 100 | 5 |
| npmpc | ipopt | scaly | 3.609 | 2.01 | 0.966 | 2.386 | 0.000 | 0.257 | 100 | 5 |
| npmpc | sqp | casadi | 1.841 | 0.25 | 1.029 | 0.686 | 0.004 | 0.121 | 100 | 5 |
| npmpc | sqp | scaly | 1.270 | 0.72 | 0.474 | 0.675 | 0.004 | 0.118 | 100 | 5 |
| race_cars | ipopt | casadi | 4.963 | 0.65 | 0.379 | 4.372 | 0.000 | 0.212 | 1367 | 5 |
| race_cars | ipopt | scaly | 4.714 | 1.08 | 0.183 | 4.242 | 0.000 | 0.290 | 1367 | 5 |
| race_cars | sqp | casadi | 2.668 | 0.81 | 0.222 | 2.352 | 0.008 | 0.086 | 1367 | 5 |
| race_cars | sqp | scaly | 2.489 | 0.74 | 0.053 | 2.346 | 0.007 | 0.083 | 1367 | 5 |
| unbumpercars | ipopt | casadi | 146.563 | 0.34 | 142.032 | 0.000 | 0.000 | 0.278 | 200 | 5 |
| unbumpercars | ipopt | scaly | 21.106 | 0.77 | 17.022 | 0.000 | 0.000 | 0.270 | 200 | 5 |
| unbumpercars | sqp | casadi | 106.583 | 0.31 | 105.222 | 1.307 | 0.011 | 0.044 | 200 | 5 |
| unbumpercars | sqp | scaly | 11.331 | 1.74 | 10.045 | 1.244 | 0.010 | 0.032 | 200 | 5 |

Scaly reduces total SQP time relative to CasADi by 7% for race-car, 31% for neural-process MPC,
57% for chain, and 89% for unbumpercars. With IPOPT, Scaly reduces total time by 5% for race-car,
19% for neural-process MPC, and 86% for unbumpercars. Chain does not provide an IPOPT/CasADi runner.

## Episode agreement

The comparisons below use every recorded control step. Mismatch counts sum across all five episode
pairs.

| Problem | Solver | Repetitions | Per-step iteration mismatches | Per-step oracle-count mismatches | Maximum state difference | Maximum control difference |
|---|---|---:|---:|---:|---:|---:|
| chain | sqp | 5 | 0 | 0 | 4.77e-09 | 1.6e-08 |
| npmpc | ipopt | 5 | 0 | 0 | 4.05e-13 | 1.99e-15 |
| npmpc | sqp | 5 | 0 | 0 | 6.64e-10 | 3.36e-12 |
| race_cars | ipopt | 5 | 0 | 0 | 9.24e-14 | 5.53e-12 |
| race_cars | sqp | 5 | 0 | 0 | 1.07e-11 | 6.66e-09 |
| unbumpercars | ipopt | 5 | 0 | 0 | 1.84e-12 | 1.35e-12 |
| unbumpercars | sqp | 5 | 5 | 0 | 4.58e-09 | 3.69e-09 |

Every IPOPT pair matches per-step iterations and oracle-call counts. The chain, neural-process MPC,
and race-car SQP pairs also match. Unbumpercars has five SQP iteration mismatches across 1,000 paired
steps, so its SQP total is an observed closed-loop result rather than a strict equal-work
comparison.

## Reproduce the study

```bash
SCALY_VECTOR_LIBM=glibc SCALY_CC=gcc CC=gcc CXX=clang++ uv run benchmarks/run.py study --out-dir benchmarks/results/<study-name>
uv run benchmarks/run.py report benchmarks/results/<study-name>
```

The study command runs the frozen grids with five fresh processes, order seed 0, the
`performance` CPU governor, boost disabled, a 180-second compile limit, and a 50 MiB source limit.
The report command regenerates the Markdown tables from the saved artifacts. The current artifacts
are in `benchmarks/results/study-2026-09-23-c77-c79-libmvec`.
