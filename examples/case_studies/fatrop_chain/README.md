# Case study E1: the Fatrop hanging chain

The hanging-chain MPC problems of the Fatrop paper (Vanroye et al., IROS 2023, Tables III and IV),
solved as published with rockit and CasADi, then with Scaly's oracles in the same solvers. The paper
could not compile CasADi's C for these two problems ("compilation failed due to insufficient
available memory") and reported Fatrop's time without function evaluation. This study measures what
that compile costs, and what the oracles cost once they are compiled, by CasADi and by Scaly.

```bash
examples/case_studies/fatrop_chain/baseline/setup.sh     # pins fatrop_benchmarks and applies one fix
uv run --with rockit-meco==0.6.7 examples/case_studies/fatrop_chain/compare.py --out examples/case_studies/fatrop_chain/results/solve.json
uv run --with rockit-meco==0.6.7 examples/case_studies/fatrop_chain/sweep.py --dim 3 --out examples/case_studies/fatrop_chain/results/sweep_3d.json
uv run examples/case_studies/fatrop_chain/scaly_impl.py   # the Scaly problem on its own, through IPOPT
uv run --with jupyterlab jupyter lab examples/case_studies/fatrop_chain/fatrop_chain.ipynb
```

`fatrop_chain.ipynb` explains the study, solves the chain with Scaly's oracles in IPOPT and in Fatrop
(live, in seconds), shows the generated C, and plots the recorded comparison and sweep. Its
`RUN_BASELINES` flag reruns the CasADi side, which takes tens of minutes.

## The problem

`no_masses = 6` masses on springs between the ground and an actuated end mass whose velocity is the
control, `|u| <= 1`, in 2D (26 states, `T = 4 s`) and 3D (39 states, `T = 2 s`), 25 multiple-shooting
intervals of one RK4 step each, tracking cost on the end mass and on the velocities. The upstream
code is `hanging_chain/` of
[fatrop_benchmarks](https://gitlab.kuleuven.be/robotgenskill/fatrop/fatrop_benchmarks) at `9e7025a`
(2023-08-03). One line of `benchmark_tools.py` needs a sixth argument for rockit 0.6's integrator
(`baseline/fatrop_benchmarks.patch`); nothing else is changed.

`scaly_impl.py` writes the same nonlinear program: the same variable order, the same rows, the same
initial guess. Evaluated at random points it matches rockit's objective and all 726 (2D) or 1089 (3D)
constraints to 4e-15 relative.

## Files

| File | What it is |
| --- | --- |
| `baseline/setup.sh`, `baseline/fatrop_benchmarks.patch` | the pinned upstream repository and its one-line fix |
| `baseline/run_casadi.py` | the CasADi side: rockit's NLP to `nlpsol` (Fatrop 1.1.8 or IPOPT 3.14.11, both from the CasADi 3.8.0 wheel), oracles in CasADi's VM or compiled |
| `baseline/cc_timed.sh` | a compiler wrapper that logs wall time and peak memory, used by both sides |
| `scaly_impl.py` | the problem in Scaly, `si.rk4` over a model that maps the link force over the chain |
| `fatrop_dropin.py`, `fatrop_scaly.c` | Fatrop's C interface filled from five Scaly Functions (objective, gradient, dynamics residual, stage Jacobians, stage Lagrangian Hessians), linked against the same `libfatrop` CasADi's plugin uses |
| `run_scaly.py` | the Scaly side: IPOPT drop-in, Fatrop drop-in, Scaly SQP |
| `compare.py` | every variant in fresh processes; writes `results/solve.json` and the table below |
| `sweep.py`, `time_kernel.c` | the chain-size sweep of IPOPT's two kernels, timed from C |
| `fatrop_chain.ipynb` | the study as a notebook: explanation, the Scaly side live, the recorded results plotted |

`tests/integration/test_case_study_fatrop_chain.py` checks the stage Hessian and Jacobian path on a
small chain against NumPy.

## Results

Apple M3 Max, macOS 26, Apple clang 21, `-O2 -mcpu=native -fno-math-errno` on both sides. This is
not the reference machine of [fairness](../../../docs/results/fairness.md); these numbers do not
belong on the results pages. Each cell waited for a one-minute load average below 6 (16 cores; the machine idles near 4).

### Solving the paper's problems

The fastest of five fresh processes per row (one for the CasADi C rows, whose compile takes minutes),
each the minimum over 30 solves. Function evaluation is the time inside the oracles: CasADi's
`t_wall_nlp_*` timers, Scaly's solver statistics, and for Fatrop with Scaly's oracles a C timer around
each oracle call in `fatrop_scaly.c`. "Rest" is everything else in the solve: the solver, its linear
algebra, and the scatter of oracle values into the solver's blocks. The CasADi rows are timed in Python
around the `nlpsol` call, the Scaly rows in C. Agreement is the largest difference from CasADi's
IPOPT solution; the drop-in pairs agree with each other to 7e-15.

| Chain | Solver | Oracles | Iterations | Solve, ms | Function evaluation, ms | Rest, ms | Setup, s | Compile, s | Peak compiler memory, GB | Agreement |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2D | Fatrop 1.1.8 | CasADi VM | 13 | 24.61 | 21.41 | 3.20 | 1.1 | 0.0 | 0.00 | 3e-08 |
| 2D | IPOPT 3.14.11 (CasADi's) | CasADi VM | 14 | 42.09 | 23.03 | 19.06 | 1.1 | 0.0 | 0.00 | 0e+00 |
| 2D | Fatrop 1.1.8 | CasADi C (JIT) | 13 | 6.97 | 3.90 | 3.07 | 75.4 | 72.7 | 2.91 | 3e-08 |
| 2D | IPOPT 3.14.11 (CasADi's) | CasADi C (JIT) | 14 | 22.03 | 2.91 | 19.12 | 53.8 | 51.4 | 2.55 | 0e+00 |
| 2D | Fatrop 1.1.8 | Scaly C | 13 | 4.97 | 2.63 | 2.34 | 3.1 | 1.4 | 0.12 | 3e-08 |
| 2D | IPOPT 3.14.19 (Scaly's) | Scaly C | 14 | 24.27 | 5.55 | 18.72 | 5.8 | 3.3 | 0.18 | 4e-15 |
| 2D | Scaly SQP + PIQP | Scaly C | 4 | 22.98 | 1.67 | 21.31 | 6.2 | 3.5 | 0.23 | 2e-06 |
| 3D | Fatrop 1.1.8 | CasADi VM | 14 | 51.28 | 43.61 | 7.67 | 1.8 | 0.0 | 0.00 | 3e-07 |
| 3D | IPOPT 3.14.11 (CasADi's) | CasADi VM | 14 | 90.03 | 45.09 | 44.94 | 1.8 | 0.0 | 0.00 | 0e+00 |
| 3D | Fatrop 1.1.8 | CasADi C (JIT) | 14 | 16.23 | 8.59 | 7.64 | 271.0 | 267.0 | 5.27 | 3e-07 |
| 3D | IPOPT 3.14.11 (CasADi's) | CasADi C (JIT) | 14 | 56.36 | 10.09 | 46.27 | 184.8 | 180.9 | 4.70 | 0e+00 |
| 3D | Fatrop 1.1.8 | Scaly C | 14 | 15.91 | 9.81 | 6.11 | 4.1 | 2.0 | 0.10 | 3e-07 |
| 3D | IPOPT 3.14.19 (Scaly's) | Scaly C | 14 | 61.60 | 16.90 | 44.70 | 14.1 | 9.7 | 0.38 | 7e-15 |
| 3D | Scaly SQP + PIQP | Scaly C | 4 | 62.59 | 6.65 | 55.94 | 13.4 | 9.6 | 0.43 | 2e-05 |

Compile is the C compiler's own wall time, from `baseline/cc_timed.sh`; setup is everything up to the
first solution. The two IPOPT rows run different builds of IPOPT and MUMPS, so only their function
evaluation compares; the rest of their time is the build's (see fairness.md, "The IPOPT build is a
first-order confound").

### Chain-size sweep

The 3D chain at `N = 25` for 3 to 15 masses: IPOPT's two kernels, the Lagrangian Hessian and the
constraint Jacobian, each built in a fresh process, compiled through `baseline/cc_timed.sh` with a
180 s budget, and timed from C (`time_kernel.c`, the best of 200 calls). A backend that misses the
budget is not tried at larger sizes. Every kernel is checked against CasADi's SX evaluation of rockit's
NLP at the same point. "Scaly (stage oracles)" are the Fatrop drop-in's kernels (dense stage blocks,
not the NLP's sparse matrices), so its columns time the same derivatives in a different layout. These
cells waited for a load below 10, not 6: other sessions kept the machine busy for most of an hour.

| Masses | States | Backend | Hessian, µs | Jacobian, µs | Build, s | Compile, s | Peak compiler memory, MB | Source, kB | Machine code, kB | Max rel. error |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 21 | Scaly (IPOPT oracles) | 182.0 | 164.0 | 1.4 | 2.9 | 144 | 1808 | 139 | 1e-15 |
| 3 | 21 | Scaly (stage oracles) | 138.0 | 49.0 | 0.8 | 1.1 | 96 | 197 | 76 | 2e-16 |
| 3 | 21 | CasADi SX | 185.0 | 59.0 | 3.8 | 70.0 | 1728 | 13606 | 4258 | 9e-18 |
| 3 | 21 | CasADi MX | 822.0 | 306.0 | 2.3 | 113.7 | 1683 | 2593 | 1658 | 2e-16 |
| 3 | 21 | CasADi map(SX) | 404.0 | 239.0 | 0.4 | 5.6 | 406 | 3928 | 711 | 2e-15 |
| 4 | 27 | Scaly (IPOPT oracles) | 264.0 | 293.0 | 1.5 | 3.8 | 177 | 2906 | 188 | 1e-15 |
| 4 | 27 | Scaly (stage oracles) | 219.0 | 67.0 | 0.9 | 1.3 | 144 | 268 | 112 | 1e-15 |
| 4 | 27 | CasADi SX | 197.0 | 70.0 | 4.7 | 61.5 | 2086 | 17943 | 5900 | 1e-16 |
| 4 | 27 | CasADi MX | 1106.0 | 468.0 | 2.8 | 165.6 | 2573 | 3483 | 2341 | 2e-16 |
| 4 | 27 | CasADi map(SX) | 523.0 | 317.0 | 0.5 | 6.9 | 480 | 5466 | 910 | 2e-15 |
| 5 | 33 | Scaly (IPOPT oracles) | 346.0 | 408.0 | 1.8 | 4.9 | 226 | 4086 | 248 | 2e-15 |
| 5 | 33 | Scaly (stage oracles) | 369.0 | 88.0 | 1.0 | 1.3 | 88 | 247 | 131 | 2e-15 |
| 5 | 33 | CasADi SX | 227.0 | 84.0 | 5.8 | 94.1 | 2606 | 22843 | 7498 | 2e-16 |
| 5 | 33 | CasADi MX | compile exceeded 180 s | | | | | | | | |
| 5 | 33 | CasADi map(SX) | 817.0 | 457.0 | 0.7 | 9.7 | 615 | 7726 | 1273 | 3e-15 |
| 6 | 39 | Scaly (IPOPT oracles) | 468.0 | 539.0 | 2.2 | 8.5 | 251 | 4839 | 357 | 8e-15 |
| 6 | 39 | Scaly (stage oracles) | 514.0 | 135.0 | 1.1 | 1.6 | 98 | 294 | 164 | 8e-15 |
| 6 | 39 | CasADi SX | 267.0 | 103.0 | 7.4 | 132.3 | 3254 | 27487 | 9134 | 1e-16 |
| 6 | 39 | CasADi map(SX) | 964.0 | 541.0 | 0.8 | 11.2 | 619 | 9262 | 1475 | 8e-15 |
| 8 | 51 | Scaly (IPOPT oracles) | 677.0 | 771.0 | 2.8 | 10.4 | 372 | 7079 | 489 | 1e-15 |
| 8 | 51 | Scaly (stage oracles) | 899.0 | 217.0 | 1.1 | 2.1 | 123 | 400 | 201 | 8e-16 |
| 8 | 51 | CasADi SX | compile exceeded 180 s | | | | | | | | |
| 8 | 51 | CasADi map(SX) | 1478.0 | 811.0 | 1.4 | 18.8 | 920 | 14146 | 2207 | 1e-15 |
| 10 | 63 | Scaly (IPOPT oracles) | 864.0 | 1239.0 | 3.1 | 9.7 | 464 | 9615 | 553 | 2e-14 |
| 10 | 63 | Scaly (stage oracles) | 1349.0 | 368.0 | 1.4 | 2.6 | 148 | 522 | 270 | 2e-14 |
| 10 | 63 | CasADi map(SX) | 2276.0 | 1172.0 | 1.9 | 30.4 | 1119 | 19884 | 3231 | 2e-14 |
| 12 | 75 | Scaly (IPOPT oracles) | 1019.0 | 1564.0 | 3.5 | 11.1 | 552 | 12523 | 552 | 1e-15 |
| 12 | 75 | Scaly (stage oracles) | 1919.0 | 528.0 | 1.2 | 5.3 | 232 | 532 | 398 | 9e-16 |
| 12 | 75 | CasADi map(SX) | 2982.0 | 1449.0 | 2.6 | 40.1 | 1372 | 24789 | 4284 | 1e-15 |
| 15 | 93 | Scaly (IPOPT oracles) | 1269.0 | 2021.0 | 4.0 | 12.3 | 794 | 15364 | 582 | 5e-14 |
| 15 | 93 | Scaly (stage oracles) | 3139.0 | 800.0 | 1.5 | 5.5 | 348 | 691 | 393 | 5e-14 |
| 15 | 93 | CasADi map(SX) | 4019.0 | 1783.0 | 3.0 | 58.1 | 1592 | 31112 | 5667 | 3e-14 |

The sweep above was timed one call per sample. `time_kernel.c` now times batches of calls,
because the macOS clock ticks in microseconds and E2's kernels take less than one; every kernel
here takes at least 49 µs, so the per-call timing is within 2% of what the batched harness gives.

- CasADi SX is the fastest encoding while it compiles: at 6 masses its Hessian takes 267 µs against
  468 µs for Scaly, after a 132 s compile that peaks at 3.3 GB. From 8 masses it does not compile
  within 180 s, and MX already fails from 5 masses.
- From 8 masses on, `map(SX)` is the best CasADi that compiles. Against it Scaly's Hessian is 2.2x
  (8 masses) to 3.2x (15 masses) faster, the stage Jacobian 3.7x to 2.2x, with compiles of 2 to 12 s
  against 19 to 58 s.
- Scaly's IPOPT Jacobian is 2.5 to 4.6x its own stage Jacobian at every size; its source grows to 15 MB,
  most of it the constant seed tables of the recombination described in the report.
