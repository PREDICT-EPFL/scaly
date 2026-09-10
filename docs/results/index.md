# Benchmark results

The latest compiler closeout was interrupted; its partial measurements appear first. Earlier
complete Hessian scalability sweeps and canonical closed-loop runs follow with their scope and
limitations.

Everything here is measured against **CasADi**, in both its SX (scalar) and MX (block) forms, on
the same problem, and every measurement is gated by a correctness check: each backend's compact
derivative is scattered into a dense matrix using its own sparsity pattern and compared entry by
entry against a reference built by a different construction — NumPy for the race-car Jacobian and
for the chain, race-car and neural-process model predictive control (MPC) Lagrangian Hessians, Alloy's dense per-stage Jacobian
of an unrolled twin for the chain and neural-process MPC Jacobians, and a CasADi Lagrangian Hessian
for the safety filter. A cell that does not agree produces no timing.

!!! warning "The Jacobian tables await a same-machine refresh"

    The sweep now benchmarks the full constraint Jacobian taken from each optimizer's solver
    descriptor. The published race-car and neural-process MPC Jacobian tables below predate that
    correction and measured equality rows only. They remain labeled as historical rather than being
    relabeled with numbers from a different kernel.

!!! warning "The historical IPOPT numbers below are not yet a fair comparison"

    An audit in August 2026 found that the `ipopt+alloy` and `ipopt+casadi` columns differ in two
    things besides the oracle provider: they link **different builds of IPOPT** (worth 2.7× to 17× in
    solver-internal time on its own) and the CasADi column evaluates its oracles through CasADi's
    **virtual machine** rather than compiled C. On the one problem where a fully controlled
    comparison has been built, alloy's oracle advantage is **1.13×**, not the 4.5× the shipped column
    reports. Every closed-loop IPOPT multiple below is therefore an upper bound on the oracle effect
    and is marked as such. The kernel sweeps, the code-size figures and the two `alloy-sqp` columns
    are unaffected. [Read the audit](fairness.md).

## Partial Track C closeout, 2026-09-09 to 2026-09-10

!!! warning "Interrupted study, partial results only"

    The run started on 2026-09-09 and stopped during chain repetition 4. It saved 671 of the
    planned 690 sweep rows and 30 of 40 closed-loop episodes. The last log update was
    2026-09-10 at 00:45 Zurich time. The process was absent when checked that morning.
    The cause is unconfirmed. No final study manifest or automatic report was produced.
    These tables summarize saved results after the interruption. They do not establish a
    complete study or ratify the paper's suite-wide claims.

The saved runs used the frozen protocol: five fresh processes, order seed 0, the performance
CPU governor, boost off, a 180-second kernel compile limit, and the 50 MiB source cap. The source
snapshot still matched every recorded file checksum when checked on 2026-09-10, before this
write-up. All 500 saved timings passed the workload correctness checks. No saved row reports a
correctness or runtime error. An interrupted attempt has no result and is not counted as a timeout.

| Workload | Saved / planned rows | Successful timings | Compile timeouts | Skips after timeout | Episodes |
|---|---:|---:|---:|---:|---:|
| race_cars | 225 / 225 | 210 | 5 | 10 | 10 / 10 |
| npmpc | 240 / 240 | 166 | 30 | 44 | 10 / 10 |
| unbumpercars | 150 / 150 | 85 | 25 | 40 | 10 / 10 |
| chain | 56 / 75 | 39 | 10 | 7 | 0 / 10 |
| Total | 671 / 690 | 500 | 70 | 101 | 30 / 40 |

The completed workload sweeps give these process means. The CasADi column names the fastest
completed encoding at that size. All displayed timings have five successful processes.
CV denotes the sample coefficient of variation across processes.

| Problem | Size | Alloy, µs | Alloy CV, % | Best completed CasADi | CasADi, µs |
|---|---:|---:|---:|---|---:|
| Race-car | N=40 | 19.834 | 0.83 | SX | 15.906 |
| Race-car | N=500 | 228.815 | 0.77 | SX | 216.058 |
| Neural-process MPC | N=12 | 34.725 | 0.30 | MX | 43.160 |
| Neural-process MPC | N=200 | 592.402 | 0.94 | called MX | 4964.348 |
| Unbumpercars | C=8 | 876.016 | 1.34 | MX | 10761.821 |
| Unbumpercars | C=16 | 2086.643 | 0.38 | none completed | |
| Unbumpercars | C=32 | 6985.492 | 0.52 | none completed | |

Alloy remains 24.7% slower than SX
at race-car N=40, so the 20% runtime gate is not met across the useful range. At N=500 the gap
is 5.9%. Neural-process MPC at N=12 is 19.5% faster than completed MX. Its N=200 comparison
is still conditional on the compile budget: plain MX failed at N=100 and all three matrix multiplication variants
failed at N=200. Unbumpercars at C=8 is 12.3× faster than completed MX. Alloy now completes
C=32, but no CasADi encoding completes C=16 or C=32 under the budget, so those sizes establish
completed Alloy evaluations rather than a measured runtime advantage over those missing builds.

### Completed closed-loop workloads

All 30 saved episodes succeeded. Each row below averages five processes. Times are per step.
QP denotes time spent solving the quadratic program.

| Problem | Solver | Provider | Total, ms | Total CV, % | Function evaluation, ms | QP, ms | Globalization, ms | Glue, ms | Steps | Processes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| npmpc | sqp | alloy | 1.304 | 0.87 | 0.504 | 0.675 | 0.004 | 0.120 | 100 | 5 |
| npmpc | sqp | casadi | 1.860 | 1.57 | 1.039 | 0.694 | 0.004 | 0.123 | 100 | 5 |
| race_cars | sqp | alloy | 2.569 | 0.83 | 0.134 | 2.344 | 0.007 | 0.083 | 1367 | 5 |
| race_cars | sqp | casadi | 2.651 | 0.78 | 0.219 | 2.339 | 0.008 | 0.085 | 1367 | 5 |
| unbumpercars | sqp | alloy | 10.778 | 0.58 | 9.510 | 1.228 | 0.010 | 0.031 | 200 | 5 |
| unbumpercars | sqp | casadi | 106.724 | 0.69 | 105.364 | 1.305 | 0.011 | 0.044 | 200 | 5 |

| Problem | Solver | Repetitions | Per-step iteration mismatches | Per-step oracle-count mismatches | Maximum state difference | Maximum control difference |
|---|---|---:|---:|---:|---:|---:|
| npmpc | sqp | 5 | 0 | 0 | 4.86e-10 | 2.04e-12 |
| race_cars | sqp | 5 | 0 | 0 | 4.68e-12 | 3.39e-09 |
| unbumpercars | sqp | 5 | 10 | 0 | 2.62e-07 | 2.02e-07 |

Race-car and neural-process MPC match per-step iteration and oracle-call counts across providers.
Unbumpercars has ten iteration mismatches across 1,000 paired steps. Its total-time ratio is an
observed closed-loop result, not a strictly equal-work comparison. There are no chain episodes
from this run. The providers use the same sequential quadratic programming source and PIQP library but separate
compiled wrappers.

[The partial scalability tables](scalability.md#partial-track-c-closeout-2026-09-09-to-2026-09-10)
retain every size for the three completed sweeps and describe the missing chain attempts.

## Interim Track C snapshot, 2026-09-09

!!! warning "This is an interim compiler checkpoint"

    These measurements cover the Track C work through layout-aware matrix products, bounded
    scalarization, baked seed tiles, and shared arithmetic rewrites. Track C is not finished. In
    particular, C-8's range-based loop compiler and its affine-index work remain open. Replace this
    checkpoint after that work rather than treating these numbers as the final compiler result.

The branch ran the complete frozen study: 690 kernel attempts and 40 SQP episodes, arranged in
20 provider pairs. A clean
`dev` worktree then ran every matching Alloy kernel and Alloy SQP episode, 115 kernel attempts and
20 episodes. Both runs used five fresh processes, seed 0, the `performance` governor, boost off,
and the same machine. The branch's native compile flags are part of the measured change, so this
is a comparison of the two configured compilers, not a pass-only ablation.

Every timed kernel passed its independent dense-reference check. The branch recorded 504 successful
timings, 76 compile timeouts, 105 skips after an earlier failure, and five source-cap skips. The
`dev` Alloy run recorded 110 successful timings and five source-cap skips.

| Problem | Point | Branch Alloy, µs | `dev` Alloy, µs | Change | Best current CasADi | Current comparison |
|---|---:|---:|---:|---:|---|---:|
| Race-car | N=40 | 20.725 | 26.521 | -21.9% | SX, 15.894 µs | 1.30× slower |
| Race-car | N=500 | 243.773 | 329.688 | -26.1% | SX, 212.714 µs | 1.15× slower |
| Neural-process MPC | N=12 | 46.586 | 83.626 | -44.3% | MX, 43.759 µs | 1.06× slower |
| Neural-process MPC | N=200 | 774.717 | 1395.931 | -44.5% | called MX, 4988.224 µs | 6.44× faster |
| Unbumpercars | C=8 | 976.155 | 3462.814 | -71.8% | MX, 10749.947 µs | 11.01× faster |
| Unbumpercars | C=16 | 2270.197 | 7316.158 | -69.0% | none completed | only Alloy timed |
| Chain | M=5 | 828.436 | 2609.411 | -68.3% | mapped SX, 635.349 µs | 1.30× slower |

Race-car remains slower than SX, but it comes within 20% at N=100, 200, and 500. Neural-process
MPC is within 17% of MX through N=50, then wins once MX exceeds the compile budget. The
layout-aware matrix-product lowering changes unbumpercars most: Alloy's advantage over MX at C=8
grows from 2.9× on `dev` to 11.0×. Explicit scalarization cuts the chain kernel by 60–68% against
`dev`, but mapped SX remains 1.30–2.00× faster across M=3, 5, and 9.

Some generated kernels take longer to build or use more source. Neural-process kernel compilation rises by 17–25%. Unbumpercars
kernel compilation rises 2.27× at C=8 and 3.16× at C=16. At chain M=9, executable code grows
2.22× and compilation rises 29%. Race-car moves the other way: executable code drops by about
half and runtime falls 19–27%. The [interim scalability tables](scalability.md#interim-track-c-compiler-snapshot-2026-09-09)
give every size and the source, workspace, and compilation tradeoffs.

The same compiler changes carry into the closed loop:

| Problem | Branch total, ms | `dev` total, ms | Change | Branch function evaluation, ms | `dev` function evaluation, ms | Change | Current CasADi total, ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| race_cars | 2.574 | 2.613 | -1.5% | 0.140 | 0.183 | -23.5% | 2.681 |
| npmpc | 1.369 | 1.649 | -17.0% | 0.570 | 0.852 | -33.1% | 1.863 |
| unbumpercars | 11.302 | 37.864 | -70.2% | 10.032 | 36.572 | -72.6% | 106.683 |
| chain | 7.728 | 10.165 | -24.0% | 1.537 | 3.976 | -61.3% | 14.362 |

All 60 episodes completed successfully. On the branch, race-car, neural-process MPC, and chain
matched per-step iterations and oracle counts between providers in every repetition. Unbumpercars
had 15 iteration mismatches across 1,000 paired steps, with matching oracle counts and maximum
state and control differences below 3.4e-8. Race-car's lower function-evaluation cost barely moves
the total because the quadratic program dominates each step.

## Hessian scalability baseline, September 2026

These generated-C microbenchmarks measure the exact sparse Lagrangian Hessian with synthetic
inputs. The reference desktop used `performance`, boost off, five fresh processes per cell,
and varied backend order. Race-car and neural-process measurements combine the pilot and its
range extension. Unbumpercars uses a fresh full grid with the corrected synthetic-input policy;
its original pilot remains recorded separately.

The table shows the canonical size for each problem. Runtime is the process mean. Dispersion is
the sample coefficient of variation (CV) across processes. The CasADi column is the fastest
completed encoding among those tested at that size.

| Problem | Size | Alloy mean, µs | Alloy CV, % | Best CasADi encoding | CasADi mean, µs | CasADi CV, % |
|---|---|---:|---:|---|---:|---:|
| Race-car | N=40 | 26.500 | 0.45 | SX | 17.035 | 0.07 |
| Unbumpercars | C=8 | 3478.775 | 0.53 | MX | 9949.845 | 1.15 |
| Neural-process model predictive control | N=12 | 83.458 | 1.47 | MX | 52.672 | 2.17 |

Race-car covers N=1–500. Alloy takes 1.44–1.62 times SX runtime throughout the grid, so the
20% runtime target remains unmet. Its executable source stays near 50 KB, but its large-horizon
caller workspace exceeds SX's.

Neural-process covers N=6–200. Alloy remains slower than plain MX through N=25. Under the frozen
compile budget it is 1.22 and 1.24 times faster than the best completed CasADi encoding at N=50
and N=100, and 3.56 times faster at N=200, where only call-node MX completes among the CasADi
variants. The large-horizon source and caller-workspace comparison also favors Alloy.

Unbumpercars covers C=2–32. Alloy is 1.06, 1.63, and 2.86 times faster than MX at C=2,4,8.
Only Alloy completes at C=16 among the tested encodings. No encoding supplies a timing at C=32:
Alloy reaches the 50 MiB generated-source cap, mostly because of static metadata; the CasADi
variants are skipped after smaller-size failures.

The primary grids contain 615 rows: 445 successful timings, 75 compilation timeouts, 90 skips
after smaller-size failures, and five source-limit skips. No correctness check failed in these
completed sweeps. Every timed primary cell has five successful processes. The
[initial pilot](scalability.md#initial-frozen-protocol-pilot-2026-09-05) remains intact; the
[range tables and reproduction commands](scalability.md#extended-hessian-sweeps-2026-09-06)
record the previous full grids and their input policies. See also the
[measurement protocol](fairness.md#the-measurement-protocol).

Closed-loop timings are a separate experiment below.

## Canonical closed-loop SQP runs, 2026-09-05

These runs execute the full receding-horizon controller with sequential quadratic programming
(SQP), compiled Alloy or CasADi oracles, and the shared PIQP library. Both providers use the
same SQP implementation and problem settings. Each provider compiles its own solver-and-oracle
wrapper, so the complete binaries are not identical. CasADi uses each problem's existing
SQP oracle construction, with every generated oracle transformed and only the upper Hessian
triangle requested. This column does not select an encoding from the kernel sweep. All runs use the
[reference-machine protocol](fairness.md#the-measurement-protocol), performance selected,
boost disabled, seed 0, and five fresh processes per provider with separate empty caches.
Both complete wrappers compile with cc 13.3.0 at `-O2`. The study manifest records that compiler
separately from the clang compiler used for the kernel sweeps.

Timing is the mean native solver time per control step, including the first step, averaged across
the five processes. The native statistics separate function evaluation, solving the quadratic
program (QP), globalization, and glue code. Timing excludes Python dispatch,
controller construction, plant simulation, and recording. CV is the sample coefficient of variation
of the five process means. Construction and first-solve wall times remain in the saved mode tables.

| Problem | Provider | Total, ms | Total CV, % | Function evaluation, ms | QP, ms | Globalization, ms | Glue, ms |
|---|---|---:|---:|---:|---:|---:|---:|
| race_cars | alloy | 2.632 | 0.77 | 0.183 | 2.356 | 0.007 | 0.085 |
| race_cars | casadi | 2.669 | 0.84 | 0.225 | 2.349 | 0.007 | 0.088 |
| unbumpercars | alloy | 37.838 | 0.20 | 36.552 | 1.242 | 0.010 | 0.033 |
| unbumpercars | casadi | 110.892 | 0.61 | 109.527 | 1.313 | 0.011 | 0.041 |
| npmpc | alloy | 1.658 | 1.02 | 0.854 | 0.680 | 0.005 | 0.120 |
| npmpc | casadi | 1.693 | 0.68 | 0.878 | 0.690 | 0.004 | 0.121 |
| chain | alloy | 10.213 | 1.44 | 3.998 | 5.482 | 0.008 | 0.725 |
| chain | casadi | 14.273 | 0.31 | 8.053 | 5.475 | 0.008 | 0.736 |

### Episode agreement

The comparisons below use every recorded control step. Matching mean iteration counts alone
would not establish that the providers did the same work. Mismatch counts sum across all five
episode pairs. The oracle column counts individual counter entries, so one step can contribute five.

| Problem | Steps per episode | Successful solves | Per-step iteration mismatches | Per-step oracle-count mismatches | Maximum state difference | Maximum control difference |
|---|---:|---|---:|---:|---:|---:|
| race_cars | 1367 | 13670/13670 | 0 | 0 | 6.45e-12 | 6.02e-09 |
| unbumpercars | 200 | 2000/2000 | 10 | 50 | 3.05e-07 | 2.35e-07 |
| npmpc | 100 | 1000/1000 | 0 | 0 | 5.12e-09 | 3.51e-11 |
| chain | 90 | 900/900 | 0 | 0 | 4.9e-09 | 1.92e-08 |

Race-car runs use N=40 and complete one 1,367-step lap of `fsds_competition_1`.
The race-car comparison has matching per-step iterations and all five oracle-call counters in
every repetition. QP work dominates the total. The measured total-time gap is only 1.4%,
with process CVs of 0.77% and 0.84%; this does not establish a substantial total-time improvement.

Unbumpercars uses C=8 and 200 control steps. All ten episodes completed without solver failures
or collisions. Steps 148 and 198 have different iteration and oracle-call counts between
providers, despite maximum state and control differences below 4e-7. Its total timing ratio
is an observed closed-loop result, not a strict equal-work oracle-cost comparison.

Neural-process MPC uses N=12 and 100 control steps. Chain uses M=5, a controller horizon of
N=12, and 90 control steps. Both comparisons have matching per-step iterations and oracle-call
counts in every pair. The separate chain Hessian microbenchmark uses N=40.

### Reproduce the closed-loop runs

One command runs all four problems into an unused directory and renders both tables above:

```bash
uv run benchmarks/run.py study --out-dir benchmarks/results/followup/<date> --only closed-loop
```

A single problem is the underlying command, shown here for the race-car run:

```bash
uv run benchmarks/run.py closed-loop --problem race_cars --solver sqp --oracle alloy,casadi --repetitions 5 --order-seed 0 --headline --boost off --out-dir benchmarks/results/followup/<date>/closed-loop/race_cars
```

Raw episodes, trajectories, per-step `telemetry.csv`, mode tables, and provenance are local
gitignored artifacts under `benchmarks/results/followup/2026-09-05/closed-loop/<problem>/`.
`uv run benchmarks/run.py report <study-dir>` reads the telemetry, compares the providers step by
step, and writes `closed_loop.summary.json` and `report.md`. These are measured runs with retained
artifacts, not an immutable publication archive.

The older measurements below retain their original limitations.

## Race-car equality Jacobian — historical

A 4-state, 2-control bicycle model with a `tanh` rolling-resistance term, integrated with RK4 over
a horizon, differentiated as a compact sparse Jacobian. The current sweep adds the corridor rows
from the solver descriptor; this table is the last equality-only run.

| Stages | Alloy µs | CasADi SX µs | CasADi MX µs | Alloy KB | SX KB | MX KB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.09 | 0.09 | 0.13 | 19.3 | 14.1 | 42.2 |
| 10 | 1.05 | 0.91 | 1.50 | 17.9 | 95.0 | 374.6 |
| 50 | 5.24 | 4.70 | 8.39 | 22.1 | 455.1 | 1996.0 |
| 100 | 10.27 | 9.28 | 18.72 | 28.3 | 907.0 | 4019.8 |
| 200 | 21.20 | 18.44 | compile > 180 s | 40.6 | 1810.8 | 8676.0 |
| 500 | 54.46 | 45.76 | skipped | 78.0 | 4528.9 | — |

The size column is the point. Alloy's growth is entirely in constant index tables — data, not
code — because the stage structure is expressed with `vmap` and survives lowering as a real loop.
Construction time follows: 248 ms to build the 500-stage case against SX's 481 ms.

Full tables, workspace figures and the reasoning are in [the scalability sweep](scalability.md).

## Hanging chain of masses — historical Jacobian result

The classic chain NMPC: `M` point masses on springs under gravity, RK4 over a 40-step horizon, with
the equality Jacobian as the kernel. This is the suite's largest sparse structured problem, and it is
the one where loop-preserving lowering costs the most. Both backends compiled at `-O3` and timed in a
C++ loop; one variant at a time on an idle machine.

| M | backend | runtime | lines | source | compile |
|---:|---|---:|---:|---:|---:|
| 5 | alloy | 587 µs | **583** | **124 KB** | **0.87 s** |
| 5 | CasADi MX unrolled | **244 µs** | 26 822 | 1.2 MB | 25.4 s |
| 5 | CasADi SX unrolled | — | 215 190 | 4.5 MB | **> 600 s** |
| 17 | alloy | 3.78 ms | **625** | **690 KB** | **1.75 s** |
| 17 | CasADi SX + `map` | **2.52 ms** | 494 159 | 20.0 MB | 38.4 s |
| 17 | every other CasADi encoding | — | 318 036 – 1 784 091 | 14.7–36.9 MB | **> 600 s** |

In this earlier equality-Jacobian experiment, Alloy evaluates 1.5–2.4 times slower while
generating less source and compiling faster. These measurements use an older kernel and protocol;
they do not establish the performance of the current Lagrangian Hessian.

These numbers had never been published; the sweep ran and the results went nowhere. Details and the
full variant matrix are in [the fairness audit](fairness.md#the-chain-sweep-a-suspected-handicap-that-was-not-one-and-a-result-that-is-not-published).

## A neural network at every node of a horizon

A conditional neural process — a small dense network, `9 → 32 → 32 → 2`, with its weights and its
latent code read out of the parameter vector — used as the dynamics model inside a Furuta-pendulum
swing-up controller, from the paper *Neural Process Model Predictive Control*. This is the shape most
deployed learning-based MPC actually has, and it is the workload where loop-preserving lowering
matters most, because the thing being repeated at every horizon node is a matmul rather than a
handful of scalar operations.

**Alloy's generated source barely moves along either axis.** 417 lines for the sparse equality
Jacobian at every horizon from 6 to 200 and every decoder width from 16 to 256, and 1041 for the
exact Lagrangian Hessian everywhere but the narrowest decoder, where it is 1043. CasADi SX reaches 1.31 million lines at a 100-step horizon, a factor of
3142, and stops being compilable: clang exceeds a fifteen-minute budget there. CasADi MX runs out too
on the harder kernel — 227 s to compile the 50-step Hessian, and past the budget at 100. Alloy
compiles the 200-step Hessian in under a second.

**On runtime, the horizon is a tie against MX and the decoder width is not.** Alloy's cost grows
about 4× per doubling of width, which is what a matmul-dominated kernel should cost; MX grows 891×
over a 16× width increase, crossing from 1.2× faster than alloy at the shipped width to 2.6× slower
at 256. Against SX on the horizon axis, alloy is 1.55–4.46× faster wherever SX still compiles; the
one cell where SX is ahead is the narrowest decoder, at 0.90.

**In closed loop this is the one problem where a fully controlled comparison has been built.** The
same 100-step swing-up episode, same plant, same warm starts. Both IPOPT columns are generated C
behind a single `ctypes` call, compiled by the same compiler at the same level, linked against the
same `libipopt.so`, and timed by `clock_gettime` inside C. Trajectories agree to 8.8e-13 in state.

| column | mean solve | mean FE |
|---|---:|---:|
| IPOPT, alloy oracles | **3.48 ms** | 1.36 ms |
| IPOPT, CasADi oracles (code-generated) | 3.93 ms | — |
| alloy-sqp, alloy oracles | **1.29 ms** | 0.66 ms |
| alloy-sqp, CasADi oracles | 1.57 ms | 0.92 ms |

So the oracle-provider margin is **1.13×** on IPOPT and **1.22×** on the SQP, with function
evaluation itself a wash — two independent fair comparisons landing in the same place. All four
columns bring the pendulum upright at the same step with identical iteration counts inside each
solver.

**What the shipped `ipopt+casadi` column reports instead is 17.0 ms, and that is a configuration, not
a result.** Three things account for the difference, and only the first is CasADi's to own:

| configuration | mean solve | mean FE | build |
|---|---:|---:|---:|
| interpreted SX — what the suite ships | 17.01 ms | 12.87 ms | 0.5 s |
| interpreted MX | 7.79 ms | 3.85 ms | 0.02 s |
| compiled MX (`jit`, `-O3`) | 5.23 ms | 1.47 ms | 24 s |
| compiled SX (`jit`, `-O3`) | 7.25 ms | 3.15 ms | **1017 s** |
| code-generated MX, CasADi's IPOPT | 5.15 ms | — | 23 s |
| code-generated MX, alloy's IPOPT | **3.93 ms** | — | 13 s |

Expanding to scalar SX is a trap here: it triples the interpreter's work, and compiled it costs a
seventeen-minute build to land *slower* than compiled MX. The last two rows are the same artifact
differing only in which `libipopt.so.3` the loader resolves — 1.31× for the solver build alone. The
full decomposition is in [the fairness audit](fairness.md).

The durable results on this problem are therefore code size, build time and the decoder-width axis,
plus a real but modest 1.1–1.2× on oracle-driven solve time.

The other durable gap is **build cost**: about a second for alloy's whole oracle set, 24 s for
compiled CasADi MX, and sixteen minutes for CasADi's scalar expansion — the code-size result
reappearing in the closed loop.

The paper's authors write that they shrank this network "as small as possible while providing the
necessary performance" to fit a 20 ms sampling time, and their own controller is uncompiled, so the
17.34 ms row is a faithful reproduction of *their system* even though it is not a fair CasADi
baseline. Combining the measured closed-loop iteration count with per-iteration oracle cost measured
at wider decoders puts the fast column at 4.6 ms at width 64 and 17 ms at width 128 — two doublings
past the width they settled on, inside the same budget, though at 128 the honest range is 13–17 ms
depending on how the extrapolation is anchored. That step is an extrapolation rather than a
measurement and is labelled as one in `benchmarks/problems/npmpc/README.md`, which also carries the
formulation, the vendored reference data, and what the cross-implementation gate against the authors'
own released episode does and does not establish.

Per-cell numbers are in [the scalability sweep](scalability.md#neural-process-mpc-on-the-furuta-pendulum-npmpc).

## Safety filter with a solver in the loop

Vehicles with pairwise control-barrier constraints over a `6 → 256 → 128 → 3` neural dynamics model,
solved with IPOPT at every step of a closed loop. The two columns solve the same problem, but they do
*not* run the same solver or the same kind of oracle — see the caveat below the table.

**This is the suite's strongest oracle result, and it is the one that survived the audit intact** —
it was simply being quoted about twice too high. At C=8, comparing alloy against CasADi's *best*
configuration rather than the one the suite ships:

| column | mean solve | mean FE |
|---|---:|---:|
| alloy | **58.9 ms** | 52.3 ms |
| CasADi, best configuration (MX, code-generated, `-O3`) | 146.2 ms | 136.0 ms |
| CasADi, as the suite ships it (SX, interpreted) | 300.6 ms | 283.7 ms |

So **2.48×**, not the 4.5–9.0× previously published. The isolated exact Lagrangian Hessian says the
same thing from a different direction — both sides compiled at `-O3`, timed in a C++ loop with no
Python anywhere, alloy is 1.16× / 1.77× / **3.03×** ahead at C = 2 / 4 / 8. Two independent
measurements at the same magnitude, and the advantage grows with the vehicle count, which is the
mapped Hessian doing what it exists for.

Two things this problem taught the rest of the suite. `expand=True`, the shipped default, costs
CasADi 1.87× here, and the right setting is per problem rather than a suite-wide default. Compiling
CasADi's oracles is worth only 1.08× on this workload, against 2.6–55× on the other two, because an
MX graph over matmuls is already dispatching into compiled block kernels.

The older continuous-time model's 2.8–3.9× has not been re-measured under the same treatment and
should be read as an upper bound. Details in [the fairness audit](fairness.md#unbumpercars-c8-40-steps-exact-lagrangian-hessian).

What the problem exists to exercise is the exact sparse Lagrangian Hessian, computed through
preserved `VMAP` structure — the construction most of alloy's sparse machinery is built to make
cheap. It is the default here, and it roughly halves IPOPT's iteration count against a
limited-memory approximation.

Per-cell numbers are in [the scalability sweep](scalability.md#discrete-time-hcbf-safety-filter-unbumpercars);
formulation, provenance and closed-loop behaviour live with the problem, in
`benchmarks/problems/unbumpercars/README.md`.

## Reproducing

```bash
# every headline sweep and closed loop on these pages, one directory, then the rendered tables
uv run benchmarks/run.py study --out-dir benchmarks/results/followup/<date>

# the full sweep
uv run python benchmarks/run.py sweep

# one workload
uv run python benchmarks/run.py sweep --workloads race_cars

# the closed-loop safety filter
uv run python benchmarks/run.py closed-loop --problem unbumpercars --solver ipopt --oracle alloy

# the neural-process-MPC swing-up, fastest column
uv run python benchmarks/run.py closed-loop --problem npmpc --solver sqp --oracle alloy

# both oracle providers in one run, for a direct comparison
uv run python -m benchmarks.problems.unbumpercars.run_closed_loop --solver ipopt --oracle both
```

The fairness audit's own scripts live under `benchmarks/results/fairness/` and are described on
[that page](fairness.md#reproducing-this-page).

Each sweep cell compiles its own Google Benchmark binary containing alloy and the chosen backend.
Cells that exceed the per-cell compile timeout (180 s by default) or the generated source cap
(50 MB) are skipped, and once a backend fails at one size, larger sizes for that backend are
skipped too — both source size and compile cost increase monotonically with the horizon, so there
is nothing to learn from trying.

Raw per-cell data lands in `benchmarks/results/sweep/scalability.csv`, with each cell's generated
code, samples, binary and logs beside it.

## Reading these fairly

A few caveats worth stating rather than burying.

- **The comparison is per problem, not in general.** These are problems with repeated structure and
  exploitable sparsity, which is what alloy is built for and where its advantage is real. A dense
  unstructured problem would not show the same picture.
- **CasADi SX is genuinely fast, and pulls ahead as problems grow.** It trails alloy at the
  smallest horizons and leads by around 19% at 500 stages. The argument for alloy on these problems
  is source size, construction time and compile time, not raw evaluation speed. On the race-car
  closed loop the same thing shows up on the oracle side: once CasADi is compiled, its function
  evaluation is **3.1× faster** than alloy's (0.40 against 1.22 ms per solve at N=40).
- **The closed-loop IPOPT multiples are not oracle measurements.** They are contaminated by a
  different IPOPT build and by CasADi running interpreted. [The fairness audit](fairness.md) has the
  decomposition and the per-problem configuration tables.
- **Numbers come from one machine.** Absolute timings vary with hardware; the ratios are the
  durable part.
- **Historical results are labelled.** [The scalability page](scalability.md) keeps measurements
  from retired formulations, marked as such, so an old number is never mistaken for a current one.
