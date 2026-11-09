# Fatrop stage Functions

Issue #240, PR #285, third in the #282 / #280 / #285 stack. The implementation
uses mapped dense stage derivatives for `DiscreteOCP.stage_problem()` and keeps
the whole-problem path for ordinary problems. The backend's `stage_oracles`
declaration selects mapped stage derivatives when the problem supplies stage
Functions. Fatrop opts in, and other backends default to whole-problem oracles.
The builder API is unchanged.

## Chain measurements

Measured on la015 on 2026-11-09 from the clean, pushed initial implementation on
`t3code/240-fatrop-stage-functions`. The raw provenance identifies that revision.
This is focused implementation evidence, separate from the published benchmark
study. The problem is `benchmarks.problems.chain.chain_ocp`, the 3D chain, with
M masses and N shooting intervals. Each comparison uses the same stage form,
initial trajectory, physical parameters, bounds, Fatrop 1.2.0 and `tol=1e-10`.
Removing only `stage_functions` from the problem selects the whole-problem path.

GCC 13.3.0 compiles both paths at `-O2 -march=native -fno-math-errno`, with glibc
2.39 vector math. Python is 3.14.3. Every CPU policy used the performance governor
with boost disabled. Ted set the policy before the run. One fresh process and
empty compile cache per cell, rotated path order, one compile/warmup solve,
then 30 solves from the same initial guess. The C-side statistics time oracles,
including bounds. Python dispatch and compilation do not enter `t_fe`.

All eight cells and all 240 timed solves succeeded. There were no failures,
skipped cells or iteration mismatches. Maximum solution difference across the
four path pairs was 8.993e-15. The native iteration counts were constant within
each cell. All reported violations were at most 1.1e-8, including Fatrop's bound
relaxation.

Function evaluation in milliseconds. Parentheses give the sample standard
deviation over the 30 solves, not uncertainty between processes.

| M | N | Iterations, both | Whole-problem median (SD) | Stage median (SD) | Maximum solution difference |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 10 | 11 | 0.818 (0.004) | 0.643 (0.019) | 9.437e-16 |
| 3 | 25 | 11 | 1.948 (0.017) | 1.546 (0.005) | 2.220e-15 |
| 5 | 10 | 11 | 4.803 (0.016) | 4.002 (0.061) | 7.327e-15 |
| 5 | 25 | 14 | 15.143 (0.116) | 12.617 (0.050) | 8.993e-15 |

Mapped-stage function evaluation took 17 to 21 percent less time in this run.
The coefficient of variation for function evaluation was at most 3.1 percent.
`summary.json` also records the total, solver and wrapper times and oracle counts.

The two wrappers do different amounts of derivative work. With 11 iterations,
whole-problem evaluation calls the gradient 23 times and the Jacobian 24 times,
while mapped stages call each 12 times. Both call the base 38 times and the
Hessian 11 times. At 14 iterations, these counts are 29/30 versus 15/15, with
47 base and 14 Hessian calls. The mapped Lagrangian blocks include their gradient
rows, so their Hessian callback avoids the whole-problem path's extra gradient
and Jacobian calls. The stage path also skips the next-state coefficient probe.
This comparison measures the complete oracle workload of each path, not an
isolated Jacobian kernel speedup.

## Reproduction

The posted measurement plan and launch authorization are recorded on #240.
Run from the project root on la015 with the CPU policy above, a clean pushed
revision and no concurrent measurements or builds:

```bash
SCALY_CC=gcc SCALY_VECTOR_LIBM=glibc uv run internal/notes/fatrop_stages_2026-11-09/measure.py --out benchmarks/results/fatrop-stages-reproduction
```

The completed run used tmux session `study-fatrop-stages-2026-11-09`, output
`benchmarks/results/fatrop-stages-2026-11-09/` and the sibling `.log`. Every cell
retains its native samples, solution, generated module/cache and provenance.
The script holds `~/.scaly-measuring` for the run and checks CPU policy before
and after. The checked-in `summary.json` is copied from that run.

## Correctness evidence

The core derivative test compares dense blocks with whole-problem sparse
derivatives and finite differences of an unrolled NumPy formulation. It covers
per-stage and global parameters, carried global variables, path and terminal
constraints, and the Lagrangian gradient convention. A one-percent Hessian
perturbation fails it.

Fatrop plugin tests compare primal and multiplier groups and iterations through
both paths, including N=1, zero controls, missing costs/constraints, changing
parameters and nested solves. The chain M=3/N=10 and race-car N=10 workload
gates compare the builder's general and stage forms. Reversing the stage
dynamics Jacobian makes both gates fail.

The free-final-time double integrator solves at T=2.012585018363112 in 12
iterations through both Fatrop paths. Its stage solution agrees with IPOPT's
general form within rtol=1e-6 and atol=1e-7.

`wt hook pre-merge --yes` passed, including 2548 tests passed and 52 skipped,
ty, Ruff, benchmark smoke, docs and the private-name check. The clean strict
docs build passed. Both Fatrop solver-guide fences, both paper-chain scripts
and the README's Python examples including export ran. Generated-C snapshots
are unchanged. Global-variable bounds remain #283, and #215 is outside scope.

## Investigation of the earlier Jacobian speedup

The 17–21 percent solve measurement and the earlier 3D chain sweep measure
different quantities. The earlier result is an isolated Jacobian comparison.
Hessians dominate the current solve's function evaluation. At benchmark M=3,
N=25, the Jacobian already takes 2.99 times less time through stage Functions,
but improving that kernel alone saves only 6.48 percentage points of projected
function evaluation. The Hessian contributes 88.6 percent of the original
workload. This accounts for most of the apparent discrepancy.

There is also a compiler gap. On the paper's formulation, inserting only
devrush's affine-loop recovery pass into the current compiler cuts the stage
Hessian's time by 64–68 percent. It removes almost all the difference from
devrush at M=3 and makes the current kernel faster at M=6 and M=15. This is a
compiler follow-up, separate from the stage-block assembly in this PR.

The historical compiler and the frozen devrush tip must also be distinguished.
On la015, reconstructing the tree that introduced the old results gives a
2.42–3.30 times Jacobian advantage for stages. The tip gives 0.52–1.22 times at
the same sizes: its whole-problem Jacobian improved much more than its stage
Jacobian. The newer constant-Jacobian and selection derivative optimizations do
not explain this change at M=6. Disabling them produces identical C.

No production code changed during this investigation. The final additions to
the PR are this report and [kernel_summary.json](kernel_summary.json).

### What the comparisons hold constant

"Current" below means the compiler on PR #285's approved builder/backend stack,
with the actual stage assembly in `src/scaly/solvers/_stage.py`. It does not
mean a separate checkout of `origin/main`, which lacks this builder path.
"Tip" means frozen devrush at
`6aa1e6b489d77c742ac1af0afd99417733fb713e`. "Historical" means external tree
`78a86faa`, dated 2026-09-28, which introduced
`examples/case_studies/fatrop_chain/README.md` and its saved sweep. Those
artifacts do not identify the measured revision, so this reconstruction cannot
establish that it is the exact original compiler.

All new timings use la015, the performance governor, boost disabled, GCC 13.3.0,
`-O2 -march=native -fno-math-errno`, and `-lmvec -lm`. The current compiler uses
its native eight-lane setting with glibc vector math. The tip uses its host
target with 64-byte vectors; its scalar loops also remain subject to GCC's
automatic vectorization. The historical compiler has no Target API. No
fast-math option is enabled. Compile logs record the actual commands; the
generic sweep-command entry in provenance is not the command for these kernels.

Each cell runs in a fresh process. A native C application binary interface
driver records 200 samples, each a batch of calls lasting at least one
millisecond, after warmup and calibration. Python dispatch, compilation and
solver work do not enter these times. These are medians, whereas the old Apple
table reports the best of 200 samples. The new Jacobian minima at the three
historical sizes are within 1.3 percent of their medians. This reporting choice
cannot explain the large differences.

The primary grid has 40 cells: eight paper sizes, N=25, and five forms per size.
The forms are current whole-problem oracles, actual current stage Functions,
a current port of the old concatenated stage contract, tip whole-problem
oracles, and tip stage Functions. The paper counts free masses, with
`nx = 3*(2*M+1)`. Both compilers receive its equations, constants, RK4 step,
seeded nonzero point and mixed-sign multipliers. Layouts, row order and signs
are normalized before checking. This uses a common point for all new runs;
the original Apple input arrays were not retained for replay.

Whole-problem Jacobians return compact sparse values, including initial and
control identity rows. Stage Jacobians return dense local blocks. The actual
current stage form also returns initial, path and terminal blocks. The old
stage contract concatenates dynamics Jacobians with residuals in `[u, x]`
order. Its Hessian concatenates the dense Hessian and Lagrangian gradient.
The actual current form includes that gradient row too, plus initial and
terminal contributions. Whole-problem Hessians return the compact lower
triangle. A "fused contract" row below reproduces the old stage outputs on
the current compiler; it is a layout and output-contract control, not an
isolated change to fusion alone.

The primary and historical outputs pass an independent CasADi stage reference,
with maximum normalized error 2.40e-15. The recent-chain derivatives pass
NumPy complex-step Jacobian and exact hyper-dual Hessian references. An offline
NumPy check of all 16 recent-chain value profiles also passes, with maximum
normalized error 1.89e-15. Compiler and copy controls compare with the retained,
already checked native outputs. The scatter control checks every matrix entry.

All 40 primary cells, 44 controls, six historical cells, four sparse-derivative
controls and three noise repeats completed. There were no timeouts or omitted
cells. Two failed diagnostic attempts remain in the raw record: the first
M=6 copy-removal driver redirected a constants buffer and failed its numerical
gate; the first scatter library lacked `-lm` and failed to load before timing.
Only the unfinished controls were resumed after fixing those drivers.

The primary sample coefficient of variation has median 0.55 percent and 90th
percentile 1.85 percent. Three samples exceeded 5 percent, the current M=3
whole gradient and historical M=3/M=15 whole Hessians. Repeating their identical
compiled C gave coefficients of variation of 0.55, 0.35 and 0.33 percent.
Medians moved by 0.04, 0.18 and 0.89 percent. Tables use those three repeats,
marked `*`; both attempts remain in the data. Dispersion describes samples in
one process, not uncertainty between processes.

### Per-kernel results on the paper's formulation

All times below are microseconds per native call. `g` is constraints alone;
the solver's fused objective/constraint `base` is recorded separately in the
JSON. Gradient means the objective gradient; stage Hessian time includes its
Lagrangian gradient row. The tables compare complete returned layouts as
described above.

#### Current compiler

| Paper M | nx | Form | Gradient | Jacobian | Hessian | g |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 3 | 21 | whole | 0.816* | 77.21 | 726.60 | 1.664 |
| 3 | 21 | stage | 0.215 | 82.53 | 661.72 | 1.730 |
| 4 | 27 | whole | 0.931 | 314.42 | 1163.54 | 2.189 |
| 4 | 27 | stage | 0.261 | 357.12 | 1341.25 | 2.110 |
| 5 | 33 | whole | 1.252 | 486.69 | 1659.29 | 2.601 |
| 5 | 33 | stage | 0.337 | 512.20 | 1698.83 | 2.575 |
| 6 | 39 | whole | 2.065 | 423.10 | 2175.57 | 3.660 |
| 6 | 39 | stage | 0.380 | 568.34 | 2419.40 | 3.692 |
| 8 | 51 | whole | 2.380 | 546.55 | 3168.89 | 4.805 |
| 8 | 51 | stage | 0.494 | 1141.85 | 3883.25 | 4.813 |
| 10 | 63 | whole | 3.570 | 685.69 | 3882.95 | 6.023 |
| 10 | 63 | stage | 0.615 | 1350.96 | 5710.20 | 6.013 |
| 12 | 75 | whole | 3.490 | 846.04 | 4593.75 | 6.768 |
| 12 | 75 | stage | 0.818 | 1917.56 | 8160.80 | 6.754 |
| 15 | 93 | whole | 4.323 | 1367.68 | 6424.32 | 12.214 |
| 15 | 93 | stage | 1.060 | 3649.19 | 11147.97 | 12.227 |

#### Frozen devrush tip

| Paper M | nx | Form | Gradient | Jacobian | Hessian | g |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 3 | 21 | whole | 0.292 | 109.76 | 346.72 | 2.754 |
| 3 | 21 | stage | 0.234 | 89.64 | 186.36 | 2.557 |
| 4 | 27 | whole | 0.373 | 179.82 | 442.27 | 3.387 |
| 4 | 27 | stage | 0.292 | 148.95 | 282.30 | 3.330 |
| 5 | 33 | whole | 0.428 | 243.05 | 695.32 | 4.127 |
| 5 | 33 | stage | 0.352 | 223.24 | 653.17 | 4.131 |
| 6 | 39 | whole | 0.496 | 295.25 | 961.74 | 5.045 |
| 6 | 39 | stage | 0.662 | 298.00 | 829.33 | 4.855 |
| 8 | 51 | whole | 0.635 | 380.48 | 1396.94 | 6.830 |
| 8 | 51 | stage | 0.527 | 513.63 | 1487.07 | 7.020 |
| 10 | 63 | whole | 1.174 | 478.02 | 1786.58 | 8.610 |
| 10 | 63 | stage | 0.647 | 773.50 | 2172.23 | 8.717 |
| 12 | 75 | whole | 0.913 | 577.76 | 2060.13 | 10.206 |
| 12 | 75 | stage | 0.772 | 1093.44 | 3178.60 | 10.747 |
| 15 | 93 | whole | 1.131 | 881.34 | 2715.29 | 12.741 |
| 15 | 93 | stage | 0.941 | 1697.71 | 5024.15 | 12.816 |

#### Historical reconstruction on la015

| Paper M | nx | Form | Gradient | Jacobian | Hessian | g |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 3 | 21 | whole | 0.290 | 301.45 | 364.21* | 2.724 |
| 3 | 21 | stage | 0.234 | 91.37 | 190.30 | 2.561 |
| 6 | 39 | whole | 0.498 | 873.10 | 1028.80 | 5.053 |
| 6 | 39 | stage | 0.663 | 306.18 | 940.36 | 4.852 |
| 15 | 93 | whole | 1.127 | 4267.95 | 2895.94* | 12.747 |
| 15 | 93 | stage | 0.940 | 1763.95 | 5333.06 | 12.838 |

#### Current compiler with the old fused stage contract

| Paper M | nx | Form | Gradient | Jacobian | Hessian | g |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 3 | 21 | fused | 0.325 | 56.12 | 628.81 | 1.638 |
| 4 | 27 | fused | 0.588 | 318.82 | 1368.67 | 2.076 |
| 5 | 33 | fused | 0.491 | 473.26 | 1909.01 | 2.459 |
| 6 | 39 | fused | 0.828 | 489.04 | 2231.61 | 3.204 |
| 8 | 51 | fused | 1.185 | 1021.75 | 3964.32 | 6.706 |
| 10 | 63 | fused | 0.917 | 1175.77 | 5660.55 | 8.628 |
| 12 | 75 | fused | 1.131 | 1616.84 | 7688.39 | 9.832 |
| 15 | 93 | fused | 1.971 | 3221.68 | 11072.80 | 12.725 |

### Explaining the original solve measurement

The benchmark chain counts all masses, including the fixed and actuated
endpoints. Its `nx = 3*(2*M-3)`, so benchmark M=5 has the same 21 states as paper
M=3. The original benchmark profiles are:

| Benchmark M | N | Form | Gradient | Jacobian | Hessian | g | base |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 3 | 10 | whole | 0.135 | 4.183 | 62.231 | 0.468 | 0.483 |
| 3 | 10 | stage | 0.034 | 1.774 | 53.130 | 0.473 | 0.490 |
| 3 | 25 | whole | 0.379 | 7.887 | 156.539 | 0.565 | 0.611 |
| 3 | 25 | stage | 0.103 | 2.639 | 132.760 | 0.572 | 0.713 |
| 5 | 10 | whole | 0.314 | 29.183 | 370.528 | 1.137 | 1.172 |
| 5 | 10 | stage | 0.059 | 22.284 | 332.108 | 1.155 | 1.172 |
| 5 | 25 | whole | 0.824 | 73.019 | 928.266 | 1.361 | 1.452 |
| 5 | 25 | stage | 0.220 | 55.186 | 829.324 | 1.469 | 1.446 |

For each original solve, multiplying these standalone kernel medians by the
recorded native call counts predicts `t_fe` within 2.9 percent. Count `base`
once: `n_eval_f` and `n_eval_g` report the same shared calls. Bounds and the
timer overhead remain outside this approximation. The profiles use one fixed
solved point and random multipliers rather than every iterate of the solve.

| Benchmark M | N | Projected whole/stage, ms | Observed whole/stage, ms | Whole Hessian share | Whole Jacobian share |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 10 | 0.806 / 0.625 | 0.818 / 0.643 | 84.9% | 12.4% |
| 3 | 25 | 1.943 / 1.520 | 1.948 / 1.546 | 88.6% | 9.7% |
| 5 | 10 | 4.828 / 3.966 | 4.803 / 4.002 | 84.4% | 14.5% |
| 5 | 25 | 15.278 / 12.510 | 15.143 / 12.617 | 85.1% | 14.3% |

The whole-problem Hessian accounts for 84–89 percent of this projected work;
the Jacobian accounts for 10–15 percent. At benchmark M=3/N=25 the projection
assigns 13.46 percentage points of reduction to the faster Hessian, 6.48 to the
faster Jacobian, 1.69 to fewer calls, 0.33 to the gradient, and -0.20 to `base`.
The predicted reduction is 21.8 percent; the observed reduction is 20.6 percent.
At M=5/N=25 the corresponding contributions are 9.07, 3.50, 5.44, 0.11 and
0.00 points, versus an observed 16.7 percent reduction and a projected 18.1
percent. These are arithmetic projections, not a new instrumented solve or an
exact causal budget. Their residuals set the accuracy of the explanation.

### Causes, ranked by their measured effect

1. **The quantity and its weights.** A 3-times faster Jacobian can coexist with
   17–21 percent less function evaluation because the Hessian dominates. This
   is the largest explanation of the headline comparison. It is measured on
   the original problem with the actual call counts, not inferred from the
   paper's different sizes.

2. **Affine loops and explicit widening.** The current compiler retains
   flattened derivative index maps. For example, the M=3 fused Hessian's
   `lanes_13` helper widens a 576-entry loop with `/24`, `/6`, `%4`, `%6` and
   twelve temporary gather arrays. Devrush emits nested 24-by-4-by-6
   affine loops for the same 24 tangent directions. The old
   [`passes/program/delinearize_loops.py`](https://github.com/PREDICT-EPFL/scaly/blob/6aa1e6b489d77c742ac1af0afd99417733fb713e/src/scaly/passes/program/delinearize_loops.py) recovers those coordinates. Inserting
   only that pass before workspace packing in the current process gives:

   | Paper M | Current stage J / H | With affine recovery J / H | Tip stage J / H |
   | ---: | ---: | ---: | ---: |
   | 3 | 82.53 / 661.72 | 56.82 / 209.49 | 89.64 / 186.36 |
   | 6 | 568.34 / 2419.40 | 211.73 / 794.84 | 298.00 / 829.33 |
   | 15 | 3649.19 / 11147.97 | 1259.70 / 4049.30 | 1697.71 / 5024.15 |

   This removes 95 percent of the M=3 current-to-tip stage Hessian difference
   and exceeds the tip's speed at M=6/M=15. The same experiment cuts the current
   whole-problem Hessians by 63–65 percent. On the original recent-chain N=25
   profiles it changes whole/stage Hessians from 156.54/132.76 to 71.08/59.95
   microseconds at M=3, and from 928.27/829.32 to 347.20/297.06 at M=5. It helps
   both forms. The adapted pass changes only its imports and removes an
   inapplicable enum reference. These experiments modify only a process-local pipeline, not
   production code or a PR file.

   Disabling Scaly's explicit widening at paper M=6 changes the actual stage
   Jacobian/Hessian from 568.34/2419.40 to 439.90/1473.91 microseconds. The tip's
   corresponding target-width switch changes less than one percent. GCC still
   vectorizes scalar loops in both cases. The result supports improving gather
   widening and index recovery, not disabling all vectorization.

3. **A much faster whole-problem Jacobian since the old study.** Historical
   versus tip whole Jacobians take 301.45 versus 109.76 microseconds at M=3,
   873.10 versus 295.25 at M=6, and 4267.95 versus 881.34 at M=15. Stage times
   change only 2–4 percent. Consequently, whole/stage ratios change from
   3.30/2.85/2.42 to 1.22/0.99/0.52. This is a measured revision effect on one
   machine, separate from the Apple comparison.

   The sparse coloring widths remain 36/42/60 at those sizes. Generated
   Jacobian source falls from 1.54/4.42/14.76 MB to 0.24/0.60/1.34 MB, and
   workspace falls from 44712/92646/378675 to 23121/56934/142308 doubles.
   [Devrush's later expression rewrites](https://github.com/PREDICT-EPFL/scaly/blob/6aa1e6b489d77c742ac1af0afd99417733fb713e/src/scaly/passes/expr.py) compose gathers through transposes, sums
   and scattered blocks; its loop pass also recovers periodic table indices
   and folds constant table reads. These remove intermediate compressed
   arrays and large recombination tables. The source and workspace changes
   support this explanation. The timing difference is measured for the group
   of revisions, not partitioned between those individual rewrites.

   In particular, the newer `_sparse_jacobian_linear_part` and
   `_sparse_jacobian_at_selections` rules are a ruled-out explanation at M=6.
   Neither accepts this model. Baseline, either rule disabled, and both
   disabled all produce the same C hash and take 295.36–295.69 microseconds.
   Their later presence on devrush is not evidence that they caused this gain.

4. **The C compiler and formulation matter, but their effects do not add.**
   Recompiling identical paper M=6 C with Clang 20.1.8 instead of GCC 13.3.0
   changes current stage Jacobian/Hessian times from 568.34/2419.40 to
   455.74/1246.48 microseconds. Tip stage times change from 298.00/829.33 to
   238.92/1480.27. Clang helps one Hessian and hurts the other. It closes the
   relative Hessian difference in this x86 control, but it does not reproduce
   Apple clang 21 or Apple M3 Max hardware. This overlaps with the loop and
   widening effects above and must not be added to them.

   Even at equal dimensions, the formulations differ. At 21 states, the
   benchmark M=5 whole/stage Jacobian ratio is 1.32 versus 0.94 for paper M=3;
   at 39 states it is 0.98 for benchmark M=8 versus 0.74 for paper M=6; at 93
   states it is 0.46 for benchmark M=17 versus 0.37 for paper M=15. The benchmark
   maps accelerations over masses and expresses each interior link force
   twice before optimization. The paper maps each link once. The benchmark
   also has runtime physical parameters, different constants, gravity axis,
   time step, cost weights and a terminal cost. These matched-dimension
   controls measure their combined effect; they do not isolate each choice.

### Smaller costs and hypotheses the controls do not support

The current compiler already fuses ranges, has blocked matrix panels and maps
local dense derivative Functions over stages. Both implementations preserve
the stage map, and neither inspected M=3 Hessian uses a blocked general matrix
multiplication kernel. The evidence does not support attributing the gap to
missing stage mapping, universal scalar expansion, a blocked matrix multiply,
or one derivative direction per variable of the whole horizon.

Returning a pure Jacobian instead of the old Jacobian-plus-residual contract
changes current fused-contract Jacobian times by 3.8, 1.7 and 0.2 percent at
M=3/6/15. Separately, replacing the actual stage contract with the old contract
changes its Jacobian by 12–32 percent at those sizes, but this changes ordering
and outputs as well as fusion. Its Hessian does not improve consistently over
the eight-size grid. Recomputing the small `base` oracle cannot explain the
large Hessian gap.

The current mapped Hessian writes scratch storage and then copies it to its
output; the old form writes the flat mapped output directly. Redirecting only
that destination and removing the final copy changes current fused-contract
Hessian times from 628.81/2231.61/11072.80 to 626.42/2161.99/11035.31
microseconds at M=3/6/15. This is at most 3.2 percent, much smaller than the
affine-loop effect.

The old wrapper packs dense blocks with `blasfeo_pack_dmat`; the new wrapper
writes permuted entries with `BLASFEO_DMATEL`. An isolated, paired transfer
control takes 7.43/25.27/136.61 microseconds for individual writes and
3.89/11.74/69.71 for bulk packing at M=3/6/15. It uses the same CasADi-bundled
BLASFEO build for both methods because the vendored Fatrop library hides the
pack symbols. It times only dynamics-block transfer, not a complete wrapper.
Both actual wrappers place this work outside function evaluation, so it
explains none of the `t_fe` comparison.

### What the Apple comparison can establish

The saved Apple M3 Max table uses macOS 26, Apple clang 21,
`-O2 -mcpu=native -fno-math-errno`, a minimum rather than a median, and a
different load policy. It reports only the two derivative kernels in the
sweep. It also reports Hessian regressions: stage/whole is 514/468 microseconds
at M=6 and 3139/1269 at M=15. Its Jacobian result never established a 3–4 times
improvement in all function evaluation.

The historical reconstruction on la015 recovers a substantial Jacobian
advantage without Apple hardware. It does not recover every Apple ratio.
The old Apple whole/stage Jacobian ratios at M=3/6/15 are 3.35/3.99/2.53;
the historical la015 ratios are 3.30/2.85/2.42. These ratios are reported
separately from all la015 timing tables. No measurement here partitions
hardware, Apple clang, load policy and unknown original revision effects.
Applying only the tip's Apple M3 render policy on x86 changes M=6 times by
less than one percent. That control does not run on Apple silicon or change
the target used when constructing derivatives, so it cannot rule out those
effects.

### Proposed follow-ups, not filed

- Recover affine loop coordinates before widening and workspace packing, with
  differential tests for the flattened derivative gathers. Recheck both
  solver forms, snapshots and native solve counts before making a compiler
  performance claim.
- Make the widening cost decision account for scalar gathers and integer
  index arithmetic. Compare the resulting C with Clang as well as GCC.
- Reconstruct the historical-to-tip Jacobian improvement one rewrite at a time
  if an individual-pass attribution is needed. The constant-Jacobian and
  selection rules are already ruled out at M=6; avoid crediting them on the
  basis of commit dates.
- Consider direct mapped-output allocation and bulk block packing after the
  larger compiler costs. Their controls establish small kernel savings and
  a separate wrapper saving, not a large function-evaluation improvement.
- If Apple-specific claims are needed, rerun the same gated kernels on an
  Apple M3 Max with recorded revision and compiler provenance. Keep those
  absolute times separate from la015.

### Investigation artifacts and reproduction

Plans, launch authorization and continuation records are in the PR discussion.
All native studies ran serially under `~/.scaly-measuring`, from the same clean
pushed implementation, with CPU policy checked between cells. The original
benchmark smoke and smallest-size sweep passed before launching. No builds or
tests competed with a timing process. The machine lock is released.

The following ignored directories remain on la015, with sibling logs:

- `benchmarks/results/fatrop-kernel-investigation-2026-11-09/`
- `benchmarks/results/fatrop-kernel-controls-2026-11-09/`, including both failed
  attempts and the continuation logs
- `benchmarks/results/fatrop-kernel-historical-2026-11-09/`
- `benchmarks/results/fatrop-kernel-attribution-2026-11-09/`

Generated C, headers, binaries, inputs, outputs, compiler commands and 200 raw
samples per timed kernel remain in each cell. The ignored
`benchmarks/results/fatrop-kernel-investigation-2026-11-09-tools/` contains the
unmodified frozen tip and historical archives, drivers, reference gates and
analysis scripts. `kernel_summary.json` records their SHA-256 digests, the full
per-kernel medians and dispersion, control results and solve-time projections.
The maintainer's no-code-on-PR instruction takes precedence over the benchmark
skill's usual tracked-analysis-script convention. Reproduction needs this
retained tool directory as well as the tracked data.

From a clean pushed checkout on la015 with the same CPU policy, run these
commands serially, using fresh output directories. `run.py` measures the full
primary grid; the control controller uses that directory. The historical
controller's output directory is fixed in its retained script.

```bash
SCALY_CC=gcc SCALY_VECTOR_LIBM=glibc uv run --no-sync benchmarks/results/fatrop-kernel-investigation-2026-11-09-tools/run.py --out benchmarks/results/fatrop-kernel-reproduction
SCALY_CC=gcc SCALY_VECTOR_LIBM=glibc uv run --no-sync benchmarks/results/fatrop-kernel-investigation-2026-11-09-tools/controls.py --out benchmarks/results/fatrop-controls-reproduction --primary benchmarks/results/fatrop-kernel-reproduction
SCALY_CC=gcc SCALY_VECTOR_LIBM=glibc uv run --no-sync benchmarks/results/fatrop-kernel-investigation-2026-11-09-tools/historic_worker.py --branch devrush --path whole --masses 6 --out benchmarks/results/fatrop-historical-reproduction-M6-whole
```

The current summary is generated without retiming by the retained
`summarize.py`. The actual scatter measurements used the corrected link flag;
the retained first-attempt drivers explain the rejected outputs and load error.
Only this note and the data summary enter the PR. No follow-up issue was filed.
