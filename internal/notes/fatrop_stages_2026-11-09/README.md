# Fatrop stage Functions

Issue #240, PR #285, third in the #282 / #280 / #285 stack. The implementation
uses mapped dense stage derivatives for `DiscreteOCP.stage_problem()` and keeps
the whole-problem path for ordinary problems. This selection rule is an open
point for the maintainer. The builder API is unchanged.

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
