# Benchmark protocol

Maintained. The full measurement protocol and the reasoning behind each rule, moved out of the
published [fairness page](../../docs/benchmarks/fairness.md), which keeps a short reader-facing
version. Rationale for what a comparison holds constant goes here; the matching task goes in
[GitHub Issues](https://github.com/PREDICT-EPFL/scaly/issues).

The figures on the published pages read `docs/assets/benchmarks/sweep.summary.csv` (the four
`sweep/<problem>/<problem>.summary.csv` files of a study, concatenated) and
`docs/assets/benchmarks/closed_loop.summary.csv` (written by `benchmarks/run.py report`). Paired
per-step ratios from that report feed `closed_loop_speedup.summary.csv`, derived by
`internal/notes/perf_2026_10_13/compare.py --publish`. `sweep.csv` concatenates the raw sweep rows,
which keep the source size of cells that never compiled, and `closed_loop.steps.csv` holds every
closed-loop step. All of them come from the October 13, 2026 study,
`benchmarks/results/study-2026-10-13-post-179`, measured at `38bd5896`. The two sweep files are that
study rewritten in the current sweep format, with sizes re-measured from its kernel objects and the
duplicate GEMM encodings removed, as stand-in data until the next study replaces them.

## Reference machine

Every published timing comes from one machine.

| | |
|---|---|
| System | Minisforum F7BSC desktop |
| CPU | AMD Ryzen 9 7940HS, 8 cores and 16 threads, simultaneous multithreading enabled |
| Memory | 30 GB |
| Operating system | Ubuntu 24.04.4 LTS, kernel 7.0.0-28-generic, glibc 2.39 |
| Compilers | gcc 13.3.0 for Scaly just-in-time compilation, clang 20.1.8 for the Google Benchmark harness |
| Stack | Python 3.14.3, Scaly `main` at `38bd5896`, CasADi 3.8.0, NumPy 2.4.6, PIQP 0.6.4 |
| CPU policy | `amd-pstate-epp`, `performance` governor, boost disabled |

Absolute timings from another machine must not share these tables.

## Measurement protocol

The study uses the following controls:

- It runs one benchmark process at a time.
- It uses one fresh process per cell or provider, with an empty compilation cache. The October 12
  study used five; [Why one process](#why-one-process) records the change.
- It rotates backend order from seed 0.
- It checks the `performance` governor and disabled boost state before and after each headline run.
- It compiles sweep kernels as C with Clang, using `-O3 -march=native -fno-math-errno
  -fveclib=libmvec`. Only the Google Benchmark wrapper is C++.
- It records the compiler, flags, CPU policy, affinity, package versions, run order, and command in
  the artifact provenance.
- It stops a kernel compile after 180 seconds and skips generated source above 50 MiB.
- After a backend fails at one size, it skips that backend at larger sizes.
- It reports each closed loop's spread over steps and the paired per-step speedup, and retains
  every failed attempt.

The study completed 138 sweep attempts. Of these, 103 produced timings, 15 reached the compilation
limit, and 20 were skipped after an earlier failure. All timed kernels passed both their symbolic
and compiled-C correctness gates. All 23 Scaly cells and all 15 closed-loop episodes completed.

## Numerical policy

The headline study uses glibc vector math. The decision on 2026-09-23 was to compare the
fastest validated configurations tested for each provider on the reference machine. This is a
selection policy, not proof that every kernel benefits from vector math. All headline cells
come from one complete study with the same policy. Scalar and vector cells are not mixed.

### Vector-math study policy

Scaly's ahead-of-time sweep modules use `vector_libm="glibc"`, and just-in-time closed-loop
builds use `SCALY_VECTOR_LIBM=glibc`. The Clang sweep compiler receives `-fveclib=libmvec`
for both providers, and both link `-lmvec`. GCC closed-loop builds use
`-O2 -march=native -fno-math-errno` and link `-lmvec` for both providers. GCC has no equivalent
vector-library selection option. Scaly emits explicit vector calls where legal.

The benchmark rejects Clang for measured just-in-time builds under this policy because Scaly JIT
does not pass its vector-library selection flag. Clang remains supported for sweep builds.
Compile logs and provenance record the commands. `vector-symbols.json` records actual sweep
object references. Both providers' race-car N=5 objects reference vector sine and cosine;
Scaly also references vector tanh. A link flag alone would not establish vector use.

Vector math can change rounding. Keep the existing independent numerical checks, record numerical
differences and the glibc version, and check closed-loop trajectories, solver iterations, and
oracle-call counts again. Bitwise equality with scalar libm is not a requirement for vector-libm
results. The lane-width bitwise checks with vector math disabled remain in force.

This decision does not enable reciprocal multiplication, reduction reassociation, or other
fast-math transformations. Preserve the original reduction order and keep compiler contraction
flags equal between providers. Reciprocal experiments remain separate because they can change
overflow and underflow as well as rounding. The [code-generation reference](../../docs/api/codegen.md#render-options)
defines these options.

## Kernel comparisons

The sweep measures the exact sparse Lagrangian Hessian from each problem's solver descriptor.
Each backend receives the same numerical sample for a cell and writes its compact result into
preallocated buffers. Google Benchmark calls the generated C symbol directly, so Python dispatch
does not enter the timing.

Every cell checks the compact result after scattering it with that backend's sparsity pattern. The
race-car, chain, and neural-process MPC Hessians use NumPy references built stage by stage with
complex-step first derivatives and hyper-dual second derivatives. The unbumpercars Hessian uses an
independently constructed CasADi MX Hessian of the same formulation. This last check can catch
construction and sparsity errors, but it cannot detect an error shared by both CasADi expressions.

CasADi's encoding materially changes both runtime and compilation cost. The results therefore report
the fastest completed CasADi encoding at each size and name that encoding. A missing CasADi timing
means that no tested encoding completed under the compile budget. It does not establish that no
possible CasADi formulation can complete.

Executable size is the compiled kernel object, so it counts the constant tables a kernel keeps, in
whatever form each generator stores them. Source size depends on how each generator writes C. It
shows how much code the C compiler has to process, not what the kernel adds to a binary.

## Closed-loop comparisons

The SQP pairs use the same Scaly SQP implementation, PIQP subsolver, settings, warm starts, and
generated C application binary interface. Only the oracle provider changes. CasADi transforms each
generated oracle and supplies the same upper Hessian triangle as Scaly.

Each closed loop uses the fastest validated CasADi Hessian encoding at its canonical size in the
sweep: SX for race cars N=40 and chain M=5, MX for neural-process MPC N=12 and unbumpercars C=8.
Every oracle uses that encoding, not only the Hessian. `check_fastest_casadi_encoding` checks the
controller construction against the published sweep. The October 13 sweep confirms those choices.

The IPOPT pairs use the same IPOPT 3.14.19 library, nonlinear program, variable and parameter layout,
bounds, options, warm starts, and compiled C boundary. Chain has no IPOPT/CasADi runner,
so the results make no IPOPT provider comparison for that problem.

Native timers cover solver time per step. The telemetry separates function evaluation, quadratic
program solution for SQP, globalization, and wrapper work. Timing excludes controller construction,
Python dispatch, plant simulation, and recording. Each provider compiles its own wrapper, so total
time compares the complete generated solver path, not one shared binary with a swapped function
pointer.

Function-evaluation timers differ in scope. Scaly includes its bounds-evaluation kernel, which has
no CasADi equivalent. CasADi sums its oracle timers, which may not cover sparse-matrix copies.
Scaly's fused `base` kernel computes the objective and constraints together on both requests, while
CasADi evaluates them separately. These implementation differences remain part of the measurement.

Every IPOPT provider pair has identical per-step iteration and oracle-call counts. The chain,
neural-process MPC, and race-car SQP pairs do too. Unbumpercars has three SQP iteration mismatches
across 200 paired steps, with no oracle-count mismatches and maximum state and control differences
below 7e-9. Its total-time ratio is an observed closed-loop result, not a strict equal-work
oracle-cost comparison.

## Reading the results

Kernel timings isolate derivative evaluation. Closed-loop totals include the shared solver work, so
a faster oracle may have little effect when the quadratic program or nonlinear solver dominates.
Race-car demonstrates this distinction: Scaly's SQP median per-step function-evaluation speedup is
2.96×, while the median total ratio is 1.04×. The IPOPT median total ratio is 1.08×. Chain SQP is
nearly equal in total time, with CasADi's function evaluation faster.

Compilation timeouts bound the claims. At unbumpercars C=16 and C=32, the study establishes that
Scaly completed within the fixed budget while the tested CasADi encodings did not. It does not assign
a runtime ratio to a missing kernel.

## Measurement limits

One process gives no estimate of between-process dispersion. The closed-loop standard deviation
is the sample standard deviation over native step times, not measurement noise. Speedup medians
and 10th/90th percentiles describe the paired per-step ratios, not confidence intervals. They
need not equal the ratio of provider means. The complete sweep also reports source size,
workspace and compilation cost. These measurements cover the reference x86 machine and do not
establish Apple M4 performance. Rerun a questionable cell with `--repetitions N` to estimate noise.

## Why one process

Decided on 2026-10-13, after the October 12 study, for every later study. Each extra process
recompiled every cell, which made repetitions the bulk of a study's time, and the spread they
measured was small next to the reported gaps:

- Sweep runtimes moved by a coefficient of variation of 0.58% at the median cell across five
  processes, 2.06% at the 90th percentile and 5.49% at worst (Scaly's npmpc N=25 Hessian, whose
  slowest process was 13% slower than its fastest). That worst case is accepted.
- Kernel compile times moved by 0.6% at the median and 3.6% at worst.
- Closed-loop solver time per step moved by 0.1% to 1.5% across processes for every controller.

The report describes one deterministic episode by its spread over steps. The standard deviation
over steps measures how the controller's work varies along the episode, not measurement noise.
The per-step ratio of CasADi's time to Scaly's pairs matching steps. Its median and 10th to 90th
percentiles give the speedup with its spread. Iteration and oracle-count agreement must still be
checked, and mismatches qualify the comparison. On the October 12 data that ratio's 10th and 90th
percentiles stay within 9% of the median for every pair. `--repetitions N` remains for checking a
cell that looks noisy.

## Why the CPU policy is fixed

Every provider runs with the `performance` governor and boost disabled. Holding the CPU policy
fixed prevents frequency-policy differences from becoming part of the provider comparison.

## Why the native compiler flags are fair

Scaly compiles on the machine it runs on, so both providers use `-march=native` on the reference
machine. Both receive the same optimization level and `-fno-math-errno`, which permits math calls
to inline without preserving an unused `errno`. Compiler contraction flags also remain equal.
The provenance sidecars and compile logs record the exact commands.

## Why chain uses common-subexpression elimination

The harness applies CasADi common-subexpression elimination with `ca.cse` to the chain
nonlinear-program expressions before constructing `nlpsol`. This controls generated-source size
and compilation cost for that formulation.

## CasADi transformation and the single GEMM encoding

The sweep transforms every CasADi oracle before code generation. It measures one general matrix
multiplication (GEMM) encoding, with CasADi's own product kernel. CasADi 3.8 can generate BLAS or
BLASFEO calls for a dense product tagged `"classic"` or `"blasfeo"`, but the products that automatic
differentiation creates never carry the tag, and `transform` drops it from the others. In the
October 13 study the classic and BLASFEO encodings generated C identical to the reference one, so
the sweep no longer runs them.

## What a fair closed-loop comparison needs

Published comparisons hold the following controls fixed.

The oracle comparison is two columns that differ in one thing:

| oracle provider | solver |
|---|---|
| scaly generated C | one IPOPT build, one compiler, one optimization level, one C-side timer |
| CasADi generated C | the same |

Bounds, warm starts, tolerances, iteration limits, and input buffers are held fixed.
Both iteration counts and per-kernel call counts are compared. Two protocols are worth reporting separately:
a warm steady state where the solver object already exists, and a full lifecycle including
construction and teardown, since scaly's deployed path pays the latter on every step.

The comparison tests several CasADi encodings and reports the fastest that completes for each
problem and size, including cases where CasADi wins. A single encoding chosen for all problems
would miss the differences between their best representations.

Do not mix machines. Every number on this site comes from the machine described above. Absolute
times depend on the machine, so timings from different machines cannot establish provider speedups.

Python call overhead, build time, and generated source size are reported separately
from native function evaluation. They measure different costs of using and deploying
the generated model. The sweep's `coloring_width` is also implementation-specific:
Scaly reports the compressed tangent directions it executes, while CasADi leaves
the field blank. It cannot support a cross-backend comparison.

The sweep reports two sizes. `executable_bytes` is the total `size` reports for the compiled kernel
object: machine code, constant tables and static data, built by the same command as the timed kernel
and without the benchmark wrapper. `source_bytes` is the generated C translation unit. Report both,
with the constant data in each, `executable_constant_bytes` and `source_constant_bytes`: the same
tables weigh very differently in the two, from about three bytes of text and eight compiled for a
0/1 entry to one line of text and 281 kB compiled for CasADi's zero buffer.
These costs need their own rows and must not be folded into a speed-up.

## Reproduce the comparison

```bash
SCALY_VECTOR_LIBM=glibc SCALY_CC=gcc CC=gcc CXX=clang++ uv run benchmarks/run.py study --out-dir benchmarks/results/<study-name>
uv run benchmarks/run.py report benchmarks/results/<study-name>
```

The first command runs the frozen sweep grids and canonical closed loops. The second command rebuilds
the summary tables from the retained comma-separated value files and telemetry.

## Study coverage and reproduction

The October 13 study completed 138 planned sweep attempts and 15 closed-loop episodes. Of the
sweep attempts, 103 produced timings, 15 reached the 180-second compilation limit, and 20 were
skipped after that encoding failed at a smaller size. Every timed kernel passed its correctness
check, and every Scaly size completed its process. The sweep uses Clang 20.1.8, and the
closed-loop builds use GCC 13.3.0.

The study command runs the frozen grids with one fresh process, order seed 0, the `performance`
CPU governor, boost disabled, a 180-second compile limit, and a 50 MiB source limit. `--only sweep`
restricts it to the sweep grids. The report command regenerates the Markdown tables and the summary
CSV files from saved artifacts. Raw CSV files, generated code, samples, binaries, logs, and
provenance live below the study directory.

Sweep column meanings: mean is the single process's Google Benchmark timing. With more than one
successful process, CV is the sample coefficient of variation, otherwise it is blank.
Source bytes count the generated C file and executable bytes the compiled kernel object. Workspace
counts caller-owned doubles. Kernel compile time includes compilation of the generated kernel, but not the
benchmark wrapper or linker. `timeout` means compilation exceeded the limit;
`skipped_after_failure` means the attempt did not run after an earlier failure.

Closed-loop column meanings: the QP column records quadratic-program solution time for SQP and
internal solver time for the IPOPT cases that report it there. Globalization includes line-search
work. Glue is the wrapper's remaining work. Not every backend exposes the same breakdown, so the
parts do not always sum to the total.

### Episode agreement in the October 13 study

Every solver step succeeded. Mismatch counts cover the single episode pair for each controller.

| Problem | Solver | Repetitions | Per-step iteration mismatches | Per-step oracle-count mismatches | Maximum state difference | Maximum control difference |
|---|---|---:|---:|---:|---:|---:|
| chain | sqp | 1 | 0 | 0 | 3.53e-09 | 1.24e-08 |
| npmpc | ipopt | 1 | 0 | 0 | 1.21e-12 | 6.09e-15 |
| npmpc | sqp | 1 | 0 | 0 | 3.02e-09 | 1.46e-11 |
| race_cars | ipopt | 1 | 0 | 0 | 5.68e-14 | 5.19e-12 |
| race_cars | sqp | 1 | 0 | 0 | 4.83e-12 | 4.17e-09 |
| unbumpercars | ipopt | 1 | 0 | 0 | 9.08e-12 | 6.62e-12 |
| unbumpercars | sqp | 1 | 3 | 0 | 6.09e-09 | 4.91e-09 |
