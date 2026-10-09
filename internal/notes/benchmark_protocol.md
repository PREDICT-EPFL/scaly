# Benchmark protocol

Maintained. The full measurement protocol and the reasoning behind each rule, moved out of the
published [fairness page](../../docs/benchmarks/fairness.md), which keeps a short reader-facing
version. Rationale for what a comparison holds constant goes here; the matching task goes in
[GitHub Issues](https://github.com/PREDICT-EPFL/scaly/issues).

The figures on the published pages read `docs/assets/benchmarks/sweep.summary.csv` (the four
`sweep/<problem>/<problem>.summary.csv` files of a study, concatenated) and
`docs/assets/benchmarks/closed_loop.summary.csv` (written by `benchmarks/run.py report`). The files
currently there are stubs rebuilt from the tables of the September 23, 2026 study,
`benchmarks/results/study-2026-09-23-c77-c79-libmvec`, whose raw artifacts are gone. They carry
means, coefficients of variation, compile means and executable bytes only; replace them with a real
study's output.

## Reference machine

Every published timing comes from one machine.

| | |
|---|---|
| System | Minisforum F7BSC desktop |
| CPU | AMD Ryzen 9 7940HS, 8 cores and 16 threads, simultaneous multithreading enabled |
| Memory | 30 GB |
| Operating system | Ubuntu 24.04.4 LTS, kernel 7.0.0-28-generic, glibc 2.39 |
| Compilers | gcc 13.3.0 for Scaly just-in-time compilation, clang 20.1.8 for the Google Benchmark harness |
| Stack | Python 3.14.3, Scaly 0.1.0a1, CasADi 3.8.0, NumPy 2.4.6 |
| CPU policy | `amd-pstate-epp`, `performance` governor, boost disabled |

Absolute timings from another machine must not share these tables.

## Measurement protocol

The study uses the following controls:

- It runs one benchmark process at a time.
- It uses five fresh processes per cell or provider, with separate empty compilation caches.
- It rotates backend order from seed 0.
- It checks the `performance` governor and disabled boost state before and after each headline run.
- It compiles sweep kernels with `-O3 -march=native -fno-math-errno -fveclib=libmvec` with Clang.
- It records the compiler, flags, CPU policy, affinity, package versions, run order, and command in
  the artifact provenance.
- It stops a kernel compile after 180 seconds and skips generated source above 50 MiB.
- After a backend fails at one size, it skips that backend at larger sizes.
- It reports dispersion across successful processes and retains every failed attempt.

The study completed 690 sweep attempts. Of these, 520 produced timings, 15 reached the compilation
limit, and 155 were skipped after an earlier failure.

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

The benchmark rejects Clang, `zig cc` included, for measured just-in-time builds under this policy
because Scaly JIT does not pass its vector-library selection flag. A closed-loop command without a
CasADi oracle has no second provider to hold equal, so it accepts any compiler. Clang remains supported for sweep builds.
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

Executable-source and metadata sizes are not normalized between generators. Scaly emits several
sparsity views for consumers, while CasADi emits one compressed-column pattern. Use those columns to
understand the measured artifacts, not as a tool-independent measure of information content.

## Closed-loop comparisons

The SQP pairs use the same Scaly SQP implementation, PIQP subsolver, settings, warm starts, and
generated C application binary interface. Only the oracle provider changes. CasADi transforms each
generated oracle and supplies the same upper Hessian triangle as Scaly.

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
neural-process MPC, and race-car SQP pairs do too. Unbumpercars has five SQP iteration mismatches
across 1,000 paired steps, with no oracle-count mismatches and maximum state and control differences
below 5e-9. Its total-time ratio is an observed closed-loop result, not a strict equal-work
oracle-cost comparison.

## Reading the results

Kernel timings isolate derivative evaluation. Closed-loop totals include the shared solver work, so
a faster oracle may have little effect when the quadratic program or nonlinear solver dominates.
Race-car demonstrates this distinction: Scaly's SQP function evaluation is 4.2× faster than CasADi's,
while its total step is only 7% faster. The IPOPT total is 5% faster.

Compilation timeouts bound the claims. At unbumpercars C=16 and C=32, the study establishes that
Scaly completed within the fixed budget while the tested CasADi encodings did not. It does not assign
a runtime ratio to a missing kernel.

## Measurement limits

Race-car N=500 has a 10.73% coefficient of variation across five processes. The
[full sweep](../../docs/benchmarks/scalability.md) reports that dispersion alongside the mean, rather than selecting
the fastest process. The complete sweep also reports source size,
workspace, and compilation cost. These measurements cover the reference x86 machine and do not
establish Apple M4 performance.

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

## Why CasADi transformation and GEMM variants remain in the sweep

The sweep transforms every CasADi oracle before code generation. It also measures three general
matrix multiplication (GEMM) encodings under the same compile budget. CasADi's BLAS selector does
not propagate to matrix products created by automatic differentiation, so each encoding needs its
own measurement.

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

The executable count removes `static const` declarations from the C translation unit. Static
metadata contains those declarations and the generated header, so the two counts sum to the full C
and header artifact. The classification follows generated C syntax: index, seed, and numeric
constant arrays are metadata, while an inline numeric literal remains executable source. Scaly and
CasADi do not emit constants in the same form. Their sparse headers differ too: Scaly includes
coordinate, row-compressed, and column-compressed views for consumers, while CasADi emits one
column-compressed pattern. The metadata count is therefore the shipped source artifact, not a
normalized measure of sparsity information. Report both counts. These costs need their own rows and
must not be folded into a speed-up.

## Reproduce the comparison

```bash
SCALY_VECTOR_LIBM=glibc SCALY_CC=gcc CC=gcc CXX=clang++ uv run benchmarks/run.py study --out-dir benchmarks/results/<study-name>
uv run benchmarks/run.py report benchmarks/results/<study-name>
```

The first command runs the frozen sweep grids and canonical closed loops. The second command rebuilds
the summary tables from the retained comma-separated value files and telemetry.

## Study coverage and reproduction

The September 23 study completed 690 planned sweep attempts and 75 closed-loop episodes. Of the
sweep attempts, 520 produced timings, 15 reached the 180-second compilation limit, and 155 were
skipped after that encoding failed at a smaller size. Every timed kernel passed its correctness
check, and every Scaly size completed all five processes. The sweep uses Clang 20.1.8, and the
closed-loop builds use GCC 13.3.0.

The study command runs the frozen grids with five fresh processes, order seed 0, the `performance`
CPU governor, boost disabled, a 180-second compile limit, and a 50 MiB source limit. `--only sweep`
restricts it to the sweep grids. The report command regenerates the Markdown tables and the summary
CSV files from saved artifacts. Raw CSV files, generated code, samples, binaries, logs, and
provenance live below the study directory.

Sweep column meanings: mean is the mean of the five processes' Google Benchmark timings, and CV the
sample coefficient of variation. Executable bytes exclude static metadata. Workspace counts
caller-owned doubles. Kernel compile time includes compilation of the generated kernel, but not the
benchmark wrapper or linker. `timeout` means compilation exceeded the limit;
`skipped_after_failure` means the attempt did not run after an earlier failure.

Closed-loop column meanings: the QP column records quadratic-program solution time for SQP and
internal solver time for the IPOPT cases that report it there. Globalization includes line-search
work. Glue is the wrapper's remaining work. Not every backend exposes the same breakdown, so the
parts do not always sum to the total.

### Episode agreement in the September 23 study

Mismatch counts sum across all five episode pairs.

| Problem | Solver | Repetitions | Per-step iteration mismatches | Per-step oracle-count mismatches | Maximum state difference | Maximum control difference |
|---|---|---:|---:|---:|---:|---:|
| chain | sqp | 5 | 0 | 0 | 4.77e-09 | 1.6e-08 |
| npmpc | ipopt | 5 | 0 | 0 | 4.05e-13 | 1.99e-15 |
| npmpc | sqp | 5 | 0 | 0 | 6.64e-10 | 3.36e-12 |
| race_cars | ipopt | 5 | 0 | 0 | 9.24e-14 | 5.53e-12 |
| race_cars | sqp | 5 | 0 | 0 | 1.07e-11 | 6.66e-09 |
| unbumpercars | ipopt | 5 | 0 | 0 | 1.84e-12 | 1.35e-12 |
| unbumpercars | sqp | 5 | 5 | 0 | 4.58e-09 | 3.69e-09 |
