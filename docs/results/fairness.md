# Are the comparisons fair?

The study that started on 2026-09-10 fixes the known first-order confounds in the earlier benchmark
columns. Within each closed-loop pair, both providers now use the same solver build and compiled C
entry points. The sweep compiles every generated kernel with the same compiler and flags.

The comparison still has limits. CasADi has several encodings, compilation failures remove some
candidates at larger sizes, and the unbumpercars SQP providers do not take identical iterations.
This page states what the current results hold constant and where the numbers need qualification.

## Reference machine

Every published timing comes from one machine.

| | |
|---|---|
| System | Minisforum F7BSC desktop |
| CPU | AMD Ryzen 9 7940HS, 8 cores and 16 threads, simultaneous multithreading enabled |
| Memory | 30 GB |
| Operating system | Ubuntu 24.04.4 LTS, kernel 7.0.0-28-generic, glibc 2.39 |
| Compilers | gcc 13.3.0 for Scaly just-in-time compilation, clang 20.1.8 for the Google Benchmark harness |
| Stack | Python 3.14.3, Scaly 0.1.0, CasADi 3.8.0, NumPy 2.4.6 |
| CPU policy | `amd-pstate-epp`, `performance` governor, boost disabled |

Absolute timings from another machine must not share these tables.

## Measurement protocol

The study uses the following controls:

- It runs one benchmark process at a time.
- It uses five fresh processes per cell or provider, with separate empty compilation caches.
- It rotates backend order from seed 0.
- It checks the `performance` governor and disabled boost state before and after each headline run.
- It compiles sweep kernels with `-O3 -march=native -fno-math-errno`.
- It records the compiler, flags, CPU policy, affinity, package versions, run order, and command in
  the artifact provenance.
- It stops a kernel compile after 180 seconds and skips generated source above 50 MiB.
- After a backend fails at one size, it skips that backend at larger sizes.
- It reports dispersion across successful processes and retains every failed attempt.

The study completed 690 sweep attempts. Of these, 514 produced timings, 16 reached the compilation
limit, and 160 were skipped after an earlier failure.

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
bounds, options, warm starts, and compiled C boundary. This removes the solver-build, interpreted
CasADi, and Python-timer differences that affected older results. Chain has no IPOPT/CasADi runner,
so the results make no IPOPT provider comparison for that problem.

Native timers cover solver time per step. The telemetry separates function evaluation, quadratic
program solution for SQP, globalization, and wrapper work. Timing excludes controller construction,
Python dispatch, plant simulation, and recording. Each provider compiles its own wrapper, so total
time compares the complete generated solver path rather than one shared binary with a swapped
function pointer.

Every IPOPT provider pair has identical per-step iteration and oracle-call counts. The chain,
neural-process MPC, and race-car SQP pairs do too. Unbumpercars has ten SQP iteration mismatches
across 1,000 paired steps, with no oracle-count mismatches and maximum state and control differences
below 3e-7. Its total-time ratio is an observed closed-loop result, not a strict equal-work
oracle-cost comparison.

## Reading the results

Kernel timings isolate derivative evaluation. Closed-loop totals include the shared solver work, so
a faster oracle may have little effect when the quadratic program or nonlinear solver dominates.
Race-car demonstrates this distinction: Scaly's SQP function evaluation is 1.63× faster, while its
total step is only 4% faster. The IPOPT totals are equal within process dispersion.

Compilation timeouts bound the claims. At unbumpercars C=16 and C=32, the study establishes that
Scaly completed within the fixed budget while the tested CasADi encodings did not. It does not assign
a runtime ratio to a missing kernel.

## Why the CPU policy is fixed

The following audit experiment selected the CPU policy. It is not a headline benchmark result.

The harness protocol check on 2026-09-05 used `performance`, five fresh processes per backend,
and alternating backend order on the race-car Hessian at N=40. Google Benchmark ran for at least
0.5 seconds per cell. Raw rows and provenance are local, gitignored artifacts under
`benchmarks/results/protocol/boost_on/race.csv` and `benchmarks/results/protocol/boost_off/race.csv`.

| Backend | Boost-on mean, µs | Boost-off mean, µs | Boost-on coefficient of variation | Boost-off coefficient of variation |
|---|---:|---:|---:|---:|
| Scaly | 20.486 | 26.661 | 0.714% | 0.822% |
| CasADi mapped SX | 46.049 | 59.535 | 0.664% | 0.278% |

Boost remains disabled. It reduced the average coefficient of variation across these two columns,
though Scaly's dispersion increased slightly and both kernels became slower. Five samples on one
workload do not establish a general variance advantage. This check selects a machine configuration,
not a paper result. The completed study uses the selected configuration for every column.

## Why the native compiler flags are fair

The following flag ablation explains the shared compiler policy. It isolates compiler flags on one
kernel and does not replace the current sweep.

The sweep compiles both providers' kernels with
`-O3 -march=native -fno-math-errno`; the closed-loop CasADi IPOPT wrapper adds the same two flags
to the optimization level it shares with Scaly's JIT, and the JIT applies them too. Scaly compiles
on the machine it runs on, so the native target is the deployment reality, and only binaries
distributed to other machines, such as the solver plugin wheels, stay at the portable x86-64
baseline. Both providers get identical flags, so the comparison stays controlled, but the flags do
not move both encodings equally, which is why the baseline is a handicap rather than a neutral
choice. The portable target withholds fused multiply-add (FMA), and clang's default
`-ffp-contract=on` fuses only within one expression: Scaly renders compound expressions and gets
the contraction, while SX emits one operation per statement (`a=(a*b); a=(a+c);`) and never forms
an FMA. Race-car Hessian at N=50, µs, Scaly then SX: `-O3` alone 32.9 and 21.3; `-mfma` alone
27.8 and 20.7; `-march=native` 26.9 and 20.7; `-march=native -ffp-contract=off` 35.2 and 20.5. The
gain is contraction, not vector width, and the native SX object contains no FMA and no vector
instruction at all. `-fno-math-errno` lets `sqrt` and the other libm calls inline instead of
setting `errno` nothing reads: 31.8 against 33.4 for Scaly and 20.6 against 21.3 for SX on the
same cell. The completed study applies the native flags to both providers. Its
`.provenance.json` sidecars, `study.json`, and compile logs record the exact commands.

## Controlled audit evidence

The experiments below isolate configuration variables that the complete study holds fixed. Their
absolute timings are audit measurements, not current benchmark headlines. The current study results
remain on the [overview](index.md) and [scalability page](scalability.md).

## The IPOPT build is a first-order confound

This is the finding that matters most, because it undercuts the attribution of every
`ipopt+scaly` versus `ipopt+casadi` gap in the suite.

The pre-audit columns did not run the same solver. Scaly's column loaded a locally built **IPOPT 3.14.19** with
MUMPS, METIS and OpenBLAS linked statically; CasADi calls the **IPOPT 3.14.11** that ships in its
wheel, dynamically linked against `libcoinmumps`, `libcoinmetis` and `libcasadi-tp-openblas`. The
suite's own provenance records the CasADi version and the C compiler, and neither IPOPT version.

Scaly's generated wrapper carries `DT_NEEDED libipopt.so.3` and a `RUNPATH`, which `LD_LIBRARY_PATH`
overrides, so the same scaly oracles can be pointed at CasADi's IPOPT instead. Every row below is
the same generated C, the same warm starts and the same episode; only the `libipopt.so.3` the loader
resolves changes. **Iteration counts are identical in every pair**, and where the oracle timer is
comparable, so is function-evaluation time.

| problem | IPOPT build | solve | FE | non-FE | iterations |
|---|---|---:|---:|---:|---:|
| `npmpc` N=12, 100 steps | scaly 3.14.19 | 3.64 ms | 1.34 ms | **2.30 ms** | 9.99 |
| | CasADi 3.14.11 | 7.67 ms | 1.34 ms | **6.33 ms** | 9.99 |
| `chain` M=5 N=12, 60 steps | scaly 3.14.19 | 9.24 ms | 3.70 ms | **5.54 ms** | 8.05 |
| | CasADi 3.14.11 | 101.51 ms | 5.56 ms | **95.95 ms** | 8.05 |
| `race_cars` N=40, 150 steps | scaly 3.14.19 | 4.60 ms | 1.22 ms | **3.37 ms** | 9.70 |
| | CasADi 3.14.11 | 13.47 ms | 1.26 ms | **12.21 ms** | 9.70 |
| `unbumpercars` C=8, 40 steps | scaly 3.14.19 | 53.45 ms | 49.92 ms | **3.53 ms** | 14.10 |
| | CasADi 3.14.11 | 58.86 ms | 49.56 ms | **9.30 ms** | 14.10 |

Function evaluation is unchanged, as it must be — the oracles are the same compiled C in both rows.
Everything else takes 2.7× longer on the smallest problem and **17× longer on the chain**, where the
KKT system is largest and the linear solver dominates. Repeating the `npmpc` pair under
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` moves nothing, so this is not BLAS threading; it is a
property of the two builds.

Two caveats on that table. The chain row's FE also moves (3.70 → 5.56 ms), which it should not; on a
96 ms solve that is most likely cache and frequency effects rather than a real difference, and it is
small next to the 90 ms it sits beside. And `unbumpercars` is 93% function evaluation at C=8, so the
confound is real there but not where its claim lives.

The swap does not work in the other direction. CasADi's IPOPT plugin links IPOPT's **C++** interface
built against the pre-C++11 `std::string` ABI, so preloading scaly's IPOPT fails to resolve
`Ipopt::StreamJournal`. Getting the fourth cell of the matrix needs the CasADi column
code-generated — which, as it turns out, is possible.

## The CasADi column can be code-generated, all the way down

`ca.CodeGenerator().add(nlpsol_instance)` works for the IPOPT plugin. It emits self-contained C that
calls `IpStdCInterface.h` directly, with the oracles as generated C functions in the same
translation unit, and it creates and frees the IPOPT problem object per solve exactly as scaly's
wrapper does. Compiling it with scaly's compiler and flags, linking it against scaly's own
`libipopt.so`, and calling it through one `ctypes` call gives a CasADi column that shares scaly's
IPOPT build, its optimization level, its interface into IPOPT and its timer placement. A three-line
C shim wraps the generated entry point in `clock_gettime(CLOCK_MONOTONIC)`.

Two things are easy to get wrong here, and both produce wrong numbers that look right:

- **CasADi's generated code uses the `res` array as scratch for nested calls**, clobbering the
  leading output slots. Because `casadi_copy` skips a `NULL` destination without complaining, a
  `res` array set up once makes every call after the first return the *first* call's answer, with a
  success status. The output pointers have to be re-set on every call.
- **A `nlpsol` cannot be code-generated and timed in the same process that built it.** Constructing
  the `nlpsol` dlopens the wheel's `libipopt.so.3`, and every later `DT_NEEDED libipopt.so.3`
  resolves to that one regardless of the generated library's `RUNPATH`. Build in one process, time in
  another that only `ctypes`-loads the artifact.

### The controlled 2×2

`npmpc`, the canonical 100-step episode, both columns generated C behind one `ctypes` call, both
compiled by gcc 13.3 at `-O2`, both timed by `clock_gettime` inside C, identical bounds and warm
starts. State trajectories agree to 8.8e-13 across all four cells.

| oracle provider | IPOPT 3.14.19 (scaly's build) | IPOPT 3.14.11 (CasADi's wheel) |
|---|---:|---:|
| **scaly generated C** | **3.48 ms** | 7.04 ms |
| **CasADi generated C** | **3.93 ms** | 5.83 ms |

What the table says:

- Holding the IPOPT build fixed, **scaly's oracles are worth 1.13×** on total solve time (3.48
  against 3.93). That was the controlled oracle-provider result in this audit.
- Holding the oracle provider fixed, **the IPOPT build is worth 2.02× for scaly's column and 1.48×
  for CasADi's** — a larger effect than the thing being measured.
- The two effects are not additive: with CasADi's IPOPT, the CasADi column is *faster* than scaly's
  (5.83 against 7.04). Scaly's wrapper recreates the IPOPT problem and reapplies every option on
  each solve, and that costs more against the wheel build. This is not explained and should not be
  over-read; it is a reason to report the matrix rather than one cell of it.
- The 1.13× here and the 1.22× from the already-fair `scaly-sqp` pair (1.29 against 1.57 ms) agree in
  magnitude, which they did not before. Two independent fair comparisons landing in the same place is
  the best evidence available that this is the real number.

## `expand` and `jit`, per problem

Neither had been swept per problem. Both matter, and they do not point the same way.

The tables in this historical configuration study are retained for their closed-loop evidence. The
kernel sweep now keeps its encoding labels literal: `expand=True` for `casadi_sx`, and `False` for
`casadi_mx`, `casadi_call_mx` and `casadi_map_sx`.

### `npmpc`, canonical episode, 100 steps

| column | build | solve | p95 | FE | Python-level | iterations |
|---|---:|---:|---:|---:|---:|---:|
| scaly (JIT `-O2`, C timer) | 0.3 s | **3.77 ms** | 6.26 | 1.49 ms | 3.85 ms | 9.99 |
| CasADi interp-SX, pre-audit | 0.5 s | 17.01 ms | 19.94 | 12.87 ms | 17.32 ms | 9.99 |
| CasADi interp-MX | 0.02 s | 7.79 ms | 10.14 | 3.85 ms | 8.01 ms | 9.99 |
| CasADi jit-MX `-O3` | 23.9 s | 5.23 ms | 6.55 | 1.47 ms | 5.43 ms | 9.99 |
| CasADi jit-SX `-O3` | **1017 s** | 7.25 ms | 9.28 | 3.15 ms | 7.55 ms | 9.99 |
| CasADi codegen-MX `-O2` (C timer) | 12.7 s | 5.56 ms | 6.48 | — | 5.60 ms | — |
| CasADi codegen-MX `-O3` (C timer) | 23.4 s | 5.15 ms | 6.83 | — | 5.18 ms | — |

The pre-audit column is the slowest of the six. Expanding to scalar SX is a trap on this problem: it
triples the interpreter's work, and compiled it costs a seventeen-minute build to land *slower* than
compiled MX. Compiled, CasADi's function evaluation (1.47 ms) is a wash against scaly's (1.49 ms) —
which is exactly what the kernel sweeps said, so the interpreted column's 8.6× gap was a
configuration artifact, not a result.

### `race_cars`, N=40, 149 steps of the canonical lap

| column | solve | p95 | FE | non-FE | iterations |
|---|---:|---:|---:|---:|---:|
| scaly (JIT `-O2`, C timer) | **4.60 ms** | 5.07 | 1.225 ms | 3.375 ms | 9.70 |
| CasADi interp-SX, pre-audit | 9.16 ms | 10.04 | 1.810 ms | 7.350 ms | 9.70 |
| CasADi interp-SX + `ca.cse` | 9.12 ms | 10.20 | 1.680 ms | 7.443 ms | 9.70 |
| CasADi interp-MX | 44.52 ms | 50.31 | 35.276 ms | 9.238 ms | 9.70 |
| CasADi jit-SX `-O3` | 7.95 ms | 8.72 | **0.398 ms** | 7.553 ms | 9.70 |
| CasADi jit-MX `-O3` | 8.03 ms | 8.72 | 0.643 ms | 7.385 ms | 9.70 |
| CasADi codegen-SX `-O2` (C timer) | 7.62 ms | 8.32 | — | — | — |
| CasADi codegen-MX `-O2` (C timer) | 8.08 ms | 9.11 | — | — | — |

Here `expand=True` is the right default and it is not close: interpreted MX is five times worse than
interpreted SX, because the MX evaluator's per-node overhead dominates a kernel this cheap. The audit also found that **once CasADi is compiled its function evaluation is 3.1× faster than scaly's** — 0.398
against 1.225 ms. The sign of the oracle comparison reverses on this problem. Scaly's total is still
1.7× better, but on this problem that is the IPOPT build, not the oracles.

`ca.cse` is worth 7% of function evaluation here and nothing on the total.

## Timers: what each column's number actually covers

Scaly's `t_total` is `clock_gettime` inside the generated C entry point, so it excludes the ctypes
call, the input coercion and the output allocation. The CasADi columns' `t_total` is
`time.perf_counter()` in Python around the SWIG call, so it *includes* the numpy→`DM` conversion of
every argument — and the difference is then booked as `t_solver`, which reads as IPOPT's time.

Measured on the `npmpc` episode, per step:

| column | Python-level | column's reported total | difference |
|---|---:|---:|---:|
| `ipopt+scaly` | 3.49 ms | 3.41 ms | 0.086 ms |
| `sqp+scaly` | 1.29 ms | 1.23 ms | 0.060 ms |
| `ipopt+casadi` | 17.81 ms | 17.43 ms | 0.380 ms |
| `sqp+casadi` | 1.51 ms | 1.45 ms | 0.060 ms |

So scaly's Python boundary costs 60–90 µs per solve — 2.5% of an IPOPT step and 5% of an SQP step.
That is the answer to "are we measuring overhead": for scaly, no.

The CasADi IPOPT column now uses the stronger fix measured in the audit. The harness code-generates
the complete `nlpsol`, calls it through `ctypes`, and measures `clock_gettime` inside the C
entry point. Both columns now have the same kind of boundary.

## Why chain uses common-subexpression elimination

The chain audit isolated CasADi common-subexpression elimination (CSE) before `nlpsol`. At M=5,
unrolled SX generated 215,190 lines with CSE and 309,030 without it. The version without CSE did not
compile inside the ten-minute audit budget. On race-car, the same preprocessing changed function
evaluation by 7% and did not change total solve time. The harness therefore applies `ca.cse` to the
chain nonlinear-program expressions only. This is a measured formulation choice, not a global
advantage assigned to either provider.

## Why CasADi transformation and GEMM variants remain in the sweep

The CasADi 3.8 audit isolated the passes behind `Function.transform()`. On the race-car SX Hessian
at N=25, CSE changed 15.0 µs to 8.2 µs. The other tested passes did not change the result. Generated
source fell by 15% to 30% across the audit cells. The current sweep therefore transforms every
CasADi oracle before code generation.

CasADi's BLAS selector does not propagate to matrix products created by automatic differentiation.
The audit also found that batching repeated network products into one general matrix multiplication
(GEMM) made the tested Hessians slower, although it let one larger NPMPC graph compile. The sweep
keeps the three GEMM encodings as recorded alternatives and reports them under the same compile
budget.

## Where the suite handicaps Scaly

- **Per-solve IPOPT setup.** Scaly's wrapper calls `CreateIpoptProblem`, every option setter and
  `FreeIpoptProblem` on *every* solve, because bounds are parameter-dependent, and all of it is
  inside its own timer. CasADi's `nlpsol` does that once at construction, outside every timer. (Its
  *generated* C does it per solve, like scaly's — one more reason the code-generated column is the
  right comparison.)
- **`t_fe` is not the same quantity on both sides.** Scaly's includes a bounds-evaluation kernel
  that CasADi has no analogue for; CasADi's is the sum of its `t_wall_nlp_*` callback timers, which
  may not cover the interface's sparse-matrix copies.
- **Fused oracles.** Scaly's `base` kernel computes `f` and `g` together and is called for both, so
  it evaluates `g` on every `f` call and vice versa, where CasADi evaluates them separately.

## What a fair closed-loop comparison needs

The rules this page argues for, collected. They are the admissibility test a number has to pass
before it appears anywhere else on this site.

The oracle comparison is two columns that differ in one thing:

| oracle provider | solver |
|---|---|
| scaly generated C | one IPOPT build, one compiler, one optimization level, one C-side timer |
| CasADi generated C | the same |

with bounds, warm starts, tolerances, iteration limits and input buffers held fixed, and with
iteration counts *and* per-kernel call counts compared rather than iteration counts alone. Two
protocols are worth reporting separately: a warm steady state where the solver object already exists,
and a full lifecycle including construction and teardown, since scaly's deployed path pays the latter
on every step.

**Report CasADi's best encoding for that problem, and name the cells CasADi wins.** No comparison may
be made against a formulation chosen on CasADi's behalf. The encodings of the same math span an order
of magnitude and the winner changes between problems, so the best one has to be found per problem
rather than nominated once.

**Do not mix machines.** Every number on this site comes from the machine described above. Absolute
times differ by roughly 2x against an Apple M-series, so a mixed table invents a result.

Alongside the oracle comparison, three numbers that are *not* it and must not be presented as if they
were: the Python-level cost of a step, which is what an application actually pays; the build cost of
the oracle set; and the generated artifact size, split into executable source and static metadata.
The sweep's `coloring_width` is also implementation-specific: Scaly reports the compressed tangent
directions it executes, while CasADi leaves the field blank. Do not use it as a cross-backend
comparison.
The executable count removes `static const` declarations from the C translation unit. Static
metadata contains those declarations and the generated header, so the two counts sum to the full C
and header artifact. The classification follows generated C syntax: index, seed, and numeric
constant arrays are metadata, while an inline numeric literal remains executable source. Scaly and
CasADi do not emit constants in the same form. Their sparse headers differ too: Scaly includes
coordinate, row-compressed, and column-compressed views for consumers, while CasADi emits one
column-compressed pattern. The metadata count is therefore the shipped source artifact, not a
normalized measure of sparsity information. Report both counts. These costs need their own rows
rather than being folded into a speed-up.

## Reproduce the comparison

```bash
uv run benchmarks/run.py study --out-dir benchmarks/results/<study-name>
uv run benchmarks/run.py report benchmarks/results/<study-name>
```

The first command runs the frozen sweep grids and canonical closed loops. The second command rebuilds
the summary tables from the retained comma-separated value files and telemetry.
