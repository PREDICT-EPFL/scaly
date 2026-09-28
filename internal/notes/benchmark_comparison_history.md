# Benchmark comparison history

This note preserves the cross-study comparisons and configuration audits moved out of the public
results on 2026-09-24. The public results describe the latest complete study. Historical findings
below retain their original context and are not claims about the latest compiler.

The retained studies are `benchmarks/results/study-2026-09-23-c77-c79-final` for scalar libm and
`benchmarks/results/study-2026-09-23-c77-c79-libmvec` for vector libm. The September 10 study is the
comparison baseline used below. Actionable follow-up remains in [the todo list](../todo.md).

## Change from the scalar-libm studies

These Scaly measurements keep the earlier studies separate from the current policy. The September 10
study is the previously published baseline. The completed September 23 scalar-libm study includes
the compiler changes before the final short-loop lane cap. The final column includes that cap and
libmvec, so the difference between the last two columns is not a pure libmvec comparison.

| Problem | Size | September 10 scalar libm, µs | September 23 scalar libm, µs | September 23 libmvec, µs |
|---|---:|---:|---:|---:|
| Race-car | N=200 | 89.722 | 29.286 | 12.502 |
| Neural-process MPC | N=200 | 595.213 | 307.330 | 235.705 |
| Chain | M=9 | 550.460 | 282.777 | 282.649 |
| Unbumpercars | C=2 | 207.925 | 345.619 | 185.986 |
| Unbumpercars | C=32 | 6872.175 | 5361.081 | 6245.988 |

All final Scaly Hessian means improve on the previously published baseline. Unbumpercars C=32 is
16.5% slower than the intermediate scalar-libm study, although it remains 9.1% faster than the
published baseline. The final study also has closed-loop regressions, listed below. These results
do not establish a regression-free compiler change.

## Closed-loop changes

Against the previously published Scaly results, unbumpercars function evaluation is slower by 3.7%
with IPOPT, from 16.419 to 17.022 ms, and 5.6% with SQP, from 9.513 to 10.045 ms. Total time is
also slower, from 20.375 to 21.106 ms with IPOPT and from 10.786 to 11.331 ms with SQP.
Neural-process MPC IPOPT total time rises 1.5%, from 3.557 to 3.609 ms, although function evaluation
improves from 1.006 to 0.966 ms. Its total CV is 2.01%, so the total-time difference is smaller than
that run's process dispersion. The remaining Scaly closed-loop totals improve on the published baseline.

## Study artifacts and policy changes

The September 10 study and the completed September 23 scalar-libm study used
`SCALY_VECTOR_LIBM=none` and `vector_libm="none"`. Their artifacts remain separate from
`study-2026-09-23-c77-c79-libmvec`. The short-loop lane cap also changed between the latest
scalar and vector studies, so that comparison does not isolate libmvec's effect.

## Remaining regressions and limits

Publishing the complete measurements does not establish a no-regression result. Compared with
September 10, unbumpercars function evaluation is 3.7% slower with IPOPT and 5.6% slower with
SQP. Its total solver times are 3.6% and 5.1% slower. Neural-process MPC function evaluation
improves, but its IPOPT total increases by 1.5%; the non-oracle portion accounts for the increase.

The vector-policy study also has slower unbumpercars C=16 and C=32 Hessians than the preceding
scalar-policy study: 1,422.562 versus 1,280.988 µs and 6,245.988 versus 5,361.081 µs. These
remain faster than September 10. Race-car N=1 also rises from 0.319 to 0.360 µs between
these two recent runs. The two recent runs changed both math policy and short-loop
lane caps, so they do not establish that libmvec caused the slowdown. Large unbumpercars mapped
Jacobian and Hessian helpers contain substantial matrix arithmetic without transcendental calls.
Their register and scratch-memory costs need separate measurement before changing the width policy.

Race-car N=500 has a 10.73% coefficient of variation across five processes. Keep that dispersion
visible rather than selecting the fastest process. The complete sweep also reports source size,
workspace, and compilation cost, which can increase despite faster evaluation. These measurements
cover the reference x86 machine; they do not establish Apple M4 performance.

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

The historical flag ablation compiled both providers' kernels with
`-O3 -march=native -fno-math-errno`. The closed-loop CasADi IPOPT wrapper adds the same two
flags to the optimization level it shares with Scaly's JIT,
and the JIT applies them too. Scaly compiles on the machine it runs on, so the native target is the
deployment reality, and only binaries distributed to other machines, such as the solver plugin
wheels, stay at the portable x86-64 baseline. Both providers get identical flags, so the comparison
stays controlled, but the flags do not move both encodings equally, which is why the baseline is a
handicap and not a neutral choice. The portable target withholds fused multiply-add (FMA), and
clang's default `-ffp-contract=on` fuses only within one expression: Scaly renders compound
expressions and gets the contraction, while SX emits one operation per statement
(`a=(a*b); a=(a+c);`) and never forms an FMA. Race-car Hessian at N=50, µs, Scaly then SX: `-O3`
alone 32.9 and 21.3; `-mfma` alone 27.8 and 20.7; `-march=native` 26.9 and 20.7;
`-march=native -ffp-contract=off` 35.2 and 20.5. The gain is contraction, not vector width, and the
SX object in that historical experiment contains no FMA and no vector instruction at all.
The current vector-math study does emit vector calls in SX. `-fno-math-errno` lets `sqrt` and
the other libm calls inline instead of setting `errno` nothing reads: 31.8 against 33.4 for Scaly
and 20.6 against 21.3 for SX on the same cell. The completed study applies the native flags to both
providers. Its `.provenance.json` sidecars, `study.json`, and compile logs record the exact
commands.

## Controlled audit evidence

The experiments below isolate configuration variables that the complete study holds fixed. Their
absolute timings are audit measurements, not current benchmark headlines. The current study results
remain on the [overview](../../docs/benchmarks/index.md) and [scalability page](../../docs/benchmarks/scalability.md).

## The IPOPT build is a first-order confound

This finding undercuts the attribution of every pre-audit `ipopt+scaly` versus `ipopt+casadi` gap
in the suite.

The pre-audit columns did not run the same solver. Scaly's column loaded a locally built IPOPT
3.14.19 with MUMPS, METIS and OpenBLAS linked statically; CasADi calls the IPOPT 3.14.11 that ships
in its wheel, dynamically linked against `libcoinmumps`, `libcoinmetis` and `libcasadi-tp-openblas`.
The suite's own provenance records the CasADi version and the C compiler, and neither IPOPT version.

Scaly's generated wrapper carries `DT_NEEDED libipopt.so.3` and a `RUNPATH`, which `LD_LIBRARY_PATH`
overrides, so the same scaly oracles can be pointed at CasADi's IPOPT instead. Every row below is
the same generated C, the same warm starts and the same episode; only the `libipopt.so.3` the loader
resolves changes. Iteration counts are identical in every pair, and where the oracle timer is
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

Function evaluation is unchanged, as it must be, since the oracles are the same compiled C in both
rows. Everything else takes 2.7× longer on the smallest problem and 17× longer on the chain, where
the KKT system is largest and the linear solver dominates. Repeating the `npmpc` pair under
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` moves nothing, so this is not BLAS threading; it is a
property of the two builds.

Two caveats on that table. The chain row's FE also moves (3.70 to 5.56 ms), which it should not; on
a 96 ms solve that is most likely cache and frequency effects, and it is small next to the 90 ms it
sits beside. And `unbumpercars` is 93% function evaluation at C=8, so the confound is real there but
not where its claim lives.

The swap does not work in the other direction. CasADi's IPOPT plugin links IPOPT's C++ interface
built against the pre-C++11 `std::string` ABI, so preloading scaly's IPOPT fails to resolve
`Ipopt::StreamJournal`. Getting the fourth cell of the matrix needs the CasADi column
code-generated, which is possible.

## The CasADi column can be code-generated, all the way down

`ca.CodeGenerator().add(nlpsol_instance)` works for the IPOPT plugin. It emits self-contained C that
calls `IpStdCInterface.h` directly, with the oracles as generated C functions in the same
translation unit, and it creates and frees the IPOPT problem object per solve exactly as scaly's
wrapper does. Compiling it with scaly's compiler and flags, linking it against scaly's own
`libipopt.so`, and calling it through one `ctypes` call gives a CasADi column that shares scaly's
IPOPT build, its optimization level, its interface into IPOPT and its timer placement. A three-line
C shim wraps the generated entry point in `clock_gettime(CLOCK_MONOTONIC)`.

Two things are easy to get wrong here, and both produce wrong numbers that look right:

- CasADi's generated code uses the `res` array as scratch for nested calls, clobbering the leading
  output slots. Because `casadi_copy` skips a `NULL` destination without complaining, a `res` array
  set up once makes every call after the first return the first call's answer, with a success
  status. The output pointers have to be re-set on every call.
- A `nlpsol` cannot be code-generated and timed in the same process that built it. Constructing the
  `nlpsol` dlopens the wheel's `libipopt.so.3`, and every later `DT_NEEDED libipopt.so.3` resolves
  to that one regardless of the generated library's `RUNPATH`. Build in one process, time in another
  that only `ctypes`-loads the artifact.

### The controlled 2×2

`npmpc`, the canonical 100-step episode, both columns generated C behind one `ctypes` call, both
compiled by gcc 13.3 at `-O2`, both timed by `clock_gettime` inside C, identical bounds and warm
starts. State trajectories agree to 8.8e-13 across all four cells.

| oracle provider | IPOPT 3.14.19 (scaly's build) | IPOPT 3.14.11 (CasADi's wheel) |
|---|---:|---:|
| **scaly generated C** | **3.48 ms** | 7.04 ms |
| **CasADi generated C** | **3.93 ms** | 5.83 ms |

What the table says:

- Holding the IPOPT build fixed, scaly's oracles are worth 1.13× on total solve time (3.48 against
  3.93). That was the controlled oracle-provider result in this audit.
- Holding the oracle provider fixed, the IPOPT build is worth 2.02× for scaly's column and 1.48× for
  CasADi's, a larger effect than the thing being measured.
- The two effects are not additive: with CasADi's IPOPT, the CasADi column is faster than scaly's
  (5.83 against 7.04). Scaly's wrapper recreates the IPOPT problem and reapplies every option on
  each solve, and that costs more against the wheel build. This is not explained and should not be
  over-read; it is a reason to report the matrix and not one cell of it.
- The 1.13× here and the 1.22× from the already-fair `scaly-sqp` pair (1.29 against 1.57 ms) agree
  in magnitude, which they did not before. Two independent fair comparisons landing in the same
  place is the best evidence available that this is the real number.

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
| CasADi codegen-MX `-O2` (C timer) | 12.7 s | 5.56 ms | 6.48 | n/a | 5.60 ms | n/a |
| CasADi codegen-MX `-O3` (C timer) | 23.4 s | 5.15 ms | 6.83 | n/a | 5.18 ms | n/a |

The pre-audit column is the slowest of the six. Expanding to scalar SX is a trap on this problem: it
triples the interpreter's work, and compiled it costs a seventeen-minute build to land slower than
compiled MX. Compiled, CasADi's function evaluation (1.47 ms) is a wash against scaly's (1.49 ms),
which is what the kernel sweeps said, so the interpreted column's 8.6× gap was a configuration
artifact.

### `race_cars`, N=40, 149 steps of the canonical lap

| column | solve | p95 | FE | non-FE | iterations |
|---|---:|---:|---:|---:|---:|
| scaly (JIT `-O2`, C timer) | **4.60 ms** | 5.07 | 1.225 ms | 3.375 ms | 9.70 |
| CasADi interp-SX, pre-audit | 9.16 ms | 10.04 | 1.810 ms | 7.350 ms | 9.70 |
| CasADi interp-SX + `ca.cse` | 9.12 ms | 10.20 | 1.680 ms | 7.443 ms | 9.70 |
| CasADi interp-MX | 44.52 ms | 50.31 | 35.276 ms | 9.238 ms | 9.70 |
| CasADi jit-SX `-O3` | 7.95 ms | 8.72 | **0.398 ms** | 7.553 ms | 9.70 |
| CasADi jit-MX `-O3` | 8.03 ms | 8.72 | 0.643 ms | 7.385 ms | 9.70 |
| CasADi codegen-SX `-O2` (C timer) | 7.62 ms | 8.32 | n/a | n/a | n/a |
| CasADi codegen-MX `-O2` (C timer) | 8.08 ms | 9.11 | n/a | n/a | n/a |

Here `expand=True` is the right default by a wide margin. Interpreted MX is five times worse than
interpreted SX, because the MX evaluator's per-node overhead dominates a kernel this cheap. The
audit also found that once CasADi is compiled its function evaluation is 3.1× faster than scaly's,
0.398 against 1.225 ms. The sign of the oracle comparison reverses on this problem. Scaly's total is
still 1.7× better, but on this problem that is the IPOPT build, not the oracles.

`ca.cse` is worth 7% of function evaluation here and nothing on the total.

## Timers: what each column's number actually covers

Scaly's `t_total` is `clock_gettime` inside the generated C entry point, so it excludes the ctypes
call, the input coercion and the output allocation. The CasADi columns' `t_total` is
`time.perf_counter()` in Python around the SWIG call, so it includes the numpy to `DM` conversion of
every argument, and the difference is then booked as `t_solver`, which reads as IPOPT's time.

Measured on the `npmpc` episode, per step:

| column | Python-level | column's reported total | difference |
|---|---:|---:|---:|
| `ipopt+scaly` | 3.49 ms | 3.41 ms | 0.086 ms |
| `sqp+scaly` | 1.29 ms | 1.23 ms | 0.060 ms |
| `ipopt+casadi` | 17.81 ms | 17.43 ms | 0.380 ms |
| `sqp+casadi` | 1.51 ms | 1.45 ms | 0.060 ms |

Scaly's Python boundary costs 60 to 90 µs per solve, 2.5% of an IPOPT step and 5% of an SQP step.
For scaly, the columns are not measuring overhead.

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

- Per-solve IPOPT setup. Scaly's wrapper calls `CreateIpoptProblem`, every option setter and
  `FreeIpoptProblem` on every solve, because bounds are parameter-dependent, and all of it is
  inside its own timer. CasADi's `nlpsol` does that once at construction, outside every timer. Its
  generated C does it per solve, like scaly's, one more reason the code-generated column is the
  right comparison.
- `t_fe` is not the same quantity on both sides. Scaly's includes a bounds-evaluation kernel that
  CasADi has no analogue for; CasADi's is the sum of its `t_wall_nlp_*` callback timers, which may
  not cover the interface's sparse-matrix copies.
- Fused oracles. Scaly's `base` kernel computes `f` and `g` together and is called for both, so it
  evaluates `g` on every `f` call and vice versa, where CasADi evaluates them separately.

