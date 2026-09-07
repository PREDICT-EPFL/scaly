# Are the comparisons fair?

Every number on the [overview](index.md) and in the [scalability sweep](scalability.md) is a
comparison, and a comparison is worth exactly as much as the weakest thing it holds constant. This
page is the audit of what the suite actually holds constant, what it does not, and which published
claims changed once each defect was measured rather than assumed.

The answer cuts both ways. Some columns were handing CasADi a configuration nobody would deploy.
Others were handing alloy one. And the largest single effect on the headline closed-loop numbers was
neither library's fault. The two columns were linking two different builds of IPOPT.

Everything here was measured on the reference machine described below, serially, with compile jobs
kept out of the timed runs.

## The reference machine

Every published timing comes from this machine. Numbers from anywhere else are not comparable and
must not share a table with these: absolute times differ by roughly 2x against an Apple M-series, and
the race-car cell that reads 10.27 us in an older table measures 19.47 us here.

| | |
|---|---|
| System | Minisforum F7BSC (`Micro Computer (HK) Tech Limited`), desktop chassis, no battery |
| CPU | AMD Ryzen 9 7940HS, 8 cores / 16 threads, SMT on, 0.40-5.26 GHz, 16 MiB L3 |
| Memory | 30 GB |
| OS | Ubuntu 24.04.4 LTS, kernel 7.0.0-28-generic, glibc 2.39 |
| Compilers | gcc 13.3.0 (alloy's JIT), clang 20.1.8 (the Google Benchmark harness) |
| Stack | Python 3.14.3, CasADi 3.8.0 (3.7.2 for every measurement above the CasADi 3.8 section; the upgrade alone moved nothing), NumPy 2.4.6, SciPy 1.18.0 |
| Frequency during the audit | `amd-pstate-epp` driver, `powersave` governor, boost enabled |

**That last row is the one to fix before any headline run.** With `powersave` and boost on, the clock
moves between 0.40 and 5.26 GHz according to load and package temperature, and it shows: one
unbumpercars figure moved from 52.7 to 58.9 ms between two runs of the same episode minutes apart,
about 12%, which is larger than several of the effects on this page. Only `performance` and
`powersave` are available with this driver, so the headline protocol should pin `performance`, and
disabling boost is worth testing for dispersion even at a lower absolute clock. Neither was set during that audit,
so **its numbers are pilot data**: the ratios are informative, the absolute values and
anything under about 15% are not yet.

The harness protocol check on 2026-09-05 used `performance`, five fresh processes per backend,
and alternating backend order on the race-car Hessian at N=40. Google Benchmark ran for at least
0.5 seconds per cell. Raw rows and provenance are local, gitignored artifacts under
`benchmarks/results/protocol/boost_on/race.csv` and `benchmarks/results/protocol/boost_off/race.csv`.

| Backend | Boost-on mean, µs | Boost-off mean, µs | Boost-on coefficient of variation | Boost-off coefficient of variation |
|---|---:|---:|---:|---:|
| Alloy | 20.486 | 26.661 | 0.714% | 0.822% |
| CasADi mapped SX | 46.049 | 59.535 | 0.664% | 0.278% |

Boost remains disabled. It reduced the average coefficient of variation across these two columns,
though Alloy's dispersion increased slightly and both kernels became slower. Five samples on one
workload do not establish a general variance advantage. This check selects a machine configuration,
not a paper result. The pilot must use the same recorded configuration for every column.

The [initial Hessian scalability pilot](scalability.md#initial-frozen-protocol-pilot-2026-09-05)
uses this selected configuration with five fresh processes per cell. Its tables, dispersion, and
compile failures are recorded on the scalability page. Those measurements use synthetic inputs
and do not measure closed-loop solver performance.

## The measurement protocol

Rules the numbers on this page follow, and that a headline run must follow more strictly.

- **One thing at a time.** No compile job, sweep or second benchmark running during a timed run.
  Several early figures in this audit had to be thrown away because three compilers were competing
  for the cores.
- **Kernel figures come from the Google Benchmark harness**, which calls the generated C symbol with
  preallocated buffers and no Python in the loop. Both closed-loop IPOPT columns now use
  `clock_gettime` around a generated C entry point.
- **Request only the oracle result that is timed.** CasADi's generated `nlp_jac_g` exposes `g` as
  output 0 and `jac_g_x` as output 1; the harness supplies only result pointer 1. The Hessian
  supplies its selected output, and every result-pointer array is reset before each call because
  generated `res` slots are also used as scratch by nested calls. No parameter-sensitivity result
  is requested by either timed kernel.
- **Record the kernel input policy.** Canonical cells consume harvested closed-loop inputs when
  available; otherwise they use synthetic samples shared by the compared encodings. A kernel
  sample must permit finite derivative evaluation, but need not be a feasible controller initial
  state. Unbumpercars synthetic kernels use seeded arena states without collision rejection:
  the closed-loop placement rule cannot fit the larger scalability counts in the default arena.
  A problem gate checks finite Hessian reference values at C=16 and C=32. The original
  collision-free-input pilot remains separate from the full-grid rerun under this policy.
- **Correctness gates before timing.** Every cell checks its compact derivative against a dense
  reference built by a different construction, and produces no timing if it disagrees. The race-car
  Jacobian and the chain, race-car and neural-process MPC Lagrangian Hessians have NumPy references:
  complex-step first derivatives and exact hyper-dual second derivatives, assembled stage by stage.
  The chain and neural-process MPC Jacobian references are Alloy's dense per-stage Jacobians of an
  unrolled twin, so they test the mapped construction against an unrolled one rather than against
  another tool. The unbumpercars Hessian reference is a CasADi `MX` Hessian of the same
  formulation, which means that cell's CasADi column is checked against its own tool.
- **`ca.cse` is applied to the chain NLP expressions before `nlpsol`, and nowhere else.** On the
  chain it is what keeps the unrolled encodings buildable at all: without it unrolled `SX` is
  309 030 lines at M=5 against 215 190 with it (§"The chain sweep"). A focused build confirms that
  the preprocessing reaches the extracted `nlpsol` oracle; the current sweep records that oracle's
  generated artifact, while the line counts in the historical chain table below describe the older
  hand-built mirror. On `race_cars` it was worth 7% of function evaluation and nothing on the total
  (§"`expand` and `jit`"); it remains disabled for `npmpc` and `unbumpercars`. This is a measured
  per-problem choice, not a suite-wide rule.
- **Report CasADi's best encoding**, not the one our mirror happens to build. §"Does CasADi have
  loop-preserving codegen?" is why: the encodings of the same math span an order of magnitude, and
  the winner changes between problems.
- The sweep now defaults to three fresh processes per cell and varies backend order with a recorded
  seed. Raw rows retain every attempt, and `.summary.csv` reports dispersion across successful runs.
  Generated-code compilation, wrapper compilation, and linking have separate columns.
  `--headline --boost on|off` requires at least three repetitions and checks the performance governor
  and chosen boost state before and after each cell. Provenance records frequency policies and CPU
  affinity. These controls do not retroactively validate the historical measurements on this page.
- Closed-loop `--repetitions` uses separate processes and empty compilation caches. `modes.csv`
  derives startup and per-step wall costs from recorded construction and solve calls. `jit` includes
  construction, including standalone result-check oracles, while `prebuilt` excludes construction
  and loading. Steady cost excludes the first
  solve. `--casadi-interpreted` supplies the third CasADi mode using the wheel's IPOPT and identifies
  that provider explicitly, so that row cannot support a controlled oracle-cost comparison.

## The short version

| defect | direction | effect where measured | status |
|---|---|---|---|
| The two IPOPT columns linked **different IPOPT builds** | favoured alloy | **2.7× to 17×** in solver-internal time, at identical iteration counts | measured, fixed |
| `ipopt+casadi` ran CasADi's **virtual machine**, not compiled C | favoured alloy | 1.2× to 8.6× in function evaluation | measured, fixed |
| CasADi `expand` ignored the encoding | varied | literal SX needs `True`; MX, called MX and mapped SX need `False` | measured, fixed |
| CasADi's total was timed in **Python**, alloy's in **C** | favoured alloy | 0.2–0.4 ms/step, and booked as *solver* time | measured, fixed |
| The chain sweep's `casadi_sx` column was **not** an SX column | mislabelled, not unfair | forcing `map` is what makes CasADi compilable here at all | measured, fixed |
| Alloy **loses** the chain equality Jacobian on runtime | favours CasADi | 2.4× at M=5, 1.5× at M=17 — and unpublished | measured |
| CasADi `ca.cse` policy was implicit | favours alloy | negligible on `race_cars` (1.81 → 1.68 ms FE) | measured, fixed: chain pre-`nlpsol` only |
| Alloy's JIT compiles at `-O2`, the sweep at `-O3` | favours CasADi | ≤6% on function evaluation | measured, minor |
| Alloy recreates the IPOPT problem object every solve, inside its own timer | favours CasADi | not isolated | known |
| Alloy's `t_fe` includes a bounds kernel CasADi has no analogue for | favours CasADi | small | known |

**The one-line conclusion.** With every controllable difference removed on the one problem where the
fully controlled comparison was built, alloy's oracle advantage over CasADi is **1.13×** on total
solve time, not the 4.5× the shipped column reports. The rest was the IPOPT build and the
interpreter.

## The IPOPT build is a first-order confound

This is the finding that matters most, because it undercuts the attribution of every
`ipopt+alloy` versus `ipopt+casadi` gap in the suite.

The two columns do not run the same solver. Alloy dlopens a locally built **IPOPT 3.14.19** with
MUMPS, METIS and OpenBLAS linked statically; CasADi calls the **IPOPT 3.14.11** that ships in its
wheel, dynamically linked against `libcoinmumps`, `libcoinmetis` and `libcasadi-tp-openblas`. The
suite's own provenance records the CasADi version and the C compiler, and neither IPOPT version.

Alloy's generated wrapper carries `DT_NEEDED libipopt.so.3` and a `RUNPATH`, which `LD_LIBRARY_PATH`
overrides, so the same alloy oracles can be pointed at CasADi's IPOPT instead. Every row below is
the same generated C, the same warm starts and the same episode; only the `libipopt.so.3` the loader
resolves changes. **Iteration counts are identical in every pair**, and where the oracle timer is
comparable, so is function-evaluation time.

| problem | IPOPT build | solve | FE | non-FE | iterations |
|---|---|---:|---:|---:|---:|
| `npmpc` N=12, 100 steps | alloy 3.14.19 | 3.64 ms | 1.34 ms | **2.30 ms** | 9.99 |
| | CasADi 3.14.11 | 7.67 ms | 1.34 ms | **6.33 ms** | 9.99 |
| `chain` M=5 N=12, 60 steps | alloy 3.14.19 | 9.24 ms | 3.70 ms | **5.54 ms** | 8.05 |
| | CasADi 3.14.11 | 101.51 ms | 5.56 ms | **95.95 ms** | 8.05 |
| `race_cars` N=40, 150 steps | alloy 3.14.19 | 4.60 ms | 1.22 ms | **3.37 ms** | 9.70 |
| | CasADi 3.14.11 | 13.47 ms | 1.26 ms | **12.21 ms** | 9.70 |
| `unbumpercars` C=8, 40 steps | alloy 3.14.19 | 53.45 ms | 49.92 ms | **3.53 ms** | 14.10 |
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
built against the pre-C++11 `std::string` ABI, so preloading alloy's IPOPT fails to resolve
`Ipopt::StreamJournal`. Getting the fourth cell of the matrix needs the CasADi column
code-generated — which, as it turns out, is possible.

## The CasADi column can be code-generated, all the way down

`ca.CodeGenerator().add(nlpsol_instance)` works for the IPOPT plugin. It emits self-contained C that
calls `IpStdCInterface.h` directly, with the oracles as generated C functions in the same
translation unit, and it creates and frees the IPOPT problem object per solve exactly as alloy's
wrapper does. Compiling it with alloy's compiler and flags, linking it against alloy's own
`libipopt.so`, and calling it through one `ctypes` call gives a CasADi column that shares alloy's
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

| oracle provider | IPOPT 3.14.19 (alloy's build) | IPOPT 3.14.11 (CasADi's wheel) |
|---|---:|---:|
| **alloy generated C** | **3.48 ms** | 7.04 ms |
| **CasADi generated C** | **3.93 ms** | 5.83 ms |

What the table says:

- Holding the IPOPT build fixed, **alloy's oracles are worth 1.13×** on total solve time (3.48
  against 3.93). That is the honest oracle-provider result for this problem.
- Holding the oracle provider fixed, **the IPOPT build is worth 2.02× for alloy's column and 1.48×
  for CasADi's** — a larger effect than the thing being measured.
- The two effects are not additive: with CasADi's IPOPT, the CasADi column is *faster* than alloy's
  (5.83 against 7.04). Alloy's wrapper recreates the IPOPT problem and reapplies every option on
  each solve, and that costs more against the wheel build. This is not explained and should not be
  over-read; it is a reason to report the matrix rather than one cell of it.
- The 1.13× here and the 1.22× from the already-fair `alloy-sqp` pair (1.29 against 1.57 ms) agree in
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
| alloy (JIT `-O2`, C timer) | 0.3 s | **3.77 ms** | 6.26 | 1.49 ms | 3.85 ms | 9.99 |
| CasADi interp-SX — *shipped* | 0.5 s | 17.01 ms | 19.94 | 12.87 ms | 17.32 ms | 9.99 |
| CasADi interp-MX | 0.02 s | 7.79 ms | 10.14 | 3.85 ms | 8.01 ms | 9.99 |
| CasADi jit-MX `-O3` | 23.9 s | 5.23 ms | 6.55 | 1.47 ms | 5.43 ms | 9.99 |
| CasADi jit-SX `-O3` | **1017 s** | 7.25 ms | 9.28 | 3.15 ms | 7.55 ms | 9.99 |
| CasADi codegen-MX `-O2` (C timer) | 12.7 s | 5.56 ms | 6.48 | — | 5.60 ms | — |
| CasADi codegen-MX `-O3` (C timer) | 23.4 s | 5.15 ms | 6.83 | — | 5.18 ms | — |

The shipped column is the worst of the six. Expanding to scalar SX is a trap on this problem: it
triples the interpreter's work, and compiled it costs a seventeen-minute build to land *slower* than
compiled MX. Compiled, CasADi's function evaluation (1.47 ms) is a wash against alloy's (1.49 ms) —
which is exactly what the kernel sweeps said, so the interpreted column's 8.6× gap was a
configuration artifact, not a result.

### `race_cars`, N=40, 149 steps of the canonical lap

| column | solve | p95 | FE | non-FE | iterations |
|---|---:|---:|---:|---:|---:|
| alloy (JIT `-O2`, C timer) | **4.60 ms** | 5.07 | 1.225 ms | 3.375 ms | 9.70 |
| CasADi interp-SX — *shipped* | 9.16 ms | 10.04 | 1.810 ms | 7.350 ms | 9.70 |
| CasADi interp-SX + `ca.cse` | 9.12 ms | 10.20 | 1.680 ms | 7.443 ms | 9.70 |
| CasADi interp-MX | 44.52 ms | 50.31 | 35.276 ms | 9.238 ms | 9.70 |
| CasADi jit-SX `-O3` | 7.95 ms | 8.72 | **0.398 ms** | 7.553 ms | 9.70 |
| CasADi jit-MX `-O3` | 8.03 ms | 8.72 | 0.643 ms | 7.385 ms | 9.70 |
| CasADi codegen-SX `-O2` (C timer) | 7.62 ms | 8.32 | — | — | — |
| CasADi codegen-MX `-O2` (C timer) | 8.08 ms | 9.11 | — | — | — |

Here `expand=True` is the right default and it is not close: interpreted MX is five times worse than
interpreted SX, because the MX evaluator's per-node overhead dominates a kernel this cheap. And the
headline: **once CasADi is compiled its function evaluation is 3.1× faster than alloy's** — 0.398
against 1.225 ms. The sign of the oracle comparison reverses on this problem. Alloy's total is still
1.7× better, but on this problem that is the IPOPT build, not the oracles.

`ca.cse` is worth 7% of function evaluation here and nothing on the total.

### `unbumpercars`, C=8, 40 steps, exact Lagrangian Hessian

This is the column behind the published "4.5–9.0×", and it behaves differently from the other two:
at C=8 the solve is **93% function evaluation**, so the IPOPT-build confound barely matters and the
oracle configuration is nearly the whole story.

| column | solve | p95 | FE | non-FE | Python-level | build |
|---|---:|---:|---:|---:|---:|---:|
| alloy (JIT `-O2`, C timer) | **58.9 ms** | 93.9 | 52.3 ms | 6.6 ms | 59.2 ms | 2.4 s |
| CasADi interp-SX — *shipped* | 300.6 ms | 479.5 | 283.7 ms | 16.9 ms | 320.9 ms | ~10 s |
| CasADi interp-MX | 160.8 ms | 252.0 | 146.6 ms | 14.1 ms | 181.0 ms | ~1 s |
| CasADi jit-MX `-O3` | 146.2 ms | 231.4 | 136.0 ms | 10.2 ms | 164.0 ms | 57 s |

Three things fall out of that table.

**`expand=True` is the wrong default here and costs CasADi 1.87×** — 300.6 against 160.8 ms. It is the
right default on `race_cars` and the wrong one on both problems with a matmul in the stage, which is
the pattern: the MX evaluator's per-node overhead only wins on kernels too cheap to amortize it.

**Compiling CasADi's oracles is worth only 1.08× on function evaluation here** (146.6 → 136.0 ms, for
a 57-second build), because an MX graph over matmuls is already dispatching into compiled block
kernels. There is little interpreter left to remove. On the other two problems the same switch is
worth 4.1× (`npmpc` SX), 2.6× (`npmpc` MX), 4.6× (`race_cars` SX) and 55× (`race_cars` MX, where an
interpreted graph over four-state scalar stages is pathological).

**So the corrected multiple is 2.48×, not 4.5–9.0×** — alloy against the best CasADi configuration
rather than against the shipped one. And it is corroborated independently: the
[isolated exact Lagrangian Hessian](scalability.md#isolated-exact-lagrangian-hessian-sphessgammazz-current),
compiled on both sides at `-O3` and timed in Google Benchmark with no Python anywhere, puts alloy
1.16× / 1.77× / **3.03×** ahead at C = 2 / 4 / 8. Two independent measurements, same magnitude,
growing in the car count.

This is the suite's strongest oracle result, and it survived the audit essentially intact — it was
just being quoted about twice too high.

One methodological trap worth recording, because it produced a wrong number that looked reasonable.
Rebuilding the NLP by composing this problem's instrumentation `ca.Function`s into a new oracle hands
`nlpsol` a *nested-call* MX graph rather than the flat expression. CasADi's codegen of the derived
Hessian then explodes — 65 MB of C against 3.7 MB for the same oracles built from the expression —
and interpreted MX measures 1578 ms instead of 161. The configuration probe has to intercept the real
`ca.nlpsol` call, not reconstruct its argument.

For reference, the flat MX oracle set at C=8 is 3.69 MB of generated C across five kernels (the exact
Lagrangian Hessian alone is 1.96 MB / 62 426 lines with a 606 258-double workspace), against alloy's
704 KB / 15 790 lines for its whole generated solver including the IPOPT driver.

## Timers: what each column's number actually covers

Alloy's `t_total` is `clock_gettime` inside the generated C entry point, so it excludes the ctypes
call, the input coercion and the output allocation. The CasADi columns' `t_total` is
`time.perf_counter()` in Python around the SWIG call, so it *includes* the numpy→`DM` conversion of
every argument — and the difference is then booked as `t_solver`, which reads as IPOPT's time.

Measured on the `npmpc` episode, per step:

| column | Python-level | column's reported total | difference |
|---|---:|---:|---:|
| `ipopt+alloy` | 3.49 ms | 3.41 ms | 0.086 ms |
| `sqp+alloy` | 1.29 ms | 1.23 ms | 0.060 ms |
| `ipopt+casadi` | 17.81 ms | 17.43 ms | 0.380 ms |
| `sqp+casadi` | 1.51 ms | 1.45 ms | 0.060 ms |

So alloy's Python boundary costs 60–90 µs per solve — 2.5% of an IPOPT step and 5% of an SQP step.
That is the answer to "are we measuring overhead": for alloy, no.

The CasADi IPOPT column now uses the stronger fix measured in the audit. The harness code-generates
the complete `nlpsol`, calls it through `ctypes`, and measures `clock_gettime` inside the C
entry point. Both columns now have the same kind of boundary.

## Does CasADi have loop-preserving codegen? Partly

The code-size claim is the strongest one alloy makes, and it compares alloy's VMAP output against
CasADi mirrors that unroll the horizon with a Python `for` loop. CasADi does have `Function.map`, which is meant to preserve exactly that structure, so the claim is
only worth making against it.

The answer is problem-dependent, which makes it more interesting than either extreme.

**On cheap scalar stages, `map` is a real mechanism for MX.** On the race-car equality Jacobian
([measured earlier](scalability.md#comparison-with-casadi-functionmapn-serial-tracking)), SX with
`.map` is byte-for-byte identical to unrolling, because SX flattens to scalars before codegen and the
abstraction does not survive graph construction. MX with `.map` does keep a loop and is the most
source-efficient CasADi configuration available — 317 KB at N=100 against 928 KB unrolled — but its
workspace grows linearly (80 506 doubles at N=200) and it runs about 2× slower than unrolled SX.

**On a dense-network stage, `map` is a pessimization.** Rebuilding the `npmpc` stage residual as its
own `ca.Function` and mapping it over the horizon makes the generated sparse Jacobian *larger* than
unrolling, by an order of magnitude, and it still grows linearly:

| N | MX unrolled | MX with `.map` | SX unrolled | alloy |
|---:|---:|---:|---:|---:|
| 6 | 2 470 lines / 90 KB / **16.0 µs** | 18 598 / 1 261 KB / 108.2 µs | 80 110 / 1 857 KB / 25.3 µs | 417 lines |
| 12 | 4 293 / 164 KB / **28.6 µs** | 35 494 / 2 547 KB / 204.3 µs | 158 645 / 4 132 KB / 45.8 µs | 417 |
| 25 | 8 242 / 323 KB / **61.3 µs** | 72 101 / 5 474 KB / 510.1 µs | 328 806 / 8 550 KB / 92.5 µs | 417 |
| 50 | 15 836 / 630 KB | 142 500 / 11 108 KB | — | 417 |

Both CasADi variants produce the same nonzero count (126/252/525/1050) and agree exactly, so this is
a codegen difference and not a formulation one. Mapping costs 6.8× to 8.3× in evaluation time on top
of 7.5× to 9× in source size. (Runtimes here are through `ca.external`, so they carry CasADi's
dispatch overhead; the ratios between the three columns are what to read, not the absolute values.)

The defensible claim is therefore narrower than "alloy's loop-preserving lowering beats CasADi on
code size", and it is still strong: **across every CasADi encoding tested — SX unrolled, SX mapped,
MX unrolled, MX mapped — generated source grows with the horizon, and alloy's does not.** Encodings
not yet tested (`map`'s `"inline"` and `"unroll"` modes, `mapaccum`/`fold` for recurrences) are the
remaining gap in that sentence, and until they are tested the claim should carry "of the encodings
tested".

## The chain sweep: a suspected handicap that was not one, and a result that is not published

At the time of this audit, the chain sweep passed `map_stages=True` for both CasADi cells. Because
`_ca_eq` forces `z` and `p` to `ca.MX` in that mode, neither label described its outer graph. The
harness now records the unrolled scalar and matrix graphs as `casadi_sx` and `casadi_mx`. It records
the called and serially mapped elemental functions as `casadi_call_mx` and `casadi_map_sx`.

The source-size rows below are the historical hand-built mirror. The current sweep builds the full
`nlpsol` first, applies the chain `ca.cse` preprocessing to its `(f, g)` expressions, and extracts
`nlp_hess_l` or `nlp_jac_g`, so its artifact and pointer metadata describe the oracle the optimizer
actually calls.

The obvious next inference — that forcing `map` is therefore a handicap — is wrong, and it is worth
recording why, because a Python-level probe says it is a 9× handicap and a compiled one says the
opposite. Every variant below is the same kernel through the same Google Benchmark harness, clang 20
at `-O3`, one variant at a time on an otherwise idle machine. Chain M=5, N=40:

| variant | lines | source | workspace | compile | runtime |
|---|---:|---:|---:|---:|---:|
| **alloy** | **583** | **124 KB** | 20 160 | **0.87 s** | 587 µs |
| CasADi SX + `map`, *audit configuration* | 68 776 | 2.6 MB | 230 015 | 3.96 s | 251 µs |
| CasADi SX unrolled | 215 190 | 4.5 MB | 198 | **> 600 s** | — |
| CasADi SX unrolled, no `cse` | 309 030 | 6.4 MB | 237 | **> 600 s** | — |
| CasADi MX + `map`, *audit configuration* | 48 496 | 2.5 MB | 235 738 | 16.3 s | 383 µs |
| CasADi MX called per stage | 26 822 | 1.2 MB | 68 646 | 25.4 s | **244 µs** |

And at M=17, where only one CasADi variant still compiles at all:

| variant | lines | source | workspace | compile | runtime |
|---|---:|---:|---:|---:|---:|
| **alloy** | **625** | **690 KB** | 115 320 | **1.75 s** | 3.78 ms |
| CasADi SX + `map`, *audit configuration* | 494 159 | 20.0 MB | 1 423 642 | 38.4 s | **2.52 ms** |
| CasADi SX unrolled | 1 118 065 | 23.8 MB | 493 | **> 600 s** | — |
| CasADi SX unrolled, no `cse` | 1 784 091 | 36.9 MB | 847 | **> 600 s** | — |
| CasADi MX + `map`, *audit configuration* | 318 036 | 17.6 MB | 1 462 701 | **> 600 s** | — |
| CasADi MX called per stage | 370 974 | 14.7 MB | 480 919 | **> 600 s** | — |

Three corrections fall out of this table.

**`map` is not the chain's handicap.** For SX, the map makes CasADi compilable here. Unrolled SX is
215 190 lines, and clang does not finish it in ten minutes. At M=17, the mapped SX variant is the
only CasADi encoding measured in the audit that compiles inside ten minutes. Every other measured
encoding times out. The literal `casadi_sx` sweep now records the M=3 cell. It hits the default
compile timeout at M=5 and skips larger sizes.

For MX, the audit compared mapping with per-stage function calls. Mapping is a mild handicap at
383 µs against 244 µs. It also makes the source larger at 48 496 lines against 26 822 lines. The
audit did not measure the literal inline `casadi_mx` form that the harness now provides. A Python
probe reported a 9× handicap. It measured `ca.Function.__call__` on a graph with a 230 000-double
workspace. The
[sweep methodology](scalability.md#why-a-c-harness-and-not-the-google-benchmark-python-bindings)
explains this dispatch cost. It does not survive compilation.

**Alloy loses this kernel on runtime, by 2.4× at M=5 and 1.5× at M=17.** That has never been
published, and it should be: the chain is the suite's largest sparse structured problem and it is the
one where alloy's loop-preserving lowering costs the most. The gap narrowing as the problem grows is
the expected shape — the per-iteration callee dispatch amortizes — but at these sizes it is real.

**Alloy wins the same kernel on everything else, by a lot.** 583 lines against 26 822–309 030; 124 KB
against 1.2–6.4 MB; 0.87 s to compile against 4–25 s, or never. At M=17 it is 625 lines against
494 159, and 690 KB against 20 MB. `ca.cse` is load-bearing on the CasADi side here — without it the
unrolled SX source grows from 215 190 to 309 030 lines — and the chain mirror is the one place in the
suite that already calls it.

So the chain's honest summary is a trade, not a win: **alloy generates three orders of magnitude less
code and compiles it in a second, and evaluates it 1.5–2.4× slower.** That is a coherent story and a
publishable one; the current situation, where the sweep runs and the numbers go nowhere, is worse.

## CasADi 3.8: `transform`, dense kernels, and the race-car horizon

CasADi 3.8.0 (September 2026) shipped three things that could have changed this page: a
`Function.transform()` pipeline of graph simplifications, specialised dense and dense-sparse
matrix-product kernels with a per-product BLAS selector (`mtimes(A, B, "reference" | "classic" |
"blasfeo")`), and ONNX import. Measured on 2026-09-05 with the same protocol as the rest of this
page (exact sparse Lagrangian Hessian, Google Benchmark, `-O3`, every cell through the dense-reference
gate). Raw CSVs live under `benchmarks/results/ca38/`.

**Upgrading alone changes nothing.** The unbumpercars `casadi_mx` Hessian moved from 8038 to 7958 µs
at C=8 and the compiled closed loop from 134.3 to 135.7 ms, both inside noise. 3.7 already inlined
dense-dense products in generated C; 3.8 routes them through a shared helper and adds a dense-sparse
kernel.

**`transform` is `cse`, and it halves CasADi's scalar kernels.** Isolating the passes on the race-car
SX Hessian at N=25: `cse` alone takes 15.0 µs to 8.2 µs, and `combine_terms`, `const_folding`,
`ref_count` and `empty_inputs` each change nothing. `cse` removes 11% of the instructions but 80% of
the `sin` and `cos` calls (825 to 175 and 850 to 175), because each reverse-over-forward sweep
re-derives the RK4 stage's trigonometry as fresh SX nodes and only CSE merges them. Per problem,
alloy against the best CasADi encoding before and after `transform({})`:

| problem, size | alloy | best CasADi | best CasADi + `transform` | passes' effect |
|---|---:|---:|---:|---:|
| race_cars N=25 | 14.1 µs | `sx` 15.0 | `sx` 8.9 | 1.7× |
| race_cars N=50 | 40.0 µs | `sx` 29.8 | `sx` 16.4 | 1.8× |
| race_cars N=100 | 151.0 µs | `sx` 62.9 | `sx` 33.6 | 1.9× |
| npmpc N=12 | 63.6 µs | `mx` 43.0 | `mx` 39.3 | 1.09× |
| npmpc_decoder W=64 | 269.6 µs | `mx` 238.1 | `mx` 231.3 | 1.03× |
| unbumpercars C=8 | 2628 µs | `mx` 7958 | `mx` 7631 | 1.04× |

Generated source shrinks 15 to 30% everywhere. The rule that follows is the one this page already
states: the CasADi column is CasADi's best configuration, so every CasADi kernel is taken after
`transform({})`. Sweeps enable this by default. Compiled CasADi IPOPT solvers use transformed
versions of every generated oracle, including values and derivatives. The SQP CasADi adapter
also transforms every generated oracle and selects the upper Hessian triangle before code
generation. The initial C3 runs exposed that this adapter had omitted both steps. Those runs are
retained under `benchmarks/results/followup/2026-09-05/diagnostic-sqp-before-protocol/` and are
not headline evidence. This closes the open question of `ca.cse` per problem, in favour of always.

**The dense kernels and the BLAS selector cannot reach the timed oracles.** The mode is an attribute
of the `MX` multiplication node, read at evaluation and code generation. `ad_forward` and `ad_reverse`
in `multiplication.cpp` build their sensitivities with plain `mac(...)`, so every product created by
differentiation runs on the reference kernel: the C=4 unbumpercars Hessian with `"blasfeo"` contains
4 `dgemm` calls and about 450 reference products. `SX` has no matrix node at all, and `expand=True`
removes every mode. Batching each repetition's network layer into one matrix-matrix product
(`casadi_mx_gemm*` backends, on unbumpercars and npmpc) is a loss at every size because the Hessian of
a wide product is denser than the sum of per-repetition Hessians:

| cell | `casadi_mx` per stage | batched, reference | batched, BLASFEO |
|---|---:|---:|---:|
| unbumpercars C=8 | 7958 µs | 10020 | 10019 |
| npmpc N=25 | 93.1 µs | 162.3 | 166.0 |
| npmpc_decoder W=256 | 12453 µs | 20267 | 20284 |

The one thing batching buys is reach: at npmpc N=50 the batched graph still compiles inside the
timeout where unrolled `casadi_mx` does not. The backends stay in the sweep as the recorded negative.

**The unrolled race-car cost caused quadratic work during the audit.** The expression DAG is
hash-consed at construction, so alloy's race-car Hessian already evaluated 14 transcendentals per
stage, CasADi's count after CSE. Alloy's runtime was nevertheless superlinear in the horizon
(0.51, 0.56, 0.80, 1.51 µs per stage at N = 10, 25, 50, 100) where CasADi was linear. The sweep's
workspace column showed 0 doubles up to N=10, then 2400, 73 296 and 266 496. At the time, the cost
in `_race_car_nlp` was a Python loop over stages. Every `z[i*NZ:(i+1)*NZ]` slice had a pad as its
adjoint, and the coloured forward sweep over N pads materialised N full-length seed buffers, each
zero-filled then written in 24 places. The equality and corridor rows were already vmapped.
Rewriting the cost as one `vmap` over a six-residual stage function, with the per-stage weights in a
constant vector, computed the same Hessian to 0.0 on every shared entry and gave:

| N | alloy, unrolled cost | alloy, vmapped cost | CasADi SX + `transform` |
|---:|---:|---:|---:|
| 25 | 14.3 µs, 3216 lines | 12.5 µs, 1285 lines | 8.9 µs |
| 50 | 40.2 µs, 5216 lines | 25.7 µs, 1285 lines | 16.4 µs |
| 100 | 148.6 µs, 9216 lines | 50.4 µs, 1285 lines | 33.6 µs |

Linear, constant source, and 1.5× behind CSE'd SX at every horizon instead of 4.5× at N=100. The
remaining factor is the small-stage handicap already described for narrow decoders: 1492 loops of
trip count 4 and four seed colours on a kernel this cheap. The benchmark now maps the race-car
cost. Alloy's program passes also combine left-associated sums of single-use pads into one
zero-fill and one scatter-add per term, preserving addition order for overlapping entries.

**ONNX import is a modelling convenience, not a performance change.** Symbolic import produces the
same `MX` graph one writes by hand and lands in the kernels above.

Reproduce with `uv run benchmarks/run.py sweep --workloads race_cars,npmpc,npmpc_decoder,unbumpercars`
with `--casadi-transform` and `--no-casadi-transform`; the gemm backends are `casadi_mx_gemm`, `casadi_mx_gemm_classic`
and `casadi_mx_gemm_blasfeo`.

## Where the suite handicaps alloy

- **Optimization level.** Alloy's JIT hardcodes `-O2`; the sweep harness compiles both backends at
  `-O3`, and CasADi's own JIT is handed `-O3`. Measured on the `npmpc` episode across both
  compilers and both levels, function evaluation moves by at most 6%:

  | | `-O2` | `-O3` |
  |---|---:|---:|
  | gcc 13.3 | 1.325 ms | 1.280 ms |
  | clang 20 | 1.309 ms | 1.281 ms |

  So this is not where the story is — but the level should be chosen deliberately rather than
  inherited from a hardcoded string, and there is currently no way to override it.
- **Per-solve IPOPT setup.** Alloy's wrapper calls `CreateIpoptProblem`, every option setter and
  `FreeIpoptProblem` on *every* solve, because bounds are parameter-dependent, and all of it is
  inside its own timer. CasADi's `nlpsol` does that once at construction, outside every timer. (Its
  *generated* C does it per solve, like alloy's — one more reason the code-generated column is the
  right comparison.)
- **`t_fe` is not the same quantity on both sides.** Alloy's includes a bounds-evaluation kernel
  that CasADi has no analogue for; CasADi's is the sum of its `t_wall_nlp_*` callback timers, which
  may not cover the interface's sparse-matrix copies.
- **Fused oracles.** Alloy's `base` kernel computes `f` and `g` together and is called for both, so
  it evaluates `g` on every `f` call and vice versa, where CasADi evaluates them separately.

## What a fair closed-loop comparison needs

The rules this page argues for, collected. They are the admissibility test a number has to pass
before it appears anywhere else on this site.

The oracle comparison is two columns that differ in one thing:

| oracle provider | solver |
|---|---|
| alloy generated C | one IPOPT build, one compiler, one optimization level, one C-side timer |
| CasADi generated C | the same |

with bounds, warm starts, tolerances, iteration limits and input buffers held fixed, and with
iteration counts *and* per-kernel call counts compared rather than iteration counts alone. Two
protocols are worth reporting separately: a warm steady state where the solver object already exists,
and a full lifecycle including construction and teardown, since alloy's deployed path pays the latter
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
The sweep's `coloring_width` is also implementation-specific: Alloy reports the compressed tangent
directions it executes, while CasADi leaves the field blank. Do not use it as a cross-backend
comparison.
The executable count removes `static const` declarations from the C translation unit. Static
metadata contains those declarations and the generated header, so the two counts sum to the full C
and header artifact. The classification follows generated C syntax: index, seed, and numeric
constant arrays are metadata, while an inline numeric literal remains executable source. Alloy and
CasADi do not emit constants in the same form. Their sparse headers differ too: Alloy includes
coordinate, row-compressed, and column-compressed views for consumers, while CasADi emits one
column-compressed pattern. The metadata count is therefore the shipped source artifact, not a
normalized measure of sparsity information. Report both counts. These costs need their own rows
rather than being folded into a speed-up.

## Reproducing this page

The scripts behind these measurements live under `benchmarks/results/fairness/`, which is scratch
space rather than part of the suite. That is a defect, not a design: a number the paper quotes has to
come out of the benchmark harness so it is reproducible by someone who is not us. Moving them is
tracked internally, and this page becomes the record of what changed and why once they land.
