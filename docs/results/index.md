# Benchmark results

These are the latest published measured numbers. They are regenerated from the benchmark suite in this
repository rather than copied from a paper, so when alloy improves or a new problem lands, this is
the page that changes first.

Everything here is measured against **CasADi**, in both its SX (scalar) and MX (block) forms, on
the same problem, and every measurement is gated by a correctness check: each backend's compact
derivative is scattered into a dense matrix using its own sparsity pattern and compared entry by
entry against a reference built by a different construction — NumPy for the race-car Jacobian and
for the chain, race-car and neural-process MPC Lagrangian Hessians, Alloy's dense per-stage Jacobian
of an unrolled twin for the chain and neural-process MPC Jacobians, and a CasADi Lagrangian Hessian
for the safety filter. A cell that does not agree produces no timing.

!!! warning "The Jacobian tables await a same-machine refresh"

    The sweep now benchmarks the full constraint Jacobian taken from each optimizer's solver
    descriptor. The published race-car and neural-process MPC Jacobian tables below predate that
    correction and measured equality rows only. They remain labeled as historical rather than being
    relabeled with numbers from a different kernel.

!!! warning "The solver-in-the-loop numbers on this page are not yet a fair comparison"

    An audit in August 2026 found that the `ipopt+alloy` and `ipopt+casadi` columns differ in two
    things besides the oracle provider: they link **different builds of IPOPT** (worth 2.7× to 17× in
    solver-internal time on its own) and the CasADi column evaluates its oracles through CasADi's
    **virtual machine** rather than compiled C. On the one problem where a fully controlled
    comparison has been built, alloy's oracle advantage is **1.13×**, not the 4.5× the shipped column
    reports. Every closed-loop IPOPT multiple below is therefore an upper bound on the oracle effect
    and is marked as such. The kernel sweeps, the code-size figures and the two `alloy-sqp` columns
    are unaffected. [Read the audit](fairness.md).

## The short version

**Alloy was close to CasADi SX on speed, at a fraction of the generated source.** On the historical
race-car equality Jacobian, alloy was slightly ahead at the smallest horizons and fell behind as they grew —
level at one stage, about 15% slower at 200 and 19% at 500. Meanwhile the generated source at a
50-stage horizon is 22 KB against SX's 455 KB, and at 500 stages 78 KB against 4.5 MB. Alloy's line
count is *flat*: 482 lines at ten stages and 482 at five hundred, because the whole interstage
Jacobian is one loop calling one function.

**Against CasADi MX, alloy is about twice as fast** wherever MX still compiles at all. MX exceeds a
180-second compile budget at 200 stages.

**On the hanging chain, alloy loses on runtime and wins everything else.** 1.5–2.4× slower than
CasADi's fastest configuration, from 583 lines against 26 822 and a 0.87-second compile against 25
seconds — or against never, since CasADi's scalar expansion does not compile at this size in ten
minutes.

**On a dense network evaluated at every node of a horizon, the code-size result is categorical.**
Alloy's generated source for the neural-process-MPC Jacobian is 417 lines at every horizon from 6 to
200 *and* every decoder width from 16 to 256; CasADi SX reaches 1.31 million lines and stops
compiling, and CasADi MX needs 227 seconds for the 50-stage Hessian that alloy compiles in 0.8. In
closed loop, with both sides code-generated and compiled, the same swing-up episode solves in 1.29 ms
per step against 1.57 ms — a modest win, and a much smaller one than the code-size figures suggest,
because on oracle evaluation alone alloy and a compiled CasADi are close.

**On a solver-in-the-loop workload the reported gap is larger, and most of it is not the oracles.**
Driving the same nonlinear program through IPOPT and changing which library provides the oracles,
alloy runs 4.5× faster on the safety filter's discrete model. But the two columns also run two
different IPOPT builds and the CasADi one runs interpreted, and the audit puts the share attributable
to the oracles at a fraction of that. Where the comparison *is* controlled — the two `alloy-sqp`
columns, both code-generated compiled C differing only in who generated it — the margin is
**1.2–1.4×**. That is the number to quote for oracle performance.

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

## Hanging chain of masses — where alloy loses

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

So it is a trade, not a win: **alloy generates three orders of magnitude less code and compiles it in
a second, and evaluates it 1.5–2.4× slower.** The gap narrows as the problem grows, which is the
expected shape — the per-iteration callee dispatch amortizes — but at these sizes it is real, and
CasADi's scalar expansion stops being compilable well before alloy's output stops being small.

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
