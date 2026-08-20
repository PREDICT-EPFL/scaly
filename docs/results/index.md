# Benchmark results

These are the current measured numbers. They are regenerated from the benchmark suite in this
repository rather than copied from a paper, so when alloy improves or a new problem lands, this is
the page that changes first.

Everything here is measured against **CasADi**, in both its SX (scalar) and MX (block) forms, on
the same problem with the same solver, and every measurement is gated by a correctness check: each
backend's compact derivative is scattered into a dense matrix using its own sparsity pattern and
compared entry by entry against an independent reference — an alloy-computed Jacobian for the
race-car sweep, a CasADi-computed Lagrangian Hessian for the safety filter. A cell that does not
agree produces no timing.

## The short version

**Alloy is close to CasADi SX on speed, at a fraction of the generated source.** On the race-car
equality Jacobian, alloy is slightly ahead at the smallest horizons and falls behind as they grow —
level at one stage, about 15% slower at 200 and 19% at 500. Meanwhile the generated source at a
50-stage horizon is 22 KB against SX's 455 KB, and at 500 stages 78 KB against 4.5 MB. Alloy's line
count is *flat*: 482 lines at ten stages and 482 at five hundred, because the whole interstage
Jacobian is one loop calling one function.

**Against CasADi MX, alloy is about twice as fast** wherever MX still compiles at all. MX exceeds a
180-second compile budget at 200 stages.

**On a dense network evaluated at every node of a horizon, the code-size result is categorical.**
Alloy's generated source for the neural-process-MPC Jacobian is 417 lines at every horizon from 6 to
200 *and* every decoder width from 16 to 256; CasADi SX reaches 1.31 million lines and stops
compiling, and CasADi MX needs 227 seconds for the 50-stage Hessian that alloy compiles in 0.8. In
closed loop, with both sides code-generated and compiled, the same swing-up episode solves in 1.29 ms
per step against 1.57 ms — a modest win, and a much smaller one than the code-size figures suggest,
because on oracle evaluation alone alloy and a compiled CasADi are close.

**On a solver-in-the-loop workload, the gap is larger.** Driving the same nonlinear program
through IPOPT with the same options, and changing only which library provides the oracles, alloy
runs 2.8–3.9× faster on the continuous-time model and 4.5–9.0× faster on the discrete one. The
advantage grows with both the vehicle count and the network size, and IPOPT's iteration counts are
identical in every cell — so the difference is oracle evaluation, not a different solve.

## Race-car equality Jacobian

A 4-state, 2-control bicycle model with a `tanh` rolling-resistance term, integrated with RK4 over
a horizon, differentiated as a compact sparse Jacobian.

| Stages | Alloy µs | CasADi SX µs | CasADi MX µs | Alloy KB | SX KB | MX KB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.09 | 0.09 | 0.13 | 19.3 | 14.1 | 42.2 |
| 10 | 1.05 | 0.91 | 1.50 | 17.9 | 95.0 | 374.6 |
| 50 | 5.24 | 4.70 | 8.39 | 22.1 | 455.1 | 1996.0 |
| 100 | 10.27 | 9.28 | 18.72 | 28.3 | 907.0 | 4019.8 |
| 200 | 21.20 | 18.44 | compile > 180 s | 40.6 | 1810.8 | 8676.0 |
| 500 | 54.46 | 45.76 | skipped | 78.0 | 4528.9 | — |

The size column is the point. Alloy's growth is entirely in constant index tables — data, not
code — because the stage structure is expressed with `map_` and survives lowering as a real loop.
Construction time follows: 248 ms to build the 500-stage case against SX's 481 ms.

Full tables, workspace figures and the reasoning are in [the scalability sweep](scalability.md).

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

**In closed loop the whole-solve number is the interesting one.** The same 100-step swing-up episode,
same plant, same warm starts, changing only the solver and the oracle provider:

| column | mean solve | mean FE | FE share |
|---|---|---|---|
| IPOPT, alloy oracles | 3.85 ms | 1.53 ms | 40% |
| IPOPT, CasADi oracles | 17.34 ms | 12.93 ms | 75% |
| alloy-sqp, alloy oracles | **1.29 ms** | 0.66 ms | 51% |
| alloy-sqp, CasADi oracles | 1.57 ms | 0.92 ms | 58% |

**Read the IPOPT/CasADi row as a statement about a configuration, not about CasADi.** `nlpsol`
evaluates through CasADi's own virtual machine unless told to compile, and that row does not compile.
Measured against a compiled CasADi instead, the same solve takes 5.37 ms with 1.46 ms of function
evaluation — so **the oracle comparison is a wash**, which is exactly what the kernel sweeps above
predict, putting CasADi MX at 0.83–0.90× of alloy at this decoder width. The uncompiled row is
retained only because the other closed-loop problems on this page are configured the same way, and
making them consistent is a single pending piece of work rather than three inconsistent ones.

The comparison that *is* fair here is the two alloy-sqp columns: both are code-generated, compiled C
differing only in who generated it, and there alloy is ahead by 1.4× on function evaluation (0.66
against 0.92 ms) and 1.2× on the whole solve. Alongside that, all four columns bring the pendulum
upright at the same step with identical iteration counts inside each solver and trajectories agreeing
to 2.4e-13 in state.

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

Vehicles with pairwise control-barrier constraints over a small neural dynamics model, solved with
IPOPT at every step of a closed loop. Both columns solve the identical problem with the identical
solver; only the library providing the oracles differs.

| Filter model | Speed-up, alloy over CasADi |
|---|---|
| continuous-time | 2.8–3.9× |
| discrete MLP (the default) | 4.5–9.0× |

Across two to eight vehicles, with the advantage growing in both the vehicle count and the network
size. IPOPT's iteration count is identical in every cell, which is the useful control: the solver
walks the same path either way, so what is being measured is the cost of evaluating the oracles and
getting into and out of them.

What the problem exists to exercise is the exact sparse Lagrangian Hessian, computed through
preserved `map` structure — the construction most of alloy's sparse machinery is built to make
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
  is source size, construction time and compile time, not raw evaluation speed.
- **Numbers come from one machine.** Absolute timings vary with hardware; the ratios are the
  durable part.
- **Historical results are labelled.** [The scalability page](scalability.md) keeps measurements
  from retired formulations, marked as such, so an old number is never mistaken for a current one.
