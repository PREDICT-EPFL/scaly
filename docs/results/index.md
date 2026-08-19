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
