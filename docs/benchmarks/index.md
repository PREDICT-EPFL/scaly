# Benchmarks

Scaly and CasADi both generate C for optimization models. These benchmarks compare the two on
four optimal-control problems, and answer three questions:

- **How fast are the derivatives?** The time the generated code takes to evaluate the Hessian of
  the Lagrangian, the second derivatives a solver needs at every iteration, as the problem grows.
- **How fast is a controller?** The time a complete controller takes per control step when the
  same solver runs on Scaly's generated functions instead of CasADi's.
- **What does it cost to build?** The time from the model to compiled code, and the size of that
  code.

All numbers come from one study run on October 29, 2026, at revision
[`06e10c16`](https://github.com/PREDICT-EPFL/scaly/commit/06e10c16), against CasADi 3.8.0, on an AMD
Ryzen 9 7940HS desktop. [Are the comparisons fair?](fairness.md) explains what the two sides
share and where the comparison has limits.

## The problems

| Problem | What it is | Size parameter |
| --- | --- | --- |
| Race-car MPC | Model predictive control (MPC) of a Formula Student car following a race track | horizon `N` |
| Neural-process MPC | MPC of a Furuta pendulum whose dynamics are a learned neural network | horizon `N` |
| Unbumpercars | A safety filter that keeps `C` cars apart, with a neural model inside pairwise constraints | number of cars `C` |
| Chain of masses | MPC of a chain of point masses connected by nonlinear springs | number of masses `M` |

## Derivatives as the problem grows

The first figure shows the time the generated code takes to evaluate the Hessian at each size.
CasADi can express the same model in several ways, called implementations, and the fastest one
depends on the problem and its size. The wide gray line follows the best of them at each size, and
the arrow in each panel marks Scaly's largest lead over it. A line ends where its implementation
reached the 180-second compile limit.

<div class="vega-chart" data-spec="../assets/benchmarks/sweep_runtime.vl.json"></div>

At the largest race-car horizon, `N = 400`, Scaly's Hessian is 5.73× faster than the fastest
completed CasADi implementation. The ratio is 21.6× at neural-process MPC `N = 200` and 25.5×
at unbumpercars `C = 8`. On neural-process MPC, MX stays within 2.5 to 3.0× of Scaly up to
`N = 50`, then stops compiling and leaves only slower ones. On unbumpercars, only Scaly compiles
at `C = 16` and `C = 32`, so those sizes have no runtime ratio.

On the chain, Scaly is 1.20× faster at `M = 3`, but slower than CasADi at `M = 5` and `M = 9`.
CasADi's SX implementation takes 82.2 µs against Scaly's 96.3 µs at `M = 5`,
and mapped SX takes 2.38 ms against 4.32 ms at `M = 9`. The CasADi-over-Scaly ratios are
0.85× and 0.55×.

The second figure shows the size of the object file each Hessian compiles to.

<div class="vega-chart" data-spec="../assets/benchmarks/sweep_size.vl.json"></div>

Scaly's object file is smaller than every CasADi implementation's at every size. On the race car it
stays between 33 and 52 kB at every horizon, because the repeated stage stays a loop over one
compiled body, while CasADi's grow with the horizon.

The [scalability page](scalability.md) adds which implementations compile at each size, the time to
build each one and what its object file contains.

## Controllers

Each controller runs a full simulated episode twice, once with Scaly's generated functions and
once with CasADi's, with the same solver, settings and warm starts. Each bar divides the mean time
per control step into function evaluation, every objective, constraint and derivative the solver
asks for, and the rest of the solver's work. Each problem has its own time axis.

<div class="vega-chart" data-spec="../assets/benchmarks/closed_loop_share.vl.json"></div>

How much of a faster function evaluation reaches the whole step depends on its share of the
step. Unbumpercars evaluates a neural network for every pair of cars, which takes 97 to 99% of
CasADi's step. Scaly's mean function evaluation is 9.7 to 10.1× faster, and the whole step is
8.1 to 9.1× faster. The race car spends most of its step inside the solver. Function evaluation
is 2.8 to 3.0× faster with Scaly but only 4 to 8% of CasADi's step. On the chain, CasADi's
function evaluation takes 41% less time than Scaly's, but the whole step differs by only 1%.

Before the first step, each side builds its controller: it constructs the problem, generates C
for its functions and compiles it, starting from an empty cache.

<div class="vega-chart" data-spec="../assets/benchmarks/closed_loop_build.vl.json"></div>

The [closed-loop page](closed_loop.md) compares the time of every step with the controller's
sampling time. Scaly's controllers always finish within it. With CasADi, the unbumpercars
controllers take longer than their 100 ms sampling time on 55 of 200 steps with sequential
quadratic programming (SQP) and on 195 with IPOPT. Unbumpercars SQP does slightly different work
with each provider, as the [closed-loop comparison](closed_loop.md#do-both-sides-do-the-same-work)
explains.

## Reproduce the study

```bash
SCALY_VECTOR_LIBM=glibc SCALY_CC=clang CC=clang CXX=clang++ uv run benchmarks/run.py study --out-dir benchmarks/results/<study-name> --repetitions 1 --order-seed 0 --headline --boost off
uv run benchmarks/run.py report benchmarks/results/<study-name>
```

The first command runs every benchmark with one fresh process per cell or provider and an empty
compilation cache. It checks the `performance` governor and disabled boost state, and uses order
seed 0. The second command summarizes the results. The [fairness page](fairness.md#what-the-results-do-not-show)
explains the limits of one-process measurements and how to check variation between runs. The
[benchmark README](https://github.com/PREDICT-EPFL/scaly/tree/main/benchmarks) describes the
harness and its options.
