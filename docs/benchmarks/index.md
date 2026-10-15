# Benchmarks

Scaly and CasADi both generate C for optimization models. These benchmarks compare the two on
four optimal-control problems, and answer three questions:

- **How fast are the derivatives?** The time the generated code takes to evaluate the Hessian of
  the Lagrangian, the second derivatives a solver needs at every iteration, as the problem grows.
- **How fast is a controller?** The time a complete controller takes per control step when the
  same solver runs on Scaly's generated functions instead of CasADi's.
- **What does it cost to build?** The time from the model to compiled code, and the size of that
  code.

All numbers come from one study run on October 15, 2026, against CasADi 3.8.0, on an AMD
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

Scaly is faster at every size but one, and its lead grows with the problem. The CasADi
implementation that comes closest changes with the size. On neural-process MPC, MX stays within 1.7
to 2.3× of Scaly up to `N = 50`, then stops compiling and leaves only slower ones. On
unbumpercars, only Scaly compiles at `C = 16` and `C = 32`. The one exception is the chain at
`M = 5`, where CasADi's SX implementation takes 18% less time than Scaly.

The second figure shows the size of the object file each Hessian compiles to.

<div class="vega-chart" data-spec="../assets/benchmarks/sweep_size.vl.json"></div>

Scaly's object file is smaller than every CasADi implementation's at every size. On the race car it
stays between 29 and 46 kB at every horizon, because the repeated stage stays a loop over one
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
step. Unbumpercars evaluates a neural network for every pair of cars, which takes 97 to 98% of
CasADi's step, so most of Scaly's 8.8 to 9.1× faster function evaluation reaches the step. The
race car spends most of its step inside the solver: function evaluation is 2.7 to 3.1× faster with
Scaly but only 5 to 8% of CasADi's step. On the chain, CasADi's function evaluation takes 13% less
time than Scaly's, too small a part of the step to change it.

Before the first step, each side builds its controller: it constructs the problem, generates C
for its functions and compiles it, starting from an empty cache.

<div class="vega-chart" data-spec="../assets/benchmarks/closed_loop_build.vl.json"></div>

The [closed-loop page](closed_loop.md) compares the time of every step with the controller's
sampling time. Scaly's controllers always finish within it. With CasADi, the unbumpercars
controllers take longer than their 100 ms sampling time on 50 of 200 steps with SQP and on 186 with
IPOPT.

## Reproduce the study

```bash
SCALY_VECTOR_LIBM=glibc SCALY_CC=gcc CC=gcc CXX=clang++ uv run benchmarks/run.py study --out-dir benchmarks/results/<study-name>
uv run benchmarks/run.py report benchmarks/results/<study-name>
```

The first command runs every benchmark with one fresh process per cell or provider and an empty
compilation cache. It checks the `performance` governor and disabled boost state, and uses order
seed 0. The second command summarizes the results. The [fairness page](fairness.md#what-the-results-do-not-show)
explains the limits of one-process measurements and how to check variation between runs. The
[benchmark README](https://github.com/PREDICT-EPFL/scaly/tree/main/benchmarks) describes the
harness and its options.
