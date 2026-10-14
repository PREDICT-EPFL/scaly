# Benchmarks

Scaly and CasADi both generate C for optimization models. These benchmarks compare the two on
four optimal-control problems, and answer three questions:

- **How fast are the derivatives?** The time the generated code takes to evaluate the Hessian of
  the Lagrangian, the second derivatives a solver needs at every iteration, as the problem grows.
- **How fast is a controller?** The time a complete controller takes per control step when the
  same solver runs on Scaly's generated functions instead of CasADi's.
- **What does it cost to build?** The time from the model to compiled code, and the size of that
  code.

All numbers come from one study run on October 14, 2026, against CasADi 3.8.0, on an AMD
Ryzen 9 7940HS desktop. [Are the comparisons fair?](fairness.md) explains what the two sides
share and where the comparison has limits.

## The problems

| Problem | What it is | Size parameter |
| --- | --- | --- |
| Race-car MPC | Model predictive control (MPC) of a Formula Student car following a race track | horizon `N` |
| Neural-process MPC | MPC of a Furuta pendulum whose dynamics are a learned neural network | horizon `N` |
| Chain of masses | MPC of a chain of point masses connected by nonlinear springs | number of masses `M` |
| Unbumpercars | A safety filter that keeps `C` cars apart, with a neural model inside pairwise constraints | number of cars `C` |

## Derivatives as the problem grows

Each point divides the time CasADi takes to evaluate the Hessian by the time Scaly takes, at one
size. CasADi can express the same model in several ways, called encodings, and the fastest one
depends on the problem and its size. Each point compares with the fastest encoding at that size,
and its color and shape say which one it was. A cross marks a size where CasADi did not compile,
because every encoding reached the 180-second compile limit there or at a smaller size.

<div class="vega-chart" data-spec="../assets/benchmarks/hessian_speedup.vl.json"></div>

Scaly is faster at every size but one, and the gap grows with the problem. On the race car it
goes from 1.1× at `N = 1` to between 5.6 and 6.7× from `N = 40` on. Unbumpercars goes from 4.2×
at `C = 2` to 17× at `C = 8`, and at `C = 16` and `C = 32` only Scaly compiles. On neural-process
MPC, Scaly stays 1.6 to 2.3× ahead of CasADi's MX encoding up to `N = 50`. At the two longest
horizons MX no longer compiles, and the gap to the slower encodings that still do reaches 4.9 and
17×. The exception is the chain at `M = 5`, where CasADi's SX encoding takes 18% less time than
Scaly. At `M = 9`, SX no longer compiles and Scaly is 8.7× faster than the fastest encoding left.

The [scalability page](scalability.md) shows every encoding separately, with the time to build
each one and the size of its code.

## Controllers

Each controller runs a full simulated episode twice, once with Scaly's generated functions and
once with CasADi's, with the same solver, settings and warm starts. Each pair of bars divides the
mean time per control step into function evaluation, every objective, constraint and derivative
the solver asks for, and the rest of the solver's work. Each controller has its own time axis, and
the number next to Scaly's bar says how many times faster its step is on average.

<div class="vega-chart" data-spec="../assets/benchmarks/closed_loop_share.vl.json"></div>

How much of a faster function evaluation reaches the whole step depends on its share of the
step. Unbumpercars evaluates a neural network for every pair of cars, which takes 97 to 98% of
CasADi's step, so Scaly's 8.7 to 9.2× faster function evaluation makes the step 7.3 to 8.2×
faster. The race car spends most of its step inside the solver. There function evaluation is 2.6
to 3.1× faster with Scaly but only 5 to 8% of the step, so the step gains 4 to 8%. On the chain,
CasADi's function evaluation takes 14% less time than Scaly's, and the two steps take the same
time to within 1%.

Before the first step, each side builds its controller: it constructs the problem, generates C
for its functions and compiles it, starting from an empty cache.

<div class="vega-chart" data-spec="../assets/benchmarks/closed_loop_build.vl.json"></div>

Scaly has the race-car controllers ready in under 3 seconds, against 9 and 16 seconds for
CasADi, and the unbumpercars controllers 5 to 9 times sooner. On neural-process MPC Scaly is 1.3
and 1.7 times sooner, and on the chain both take about 30 seconds, CasADi slightly less.

The [closed-loop page](closed_loop.md) compares the time of every step with the controller's
sampling time. Scaly's controllers always finish within it. With CasADi, the unbumpercars
controllers take longer than their 100 ms sampling time on 48 of 200 steps with SQP and on 186 with
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
