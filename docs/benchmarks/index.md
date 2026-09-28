# Benchmarks

Scaly and CasADi both generate C for optimization models. These benchmarks compare the two on
four optimal-control problems, in two ways:

- **Derivative evaluation.** How fast the generated code evaluates the Hessian of the
  Lagrangian, the second derivatives a solver needs at every iteration, as the problem grows.
- **Closed-loop control.** How long a complete controller takes per control step when the same
  solver runs on Scaly's generated functions instead of CasADi's.

All numbers come from one study run on September 23, 2026, against CasADi 3.8.0, on an AMD
Ryzen 9 7940HS desktop. [Are the comparisons fair?](fairness.md) explains what the two sides
share and where the comparison has limits.

## The problems

| Problem | What it is | Size parameter |
| --- | --- | --- |
| Race-car MPC | Model predictive control (MPC) of a Formula Student car following a race track | horizon `N` |
| Neural-process MPC | MPC of a Furuta pendulum whose dynamics are a learned neural network | horizon `N` |
| Chain of masses | MPC of a chain of point masses connected by nonlinear springs | number of masses `M` |
| Unbumpercars | A safety filter that keeps `C` cars apart, with a neural model inside pairwise constraints | number of cars `C` |

## Derivative evaluation

The figure shows the time to evaluate the Hessian once. CasADi can express the same model in
several ways, and the fastest one depends on the problem, so its line follows whichever
encoding was fastest at each size. Hover over a point to see which one it was. A line ends where
no CasADi encoding finished compiling within 180 seconds.

<div class="vega-chart" data-spec="../assets/benchmarks/hessian_overview.vl.json"></div>

Scaly is faster in every comparison but one, and the gap tends to grow with size. On the race car
it goes from 1.1× at `N = 1` to 6.5× at `N = 200`. On neural-process MPC it stays between 1.5 and
2× while CasADi's MX encoding compiles, and reaches 6 to 21× at the longer horizons, where only
slower CasADi encodings still compile. Unbumpercars goes from 5× at `C = 2` to 22× at `C = 8`, and
at `C = 16` and `C = 32` only Scaly compiles. The exception is the chain at `M = 5`, where CasADi's
SX encoding is 1.2× faster. At `M = 9`, SX no longer compiles and Scaly is 8.8× faster than the
fastest remaining encoding.

The [scalability page](scalability.md) shows every CasADi encoding separately, with compile
time and code size.

## Closed-loop control

Each controller runs a full simulated episode with the same solver and the same settings on both
sides. Only the generated model functions change. The figure shows how much faster the Scaly
version is, for the whole solve and for function evaluation alone: every objective, constraint
and derivative the solver asks for.

<div class="vega-chart" data-spec="../assets/benchmarks/closed_loop_speedup.vl.json"></div>

Function evaluation is 1.8 to 53× faster. How much of that reaches the total depends on how much
of the step the solver spends on its own numerical work. The race car spends most of its step
solving quadratic subproblems, so a 4.2× faster function evaluation makes the whole step only 7%
faster. Unbumpercars is dominated by function evaluation, and the whole step is 7 to 9× faster.

The [closed-loop page](closed_loop.md) breaks the step time down and lists the measured times.

## Reproduce the study

```bash
SCALY_VECTOR_LIBM=glibc SCALY_CC=gcc CC=gcc CXX=clang++ uv run benchmarks/run.py study --out-dir benchmarks/results/<study-name>
uv run benchmarks/run.py report benchmarks/results/<study-name>
```

The first command runs every benchmark and the second summarizes the results. The
[benchmark README](https://github.com/PREDICT-EPFL/scaly/tree/main/benchmarks) describes the
harness and its options.
