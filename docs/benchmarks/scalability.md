# Scalability

This page follows the Hessian of the Lagrangian as each [benchmark problem](index.md#the-problems)
grows: how long the generated code takes to run, how long it takes to compile, and how large it
is.

Each point is one generated C kernel, compiled on its own and timed with
[Google Benchmark](https://github.com/google/benchmark) in five fresh processes. Before timing, the
harness checks the kernel's result against an independent reference, and a kernel that fails the
check gets no timing. Compilation stops after 180 seconds. An encoding that runs out of time at
one size is not tried at larger sizes, which is why some lines end early.

CasADi can build the same model from scalar operations or from matrix operations and function
calls, and the choice changes both speed and compile time. The figures include each encoding the
study measured:

- **CasADi SX**: every operation unrolled into scalars.
- **CasADi MX**: matrix operations kept whole.
- **CasADi MX with calls**: MX with subfunctions kept as calls instead of inlined.
- **CasADi mapped SX**: an SX function applied to every stage with `Function.map`.
- **CasADi MX with GEMM**: MX with matrix products lowered to general matrix multiplication
  (GEMM) calls. The study also measured the classic and BLASFEO GEMM lowerings, which fall within
  about 2% of this one and are left out.

## Evaluation time

<div class="vega-chart" data-spec="../../assets/benchmarks/hessian_time.vl.json"></div>

On the race car and neural-process MPC, Scaly's evaluation time grows linearly with the horizon.
On the race car, CasADi's SX encoding is 1.1× slower at `N = 1`, and the gap widens to between
5.5× and 6.5× from `N = 50` on. On
the problems that contain neural networks, SX compiles only at the shortest neural-process horizon
and not at all on unbumpercars, and the encodings that do compile pay for their calls and matrix
products.

## Compile time

<div class="vega-chart" data-spec="../../assets/benchmarks/compile_time.vl.json"></div>

The horizontal line is the 180-second limit. Scaly compiles the race car in under half a second at
every horizon. It keeps the repeated stage as a loop over one compiled body, so the executable
code stays at about 120 kB from `N = 5` to `N = 500`. Compile time is also what ends most CasADi
lines: the unrolled encodings produce source files that the C compiler takes minutes to digest.

## Code size

<div class="vega-chart" data-spec="../../assets/benchmarks/code_size.vl.json"></div>

Code size counts the executable part of the generated C source, leaving out constant tables such
as sparsity patterns. The two generators store those tables differently, so this is a guide to
how much code the compiler has to process rather than an exact measure of the model.

Each point is the mean of the five runs. The tooltips of the evaluation-time figures also give the
coefficient of variation across those runs. It is at most 8%, except for Scaly on the race car at
`N = 500`, where it reaches 11%.
