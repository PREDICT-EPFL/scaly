# Scalability

This page follows the Hessian of the Lagrangian as each [benchmark problem](index.md#the-problems)
grows: how long the generated code takes to run, which CasADi implementations still compile, how
long it takes to get from the model to compiled code, and how large that code is.

Each measurement is the C function that one implementation generates for the Hessian at one size,
compiled on its own and timed with [Google Benchmark](https://github.com/google/benchmark) in a
fresh process. Before timing, the harness checks its result against a separately constructed
reference, and a function that fails the check gets no timing. Compilation stops after 180
seconds, and an implementation that runs out of time at one size is not tried at larger sizes.

CasADi can build the same model from scalar operations or from matrix operations and function
calls, and the choice changes both speed and compile time. The study measures five implementations:

- **CasADi SX**: every operation unrolled into scalars.
- **CasADi MX**: matrix operations kept whole.
- **CasADi MX with calls**: MX with subfunctions kept as calls instead of inlined.
- **CasADi mapped SX**: an SX function applied to every stage with `Function.map`.
- **CasADi MX with GEMM**: MX with each network layer computed for every stage or car at once,
  as one general matrix multiplication (GEMM). Only the two problems with a neural network have
  it.

Unbumpercars has no single repeated stage, so it has no calls or mapped implementation.

## Evaluation time

In this figure and the other two line figures on this page, the wide gray line follows CasADi's
best implementation at each size, and the arrow in each panel marks Scaly's largest lead over it.

<div class="vega-chart" data-spec="../../assets/benchmarks/sweep_runtime.vl.json"></div>

On the race car, Scaly's evaluation time grows roughly linearly with the horizon, at about 70 ns
per stage from `N = 100` to `N = 400`, against about 400 ns for SX, CasADi's fastest implementation
there. On the problems that contain neural networks, SX compiles only at the shortest
neural-process horizon and not at all on unbumpercars. The implementations that do compile keep
the network's matrix products and calls, and pay for them at every evaluation.

Each point comes from one process. Google Benchmark repeats the C call inside that process for at
least half a second, but the study has no estimate of variation between processes, as the
[fairness page](fairness.md#what-the-results-do-not-show) explains.

Scaly is slower than CasADi on the chain at `M = 5` and `M = 9`. The CasADi-over-Scaly
Hessian ratios are 1.20× at `M = 3`, 0.85× at `M = 5` and 0.55× at `M = 9`. SX is fastest at
the two smaller sizes, and mapped SX is the only CasADi implementation that compiles at `M = 9`.

## Which implementations compile

Each cell divides an implementation's evaluation time by Scaly's at the same size. A gray cell
reached the 180-second compile limit, and "not run" marks the larger sizes skipped after it.

<div class="vega-chart" data-spec="../../assets/benchmarks/sweep_ratio.vl.json"></div>

Of the 85 CasADi Hessians in the study, 58 compiled and were timed, 11 reached the limit and 16 were
skipped after it. Mapped SX compiled neural-process MPC at `N = 25` and `N = 50` in 174 and
177 seconds, close enough to the limit that another run could stop it. Two CasADi Hessians are
faster than Scaly's, SX on the chain at `M = 5` and mapped SX at `M = 9`. No implementation is
best everywhere. Where it compiles, SX is the fastest on the race car and the chain, and MX on
the two problems with a neural network. On the race car, MX with calls and mapped SX are the
slowest at run time, up to 15× slower than Scaly. At the largest sizes they are sometimes the
only ones left. MX with calls is the only one at the longest neural-process horizon, and mapped
SX the only one at `M = 9`.

## From model to object file

Turning the model into an object file takes three steps, which do different work on each side:

| Step | Scaly | CasADi |
| --- | --- | --- |
| Build the Hessian | Traces the model into an expression graph, differentiates it, and finds the Hessian's sparsity pattern and coloring | Builds the SX or MX expressions and the IPOPT interface, which differentiates them into every oracle IPOPT needs, then simplifies the Hessian with `transform` |
| Generate C | Runs the [optimization passes](../how_it_works/lowering.md), which turn repeated stages into loops, and writes the C | Writes the expression graph as C with `CodeGenerator` |
| Compile with Clang | Compiles the C file at `-O3` | The same |

The first figure adds the three up. An open marker above the 180-second line marks a compile
stopped at the limit.

<div class="vega-chart" data-spec="../../assets/benchmarks/sweep_build.vl.json"></div>

Scaly keeps the race car's repeated stage as a loop over one compiled body, so its C source stays at
77 kB at every horizon, while CasADi's SX implementation writes 8.2 MB of C at `N = 400`. On
neural-process MPC and unbumpercars, Scaly takes about 2 to 5 seconds at every size. The chain
takes 3.2 seconds at `M = 3`, 13.3 at `M = 5` and 8.9 at `M = 9`. CasADi's mapped SX builds
faster at `M = 5`, in 10.8 seconds.

The next figure splits that time into its three steps, at each problem's size in the
[closed-loop benchmarks](closed_loop.md). Each problem has its own time axis. A bar too long to
fit is cut at the end of the axis: its parts keep the scale of the other bars, a break marks the
cut, and its true total follows the ellipsis.

<div class="vega-chart" data-spec="../../assets/benchmarks/sweep_build_stages.vl.json"></div>

CasADi generates C in under a second and spends 79 to 99% of its time in Clang. Scaly spends 25
to 75% of its time generating C, since that step includes its optimization passes, and usually
hands Clang a smaller file. On the chain, generating C takes 10 of Scaly's 13.3 seconds.

## Binary size

Binary size is the size of the object file Clang produces, as `size` reports it: machine code,
constant tables and static data, without the benchmark wrapper.

<div class="vega-chart" data-spec="../../assets/benchmarks/sweep_size.vl.json"></div>

Scaly's object file is smaller than every CasADi implementation's at every size. On the race car it
stays between 33 and 52 kB at every horizon. On neural-process MPC it stays between 62 and 190 kB.

The next figure compares the generated C file and the object file at each problem's closed-loop
size, and splits each into constant tables, such as sparsity patterns and numeric constants, and
everything else. Long bars are cut as in the time figure above.

<div class="vega-chart" data-spec="../../assets/benchmarks/sweep_size_contents.vl.json"></div>

The same tables weigh very differently in the two files. CasADi's mapped SX implementation stores a
third to more than half of its object file as tables. CasADi's MX implementation of unbumpercars
keeps 286 kB of tables in an object file whose C source holds under 3 kB of them, because a
zero-filled buffer takes one line of C. The two generators store constants differently, so these
sizes compare what each one ships, not the information in it.
