# Compiler revision comparison

This study compares Scaly before, at [`e7f80e1f`](https://github.com/PREDICT-EPFL/scaly/commit/e7f80e1f),
and after, at [`5335c46d`](https://github.com/PREDICT-EPFL/scaly/commit/5335c46d), measured on November 12, 2026.
Both revisions use the same benchmark definitions, numerical inputs, native solver libraries
and compiler flags on the reference AMD Ryzen 9 7940HS desktop.
The results describe only these two revisions. They do not measure subsequent changes on `main`.

The changes depend on the kernel and size. Nine of the 27 Scaly kernel cells are faster,
16 are slower, and two remain unresolved at the measured noise level.
The chain Hessian at `M = 9` evaluates about seven times as fast, with a 95% interval of
5.5 to 9.3 times. At `M = 5`, evaluation takes 71% more time.
Neural-process model predictive control (MPC) Hessians are 16 to 21% slower across the
three measured horizons. The decoder-width Hessians take 21 to 42% more time.

## Kernel measurements

The first figure shows the evaluation time of each kernel at both revisions. Its selector also
shows generated-C compilation time, source size and operations per iteration.
Each panel has its own logarithmic axes.
Decoder width varies the hidden-layer width at a fixed horizon of 12.

<div class="vega-chart" data-spec="../../assets/benchmarks/compiler_run_values.vl.json"></div>

The second figure divides after by before. A ratio below one means less time, less code or
fewer operations. Timing bars show 95% intervals. Faint points mark equal values or unresolved
changes. For evaluation time, the entire interval must also clear the shaded 1.6% reference-kernel
variation range described below.
Compilation covers the generated C kernel, excluding the measurement wrapper and linker.
Operation counts cover one iteration of the equally repeated blocks that account for the most
arithmetic. They exclude work elsewhere in the kernel. A compiler change can change which blocks are included.

<div class="vega-chart" data-spec="../../assets/benchmarks/compiler_run_ratios.vl.json"></div>

Faster evaluation can cost more to build. At chain `M = 9`, generated C grows from 598 kB to
2,320 kB and C compilation grows from 5.88 to 20.76 seconds. The reported operation count falls
from 270,299 to 58,541. At `M = 5`, that count grows from 17,315 to 21,679.

Unbumpercars evaluation time also depends on size. Its Hessian takes 45% more time at `C = 2`,
3.5% more at `C = 8`, and 47% less at `C = 32`.

## Complete controllers

The chain controller comparison uses `M = 5`. Its mean function-evaluation time grows from
0.512 to 1.059 ms with IPOPT and from 0.224 to 0.435 ms with sequential quadratic programming
(SQP). The whole control step is 8.4% and 5.3% slower, respectively.

Unbumpercars control steps take 6.3% less time with IPOPT and 23% less with SQP. The latter
comparison has iteration differences at three of 200 steps, and its saved measurements do not
record oracle-call counts, so part of the 23% may come from different solver work.
Neural-process MPC SQP also has an iteration difference at one of 100 steps.
Neural-process MPC and race-car mean step times change by less than 1.4%, although race-car
function evaluation is 10 to 14% slower.

Each controller runs once per revision. Six of the seven CasADi controller pairs
have mean step times within 1.6%, but neural-process MPC SQP takes 10% less time in the second
run, with function evaluation down 17%. Controller changes of that size are indicative.
The kernel noise estimate below does not apply to these complete episodes.

## Measurement limits

Three fresh processes build each Scaly kernel. Four alternating timings of each saved binary
compare the revisions, increased to eight for noisy cells. CasADi's generated code does not
depend on the Scaly revision, so its identical compiled kernels are retimed as a reference that
should not change. Their kernel timing ratios fall within 1.6% of one. A reported evaluation-time
change must have its entire 95% interval outside that observed range. Each interval is computed
from the three build means of its own kernel and size, without a correction for comparing
27 cells together.

Every timed kernel passes its correctness gate, and every controller step succeeds.
Variation along one episode's steps does not estimate variation between processes.

Individual kernels use Clang 20.1.8 at `-O3`. Controllers use GCC 13.3.0 at `-O2`.
Both use native CPU instructions, glibc vector math, the performance governor and disabled boost.
The sparse chain Jacobian is checked against an unrolled Scaly stage, so that reference can
share differentiation errors. The Hessian checks use NumPy stage references, except unbumpercars,
which uses a separately constructed CasADi model.

The three-size grids leave intermediate scaling unmeasured. Newly added APIs have no matching
predecessor measurement here. This study uses one CasADi implementation per workload as a
reference. The [provider comparison](index.md) measures the wider set of implementations
in its separate October 29 study.
