# Extremum C generation evidence

Measured on la015 with `cc -O3 -fno-math-errno -c`, three fresh object compilations
per source. No function run-time measurement was taken. Concurrent compiler-run
activity makes these compile times descriptive, not evidence of a speedup.

The `wide` snapshot is the only changed corpus entry. Its lane wrappers call
two scalar inline helpers that propagate either operand's NaN and evaluate each
operand once. A guard refuses fast-math flags that would discard this behavior.
The other snapshots remain byte-identical.

Comparison with main at `fb579fce03a68af1ba8e4238904faae387789653`:

| Quantity | main | This change |
| --- | ---: | ---: |
| Generated source bytes | 75178 | 75556 |
| Static elementwise Program operation nodes | 88 | 88 |
| Object compile seconds, three samples | 0.1001, 0.1226, 0.1240 | 0.2338, 0.0861, 0.1681 |

Counts are unique nodes from `walk_program(lower_function(_wide()))`, restricted
to the program operations in the elementwise table. They count each maximum and
minimum as one operation. Each inline extremum helper performs two scalar comparisons and two conditional
choices rather than one `fmin` or `fmax` call. Nested extrema call the helper
without repeating operand text. The source-size regression test bounds a
12-entry chain below 10 KB and checks that each expensive operand appears once.
The full operation histogram is unchanged: add 38, mul 26, div 3, and one each of
abs, acos, asin, atan, atan2, ceil, cos, cosh, erf, exp, floor, log, maximum,
minimum, pow, sin, sinh, sqrt, sub, tan and tanh.

The tie-convention gate was checked by temporarily changing the elementwise tie
weight from 0.5 to 0.25. Both extremum exact-partial cases failed; the change was
then restored. The initial public-builder tests failed because the builders did
not exist.

Integer index expressions remain inside the existing float64-only Function leaf
boundary until #56. Tests inspect their int64 dtype and nondifferentiability,
then cast them to float64 to compare compiled results with NumPy. Integer minimum
accumulators use an exact int64 identity so these builders never round indices
through a floating reduction.

The composed-extrema tie test and the transposed-view norm test adapt the split
convention cases in frozen `origin/devrush`, `tests/core/ad/test_derivatives.py`
and `tests/core/ad/test_reduction_derivative_edge_cases.py`. Configurable tie
conventions and segment extrema from those tests are outside this task.

Sparse extremum derivative masks and tie counts still scale with the full logical
shape. Follow-up #413 owns stored-coordinate scaling while preserving implicit
zero multiplicities and selection masks.
