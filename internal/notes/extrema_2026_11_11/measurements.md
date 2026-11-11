# Extremum C generation evidence

Measured on la015 with `cc -O3 -fno-math-errno -c`, three fresh object compilations
per source. No function run-time measurement was taken. Concurrent compiler-run
activity makes these compile times descriptive, not evidence of a speedup.

The `wide` snapshot is the only changed corpus entry. Its two scalar lane wrappers
replace `fmax` and `fmin` with comparisons and conditionals that propagate either
operand's NaN. Helper call names change accordingly. The other snapshots remain
byte-identical.

Comparison with main at `fb579fce03a68af1ba8e4238904faae387789653`:

| Quantity | main | This change |
| --- | ---: | ---: |
| Generated source bytes | 75178 | 75260 |
| Static elementwise Program operation nodes | 88 | 88 |
| Object compile seconds, three samples | 0.1001, 0.1226, 0.1240 | 0.1002, 0.1155, 0.2076 |

Counts are unique nodes from `walk_program(lower_function(_wide()))`, restricted
to the program operations in the elementwise table. They count each maximum and
minimum as one operation. Each emitted extremum now performs three scalar
comparisons and two conditional choices rather than one `fmin` or `fmax` call.
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
