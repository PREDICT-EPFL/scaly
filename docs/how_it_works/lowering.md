# Lowering and optimization

Lowering is the one step from the expression dialect to the program dialect,
described in the [architecture overview](architecture.md#why-two-dialects). This
page shows what the first program looks like and what each optimization pass
then does to it. Every listing is real output. The listings trim the attributes
of `prog.proc` lines, and trim more where they say so.

## One loop per operation

Lowering emits each expression operation on its own, with a buffer for its
result and a loop that fills it. For this function

```python
import scaly as sc

@sc.function(sc.G(sc.L("x", 8), sc.L("y", 8)), sc.L("out", ...))
def f(inputs):
    x, y = inputs
    return (x.sin() + y) * y
```

the program straight after lowering, before any optimization pass, is:

```
prog.module {
  prog.proc @f(%x: memref<8xfloat64>, %y: memref<8xfloat64>, %out: memref<8xfloat64>) {
    %t0 = prog.buffer : memref<8xfloat64, private>
    prog.for %i_t0 = 0 to 8 step 1 {kind=global} {
      prog.store prog.sin(prog.load %x[%i_t0]), %t0[%i_t0] : float64
    }
    %t1 = prog.buffer : memref<8xfloat64, private>
    prog.for %i_t1 = 0 to 8 step 1 {kind=global} {
      prog.store prog.add(prog.load %t0[%i_t1], prog.load %y[%i_t1]), %t1[%i_t1] : float64
    }
    prog.for %i_out = 0 to 8 step 1 {kind=global} {
      prog.store prog.mul(prog.load %t1[%i_out], prog.load %y[%i_out]), %out[%i_out] : float64
    }
  }
}
```

This form is easy to produce and to verify, one rule per operation, but it
writes two temporaries that nobody needs. Removing them is the passes' job,
shown [below](#elementwise-fusion). The last operation writes straight into the
output buffer, and a contiguous slice or a reshape becomes a view into existing
storage rather than a copy. A called function becomes its own procedure and a
`prog.call`, and `sc.vmap` becomes a loop around that call.

A few limits come from lowering rather than from the expression dialect. Matrix
multiplication takes operands of rank at most two, transpose works up to rank
four. An unsupported case raises
`LoweringError`. A solver call is not lowered at all. Its plugin renders the
wrapper, and only the oracles it calls go through this page's pipeline.

## Loops or scalar code

A small stage function is usually faster as straight-line scalar code than as a
set of three-iteration loops. With the default `auto` policy, a small stage
mapped over a horizon is expanded and then merged into the horizon loop:

```python
@sc.function(sc.L("x", 3), sc.L("y", ...))
def stage(x):
    return x.sin() * x

@sc.function(sc.L("xs", 15), sc.L("ys", ...))
def horizon(xs):
    return sc.vmap(stage, 5, [xs])
```

```
prog.module {
  prog.proc @horizon(%xs: memref<15xfloat64>, %ys: memref<15xfloat64>) {
    prog.for %v0 = 0 to 5 step 1 {kind=global} {
      %v1 = prog.assign prog.mul(3, %v0) : int64 {declare=True}
      %v2 = prog.assign prog.load %xs[%v1] : float64 {declare=True}
      %v3 = prog.assign %v1 : int64 {declare=True}
      %v4 = prog.assign prog.load %xs[prog.add(%v3, 1)] : float64 {declare=True}
      %v5 = prog.assign prog.load %xs[prog.add(%v3, 2)] : float64 {declare=True}
      prog.store prog.mul(prog.sin(%v2), %v2), %ys[prog.mul(3, %v0)] : float64
      prog.store prog.mul(prog.sin(%v4), %v4), %ys[prog.add(1, prog.mul(3, %v0))] : float64
      prog.store prog.mul(prog.sin(%v5), %v5), %ys[prog.add(2, prog.mul(3, %v0))] : float64
    }
  }
}
```

The horizon loop survives, and the stage body inside it is three scalar
statements. Ending the stage with `return (x.sin() * x).block()` keeps the
stage as a procedure with its own loop:

```
prog.module {
  prog.proc @stage(%x: memref<3xfloat64>, %y: memref<3xfloat64>) {
    prog.for %i_y = 0 to 3 step 1 {kind=global} {
      %v0 = prog.assign prog.load %x[%i_y] : float64 {declare=True}
      prog.store prog.mul(prog.sin(%v0), %v0), %y[%i_y] : float64
    }
  }
  prog.proc @horizon(%xs: memref<15xfloat64>, %ys: memref<15xfloat64>) {
    prog.for %it_ys = 0 to 5 step 1 {kind=global} {
      prog.call @stage(%xs[prog.mul(3, %it_ys)], %ys[prog.mul(%it_ys, 3)]) {callee_needs_w=False, n_in=1, n_out=1, w_self=0}
    }
  }
}
```

The expansion is done by the `scalarize` pass. It runs the procedure's loops
at compile time with symbolic values, folding arithmetic as it goes, so a
multiplication by a known zero disappears instead of becoming a statement. This
is partial evaluation[^pe] with every loop bound known.

The choice is driven by `Expr.lowering`, a field every expression node carries.
It is `"auto"` unless you set it with `.scalar()` or `.block()`, and it
propagates to the nodes built from the hinted one. The hints of a function's
nodes combine into the policy for its procedure:

| Hint | Effect on the containing procedure |
| --- | --- |
| none | `auto`: expand if the procedure is small and not the entry point |
| `.scalar()` | Expand, ignoring the size limits, including the entry point |
| `.block()` | Keep loops and buffers |

A `block` hint anywhere in a function wins over `.scalar()`. Under
`auto`, the size limits in `passes/program/scalarize.py` include 4096 scalar
operations per procedure and 16,384 across the program. Only procedures whose values
are all `float64` expand, because an integer store truncates and expansion
would drop that. A solver call also blocks expansion of its caller. A
derivative built from a hinted function inherits the hint, so a scalar stage model gets scalar derivative procedures too.

A hint only chooses between loops and scalar code. Every other pass still runs
on a `.block()` procedure.

## The pass pipeline

The passes run in the order of `PASS_PIPELINE` in
`src/scaly/passes/program/__init__.py`:

| Pass | What it does |
| --- | --- |
| `hoist_invariant` | Moves work that depends only on broadcast inputs out of a mapped loop |
| `scalarize` | Expands selected procedures into scalar statements |
| `fold_tiles` | Shrinks a constant table made of one repeated tile to that tile |
| `fuse_ranges` | Merges expanded stages into their mapped loop and drops unused outputs |
| `prune_procedures` | Removes procedures that nothing calls any more |
| `combine_scatter_sums` | Accumulates a sum of zero-padded scatters into one buffer |
| `fuse_elementwise` | Substitutes an elementwise producer into its consumer's loop |
| `fold_arith` | Replaces reads of constant buffers with constants and applies identities |
| `unroll_unit_loops` | Removes empty loops and inlines one-iteration loops |
| `fold_arith_after_unroll` | Runs `fold_arith` again on what unrolling exposed |
| `pack_workspace` | Shares storage between temporaries whose lifetimes do not overlap |
| `coalesce_stores` | Pairs stores to adjacent `float64` elements |
| `prepare_scalar` | Splits deeply nested scalar expressions into bounded statements |

The `reciprocal` and `lanes` options add two passes before `pack_workspace`.
`optimize_program` in the same file shows where. Each pass returns a new
program, and lowering verifies the result before rendering C.

## Work shared across stages

A stage often reads a parameter that is the same at every stage, such as a
weight matrix. The stage function cannot know that, so each call recomputes
everything derived from it:

```python
@sc.function(sc.G(sc.L("x", 3), sc.L("w", 9)), sc.L("y", ...))
def stage(inputs):
    x, w = inputs
    return (w.reshape((3, 3)).exp() @ x).sin()

@sc.function(sc.G(sc.L("xs", 15), sc.L("w", 9)), sc.L("ys", ...))
def horizon(inputs):
    xs, w = inputs
    return sc.vmap(stage, 5, [xs, w])
```

Here `w` has one chunk, so `vmap` passes the same nine values to all five
calls. Lowered, the loop calls `@stage` five times, and `@stage` evaluates
`exp` on all nine entries of `w` each time. `hoist_invariant` splits the stage
in two. The part that reads only `w` becomes a procedure called once before
the loop, and its result is passed to the rest of the stage as an extra input
(listing after `hoist_invariant`, trimmed to the horizon procedure):

```
prog.proc @horizon(%xs: memref<15xfloat64>, %w: memref<9xfloat64>, %ys: memref<15xfloat64>) {
  %it_ys_t0 = prog.buffer : memref<3x3xfloat64, private>
  prog.call @stage_hoist_1(%w[0], %it_ys_t0) {n_in=1, n_out=1}
  prog.for %it_ys = 0 to 5 step 1 {kind=global} {
    prog.call @stage_hoisted_1(%xs[prog.add(0, prog.mul(3, %it_ys))], %w[0], %it_ys_t0, %ys[prog.mul(%it_ys, 3)]) {n_in=3, n_out=1}
  }
}
```

Loop-invariant code motion is a standard compiler optimization[^dragon]. What
is particular here is that the invariance is found across a call boundary: the
mapped call says which arguments are broadcast, and the pass rewrites the
callee accordingly. After the later passes, the exponentials run once before
the loop and each iteration does only the matrix-vector product and the sine.

## Mapped stages and their consumers

When the caller uses only part of a mapped result, `fuse_ranges` propagates
that use into the loop. Taking the first element of each stage output:

```python
@sc.function(sc.L("x", 3), sc.L("y", ...))
def stage(x):
    return x.sin() * x

@sc.function(sc.L("xs", 15), sc.L("firsts", ...))
def firsts(xs):
    return sc.vmap(stage, 5, [xs])[::3]
```

```
prog.module {
  prog.proc @firsts(%xs: memref<15xfloat64>, %firsts: memref<5xfloat64>) {
    prog.for %v0 = 0 to 5 step 1 {kind=global} {
      %v1 = prog.assign prog.load %xs[prog.mul(3, %v0)] : float64 {declare=True}
      prog.store prog.mul(prog.sin(%v1), %v1), %firsts[%v0] : float64
    }
  }
}
```

The 15-element intermediate result is gone, and so is the work for the two
outputs per stage that nothing reads. The loop writes straight into `firsts`.
The pass does this only for stages that `scalarize` has already expanded, and it
leaves a buffer in place wherever a dependency, an overlapping write or a
pointer escaping into a call makes the substitution unsafe.

## Elementwise fusion

`fuse_elementwise` substitutes the expression that fills a buffer into the loop
that reads it, then deletes the buffer and its loop. The program from the first
section ends as one loop:

```
prog.module {
  prog.proc @f(%x: memref<8xfloat64>, %y: memref<8xfloat64>, %out: memref<8xfloat64>) {
    prog.for %i_out = 0 to 8 step 1 {kind=global} {
      %v0 = prog.assign prog.load %y[%i_out] : float64 {declare=True}
      prog.store prog.mul(prog.add(prog.sin(prog.load %x[%i_out]), %v0), %v0), %out[%i_out] : float64
    }
  }
}
```

Fusion trades memory traffic for recomputation when the consumer reads each
produced value more than once[^halide]. Broadcasting is the common case. Here
three sines feed twelve products:

```python
@sc.function(sc.G(sc.L("x", 3), sc.L("y", (4, 3))), sc.L("out", ...))
def f(inputs):
    x, y = inputs
    return x.sin() * y
```

```
prog.module {
  prog.proc @f(%x: memref<3xfloat64>, %y: memref<4x3xfloat64>, %out: memref<4x3xfloat64>) {
    %s0 = prog.buffer : memref<3xfloat64, private>
    prog.for %i_t0 = 0 to 3 step 1 {kind=global} {
      prog.store prog.sin(prog.load %x[%i_t0]), %s0[%i_t0] : float64
    }
    prog.for %i_out = 0 to 12 step 1 {kind=global} {
      prog.store prog.mul(prog.load %s0[prog.mod(%i_out, 3)], prog.load %y[%i_out]), %out[%i_out] : float64
    }
  }
}
```

Fusing would call `sin` twelve times instead of three, so the pass keeps the
buffer. It fuses freely when the producer is cheap arithmetic, and counts
expensive calls such as `sin`, `exp` or `pow` through chains of producers. It
also refuses to move a read past a write to the same buffer.

## Sums of scattered pieces

The gradient of a function of overlapping slices adds one zero-padded
contribution per slice. Lowered naively, each contribution is a full-size
temporary filled with zeros and then summed:

```python
@sc.function(sc.L("x", 6), sc.L("c", ...))
def cost(x):
    return (x[0:4].sin()).sum() + (x[2:6].cos()).sum()
```

```
prog.module {
  prog.proc @cost_grad_c_x(%x: memref<6xfloat64>, %grad_c_x: memref<6xfloat64>) {
    %t0 = prog.buffer : memref<4xfloat64, private> = prog.alias %x offset 2
    %t4 = prog.buffer : memref<4xfloat64, private> = prog.alias %x offset 0
    prog.for %z_t3 = 0 to 6 step 1 {kind=global} {
      prog.store 0, %grad_c_x[%z_t3] : float64
    }
    prog.for %i_t3 = 0 to 4 step 1 {kind=global} {
      %v0 = prog.assign prog.add(2, %i_t3) : int64 {declare=True}
      prog.store prog.sub(prog.load %grad_c_x[%v0], prog.sin(prog.load %t0[%i_t3])), %grad_c_x[%v0] : float64
    }
    prog.for %i_t6 = 0 to 4 step 1 {kind=global} {
      prog.store prog.add(prog.load %grad_c_x[%i_t6], prog.cos(prog.load %t4[%i_t6])), %grad_c_x[%i_t6] : float64
    }
  }
}
```

`combine_scatter_sums` zeroes the output once and has each contribution add
into its own window of it, so the two six-element temporaries never exist. The
`prog.alias` lines are the slices of `x`, which are views rather than copies.

## Repeated constant tables

A constant array that repeats one tile is stored as that tile and indexed
modulo its length:

```python
import numpy as np

@sc.function(sc.L("x", 12), sc.L("y", ...))
def scaled(x):
    return x * sc.const(np.tile([1.0, 2.0, 3.0], 4))
```

Before `fold_tiles`:

```
%k0 = prog.buffer : memref<12xfloat64, constant> = dense<[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]>
prog.for %i_y = 0 to 12 step 1 {kind=global} {
  prog.store prog.mul(prog.load %x[%i_y], prog.load %k0[%i_y]), %y[%i_y] : float64
}
```

After:

```
%k0 = prog.buffer : memref<3xfloat64, constant> = dense<[1.0, 2.0, 3.0]>
prog.for %i_y = 0 to 12 step 1 {kind=global} {
  prog.store prog.mul(prog.load %x[%i_y], prog.load %k0[prog.mod(%i_y, 3)]), %y[%i_y] : float64
}
```

The pass compares values bit for bit, so `-0.0` and `0.0` or two NaN payloads
are different values. It leaves a table alone if it is aliased by a view,
passed to a call or written, since other code could then depend on its full
length.

## Vector lanes

With the `lanes` option, the passes group independent loop iterations so the C
compiler can map them onto vector instructions. The
[code generation guide](../guide/codegen.md#cpu-targets-vector-lanes-and-math-libraries)
covers the options. This is the C for the fused loop of the first section, with
ten elements and `render_c_source(f, lanes=4, dialect="c")`, trimmed to the
lane helper and its caller:

```c
static inline void f_lanes_1(long long i_out_chunk, long long f_lanes_1_valid, double* out, const double* x, const double* y) {
  for (long long i_out_lane = 0; i_out_lane < f_lanes_1_valid; ++i_out_lane) {
    double v0 = y[(0 + (((i_out_chunk * SCALY_WIDTH_f_lanes_1) + i_out_lane) < 9 ? ((i_out_chunk * SCALY_WIDTH_f_lanes_1) + i_out_lane) : 9))];
    out[(0 + (((i_out_chunk * SCALY_WIDTH_f_lanes_1) + i_out_lane) < 9 ? ((i_out_chunk * SCALY_WIDTH_f_lanes_1) + i_out_lane) : 9))] = ((sin(x[(0 + (((i_out_chunk * SCALY_WIDTH_f_lanes_1) + i_out_lane) < 9 ? ((i_out_chunk * SCALY_WIDTH_f_lanes_1) + i_out_lane) : 9))]) + v0) * v0);
  }
}
int f(const double** arg, double** res, int* iw, double* w, int mem) {
  ...
  for (long long i_out_chunk = 0; i_out_chunk < 10 / SCALY_WIDTH_f_lanes_1; ++i_out_chunk) {
    f_lanes_1(i_out_chunk, SCALY_WIDTH_f_lanes_1, res[0], arg[0], arg[1]);
  }
#if (10 % SCALY_WIDTH_f_lanes_1) != 0
  f_lanes_1(10 / SCALY_WIDTH_f_lanes_1, 10 % SCALY_WIDTH_f_lanes_1, res[0], arg[0], arg[1]);
#endif
  return SCALY_SUCCESS;
}
```

The full groups and the final partial group call the same helper. The partial
group passes the number of valid lanes, and every index is clamped to the last
element, 9, so no lane reads past the end of an input. Only valid lanes are
stored. The arrays therefore need no padding, and their layout does not depend
on the width.

`SCALY_WIDTH_f_lanes_1` is a preprocessor macro, so the width is fixed when the
C is compiled. It is the requested width, capped for each loop by an estimate
of how many values the loop body keeps live against the number of vector
registers on the target, so a large body gets fewer lanes. It is also capped
at the smallest power of two not below the loop's iteration count. In the default
`gnu` dialect the helper uses a vector type, loading and storing whole groups
when all lanes are valid and falling back to a per-lane loop for the partial
group. A mapped loop is widened after its stage has been inlined. A loop stays
scalar if it touches integer values, still contains a call, or
carries a dependency between iterations other than an ordered sum.

## Temporary storage

`pack_workspace` lets temporaries share memory when their lifetimes do not
overlap. In this function, the product `A @ B` is dead once its sum is taken,
before `B @ A` is computed:

```python
@sc.function(sc.G(sc.L("A", (40, 40)), sc.L("B", (40, 40))), sc.L("out", ...))
def f(inputs):
    A, B = inputs
    return (A @ B).sum() + (B @ A).sum()
```

| Temporary | Elements | Holds | Live from | Until | Slot |
| --- | --- | --- | --- | --- | --- |
| `t0` | 1600 | `A @ B` | first product | first sum | `s0` |
| `t1` | 1 | first sum | first sum | final addition | `s1` |
| `t2` | 1600 | `B @ A` | second product | second sum | `s0` |
| `t3` | 1 | second sum | second sum | final addition | `s2` |

The two products share slot `s0`. The two sums cannot share, because `t1` is
still live when `t3` is written. The pass walks the temporaries in the order
they are first written and gives each the first free slot of the same data type,
a greedy interval assignment like linear-scan register allocation[^linscan].
A view such as a slice keeps the buffer it points into live until the view's
last use.

A slot of 1024 or more elements moves to the caller's workspace `w`, and
smaller ones stay local arrays:

```c
double* s0 = w + 0;
double s1[1];
double s2[1];
```

The header reports `#define f_SZ_W 1600`, where 3200 would be needed without
sharing. A procedure's requirement is its own slots plus the largest
requirement among the procedures it calls, which reuse the space after the
caller's own slots. [Working memory](../guide/codegen.md#working-memory) in the
guide shows how to read the number.

## Arithmetic semantics

Scaly simplifies expressions as algebra over real numbers, not as IEEE 754
floating-point arithmetic[^goldberg]. The two disagree at invalid and
exceptional inputs:

```python
import numpy as np

@sc.function(sc.L("x", 4), sc.L("y", ...))
def f(x):
    return 0 / x + x / x

print(f(np.array([0.0, np.nan, np.inf, 2.0])))  # [1. 1. 1. 1.]
```

| `x` | `0.0` | `nan` | `inf` | `2.0` |
| --- | --- | --- | --- | --- |
| NumPy, `0 / x + x / x` | `nan` | `nan` | `nan` | `1.0` |
| Scaly | `1.0` | `1.0` | `1.0` | `1.0` |

`0 / x` became zero and `x / x` became one before any C existed. In general,
simplification can change how NaN and infinity propagate, the sign of zero,
overflow, underflow and intermediate rounding. Do not use an invalid operation
in a model to detect invalid input.

The same holds for constants. The identities apply to a constant operand like
any other, and they can apply before the constants are evaluated. So
`sc.const(np.inf) * 0` becomes `0`, because `x * 0 = 0` applies first, and it
never reaches the NaN that IEEE 754 gives. This is intended: simplification
treats every operand as a real number. To put a NaN in a model, write
`sc.const(np.nan)`, which is kept as is.

Scalar expansion and lane widening keep the order in which a reduction adds its
terms. Scalar and loop forms of the same function are still not promised to
give bit-identical results, because the expression trees differ and the C
compiler rounds each one its own way. Neither the JIT nor the recipe for
exported code enables `-ffast-math`. Both pass `-fno-math-errno`, and the
lowering hints do not change these flags.

`reciprocal=True` allows one further change. A division `x / y` whose divisor
does not change inside a loop becomes a multiplication by `1 / y` computed
before the loop. For `x / s`, with `x` of four elements and `s` a scalar input,
the `reciprocal=True` render option gives, trimmed to the procedure body:

```
%inv = prog.assign prog.div(1, prog.load %s[0]) : float64 {declare=True}
prog.for %i_y = 0 to 4 step 1 {kind=global} {
  prog.store prog.mul(prog.load %x[%i_y], %inv), %y[%i_y] : float64
}
```

This changes rounding and can overflow where the division does not:
`1e-310 / 1e-310` is `1.0`, but `1 / 1e-310` is already `inf`. Vector math
libraries can also round differently from scalar math calls. Neither option
enables any other fast-math behaviour.

[^pe]: Neil D. Jones, Carsten K. Gomard and Peter Sestoft, *Partial Evaluation
    and Automatic Program Generation*, Prentice Hall, 1993.
    [Online edition](https://raspi.itu.dk/people/sestoft/pebook/).
[^dragon]: Alfred V. Aho, Monica S. Lam, Ravi Sethi and Jeffrey D. Ullman,
    *Compilers: Principles, Techniques, and Tools*, 2nd edition,
    Addison-Wesley, 2006, chapter 9.
[^halide]: Jonathan Ragan-Kelley et al., "Halide: a language and compiler for
    optimizing parallelism, locality, and recomputation in image processing
    pipelines", PLDI 2013.
    [doi:10.1145/2491956.2462176](https://doi.org/10.1145/2491956.2462176).
[^linscan]: Massimiliano Poletto and Vivek Sarkar, "Linear scan register
    allocation", ACM Transactions on Programming Languages and Systems 21(5),
    1999. [doi:10.1145/330249.330250](https://doi.org/10.1145/330249.330250).
[^goldberg]: David Goldberg, "What every computer scientist should know about
    floating-point arithmetic", ACM Computing Surveys 23(1), 1991.
    [doi:10.1145/103162.103163](https://doi.org/10.1145/103162.103163).
