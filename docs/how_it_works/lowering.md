# Lowering and optimization

Lowering is the step that turns a `Function` — a graph of values — into a program: loops, buffers,
loads and stores. It is the only path from the expression dialect to generated code. There is no
second renderer and no interpreter behind it, so anything lowering cannot express is a loud error
rather than a slow fallback.

## The rule registry

`passes/lowering.py` dispatches on `ExprOp` through a registry. Each rule is registered with
`@lowers(...)` and is responsible for one operation:

```python
@lowers(ExprOp.RESHAPE)
def _lower_reshape(ctx: LowerCtx, node: Expr) -> None:
    ...
```

The elementwise family is the exception, and a deliberate one: every unary op shares a single rule
driven by the `_UNARY` map from `ExprOp` to `ProgramOp`, and every binary op shares another. Adding
`asinh` is therefore a map entry, not a rule; adding a new structural operation is a rule.

`lower_function` walks the graph in topological order, emits one procedure per reached `Function`,
deduplicates callees so a block used a hundred times is lowered once, runs the optimization
pipeline, and verifies the result before returning it.

What is covered today: elementwise unary and binary with NumPy broadcasting; `reshape` as an alias;
`const` of any size through a constant buffer; general `slice` including integer, multi-dimensional
and strided forms; `sum`; `matmul` up to rank 2; `transpose` up to rank 4; `gather` and `scatter`
of any size through a `static const` index table; `stack` and `concat` on any axis; `call` across
multiple procedures; and `VMAP`. Solver calls use `ExprOp.SOLVER_CALL` and a callee carrying a solver descriptor.
The solver callee stays opaque and its wrapper is rendered separately, while its oracle functions
lower normally.

Not covered: device placement other than the host, and the operations listed as deferred in
[the expression dialect](expr_ir.md#operations). Both raise `LoweringError`.

## Program forms

**Loopy code**, or **loop form**, retains buffers and loops. **Scalarized code**, or **scalar form**,
expands eligible procedures into scalar calculations. Both are forms of the program dialect, and
one program can contain procedures in both forms. Loopy code can still contain scalar calculations
and undergo optimization. The existing `.block()` hint requests loopy form; it does not mean
matrix tiling or disabling optimization.

## The optimization pipeline

`optimize_program` runs the explicit `PASS_PIPELINE` sequence in `passes/program/__init__.py`
at the tail of lowering, between the initial program and the verifier. Each pass has its own
module. Adding an optimization means adding its function to this sequence at the required
position. Imports do not determine execution order.

Five passes run today, in this order.

**`scalarize`** expands selected float64 procedures before any buffer fusion or workspace reuse.
It substitutes constant loop indices, tracks the current scalar value of each buffer element,
and expands eligible pure callees. Views become scalar references, repeated expressions share
one value, and constant arithmetic and zero/one identities fold per element. Reductions keep
their original accumulation order. Shared values become typed scalar declarations. Single-use
arithmetic stays in expression trees, with a temporary inserted at depth 32 to bound rendering
depth and C parser nesting.

The `Expr.lowering` hint selects the containing procedure:

- `expr.scalar()` requests expansion even at the entry point and overrides its automatic size limits.
- `expr.block()` or `expr.opaque()` prevents expansion of the containing procedure. These hints
  take precedence if a body contains conflicting hints.
- `auto` admits at most 4,096 distinct scalar arithmetic operations per procedure after folding
  and sharing. Constants, variable references, and loads do not count as arithmetic. A separate
  program-wide limit of 16,384 counts arithmetic operations plus scalar declarations and output
  stores, so large copies also consume the budget. Earlier explicit scalarizations consume this
  capacity for later automatic candidates, but explicit requests always bypass the automatic limits.
- Before attempting automatic expansion, the pass limits work to 65,536 units per procedure:
  parameter elements, local buffer elements, executed stores, and nested call work. This prevents
  a large allocation or loop expansion merely to discover that the result exceeds the code budget.
  These limits are compiler heuristics, not API guarantees.

Automatic expansion leaves the entry point's mapped horizon intact. A procedure expands only if
all of its callees are eligible too. Solver calls retain their call boundaries. Float32 and integer
procedures retain their store boundaries because those stores can round or truncate values.
Constant tangents already inside a body fold during expansion. Specializing a mapped callee for
constant arguments is a separate transformation.

### Arithmetic semantics

Alloy applies algebraic simplifications without a math-mode option. Expression simplification
and scalar expansion can remove neutral elements, multiply by zero, cancel equal symbolic terms,
and simplify constant powers. For example, expression simplification can replace `x / x` with
one, and scalar expansion can replace `0 / x` with zero. The available rules and known constants
differ by compilation stage; a lowering hint does not select an IEEE 754 compliance mode.

These rules do not preserve NaN or infinity propagation, signed zero, or floating-point exception
behavior. They can also change intermediate rounding, overflow, or underflow. In particular,
symbolic `0 / x` can become zero even when the runtime value of `x` is zero or NaN. Do not rely
on an invalid operation surviving graph simplification to detect invalid model inputs.

When scalar expansion knows all operands, it evaluates constant arithmetic before applying
symbolic identities. Known `inf * 0` produces NaN. An invalid constant operation that the folder
cannot evaluate, such as `0 / 0` or `sqrt(-1)`, remains a runtime operation. Integer index division
retains C's truncation toward zero; floating algebraic rules do not relax index semantics or
permit removal of dtype rounding boundaries.

The scalarization pass keeps reduction accumulation order. This does not promise bit-identical
results across scalarized and loopy code: emitted expression trees and the selected C compiler
flags can also affect rounding. Alloy does not enable `-ffast-math` by default.

### Loopy-code optimizations

**`combine_scatter_sums`** replaces a left-associated sum of single-use zero-filled scatters with one zero-fill
and one scatter-add per term. Slice adjoints produce these scatters. Combining them removes
the full-length temporary buffer for each slice, including when slices overlap. Shared buffers
and aliases keep their storage. The pass leaves a sum unchanged if moving a scatter would
read a source after it changes.

**`fuse_elementwise`** inlines a single-use producer into its one consumer. A producer here is a
loop with a single store whose index is the loop variable — the shape every elementwise, slice and
gather lowering produces. The pass substitutes the producer's right-hand side at the consumer's
load site and deletes both the producer loop and its buffer. Chains collapse into one loop and the
intermediate round-trips through memory disappear.

It declines in one case. Operations that lower to a libm call — the transcendentals, `pow`,
`atan2` — are not duplicated into a consumer that would evaluate them more than once, which happens
when the consumer's iteration domain is larger than the producer's (a broadcast) or when there is
more than one read. Cheap arithmetic is always safe to duplicate; a `sin` is not.

**`unroll_unit_loops`** erases loops that are statically empty and inlines loops that run exactly
once, substituting the loop variable with its only value. It runs after fusion so that fusion sees
canonical loop-shaped producers first, and it removes the resulting single-iteration noise before
anything renders.

**`pack_workspace`** decides where temporaries live. It lifetime-packs the private buffers of each
procedure into shared slots — buffers whose lifetimes do not overlap reuse a slot — then spills
slots of 1024 doubles or more into the caller-provided `w[]` array while smaller ones stay as local
C arrays. The spilled total is what the generated header reports as `f_SZ_W`.

This is the pass that makes large workloads compile at all. Without it the biggest benchmark cells
declare every temporary as a C local and overflow the 8 MB stack. Zero-copy alias buffers are
handled explicitly: they own no slot but extend the lifetime of whatever they point into.

## A limitation worth knowing: recursion depth

The passes walk node trees recursively, so **depth in your expression becomes depth on the Python
stack**. `fuse_elementwise` is the binding constraint — it re-enters itself for every inlined
producer and once per argument, at roughly five stack frames per level of chaining. Against
CPython's default limit of 1000 frames, that runs out somewhere just under 200 chained scalar
operations, and it surfaces as a bare `RecursionError` during lowering rather than as a diagnosable
alloy error.

The shape that hits this is an accumulator written as a left fold — a per-stage cost built with
`cost = cost + <stage terms>` over a few dozen stages. Two ways around it, in order of preference:

1. **Write the accumulation as one flat reduction.** A sum of weighted squares is
   `al.dot(al.const(weights), al.stack(residuals) ** 2)`: a single `sum` over a wide `stack`, so
   the tree is shallow no matter how many stages there are. Recursion cost tracks tree *depth*, not
   *width* — wide nodes are free. This is what the race-car objective does.
2. **Sum pairwise**, as a balanced binary reduction, when a flat reduction does not fit. That takes
   depth from `O(n)` to `O(log n)`.

Raising `sys.setrecursionlimit` is not a fix: it trades a clean `RecursionError` for a possible
interpreter crash on the real C stack. Making the passes iterative is tracked on the roadmap, and
`tests/passes/test_program.py` pins both the current headroom and the failure, with an `xfail` that
flips the day the passes stop recursing.

## Watching it happen

Every step above is observable. Mark a function, call it, and the recorder captures the expression
graph, the lowered program, the result of each individual pass, and the generated C:

```python
from alloy.viz import visualize, serve

visualize(f)
f(x_value)
serve()
```

See [Visualization](../guide/visualization.md).
