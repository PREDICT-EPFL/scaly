# Lowering and optimization

Lowering turns a `Function`, a graph of values, into a program: loops, buffers, loads and stores.
It is the only path from the expression dialect to generated code. There is no second renderer
and no interpreter behind it, so anything lowering cannot express raises.

## The rule registry

`passes/lowering.py` dispatches on `ExprOp` through a registry. Each rule is registered with
`@lowers(...)` and is responsible for one operation:

```python
@lowers(ExprOp.RESHAPE)
def _lower_reshape(ctx: LowerCtx, node: Expr) -> None:
    ...
```

The elementwise family is the exception: every unary op shares a single rule driven by the
`_UNARY` map from `ExprOp` to `ProgramOp`, and every binary op shares another. Adding `asinh` is
therefore a map entry; adding a new structural operation is a rule.

`lower_function` walks the graph in topological order, emits one procedure per reached `Function`,
deduplicates callees so a block used a hundred times is lowered once, runs the optimization
pipeline, and verifies the result before returning it.

Before allocating buffers, lowering normalizes a private copy of each ordinary Function's outputs
with the expression rewrite rules. The copy keeps the declared inputs and output metadata.
Lowering captures the Function's effective hint before rewriting and applies it to the procedure
without copying shared output expressions. The user's graph and the graph used by differentiation
stay intact.

Covered today: elementwise unary and binary with NumPy broadcasting; `reshape` as an alias;
`const` of any size through a constant buffer; general `slice` including integer,
multi-dimensional and strided forms; `sum`; `matmul` up to rank 2; `transpose` up to rank 4;
`gather` and `scatter` of any size through a `static const` index table; `stack` and `concat` on
any axis; `call` across multiple procedures; and `VMAP`. Solver calls use `ExprOp.SOLVER_CALL` and
a callee carrying a solver descriptor. The solver callee stays opaque and its wrapper is rendered
separately, while its oracle functions lower normally.

Not covered: device placement other than the host, and the operations listed as absent in
[the expression dialect](expr_ir.md#operations). Both raise `LoweringError`.

## Program forms

Loopy code, or loop form, keeps buffers and loops. Scalarized code, or scalar form, expands
eligible procedures into scalar calculations. Both are forms of the program dialect, and one
program can contain procedures in both forms. Loopy code can still contain scalar calculations
and undergo optimization. The `.block()` hint requests loopy form; it does not mean matrix tiling
or disabling optimization.

## The optimization pipeline

`optimize_program` runs the explicit `PASS_PIPELINE` sequence in `passes/program/__init__.py`
at the tail of lowering, between the initial program and the verifier. Each pass has its own
module. Adding an optimization means adding its function to this sequence at the required
position. Imports do not determine execution order.

The sequence starts with hoisting and scalar expansion, then cleans up loops and storage, and
ends with explicit store pairing and scalar preparation.

### `hoist_invariant`

Runs first, on the loop-shaped program. A `VMAP` that broadcasts an argument (stride 0) lowers to
a loop whose call passes the same pointer at every trip, but the callee cannot know that and
recomputes everything derived from it. The pass finds, inside the callee, the private buffers
written only by statements that read invariant inputs, constant buffers and other invariant
buffers, and splits the callee: a `_hoist_<positions>` prologue computes those buffers once before
the loop, and a `_hoisted_<positions>` body takes them as extra inputs. The suffix names the
invariant argument positions, so one callee mapped two ways gets two distinct splits. A buffer
that any per-trip statement also writes, such as a zero-filled accumulator, stays in the body.
Call sites whose arguments all vary keep the original callee, which is dropped once nothing calls
it.

Under `auto` the prologue has `scalarize_mode="inline"`: `scalarize` inlines it into an expanding
caller but never expands it on its own, since code that runs once per call gains nothing from
expansion and would grow the source with the invariant argument's size. Other selection modes are
`disabled` and `procedure`; the separate lowering hint controls automatic budgets. Generated
procedure names reserve existing names and separate argument positions unambiguously. Opaque and
solver-bearing calls stay in place.

### `scalarize`

Expands selected float64 procedures before any buffer fusion or workspace reuse. It substitutes
constant loop indices, tracks the current scalar value of each buffer element, and expands
eligible pure callees. Views become scalar references, repeated expressions share one value, and
constant arithmetic and zero/one identities fold per element. Reductions keep their original
accumulation order. Shared values become typed scalar declarations. Single-use arithmetic stays
in expression trees, with a temporary inserted at depth 32 to bound rendering depth and C parser
nesting.

The `Expr.lowering` hint selects the containing procedure:

- `expr.scalar()` requests expansion even at the entry point and overrides its automatic size limits.
- `expr.block()` or `expr.opaque()` prevents expansion of the containing procedure. These hints
  take precedence if a body contains conflicting hints.
- `auto` admits at most 4,096 distinct scalar arithmetic operations per procedure after folding
  and sharing. Constants, variable references and loads do not count as arithmetic. A separate
  program-wide limit of 16,384 counts arithmetic operations plus scalar declarations and output
  stores, so large copies also consume the budget. Earlier explicit scalarizations consume this
  capacity for later automatic candidates, but explicit requests always bypass the automatic limits.
- Before attempting automatic expansion, the pass limits work to 65,536 units per procedure:
  parameter elements, local buffer elements, executed stores, and nested call work. This avoids
  a large allocation or loop expansion only to discover that the result exceeds the code budget.
  These limits are compiler heuristics, not API guarantees.
- Derivative Functions that AD (automatic differentiation) builds for a `call` or `vmap` callee
  (its tangent, adjoint and adjoint-tangent bodies) inherit the callee's effective hint: `block`
  or `opaque` anywhere in the callee makes the derived body `block`, otherwise `scalar` anywhere
  makes it `scalar`, and `auto` inherits nothing. A `.scalar()` on a stage output therefore selects
  the stage's Hessian procedures too.

Automatic expansion leaves the entry point's mapped horizon intact. A procedure expands only if
all of its callees are eligible too. Solver calls keep their call boundaries. Float32 and integer
procedures keep their store boundaries because those stores can round or truncate values.
Constant tangents already inside a body fold during expansion. Specializing a mapped callee for
constant arguments is a separate transformation.

### `prune_procedures`

Removes procedures made unreachable by call expansion. It keeps the entry point, solver oracles
and callees of retained kernels, follows their calls, and preserves callee-before-caller order.
Hoisting uses the same reachability analysis after splitting callees.

### Arithmetic semantics

Scaly applies algebraic simplifications without a math-mode option. Expression simplification
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
keeps C's truncation toward zero; floating algebraic rules do not relax index semantics or permit
removal of dtype rounding boundaries.

The scalarization pass keeps reduction accumulation order. This does not promise bit-identical
results across scalarized and loopy code: emitted expression trees and the selected C compiler
flags can also affect rounding. Scaly does not enable `-ffast-math` by default.

### Loopy-code optimizations

`combine_scatter_sums` replaces a left-associated sum of single-use zero-filled scatters with one
zero-fill and one scatter-add per term. Slice adjoints produce these scatters. Combining them
removes the full-length temporary buffer for each slice, including when slices overlap. Shared
buffers and aliases keep their storage. The pass leaves a sum unchanged if moving a scatter would
read a source after it changes.

`fuse_elementwise` inlines a single-use producer into its one consumer. A producer here is a loop
with a single store whose index is the loop variable, the shape every elementwise, slice and
gather lowering produces. The pass substitutes the producer's right-hand side at the consumer's
load site and deletes both the producer loop and its buffer. Chains collapse into one loop and the
intermediate round-trips through memory disappear.

It checks the fully expanded producer chain, including source writes and aliases. Operations that
lower to a libm call (the transcendentals, `pow`, `atan2`) are not duplicated into a consumer that
would evaluate them more than once, which happens when the consumer's iteration domain is larger
than the producer's (a broadcast) or when there is more than one read. Cheap arithmetic can be
duplicated only when moving its source reads is safe.

`fold_arith` runs the arithmetic identities shared with the expression dialect
(`passes/arith.py`) over every loop body after fusion, where index substitution exposes `x * 1`,
`x + 0` and constant index arithmetic, and evaluates all-constant scalars with the operation's
dtype. A loop that fills a private, otherwise unwritten buffer with one constant becomes a constant
buffer so its readers fold on the next round. Invalid constants such as `0 / 0` stay runtime
operations.

`unroll_unit_loops` erases loops that are statically empty and inlines loops that run exactly
once, substituting the loop variable with its only value. It runs after fusion so that fusion sees
canonical loop-shaped producers first, and it removes the resulting single-iteration noise before
anything renders.

A second arithmetic cleanup, recorded as `fold_arith_after_unroll`, resolves constant indices and
identities exposed by loop substitution, then removes unused buffer declarations.

`pack_workspace` decides where temporaries live. It lifetime-packs the private buffers of each
procedure into shared slots (buffers whose lifetimes do not overlap reuse a slot), then spills
slots of 1024 doubles or more into the caller-provided `w[]` array while smaller ones stay as local
C arrays. The spilled total is what the generated header reports as `f_SZ_W`.

This pass is what lets large workloads compile at all. Without it the biggest benchmark cells
declare every temporary as a C local and overflow the platform's default thread stack (8 MB on
the reference machine). Zero-copy alias buffers are handled explicitly: they own no slot but
extend the lifetime of whatever they point into.

`coalesce_stores` replaces eligible adjacent float64 stores with an explicit `STORE_PAIR`
statement after workspace packing. The two values are evaluated before either lane is written.
Pair selection accounts for physical buffer aliases and refuses a pair when the second value
reads the first destination. Odd tails stay scalar.

`prepare_scalar` inserts typed temporaries to bound expression depth before C rendering. It
shares the scheduler used by scalar expansion, but limits sharing to a statement so loads keep
their timing across writes and calls. Generated names reserve existing C identifier spellings.

## Deep expressions

Depth in your expression does not become depth on the Python stack. The program-dialect passes
and the C renderer walk node graphs iteratively: rewrites go through the shared driver
`scaly.ir.match.rewrite`, which uses an explicit stack and one identity-keyed memo, and the
remaining traversals (`_max_load_executions`, `_count_buf_loads`, the scalarizer's value
substitution, `_emit_scalar`) keep their own explicit stacks. A left fold of several thousand
chained scalar operations lowers, renders, compiles and runs; `tests/passes/test_program.py` pins
folds at 400 and 3000 and a flat per-stage reduction at 100 stages.

The generated C stays bounded too. Clang caps bracket nesting at 256, so `prepare_scalar` splits a
fused chain deeper than `MAX_SCALAR_DEPTH` (32, shared with the scalarizer's temporary scheduling)
into scalar temporaries. Store values, indices and call offsets use the same depth bound. Range
expressions stay unchanged so start, stop and step keep their evaluation frequency; the depth
bound does not apply to hand-built deep range expressions. Depth is still cheaper to avoid than to
render: a wide flat reduction (`sc.dot(sc.const(weights), sc.stack(residuals) ** 2)`) or a pairwise
sum reads better in the generated source than a long fold, but neither is required for
correctness.

What still recurses is proportional to statement nesting, not expression depth: `FOR` bodies in
`unroll_unit_loops`, the scalarizer's `run` and the renderer's `_emit_statement`, and `CALL`
chains in the scalarizer. Loop nests are a few levels deep.

## Watching it happen

Every step above is observable. Mark a function, call it, and the recorder captures the expression
graph, the normalized outputs of each reached Function, the lowered program, the result of each
pass, and the generated C:

```python
from scaly.viz import visualize, serve

visualize(f)
f(x_value)
serve()
```

See [Visualization](../guide/visualization.md).
