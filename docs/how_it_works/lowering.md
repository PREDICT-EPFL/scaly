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

The elementwise family is the exception: an op with the `elementwise` trait shares a single rule,
and the trait's value is the `ProgramOp` that computes one entry. Adding `asinh` is therefore a
trait; adding a new structural operation is a rule.

`lower_function` walks the graph in topological order, emits one procedure per reached `Function`,
deduplicates callees so a block used a hundred times is lowered once, runs the optimization
pipeline, and verifies the result before returning it.

It lowers for a target, the `sc.Target` describing the processor the code is tuned for (the one in
force unless `target=` names another). A rule reads it as `ctx.target`, and it is recorded on the
`PROGRAM` node as its `tuned_for` attribute for the program passes, so a choice that depends on the
vector width or a cache size is made where the rest of the lowering is decided, and the renderer
still only spells what it is given. The listing leaves the target out: it says what the program was tuned for, not
what it computes.

Before allocating buffers, lowering normalizes a private copy of each ordinary Function's outputs
with the expression rewrite rules. The copy keeps the declared inputs and output metadata.
Lowering captures the Function's effective hint before rewriting and applies it to the procedure
without copying shared output expressions. The user's graph and the graph used by differentiation
stay intact.

Covered today: elementwise unary and binary with NumPy broadcasting; `reshape` as an alias;
`const` of any size through a constant buffer; general `slice` including integer,
multi-dimensional and strided forms; `sum`; `matmul` up to rank 2; `transpose` up to rank 4;
`gather`, `segment_reduce` and `put`/`put_add` at constant indices, of any size, through indices
that are arithmetic on the loop index where they are affine and a `static const` table where they
are not; `put`/`put_add` and `take` at run-time indices; `stack` and `concat` on
any axis; `call` across multiple procedures; and `VMAP`. A call to a Function with an extern body
(`ExprOp.EXTERN_CALL`, a solver for instance) stays opaque: its callee renders its C separately,
while the Functions that C calls lower normally.

Not covered: device placement other than the host, and the operations listed as absent in
[the expression dialect](ir.md#operations). Both raise `LoweringError`.

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
position; a package outside the compiler inserts its pass at a named slot instead
(`insert_after`, `insert_before`). Imports do not determine execution order.

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
- `auto` leaves a procedure unexpanded when it holds a matrix times a matrix whose rows fill the
  target's middle column block (8 columns on Apple silicon) over four terms or more: expanded, its
  outputs become scalar chains the C compiler does not vectorize, where its loops vectorize across
  the block.
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

Automatic expansion leaves the entry point's call boundaries intact, its mapped horizon among
them: an entry point that calls a procedure, in a loop or not, keeps its loopy form. One that calls
nothing expands under the same op budgets as a callee, but only up to 4,096 units of work, since a
rejected attempt costs generation time for nothing. Small standalone kernels, such as a dense
Jacobian written without calls, are then straight-line code whose identity seeds fold away. A
procedure expands only if all of its callees are eligible too. Solver calls keep their call boundaries. Float32 and integer
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
and simplify constant powers. For example, expression simplification can replace `x - x` with
zero, and scalar expansion can replace `0 / x` with zero. The available rules and known constants
differ by compilation stage; a lowering hint does not select an IEEE 754 compliance mode.

These rules do not preserve NaN or infinity propagation, signed zero, or floating-point exception
behavior. They can also change intermediate rounding, overflow, or underflow. In particular,
symbolic `0 / x` can become zero even when the runtime value of `x` is zero or NaN. Do not rely
on an invalid operation surviving graph simplification to detect invalid model inputs.

A quotient of a term by itself is the exception. `x / x` stays a division, so it is NaN where
`x` is zero, infinite or NaN, as in NumPy.

When scalar expansion knows all operands, it evaluates constant arithmetic before applying
symbolic identities. Known `inf * 0` produces NaN. An invalid constant operation that the folder
cannot evaluate, such as `0 / 0` or `sqrt(-1)`, remains a runtime operation. Integer index division
keeps C's truncation toward zero; floating algebraic rules do not relax index semantics or permit
removal of dtype rounding boundaries.

The scalarization pass keeps reduction accumulation order. This does not promise bit-identical
results across scalarized and loopy code: emitted expression trees and the selected C compiler
flags can also affect rounding. Scaly does not enable `-ffast-math` by default.

That order is the lowering's, and it is not a single chain. A `sum` or a `dot` of eight elements
or more adds into four partial sums, element `k` into partial `k % 4`, the elements past the last
whole block of four into the first, and combines them as `(p0 + p1) + (p2 + p3)`; a matrix-vector
product with eight columns or more runs four rows per pass, each row in four partial sums the same
way, and stores each output once. One chain of dependent adds runs at the latency of an add; four
overlap, and the C compiler pairs them into vector lanes, which is what `-ffast-math` would buy by
reordering freely. The rounding is a blocked sum's, as in NumPy's pairwise `sum`, and it is fixed
by the generated code rather than by the compiler. One reduction takes its count of partial sums
from the target. It is the row of a triangular solve with one right-hand side and no transpose, a
dot product as long as the row, which runs in `Target.sum_lanes` of them, a vector for each
multiply-add unit and no fewer than four, eight on the reference machine. Shorter reductions keep
one chain, and so do
the matrix products whose reduction axis is the matrix's slow one, `x @ b` and `a @ b`: each output
sums its terms in order of `k`. An output row runs in blocks of 8, 4 and 2 vector registers'
worth of columns, 16, 8 and 4 on a 128-bit machine (`Target.row_blocks`), whose sums stay in
registers across the reduction, each output stored once; the columns no block covers, and every
column of a row wider than four of the widest blocks (`Target.row_blocked_max`, 64 columns on
Apple silicon), put the reduction loop outermost and vectorize over the row instead, adding into
the output at every step. A block keeps its sums in private buffers of one vector's lanes, each
updated in a loop of kind `vector` that the C renderer writes as GNU vector statements, so the
generator vectorizes them rather than leaving it to the C compiler. Each output's terms are summed
in the same order whichever target the code is rendered for, and a block's multiply-adds stay one
expression each, which the C compiler fuses. The code around the blocks is left to the compiler's
own vectorizer, which fuses a multiply and an add in some shapes and not in others, so another
target's widths can still change the last bits. Under `rounding="portable"` every target takes the
M3's widths.

A matrix of as many rows as a register tile holds or more runs in tiles instead, four rows by eight
columns on the M3 (`Target.product_tile`), chosen from the processor's multiply-add units, their
latency and its registers as BLIS's analytical model chooses them, so that enough independent sums
hide the latency and each step of `k` loads four elements of `a` and four vectors of `b` for sixteen
vector multiply-adds. Every tile of rows reads all of `b`, so the rows go in the fewest tiles the
registers hold, of even heights. A tile may take half as many rows again as the model's
(`Target.tile_rows_max`, six on the M3), which measured as fast on a `b` in the cache and faster on
a larger one. Ten rows are two tiles of five, where tiles of four would leave two rows to read `b` a
third time. The columns fewer than a vector take a tile of scalar sums. Each output still sums in
order of `k`, so a tile computes what a row block does.

A constant table of floating-point values larger than the level-1 data cache (`Target.l1d_bytes`)
is declared aligned to 64 bytes. The C compiler aligns a table to its element, eight bytes, and the
vector loads of a constant `b` then straddle two cache lines every few loads, which costs when each
line comes from the level-2 cache. Tables that fit the cache, and index tables, are left as the
compiler places them.

A matrix reads `b` once per tile of rows, or once per row when it has fewer rows than a tile, so a
large `b` runs its column blocks outermost instead. In tiles that is a `b` larger than the level-1
data cache from 64 rows, or larger than half the level-2 cache (`Target.l2_bytes`) from sixteen,
which every tile of rows would otherwise read from memory again; below that a tile of rows reads
`b` in place fast enough. In rows, it is a `b` wider than that limit from six rows or larger than
the level-1 data cache from sixteen, which on a target of several lanes means a product of up to
three rows, the fewest a tile does not hold.
Each block copies its panel of `b` into a contiguous buffer, a chunk of `k` at a time small enough
to stay in the cache, and every row of `a` passes over the chunk with its sums in registers. The
copy is what keeps the panel in the cache when `n` is a power of two: read in place, down a column
of the matrix, its rows would share a few of the cache's sets and evict each other. The first chunk
starts each output's sum at zero and every later one resumes it from the stored partial sum, so
each output is still one chain of multiply-adds in order of `k`.

A `transpose` lowers to one flat loop over its output whose coordinates divide the loop variable.
Fusion can then inline it into its one consumer. A gather reading a few entries of a transposed
product, as the recovery of a sparse derivative does, then computes only those entries. Read through
a gather's index table, the transposed coordinates are a division and a remainder of the table's
entry, and `fold_arith` computes those when the code is generated. The two indices compose into one
table, so a transpose that only moves data is never copied in front of a gather at constant
indices. In front of a gather at run-time indices it stays a copy, and `delinearize_loops` splits
any transpose that stays back into a loop per axis without the divisions. A gather whose map repeats
with a period reads a table at the remainder, and the split takes the index apart at that read. The
result is one loop per period and one within it, with the table read at the inner coordinate.

A `max` or `min` reduction propagates NaN as `np.max` does, so its step is a select C's `fmax`
cannot express: `((cur < x) || (x != x)) ? x : cur`. The renderer spells that select
`SCALY_FMAX_NAN(cur, x)` (and the minimum `SCALY_FMIN_NAN`), defined at the top of a source that
uses it as clang's `__builtin_elementwise_maximum`, the IEEE 754-2019 maximum and one instruction on
AArch64, where the compiler has it, and as the select anywhere else. The two differ only in the
sign of a zero result, which the reduction's lanes already leave open.

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

The same temporaries make selects what the C compiler can vectorize. C evaluates only one branch of
`c ? a : b` and skips the right operand of `&&` and `||`, and a load that C would skip is one the
compiler may not move ahead of the condition, so it keeps a branch and the loop around it stays
scalar. A graph evaluates every operand anyway, so inside a loop the scheduler names such an operand
in a temporary before the statement: `where(mask, 1.0 / x, 0.0)` renders as the reciprocal of every
element and then a select.

Computing a branch for every element is a trade against a branch the processor predicts, which
skips the work. The scheduler takes it only where it pays:

- One of the select's two branches must be no work at all (a constant, a named value, a load). A
  piecewise function with arithmetic in every piece keeps its branches.
- A variable must enter the condition. A select on a flag keeps its branches, and the C compiler
  tests the flag once outside the loop.
- The branch must be a float computed with operations that are one instruction under every
  compiler and cannot fault. A libm call, a square root, a minimum, a floor, an integer division
  and a float converted to an integer stay under their condition.
- A running sum keeps its branch, because the sum is a chain through one element whatever the
  compiler does with the select.
- Straight-line code keeps its branches, since it has no loop to vectorize.

A loop the C compiler vectorizes can differ in the last bit from the same loop left scalar, because
the compiler fuses multiply-adds differently in vector code.

## Deep expressions

Depth in your expression does not become depth on the Python stack. The program-dialect passes
and the C renderer walk node graphs iteratively: rewrites go through the shared driver
`scaly.ir.match.rewrite`, which uses an explicit stack and one identity-keyed memo, and the
remaining traversals (`_max_load_executions`, `_count_buf_loads`, the scalarizer's value
substitution, `_emit_scalar`) keep their own explicit stacks. A left fold of several thousand
chained scalar operations lowers, renders, compiles and runs; `tests/core/passes/test_program.py` pins
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
