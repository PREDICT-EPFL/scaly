# How differentiation works

Scaly differentiates graphs. Every derivative is a new expression-dialect graph built from the old
one, so a derivative is an ordinary `Function` that is lowered, optimized and compiled like the
function it came from. There is no tape and no runtime.

The user-facing side is in [Derivatives](../guide/derivatives.md). This page covers the parts that
matter when you change them.

## Two modes, kept separate

Forward and reverse mode are independent implementations in `ad/forward.py` and `ad/reverse.py`.
They share no dispatch and no rule table. The one thing they have in common, the structural "does
this depend on that" query, sits below both in `ad/sparsity.py`, which contains no automatic
differentiation (AD).

`ad/derivatives.py` assembles the whole-derivative constructions from them:

| Construction | How |
| --- | --- |
| `gradient(y, x)` | one reverse sweep with a seed of ones. Requires a scalar `y`. |
| `jacobian(y, x)` | forward mode, batched: the `x.size` identity columns are stacked into one seed of shape `(x.size, *x.shape)` and pushed through `jvp_many` in a single pass, then transposed. |
| `hessian(y, x)` | `jacobian(gradient(y, x), x)`: one reverse sweep, then batched forward over its graph. |

Batching the Jacobian keeps it from becoming a loop of independent sweeps. The multi-seed rules
share the expensive parts across columns (one `cos` serves every column of a `sin`'s derivative) and
turn per-column chain-rule unrolling into small matrix products.

`SUM` has a structural multi-seed rule as well. It reshapes the seed-batched tangent to
`(nseed, input.size)` and multiplies each row by an all-ones vector. The reduction stays in one
graph instead of becoming one Python-unrolled reduction per seed, so generated source does not grow
with the seed count. The generated loop still performs one extra multiply for each summed element
and seed. The benchmark harness's `dispatch_arithmetic` metric counts those multiplies, so they
appear in the reported work.

Not every operation has a multi-seed rule yet. `abs`, `asin`, `acos`, `atan`, `atan2`, `minimum`,
`maximum`, `floor`, `ceil`, and any `transpose` whose result would be rank 4 or higher (the seed
axis would push it past the rank-4 lowering limit) fall back to evaluating seeds one at a time. The
fallback is silent by default because it is correct, only slower. Set `SCALY_STRICT_JVP_MANY=1` to
make it raise instead, which is what you want when adding a rule and checking that it is used.

## Differentiating across a call

A `call` node exists so that the callee's structure survives into generated code. AD has to
preserve that, or a horizon of a hundred identical stages becomes a hundred copies of the stage
derivative.

Forward mode builds a derivative function and calls it. One derivative `Function` propagates all
active formal inputs together for a callee output. Its cache key includes the active formal
indices, the seed count, and any baked seed values. The function takes only the primal arguments
and seed inputs its derivative depends on, so a stage derivative can have a narrower signature than
the stage itself. Constant and runtime seeds use the same helper construction; periodic
specialization and local coloring select their own seed layouts. Hint inheritance belongs to
`Function`, so both AD modes use the same policy: every derived function inherits the callee's
effective lowering hint as described in [Lowering](lowering.md#the-optimization-pipeline).

Reverse mode inlines an ordinary call. It substitutes the actual arguments for the formals in the
callee's adjoint graph, with memoized subgraphs and a topologically cached substitution, so repeated
call structure does not re-walk the same graph once per occurrence.

For `VMAP`, reverse mode caches one adjoint function for the active formals and wraps it once over
the primal slices together with the per-iteration cotangent. Constant-index gathers then split its
concatenated adjoints back apart. A broadcast formal reduces across iterations, disjoint windows
take one scatter, and overlapping windows take a fixed number of scatters, one per distinct overlap
offset, whose sum accumulates the repeated destinations. Cotangents from formals bound to the same
outer expression are accumulated by the enclosing reverse pass.

As a result, dense and sparse Jacobians, gradients and Hessians all work on graphs containing
`VMAP` without the derivative code growing with the VMAP length.

When the `jvp_many` seeds over a `VMAP` are constant and the per-iteration seed tiles repeat with a
period of at most eight iterations, forward mode bakes each distinct tile into a const-seed callee
and maps it over that tile's residue class of iterations. This avoids runtime seed tables and seed
gathers. Other constant patterns keep the local-coloring and runtime-seed paths.

Different formals can need different local seed counts. Forward mode packs their specialized
results into one concatenated callee output when the iteration count and shared input slices agree.
One mapped call then shares the primal and adjoint expressions across those results. Gathers recover
each result's original seed layout, including zero seed rows. Generic runtime seeds instead enter
one joint derivative helper for all active formals.

A mapped tangent body grows with its seeds, and a body expanded into scalar code (a callee marked
`.scalar()`, whose derivatives inherit the hint) grows in machine code with them. Once that code no
longer fits the processor's instruction cache, every iteration fetches it again from the next level.
Forward mode estimates an expanded body's code from its expression graph, at about five bytes an
operation. When that passes the target's `body_bytes`, half its instruction cache, the seeds are
split into groups, each its own mapped call over every iteration, with enough groups that each body
comes to about that budget. A body kept in loops stays compact at any size and is never split.
Every group recomputes the primal and the adjoint, estimated as the callee's own body, and a formal
split into groups no longer shares them with the other formals' packed body. So the seeds stay in
one body when that part alone is past the budget, when a body of one seed would not fit either, or
when the recomputation would add more than half the body's work. A group's tangents are the same
expressions, simplified within the group, so a result can differ from the single body's in its last
bits.

The target is the one in force while the derivative is built, with two limits. Each formal's body
is judged on its own, so bodies that packing then joins can pass the budget together. And the
derivative helpers the process caches (of a scan, a while loop, a call with a custom JVP, and a
template's derived instances) keep the grouping of the target they were first built under.

One case where one-sided coloring would lose that guarantee is a shared stride-0 formal marked
differentiable. It gives the Hessian a dense row and column, so coloring the global pattern as a
Jacobian needs one color per iteration. `sparse_hessian` instead symmetrizes its structural pattern
and uses one global star-colored JVP batch, keeping the color count constant with the VMAP length.
The structured VMAP Jacobian path remains one-sided and uses `column_coloring` on each local tile.

## Sparse derivatives

Structural analysis and value construction are separate modules.

`ad/sparsity.py` answers "where can a nonzero be?" from graph shape alone. It is value-independent,
symbolic and free of any AD import, which is what lets it sit below the frontend. It propagates
compressed sparse row (CSR) Boolean arrays internally, covers the structural and arithmetic
operations exactly, handles `matmul` conservatively, and applies a sparse Boolean chain rule through
`call`. SciPy stays behind this module; public patterns remain `SparsityType` coordinate lists.
`column_coloring`, `star_coloring` and `color_groups` expose the graph-coloring vocabulary on top of
it. Star coloring is for square symmetric patterns. It is proper and forbids a two-coloured path of
three edges, which makes each Hessian entry recoverable from one compressed product.

`ad/sparse.py` is the half that needs AD. `sparse_jacobian(y, x)` returns a `SparseJacobian`: the
pattern, plus a compact `values` expression holding exactly the nonzero entries.

It tries a structured decomposition first. When `y` is a `VMAP` node, or an axis-0 `concat` of
`VMAP` nodes, whose outer tensors are exactly `x`, each piece is handled locally. Scaly computes the
sparsity tile on the callee, colors that small tile, materializes a constant local seed matrix,
pushes it through a multi-seed forward pass specialized on those constant seeds (so constant-seed
dead code elimination and common subexpression elimination apply), wraps the result back in a
`VMAP` over the original slicing, and assembles the global nonzero buffer.

When the per-formal contributions partition the nonzeros exactly, verified entry by entry and the
common case for the `z`/`znext` windows of a multistage constraint, the gathered values are emitted
back to back and the pattern is permuted to match, so no full-size scatter temporary is
materialized. Overlapping supports fall back to per-formal constant-index scatters that are summed.

This is why the compact ordering is piece-ordered instead of row-major, and why `(rows, cols)` is
the authority on coordinates. See [the generated interface](generated_interface.md#sparse-outputs) for what that means on the C
side.

Anything that is not a `VMAP` piece goes through `sparse_jacobian_colored`, which computes the
global pattern, greedily colors structurally independent columns, builds one compressed seed per
color, evaluates the forward passes, simplifies and CSEs each compressed column, and gathers each
nonzero from its `(row, color[col])` slot. `sparse_jacobian_reference` computes the dense Jacobian
and gathers from it. It is slow and obviously correct, and small tests are written differentially
against it.

`sparse_hessian` takes a different global path. It differentiates the gradient, symmetrizes its
structural pattern, star-colors that graph once, and uses a constant recovery table to gather every
entry from the compressed products. It bypasses the top-level `_sparse_jacobian_vmap` construction
shortcut, while global JVP rules retain one-sided coloring on each local VMAP tile.

The compressed products are a sum of blocks, each scattered to its place in a colors-by-columns
array, and the recovery gathers the nonzeros from that array. The simplifier composes the three
when the graph is built, so the array is never formed. A gather of a transpose reads what was
transposed, a gather of a sum of placed arrays is the sum of the terms' gathers, and a gather of
scattered values reads the values that reach each place, added in the order they were scattered.
Each block's values then go straight to the nonzeros they contribute to, and the sums are added in
the order the array would have added them. The gather reads through slices and concatenations the same
way. Under a gather, a product with a constant that is mostly zeros, as a seed matrix is, becomes a
scatter of the entries the constant keeps, so the zeros are never multiplied, and a zero times
anything is then a positive zero. A part that is computed, not placed, stays whole and is gathered
after, since it vectorizes that way, and so does a product that no gather reads.
