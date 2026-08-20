# How differentiation works

Alloy differentiates graphs, not values. Every derivative is a new expression-dialect graph built
from the old one, so a derivative is an ordinary `Function` that gets lowered, optimized and
compiled exactly like the thing it came from. There is no tape and no runtime.

The user-facing side is in [Derivatives](../guide/derivatives.md). This page is about the parts
that matter when you are changing them.

## Two modes, kept separate

Forward and reverse mode are independent implementations in `ad/forward.py` and `ad/reverse.py`.
They share no dispatch and no rule table; the only thing they had in common — the structural
"does this depend on that" query — sits below both in `ad/sparsity.py`, which contains no AD.

`ad/derivatives.py` assembles the whole-derivative constructions from them:

| Construction | How |
| --- | --- |
| `gradient(y, x)` | one reverse sweep with a seed of ones. Requires a scalar `y`. |
| `jacobian(y, x)` | forward mode, batched: the `x.size` identity columns are stacked into one seed of shape `(x.size, *x.shape)` and pushed through `jvp_many` in a single pass, then transposed. |
| `hessian(y, x)` | `jacobian(gradient(y, x), x)`: one reverse sweep, then batched forward over its graph. |

Batching the Jacobian is what keeps it from being a loop of independent sweeps. The multi-seed
rules share the expensive parts across columns — one `cos` serves every column of a `sin`'s
derivative — and turn per-column chain-rule unrolling into small matrix products.

Not every operation has a multi-seed rule yet: `abs`, `asin`, `acos`, `atan`, `atan2`, `minimum`,
`maximum`, `floor`, `ceil`, and any `transpose` whose result is rank 4 or higher (the seed axis
would push it past the rank-4 lowering limit) fall back to evaluating seeds one at a time. The fallback is silent by default because it is correct, just slower. Set
`ALLOY_STRICT_JVP_MANY=1` to make it raise instead, which is what you want when you are adding a
rule and need to know whether it is being used.

## Differentiating across a call

The whole point of a `call` node is that the callee's structure survives into generated code. AD
has to preserve that, or a horizon of a hundred identical stages becomes a hundred copies of the
stage derivative.

**Forward mode builds a derivative function and calls it.** For each distinct
(callee, output, formal input) triple it constructs a derivative `Function` once, caches it, and
emits a new `call` node to that function. The constructed function takes only the arguments its
derivative actually depends on — determined structurally — plus the seed if the seed survives, so
the derivative of a stage function has a narrower signature than the stage itself.

**Reverse mode inlines for an ordinary call**, substituting the actual arguments for the formals in
the callee's adjoint graph, with memoized subgraphs and a topologically cached substitution so that
repeated call structure does not re-walk the same graph once per occurrence.

**For `map`, reverse mode caches one adjoint function** for the active formals and maps it once
over the primal slices together with the per-iteration cotangent. Constant-index gathers then split
its concatenated adjoints back apart: a broadcast formal reduces across iterations, disjoint
windows take one scatter, and overlapping windows take a fixed number of scatters — one per
distinct overlap offset — whose sum accumulates the repeated destinations. Cotangents from formals bound to the same outer
expression are accumulated by the enclosing reverse pass.

The result is that `jac`, `grad`, `hess` and `sphess` all work on graphs containing `map` without
the derivative code growing with the map length.

One case where that guarantee does not hold: a shared stride-0 formal marked differentiable gives
the Hessian a dense row and column. One-sided column coloring then degenerates to one color per
iteration and the generated `sphess` source grows with the map length. The values stay exact.
Symmetric or star coloring would restore the constant behaviour and is not implemented.

## Sparse derivatives

Structural analysis and value construction are separated on purpose.

`ad/sparsity.py` answers "where can a nonzero be?" from graph shape alone — value-independent,
symbolic, and free of any AD import, which is what lets it sit below the frontend. It propagates
compressed sparse row Boolean arrays internally, covers the structural and arithmetic operations
exactly, handles `matmul` conservatively, and applies a sparse Boolean chain rule through `call`.
SciPy stays behind this module; public patterns remain `SparsityType` coordinate lists.
`column_coloring` and `color_groups` expose the graph-coloring vocabulary on top of it.

`ad/sparse.py` is the half that needs AD. `sparse_jacobian(y, x)` returns a `SparseJacobian`: the
pattern, plus a compact `values` expression holding exactly the nonzero entries.

It tries a structured decomposition first. When `y` is a `map` node — or an axis-0 `concat` of
`map` nodes — whose outer tensors are exactly `x`, each piece is handled locally: compute the
sparsity tile on the *callee*, color that small tile, materialize a constant local seed matrix,
push it through a multi-seed forward pass specialized on those constant seeds (so constant-seed
dead code elimination and CSE apply), wrap the result back in a `map` over the original slicing,
and assemble the global nonzero buffer.

When the per-formal contributions partition the nonzeros exactly — verified entry by entry, and the
common case for the `z`/`znext` windows of a multistage constraint — the gathered values are
emitted back to back and the pattern is permuted to match, so no full-size scatter temporary is
ever materialized. Overlapping supports fall back to per-formal constant-index scatters that are
summed.

This is why the compact ordering is piece-ordered rather than row-major, and why `(rows, cols)` is
the authority on coordinates rather than an afterthought. See
[the ABI](c_abi.md#sparse-outputs) for what that means on the C side.

Anything that is not a `map` piece goes through `sparse_jacobian_colored`: compute the global
pattern, greedily color structurally independent columns, build one compressed seed per color,
evaluate the forward passes, simplify and CSE each compressed column, and gather each nonzero from
its `(row, color[col])` slot. `sparse_jacobian_reference` computes the dense Jacobian and gathers
from it — slow, obviously correct, and the differential reference small tests are written against.
