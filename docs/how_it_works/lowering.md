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
multiple procedures; and `VMAP`. A Function carrying a solver descriptor is deliberately not lowered — it stays
opaque and its wrapper is rendered separately, while the oracle functions it drives lower normally.

Not covered: device placement other than the host, and the operations listed as deferred in
[the expression dialect](expr_ir.md#operations). Both raise `LoweringError`.

## Mixed scalar and block lowering

Every expression carries a lowering hint — `auto`, `scalar`, `block` or `opaque`. The intent is the
one alloy was designed around: SX-like scalar regions and MX-like block regions coexisting in one
graph, with materialization inserted at the boundary.

Today `auto` is what actually runs. The hints are recorded, printed and inspectable, and `opaque`
already has meaning at a solver boundary, but the lowerer does not yet partition a graph on them.
Making them drive region formation is the open work; see the roadmap.

## The optimization pipeline

`optimize_program` runs a registered, ordered list of program-to-program rewrites at the tail of
lowering, between the naive lowering and the verifier. Adding an optimization means adding one
function:

```python
@register_pass("my_pass")
def my_pass(prog: ProgramNode) -> ProgramNode:
    ...
```

Three passes run today, in this order.

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
