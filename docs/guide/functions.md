# Building functions

A `Function` is a named expression graph with declared input and output trees. The same trees
describe symbolic calls with `Expr` leaves and numerical calls with NumPy-array leaves, and they
give ty enough information to reject a call with the wrong structure.

## Declare a function

Build trees from two constructors. `sc.L(name, shape)` declares one tensor. `sc.G(*trees)` groups
trees and may be nested; the type stubs cover up to eight children, the runtime accepts any count.

An integer shape means a rank-1 tensor, `()` is a scalar and a tuple is used as written. Pass a
`TensorType` when you need an explicit dtype or differentiability flag.

```python
import scaly as sc
import numpy as np

@sc.function(
    sc.G(sc.L("x", 3), sc.L("A", (2, 3))),
    sc.G(sc.L("sum", ...), sc.L("projection", 2)),
)
def features(inputs: tuple[sc.Expr, sc.Expr]) -> tuple[sc.Expr, sc.Expr]:
    x, A = inputs
    return x.sum(), A @ x
```

The decorator traces the body once with symbolic inputs. The body takes one value with the declared
input structure and returns one value with the declared output structure. Use `...` when an output
shape should be inferred; a written output shape is checked immediately, as are the output count
and structure.

Names are external metadata and need not match the body's local variable names. They identify
derivative inputs and outputs, generated C buffers, sparse tables and assembly text, so each name
must be unique within its tree.

## Symbolic and numerical calls

Call a `Function` with its declared input tree. The leaves decide what the call means:

```python
symbolic = features((sc.sym("x0", 3), sc.sym("A0", (2, 3))))
numeric = features((np.ones(3), np.eye(2, 3)))

symbolic_sum, symbolic_projection = symbolic
numeric_sum, numeric_projection = numeric
```

`Expr` leaves create `CALL` nodes in a larger expression graph. Numerical leaves compile on first
use, cache the shared library and reconstruct the declared output tree.

`__call__` dispatches to two methods you can also call directly when the distinction matters:
`fn.symbolic_call(tree)` always builds a call node and `fn.numerical_call(tree)` always evaluates.

A tree that mixes `Expr` and numerical leaves is an error. Wrap the constants in `sc.const` to make
the symbolic reading explicit.

Ty checks structure statically and the runtime checks it again. Shapes are checked at runtime only,
because shapes are values in Python's type system.

## A single leaf is unpacked

Only `sc.G` introduces a tuple. A tree of one `sc.L` is that leaf, so a one-leaf input takes the
tensor itself and a one-leaf output returns the tensor itself:

```python
@sc.function(sc.L("x", 3), sc.L("scaled", ...))
def scale(x: sc.Expr) -> sc.Expr:
    return 2.0 * x

scale(np.ones(3))                      # L in, L out -> one array in, one array out
features((np.ones(3), np.eye(2, 3)))   # G in, G out -> a 2-tuple in, a 2-tuple out
```

This matters most on the way out. Do not destructure a single-output result:

```python
scaled = scale(np.ones(3))     # correct
(scaled,) = scale(np.ones(3))  # wrong
```

The second line is not always loud. It iterates the returned array along its first axis, as NumPy
or PyTorch would, so it raises for a length-3 output but succeeds whenever the leading axis has
length one, binding a scalar slice instead of the whole tensor. Symbolic calls behave the same way,
with an `Expr` in place of the array.

## Grouping and the C signature

Grouping exists for Python readability and typing. The generated signature uses the leaves in tree
order, so these declarations have the same flat input signature:

```python
nested = sc.G(
    sc.G(sc.L("state", 4), sc.L("control", 2)),
    sc.G(sc.L("weights", 10), sc.L("dt", ())),
)
flat = sc.G(
    sc.L("state", 4),
    sc.L("control", 2),
    sc.L("weights", 10),
    sc.L("dt", ()),
)
```

Choose the grouping that matches the object the caller passes.

## Compose functions

Calling a function with `Expr` leaves keeps the callee as a call in the graph:

```python
@sc.function(sc.L("x", 3), sc.L("square", ...))
def square(x: sc.Expr) -> sc.Expr:
    return x * x

@sc.function(sc.L("x", 3), sc.L("energy", ...))
def energy(x: sc.Expr) -> sc.Expr:
    return square(x).sum()
```

The generated C contains one `square` procedure and a call from `energy`. Differentiation keeps
that boundary; it does not copy the callee graph into every call site.

## Regular repetition: `vmap`

Use `sc.vmap` when every iteration applies the same function to a different slice. The mapping
stays one node through differentiation and lowers to a C loop.

```python
@sc.function(sc.L("x", 3), sc.L("sum", ...))
def reduce3(x: sc.Expr) -> sc.Expr:
    return x.sum().reshape((1,))

xs = sc.sym("xs", 15)
mapped = sc.vmap(reduce3, 5, [(xs, 0, 3)])
```

Each input specification is `(outer, start, stride)`. Iteration `i` reads a slice beginning at
`start + i * stride`. A zero stride broadcasts one slice across every iteration. See
[Sparsity](sparsity.md) for how mapped structure reduces derivative construction and generated
source size.

## Sequential repetition: `scan`

Use `sc.scan` when each step needs the result of the previous one: a rollout, a filter, a
recursion. The body's first input is the carry and its first output is the next carry, with the
same shape; its other inputs are sliced from outer tensors exactly as `vmap` slices them, and its
other outputs are stacked one slice per step.

```python
@sc.function(sc.G(sc.L("z", 2), sc.L("u", 1)), sc.G(sc.L("znext", ...), sc.L("cost", ...)))
def step(inputs):
    z, u = inputs
    znext = sc.stack([z[0] + 0.1 * z[1], z[1] + 0.1 * u[0]])
    return znext, sc.stack([sc.sumsqr(z)])

z0, us = sc.sym("z0", 2), sc.sym("us", 50)
z_final, costs = sc.scan(step, z0, [(us, 0, 1)], length=50)
```

The step count is fixed when the graph is built. The scan lowers to one loop in C whose size does
not depend on the count, and the carry alternates between two slots, so no step copies it. A
derivative of a scan is a scan: forward mode carries the tangent alongside the carry, and reverse
mode runs the steps backwards over the carries the forward pass stored, which costs
`(length + 1) * carry.size` values of workspace.

When a step changes only a few entries of a large carry, write the change with `sc.index_add` and
`sc.index_set` instead of rebuilding the carry. If the next carry is such a chain of updates rooted
at the carry, each update's values read only the carry as it stands just before that update and
none of the entries it writes, and no other output reads the carry, the body updates one carry slot
in place: each step touches only the indexed entries. Otherwise the two slots are kept, with the
same results. A loop differentiated in reverse mode stores every carry and does not update in place.

## Iteration until done: `while_loop`

Use `sc.while_loop` for an iteration that stops on a condition, such as a Newton solve. `cond` maps
the carry to one `bool`, `body` maps it to the next carry, and `max_iter` bounds the number of
steps:

```python
carry, n_iter = sc.while_loop(not_converged, newton_step, x0, max_iter=50)
```

`n_iter` is the number of steps taken, as a `float64`. The loop lowers to one C loop that calls
the condition, leaves when it is false and otherwise calls the body, so the code size does not
depend on `max_iter`.

A while loop is differentiable through the steps it took: the step count is treated as locally
constant, which it is except where the input crosses a point where the count changes. Forward mode
runs a loop that also carries the tangent. Reverse mode stores the carry at each of the at most
`max_iter` steps and runs `max_iter` backward steps, passing the cotangent unchanged through steps
the loop did not take. For a solver, the derivative through the steps only approximates the
derivative of the solution, and it improves as the stopping tolerance shrinks.

## Inspect and verify

```python
sc.verify_expr(symbolic_projection)
print(sc.format_expr(symbolic_projection))
print(sc.render_expr_assembly(features))
```

Construction checks declarations and shapes. `sc.verify_expr` checks the complete expression graph
and raises `VerifyError` at the first invalid node.

## Device placement

`fn.with_device("cuda:0")` records device placement and validates dtypes against the backend
capability table. Only host lowering is implemented today.
