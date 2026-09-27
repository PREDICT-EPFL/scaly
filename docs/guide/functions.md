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
    output=sc.G(sc.L("sum", ...), sc.L("projection", 2)),
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

Call a `Function` with one argument per parameter, each shaped as that parameter's declared tree.
The leaves decide what the call means:

```python
symbolic = features((sc.sym("x0", 3), sc.sym("A0", (2, 3))))
numeric = features((np.ones(3), np.eye(2, 3)))

symbolic_sum, symbolic_projection = symbolic
numeric_sum, numeric_projection = numeric
```

`Expr` leaves create `CALL` nodes in a larger expression graph. Numerical leaves compile on first
use, cache the shared library and reconstruct the declared output tree.

`__call__` dispatches to two methods you can also call directly when the distinction matters:
`fn.symbolic_call(...)` always builds a call node and `fn.numerical_call(...)` always evaluates.

A tree that mixes `Expr` and numerical leaves is an error. Wrap the constants in `sc.const` to make
the symbolic reading explicit.

Ty checks structure statically and the runtime checks it again. Shapes are checked at runtime only,
because shapes are values in Python's type system.

## A single leaf is unpacked

Only `sc.G` introduces a tuple. A tree of one `sc.L` is that leaf, so a one-leaf input takes the
tensor itself and a one-leaf output returns the tensor itself:

```python
@sc.function(sc.L("x", 3), output=sc.L("scaled", ...))
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

## Sparse matrices

`sc.S(name, pattern)` declares a sparse matrix whose pattern is part of the signature. The body
receives a [`SparseMatrix`](sparsity.md#sparse-matrices-as-values). A symbolic call must pass a
`SparseMatrix` with exactly that pattern, and an evaluation a SciPy sparse matrix with exactly that
pattern, where explicitly stored zeros count as stored. Any other pattern is refused. As an output,
`sc.S(name, ...)` takes the pattern the body returns. A symbolic call then returns a `SparseMatrix`
and an evaluation a `scipy.sparse.csc_array`.

```python
@sc.function(sc.G(sc.L("q", n), sc.S("A", a_pattern)), output=sc.S("K", ...))
def kkt(inputs):
    q, A = inputs
    return sc.SparseMatrix.block([[sc.SparseMatrix.diag(q).add_diagonal(1e-6), None], [A, sc.SparseMatrix.identity(m) * -1e-3]])

@sc.function(sc.G(sc.S("K", kkt.output_sparsities[0]), sc.L("b", n + m)), output=sc.L("x", ...))
def solve(inputs):
    K, b = inputs
    return sc.linalg.SparseLDL(K).solve(b)

x = solve((kkt((q, A)), b))   # SciPy matrices in and out, patterns checked at each call
```

This is interface only. The C signature carries the `(nnz,)` values vector in CSC order, and the
generated code is the same as for a `Function` over that vector. An output's pattern also becomes
the output's sparsity metadata in the generated header.

## Compose functions

Calling a function with `Expr` leaves keeps the callee as a call in the graph:

```python
@sc.function(sc.L("x", 3), output=sc.L("square", ...))
def square(x: sc.Expr) -> sc.Expr:
    return x * x

@sc.function(sc.L("x", 3), output=sc.L("energy", ...))
def energy(x: sc.Expr) -> sc.Expr:
    return square(x).sum()
```

The generated C contains one `square` procedure and a call from `energy`. Differentiation keeps
that boundary; it does not copy the callee graph into every call site.

## Regular repetition: `vmap`

Use `sc.vmap` when every iteration applies the same function to a different slice. The mapping
stays one node through differentiation and lowers to a C loop.

```python
@sc.function(sc.L("x", 3), output=sc.L("sum", ...))
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
@sc.function(sc.G(sc.L("z", 2), sc.L("u", 1)), output=sc.G(sc.L("znext", ...), sc.L("cost", ...)))
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

A step that needs to know which step it is takes the step number: with `index=True` the body's
second input is an `int64` scalar counting from zero, and the sliced inputs follow it.

```python
@sc.function(sc.G(sc.L("z", 2), sc.L("k", sc.TensorType((), sc.dtypes.int64)), sc.L("u", 1)), output=sc.L("znext", ...))
def tv_step(inputs):
    z, k, u = inputs
    t = 0.1 * k.cast("float64")  # the time at the start of step k
    return sc.stack([z[0] + 0.1 * z[1], z[1] + 0.1 * (u[0] - t * z[0])])

(z_final,) = sc.scan(tv_step, z0, [(us, 0, 1)], length=50, index=True)
```

The loop passes its own counter, so no table of step numbers is stored, and the backward scan of
reverse mode passes the same numbers counting down. The index has no derivative. Compare it with
`sc.equal(k, 0)`, not `k == 0`: `==` between expressions is Python identity, not a comparison.

When a step changes only a few entries of a large carry, write the change with `sc.index_add` and
`sc.index_set` instead of rebuilding the carry. If the next carry is such a chain of updates rooted
at the carry, each update's values read only the carry as it stands just before that update and
none of the entries it writes (a read through a comparison, a `where` condition or `copysign`'s
sign counts as reading every entry), and no other output reads the carry, the body updates one carry slot
in place: each step touches only the indexed entries. Otherwise the two slots are kept, with the
same results. A loop differentiated in reverse mode stores every carry and does not update in place.

### Indices known only at run time

`sc.gather`, `sc.scatter` and `sc.index_add` take index tables fixed when the graph is built. When
the index is itself a value, such as an entry of a table chosen by the step number or the result of
a search, use `sc.take(x, idx)`, `sc.put_add(base, idx, values)` and `sc.put(base, idx, values)`
with `idx` an `int64` vector. They index the last axis, and any leading axes are kept. An index
outside `[0, n)` reads `fill` (`take`) or drops its value (`put_add`, `put`), so a ragged list of
indices can be padded to a fixed width with `n`:

```python
# one row of a CSR matrix per step, padded to `width` entries with the index n
body = sc.Function._from_exprs(
    "row", [c, cols, pos, x, vals],
    [c, sc.stack([(sc.take(vals, pos) * sc.take(x, cols)).sum()])],
    ["c", "cols", "pos", "x", "vals"], ["n", "y"],
)
tables = [(sc.const(cols_table, dtype="int64"), 0, width), (sc.const(pos_table, dtype="int64"), 0, width)]
_, y = sc.scan(body, c0, [*tables, (x, 0, 0), (vals, 0, 0)], length=n_rows)
```

A scan whose next carry is a chain of `put`/`put_add` (and `index_add`/`index_set`) updates can also
run in place. Run-time indices are proven at the loop: when every index an update writes, and every
index through which its values read the carry, is computed from constants, the step number and
integer tables the scan slices, the loop evaluates them for all steps at generation time and checks
that no value reads an entry an update of the same step has already written. The values may read
the carry only through `take`, `gather` or a slice. A forward substitution with a sparse triangular
matrix, which reads earlier entries of the solution and writes the current one, qualifies; a carry
index read from the data does not, and keeps two slots.

All three are differentiable in their floating-point operands. Because the pattern of a run-time
index is unknown, their sparsity is conservative: an output entry may depend on every entry of its
row. With repeated indices `put` keeps the last value, and only that value gets a derivative.

## Iteration until done: `while_loop`

Use `sc.while_loop` for an iteration that stops on a condition, such as a Newton solve. `cond` maps
the carry to one `bool`, `body` maps it to the next carry, and `max_iter` bounds the number of
steps:

```python
carry, n_iter = sc.while_loop(not_converged, newton_step, x0, max_iter=50)
```

`n_iter` is the number of steps taken, as a `float64`. With `index=True`, `body` also takes the
step number as a second `int64` input, as in `scan`; `cond` does not. The loop lowers to one C loop that calls
the condition, leaves when it is false and otherwise calls the body, so the code size does not
depend on `max_iter`.

Data every step reads and none changes, such as a solver's problem data or a matrix factor, goes in
`params` rather than in the carry. `body` takes them after the carry (and the step number), `cond`
after the carry:

```python
@sc.function(sc.G(sc.L("x", n), sc.L("P", (n, n)), sc.L("q", n)), output=sc.L("x_next", n), name="newton_step")
def newton_step(inputs):  # the carry, then the params
    x, P, q = inputs
    ...

# not_converged takes the same inputs and returns one bool
carry, n_iter = sc.while_loop(not_converged, newton_step, x0, max_iter=50, params=(P, q))
```

Params reach each step where they are, so they are never copied into the carry. Derivatives flow
through them too; reverse mode sums their cotangents over the steps the loop took.

A while loop is differentiable through the steps it took: the step count is treated as locally
constant, which it is except where the input crosses a point where the count changes. Forward mode
runs a loop that also carries the tangent. Reverse mode stores the carry at each of the at most
`max_iter` steps and runs `max_iter` backward steps, passing the cotangent unchanged through steps
the loop did not take. For a solver, the derivative through the steps only approximates the
derivative of the solution, and it improves as the stopping tolerance shrinks. A reverse pass that
would store more than `sc.options(max_trajectory=...)` values (50 million by default) for one loop,
`scan` or `while_loop`, raises when it is built ([Options](options.md)).

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
