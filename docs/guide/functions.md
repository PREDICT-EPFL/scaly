# Building functions

A `Function` is a Python body traced into a named expression graph. Calling it with arrays compiles
and runs the graph; calling it with expressions puts a call to it in a larger graph. This page goes
from the shortest declaration to the most specific one, then covers composition and loops.

## A first function

Decorate a body with `sc.function`. Nothing needs declaring:

```python
import numpy as np
import scaly as sc

@sc.function
def step(x, u):
    return x + 0.1 * u

step(np.zeros(2), np.ones(2))          # compiles on first use, then runs: array([0.1, 0.1])
step(sc.sym("x", 2), sc.sym("u", 2))   # an Expr: a call to step inside a larger graph
```

The body takes one argument per parameter and returns an `Expr`, or a tuple of them for several
outputs. Inside it the arguments are `Expr`s, so build the result from the operations on the
[Expressions](../api/core.md) page, not from NumPy functions.

## Symbolic and numerical calls

The leaves of the arguments decide what a call means. All `Expr` leaves build a call node. All
numerical leaves, which may be arrays, Python numbers or lists, evaluate. The first call at a given
signature generates C, compiles it and caches the shared library, and later calls go straight to
it. A call with no arguments evaluates.

A call that mixes the two is an error. Wrap the constants in `sc.const` to make a symbolic call
explicit. `f.symbolic_call(...)` and `f.numerical_call(...)` skip the dispatch when the distinction
is the point; inside a traced body, a function without parameters is called as `f.symbolic_call()`.

## One body, many shapes

Nothing in `step` says how long `x` is, so each call binds that from its arguments. The body is
traced once per distinct binding, and the result, a `ConcreteFunction`, is kept under a name that
spells the shapes:

```python
step(np.zeros(3), np.ones(3))
list(step.instances)   # ['step__2_2', 'step__3_3']
```

The name is the instance's C symbol and the key of its compiled library, so it is the same in every
process. The tokens after `__` are the bound dimensions joined by `x` (`4x2`), `s` for a scalar,
and the dtype when it is not `float64` (`3int64`). `step.instances` is everything that reached C.

A Python float, an int, a NumPy scalar and a 0-d array are the same argument, a scalar coerced to
`float64`. A tuple is structure, not a vector, so `step((1.0, 2.0), u)` raises and suggests a list
or an array. An `Expr` argument brings its dtype, so a helper called with an `int64` index gets an
`int64` input.

## Declaring shapes

Declare a parameter's shape by passing it to the decorator, one declaration per parameter in order:

```python
NX, NU = 4, 2

@sc.function(NX, NU)
def dynamics(x, u):
    return x + 0.1 * sc.concat([x[2:], u])
```

A declaration is a *tree spec*: a shape (`4`, `(3, 3)`, `()` for a scalar), a name, an `sc.L`,
`sc.G` or `sc.linalg.S` tree, or a `TensorType`. With every shape declared, the body is traced at the
decorator, errors in it surface there, and the result is the one instance itself, named after the
function: `dynamics.is_concrete` holds and `dynamics.concrete is dynamics`.

Declare shapes when a stable C symbol matters (code shipped to another build), when the function is
a `vmap` or `scan` callee (below), or when an early error helps. `sc.function` with no declarations
and `sc.function` with every shape declared are the two ends; anything in between is allowed:

```python
@sc.function((NX, None), NX)
def project(A, x):
    return A.T @ x

project(np.ones((NX, 7)), np.ones(NX))
list(project.instances)   # ['project__7']
```

`None` leaves one dimension open, and `sc.L()` or `...` leaves the whole shape open. The call
refuses a fixed dimension that does not match. A function whose inputs still have holes is a
template: `project.is_concrete` is false, and `project.concrete` raises `NotConcrete` with the fix.

## Names

Each input leaf is named after its parameter, which is how a derivative refers to it and what the
generated C header calls it. An unnamed output is named after the function. Give names explicitly
with `sc.L`:

```python
@sc.function(sc.L("state", NX), NU, output="next_state")
def model(x, u):
    return x + 0.1 * sc.concat([x[2:], u])

model.input_names, model.output_names   # (('state', 'u'), ('next_state',))
```

`output=` takes a tree spec like the inputs, so `output="y"` names one leaf and leaves its shape to
the trace, and `output=sc.G("lo", "hi")` names two. Without `output=`, a tuple result's leaves are
named `model_0`, `model_1`, and so on. Names are unique within a function. A name that is a C or C++
keyword gets a trailing underscore in the generated code.

## Structured arguments and results

A tuple argument is structure. Its leaves become separate inputs, named `p_0`, `p_1`, ... after the
parameter `p`. Declare a tuple parameter with `sc.G`, which groups two to eight trees and nests:

```python
@sc.function(sc.G(NX, NU), ())
def rollout_step(state, dt):
    x, u = state
    return x + dt * sc.concat([x[2:], u]), (x * x).sum()

next_x, energy = rollout_step((np.zeros(NX), np.ones(NU)), 0.1)
```

The generated C signature is the leaves in order, so grouping is for readability. `rollout_step`
and a four-parameter body over the same leaves compile to the same entry.

Only a tuple introduces structure. A single-leaf result is the array or `Expr` itself, not a
one-element tuple, so do not destructure it:

```python
out = model(np.zeros(NX), np.ones(NU))     # correct
(out,) = model(np.zeros(NX), np.ones(NU))  # wrong
```

The second line is not always loud. It iterates the returned array along its first axis, so it
succeeds whenever that axis has length one, binding a slice instead of the whole result.

## Dtypes and differentiability

A leaf is `float64` unless declared otherwise. Pass `dtype=` to `sc.L` for an integer or boolean
input, and `diff=False` for an input no derivative should flow through:

```python
@sc.function(NX, sc.L((), dtype="int64"))
def pick(x, k):
    return sc.take(x, sc.stack([k]))
```

A call converts numerical arguments to the declared dtype and refuses a symbolic argument of another
dtype.

## Sparse matrices

`sc.linalg.S(pattern)` declares a sparse matrix whose pattern is part of the signature. The body receives
a [`SparseMatrix`](sparsity.md#sparse-matrices-as-values). A symbolic call must pass a
`SparseMatrix` with exactly that pattern, and an evaluation a SciPy sparse matrix with exactly that
pattern, where explicitly stored zeros count as stored. A result that is a `SparseMatrix` comes back
as one from a symbolic call and as a `scipy.sparse.csc_array` from an evaluation.

```python
n, m = 3, 2
a_pattern = np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)

@sc.function(n, sc.linalg.S(a_pattern))
def kkt(q, A):
    return sc.linalg.SparseMatrix.block([[sc.linalg.SparseMatrix.diag(q).add_diagonal(1e-6), None], [A, sc.linalg.SparseMatrix.identity(m) * -1e-3]])

@sc.function(sc.linalg.S(kkt.output_sparsities[0]), n + m)
def solve(K, b):
    return sc.linalg.SparseLDL(K).solve(b)
```

The C signature carries the `(nnz,)` values in compressed-column order; the pattern is fixed when
the graph is built. A bare template reads a `SparseMatrix` argument's pattern into the instance, so
two patterns are two instances. It does not yet read a pattern off a SciPy argument, so declare
that parameter with `sc.linalg.S`.

## Differentiating

The derivative wrappers take a `Function` and name the output and the input:
`sc.gradient(f, "cost", "x")`. One name is the input, and a name left out is the function's only
output or input, so `sc.gradient(energy)` is the gradient of a one-input, one-output function. The
result is a `Function` over the same parameters. The derivative of a template is a template too,
whose instances are the derivatives of the source's instances at the same shapes
(`cost__3_grad_cost_x`). See [Derivatives](derivatives.md).

## Compose functions

Calling a function with `Expr` arguments keeps the callee as a call in the graph:

```python
@sc.function
def square(x):
    return x * x

@sc.function(3)
def energy(x):
    return square(x).sum()
```

Tracing `energy` instantiates `square__3`. The generated C contains one `square__3` procedure and a
call to it from `energy`. Differentiation keeps that boundary; it does not copy the callee graph
into every call site.

## Regular repetition: `vmap`

Use `sc.vmap` when every iteration applies the same function to a different slice. The mapping
stays one node through differentiation and lowers to a C loop.

```python
@sc.function(3)
def reduce3(x):
    return x.sum().reshape((1,))

xs = sc.sym("xs", 15)
mapped = sc.vmap(reduce3, 5, [(xs, 0, 3)])
```

Each input specification is `(outer, start, stride)`. Iteration `i` reads a slice beginning at
`start + i * stride`. A zero stride broadcasts one slice across every iteration. The callee's
shapes are what `vmap` slices by, so a template callee must be instantiated first:
`sc.vmap(f.instantiate(3), 5, ...)`. See [Sparsity](sparsity.md) for how mapped structure reduces
derivative construction and generated source size.

## Sequential repetition: `scan`

Use `sc.scan` when each step needs the result of the previous one: a rollout, a filter, a
recursion. The body's first parameter is the carry and its first output is the next carry, with the
same shape; its other parameters are sliced from outer tensors exactly as `vmap` slices them, and
its other outputs are stacked one slice per step.

```python
@sc.function(2, 1)
def step2(z, u):
    znext = sc.stack([z[0] + 0.1 * z[1], z[1] + 0.1 * u[0]])
    return znext, sc.stack([sc.sumsqr(z)])

z0, us = sc.sym("z0", 2), sc.sym("us", 50)
z_final, costs = sc.scan(step2, z0, [(us, 0, 1)], length=50)
```

The step count is fixed when the graph is built. The scan lowers to one loop in C whose size does
not depend on the count, and the carry alternates between two slots, so no step copies it. A
derivative of a scan is a scan: forward mode carries the tangent alongside the carry, and reverse
mode runs the steps backwards over the carries the forward pass stored, which costs
`(length + 1) * carry.size` values of workspace.

A step that needs to know which step it is takes the step number: with `index=True` the body's
second parameter is an `int64` scalar counting from zero, and the sliced inputs follow it.

```python
@sc.function(2, sc.L((), dtype="int64"), 1)
def tv_step(z, k, u):
    t = 0.1 * k.cast("float64")  # the time at the start of step k
    return sc.stack([z[0] + 0.1 * z[1], z[1] + 0.1 * (u[0] - t * z[0])])

(z_final,) = sc.scan(tv_step, z0, [(us, 0, 1)], length=50, index=True)
```

The loop passes its own counter, so no table of step numbers is stored, and the backward scan of
reverse mode passes the same numbers counting down. The index has no derivative. Compare it with
`sc.equal(k, 0)`, not `k == 0`: `==` between expressions is Python identity, not a comparison.

When a step changes only a few entries of a large carry, write the change with `sc.index_add` and
`sc.index_set` instead of rebuilding the carry. They are `sc.put_add` and `sc.put` at indices fixed
when the graph is built, on the flat view of a carry with more than one axis. If the next carry is
such a chain of updates rooted at the carry, each update's values read only the carry as it stands
just before that update and none of the entries it writes (a read through a comparison, a `where`
condition, `copysign`'s sign, `floor`, `ceil` or a cast counts as reading every entry), and no other
output reads the carry,
the body updates one carry slot in place: each step touches only the indexed entries. Otherwise the
two slots are kept, with the same results. A loop differentiated in reverse mode stores every carry
and does not update in place.

### Indices known only at run time

`sc.gather`, `sc.scatter` and `sc.index_add` take index tables fixed when the graph is built. When
the index is itself a value, such as an entry of a table chosen by the step number or the result of
a search, use `sc.take(x, idx)`, `sc.put_add(base, idx, values)` and `sc.put(base, idx, values)`
with `idx` an `int64` vector. They index the last axis, and any leading axes are kept. An index
outside `[0, n)` reads `fill` (`take`) or drops its value (`put_add`, `put`), so a ragged list of
indices can be padded to a fixed width with `n`:

```python
# one row of a CSR matrix per step, padded to `width` entries with the index n
@sc.function(1, sc.L(width, dtype="int64"), sc.L(width, dtype="int64"), n, nnz)
def row(c, cols, pos, x, vals):
    return c, sc.stack([(sc.take(vals, pos) * sc.take(x, cols)).sum()])

tables = [(sc.const(cols_table, dtype="int64"), 0, width), (sc.const(pos_table, dtype="int64"), 0, width)]
_, y = sc.scan(row, c0, [*tables, (x, 0, 0), (vals, 0, 0)], length=n_rows)
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
steps. Data every step reads and none changes, such as a solver's problem data or a matrix factor,
goes in `params` rather than in the carry; `body` takes them after the carry (and the step number),
`cond` after the carry:

```python
@sc.function
def newton_step(x, P, q):
    return x - sc.linalg.solve(P, P @ x - q)

@sc.function
def not_converged(x, P, q):
    return sc.norm_2(P @ x - q) > 1e-10

carry, n_iter = sc.while_loop(not_converged, newton_step, sc.sym("x0", 3), max_iter=50, params=(sc.sym("P", (3, 3)), sc.sym("q", 3)))
```

Every shape is determined here, the carry by `init` and the rest by `params`, so `while_loop`
instantiates template callees itself. `n_iter` is the number of steps taken, as a `float64`. With
`index=True`, `body` also takes the step number as its second parameter, as in `scan`; `cond` does
not. The loop lowers to one C loop that calls the condition, leaves when it is false and otherwise
calls the body, so the code size does not depend on `max_iter`. Params reach each step where they
are, so they are never copied into the carry. Derivatives flow through them too; reverse mode sums
their cotangents over the steps the loop took.

A while loop is differentiable through the steps it took: the step count is treated as locally
constant, which it is except where the input crosses a point where the count changes. Forward mode
runs a loop that also carries the tangent. Reverse mode stores the carry at each of the at most
`max_iter` steps and runs `max_iter` backward steps, passing the cotangent unchanged through steps
the loop did not take. For a solver, the derivative through the steps only approximates the
derivative of the solution, and it improves as the stopping tolerance shrinks. A reverse pass that
would store more than `sc.options(max_trajectory=...)` values (50 million by default) for one loop,
`scan` or `while_loop`, raises when it is built ([Options](options.md)).

## Ahead of time

`f.instantiate(...)` builds an instance without calling it, from one declaration per parameter:
a shape, a `TensorType`, a tree, or an example array or `Expr`. A tuple of ints is a shape; any
other tuple is a group's parts. The result is cached like a call's:

```python
step_4 = step.instantiate(4, 4)       # step__4_4
sc.codegen.write_module(step_4, out_dir)
```

`python -m scaly.codegen module:attribute` renders a `Function` attribute with every shape
declared, such as `step_4` above, or a zero-argument factory returning one. It refuses a template
with holes and says what to export instead. See [Code generation](codegen.md).

## Declarations at a glance

| Spelling | Declares |
| --- | --- |
| `@sc.function` | every parameter bound at the call; outputs inferred |
| `3`, `(n, m)`, `()` | one leaf of that shape (`()` is a scalar), named after the parameter |
| `(n, None)` | rank 2, first dimension fixed, second bound at the call |
| `sc.L()`, `...` | one leaf of any shape |
| `"y"` | a leaf named `y`, its shape bound at the call or by the trace |
| `sc.L("y", 3)`, `sc.L("k", (), dtype="int64", diff=False)` | a named leaf with shape, dtype and differentiability |
| `sc.G(a, b, ...)` | a tuple of two to eight trees; parts named `p_0`, `p_1`, ... unless named |
| `sc.linalg.S(pattern)`, `sc.linalg.S("A", pattern)` | a sparse matrix with that pattern |
| `output=...` | the same specs, for the result; leave it out to read it off the trace |
| `name="f"` | the function's name, and so its C symbol |

## Inspect and verify

```python
print(sc.render_expr_assembly(dynamics))
sc.verify_expr(dynamics.outputs[0])
```

Construction checks declarations and shapes. `sc.verify_expr` checks the complete expression graph
and raises `VerifyError` at the first invalid node. A template's graph belongs to its instances, so
render `f.instances["f__3"]`, or instantiate first.

## Device placement

`fn.with_device("cuda:0")` records device placement and validates dtypes against the backend
capability table; on a template it applies to every instance. Only host lowering is implemented
today.
