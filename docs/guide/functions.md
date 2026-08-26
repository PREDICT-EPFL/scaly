# Building functions

A `Function` is a named graph: named inputs, named outputs, and the expression graph between them.
It is the unit of composition, of differentiation and of compilation, and almost everything in
alloy is either building one or transforming one.

## Two ways to build

The decorator is the usual one. Declare the input shapes, write the body, return a dictionary:

```python
import alloy as al

@al.function("f", {"x": 3, "A": (2, 3)})
def f(x, A):
    return {"y": (A @ x).sum()}
```

Your parameters arrive as `Expr` inputs of the declared shapes. An `int` means a rank-1 shape, a
tuple is used as given, and a `TensorType` lets you set a dtype or attach sparsity.

Building explicitly does the same thing with the pieces visible:

```python
x = al.sym("x", 3)
y = (x.sin() + x * x).sum()
f = al.Function("f", [x], [y], ["x"], ["y"])
```

`al.sym` creates a named input. `al.const` wraps a NumPy array as a constant. Both return `Expr`
nodes, and every operation on an `Expr` builds another node rather than computing anything.

Returning a tuple or a single expression instead of a dictionary works too; outputs are then named
`out0`, `out1`, and so on. Naming them is usually worth it — the names become the generated C
symbols and the handles you use in derivative requests.

## Calling one

```python
f(x_value)                    # positional, in declared input order
f(x=x_value, A=a_value)       # or by name — not both at once
```

What comes back depends on how many outputs there are: **one output returns the array itself,
several return a tuple** in declared order. `f.output_names` is the order. If you would rather have
the list either way, `f.eval_list(...)` always returns one.

A `SolverFunction` from `al.qp` or `al.nlp` is the exception — it returns a dict keyed by output
name, because a solve has half a dozen outputs and positional indexing into them is a bug waiting
to happen. See [Solvers](solvers.md).

## Shapes

Shapes are static tuples of non-negative integers, and `()` is a scalar. There are no symbolic
dimensions: a function for a horizon of 10 and a function for a horizon of 40 are two functions.

Arithmetic broadcasts the NumPy way. Mixed dtypes do not promote — combining `float32` and
`float64` raises rather than silently widening.

Everyday operations, all of which return new `Expr` nodes:

```python
x[2]            x[1:4]         x[:, 0]        # indexing and slicing
x.reshape((2, 3))              x.transpose((1, 0))
A @ x                          x.sum()        # a full reduction to a scalar
x.sin()   x.exp()   x.sqrt()   abs(x)
al.stack([a, b])               al.concat([a, b])
al.dot(a, b)     al.sumsqr(x)     al.norm_2(x)
al.gather(x, idx)              al.scatter(x, idx, shape)
al.split(x, 3)                 al.vec(x)      # flatten to rank 1
```

The full operation set is in [the expression dialect](../how_it_works/expr_ir.md#operations).

## Calling one function from another

`Function.call` puts a first-class call node in the graph:

```python
@al.function("stage", {"z": 4, "u": 2})
def stage(z, u):
    return {"znext": z + 0.1 * al.concat([z[2:], u])}

@al.function("horizon", {"z0": 4, "us": (10, 2)})
def horizon(z0, us):
    z = z0
    for k in range(10):
        z = stage.call([z, us[k]])[0]
    return {"zN": z}
```

`call` returns a tuple with one `Expr` per callee output, so take the one you want. Arguments are
checked against the callee's formals by shape, and constants are converted for you.

This matters more than it looks. The call is *not* inlined: `stage` appears once in the generated C
and `horizon` calls it ten times, so source size stays flat as the horizon grows. Derivatives
preserve the same structure — forward mode builds one derivative of `stage` and calls it ten times
rather than emitting ten copies.

## Regular repetition: `vmap`

When every iteration is the same callee applied to a different slice, say so. `al.vmap` keeps the
repetition as a single node, which survives lowering into a real loop and differentiation into a
mapped derivative:

```python
@al.function("body", {"a": 3})
def body(a):
    return {"o": a.sum().reshape((1,))}

xs = al.sym("xs", (15,))
mapped = al.vmap(body, 5, [(xs, 0, 3)])     # 5 iterations, reading xs[0:3], xs[3:6], ...
mapped.shape                                 # (5,)
```

The arguments are the callee, the number of iterations, and one `(outer, start, stride)` per callee
input. Iteration `i` reads `outer[start + i*stride : start + i*stride + formal.size]`. Outer tensors
must be rank 1; the callee's own formals and outputs may be rank 2. Outputs are concatenated flat,
so the node's shape is `(length * output.size,)`.

A `stride` of `0` broadcasts — every iteration reads the same slice, which is how a shared parameter
rides along. You can also pass a mapping from callee input name to spec, which is easier to read
when there are several.

`vmap` is what makes a multistage problem scale. A hundred-stage constraint is one node, its sparse
Jacobian is computed by coloring the *callee's* small pattern once, and the generated code stays
roughly constant in size. See [Sparsity](sparsity.md) and
[the numbers](../results/scalability.md).

There is no loop-carried state: every iteration is independent. A stride of zero broadcasts the
same slice to every iteration.

## Lowering hints

Every expression carries a preference for how it should eventually be computed:

```python
sparse_part = (x.sin() + x * x).scalar()   # prefer unrolled scalar code
dense_part = (A @ x).block()               # prefer loops
boundary = dense_part.opaque()             # keep this a named boundary
```

Today these are recorded, printed and inspectable, and the lowerer runs everything through its
default policy — with one exception: `opaque` genuinely stops the lowerer looking inside, which is
how a solver stays a call. Region formation on `scalar` and `block` is open work, so treat them as
documentation of intent for now.

## Checking a graph

Construction checks the cheap things. For everything else there is an explicit verifier:

```python
al.verify_expr(y)     # silent on success, VerifyError at the first bad node
```

Run it after a non-trivial rewrite, and write negative tests against it.

To look at what you built:

```python
y.debug()                            # topological dump with %0, %1, ... names
al.format_expr(y)                    # the same, as a string
al.render_expr_assembly(f)           # stable assembly text, good for diffs and tests
```

## Placement

A function can be placed on a device:

```python
fn_gpu = fn.with_device("cuda:0")
```

Placement is checked against the backend's capabilities when the function is constructed, so
putting a `float64` graph on a backend without `float64` fails immediately, naming the input. Today
only `host` actually lowers; other placements are tracked and printed, and raise when called.
