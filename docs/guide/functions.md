# Building functions

A `Function` is a named expression graph with declared input and output trees. The same trees
describe symbolic calls with `Expr` leaves and numerical calls with NumPy-array leaves. They also
give ty enough information to reject a call with the wrong structure.

## Declare a function

Build trees from two constructors:

- `sc.L(name, shape)` declares one tensor.

- `sc.G(*trees)` groups two to eight trees and may be nested.

An integer shape means a rank-1 tensor, `()` is a scalar, and a tuple is used as written. Pass a
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
input structure and returns one value with the declared output structure.
Use `...` when an output shape should be inferred. A written output shape is checked immediately.
The decorator also rejects the wrong output count or structure.
Names are external metadata. They need not match the body's local variable names. They identify
derivative inputs and outputs, generated C buffers, sparse tables, and assembly text, so each name
must be unique within its tree.

## Symbolic and numerical calls

Call a `Function` with its declared input tree. The leaves decide what the call means:

```python
symbolic = features((sc.sym("x0", 3), sc.sym("A0", (2, 3))))
numeric = features((np.ones(3), np.eye(2, 3)))

symbolic_sum, symbolic_projection = symbolic
numeric_sum, numeric_projection = numeric
```

`Expr` leaves create first-class `CALL` nodes in a larger expression graph. Numerical leaves
compile on first use, cache the shared library, and reconstruct the declared output tree.

`__call__` is a dispatcher over two methods you can also call directly, and should when the
distinction is the point you are making:

- `fn.symbolic_call(tree)` always builds a call node.

- `fn.numerical_call(tree)` always evaluates.

A tree that mixes `Expr` and numerical leaves is an error rather than a guess. Wrap the constants
in `sc.const` to make the symbolic reading explicit.

Structure is checked statically by ty and again at runtime. Shapes are checked at runtime because
shapes are values in Python's type system.

## A single leaf is unpacked

Only `sc.G` introduces a tuple. A tree of one `sc.L` *is* that leaf, so a one-leaf input takes the
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

The second line is wrong in a way worth knowing about, because it is not always loud. It iterates
the returned array along its first axis, exactly as NumPy or PyTorch would, so it raises for a
length-3 output but *succeeds* whenever the leading axis has length one — binding a scalar slice
instead of the whole tensor. Symbolic calls behave the same way, with an `Expr` in place of the
array.

## Grouping and the C signature

Grouping exists for Python readability and typing. The generated signature uses the leaves in tree
order. These declarations therefore have the same flat input signature:

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

Choose the grouping that matches the domain object passed by the caller.

## Compose functions

Calling a function with `Expr` leaves preserves the callee as a call in the graph:

```python
@sc.function(sc.L("x", 3), sc.L("square", ...))
def square(x: sc.Expr) -> sc.Expr:
    return x * x

@sc.function(sc.L("x", 3), sc.L("energy", ...))
def energy(x: sc.Expr) -> sc.Expr:
    return square(x).sum()
```

The generated C contains one `square` procedure and a call from `energy`. Differentiation preserves
that boundary instead of copying the callee graph into every call site.

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
