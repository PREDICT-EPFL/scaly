# Building functions

A `Function` is a named expression graph with declared input and output trees. The same trees
describe symbolic calls with `Expr` leaves and numerical calls with NumPy-array leaves. They also
give ty enough information to reject a call with the wrong structure.

## Declare a function

Build trees from two constructors:

- `al.L(name, shape)` declares one tensor.

- `al.G(*trees)` groups two to eight trees and may be nested.

An integer shape means a rank-1 tensor, `()` is a scalar, and a tuple is used as written. Pass a
`TensorType` when you need an explicit dtype or differentiability flag.

```python
import alloy as al
import numpy as np

@al.function(
    al.G(al.L("x", 3), al.L("A", (2, 3))),
    al.G(al.L("sum", ...), al.L("projection", 2)),
)
def features(inputs: tuple[al.Expr, al.Expr]) -> tuple[al.Expr, al.Expr]:
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

Use the typed call that matches the leaf kind:

```python
symbolic = features.symbolic_call((al.sym("x0", 3), al.sym("A0", (2, 3))))
numeric = features.numerical_call((np.ones(3), np.eye(2, 3)))

symbolic_sum, symbolic_projection = symbolic
numeric_sum, numeric_projection = numeric
```

`symbolic_call` creates first-class `CALL` nodes in a larger expression graph. `numerical_call`
compiles on first use, caches the shared library, and reconstructs the declared output tree.
Structure is checked statically by ty and again at runtime. Shapes are checked at runtime because
shapes are values in Python's type system.
The lower-level compatibility calls remain available:

- `fn.call([...])` accepts flat symbolic arguments and returns a flat tuple.

- `fn(...)` accepts flat numerical arguments and returns one array or a flat tuple.

- `fn.eval_list(...)` always returns a flat list.

Use the tree calls in typed code.

## Grouping and the C signature

Grouping exists for Python readability and typing. The generated signature uses the leaves in tree
order. These declarations therefore have the same flat input signature:

```python
nested = al.G(
    al.G(al.L("state", 4), al.L("control", 2)),
    al.G(al.L("weights", 10), al.L("dt", ())),
)
flat = al.G(
    al.L("state", 4),
    al.L("control", 2),
    al.L("weights", 10),
    al.L("dt", ()),
)
```

Choose the grouping that matches the domain object passed by the caller.

## Compose functions

`symbolic_call` preserves the callee as a call in the graph:

```python
@al.function(al.L("x", 3), al.L("square", ...))
def square(x: al.Expr) -> al.Expr:
    return x * x

@al.function(al.L("x", 3), al.L("energy", ...))
def energy(x: al.Expr) -> al.Expr:
    return square.symbolic_call(x).sum()
```

The generated C contains one `square` procedure and a call from `energy`. Differentiation preserves
that boundary instead of copying the callee graph into every call site.

## Regular repetition: `vmap`

Use `al.vmap` when every iteration applies the same function to a different slice. The mapping
stays one node through differentiation and lowers to a C loop.

```python
@al.function(al.L("x", 3), al.L("sum", ...))
def reduce3(x: al.Expr) -> al.Expr:
    return x.sum().reshape((1,))

xs = al.sym("xs", 15)
mapped = al.vmap(reduce3, 5, [(xs, 0, 3)])
```

Each input specification is `(outer, start, stride)`. Iteration `i` reads a slice beginning at
`start + i * stride`. A zero stride broadcasts one slice across every iteration. See
[Sparsity](sparsity.md) for how mapped structure reduces derivative construction and generated
source size.

## Inspect and verify

```python
al.verify_expr(symbolic_projection)
print(al.format_expr(symbolic_projection))
print(al.render_expr_assembly(features))
```

Construction checks declarations and shapes. `al.verify_expr` checks the complete expression graph
and raises `VerifyError` at the first invalid node.

## Lowering hints and placement

`scalar()`, `block()`, and `opaque()` record lowering preferences on expressions. `opaque()` is
also a real boundary that prevents the lowerer from looking inside. The other hints currently
document intent while the default lowering policy remains in control.
`fn.with_device("cuda:0")` records device placement and validates dtypes against the backend
capability table. Only host lowering is implemented today.
