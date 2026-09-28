# Building functions

An `Expr` represents a symbolic value. A `Function` gives a calculation named
inputs and outputs, and is the unit of composition, differentiation, and code
generation. Its declarations determine both the symbolic Python body and the
structure of numerical calls.

## Input and output declarations

This function computes a vector's sum and its projection through a matrix:

```python
import numpy as np
import scaly as sc

@sc.function(
    sc.G(sc.L("x", 3), sc.L("A", (2, 3))),
    sc.G(sc.L("sum", ...), sc.L("projection", 2)),
)
def features(inputs: tuple[sc.Expr, sc.Expr]) -> tuple[sc.Expr, sc.Expr]:
    x, A = inputs
    return x.sum(), A @ x

total, projected = features((np.array([1.0, 2.0, 3.0]), np.eye(2, 3)))
print(total)      # 6.0
print(projected)  # [1. 2.]
```

`sc.L` declares a *leaf*, one named input or output array. `sc.G` declares a
*group*, represented by a tuple that can contain leaves or other groups.
Here, the Python body receives one tuple containing `x` and `A`, and returns a
tuple containing the two results. Numerical calls follow that same structure.
After decoration, `features` is a `Function`, not the original Python function.

The annotations describe the symbolic body and let an IDE's type checker compare
its argument and return types with the declarations. They are optional and do
not change evaluation. Array shapes remain runtime checks.

The shape after a name describes the array:

| Declaration | Meaning |
| --- | --- |
| `sc.L("x", 3)` | vector with shape `(3,)` |
| `sc.L("A", (2, 3))` | matrix with two rows and three columns |
| `sc.L("dt", ())` | scalar, with no array axes |
| `sc.L("y", ...)` | output whose shape Scaly infers from the body |

Input shapes must be known when you define the function. For outputs, use `...`
unless you want Scaly to check a particular shape. A scalar with shape `()` and
a one-element vector with shape `(1,)` are different shapes.

Names such as `"projection"` let you select outputs for
[differentiation](derivatives.md). Names need not match Python variable names,
but each input name must be unique, as must each output name.

## Python execution and symbolic calculations

The decorator runs the Python body once with symbolic values of type `sc.Expr`.
An expression records a calculation rather than storing its result. For example,
`A @ x` records a matrix-vector product whose numerical inputs will arrive later.

Use Scaly operations inside the body. Arithmetic such as `x * x` is elementwise,
`A @ x` is matrix multiplication, and `x.sin()` applies sine elementwise. Use
NumPy to prepare data outside the function and `sc.const(array)` to put fixed
numerical data inside a symbolic calculation.

Python runs when the function is defined, not on each evaluation. This matters
for values captured from the surrounding Python scope:

```python
gain = 2.0

@sc.function(sc.L("x", 2), sc.L("y", ...))
def scale(x: sc.Expr) -> sc.Expr:
    return gain * x

gain = 10.0
print(scale(np.ones(2)))  # [2. 2.]
```

The multiplication already contains the constant `2.0`. Changing the Python
name `gain` cannot change that expression. A gain that varies between calls
belongs in the input declaration.

A Python loop similarly executes during construction and builds its body once
per iteration. It does not become a loop in the recorded graph. Independent
repetition can instead use [`vmap`](#regular-repetition-vmap).

### Symbolic values are not Python conditions

A condition based on a symbolic input cannot select a branch at evaluation time.
Nothing prevents using an `Expr` as a Python truth value:

```python
@sc.function(sc.L("x", ()), sc.L("y", ...))
def branch(x: sc.Expr) -> sc.Expr:
    return x if x else sc.const(10.0)

print(branch(np.array(0.0)))  # 0.0, not 10.0
```

The symbolic `x` is treated as true while the body runs, so only the expression
`x` is returned. This function has no recorded branch. Scaly has no symbolic
conditional selection.

### Supported array operations

`Expr` supports fixed shapes, indexing, broadcasting, and common arithmetic.
The available operations cover part of the NumPy API, and a NumPy operation
missing from the reference is not available on `Expr`. For example, `x.sum()` reduces all
elements, but it has no `axis` argument. The
[expression reference](../api/core.md#expressions) lists supported methods.
NumPy arrays are numerical data, so use Scaly operations to construct symbolic
calculations rather than assuming a NumPy function accepts `Expr`.

## Numerical evaluation and symbolic composition

Numerical inputs evaluate the function. The first call generates and compiles C.
Later calls reuse the compiled code. Results are NumPy arrays, including
zero-dimensional arrays for scalar outputs.

Symbolic inputs let you use one function inside another:

```python
@sc.function(sc.L("x", 3), sc.L("square", ...))
def square(x: sc.Expr) -> sc.Expr:
    return x * x

@sc.function(sc.L("x", 3), sc.L("energy", ...))
def energy(x: sc.Expr) -> sc.Expr:
    return square(x).sum()

print(energy(np.array([1.0, 2.0, 3.0])))  # 14.0
```

The symbolic calculation records a call to `square`. Derivatives work across
this boundary, so you can organize a model into functions without manually
combining their bodies. Differentiation and compilation may expand an ordinary
callee into its caller. The standalone
[composition example](https://github.com/PREDICT-EPFL/scaly/blob/main/examples/function_composition.py)
evaluates this calculation, differentiates it, and exports its C source.

All inputs to a call must be numerical or all must be symbolic. To combine a
symbolic argument with fixed data, wrap the data in `sc.const`. Explicit
`fn.symbolic_call(inputs)` and `fn.numerical_call(inputs)` methods are also
available.

## Arrays, tuples, and fixed shapes

A single `sc.L` takes or returns an array directly. Only `sc.G` introduces a tuple:

```python
squared = square(np.ones(3))
total, projected = features((np.ones(3), np.eye(2, 3)))
```

A single-output call already returns its array. Writing `(squared,) =
square(...)` instead iterates over that array. For this length-three result,
Python raises `ValueError: too many values to unpack (expected 1)`. A length-one
result would silently unpack to one element, changing its shape.

Function declarations also fix dimensions. Scaly does not infer a new function
shape from each numerical call:

```python
try:
    square(np.ones((3, 1)))
except ValueError as error:
    print(error)
# square.numerical_call: expected shape (3,) for 'x', got (3, 1)
```

A three-element column matrix and a vector have the same number of entries,
but different shapes. The same distinction applies to a scalar `()` and a
one-element vector `(1,)`.

Groups may contain other groups. For example, this input declaration expects
`((state, control), (weights, dt))`:

```python
inputs = sc.G(
    sc.G(sc.L("state", 4), sc.L("control", 2)),
    sc.G(sc.L("weights", 10), sc.L("dt", ())),
)
```

The [generated C interface](codegen.md#the-pointer-entry) uses the
individual arrays in declaration order.
Python grouping does not add C buffers. Runtime checks validate the structure
and shapes. Type annotations can also catch mismatched tuple structures before
running the code.

## Regular repetition: `vmap`

Use `sc.vmap` to apply the same function to independent slices of a longer array.
For example, sum five consecutive groups of three values:

```python
@sc.function(sc.L("x", 3), sc.L("sum", ...))
def reduce3(x: sc.Expr) -> sc.Expr:
    return x.sum()

@sc.function(sc.L("xs", 15), sc.L("sums", ...))
def sum_groups(xs: sc.Expr) -> sc.Expr:
    return sc.vmap(reduce3, 5, [xs])

print(sum_groups(np.arange(15.0)))  # [ 3. 12. 21. 30. 39.]
```

The second argument, `5`, is the number of calls. `reduce3` takes three values,
so Scaly splits the 15-element input into five consecutive chunks. The slices
are `xs[0:3]`, `xs[3:6]`, and so on. An input with only three elements would
instead be reused in all five calls.

Supply one array per function input, in declaration order, or use a dictionary
keyed by input name. For example, `{"x": xs}` means the same as `[xs]` here.
Each array must contain either all the chunks or exactly one chunk to reuse.

For overlapping or offset windows, specify `(array, start, stride)` explicitly.
On iteration `i`, Scaly reads from index `start + i * stride`, taking as many
elements as that input requires. Thus `[(xs, 0, 3)]` is an explicit version of
`[xs]` in this example. A stride of zero reuses the same slice in every call.

Outer arrays must be one-dimensional. Scaly flattens each iteration's output
and concatenates the results into one vector. For functions with several
outputs, `output=0` selects the first output, `output=1` the second, and so on.

Mapped repetition remains represented as a loop, including in generated derivatives. In optimal
control, this is useful for evaluating dynamics defects at all horizon stages.
The calls must be independent. `vmap` does not feed one iteration's output into
the next iteration. The loop body need not grow with the number of calls, but
numerical work, input and output storage, and derivative sparsity tables can.

## Function metadata

```python
print(features.input_names)    # ('x', 'A')
print(features.output_shapes)  # ((), (2,))
print(sc.render_expr_assembly(features))
```

The assembly listing shows the recorded operations.

Generated code runs on the host CPU.
