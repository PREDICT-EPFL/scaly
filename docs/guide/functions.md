# Building functions

An `Expr` represents a symbolic value. A `Function` gives a calculation named
inputs and outputs, and is the unit of composition, differentiation, and code
generation. Its declarations give the body its symbolic inputs and fix the
structure of symbolic and numerical calls.

## Input and output declarations

This function computes a vector's sum and its projection through a matrix:

```python
import numpy as np
import scaly as sc

@sc.function(
    sc.arg("x", 3), sc.arg("A", (2, 3)),
    outputs=sc.group(sc.arg("sum", ()), sc.arg("projection", 2)),
)
def features(x: sc.Expr, A: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    return x.sum(), A @ x

total, projected = features(np.array([1.0, 2.0, 3.0]), np.eye(2, 3))
print(total)      # 6.0
print(projected)  # [1. 2.]
```

`sc.arg` declares one named input or output array. `sc.group` combines several
declarations into a tuple, and a group may contain other groups. The decorator
takes one declaration per Python parameter, so the body receives `x` and `A`
separately. The keyword-only `outputs=` declares what the body returns. Here
that is a group, so the body returns a tuple and numerical calls return one.
After decoration, `features` is a `Function`, not the original Python function.

The decorator never calls the body with arrays. For each input declaration it
creates a symbolic input, the same `Expr` that `sc.sym("x", 3)` creates, runs
the body once with those symbols, and records the returned expressions. The
recorded graph is what evaluation, differentiation, and code generation use.
You rarely need `sc.sym` yourself, because declarations create the symbols.

!!! info "Declarations form a tree"
    Declarations nest the way [JAX pytrees](https://docs.jax.dev/en/latest/pytrees.html)
    do. `sc.arg` is a leaf and `sc.group` is a node whose children are leaves or
    further groups. Every symbolic call, numerical call, and result has the same
    nested tuple structure as the declarations. The API reference uses this leaf
    and tree vocabulary.

The annotations describe the symbolic body and let an IDE's type checker compare
its parameter and return types with the declarations. They are optional and do
not change evaluation. Array shapes remain runtime checks.

The shape after a name describes the array:

| Declaration | Meaning |
| --- | --- |
| `sc.arg("x", 3)` | vector with shape `(3,)` |
| `sc.arg("A", (2, 3))` | matrix with two rows and three columns |
| `sc.arg("dt", ())` | scalar, with no array axes |
| `sc.arg("y")` | shape left open, see [below](#leaving-shapes-out) |
| `sc.arg("p", sc.TensorType((3,), diff=False))` | vector that is never differentiated |

A scalar with shape `()` and a one-element vector with shape `(1,)` are
different shapes.

Names such as `"projection"` let you select outputs for
[differentiation](derivatives.md). Names need not match Python parameter names,
but each input name must be unique, as must each output name.

## Leaving shapes out

An output declared without a shape takes its shape from the returned expression.
Leaving `outputs=` out altogether does the same for the whole output tree. One
returned value is named after the function, and several are named `out0`,
`out1`, and so on:

```python
@sc.function(sc.arg("x", 3))
def cost(x: sc.Expr) -> sc.Expr:
    return (x * x).sum()

print(cost(np.array([1.0, 2.0, 3.0])))  # 14.0
print(cost.instantiate().output_names)  # ('cost',)
```

Input shapes must be known before the body can run. An input declared without a
shape makes the function a template, whose body runs once for each new set of
input shapes:

```python
@sc.function(sc.arg("x"), outputs=sc.arg("squared"))
def square(x: sc.Expr) -> sc.Expr:
    return x * x

print(square(np.ones(3)))  # [1. 1. 1.]
print(square(np.ones(5)))  # [1. 1. 1. 1. 1.]
```

Each set of shapes gives one *instance*, a concrete graph with its own generated
C. Calls with the same shapes share the instance, whether they are symbolic or
numerical. `square.instantiate((3,))` returns the instance for a tuple of input
shapes without calling the function. A fully declared function has one
instance, which `instantiate()` returns with no arguments.

Code generation needs a concrete graph, so pass an instance to export a
template ahead of time:

```python
from pathlib import Path
from scaly.codegen import write_module

write_module(square.instantiate((3,)), Path("generated"))
```

Passing `square` itself raises `TypeError: square has shape holes in ('x',);
pass shapes to bind them`. The exported symbol appends the bound shapes to the
function name, here `square_3` followed by a short suffix, so it does not depend
on which shapes were called first. [Exported C symbols](../dev/versioning.md#exported-c-symbols)
gives the exact spelling.

An open input is a differentiable float64 array. A `TensorType` in place of the
shape fixes the dtype or marks an input as never differentiated, and fixes the
shape with it.

### Bare helpers

A decorator with no declarations at all reads the parameter structure and shapes
from each call:

```python
@sc.function()
def product(x: sc.Expr, gain: sc.Expr) -> sc.Expr:
    return x * gain

@sc.function(sc.arg("x", 3), outputs=sc.arg("y", 3))
def doubled(x: sc.Expr) -> sc.Expr:
    return product(x, sc.const(2.0))

print(doubled(np.ones(3)))  # [2. 2. 2.]
```

Every array passed to a bare function must be an `Expr` or a NumPy array,
because without a declaration nothing distinguishes a list from a group. Its
inputs are named `in0`, `in1`, and so on.

A body without parameters has nothing to read from a call, so `@sc.function()`
on it counts as fully declared. The decorator runs the body at once, and the
single instance and its C symbol take the function name:

```python
@sc.function()
def offset() -> sc.Expr:
    return sc.const([1.0, -1.0])

print(offset.instantiate().name)  # offset
```

### Which declaration to use

Declare every input shape for anything you export, differentiate by name, or
hand to `sc.problem`. Scaly checks the shapes when the function is defined.
There is exactly one instance, and its C symbol is the function name. Output
shapes can always be left out. The body determines them and the number of
instances does not change.

Leave an input shape open when one body is useful at several sizes, such as a
norm or a stage cost shared by models of different dimension. Each caller binds
its own instance at the shapes it uses, and exporting one means choosing its
shapes with `instantiate`. The template saves repeating the body, not the
shapes. They move from the declaration to the call sites and the export.

Use a bare decorator only for small symbolic helpers called from declared
functions. Nothing is checked before the first call, the inputs carry generated
names, and a derivative cannot select them by name.

## Python execution and symbolic calculations

The decorator runs the Python body once with symbolic values of type `sc.Expr`.
An expression records a calculation rather than storing its result. For example,
`A @ x` records a matrix-vector product whose numerical inputs will arrive later.
A template runs its body at the first call with each new set of shapes instead.

Use Scaly operations inside the body. Arithmetic such as `x * x` is elementwise,
`A @ x` is matrix multiplication, and `x.sin()` applies sine elementwise. Use
NumPy to prepare data outside the function and `sc.const(array)` to put fixed
numerical data inside a symbolic calculation.

Python runs when the function is defined, not on each evaluation. The next
three subsections show where this changes the meaning of ordinary Python.

### Python values are captured once

A value read from the surrounding Python scope enters the graph as a constant:

```python
gain = 2.0

@sc.function(sc.arg("x", 2), outputs=sc.arg("y"))
def scale(x: sc.Expr) -> sc.Expr:
    return gain * x

gain = 10.0
print(scale(np.ones(2)))  # [2. 2.]
```

The multiplication already contains the constant `2.0`. Changing the Python
name `gain` cannot change that expression. A gain that varies between calls
belongs in the input declaration. For a template, each instance captures the
values current at its first call, so changing `gain` between calls at two
shapes gives those instances different constants.

A Python loop similarly executes during construction and builds its body once
per iteration. It does not become a loop in the recorded graph. Independent
repetition can instead use [`vmap`](#regular-repetition-vmap).

### Prints happen while the body runs

A `print` in the body runs when the decorator records the graph, not when the
function is evaluated:

```python
@sc.function(sc.arg("x", 2), outputs=sc.arg("y"))
def noisy(x: sc.Expr) -> sc.Expr:
    print("called with", x)
    return 2 * x

print(noisy(np.ones(2)))   # [2. 2.]
print(noisy(np.zeros(2)))  # [0. 0.]
```

The message appears once, before either call, and shows a symbolic `Expr`
rather than numbers. Neither numerical call prints anything, because the
compiled code runs instead of the Python body. The same holds for logging,
breakpoints, and any other side effect. To inspect numerical values, print the
result of the call.

### Symbolic values are not Python conditions

A condition based on a symbolic input cannot select a branch at evaluation time.
Nothing prevents using an `Expr` as a Python truth value:

```python
@sc.function(sc.arg("x", ()), outputs=sc.arg("y"))
def branch(x: sc.Expr) -> sc.Expr:
    return x if x else sc.const(10.0)

print(branch(np.array(0.0)))  # 0.0, not 10.0
```

The symbolic `x` is treated as true while the body runs, so only the expression
`x` is returned. This function has no recorded branch.

!!! warning "A Python branch on an `Expr` is silently wrong"
    Scaly has no symbolic conditional selection, and nothing raises when a body
    tests an `Expr`. The function keeps whichever branch Python took while the
    body ran, for every later evaluation.

### Supported array operations

`Expr` supports fixed shapes, indexing, broadcasting, and common arithmetic.
The available operations cover part of the NumPy API, and a NumPy operation
missing from the reference is not available on `Expr`. For example, `x.sum()` reduces all
elements, but it has no `axis` argument. The
[expression reference](../api/core.md#expressions) lists supported methods.
NumPy arrays are numerical data, so use Scaly operations to construct symbolic
calculations rather than assuming a NumPy function accepts `Expr`.

`sc.gather(x, indices)` reads flat entries of `x`, with the result shaped like
`indices`. `sc.scatter(values, indices, shape)` places values into a zero array.
Repeated indices sum in input order. `sc.segment_sum(values, ids, n)` uses the
same operation to sum values into `n` segments, with zero for empty segments.
Indices and segment ids are fixed when the function is built.

```python
@sc.function(sc.arg("values", 4), outputs=sc.arg("totals"))
def totals(values: sc.Expr) -> sc.Expr:
    return sc.segment_sum(values, [0, 2, 0, 2], 4)

print(totals(np.array([1.0, 2.0, 3.0, 4.0])))  # [4. 0. 6. 0.]
```

## Numerical evaluation and symbolic composition

Numerical inputs evaluate the function. The first call generates and compiles C.
Later calls reuse the compiled code. Results are NumPy arrays, including
zero-dimensional arrays for scalar outputs.

Symbolic inputs let you use one function inside another:

```python
@sc.function(sc.arg("x", 3), outputs=sc.arg("energy"))
def energy(x: sc.Expr) -> sc.Expr:
    return square(x).sum()

print(energy(np.array([1.0, 2.0, 3.0])))  # 14.0
```

The symbolic calculation records a call to `square`, bound at shape `(3,)`
because that is the shape of `x`. Derivatives work across this boundary, so you
can organize a model into functions without manually combining their bodies.
Differentiation and compilation may expand an ordinary callee into its caller.
The standalone
[composition example](https://github.com/PREDICT-EPFL/scaly/blob/main/examples/function_composition.py)
evaluates this calculation, differentiates it, and exports its C source.

All inputs to a call must be numerical or all must be symbolic. To combine a
symbolic argument with fixed data, wrap the data in `sc.const`. Explicit
`fn.symbolic_call(*args)` and `fn.numerical_call(*args)` methods are also
available. A function with no parameters needs them to choose, because `fn()`
has no arguments to inspect and evaluates numerically:

```python
@sc.function(outputs=sc.arg("value", ()))
def constant() -> sc.Expr:
    return sc.const(2.0)

print(type(constant()).__name__)                # ndarray
print(type(constant.symbolic_call()).__name__)  # Expr
```

## Arrays, tuples, and fixed shapes

A single `sc.arg` takes or returns an array directly. Only `sc.group` introduces
a tuple:

```python
squared = square(np.ones(3))
total, projected = features(np.ones(3), np.eye(2, 3))
```

A single-output call already returns its array. Writing `(squared,) =
square(...)` instead iterates over that array. For this length-three result,
Python raises `ValueError: too many values to unpack (expected 1)`. A length-one
result would silently unpack to one element, changing its shape.

A declared input shape is fixed. A numerical call with another shape is an
error, not a new instance:

```python
try:
    features(np.ones((3, 1)), np.eye(2, 3))
except ValueError as error:
    print(error)
# features: expected shape (3,) for 'x', got (3, 1)
```

A three-element column matrix and a vector have the same number of entries,
but different shapes. The same distinction applies to a scalar `()` and a
one-element vector `(1,)`. Only an open input accepts a new shape, by binding
a new instance: `square(np.ones((3, 1)))` returns a `(3, 1)` array.

A group can be a parameter, and groups may contain other groups. This function
takes its cost parameters as one tuple:

```python
@sc.function(
    sc.arg("x", 2), sc.group(sc.arg("weights", 2), sc.arg("offset", ())),
    outputs=sc.arg("cost", ()),
)
def weighted(x: sc.Expr, params: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    weights, offset = params
    return (weights * x * x).sum() + offset

print(weighted(np.ones(2), (np.array([1.0, 2.0]), np.array(0.5))))  # 3.5
```

The [generated C interface](codegen.md#the-pointer-entry) uses the
individual arrays in declaration order.
Python grouping does not add C buffers. Runtime checks validate the structure
and shapes. Type annotations can also catch mismatched tuple structures before
running the code.

## Regular repetition: `vmap`

Use `sc.vmap` to apply the same function to every slice along the leading axis
of its arguments. For example, sum five groups of three values:

```python
@sc.function(sc.arg("x", 3), outputs=sc.arg("sum", ()))
def reduce3(x: sc.Expr) -> sc.Expr:
    return x.sum()

@sc.function(sc.arg("xs", 15), outputs=sc.arg("sums"))
def sum_groups(xs: sc.Expr) -> sc.Expr:
    return sc.vmap(reduce3, 5)(xs.reshape((5, 3)))

print(sum_groups(np.arange(15.0)))  # [ 3. 12. 21. 30. 39.]
```

`sc.vmap(reduce3, 5)` is a callable with the same parameters as `reduce3` that
calls it five times. Each argument has a leading axis of length five, and call
`i` receives slice `i` along that axis. Here each call receives one row of the
`(5, 3)` matrix. The mapped axis is always the leading one, so reshape an
argument to put it first. The reshape only describes how the vector is read,
and the generated loop indexes the original storage. A transpose copies the
argument first.

Each output gains a leading axis of length five. The result here has shape
`(5,)`, and a callee returning a vector of length two would give `(5, 2)`.
`.vec()` flattens such a result back into one vector when the surrounding
calculation needs one. A callee with several outputs returns a tuple of mapped
arrays. A mapped callable also accepts numerical arrays directly.

An argument that every call should see whole, such as a shared parameter, is
wrapped in `sc.broadcast`:

```python
@sc.function(sc.arg("x", 3), sc.arg("gain", ()), outputs=sc.arg("y", 3))
def scaled(x: sc.Expr, gain: sc.Expr) -> sc.Expr:
    return gain * x

print(sc.vmap(scaled, 2)(np.arange(6.0).reshape(2, 3), sc.broadcast(np.array(2.0))))
# [[ 0.  2.  4.]
#  [ 6.  8. 10.]]
```

For overlapping or offset windows into a vector, `sc.window(x, start, stride)`
reads from index `start + i * stride` on call `i`, taking as many elements as
the callee declares:

```python
@sc.function(sc.arg("pair", 2), outputs=sc.arg("difference", ()))
def difference(pair: sc.Expr) -> sc.Expr:
    return pair[1] - pair[0]

print(sc.vmap(difference, 4)(sc.window(np.arange(5.0), 0, 1)))
# [1. 1. 1. 1.]
```

Mapped repetition remains represented as a loop, including in generated
derivatives. In optimal control, this is useful for evaluating dynamics defects
at all horizon stages. The calls must be independent. `vmap` does not feed one
iteration's output into the next iteration. The loop body need not grow with
the number of calls, but numerical work, input and output storage, and
derivative sparsity tables can.

## Function metadata

Shapes and the recorded operations belong to an instance:

```python
instance = features.instantiate()
print(instance.input_names)    # ('x', 'A')
print(instance.output_shapes)  # ((), (2,))
print(sc.render_expr_assembly(instance))
```

The assembly listing shows the recorded operations. `features.inputs` and
`features.outputs` hold the declarations, including any open shapes.
