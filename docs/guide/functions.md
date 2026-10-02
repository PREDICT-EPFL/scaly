# Building functions

An `Expr` represents a symbolic value. A `Function` gives a calculation named
inputs and outputs, and is the unit of composition, differentiation, and code
generation. Its declarations determine the structure of symbolic and numerical
calls, and can leave input shapes open so that one definition serves several
shapes.

## Function authoring levels

This function computes a vector's sum and its projection through a matrix:

```python
import numpy as np
import scaly as sc

@sc.function(sc.arg("x", 3), sc.arg("A", (2, 3)),
             outputs=sc.group(sc.arg("sum", ()), sc.arg("projection", 2)))
def features(x: sc.Expr, A: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    return x.sum(), A @ x

total, projected = features(np.array([1.0, 2.0, 3.0]), np.eye(2, 3))
print(total)      # 6.0
print(projected)  # [1. 2.]
```

The decorator fixes the parameter count, tuple structure, names, and shapes, and
defines how calls behave at runtime. It also gives calls their static types.
Symbolic calls to `features` return `tuple[sc.Expr, sc.Expr]`, and numerical
calls return `tuple[np.ndarray, np.ndarray]`.

Scaly traces a function by running its Python body with symbolic inputs to record
a calculation. Body annotations do not change tracing or evaluation. A type checker uses them
to check agreement with the decorator, such as a return structure that disagrees
with `outputs=`, and to check operations inside the body, such as an invalid
`Expr` method. Without body annotations, the decorator still types calls, but
checking inside the body is less precise.

A declaration can be complete or leave parts out. Body annotations work the same
way in each form:

| Declaration | Call result types | Tracing |
| --- | --- | --- |
| Fully shaped inputs and `outputs=` | Symbolic and numerical types from the declaration | At decoration |
| Fully shaped inputs only | Symbolic type from the return annotation. Numerical `np.ndarray` for an `Expr` result, `Any` for a tuple | At decoration |
| Inputs with open shapes | As above, depending on whether `outputs=` is present | Once per new set of input shapes |
| Bare `@sc.function()` | Symbolic types from annotations, numerical `Any` | Once per new structure and shape |

Use complete declarations for exported interfaces. An inferred output can be
useful when an operation determines its shape, such as compact sparse derivative
values. Bare decorators suit small symbolic helpers.

## Parameters, leaves, and groups

`sc.arg` declares a *leaf*, one named array. `sc.group` combines arguments into
a *group*, a tuple of leaves or other groups. Each Python parameter takes one
declaration, and `outputs=` is keyword-only. A group suits a tuple passed around
as one value, such as a set of cost parameters:

```python
@sc.function(sc.arg("x", 2), sc.group(sc.arg("weights", 2), sc.arg("offset", ())),
             outputs=sc.arg("cost", ()))
def weighted(x: sc.Expr, params: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    weights, offset = params
    return (weights * x * x).sum() + offset

print(weighted(np.ones(2), (np.array([1.0, 2.0]), np.array(0.5))))  # 3.5
```

A group stays a tuple even when it contains one leaf. A leaf is passed and
returned directly. Unpacking a single-output call as `(y,) = f(x)` iterates over
the array instead. For a length-three result, Python raises `ValueError: too
many values to unpack (expected 1)`. A length-one result silently loses its
axis.

The decorator has typing overloads for up to eight parameters. Each group takes
one to eight parts at runtime. Nest groups for wider structures. Names need not
match Python parameter names. Input names must be unique across all
parameters, and output names must be unique across the output tree. Names such
as `"projection"` select outputs for [differentiation](derivatives.md). The
generated C interface takes flat leaves in declaration order. Grouping adds no C
buffers.

| Declaration | Meaning |
| --- | --- |
| `sc.arg("x", 3)` | Vector with shape `(3,)` |
| `sc.arg("A", (2, 3))` | Matrix with shape `(2, 3)` |
| `sc.arg("dt", ())` | Scalar with no axes |
| `sc.arg("x")` or `sc.arg("x", ...)` | Open input shape, or output shape inferred from the trace |
| `sc.arg("p", sc.TensorType((3,), diff=False))` | Non-differentiable vector with explicit tensor metadata |

A type checker sees `Expr` or `np.ndarray` leaves and tuple structure, and Scaly
checks shapes at runtime. A scalar `()` and a one-element vector `(1,)` are different shapes.

## Inferred outputs and shape templates

Without `outputs=`, Scaly reads the output tree from the trace. One leaf is named
after the function. Several leaves are named `out0`, `out1`, and so on in flat
order. Nested tuples remain nested.

```python
@sc.function(sc.arg("x", 3))
def energy(x: sc.Expr) -> sc.Expr:
    return (x * x).sum()

print(energy.instantiate().output_names)  # ('energy',)
```

Leaving an input shape out makes a template:

```python
@sc.function(sc.arg("x"), outputs=sc.arg("squared"))
def square(x: sc.Expr) -> sc.Expr:
    return x * x

print(square(np.ones(3)))  # [1. 1. 1.]
print(square(np.ones(5)))  # [1. 1. 1. 1. 1.]

three = square.instantiate((3,))
assert three is square.instantiate((3,))
assert len(square.instances) == 2
```

Numerical and symbolic calls with the same shapes select the same instance.
Fixed parts of a partial declaration still constrain calls. Scaly caches
instances by their resolved `TensorType`s. An open input uses the default float64 tensor type. A declaration can fix dtype and differentiability with
`TensorType`. Differentiability belongs to the declaration rather than the
actual argument.

Scaly traces a function with fully shaped inputs at decoration and keeps its
name. A template traces when it is first called with a new set of shapes. Each
instance gets a name containing the bound shapes and a short hash of the group
structure, so generated C symbols do not depend on call order. Two instances
used in the same graph that would produce the same C symbol raise an error.

Use explicit bindings for ahead-of-time export:

```python
from pathlib import Path
from scaly.codegen import write_module

write_module(square.instantiate((3,)), Path("generated"))
```

`instantiate` takes a tuple of shapes in flat input-leaf order, including fixed-shape
leaves. It also accepts `TensorType` entries that agree with the declaration. A
fully declared function needs only `instantiate()`. Exporting a template without
bindings raises an error. A command-line target can name a bound instance or a
function returning one.

## Bare helpers

A bare decorator infers the input structure and shapes from calls:

```python
@sc.function()
def scale(x: sc.Expr, gain: sc.Expr) -> sc.Expr:
    return x * gain

@sc.function(sc.arg("x", 3), outputs=sc.arg("y", 3))
def scaled(x: sc.Expr) -> sc.Expr:
    return scale(x, sc.const(2.0))

print(scaled(np.ones(3)))  # [2. 2. 2.]
```

Bare functions accept `Expr` or `np.ndarray` leaves. A tuple argument is always
read as a group. Lists and other array-likes are rejected because there is no
declaration to distinguish an array from a group. Convert them with `np.asarray`
or declare the input tree. Bare symbolic leaves use float64. Each new tuple
structure and shape combination gets its own instance. `instantiate` needs
declarations, so a bare function gets instances only from calls.

Bare input leaves are named `in0`, `in1`, and so on. These names can refer to
different parameters when the same helper accepts different structures. Use
declared names for stable derivative APIs.

## Tracing and calling

Scaly runs the Python body with symbolic `Expr` arguments when it traces an
instance. That happens at decoration for a fully shaped declaration, and on the
first call with new shapes for a template. Numerical calls generate and compile
C on first use and reuse its shared library. Results are NumPy arrays, including
zero-dimensional arrays for scalar outputs.

Captured Python values become constants at tracing time:

```python
gain = 2.0

@sc.function(sc.arg("x", 2), outputs=sc.arg("y", 2))
def fixed_scale(x: sc.Expr) -> sc.Expr:
    return gain * x

gain = 10.0
print(fixed_scale(np.ones(2)))  # [2. 2.]
```

!!! warning "Captured values in templates"
    Each instance captures Python values when it is first traced. Changing `gain`
    between calls at different shapes can give those instances different constants.
    Put values that change between evaluations in the input declaration.

Python loops also run during tracing and build graph nodes per iteration. Use
`vmap` for independent repeated structure.

Use Scaly operations inside the body. `x * x` is elementwise, `A @ x` is matrix
multiplication, and `x.sin()` is elementwise sine. Prepare numerical data with
NumPy outside the function and put fixed data in the graph with `sc.const`. The
[expression reference](../api/core.md#expressions) lists the supported
operations. For example, `x.sum()` reduces all entries and has no `axis`
argument.

!!! warning "Symbolic values as Python conditions"
    An `Expr` used as a Python truth value is treated as true during tracing.
    A Python `if` cannot select a branch at evaluation time.

```python
@sc.function(sc.arg("x", ()), outputs=sc.arg("y", ()))
def branch(x: sc.Expr) -> sc.Expr:
    return x if x else sc.const(10.0)

print(branch(np.array(0.0)))  # 0.0
```

The recorded expression is just `x`, so a zero input returns `0.0` rather than
`10.0`. Scaly has no symbolic conditional selection.

Symbolic calls record a call to the selected concrete instance. Derivatives
cross these calls, so models can be composed from functions. The standalone
[composition
example](https://github.com/PREDICT-EPFL/scaly/blob/main/examples/function_composition.py)
evaluates a composed function, differentiates it, and exports its C source. All
leaves in a call must be symbolic or all numerical. Wrap fixed data in
`sc.const` when passing it alongside a symbolic argument. The explicit methods
`f.symbolic_call(*args)` and `f.numerical_call(*args)` select the call kind
directly.

With no parameters, `f()` is numerical. Use `f.symbolic_call()` to embed a
constant function:

```python
@sc.function(outputs=sc.arg("value", ()))
def constant() -> sc.Expr:
    return sc.const(2.0)

assert isinstance(constant(), np.ndarray)
assert isinstance(constant.symbolic_call(), sc.Expr)
```

## Derivatives of templates

The [named derivative wrappers](derivatives.md) preserve the source's parameter
trees and bind once per source instance. `of` and `wrt` can be omitted when the
relevant name is unique. `sc.forward` appends a seed with the type of the selected
input, and `sc.adjoint` one with the type of the selected output, including dtype
and differentiability. Lagrangian Hessians append one multiplier parameter with
the source's output tree.

```python
grad = sc.gradient(energy)
print(grad(np.array([1.0, 2.0, 3.0])))  # [2. 4. 6.]

jvp = sc.forward(square)
print(jvp(np.array([2.0, 3.0]), np.ones(2)))  # [4. 6.]
```

Names from inferred trees are checked when Scaly traces the source instance.
Derivatives of a template carry the same shape suffix as the source, including
when given an explicit name.

## Regular repetition: `vmap`

`sc.vmap(f, N)` returns a callable with the same parameter and output structures
as `f`. Each output leaf gains a leading axis of length `N`:

```python
@sc.function(sc.arg("x"), outputs=sc.arg("sum", ()))
def reduce(x: sc.Expr) -> sc.Expr:
    return x.sum()

print(sc.vmap(reduce, 5)(np.arange(15.0).reshape(5, 3)))
# [ 3. 12. 21. 30. 39.]
```

Arguments with leading axis `N` bind the callee from their remaining axes. Use
`sc.broadcast(p)` to pass one whole tensor to every iteration. For fixed-shape
callees, a flat vector containing `N` chunks also works, and a tensor containing
just one chunk is broadcast. The leading-axis interpretation takes priority when
its per-iteration shape matches the declaration.

For overlapping rank-1 windows, `sc.window(x, start, stride)` reads a slice at
`start + i * stride` on iteration `i`. Its width comes from the callee's
declared shape. An open shape needs a leading-axis view or an explicitly
instantiated callee to determine that width.

```python
@sc.function(sc.arg("pair", 2), outputs=sc.arg("difference", ()))
def difference(pair: sc.Expr) -> sc.Expr:
    return pair[1] - pair[0]

print(sc.vmap(difference, 4)(sc.window(np.arange(5.0), 0, 1)))
# [1. 1. 1. 1.]
```

Symbolic mapped calls build map nodes directly in the caller, so their structure
stays visible to differentiation and loop optimization. Slices, reshapes, and
transposes that describe contiguous windows with a constant nonnegative stride
read from their base in place. Other views preserve their element order through
a copy. A numerical mapped call assembles batched arrays from windows and
broadcasts. Use a declared outer function when repeated evaluation should
perform the slicing in compiled C.

Mapped calls return the whole output tree. Call `.vec()` on a leaf when a
surrounding calculation needs a flat vector. Each used output has its own
generated loop. Iterations must be independent. One iteration's output does not
become the next iteration's input.

## Inspecting a graph

Shape-dependent metadata belongs to the concrete instance:

```python
instance = features.instantiate()
print(instance.input_names)    # ('x', 'A')
print(instance.output_shapes)  # ((), (2,))
print(sc.render_expr_assembly(instance))
```

`Function.inputs` and `Function.outputs` hold the declarations, including any
open shapes. Resolved input and output trees, expression nodes, tensor shapes,
and sparsity metadata belong to instances.
