# Building functions

A `Function` holds a Python body and its input and output declarations. It traces the body once
for each binding of the input shapes and caches the resulting concrete graph. Symbolic calls
compose graphs; numerical calls evaluate compiled C.

## Function authoring levels

The decorator defines runtime behavior and the types of calls to the decorated function. Body
annotations serve two purposes: a type checker checks that they agree with the decorator, and it
checks operations inside the body. Annotations do not change tracing or numerical evaluation.

For a function with a stable interface, specify the parameter trees, output tree, and body types:

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

`features` is now a `Function`. Its decorator fixes the parameter count, tuple structure, names,
and shapes. It gives symbolic calls the result type `tuple[sc.Expr, sc.Expr]` and numerical calls
`tuple[np.ndarray, np.ndarray]`. The body annotations let the checker reject, for example, an
invalid `Expr` method or a return structure that disagrees with `outputs=`. Without body annotations,
the decorator still types calls, but checking inside an unannotated body is less precise.

Choose how much to declare according to the function's role:

| Authoring form | What it provides |
| --- | --- |
| Complete input and output declarations | Precise symbolic and numerical call structures, declared names and shapes, immediate tracing and validation |
| Declared inputs, omitted `outputs=` | Input call typing; output names, structure and shapes inferred from the trace. The symbolic result type comes from the body's return annotation. An `Expr` result has numerical type `np.ndarray`; a nested result has numerical type `Any` |
| Input shape holes, with or without `outputs=` | The same declared call structures at several shapes. Scaly traces and caches one instance per shape binding; shape-dependent errors occur at binding |
| Bare `@sc.function()` | Structure and shapes inferred from each call. Body annotations type the symbolic parameters and result; the numerical side is `Any`. Use it for small symbolic helpers |

These choices combine with body annotations. They do not require another decorator or a tracing
flag. Benchmarks and exported interfaces usually benefit from complete declarations and annotated
bodies. An inferred output can be useful when an operation determines its shape, such as compact
sparse derivative values.

## Parameters, leaves, and groups

`sc.arg` declares one named tensor. `sc.group` declares a tuple of leaves or nested groups. Pass
one declaration per Python parameter; `outputs=` is keyword-only.

```python
@sc.function(sc.group(sc.arg("x", 3), sc.arg("p", ())), outputs=sc.arg("cost", ()))
def cost(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    x, p = inputs
    return (x * x).sum() * p

print(cost((np.ones(3), np.array(2.0))))  # 6.0
```

The group above is one parameter. It stays a tuple even if it contains just one leaf. A leaf is
passed and returned directly; only a group introduces a tuple. Do not destructure a single-output
call as `(result,) = f(x)`: that iterates the resulting array and may silently change its shape.

The decorator supports up to eight typed parameters, and each group supports up to eight parts.
Nest groups for wider structures. Names need not match local Python variable names. Input names
must be unique across all parameters, and output names must be unique across the output tree.
The generated C interface takes flat leaves in declaration order; grouping adds no C buffers.

| Declaration | Meaning |
| --- | --- |
| `sc.arg("x", 3)` | Vector with shape `(3,)` |
| `sc.arg("A", (2, 3))` | Matrix with shape `(2, 3)` |
| `sc.arg("dt", ())` | Scalar with no axes |
| `sc.arg("x")` or `sc.arg("x", ...)` | Input shape hole, or output shape inferred from the trace |
| `sc.arg("p", sc.TensorType((3,), diff=False))` | Fixed vector with explicit tensor metadata |

A scalar `()` and a one-element vector `(1,)` have different shapes. Static types check leaf kinds
and tuple structure. Shapes remain runtime checks.

## Inferred outputs and shape templates

Omitting `outputs=` reads the output tree from the trace. One leaf is named after the function;
several leaves are named `out0`, `out1`, and so on in flat order. Nested tuples remain nested.

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

Numerical and symbolic calls with the same shapes select the same instance. Fixed parts of a
partial declaration still constrain calls. The instance key contains each resolved `TensorType`;
shape holes use the default float64 tensor type. A declaration can fix dtype and differentiability
with `TensorType`. Differentiability belongs to the declaration, rather than the actual argument.

A fully shaped input declaration traces at decoration and keeps the function's name. A template
traces at first binding. Each instance has a deterministic name containing its open shapes and a
nesting digest. Names depend on the declaration and binding, not the order of calls. Distinct
reachable instances with the same generated C identifier raise an error.

Use explicit bindings for ahead-of-time export:

```python
from pathlib import Path
from scaly.codegen import write_module

write_module(square.instantiate((3,)), Path("generated"))
```

`instantiate` takes a tuple of shapes in flat input-leaf order, including fixed leaves. It also
accepts `TensorType` entries that agree with the declaration. A fully declared function needs only
`instantiate()`. Exporting a template without bindings raises an error; a CLI target can name a
bound instance or a factory returning one.

### Bare helpers

```python
@sc.function()
def scale(x: sc.Expr, gain: sc.Expr) -> sc.Expr:
    return x * gain

@sc.function(sc.arg("x", 3), outputs=sc.arg("y", 3))
def scaled(x: sc.Expr) -> sc.Expr:
    return scale(x, sc.const(2.0))
```

Bare functions accept `Expr` or `np.ndarray` leaves. Every tuple is structure. Lists and other
array-likes are rejected because there is no declaration to distinguish an array from a group;
convert them with `np.asarray` or declare the input tree. Bare symbolic leaves use float64.
Each new tuple structure and shape combination gets its own instance. Bare functions bind only
through calls, so `instantiate(shapes)` requires adding declarations first.

Bare input leaves are named `in0`, `in1`, and so on. These names can refer to different parameters
when the same helper accepts different structures. Use declared names for stable derivative APIs.

## Tracing and calling

The Python body executes while Scaly traces an instance, with symbolic `Expr` arguments. Numerical
calls then generate and compile C on first use and reuse its shared library. Results are NumPy
arrays, including zero-dimensional arrays for scalar outputs.

Captured Python values become constants at tracing time:

```python
gain = 2.0

@sc.function(sc.arg("x", 2), outputs=sc.arg("y", 2))
def fixed_scale(x: sc.Expr) -> sc.Expr:
    return gain * x

gain = 10.0
print(fixed_scale(np.ones(2)))  # [2. 2.]
```

For templates, capture happens when each instance is first traced. Put values that change between
evaluations in the input declaration. Python loops also run during tracing and build graph nodes
per iteration; use `vmap` for independent repeated structure.

Use Scaly operations inside the body. `x * x` is elementwise, `A @ x` is matrix multiplication,
and `x.sin()` is elementwise sine. Prepare numerical data with NumPy outside the function and put
fixed data in the graph with `sc.const`. The [expression reference](../api/core.md#expressions)
lists the supported operations. For example, `x.sum()` reduces all entries and has no `axis` argument.

A symbolic value cannot select a numerical branch using Python `if`. An `Expr` used as a Python
truth value is treated as true during tracing, so `x if x else sc.const(10.0)` records only `x`.

Symbolic calls record a call to the selected concrete instance. Derivatives cross these calls,
so models can be composed from functions. All leaves in a call must be symbolic or all numerical.
Wrap fixed data in `sc.const` when passing it alongside a symbolic argument. The explicit methods
`f.symbolic_call(*args)` and `f.numerical_call(*args)` select the call kind directly.

With no parameters, `f()` is numerical. Use `f.symbolic_call()` to embed a constant function:

```python
@sc.function(outputs=sc.arg("value", ()))
def constant() -> sc.Expr:
    return sc.const(2.0)

assert isinstance(constant(), np.ndarray)
assert isinstance(constant.symbolic_call(), sc.Expr)
```

## Derivatives of templates

The [named derivative wrappers](derivatives.md) preserve the source's parameter trees and bind
once per source instance. `of` and `wrt` can be omitted when the relevant name is unique.
`forward` and `adjoint` append one seed parameter. The seed copies the selected source leaf's
whole type. Lagrangian Hessians append one multiplier parameter with the source's output tree.

```python
grad = sc.gradient(energy)
print(grad(np.array([1.0, 2.0, 3.0])))  # [2. 4. 6.]

jvp = sc.forward(square)
print(jvp(np.array([2.0, 3.0]), np.ones(2)))  # [4. 6.]
```

Names from inferred trees are checked at binding. Explicit derivative names receive the source's
specialization suffix when the source has holes.

## Regular repetition: `vmap`

`sc.vmap(f, N)` returns a callable with the same parameter and output structures as `f`. Each
output leaf gains a leading axis of length `N`:

```python
@sc.function(sc.arg("x"), outputs=sc.arg("sum", ()))
def reduce(x: sc.Expr) -> sc.Expr:
    return x.sum()

print(sc.vmap(reduce, 5)(np.arange(15.0).reshape(5, 3)))
# [ 3. 12. 21. 30. 39.]
```

Arguments with leading axis `N` bind the callee from their remaining axes. Use `sc.broadcast(p)`
to pass one whole tensor to every iteration. For fixed-shape callees, a flat vector containing
`N` chunks also works, and a tensor containing just one chunk is broadcast. The leading-axis
interpretation takes priority when its per-iteration shape matches the declaration.

For overlapping rank-1 windows, `sc.window(x, start, stride)` reads a slice at
`start + i * stride` on iteration `i`. Its width comes from the callee's declared shape. A shape
hole needs a leading-axis view or an explicitly instantiated callee to determine that width.

```python
@sc.function(sc.arg("pair", 2), outputs=sc.arg("difference", ()))
def difference(pair: sc.Expr) -> sc.Expr:
    return pair[1] - pair[0]

print(sc.vmap(difference, 4)(sc.window(np.arange(5.0), 0, 1)))
# [1. 1. 1. 1.]
```

Symbolic mapped calls build map nodes directly in the caller, so their structure stays visible
to differentiation and loop optimization. Slices, reshapes, and transposes that describe contiguous
windows with a constant nonnegative stride read from their base in place. Other views preserve
their element order through a copy. Numerical marker calls assemble batched arrays; use a declared
outer function when repeated evaluation should perform the slicing in compiled C.

Mappings return the whole output tree. Call `.vec()` on a leaf when a surrounding calculation
needs a flat vector. Each used output currently has its own generated loop. Iterations must be
independent; one iteration's output does not become the next iteration's input.

## Inspecting a graph

Shape-dependent metadata belongs to the concrete instance:

```python
instance = features.instantiate()
print(instance.input_names)    # ('x', 'A')
print(instance.output_shapes)  # ((), (2,))
print(sc.render_expr_assembly(instance))
```

`Function.inputs` and `Function.outputs` hold the declarations, including any holes. Resolved
input and output trees, expression nodes, tensor shapes, and sparsity metadata belong to instances.
Generated code runs on the host CPU.
