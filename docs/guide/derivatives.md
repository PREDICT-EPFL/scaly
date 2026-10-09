# Derivatives

Scaly differentiates the recorded expression graph. A derivative of a `Function`
is another `Function`, which you can evaluate, compose, and export as C. Named
derivative requests select which output to differentiate and with respect to
which input, while retaining the other inputs as parameters.

The examples assume basic multivariable calculus and familiarity with
[building a Function](functions.md). No experience with automatic differentiation
is needed.

## Gradients and Hessians

For a scalar cost, the gradient gives its rate of change with each input
component. The Hessian is the matrix of second derivatives and describes local
curvature. For a squared tracking cost,

\[
f(x,t)=\lVert x-t\rVert^2, \qquad
\nabla_x f(x,t)=2(x-t), \qquad
\nabla_x^2 f(x,t)=2I.
\]

The declarations below name the output `cost` and the inputs `x` and `target`:

```python
import numpy as np
import scaly as sc

@sc.function(sc.arg("x", 2), sc.arg("target", 2), outputs=sc.arg("cost"))
def tracking_cost(x: sc.Expr, target: sc.Expr) -> sc.Expr:
    return sc.sumsqr(x - target)

grad = sc.gradient(tracking_cost, "cost", "x")
hess = sc.hessian(tracking_cost, "cost", "x")
data = (np.array([3.0, 5.0]), np.array([1.0, 2.0]))
print(grad(*data))  # [4. 6.]
print(hess(*data))  # [[2. 0.]
                   #  [0. 2.]]
```

The `of` and `wrt` arguments, here `"cost"` and `"x"`, select the declared output
and input. Either can be left out when the function has only one output or only
one input. The gradient
is taken with respect to `x`, holding `target` fixed. Both derivative functions
still take `(x, target)`, because their calculations may need both values.
A derivative of a function with [open input shapes](functions.md#leaving-shapes-out)
is itself a template, bound at the shapes of each call.

The gradient has the input's shape. The Hessian has shape `(x.size, x.size)`.
Gradient and Hessian requests require a scalar output. Use a Jacobian for a
vector output.

The standalone [quadratic example](https://github.com/PREDICT-EPFL/scaly/blob/main/examples/quadratic.py)
evaluates this cost and its derivatives, then exports their generated C.

## Jacobians and flattened dimensions

A Jacobian contains all first derivatives of a vector-valued function. Entry
`J[i, j]` is the derivative of output component `i` with respect to input
component `j`. For the following measurement model,

\[
y(x)=\begin{bmatrix}x_0x_1 \\ x_0+2x_1\end{bmatrix}, \qquad
J(x)=\frac{\partial y}{\partial x}
=\begin{bmatrix}x_1 & x_0 \\ 1 & 2\end{bmatrix}.
\]

```python
@sc.function(sc.arg("x", 2), outputs=sc.arg("y"))
def measurements(x: sc.Expr) -> sc.Expr:
    return sc.stack([x[0] * x[1], x[0] + 2.0 * x[1]])

jac = sc.jacobian(measurements, "y", "x")
x_value = np.array([3.0, 4.0])
print(jac(x_value))  # [[4. 3.]
                     #  [1. 2.]]
```

The Jacobian shape is `(output.size, input.size)`. Matrix-valued inputs and
outputs are flattened in row-major order for these two axes.

The fixed row-major order matters when comparing with a NumPy calculation.
For a matrix input `X`, column `j` of the Jacobian corresponds to
`X.reshape(-1)[j]`, not to a column of `X`.

For large problems with many known zeros, use `sc.sparse_jacobian` or
`sc.sparse_hessian`. These return only the potentially nonzero entries. See
[Sparsity](sparsity.md) for retrieving their locations and reconstructing a
matrix. Sparse Hessians accept `triangle="full"`, `"lower"`, or `"upper"`.

## Products with a Jacobian

If you only need `J @ direction`, use `sc.forward`. The direction, also called a
seed, has the input's shape. For the `measurements` function above,

\[
y(x+\varepsilon d)=y(x)+\varepsilon J(x)d+O(\varepsilon^2).
\]

`sc.forward` constructs the function for \(J(x)d\):

```python
fwd = sc.forward(measurements, "y", "x")
print(fwd(x_value, np.array([1.0, 0.0])))  # [4. 1.]
```

`sc.adjoint` constructs \(J(x)^T w\), the gradient of the scalar weighted
output \(w^T y(x)\). The weights have the output's shape:

```python
adj = sc.adjoint(measurements, "y", "x")
print(adj(x_value, np.array([1.0, 2.0])))  # [6. 7.]
```

Both functions append the seed as one parameter after the original parameters.
For `f(x, target)`, call `fwd(x, target, seed)`. For a single grouped
parameter `f((x, target))`, call `fwd((x, target), seed)`.

## Lagrangian Hessians and multiplier structure

Constrained optimization uses derivatives of both the cost and constraints.
For a scalar cost `f(x)` and constraints `g(x)`, the Lagrangian is the weighted
sum

\[
\mathcal{L}(x,\sigma,\lambda)=\sigma f(x)+\lambda^T g(x), \qquad
\nabla_x^2\mathcal{L}=\sigma\nabla_x^2 f+\sum_i\lambda_i\nabla_x^2 g_i.
\]

Scaly's solver interfaces construct this derivative automatically. A direct
request uses every output of a function as one term in that weighted sum:

```python
@sc.function(sc.arg("x", 2), outputs=sc.group(sc.arg("cost"), sc.arg("constraint")))
def model(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    return sc.sumsqr(x), sc.stack([x[0] * x[1]])

lag_hess = sc.lagrangian_hessian(model, "x")
weights = (np.array(1.0), np.array([3.0]))
point = np.array([3.0, 4.0])
print(lag_hess(point, weights))  # [[2. 3.]
                                    #  [3. 2.]]
```

The multiplier structure matches the complete output structure. Every output
participates in the weighted sum. `sc.sparse_lagrangian_hessian` returns compact
values and accepts the same triangle choices as `sc.sparse_hessian`.

## Values and derivatives in one function

For the `tracking_cost` function above, `Function.factory` can combine the value
and several derivatives into one call:

```python
combined = tracking_cost.factory(
    "tracking_all",
    ["x", "target"],
    ["cost", sc.factory.Grad("cost", "x"), sc.factory.Hess("cost", "x")],
)
cost, gradient, hessian = combined(*data)
print(combined.instantiate().output_names)
# ('cost', 'grad_cost_x', 'hess_cost_x_x')
```

The input list selects the declared leaves. Each leaf becomes one parameter
of the factory function, even if the source groups them. The output list mixes
output names with derivative requests. The [function API reference](../api/functions.md)
lists the request types, including sparse and seeded derivatives.

A factory request for a forward derivative also needs `fwd:<wrt>` in the input
list. An adjoint request needs `lam:<of>`. The `sc.forward` and `sc.adjoint`
wrappers construct those inputs for a single derivative request.

Factory requests also work on shape templates. The transform runs once for each
binding, so values and derivatives keep the shape selected by the call:

```python
@sc.function(sc.arg("x"), outputs=sc.arg("cost", ()))
def energy(x: sc.Expr) -> sc.Expr:
    return sc.sumsqr(x)

combined = energy.factory("energy_all", ["x"], ["cost", sc.factory.Grad("cost", "x")])
print(combined(np.array([2.0, 3.0])))       # (array(13.), array([4., 6.]))
print(combined(np.array([2.0, 3.0, 4.0])))  # (array(29.), array([4., 6., 8.]))
```

An unbound factory source needs named input declarations. Include every input
with an open shape in the selected input list. Fixed inputs may be omitted if
the requested outputs do not depend on them. Seed and weight shapes are checked
against the bound source.

## Derivatives of expressions

You can also differentiate before wrapping expressions in a function:

```python
x = sc.sym("x", 2)
y = sc.stack([x[0] * x[1], x[0] + 2.0 * x[1]])
J = sc.jacobian(y, x)

seed = sc.sym("seed", 2)
directional = sc.jvp(y, x, seed)
weights = sc.sym("weights", 2)
(transposed,) = sc.vjp((y,), (x,), (weights,))
```

`jvp` means Jacobian-vector product and `vjp` means vector-Jacobian product.
`sc.jvp_many(y, x, seeds)` handles several directions, with the seed number as
the first axis. Expression forms of sparse derivatives return a
`SparseJacobian` containing `.values`, `.sparsity`, and `.to_dense()`.

## Derivatives with respect to intermediate expressions

Expression derivatives can select an intermediate value, such as a slice of a
state vector or the result of a function call. Scaly treats the selected
expression as an independent variable. Calculations that use its inputs
without using the selected expression stay fixed. For example, a calculation
can differentiate only the state portion of a vector that also contains a
parameter:

```python
@sc.function(sc.arg("carry", 3), outputs=sc.arg("state_grad"))
def state_derivative(carry: sc.Expr) -> sc.Expr:
    state = carry[:2]
    cost = sc.sumsqr(state - carry[2]) + sc.sumsqr(carry[1:])
    return sc.gradient(cost, state)

print(state_derivative(np.array([1.0, 3.0, 2.0])))  # [-2.  2.]
```

The second term uses `carry[1:]`, which is a different expression from `state`,
so it contributes nothing to this derivative even though the slices overlap.
Writing `carry[:2]` again gives the same expression as `state`, so a term using
that slice would contribute too. Jacobians, Hessians, sparse derivatives and
Jacobian products follow the same rule.

`sc.vjp` can select several expressions at once. It treats each selection as
independent, including when one is a slice of another. In the following example,
the cubic term contributes only to the derivative with respect to `tail`:

```python
@sc.function(
    sc.arg("carry", 3),
    outputs=sc.group(sc.arg("segment_grad"), sc.arg("tail_grad")),
)
def joint_derivatives(carry: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    segment = carry[1:]
    tail = segment[1:]
    cost = sc.sumsqr(segment) + (tail ** 3).sum()
    segment_grad, tail_grad = sc.vjp((cost,), (segment, tail), (sc.const(1.0),))
    return segment_grad, tail_grad

print(joint_derivatives(np.array([1.0, 2.0, 3.0])))
# (array([4., 6.]), array([27.]))
```

The results follow the order of `wrts`, here `(segment, tail)`.

!!! warning "Mapped slices must remain in the graph"
    A map can replace a slice and reshape with direct reads from the original
    vector. Differentiating with respect to the removed slice then returns
    zeros without raising an error. An explicit `sc.window` retains the selected
    slice, as the two forms below show.

```python
@sc.function(sc.arg("pair", 2), outputs=sc.arg("cost"))
def pair_cost(pair: sc.Expr) -> sc.Expr:
    return sc.sumsqr(pair)

@sc.function(
    sc.arg("carry", 5), outputs=sc.group(sc.arg("removed"), sc.arg("retained")),
)
def mapped_gradients(carry: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    state = carry[:4]
    removed = sc.vmap(pair_cost, 2)(state.reshape((2, 2))).sum()
    retained = sc.vmap(pair_cost, 2)(sc.window(state, 0, 2)).sum()
    return sc.gradient(removed, state), sc.gradient(retained, state)

print(mapped_gradients(np.arange(1.0, 6.0)))
# (array([0., 0., 0., 0.]), array([2., 4., 6., 8.]))
```

See [mapped windows](functions.md#regular-repetition-vmap) for the start and
stride arguments.

## Holding a value fixed in derivatives

`sc.stop_gradient` returns its argument's value and treats it as a constant in
all derivatives. Other uses of the same expression still contribute normally:

```python
@sc.function(sc.arg("x", 2), outputs=sc.arg("cost"))
def frozen_product(x: sc.Expr) -> sc.Expr:
    return (x * sc.stop_gradient(x)).sum()

point = np.array([2.0, 3.0])
print(frozen_product(point))                 # 13.0
print(sc.gradient(frozen_product)(point))    # [2. 3.]
print(sc.hessian(frozen_product)(point))     # [[0. 0.]
                                            #  [0. 0.]]
```

Only the first factor contributes to the gradient. Differentiating again keeps
the second factor fixed, so the Hessian is zero. This convention applies in
forward mode, reverse mode and sparsity analysis, including inside calls and
maps. It changes the derivative without changing the evaluated value or its
generated C. Finite-difference checks perturb the input and recompute the
stopped value, so they need not agree with this derivative.

Quadratic-program extraction rejects problems containing `stop_gradient`,
including in their bounds or inside called functions. The proof that a problem
is quadratic requires derivatives of the evaluated cost and constraints.

## Custom derivative rules

`sc.custom_derivative` gives a function rules that replace differentiating its
body. It suits a calculation whose derivative has a cheaper or more accurate
form than its steps, such as an iterative solve. The function below solves
\(x^3 + x = p\) entry by entry with 30 Newton steps. Differentiating
\(x^3 + x - p = 0\) at the solution gives

\[
\frac{\partial x}{\partial p} = \frac{1}{3x^2 + 1},
\]

one division per entry, where differentiating the body would differentiate
every Newton step:

```python
@sc.function(sc.arg("p"))
def cubic_root(p: sc.Expr) -> sc.Expr:
    x = p
    for _ in range(30):
        x = x - (x**3 + x - p) / (3.0 * x * x + 1.0)
    return x

@sc.function()
def cubic_root_jvp(p: sc.Expr, p_dot: sc.Expr) -> sc.Expr:
    x = cubic_root(p)
    return p_dot / (3.0 * x * x + 1.0)

root = sc.custom_derivative(cubic_root, jvp=cubic_root_jvp)
p = np.array([2.0, 10.0, 30.0])
print(root(p))                # [1. 2. 3.]
print(sc.jacobian(root)(p))   # [[0.25       0.         0.        ]
                              #  [0.         0.07692308 0.        ]
                              #  [0.         0.         0.03571429]]
```

The forward rule `jvp` takes the function's parameters and then one tangent per
parameter, here `p_dot` for `p`, and returns the output tangents. Every forward
derivative of `root` uses it, whether `root` is called directly, inside another
function or in a `vmap`. Several directions at once, such as the three columns
of this Jacobian, map the single-direction rule over the directions. A tangent
the rule never reads is never computed. Because `cubic_root` leaves its input
shape open, `root` and its rule are templates, bound at the shape of each call.

### Reverse rules and residuals

The reverse rule is a pair of functions in the form of JAX's `custom_vjp`. `fwd`
takes the parameters and returns `(outputs, residuals)`. `bwd` takes the
residuals and the output cotangents and returns one cotangent per parameter.
The residuals hold what `bwd` needs, often the solution or a factorization.
Here the residual is the solution itself:

```python
@sc.function()
def cubic_root_fwd(p: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    x = cubic_root(p)
    return x, x

@sc.function()
def cubic_root_bwd(x: sc.Expr, x_bar: sc.Expr) -> sc.Expr:
    return x_bar / (3.0 * x * x + 1.0)

root = sc.custom_derivative(cubic_root, jvp=cubic_root_jvp, fwd=cubic_root_fwd, bwd=cubic_root_bwd)

@sc.function(sc.arg("p", 3), outputs=sc.arg("cost"))
def cost(p: sc.Expr) -> sc.Expr:
    return sc.sumsqr(root(p))

print(sc.gradient(cost)(p))  # [0.5        0.30769231 0.21428571]
print(sc.hessian(cost)(p))   # [[-0.0625      0.          0.        ]
                             #  [ 0.         -0.01001365  0.        ]
                             #  [ 0.          0.         -0.0023688 ]]
```

A call of `root` computes its outputs and residuals together, so the gradient
reuses the solution or the factorization of that call instead of computing it
again. The outputs of `fwd` replace those of the body and should equal them.
The residuals keep their dependence on the inputs, so a second derivative, such
as the Hessian above, differentiates `bwd` and the residuals it reads.

`fwd` and `bwd` are given together. A direction without a rule differentiates
the body: with only `jvp`, a gradient goes through the Newton steps. A forward
rule receives no residuals, which is why `cubic_root_jvp` calls `cubic_root`
again to recover the solution.

### Declared sparsity

Sparsity analysis cannot see the derivative a rule computes, so the Jacobian
pattern of a function with rules is dense unless it is declared. The callable
`sparsity(of, wrt, shape)` returns the pattern of output `of` in input `wrt`,
as a boolean mask of `shape` or a `SparsityPattern`:

```python
diagonal = sc.custom_derivative(
    cubic_root, jvp=cubic_root_jvp, sparsity=lambda of, wrt, shape: np.eye(*shape, dtype=bool)
)
print(sc.sparse_jacobian(diagonal)(p))     # [0.25       0.07692308 0.03571429]
print(sc.sparse_jacobian(root)(p).shape)   # (9,)
```

!!! warning "A declared pattern is taken as given"
    Scaly does not compare the declared pattern with the rule. A pattern that
    leaves out an entry the rule computes gives wrong sparse derivatives
    without an error.

## Current limitations

Derivatives work through ordinary function calls and `vmap`. Ordinary calls may
expand during differentiation or compilation. Mapped repetition remains
represented as a loop. An active derivative through an optimization solve
raises `NotImplementedError`, including when the solve is inside an ordinary
or mapped function. A solve independent of the differentiated input does not
block its derivative. See [solver sensitivities](solvers.md#nesting-a-solver-in-a-graph)
for the limits of dependency analysis.

Differentiation through `minimum`, `maximum`, `floor`, and `ceil`
raises `NotImplementedError`, even at points where the mathematical derivative
exists:

```python
x_clip = sc.sym("x_clip", ())
clipped = x_clip.maximum(0.0)
try:
    sc.gradient(clipped, x_clip)
except NotImplementedError as error:
    print(type(error).__name__)
# NotImplementedError
```

Clipping a control with minimum and maximum operations therefore prevents
differentiation of that expression. For an
optimization problem, express control limits as variable bounds instead.

See [How differentiation works](../how_it_works/autodiff.md) for the algorithms.
