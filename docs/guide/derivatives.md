# Derivatives

Dense derivatives in Alloy are another `Function` when you start with a `Function`, and another `Expr`
when you start with an expression. Sparse expression derivatives return a `SparseJacobian` containing
values and sparsity; sparse `Function` derivatives remain `Function`s. All forms use the same graph,
and `Function` results compile through the same path.

## Use the derivative names

The derivative names are overloaded by the type of their first argument.

```
al.gradient(expr, x)                  # Expr gradient
al.jacobian(expr, x)                  # Expr dense Jacobian
al.hessian(expr, x)                   # Expr Hessian
al.sparse_jacobian(expr, x)           # Expr compact Jacobian and pattern
al.sparse_hessian(expr, x)            # Expr compact Hessian and pattern

al.gradient(fn, "f", "x")             # Function gradient of f with respect to x
al.jacobian(fn, "y", "x")             # Function dense Jacobian of y with respect to x
al.hessian(fn, "f", "x")              # Function Hessian of f with respect to x
al.sparse_jacobian(fn, "y", "x")      # Function compact Jacobian and pattern
al.sparse_hessian(fn, "f", "x")       # Function compact Hessian and pattern
al.forward(fn, "y", "x")              # J(y, x) @ fwd:x
al.adjoint(fn, "y", "x")              # J(y, x).T @ lam:y
```

Function forms take `(of, wrt)`. Pass `name=` to choose the derived function name. Pass
`extra_inputs=` to carry parameters through to the result.

Sparse Hessians accept `triangle="full"` (the default), `triangle="lower"`, or
`triangle="upper"`. The selected triangle keeps the full pattern's order.

The seeded forms add one input:

- `fwd:<input>` is the forward seed for `al.forward`.
- `lam:<output>` is the cotangent for `al.adjoint`.

Check input_names before calling a seeded derivative.

## Build several outputs together

`Function.factory` is a shorthand for asking for several named outputs from one function. It is useful
when one artifact needs a value and several derivatives. The graph itself does the merge: expressions
are interned, and lowering walks all outputs in one topological pass.

For example:

```python
(x, p), (f, g) = fn.inputs, fn.outputs
sj = al.sparse_jacobian(g, x)
merged = al.Function("all", [x, p], [f, al.gradient(f, x), sj.values],
                     ["x", "p"], ["f", "grad_f_x", "spjac_g_x"],
                     [None, None, sj.sparsity])
```

Often the first line is unnecessary. `al.sym("x", 3)` returns the interned placeholder that the
function already uses.

Use the factory when naming several derivatives is shorter than building this merged expression
directly:

```python
combined = fn.factory(
    "fn_all",
    ["x", "p"],
    ["f", al.factory.Grad("f", "x"), al.factory.SpJac("g", "x")],
)
combined.output_names  # ('f', 'grad_f_x', 'spjac_g_x')
```

A factory output is either a string naming an existing output or a typed request from al.factory.
The factory is a convenience for this request list. It is not a separate graph merging mechanism.

## The request types

Each request is a frozen dataclass. The `of` field names an output, and the `wrt` field names an input. `SpHess` also accepts a `triangle` layout.

| Request | Produces | Derived output name |
| --- | --- | --- |
| `al.factory.Jac(of, wrt)` | dense Jacobian | `jac_<of>_<wrt>` |
| `al.factory.Grad(of, wrt)` | gradient of a scalar output | `grad_<of>_<wrt>` |
| `al.factory.Hess(of, wrt)` | Hessian of a scalar output | `hess_<of>_<wrt>_<wrt>` |
| `al.factory.SpJac(of, wrt)` | compact nonzero Jacobian values and the pattern | `spjac_<of>_<wrt>` |
| `al.factory.SpHess(of, wrt, triangle="full")` | compact nonzero Hessian values and the selected pattern | `sphess_<of>_<wrt>_<wrt>` |
| `al.factory.Fwd(of, wrt)` | `J(of, wrt) @ fwd:<wrt>` | `fwd_<of>_<wrt>` |
| `al.factory.Adj(of, wrt)` | `J(of, wrt).T @ lam:<of>` | `adj_<of>_<wrt>` |

Hess and SpHess always differentiate with respect to the same input twice. A mixed partial is a
Jacobian of a gradient output. The doubled input name in the derived output stays in place because
it is part of the generated C symbol and the sparsity-table prefix. `SpHess` returns the selected
triangle in the full symmetric pattern's order; `Hess` remains dense.

The base type is `al.factory.DerivSpec`. You usually work with one of the concrete request types
instead.

## Seeds and duals

The factory creates the symbolic seed and dual expressions. Include their names in the factory input
list:

```python
adj = fn.factory("fn_adj", ["x", "p", "lam:g"], [al.factory.Adj("g", "x")])
adj.input_names  # ('x', 'p', 'lam:g')
```

## Lagrangians

A Lagrangian is a weighted combination of outputs. The convenience wrappers take the output names
first and the differentiated input second:

```python
al.lagrangian_hessian(fn, ["f", "g"], "x")
al.sparse_lagrangian_hessian(fn, ["f", "g"], "x", triangle="lower")
# inputs: ('x', 'lam:f', 'lam:g')
# output: ('sphess_gamma_x_x',)
```

Use `triangle="upper"` when the consumer expects the upper triangle. The output name and
`coloring_width` do not depend on the selected layout.

The wrappers build an auxiliary output named `gamma`:

```python
sum(lam:<name> * <name> for name in of)
```

The output order fixes which dual multiplies each constraint block. al.nlp uses the same path.

## Work directly on expressions

Use the expression forms when you are building a graph that does not need named-function metadata:

```python
seed = al.sym("seed", y.shape)
(vjp_x,) = al.vjp((y,), (x,), (seed,))
tangent = al.jvp(y, x, seed)

seeds = al.sym("seeds", (4, *x.shape))
batched = al.jvp_many(y, x, seeds)

al.jacobian(y, x)
al.gradient(y, x)
al.hessian(y, x)
```

Sparse expression derivatives return a `SparseJacobian` with `.values` and `.sparsity`.

## Cost and structure

`gradient` uses one reverse sweep. `jacobian` pushes all identity columns through batched forward
mode. `hessian` computes a gradient and then its Jacobian.

Some operations have no multi-seed forward rule yet. Alloy falls back to one seed at a time for
those graphs. Set ALLOY_STRICT_JVP_MANY=1 to raise instead.

Derivatives through call and vmap preserve those structures rather than expanding them. See
[how differentiation works](../how_it_works/autodiff.md).
