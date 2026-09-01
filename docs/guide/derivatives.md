# Derivatives

Derivative wrappers operate on either expressions or typed functions. Expression forms return an
`Expr` or `SparseJacobian`. Function forms return another typed `Function` that keeps the source's
complete input tree.

## Expression and function forms

```python
al.gradient(expr, x)
al.jacobian(expr, x)
al.hessian(expr, x)
al.sparse_jacobian(expr, x)
al.sparse_hessian(expr, x, triangle="upper")

al.gradient(fn, "f", "x")
al.jacobian(fn, "y", "x")
al.hessian(fn, "f", "x")
al.sparse_jacobian(fn, "y", "x")
al.sparse_hessian(fn, "f", "x", triangle="lower")
al.forward(fn, "y", "x")
al.adjoint(fn, "y", "x")
```

Function forms take the declared output name `of` and input name `wrt`. An unknown name fails when
the derivative is built and reports the declared choices. Pass `name=` to set the derived
function's artifact name.
Unseeded derivatives preserve the source's input tree. If `fn` takes `(x, p)`, then
`al.gradient(fn, "f", "x")` also takes `(x, p)`:

```python
grad = al.gradient(fn, "f", "x")
value = grad((x_value, p_value))
```

There is no `extra_inputs` option. Parameters and other source inputs remain available because
dropping them would make the derived expression incomplete and would break the source's typed call
structure.
Sparse Hessians accept `triangle="full"`, `"lower"`, or `"upper"`. The selected triangle preserves
the full pattern's order.

## Seeded modes

Forward and adjoint wrappers pair the source input tree with one seed leaf:

```text
forward inputs = (source_inputs, fwd:<wrt>)
adjoint inputs = (source_inputs, lam:<of>)
```

For example:

```python
fwd = al.forward(fn, "y", "x")
dy = fwd(((x_value, p_value), x_tangent))

adj = al.adjoint(fn, "y", "x")
dx = adj(((x_value, p_value), y_cotangent))
```

The seed shape is the shape of the named input or output.

## Lagrangian Hessians

A Lagrangian wrapper weights every leaf in the source output tree. Its inputs pair the source input
tree with a multiplier tree that has the source output structure:

```python
lag_hess = al.lagrangian_hessian(fn, "x")
sparse_lag_hess = al.sparse_lagrangian_hessian(
    fn,
    "x",
    triangle="lower",
)

dense = lag_hess(((x_value, p_value), (lam_f, lam_g)))
```

The wrappers build an auxiliary scalar named `gamma` from the declared output order. Pass
`aux_name=` to change that internal name. The derived output is `hess_gamma_x_x` or
`sphess_gamma_x_x`. The doubled `wrt` name is part of the generated C symbol and sparsity-table
prefix.
Solver construction uses the same mechanism. Each backend selects its Hessian triangle, while the
`Problem` cache shares the full derivative construction.

## Build several outputs together

`Function.factory` builds one function containing selected source outputs and derivative requests:

```python
combined = fn.factory(
    "fn_all",
    ["x", "p"],
    [
        "f",
        al.factory.Grad("f", "x"),
        al.factory.SpJac("g", "x"),
    ],
)
combined.output_names
# ('f', 'grad_f_x', 'spjac_g_x')
```

A request is a frozen value naming `of` and `wrt`:

| Request | Produces | Derived output name |
| --- | --- | --- |
| `al.factory.Jac(of, wrt)` | dense Jacobian | `jac_<of>_<wrt>` |
| `al.factory.Grad(of, wrt)` | scalar-output gradient | `grad_<of>_<wrt>` |
| `al.factory.Hess(of, wrt)` | scalar-output Hessian | `hess_<of>_<wrt>_<wrt>` |
| `al.factory.SpJac(of, wrt)` | compact Jacobian values and pattern | `spjac_<of>_<wrt>` |
| `al.factory.SpHess(of, wrt, triangle=...)` | compact Hessian values and pattern | `sphess_<of>_<wrt>_<wrt>` |
| `al.factory.Fwd(of, wrt)` | `J(of, wrt) @ fwd:<wrt>` | `fwd_<of>_<wrt>` |
| `al.factory.Adj(of, wrt)` | `J(of, wrt).T @ lam:<of>` | `adj_<of>_<wrt>` |

Include `fwd:<wrt>` or `lam:<of>` in the factory input list for a seeded request.
`al.factory.DerivSpec` is the common request base class.

## Work directly on expressions

Use expression forms inside a graph that does not need function metadata:

```python
seed = al.sym("seed", y.shape)
(vjp_x,) = al.vjp((y,), (x,), (seed,))
tangent = al.jvp(y, x, seed)

seeds = al.sym("seeds", (4, *x.shape))
batched = al.jvp_many(y, x, seeds)

dense_jac = al.jacobian(y, x)
sparse_jac = al.sparse_jacobian(y, x)
```

A sparse expression derivative returns `SparseJacobian` with `values`, `sparsity`, and
`to_dense()`.

## Cost and preserved structure

`gradient` uses one reverse sweep. `jacobian` pushes identity columns through batched forward mode.
`hessian` differentiates a gradient. Unsupported multi-seed rules fall back to one seed at a time
unless `ALLOY_STRICT_JVP_MANY=1` is set.
Derivatives through `CALL` and `VMAP` preserve those structures rather than expanding them. See
[How differentiation works](../how_it_works/autodiff.md).
