# Derivatives

Derivative wrappers operate on either expressions or typed functions. Expression forms return an
`Expr` or `SparseJacobian`. Function forms return another typed `Function` that keeps every
parameter of the source.

## Expression and function forms

```python
sc.gradient(expr, x)
sc.jacobian(expr, x)
sc.hessian(expr, x)
sc.sparse_jacobian(expr, x)
sc.sparse_hessian(expr, x, triangle="upper")

sc.gradient(fn, "f", "x")
sc.jacobian(fn, "y", "x")
sc.hessian(fn, "f", "x")
sc.sparse_jacobian(fn, "y", "x")
sc.sparse_hessian(fn, "f", "x", triangle="lower")
sc.forward(fn, "y", "x")
sc.adjoint(fn, "y", "x")
```

Function forms take the output name `of` and the input name `wrt`. One name is `wrt`, as in the
expression form: `sc.gradient(fn, "x")` differentiates the only output. A name left out is the
function's only output or only input, so `sc.gradient(fn)` needs no names at all for a function of
one input and one output. Names can also be passed as `of=` and `wrt=`. An unknown name fails when
the derivative is built and reports the declared choices. Pass `name=` to set the derived
function's artifact name.

Unseeded derivatives take the same arguments as the source. For a source `fn(x, p)`:

```python
grad = sc.gradient(fn, "f", "x")
value = grad(x_value, p_value)
```

The derivative of a template (a function whose shapes are bound at each call) is a template too.
Nothing is built until it is called; each instance is the derivative of the source's instance at
the same shapes, named after it (`fn__3_grad_f_x`), or `{name}__3` with `name=` given. Names the
template declares are checked when the derivative is built; a check that needs shapes, such as a
gradient needing a scalar output, happens at the call.

There is no `extra_inputs` option. Parameters and other source inputs remain available because
dropping them would leave the derived expression incomplete and break the source's typed call
structure.

Sparse Hessians accept `triangle="full"`, `"lower"` or `"upper"`. The selected triangle keeps the
full pattern's order.

## Seeded modes

Forward and adjoint wrappers take the source's arguments followed by one seed leaf:

```text
forward arguments = *source_arguments, fwd:<wrt>
adjoint arguments = *source_arguments, lam:<of>
```

For the `fn(x, p)` above:

```python
fwd = sc.forward(fn, "y", "x")
dy = fwd(x_value, p_value, x_tangent)

adj = sc.adjoint(fn, "y", "x")
dx = adj(x_value, p_value, y_cotangent)
```

The seed shape is the shape of the named input or output.

## Lagrangian Hessians

A Lagrangian wrapper weights every leaf in the source output tree. It takes the source's arguments
followed by one multiplier argument that has the source output structure:

```python
lag_hess = sc.lagrangian_hessian(fn, "x")
sparse_lag_hess = sc.sparse_lagrangian_hessian(
    fn,
    "x",
    triangle="lower",
)

dense = lag_hess(x_value, p_value, (lam_f, lam_g))
```

The wrappers build an auxiliary scalar named `gamma` from the declared output order; pass
`aux_name=` to change it. The derived output is `hess_gamma_x_x` or `sphess_gamma_x_x`. The doubled
`wrt` name is part of the generated C symbol and sparsity-table prefix.

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
        sc.factory.Grad("f", "x"),
        sc.factory.SpJac("g", "x"),
    ],
)
combined.output_names
# ('f', 'grad_f_x', 'spjac_g_x')
```

A request is a frozen value naming `of` and `wrt`:

| Request | Produces | Derived output name |
| --- | --- | --- |
| `sc.factory.Jac(of, wrt)` | dense Jacobian | `jac_<of>_<wrt>` |
| `sc.factory.Grad(of, wrt)` | scalar-output gradient | `grad_<of>_<wrt>` |
| `sc.factory.Hess(of, wrt)` | scalar-output Hessian | `hess_<of>_<wrt>_<wrt>` |
| `sc.factory.SpJac(of, wrt)` | compact Jacobian values and pattern | `spjac_<of>_<wrt>` |
| `sc.factory.SpHess(of, wrt, triangle=...)` | compact Hessian values and pattern | `sphess_<of>_<wrt>_<wrt>` |
| `sc.factory.Fwd(of, wrt)` | `J(of, wrt) @ fwd:<wrt>` | `fwd_<of>_<wrt>` |
| `sc.factory.Adj(of, wrt)` | `J(of, wrt).T @ lam:<of>` | `adj_<of>_<wrt>` |

Include `fwd:<wrt>` or `lam:<of>` in the factory input list for a seeded request.
`sc.factory.DerivSpec` is the common request base class.

## Work directly on expressions

Use expression forms inside a graph that does not need function metadata. `sc.jvp` takes a seed
shaped like `wrt`; `sc.vjp` takes a cotangent shaped like the output:

```python
seed = sc.sym("seed", x.shape)
tangent = sc.jvp(y, x, seed)

cot = sc.sym("cot", y.shape)
(vjp_x,) = sc.vjp((y,), (x,), (cot,))

seeds = sc.sym("seeds", (4, *x.shape))
batched = sc.jvp_many(y, x, seeds)

dense_jac = sc.jacobian(y, x)
sparse_jac = sc.sparse_jacobian(y, x)
```

A sparse expression derivative returns `SparseJacobian` with `values`, `sparsity` and
`to_dense()`.

## Cost and preserved structure

`gradient` uses one reverse sweep. `jacobian` pushes identity columns through batched forward mode.
`hessian` differentiates a gradient. Operations without a multi-seed rule fall back to one seed at a
time unless `SCALY_STRICT_JVP_MANY=1` is set.

Batched forward mode through `scan` and `while_loop` is one loop whose carry holds the primal and
every seed's tangent, so the Jacobian or Hessian of a rollout is a fixed number of loops for any
horizon. For a Hessian, that is one tangent scan forwards and one adjoint scan backwards (reverse
mode differentiates every used output of a scan in one backward scan), each carrying all the
seeds. When a step has fewer inputs than there are seeds, the step's own small Jacobian is formed
once per step and the seeds are multiplied by it, so the work per step is a matrix product rather
than the step's operations repeated per seed; it is still proportional to the number of seeds times
the number of steps.

Derivatives through `CALL` and `VMAP` keep those nodes instead of expanding them. See
[How differentiation works](../how_it_works/autodiff.md).

## Custom derivatives

`sc.custom_derivative(fn, jvp=..., vjp=...)` returns a copy of `fn` whose derivatives come from the
given Functions instead of from its body. The forward rule takes `(*inputs, *input_tangents)` and
returns one tangent per output. The reverse rule takes `(*inputs, *outputs, *output_cotangents)` and
returns one cotangent per input. A missing direction differentiates the body as usual.

The typical use is a solver. Differentiating a `while_loop` differentiates the steps it took, which
only approximates the derivative of the solution and costs a backward pass over every step. The
implicit-function rule at the solution is exact and costs one linear solve:

```python
# x solves x**3 + x = p, so dx/dp = 1 / (3 x^2 + 1)
p, x, xbar = sc.sym("p", n), sc.sym("x", n), sc.sym("xbar", n)
rule = sc.Function.from_exprs("implicit", [p, x, xbar], [xbar / (3 * x * x + 1)], ["p", "x", "xbar"], ["pbar"])
solve_with_rule = sc.custom_derivative(solve, vjp=rule)
```

The reverse rule receives the call's outputs, so it reuses the solution instead of solving again.
The rules are honored through calls, `vmap` and every derivative built on them.

Sparsity patterns come from the body unless `sparsity=` gives them. It is a function of `(output
index, input index)` that returns the pattern of that output in that input: a `SparsityType`, a
boolean mask, a SciPy sparse matrix, or `None` for no dependence. The declared pattern is used
wherever the Function is applied: a call, a `vmap`, or as a `scan` or `while_loop` body. A solver's
pattern through its iterations is conservative (often dense) and can be slow to work out, while the
rule's author usually knows the real one:

```python
diagonal = sc.custom_derivative(solve, vjp=rule, sparsity=lambda out, k: np.eye(n, dtype=bool))
```
