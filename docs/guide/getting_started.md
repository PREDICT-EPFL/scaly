# Getting started

This tutorial builds a model, differentiates it, turns it into an optimization problem, and calls a
generated solver. Every numerical result comes from compiled C.

## Build expressions

`al.sym` creates a named symbolic input and `al.const` creates a constant:

```python
import alloy as al
import numpy as np

z = al.sym("z", 2)
u = al.sym("u", 1)
znext = z + 0.1 * al.concat([z[1:], u])
```

Operations on an `Expr` build graph nodes. They do not evaluate Python values. Shapes and dtypes
are static, and arithmetic follows NumPy broadcasting.

```python
print(znext.shape)       # (2,)
print(al.format_expr(znext))
```

See [the expression dialect](../how_it_works/expr_ir.md#operations) for the full operation set.

## Declare a typed function

A `Function` gives a graph a named boundary. Its input and output trees describe both symbolic and
numerical calls.

```python
@al.function(
    al.G(al.L("z", 2), al.L("u", 1)),
    al.L("znext", ...),
)
def step(inputs: tuple[al.Expr, al.Expr]) -> al.Expr:
    z, u = inputs
    return z + 0.1 * al.concat([z[1:], u])
```

`al.L` declares one tensor. `al.G` groups trees. The body takes one value with the input structure
and returns one value with the output structure. `...` asks Alloy to infer the output shape during
tracing.
Run it numerically:

```python
z1 = step.numerical_call(
    (np.array([1.0, 2.0]), np.array([0.5]))
)
# array([1.2, 2.05])
```

The first call lowers the graph, renders C, compiles a shared library, and stores it in the
just-in-time cache. Later calls reuse the artifact.
Compose it symbolically:

```python
N = 20

@al.function(
    al.G(al.L("z0", 2), al.L("us", N)),
    al.G(al.L("zN", ...), al.L("cost", ...)),
)
def rollout(inputs: tuple[al.Expr, al.Expr]) -> tuple[al.Expr, al.Expr]:
    z, us = inputs
    cost = al.const(0.0)
    for k in range(N):
        u = us[k : k + 1]
        cost = cost + al.sumsqr(z) + 0.1 * al.sumsqr(u)
        z = step.symbolic_call((z, u))
    return z, cost + 10.0 * al.sumsqr(z)
```

`symbolic_call` adds a first-class `CALL` node. The generated C contains one `step` procedure and
calls it from `rollout`.
See [Building functions](functions.md) for nested trees, inferred outputs, lower-level flat calls,
and `vmap`.

## Differentiate the function

Named derivative wrappers return typed functions:

```python
grad = al.gradient(rollout, "cost", "us")

gradient_value = grad.numerical_call(
    (np.array([1.0, 0.0]), np.zeros(N))
)
```

The derivative keeps `rollout`'s complete input tree, `(z0, us)`. There is no separate parameter
list to maintain.
The common wrappers are:

```python
al.gradient(fn, "f", "x")
al.jacobian(fn, "y", "x")
al.hessian(fn, "f", "x")
al.sparse_jacobian(fn, "y", "x")
al.sparse_hessian(fn, "f", "x")
al.forward(fn, "y", "x")
al.adjoint(fn, "y", "x")
```

Differentiation is graph-to-graph. The result compiles, nests, and renders like any other
`Function`. See [Derivatives](derivatives.md) and [Sparsity](sparsity.md).

## Declare an optimization problem

A `Problem` separates the mathematical model from the solver backend:

```python
@al.problem(
    vars=al.L("us", N),
    params=al.L("z0", 2),
)
def shooting_problem(us: al.Expr, z0: al.Expr) -> al.ProblemSpec[al.Expr]:
    zN, cost = rollout.symbolic_call((z0, us))
    return al.ProblemSpec(
        minimize=cost,
        eq=(zN,),
        lb=al.const(np.full(N, -2.0)),
        ub=al.const(np.full(N, 2.0)),
    )
```

The objective is scalar. Equality groups are constrained to zero. Use `al.bounded` for one- or
two-sided inequality groups. Variable bounds have the declared variable structure.
Choose a backend:

```python
solve = al.solver(
    shooting_problem,
    "ipopt",
    options={"print_level": 0},
)
```

IPOPT, PIQP, and alloy-sqp are discovered as plugins. PIQP is accepted only when Alloy can prove
the cost quadratic, the constraints affine, and the bounds independent of the variables.

## Call the solver

All solver functions have the same five input groups and four output groups:

```text
inputs  = (vars_init, lam_box0, lam_eq0, lam_ineq0, params)
outputs = (vars,      lam_box,  lam_eq,  lam_ineq)
```

For this problem, the variable and parameter trees each have one leaf:

```python
us_opt, lam_box, lam_eq, lam_ineq = solve.numerical_call(
    (
        np.zeros(N),
        np.zeros(N),
        np.zeros(2),
        np.zeros(0),
        np.array([1.0, 0.0]),
    )
)

stats = solve.solver_stats()
print(stats.obj, stats.iter, stats.to_solver_status())
```

Multiplier categories remain in the signature when absent, using length-zero arrays. Box and
inequality multipliers are signed: positive means the upper side is active and negative means the
lower side is active.
A solver is a plain `Function`. Use `symbolic_call` to put it inside another graph; the host,
oracles, and native wrapper then compile into one shared library.
See [Solvers](solvers.md) for multi-block variables, bounded groups, `qp_problem`, sparse PIQP
data, nesting, and statistics. See [Solver backends](solver_backends.md) for backend-specific
options and warm-start behavior.

## Preserve regular repetition

The Python loop in `rollout` creates `N` call sites. When iterations are independent, `al.vmap`
represents the repetition as one node and lowers it to a C loop:

```python
@al.function(
    al.G(al.L("z", 2), al.L("u", 1), al.L("znext", 2)),
    al.L("defect", ...),
)
def defect(inputs: tuple[al.Expr, al.Expr, al.Expr]) -> al.Expr:
    z, u, znext = inputs
    return step.symbolic_call((z, u)) - znext

decision = al.sym("decision", 2 * (N + 1) + N)
states = decision[: 2 * (N + 1)]
controls = decision[2 * (N + 1) :]

defects = al.vmap(
    defect,
    N,
    [
        (states, 0, 2),
        (controls, 0, 1),
        (states, 2, 2),
    ],
)
```

Each mapping tuple is `(outer, start, stride)`. Iteration `i` reads a slice beginning at
`start + i * stride`. Derivatives preserve this mapping, so source size and derivative construction scale
with the local stage rather than an unrolled copy of every stage.

## Render C ahead of time

The ahead-of-time command uses the same lowering and renderer as the numerical call:

```bash
uv run -m alloy.codegen mymodule:solve -o generated/
```

It writes one C source file and one header exposing the universal pointer-array ABI and typed C++
helpers. Solver-bearing modules include their backend link flags. See [Code generation](codegen.md)
and [the C ABI](../how_it_works/c_abi.md).

## Next steps

- [Building functions](functions.md)
- [Derivatives](derivatives.md)
- [Sparsity](sparsity.md)
- [Solvers](solvers.md)
- [Code generation](codegen.md)
- [Architecture](../how_it_works/architecture.md)
