# Getting started

This tutorial builds a model, differentiates it, turns it into an optimization problem and calls a
generated solver. Every numerical result comes from compiled C.

## Build expressions

`sc.sym` creates a named symbolic input and `sc.const` creates a constant.

```python
import scaly as sc
import numpy as np

z = sc.sym("z", 2)
u = sc.sym("u", 1)
znext = z + 0.1 * sc.concat([z[1:], u])
```

Operations on an `Expr` build graph nodes; they do not evaluate anything. Shapes and dtypes are
static and arithmetic follows NumPy broadcasting.

```python
print(znext.shape)       # (2,)
print(sc.format_expr(znext))
```

See [the expression dialect](../how_it_works/ir.md#operations) for the full operation set.

## Declare a typed function

A `Function` gives a graph a named boundary. Its input and output trees describe both symbolic and
numerical calls.

```python
@sc.function(
    sc.G(sc.L("z", 2), sc.L("u", 1)),
    sc.L("znext", ...),
)
def step(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    z, u = inputs
    return z + 0.1 * sc.concat([z[1:], u])
```

`sc.L` declares one tensor and `sc.G` groups trees. The body takes one value with the input
structure and returns one value with the output structure. `...` infers the output shape during
tracing.

Call it with arrays to run it:

```python
z1 = step((np.array([1.0, 2.0]), np.array([0.5])))
# array([1.2, 2.05])
```

The input tree is an `sc.G` of two leaves, so the call takes a 2-tuple. The output tree is a single
`sc.L`, so the result is the array itself, see
[a single leaf is unpacked](functions.md#a-single-leaf-is-unpacked). The first call lowers the
graph, renders C, compiles a shared library and stores it in the just-in-time (JIT) cache. Later
calls reuse it.

Call it with `Expr` leaves to compose it symbolically:

```python
N = 20

@sc.function(
    sc.G(sc.L("z0", 2), sc.L("us", N)),
    sc.G(sc.L("zN", ...), sc.L("cost", ...)),
)
def rollout(inputs: tuple[sc.Expr, sc.Expr]) -> tuple[sc.Expr, sc.Expr]:
    z, us = inputs
    cost = sc.const(0.0)
    for k in range(N):
        u = us[k : k + 1]
        cost = cost + sc.sumsqr(z) + 0.1 * sc.sumsqr(u)
        z = step((z, u))
    return z, cost + 10.0 * sc.sumsqr(z)
```

`step(...)` dispatches on the leaves it is given: `numerical_call` for arrays, `symbolic_call` for
expressions. The symbolic call adds a `CALL` node, so the generated C contains one `step`
procedure and calls it from `rollout`. See [Building functions](functions.md) for nested trees,
inferred outputs, the two named call methods and `vmap`.

## Differentiate the function

Derivative wrappers return typed functions.

```python
grad = sc.gradient(rollout, "cost", "us")

gradient_value = grad((np.array([1.0, 0.0]), np.zeros(N)))
```

The derivative keeps `rollout`'s complete input tree, `(z0, us)`. There is no separate parameter
list to maintain.

The common wrappers are:

```python
sc.gradient(fn, "f", "x")
sc.jacobian(fn, "y", "x")
sc.hessian(fn, "f", "x")
sc.sparse_jacobian(fn, "y", "x")
sc.sparse_hessian(fn, "f", "x")
sc.forward(fn, "y", "x")
sc.adjoint(fn, "y", "x")
```

Differentiation is graph-to-graph. The result compiles, nests and renders like any other
`Function`. See [Derivatives](derivatives.md) and [Sparsity](sparsity.md).

## Declare an optimization problem

A `Problem` separates the model from the solver backend.

```python
@sc.problem(
    vars=sc.L("us", N),
    params=sc.L("z0", 2),
)
def shooting_problem(us: sc.Expr, z0: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    zN, cost = rollout((z0, us))
    return sc.ProblemSpec(
        minimize=cost,
        eq=(zN,),
        lb=sc.const(np.full(N, -2.0)),
        ub=sc.const(np.full(N, 2.0)),
    )
```

The objective is scalar and equality groups are constrained to zero. Use `sc.bounded` for one- or
two-sided inequality groups. Variable bounds have the declared variable structure.

Choose a backend:

```python
solve = sc.solver(
    shooting_problem,
    "ipopt",
    options={"print_level": 0},
)
```

IPOPT, PIQP and scaly-sqp are discovered as plugins. PIQP is accepted only when scaly can prove the
cost quadratic, the constraints affine and the bounds independent of the variables.

## Call the solver

All solver functions have the same five input groups and four output groups:

```text
inputs  = (vars_init, lam_box0, lam_eq0, lam_ineq0, params)
outputs = (vars,      lam_box,  lam_eq,  lam_ineq)
```

For this problem, the variable and parameter trees each have one leaf:

```python
us_opt, lam_box, lam_eq, lam_ineq = solve(
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

Absent multiplier categories stay in the signature as length-zero arrays. Box and inequality
multipliers are signed: positive means the upper side is active, negative the lower side.

A solver is a plain `Function`. Call it with `Expr` leaves to put it inside another graph; the host,
oracles and native wrapper then compile into one shared library. See [Solvers](solvers.md) for
multi-block variables, bounded groups, `qp_problem`, sparse PIQP data, nesting and statistics, and
[Solver backends](solver_backends.md) for backend options and warm-start behavior.

## Preserve regular repetition

The Python loop in `rollout` creates `N` call sites. When iterations are independent, `sc.vmap`
represents the repetition as one node and lowers it to a C loop.

```python
@sc.function(
    sc.G(sc.L("z", 2), sc.L("u", 1), sc.L("znext", 2)),
    sc.L("defect", ...),
)
def defect(inputs: tuple[sc.Expr, sc.Expr, sc.Expr]) -> sc.Expr:
    z, u, znext = inputs
    return step((z, u)) - znext

decision = sc.sym("decision", 2 * (N + 1) + N)
states = decision[: 2 * (N + 1)]
controls = decision[2 * (N + 1) :]

defects = sc.vmap(
    defect,
    N,
    [
        (states, 0, 2),
        (controls, 0, 1),
        (states, 2, 2),
    ],
)
```

Each mapping tuple is `(outer, start, stride)`; iteration `i` reads a slice beginning at
`start + i * stride`. A derivative of a vmapped function is another vmapped function, so source
size and derivative construction scale with one stage, not with the horizon.

## Render C ahead of time

The ahead-of-time (AOT) command uses the same lowering and renderer as the numerical call.

```bash
uv run scaly_codegen mymodule:solve -o generated/
```

It writes one C source file and one header exposing the pointer-array ABI and typed C++ helpers.
Solver-bearing modules include their backend link flags. See [Code generation](codegen.md) and
[the C ABI](../how_it_works/c_abi.md).

## Next steps

- [Building functions](functions.md)
- [Derivatives](derivatives.md)
- [Sparsity](sparsity.md)
- [Solvers](solvers.md)
- [Code generation](codegen.md)
- [Architecture](../how_it_works/architecture.md)
