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

## Declare a function

A `Function` gives a graph a named boundary. Decorate a Python body and declare the shape of each
parameter, in order:

```python
@sc.function(2, 1)
def step(z, u):
    return z + 0.1 * sc.concat([z[1:], u])
```

The body is traced once, with an `Expr` per parameter, and the output shape comes from the trace.
The inputs are named after the parameters and the output after the function, and those names are
what derivatives refer to and what the generated C calls the buffers. The shapes may also be left
out entirely, `@sc.function` on its own, and then each call binds them; see
[Building functions](functions.md).

Call it with arrays to run it:

```python
z1 = step(np.array([1.0, 2.0]), np.array([0.5]))
# array([1.2, 2.05])
```

The first call lowers the graph, renders C, compiles a shared library and stores it in the
just-in-time (JIT) cache. Later calls reuse it. A single output is the array itself, not a
one-element tuple.

Call it with `Expr` arguments to compose it symbolically:

```python
N = 20

@sc.function(2, N, output=sc.G("zN", "cost"))
def rollout(z, us):
    cost = sc.const(0.0)
    for k in range(N):
        u = us[k : k + 1]
        cost = cost + sc.sumsqr(z) + 0.1 * sc.sumsqr(u)
        z = step(z, u)
    return z, cost + 10.0 * sc.sumsqr(z)
```

`step(...)` dispatches on the arguments it is given: `numerical_call` for arrays, `symbolic_call`
for expressions. The symbolic call adds a `CALL` node, so the generated C contains one `step`
procedure and calls it from `rollout`. `output=` names the two results; without it they would be
`rollout_0` and `rollout_1`.

## Differentiate the function

Derivative wrappers take a function and the names of an output and an input, and return a function
of the same parameters:

```python
grad = sc.gradient(rollout, "cost", "us")

gradient_value = grad(np.array([1.0, 0.0]), np.zeros(N))
```

There is no separate parameter list to maintain. For a function with one output, the input alone
names the derivative, as in `sc.gradient(f, "x")`.

The common wrappers are:

```python
sc.gradient(fn, "f", "x")
sc.jacobian(fn, "y", "x")
sc.hessian(fn, "f", "x")
sc.sparse_jacobian(fn, "y", "x")
sc.sparse_hessian(fn, "f", "x")
sc.forward(fn, "y", "x")   # called with fn's arguments, then the tangent seed
sc.adjoint(fn, "y", "x")   # called with fn's arguments, then the cotangent seed
```

Differentiation is graph-to-graph. The result compiles, nests and renders like any other
`Function`. See [Derivatives](derivatives.md) and [Sparsity](sparsity.md).

## Declare an optimization problem

An optimization problem separates the model from the solver.

```python
@sc.opt.problem(
    vars=sc.L("us", N),
    params=sc.L("z0", 2),
)
def shooting_problem(us: sc.Expr, z0: sc.Expr) -> sc.opt.ProblemSpec[sc.Expr]:
    zN, cost = rollout(z0, us)
    return sc.opt.ProblemSpec(
        minimize=cost,
        eq=(zN,),
        lb=sc.const(np.full(N, -2.0)),
        ub=sc.const(np.full(N, 2.0)),
    )
```

The objective is scalar and equality groups are constrained to zero. Use `sc.opt.bounded` for one- or
two-sided inequality groups. Variable bounds have the declared variable structure.

Choose a method:

```python
solve = sc.opt.solver(
    shooting_problem,
    sc.opt.IPOPT(options={"print_level": 0}))
```

IPOPT, PIQP and scaly-sqp are methods discovered as plugins; `sc.opt.solver(problem)` takes the
first installed one that fits. PIQP is accepted only when scaly can prove the cost quadratic, the
constraints affine and the bounds independent of the variables.

## Call the solver

A solver takes five arguments and returns five results. The arguments are the warm start, the box
multipliers, the equality multipliers, the inequality multipliers and the parameters; the results
end with the solve's `Info` (status, iterations, objective, primal residual):

```text
arguments = vars_init, lam_box0, lam_eq0, lam_ineq0, params
results   = vars,      lam_box,  lam_eq,  lam_ineq,  info
```

For this problem, the variable and parameter trees each have one leaf:

```python
us_opt, lam_box, lam_eq, lam_ineq, info = solve(
    np.zeros(N),
    np.zeros(N),
    np.zeros(2),
    np.zeros(0),
    np.array([1.0, 0.0]),
)

stats = sc.opt.solver_stats(solve)
print(stats.obj, stats.iter, stats.to_solver_status())
```

Absent multiplier categories stay in the signature as length-zero arrays. Box and inequality
multipliers are signed: positive means the upper side is active, negative the lower side.

A solver is a plain `Function`. Call it with `Expr` leaves to put it inside another graph; the host,
oracles and native wrapper then compile into one shared library. See [Solvers](solvers.md) for
multi-block variables, bounded groups, methods, `sc.opt.QP`, sparse PIQP data, nesting and statistics, and
[Solver backends](solver_backends.md) for backend options and warm-start behavior.

## Preserve regular repetition

The Python loop in `rollout` creates `N` call sites. When iterations are independent, `sc.vmap`
represents the repetition as one node and lowers it to a C loop.

```python
@sc.function(2, 1, 2)
def defect(z, u, znext):
    return step(z, u) - znext

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
`start + i * stride`. The callee's declared shapes are what `vmap` slices by, which is why `defect`
declares them. A derivative of a vmapped function is another vmapped function, so source
size and derivative construction scale with one stage, not with the horizon.

## Render C ahead of time

The ahead-of-time (AOT) command uses the same lowering and renderer as the numerical call.

```bash
uv run scaly_codegen mymodule:solve -o generated/
```

It writes one C source file and one header, C or C++, exposing the pointer ABI and typed buffers.
Solver-bearing modules include their backend link flags. See [Code generation](codegen.md) and
[the generated interface](../how_it_works/generated_interface.md).

## Next steps

- [Building functions](functions.md)
- [Derivatives](derivatives.md)
- [Sparsity](sparsity.md)
- [Solvers](solvers.md)
- [Code generation](codegen.md)
- [Architecture](../how_it_works/architecture.md)
