# Solvers

A solver starts from a backend-free `Problem`. Declare variables, parameters, objective, and
constraints once, then choose PIQP, IPOPT, or alloy-sqp with `al.solver`. The result is an ordinary
typed `Function`, so it can run numerically or appear as a call node in a larger graph.
[Solver backends](solver_backends.md) compares the implementations. [How solvers
work](../how_it_works/solvers.md) describes their generated wrappers.

## Declare a problem

Use `al.L(name, shape)` for one tensor and `al.G(...)` to group tensors. The declared tree
determines the symbolic structure seen by the body and the NumPy structure used at calls.

```python
import alloy as al
import numpy as np

@al.problem(
    vars=al.G(al.L("u", 2), al.L("slack", 1)),
    params=al.G(al.L("target", 2), al.L("bias", 1)),
)
def tracking_problem(
    variables: tuple[al.Expr, al.Expr],
    params: tuple[al.Expr, al.Expr],
) -> al.ProblemSpec[tuple[al.Expr, al.Expr]]:
    u, slack = variables
    target, bias = params
    return al.ProblemSpec(
        minimize=al.sumsqr(u - target) + 10.0 * al.sumsqr(slack - bias),
        eq=(u[0:1] - u[1:2],),
        ineq=(al.bounded(u + slack, lo=0.0, name="safe"),),
        lb=(al.NO_LB, al.const(0.0)),
        ub=(al.NO_UB, al.NO_UB),
    )
```

`ProblemSpec` has five parts:

| Field | Meaning |
| --- | --- |
| `minimize` | scalar objective |
| `eq` | tuple of scalar or vector expressions constrained to zero |
| `ineq` | tuple of `al.bounded(expr, lo=..., hi=..., name=...)` groups |
| `lb` | optional lower bounds with the variables' structure |
| `ub` | optional upper bounds with the variables' structure |

At least one of `lo` and `hi` is required for a bounded group. A scalar bound broadcasts over its
group or variable leaf. `lb` and `ub` must have the variables’ tree structure. Use `al.NO_LB` or
`al.NO_UB` when one leaf has no bound on that side; use `None` when the entire lower or upper side
is absent. These constants are scalar `Expr` values, so the tree remains statically typed. Names are
metadata and need not match local Python variable names.
Pass a parameter tree when the public interface is fixed. If `params` is omitted, Alloy collects
named expressions closed over by the body and builds the parameter tree from them.

## Select one solver

The backend is positional. The artifact name and backend options belong to the solver construction:

```python
solve = al.solver(
    tracking_problem,
    "sqp",
    name="tracking_sqp",
    options={"max_iter": 30},
)
```

PIQP accepts only problems that Alloy can prove are quadratic programs. The cost must have a
variable-independent Hessian, every constraint must have a variable-independent Jacobian, and
bounds must not depend on the variables. Failure raises `al.NotQuadratic` when the solver is built.
IPOPT and alloy-sqp accept nonlinear problems.
A problem caches its common objective, gradient, constraint Jacobian, and bounds oracles. Solvers
that need different Hessian triangles share the common oracles and cache one Hessian per triangle.

## Solver input and output structure

Every solver has the same five input groups and four output groups:

```text
inputs  = (vars_init, lam_box0, lam_eq0, lam_ineq0, params)
outputs = (vars,      lam_box,  lam_eq,  lam_ineq)
```

The variable and box-multiplier groups have the declared variable tree. The parameter group has the
declared parameter tree. Equality and inequality multipliers are flat arrays whose lengths are
`problem.n_eq` and `problem.n_ineq`. An absent category is still present as an array of length
zero.

```python
result = solve.numerical_call(
    (
        (np.zeros(2), np.zeros(1)),
        (np.zeros(2), np.zeros(1)),
        np.zeros(1),
        np.zeros(2),
        (np.array([0.25, -0.75]), np.array([0.1])),
    )
)
(variables, lam_box, lam_eq, lam_ineq) = result
u, slack = variables
```

`lam_ineq` and `lam_box` are signed. A positive value means the upper bound is active; a negative
value means the lower bound is active. IPOPT and alloy-sqp consume warm starts. PIQP currently
ignores them because its C interface has no warm-start entry point.
`solve.input_names` and `solve.output_names` show the flattened C signature. Grouping affects
Python and static types, but not leaf order in the generated ABI.

## Matrix-data quadratic programs

`al.qp_problem(n, n_eq, n_ineq)` is the typed matrix form:

```text
minimize  0.5 x' P x + c' x
subject to A x = b
           g_lb <= G x <= g_ub
```

Its parameter structure is `((P, c), (A, b), (G, g_lb, g_ub))`. Use zero-sized arrays for absent
constraint blocks:

```python
problem = al.qp_problem(2, 0, 0)
solve_qp = al.solver(problem, "piqp")

data = (
    (np.eye(2), np.array([-1.0, -2.0])),
    (np.zeros((0, 2)), np.zeros(0)),
    (np.zeros((0, 2)), np.zeros(0), np.zeros(0)),
)
x, lam_box, lam_eq, lam_ineq = solve_qp.numerical_call(
    (np.zeros(2), np.zeros(2), np.zeros(0), np.zeros(0), data)
)
```

This helper goes through the same quadratic proof and extraction as any other `Problem`. Bounds on
`x` are not part of `QPData`; declare a problem directly when you need them.
For `0.5 * x @ P @ x`, the extracted Hessian is `0.5 * (P + P.T)`. Provide a symmetric matrix when
that distinction matters.

## Sparse PIQP data

PIQP uses dense problem data by default. Pass `options={"sparse": True}` to derive the structural
patterns of the extracted `P`, `A`, and `G` and bake compressed sparse column tables into generated
C:

```python
solve_sparse = al.solver(problem, "piqp", options={"sparse": True})
```

The sparse path rejects QP matrices computed from another solver output because solver calls are
opaque to structural dependency analysis. The dense path has no such restriction.
Problem oracles represent an absent bound with IEEE negative or positive infinity. Built-in solver
adapters translate those values to the backend’s native convention before solving.

## Nesting a solver in a graph

Use `symbolic_call` with the same declared structure to embed a solve:

```python
@al.function(
    al.G(al.L("target", 2), al.L("bias", 1)),
    al.L("u", ...),
)
def filtered_control(params: tuple[al.Expr, al.Expr]) -> al.Expr:
    target, bias = params
    nested = al.solver(tracking_problem, "sqp", name="nested_tracking")
    result = nested.symbolic_call(
        (
            (al.const(np.zeros(2)), al.const(np.zeros(1))),
            (al.const(np.zeros(2)), al.const(np.zeros(1))),
            al.const(np.zeros(1)),
            al.const(np.zeros(2)),
            (target, bias),
        )
    )
    return result[0][0]
```

The call lowers to one generated solver wrapper in the same shared library as the host function and
its oracles. `SOLVER_CALL` is currently non-differentiable, so derivatives through a solve are
zero.

## Statistics

After a numerical call, read the latest statistics from the compiled function:

```python
stats = solve.solver_stats()
status = stats.to_solver_status()
if status is not None and not status.ok:
    raise RuntimeError(status)
```

A host function can reach more than one solver. Pass the solver artifact name to
`host.solver_stats(name)` to select one. Statistics include statuses, iteration count, objective,
oracle evaluation counts, timing splits, primal violation, last step norm, accepted step length,
backtracks, and accumulated QP iterations.

## Shipping one in C

A solver-bearing function renders through the same C API as any other function. Its module also
carries the include, library, runtime-path, and link flags for every reached plugin. See [Code
generation](codegen.md) and [the C ABI](../how_it_works/c_abi.md).

## Limits

- The vendored PIQP and IPOPT libraries build on the first sync and can take 5 to 8 minutes from a
  cold checkout.

- PIQP does not consume warm starts.

- Generated wrappers use per-symbol static storage and are not reentrant.

- Backend options are compiled into the wrapper.

- Differentiation through a solve is not implemented.
