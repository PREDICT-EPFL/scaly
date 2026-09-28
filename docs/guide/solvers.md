# Solvers

A solver starts from a `Problem`, which names no backend. Declare variables, parameters, objective and
constraints once, then choose PIQP, IPOPT or scaly-sqp with `sc.solver`. The result is an ordinary
typed `Function`, so it can run numerically or appear as a call node in a larger graph.
[Solver backends](solver_backends.md) compares the implementations. [How solvers
work](../how_it_works/solvers.md) describes their generated wrappers.

## Declare a problem

Use `sc.L(name, shape)` for one tensor and `sc.G(...)` to group tensors. The declared tree
is the symbolic structure the body sees and the NumPy structure calls take.

```python
import scaly as sc
import numpy as np

@sc.problem(
    vars=sc.G(sc.L("u", 2), sc.L("slack", 1)),
    params=sc.G(sc.L("target", 2), sc.L("bias", 1)),
)
def tracking_problem(
    variables: tuple[sc.Expr, sc.Expr],
    params: tuple[sc.Expr, sc.Expr],
) -> sc.ProblemSpec[tuple[sc.Expr, sc.Expr]]:
    u, slack = variables
    target, bias = params
    return sc.ProblemSpec(
        minimize=sc.sumsqr(u - target) + 10.0 * sc.sumsqr(slack - bias),
        eq=(u[0:1] - u[1:2],),
        ineq=(sc.bounded(u + slack, lo=0.0, name="safe"),),
        lb=(sc.NO_LB, sc.const(0.0)),
        ub=(sc.NO_UB, sc.NO_UB),
    )
```

`ProblemSpec` has five parts:

| Field | Meaning |
| --- | --- |
| `minimize` | scalar objective |
| `eq` | tuple of scalar or vector expressions constrained to zero |
| `ineq` | tuple of `sc.bounded(expr, lo=..., hi=..., name=...)` groups |
| `lb` | optional lower bounds with the variables' structure |
| `ub` | optional upper bounds with the variables' structure |

At least one of `lo` and `hi` is required for a bounded group. A scalar bound broadcasts over its
group or variable leaf. `lb` and `ub` must have the variables' tree structure. Use `sc.NO_LB` or
`sc.NO_UB` when one leaf has no bound on that side; use `None` when the entire lower or upper side
is absent. These constants are scalar `Expr` values, so the tree stays statically typed. Names are
metadata and need not match local Python variable names.

Pass a parameter tree when the public interface is fixed. If `params` is omitted, Scaly collects
named expressions closed over by the body and builds the parameter tree from them.

## Select one solver

The backend is positional. The artifact name and backend options are given when the solver is built:

```python
solve = sc.solver(
    tracking_problem,
    "sqp",
    name="tracking_sqp",
    options={"max_iter": 30},
)
```

PIQP accepts only problems that Scaly can prove are quadratic programs. The cost must have a
variable-independent Hessian, every constraint must have a variable-independent Jacobian, and
bounds must not depend on the variables. Failure raises `sc.NotQuadratic` when the solver is built.
IPOPT and scaly-sqp accept nonlinear problems.

A problem caches its common objective, gradient, constraint Jacobian, and bounds oracles. Solvers
that need different Hessian triangles share the common oracles and cache one Hessian per triangle.

## Solver arguments and results

Every solver takes the same five arguments and returns the same four results:

```text
arguments = vars_init, lam_box0, lam_eq0, lam_ineq0, params
results   = vars,      lam_box,  lam_eq,  lam_ineq
```

The arguments are the warm start, the box multipliers, the equality multipliers, the inequality
multipliers and the parameters. The warm start, the box multipliers and their results have the
declared variable tree. The parameters have the declared parameter tree. Equality and inequality
multipliers are flat arrays whose lengths are `problem.n_eq` and `problem.n_ineq`. An absent
category is still passed, as an array of length zero.

```python
result = solve(
    (np.zeros(2), np.zeros(1)),
    (np.zeros(2), np.zeros(1)),
    np.zeros(1),
    np.zeros(2),
    (np.array([0.25, -0.75]), np.array([0.1])),
)
(variables, lam_box, lam_eq, lam_ineq) = result
u, slack = variables
```

`lam_ineq` and `lam_box` are signed. A positive value means the upper bound is active; a negative
value means the lower bound is active. IPOPT and scaly-sqp consume warm starts. PIQP
ignores them because its C interface has no warm-start entry point.

`solve.input_names` and `solve.output_names` show the flattened C signature. Grouping affects
Python and static types, but not leaf order in the generated ABI.

## Matrix-data quadratic programs

`sc.qp_problem(n, n_eq, n_ineq)` is the typed matrix form:

```text
minimize  0.5 x' P x + c' x
subject to A x = b
           g_lb <= G x <= g_ub
```

Its parameter structure is `((P, c), (A, b), (G, g_lb, g_ub))`, passed as the solver's fifth
argument. Use zero-sized arrays for absent constraint blocks:

```python
problem = sc.qp_problem(2, 0, 0)
solve_qp = sc.solver(problem, "piqp")

data = (
    (np.eye(2), np.array([-1.0, -2.0])),
    (np.zeros((0, 2)), np.zeros(0)),
    (np.zeros((0, 2)), np.zeros(0), np.zeros(0)),
)
x, lam_box, lam_eq, lam_ineq = solve_qp(np.zeros(2), np.zeros(2), np.zeros(0), np.zeros(0), data)
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
solve_sparse = sc.solver(problem, "piqp", options={"sparse": True})
```

The sparse path rejects QP matrices computed from another solver output because solver calls are
opaque to structural dependency analysis. The dense path has no such restriction.

Problem oracles represent an absent bound with IEEE negative or positive infinity. Built-in solver
adapters translate those values to the backend's native convention before solving.

## Nesting a solver in a graph

Call the solver with the same five arguments, built from `Expr` leaves, to embed a solve:

```python
@sc.function(2, 1, output="u")
def filtered_control(target, bias):
    nested = sc.solver(tracking_problem, "sqp", name="nested_tracking")
    result = nested(
        (sc.const(np.zeros(2)), sc.const(np.zeros(1))),
        (sc.const(np.zeros(2)), sc.const(np.zeros(1))),
        sc.const(np.zeros(1)),
        sc.const(np.zeros(2)),
        (target, bias),
    )
    return result[0][0]
```

The call lowers to one generated solver wrapper in the same shared library as the host function and
its oracles. A solve has no derivative rule of its own: a derivative that reaches one raises
`NotImplementedError`, and [`sc.custom_derivative`](derivatives.md#custom-derivatives) gives the solver `Function` one.
A solve whose arguments do not depend on what is being differentiated is a constant and needs none.

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
backtracks and accumulated QP iterations.

## Shipping one in C

A solver-bearing function renders through the same C API as any other function. Its module also
carries the include, library, runtime-path and link flags for every plugin it reaches. See [Code
generation](codegen.md) and [the generated interface](../how_it_works/generated_interface.md).

## Limits

- The vendored PIQP and IPOPT libraries build on the first sync and take 5 to 8 minutes from a
  cold checkout.
- PIQP does not consume warm starts.
- Generated wrappers use per-symbol static storage and are not reentrant.
- Backend options are compiled into the wrapper, so changing one recompiles it.
- Differentiation through a solve is not implemented.
