# Solvers

`scaly.opt` separates what is solved from how. A problem (`sc.opt.problem`) names its variables,
parameters, objective and constraints and no solver. A method (`sc.opt.PIQP`, `sc.opt.IPM`,
`sc.opt.IPOPT`, `sc.opt.SQP`) is a solver with its options. `sc.opt.solver(problem, method)` builds the `Function`
that solves one with the other: an ordinary typed `Function`, which runs numerically or appears as
a call node in a larger graph. [Solver backends](solver_backends.md) compares the methods; [How
solvers work](../how_it_works/solvers.md) describes their generated wrappers.

## Declare a problem

Use `sc.L(name, shape)` for one tensor and `sc.G(...)` to group tensors. The declared tree
is the symbolic structure the body sees and the NumPy structure calls take.

```python
import scaly as sc
import numpy as np

@sc.opt.problem(
    vars=sc.G(sc.L("u", 2), sc.L("slack", 1)),
    params=sc.G(sc.L("target", 2), sc.L("bias", 1)),
)
def tracking_problem(
    variables: tuple[sc.Expr, sc.Expr],
    params: tuple[sc.Expr, sc.Expr],
) -> sc.opt.ProblemSpec[tuple[sc.Expr, sc.Expr]]:
    u, slack = variables
    target, bias = params
    return sc.opt.ProblemSpec(
        minimize=sc.sumsqr(u - target) + 10.0 * sc.sumsqr(slack - bias),
        eq=(u[0:1] - u[1:2],),
        ineq=(sc.opt.bounded(u + slack, lo=0.0, name="safe"),),
        lb=(sc.opt.NO_LB, sc.const(0.0)),
        ub=(sc.opt.NO_UB, sc.opt.NO_UB),
    )
```

The decorator returns an `sc.opt.NLP`, the general problem. `ProblemSpec` has five parts:

| Field | Meaning |
| --- | --- |
| `minimize` | scalar objective |
| `eq` | tuple of scalar or vector expressions constrained to zero |
| `ineq` | tuple of `sc.opt.bounded(expr, lo=..., hi=..., name=...)` groups |
| `lb` | optional lower bounds with the variables' structure |
| `ub` | optional upper bounds with the variables' structure |

At least one of `lo` and `hi` is required for a bounded group. A scalar bound broadcasts over its
group or variable leaf. `lb` and `ub` must have the variables' tree structure. Use `sc.opt.NO_LB` or
`sc.opt.NO_UB` when one leaf has no bound on that side; use `None` when the entire lower or upper side
is absent. These constants are scalar `Expr` values, so the tree stays statically typed. Names are
metadata and need not match local Python variable names.

Pass a parameter tree when the public interface is fixed. If `params` is omitted, Scaly collects
named expressions closed over by the body and builds the parameter tree from them.

## Choose a method

A method is a frozen dataclass of its options, checked when it is made:

```python
solve = sc.opt.solver(tracking_problem, sc.opt.SQP(options={"max_iter": 30}), name="tracking_sqp")
solve = sc.opt.solver(tracking_problem, "ipopt")   # a method by name, with its default options
solve = sc.opt.solver(tracking_problem)            # "auto": the first installed method that fits
```

The library methods come with their own distributions (`scaly-piqp`, `scaly-ipopt`, `scaly-sqp`);
`sc.opt.IPM` is part of Scaly. Each is declared in the `scaly.methods` entry-point group;
`sc.opt.PIQP` loads its class on first use, and names the distribution to install when it is
missing. `sc.opt.REGISTRY.installed()` lists what is installed. `"auto"` tries PIQP first, then IPM,
IPOPT and SQP, and takes the first that supports the problem.

PIQP and IPM are QP methods: they accept only problems that Scaly proves quadratic. The cost must have a
variable-independent Hessian, every constraint a variable-independent Jacobian, and the bounds must
not depend on the variables; otherwise building the solver raises `sc.opt.NotQuadratic`, naming
what is not. IPOPT and SQP accept nonlinear problems. A method's `options` are the solver's own, by
name, compiled into the wrapper; PIQP's `sparse` chooses its sparse interface (below).

IPM is PIQP's algorithm generated as C, with no library behind it: the solver is specialised to the
problem's sparsity and to which of its bounds are finite, and ships as the rest of a generated module
does. It takes PIQP's settings as `options`, so `sc.opt.IPM(options={"eps_abs": 1e-9})` and
`sc.opt.PIQP(options={"eps_abs": 1e-9})` ask for the same tolerance, and it takes the library's
iterations. `sparse=True` factors the whole KKT system with `linalg.SparseLDL`; `sparse=False`
condenses it and uses a dense Cholesky. By default the solver takes the one whose iteration costs
less, decided when it is built from the problem's structure alone: the counts each iteration is made
of (the condensed matrix's outer products, the factors' multiply-adds, the entries of the sparse
factor) weighed as they were measured on Apple silicon, for the target in force
([Code generation](codegen.md#tuning-for-a-processor)). Both backends follow PIQP's path, so the
choice changes only the speed; on the Maros–Mészáros set it takes the dense backend for problems
with far more inequality rows than variables and the sparse one nearly everywhere else.
`sc.opt.ipm.choose_backend(structure)` says which it would take.

A problem caches its objective, gradient, constraint Jacobian and bounds oracles. Methods that need
different Hessian triangles share them and cache one Hessian per triangle.

## Solver arguments and results

Every solver takes the same five arguments and returns the same five results:

```text
arguments = vars_init, lam_box0, lam_eq0, lam_ineq0, params
results   = vars,      lam_box,  lam_eq,  lam_ineq,  info
```

The arguments are the warm start, the box multipliers, the equality multipliers, the inequality
multipliers and the parameters. The warm start, the box multipliers and their results have the
declared variable tree. The parameters have the declared parameter tree. Equality and inequality
multipliers are flat arrays whose lengths are `problem.n_eq` and `problem.n_ineq`. An absent
category is still passed, as an array of length zero.

```python
variables, lam_box, lam_eq, lam_ineq, info = solve(
    (np.zeros(2), np.zeros(1)),
    (np.zeros(2), np.zeros(1)),
    np.zeros(1),
    np.zeros(2),
    (np.array([0.25, -0.75]), np.array([0.1])),
)
u, slack = variables
if not sc.Status(int(info.status)).ok:
    raise RuntimeError(sc.Status(int(info.status)))
```

`info` is an `sc.opt.Info`: the solve's `status` (an `sc.Status` code; `OK` and `ACCEPTABLE` are
solutions), the iterations `iter`, the problem's `objective` at the solution, constants included,
and its `primal_residual`, the largest constraint violation. They are outputs like the others, so a graph that nests the solve can
read them too.

`lam_ineq` and `lam_box` are signed. A positive value means the upper bound is active; a negative
value means the lower bound is active. IPOPT and SQP consume warm starts. PIQP ignores them
because its C interface has no warm-start entry point, and IPM, which follows it, does too.

`solve.input_names` and `solve.output_names` show the flattened C signature. Grouping affects
Python and static types, but not leaf order in the generated ABI.

## Matrix-data quadratic programs

`sc.opt.QP(n, n_eq, n_ineq)` is a problem in matrix form, whose matrices are its parameters:

```text
minimize  0.5 x' P x + c' x
subject to A x = b
           g_lb <= G x <= g_ub
```

Its parameter structure is `((P, c), (A, b), (G, g_lb, g_ub))`, passed as the solver's fifth
argument. Use zero-sized arrays for absent constraint blocks:

```python
problem = sc.opt.QP(2, 0, 0)
solve_qp = sc.opt.solver(problem, "piqp")

data = (
    (np.eye(2), np.array([-1.0, -2.0])),
    (np.zeros((0, 2)), np.zeros(0)),
    (np.zeros((0, 2)), np.zeros(0), np.zeros(0)),
)
x, lam_box, lam_eq, lam_ineq, info = solve_qp(np.zeros(2), np.zeros(2), np.zeros(0), np.zeros(0), data)
```

A `QP` is an `NLP`, so every method solves it; it goes through the same quadratic proof and
extraction as any other problem. Bounds on `x` are not part of `QPData`; declare a problem directly
when you need them. For `0.5 * x @ P @ x`, the extracted Hessian is `0.5 * (P + P.T)`: provide a
symmetric matrix when that distinction matters.

## Normal forms

A method reads a problem through one of two normal forms, which are public for methods written
outside Scaly:

- `sc.opt.extract_qp(problem)` proves the problem quadratic and returns its `QPForm`: `P`, `c`,
  `A`, `b`, `G`, the bounds and the objective's constant `f0`, each an `Expr` of the problem's
  parameters. `form.patterns()` gives the structural patterns of `P`'s upper triangle, `A` and `G`,
  and `form.in_pattern(matrix, pattern)` a matrix's values in one.
- `sc.opt.nlp_oracles(problem)` returns its `NLPOracles`: the objective and constraint oracle, the
  gradient, the sparse constraint Jacobian, the sparse Hessian of the Lagrangian and the bounds, as
  Functions of the flat variable vector and the parameters.

## Sparse PIQP data

PIQP takes dense problem data by default. `sc.opt.PIQP(sparse=True)` derives the structural
patterns of the extracted `P`, `A` and `G` and bakes compressed sparse column tables into the
generated C:

```python
solve_sparse = sc.opt.solver(problem, sc.opt.PIQP(sparse=True))
```

The sparse path rejects QP matrices computed from another solver output because solver calls are
opaque to structural dependency analysis. The dense path has no such restriction.

Problem oracles represent an absent bound with IEEE negative or positive infinity. Each method's
wrapper translates those values to its solver's convention before solving.

## Nesting a solver in a graph

Call the solver with the same five arguments, built from `Expr` leaves, to embed a solve:

```python
@sc.function(2, 1, output="u")
def filtered_control(target, bias):
    nested = sc.opt.solver(tracking_problem, "sqp", name="nested_tracking")
    variables, *_ = nested(
        (sc.const(np.zeros(2)), sc.const(np.zeros(1))),
        (sc.const(np.zeros(2)), sc.const(np.zeros(1))),
        sc.const(np.zeros(1)),
        sc.const(np.zeros(2)),
        (target, bias),
    )
    return variables[0]
```

The call lowers to one generated solver wrapper in the same shared library as the host function and
its oracles. A solve has no derivative rule of its own: a derivative that reaches one raises
`NotImplementedError`, and [`sc.custom_derivative`](derivatives.md#custom-derivatives) gives the solver `Function` one.
A solve whose arguments do not depend on what is being differentiated is a constant and needs none.

## Statistics

Beyond `info`, the latest solve's statistics are read from the compiled function:

```python
stats = sc.opt.solver_stats(solve)
print(stats.iter, stats.t_solver, stats.n_eval_f)
```

A host function can reach more than one solver. Pass the solver's name to
`sc.opt.solver_stats(host, name)` to select one. Statistics include the status and the solver's own
native status, iteration count, objective, oracle evaluation counts, timing splits, primal
violation, last step norm, accepted step length, backtracks and accumulated QP iterations.

Statistics belong to the library methods' wrappers. An IPM solver is generated code with no wrapper,
and `info` is all it reports.

## Shipping one in C

A solver-bearing function renders through the same C API as any other function. Its module also
carries the include, library, runtime-path and link flags for every plugin it reaches. See [Code
generation](codegen.md) and [the generated interface](../how_it_works/generated_interface.md).

## Limits

- The vendored PIQP and IPOPT libraries build on the first sync and take 5 to 8 minutes from a
  cold checkout.
- PIQP and IPM do not consume warm starts.
- Generated wrappers use per-symbol static storage and are not reentrant.
- A method's options are compiled into the wrapper, so changing one recompiles it.
- Differentiation through a solve is not implemented.
