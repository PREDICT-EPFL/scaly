# Solvers

`Problem` records an optimization problem in symbolic form. `Solver` selects a
numerical backend for that problem and supplies the objective, constraints, and
derivatives it needs. The distinction lets the same problem be solved with
different backends or exported as C.

This page covers declarations, call structures, initial guesses, and solver
status. The [getting-started example](getting_started.md#optimization-problems-and-solvers)
shows these objects in a complete control problem.

## Supported problem formulations

A parameterized nonlinear program, or NLP, has the form

\[
\begin{aligned}
\min_x\quad & f(x,p) \\
\text{subject to}\quad
& h(x,p)=0, \\
& g_{\mathrm{lb}}(p)\leq g(x,p)\leq g_{\mathrm{ub}}(p), \\
& x_{\mathrm{lb}}(p)\leq x\leq x_{\mathrm{ub}}(p).
\end{aligned}
\]

Here \(x\in\mathbb{R}^n\) contains the decision variables and
\(p\in\mathbb{R}^{n_p}\) contains parameters held fixed during each solve.
The objective \(f\) is scalar. The equality residual \(h\) has
\(m_{\mathrm{eq}}\) entries, and the bounded constraint expression \(g\) has
\(m_{\mathrm{ineq}}\) entries. Bounds may depend on parameters but not on the
decision variables. Any constraint category or bound side may be absent.

A quadratic program, or QP, is a special case:

\[
\begin{aligned}
\min_x\quad & \tfrac12 x^\top P(p)x+c(p)^\top x+d(p) \\
\text{subject to}\quad & A(p)x=b(p), \\
& g_{\mathrm{lb}}(p)\leq G(p)x\leq g_{\mathrm{ub}}(p), \\
& x_{\mathrm{lb}}(p)\leq x\leq x_{\mathrm{ub}}(p).
\end{aligned}
\]

Both forms use `@sc.problem` and `ProblemSpec`. Choosing a quadratic-program
backend makes Scaly check that the objective Hessian and constraint Jacobians
are independent of the decision variables, then extract the matrices. A
nonquadratic objective or nonaffine constraint raises `sc.NotQuadratic` during solver
construction. Scaly does not prove that \(P\) is positive semidefinite, which
PIQP requires. A nonlinear-program backend can also solve a quadratic problem.

## Problem declarations are symbolic

For example, consider the parameterized quadratic problem

\[
\begin{aligned}
\min_u\quad & \lVert u-r\rVert^2 \\
\text{subject to}\quad & u_0+u_1=1, \qquad u\geq0.
\end{aligned}
\]

Here `u` is the decision variable and `target` represents the parameter \(r\).
The builder receives both as symbolic expressions:

```python
import numpy as np
import scaly as sc


@sc.problem(vars=sc.arg("u", 2), params=sc.arg("target", 2))
def allocation(u: sc.Expr, target: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    return sc.ProblemSpec(
        minimize=sc.sumsqr(u - target),
        eq=(u.sum() - 1.0,),
        lb=sc.const(0.0),
    )
```

`@sc.problem` runs the Python body once and produces an `sc.Problem`. The cost,
equality residual, and lower bound are `Expr` objects, not evaluated NumPy
values. Later numerical calls supply a target and let the solver choose `u`.
The builder does not run again for each solve.

The declarations specify the names and dimensions of variables and parameters.
In this example, each is a single leaf array, so the builder receives two
`Expr` arguments. The `ProblemSpec[sc.Expr]` annotation says that variable
bounds also have a single-expression structure. Annotations are optional and
do not change evaluation.

| Field      | Symbolic content                                          |
| ---------- | --------------------------------------------------------- |
| `minimize` | scalar objective expression                               |
| `eq`       | tuple of scalar or vector expressions constrained to zero |
| `ineq`     | tuple of `sc.bounded(...)` constraint groups              |
| `lb`, `ub` | expression leaves with the same structure as `vars`       |

An equality is a residual such as `u.sum() - 1.0`, rather than a Python
comparison.

### Bounded expressions

`sc.bounded(expr, lo=..., hi=...)` describes lower and upper limits on an
expression. It belongs inside the `ineq` tuple. For example,
`sc.bounded(u[0] - u[1], lo=-0.5, hi=0.5)` represents
\(-0.5\leq u_0-u_1\leq0.5\). At least one limit is required. A scalar limit
broadcasts over a vector expression, and numerical limits become constants.

Unlike `lb` and `ub`, which limit the variables directly, a bounded expression
can combine several variables. A variable-dependent limit belongs in the
expression itself. For example, \(u\geq-s\) becomes
`sc.bounded(u + slack, lo=0.0)`.

## Numerical calls and return values

`sc.solver` returns an `sc.Solver` for an installed backend. Its main argument
is the declared parameter structure:

```python
solve = sc.solver(allocation, "piqp")
target_value = np.array([0.2, 0.8])
result = solve(target_value)
u, lam_box, lam_eq, lam_ineq = result

status = solve.stats().to_solver_status()
if not status.ok:
    raise RuntimeError(status)
print(u)               # approximately [0.2 0.8]
print(lam_eq.shape)     # (1,)
print(lam_ineq.shape)   # (0,)
```

Scaly supports PIQP for quadratic programs, IPOPT and a custom
sequential quadratic programming (SQP) implementation for nonlinear programs. [Solver backends](solver_backends.md)
covers their differences and installation.

Every call returns four groups:

```text
(optimized variables, bound multipliers, equality multipliers, inequality multipliers)
```

Variables and bound multipliers follow the declared variable tree. Equality and
inequality multipliers are flat vectors of lengths `problem.n_eq` and
`problem.n_ineq`. Missing constraint categories produce empty arrays.
The parameter structure is separate from these four result groups.

A numerical call returns arrays even when the solve fails. Status therefore
needs checking before the result is used, as in the example. The
[statistics section](#statistics) describes what success means.

## Problem dimensions

The declarations fix the number and shape of decision variables and parameters.
For example, `vars=sc.arg("u", 2)` and `params=sc.arg("target", 2)` each declare one
leaf vector of length two. The builder receives `u` and `target` directly,
`solve` takes one target array, and `x0` and the optimized variables each have
shape `(2,)`. A scalar declaration uses `()`, distinct from the one-element
vector shape `(1,)`.

A scalar objective has one value regardless of the variable layout. Each
scalar or vector expression in `eq` contributes its number of entries to
`problem.n_eq`. The same rule applies to expressions inside `ineq` and
`problem.n_ineq`. A two-sided bound counts once per expression entry, not twice.
Variable bounds have their own multipliers and are excluded from `n_ineq`.
For `allocation`, `n_eq` is `1` and `n_ineq` is `0`.

## Grouped variables and parameters

`sc.group` groups several leaves into tuples when separate variable blocks are
more convenient. The input to the builder, initial guess, result, and variable
bounds follow that tree. Equality and inequality multipliers remain flat
vectors. Type checkers can check tuple structures, while Scaly checks numerical
array dimensions at runtime.

The following problem adds a scalar slack variable to relax the lower limit on
both entries of `u`:

\[
\min_{u,s}\ \lVert u-r\rVert^2+10s^2,
\qquad u+s\geq0,\quad s\geq0.
\]

```python
@sc.problem(
    vars=sc.group(sc.arg("u", 2), sc.arg("slack", 1)),
    params=sc.arg("target", 2),
)
def softened_tracking(
    variables: tuple[sc.Expr, sc.Expr], target: sc.Expr,
) -> sc.ProblemSpec[tuple[sc.Expr, sc.Expr]]:
    u, slack = variables
    return sc.ProblemSpec(
        minimize=sc.sumsqr(u - target) + 10.0 * sc.sumsqr(slack),
        ineq=(sc.bounded(u + slack, lo=0.0, name="safe"),),
        lb=(sc.NO_LB, sc.const(0.0)),
    )


solve_soft = sc.solver(softened_tracking, "piqp")
variables, bound_multipliers, eq_multipliers, ineq_multipliers = solve_soft(
    np.array([-1.0, 0.5]), x0=(np.zeros(2), np.zeros(1)),
)
assert solve_soft.stats().to_solver_status().ok
u_soft, slack = variables
print(np.round(u_soft, 6))        # [-0.090909  0.5     ]
print(np.round(slack, 6))         # [0.090909]
print(ineq_multipliers.shape)     # (2,)
```

The variables, `x0`, returned variables, bound multipliers, and `lb` all use
`(u_array, slack_array)`. A scalar bound broadcasts within its leaf, so
`sc.const(0.0)` bounds the one-element slack vector. It does not replace the
surrounding group. `sc.NO_LB` leaves the `u` leaf unbounded below.

`params` can also use `sc.group`. For example, `sc.group(sc.arg("target", 2),
sc.arg("weight", ()))` makes the builder's second argument a tuple of two
expressions. Numerical calls then take `(target_array, weight_scalar_array)`.

One bounded vector contributes one multiplier per entry, even if both lower
and upper limits are present. Its entries follow the order of the groups in
`ineq`. Bound and inequality multipliers are signed: positive corresponds to
an active upper bound, negative to an active lower bound.

## Absent bounds and constraints

Omitting `lb` or `ub`, or setting it to `None`, leaves that entire side of the
variable bounds absent. For grouped variables, `sc.NO_LB` and `sc.NO_UB` leave
one leaf unbounded while preserving the surrounding tuple structure. These
constants represent negative and positive infinity.

For example, the previous declaration's `lb=(sc.NO_LB, sc.const(0.0))` limits
only the slack variable. A declaration `ub=(sc.const(1.0), sc.NO_UB)` would
limit both components of `u` to one and leave the slack unbounded above.
A vector bound may use `-np.inf` or `np.inf` for individual unbounded entries,
wrapped in `sc.const` like other fixed bound data.

For `sc.bounded`, an omitted `lo` or `hi` leaves that side absent. Omitting both
raises `ValueError`, since such a group would impose no constraint. Omitting
`eq` or `ineq` leaves that whole constraint category empty. Its returned
multiplier array then has shape `(0,)`.

## Warm starts and repeated solves

A `Solver` does not automatically reuse the previous result as its next initial
point. Each ordinary call supplies zero initial variables and multipliers.
`x0=` replaces the variables, while `warm=` supplies all four result groups
from a previous solve.

For the `allocation` problem above, IPOPT can use both variables and multipliers:

```python
solve_ipopt = sc.solver(
    allocation, "ipopt", options={"warm_start_init_point": "yes"},
)
first = solve_ipopt(np.array([0.2, 0.8]), x0=np.array([0.5, 0.5]))
assert solve_ipopt.stats().to_solver_status().ok
second = solve_ipopt(np.array([0.3, 0.7]), warm=first)
assert solve_ipopt.stats().to_solver_status().ok
print(np.round(second[0], 6))  # [0.3 0.7]
```

`x0=first[0]` would reuse only the variables and reset the multipliers to zero.
Passing both `x0` and `warm` raises
`TypeError: pass either x0 or warm, not both`.

What the backend does with those inputs differs:

| Backend   | Initial variables                     | Initial multipliers                     |
| --------- | ------------------------------------- | --------------------------------------- |
| PIQP      | ignored                               | ignored                                 |
| IPOPT     | used                                  | used with `warm_start_init_point="yes"` |
| Scaly SQP | used, then clamped to variable bounds | used                                    |

For a receding horizon, Scaly does not shift the result by a stage. `warm=first`
passes exactly the values in `first`. A shifted initial trajectory can be
supplied as `x0`, or together with appropriately shifted multipliers as `warm`.

## Model predictive control

The optimization interface does not prescribe a transcription. Single
shooting records a recurrence inside the objective and constraints. Multiple
shooting declares stage states as variables and records each dynamics defect
as an equality constraint.

### Multiple shooting

In multiple shooting, defect evaluations are independent calculations over
candidate states, even though their equality constraints couple neighboring
stages. This makes them suitable for `sc.vmap`. A sequential rollout has a
loop-carried state and cannot be replaced with the same map.

The [complete multiple-shooting example](getting_started.md#bonus-repeated-stages-with-vmap)
shows the variable layout, mapped defects, and bounds for a double integrator.
The [function guide](functions.md#regular-repetition-vmap) describes mapped
input slices and shared parameters.

## Matrix-data quadratic programs

`sc.qp_problem(n, n_eq, n_ineq)` is shorthand for a `@sc.problem` declaration
with matrices and vectors as parameters. It uses the same solver interface and
quadratic-form checks as an expression-based declaration:

\[
\begin{aligned}
\min_x\quad & \tfrac12 x^\top P x+c^\top x \\
\text{subject to}\quad & Ax=b, \\
& g_{\mathrm{lb}}\leq Gx\leq g_{\mathrm{ub}}.
\end{aligned}
\]

Its parameter tree is `((P, c), (A, b), (G, g_lb, g_ub))`. For an unconstrained
two-variable problem:

```python
matrix_problem = sc.qp_problem(2, 0, 0)
solve_qp = sc.solver(matrix_problem, "piqp")
data = (
    (np.eye(2), np.array([-1.0, -2.0])),
    (np.zeros((0, 2)), np.zeros(0)),
    (np.zeros((0, 2)), np.zeros(0), np.zeros(0)),
)
x, _, _, _ = solve_qp(data)
assert solve_qp.stats().to_solver_status().ok
print(np.round(x, 6))  # [1. 2.]
```

Zero constraints still require the empty arrays shown, including their two
matrix axes. The declarations fix dimensions when `qp_problem` is constructed.
Changing matrix entries is a parameter update, while changing their dimensions
requires another problem.

For a symmetric `P`, the objective Hessian is `P`. More generally, Scaly extracts
`0.5 * (P + P.T)`. The helper has no direct variable-bound parameters. A custom
`Problem` declaration can provide those, or rows of `G` can encode bounds.

## Fixed sparse matrices without sparse constants

`sc.const` accepts dense numerical arrays and scalars, not SciPy
sparse matrices. Converting a sparse matrix with `.toarray()` supplies valid
constant data, but does not turn ordinary matrix multiplication into a sparse
operation. Sparse derivative storage and sparse solver matrices are separate
features.

For fixed linear dynamics, expressing the known sparse equations directly can
avoid assembling a large constant matrix. This three-step model predictive
control problem has fixed dynamics and a changing scalar reference:

\[
\min_{z_0,\ldots,z_3,\,u_0,\ldots,u_2}
\sum_{k=0}^{3}(z_k-r)^2+0.1\sum_{k=0}^{2}u_k^2,
\qquad z_0=1,\quad z_{k+1}-z_k-u_k=0.
\]

```python
@sc.problem(vars=sc.arg("w", 7), params=sc.arg("target", ()))
def tracking_mpc(w: sc.Expr, target: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    states, controls = w[:4], w[4:]
    return sc.ProblemSpec(
        minimize=sc.sumsqr(states - target) + 0.1 * sc.sumsqr(controls),
        eq=(states[:1] - 1.0, states[1:] - states[:-1] - controls),
    )


solve_mpc = sc.solver(tracking_mpc, "piqp", options={"sparse": True})
for target in (0.0, 2.0):
    result = solve_mpc(np.array(target))
    assert solve_mpc.stats().to_solver_status().ok
    states, controls = result[0][:4], result[0][4:]
    print(np.round(states, 3))
# [1.    0.084 0.007 0.001]
# [1.    1.916 1.993 1.999]
```

The objective Hessian and equality matrix are fixed and sparse. Changing the
target changes the linear objective term. `sparse=True` makes PIQP receive
compact matrix data extracted from these equations. There is no sparse
constant object or sparse matrix multiplication in the model itself.

The same solver can be called symbolically without passing any matrices:

```python
@sc.function(sc.arg("target", ()), outputs=sc.arg("trajectory"))
def mpc_trajectory(target: sc.Expr) -> sc.Expr:
    return solve_mpc(target)[0]


trajectory = mpc_trajectory(np.array(0.0))
assert mpc_trajectory.solver_stats("tracking_mpc_piqp").to_solver_status().ok
```

The [complete runnable example](https://github.com/PREDICT-EPFL/scaly/blob/main/examples/sparse_constant_mpc.py)
prints both the states and controls for two target values.

This is useful when the matrix structure is known at model construction. It
does not provide a general replacement for importing arbitrary SciPy sparse
matrices. With `qp_problem`, every entry of the declared matrix parameters may
vary, so numerical zeros in one call do not establish a fixed sparse pattern.

## Nesting a solver in a graph

A call with symbolic parameters records a solver call inside an ordinary
`Function`. Here `solve` is the PIQP allocation solver from
[numerical calls](#numerical-calls-and-return-values):

```python
@sc.function(sc.arg("target", 2), outputs=sc.arg("allocation"))
def allocate(target: sc.Expr) -> sc.Expr:
    result = solve(target)
    return result[0]

print(allocate(np.array([0.2, 0.8])))  # approximately [0.2 0.8]
```

The omitted initial values become constant zero expressions. Explicit `x0` or
`warm` values in a symbolic call must also be symbolic. Constant numerical data
can be wrapped with `sc.const`.

The generated code includes the solver call and surrounding calculations.

!!! warning "Solver sensitivities are not supported"
    An active derivative through a solver call raises `NotImplementedError`.
    This applies to forward and reverse differentiation and Jacobian sparsity
    analysis, including solver calls inside ordinary or mapped functions.
    A solver call independent of the differentiated input does not block its
    derivative.

## Statistics

`solve.stats()` returns statistics from the latest numerical call. A symbolic
call only builds a graph and does not produce solve statistics.

```python
result = solve(target_value)
stats = solve.stats()
status = stats.to_solver_status()
print(status.name)
print(round(stats.obj, 6))  # -0.68 for the PIQP allocation problem
assert status.ok
```

`status.ok` accepts both `OK` and `ACCEPTABLE`. An iteration limit, numerical
failure, or infeasibility is not success, even if the returned variables happen
to satisfy the constraints. `stats.native_status` retains the backend's status
code for more detailed diagnosis.

PIQP reports its quadratic objective, which omits terms independent of the
decision variables. For the allocation problem, it solves the objective
\(u^\top u-2r^\top u\), omitting \(r^\top r\). Consequently,
`stats.obj` is `-0.68` at `target_value`, even though the declared squared-error
cost is zero. Evaluating the model's cost at the returned variables gives the
original objective value. IPOPT and Scaly SQP report that original value.

Other statistics include constraint violation and times in seconds. The
available diagnostics depend on the backend. Fields that a backend cannot
report may be zero, so a zero diagnostic alone does not establish success.
The [statistics reference](../api/solvers.md#statistics) lists the fields.

For a nested solver, statistics belong to the containing compiled function.
For example, `allocate.solver_stats("allocation_piqp")` retrieves the result
of the solver call above. An explicit `name=` on `sc.solver` identifies a
particular solver when a function contains several.

## The underlying function

`solve.function` is the ordinary `Function` behind the parameter-based call.
It exposes the full initial state and parameter structure:

```text
(initial variables, initial bound multipliers,
 initial equality multipliers, initial inequality multipliers, parameters)
```

The first two groups match the variable tree. The next two are flat arrays of
lengths `problem.n_eq` and `problem.n_ineq`. The last group matches the parameter
tree. Its four output groups are the same as those of `solve(params)`.

For the allocation solver, the explicit equivalent of `solve(target_value)` is

```python
explicit_result = solve.function((
    np.zeros(2), np.zeros(2),
    np.zeros(allocation.n_eq), np.zeros(allocation.n_ineq),
    target_value,
))
```

The numerical convenience call inserts these defaults. The
[generated C interface](codegen.md#the-pointer-entry) still exposes every
input leaf, including initial multipliers.

## Solvers in generated C

`write_module` and the `scaly_codegen` command accept a `Solver` directly.
Other [code-generation functions](codegen.md) accept `solve.function`.
Generated solver code needs the relevant native solver libraries, whose link
flags are available in the rendered module comments.

Solver wrappers keep static state and are not reentrant. Concurrent calls to
the same wrapper are unsupported. Python passes tuning options on each call.
Constructing another solver with different tuning reuses the compiled module.
Changing PIQP's `sparse` or SQP's `qp` selects a different compiled interface.
An exported module accepts options through its
[`_with_options` entry](codegen.md#solver-options-in-c).
