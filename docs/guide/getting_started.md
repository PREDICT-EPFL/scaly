# Getting started

This guide introduces scaly's modelling workflow through a double-integrator
control problem: symbolic expressions, functions, derivatives, optimization,
and generated C. It assumes familiarity with Python, NumPy, and basic optimal
control, but no experience with symbolic modelling libraries such as CasADi.

The examples use scaly with the IPOPT plugin and a C compiler, as described in
[Installation](installation.md).

## Symbolic variables and expressions

Consider the discrete-time model

\[
x_k = \begin{bmatrix}p_k \\ v_k\end{bmatrix}, \qquad
x_{k+1} = f(x_k, u_k)
= \begin{bmatrix}p_k + 0.1 v_k \\ v_k + 0.1 u_k\end{bmatrix}.
\]

A symbolic variable represents an input whose numerical value is not yet
specified. In scaly, it has a name and a fixed shape. For this model, `x`
represents a two-element state vector and `u` a one-element control vector:

```python
import numpy as np
import scaly as sc

x = sc.sym("x", 2)
u = sc.sym("u", 1)
xnext = x + 0.1 * sc.concat([x[1:], u])
```

All three objects are instances of `sc.Expr`. `x` and `u` are input expressions and
`xnext` describes a calculation using them. Unlike an operation on NumPy arrays,
this addition does not calculate a numerical result. It creates an expression
node that records the addition and its operands.

Together, the inputs and operations form an **expression graph**. Its nodes
represent values, and its edges record which values an operation needs.
For `xnext`, the graph records the velocity slice, its concatenation with `u`,
the multiplication by `0.1`, and the addition to `x`. Scaly uses this graph to
calculate derivatives and generate code. Assigning a new value to the Python
name `x` later does not change the recorded graph.

Shapes, indexing, broadcasting, and arithmetic follow NumPy conventions.
Here, `x[1:]` has shape `(1,)`, so concatenating it with `u` produces a vector
of shape `(2,)`. `*` is elementwise multiplication and `@` is matrix
multiplication. The [expression reference](../api/core.md#expressions) lists
`Expr` methods, and [array builders](../api/core.md#builders) include operations
such as `sc.concat`, `sc.stack`, and `sc.sumsqr`.

You will rarely write `sc.sym` yourself. The decorator in the next section
creates these symbols from its declarations, and the rest of this guide never
calls `sc.sym` again. It appears here to show what a function body works with.

## From expressions to a Function

An `sc.Expr` describes a value. An `sc.Function` gives a calculation named
inputs and outputs so that it can be evaluated, composed with other functions,
differentiated, or exported as C. `Function` is the unit of composition in scaly.

The `@sc.function` decorator constructs one by running a Python body with
symbolic inputs:

```python
@sc.function(sc.arg("x", 2), sc.arg("u", 1), outputs=sc.arg("xnext"))
def model(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    return x + 0.1 * sc.concat([x[1:], u])
```

`sc.arg` declares one named array. Here, each Python parameter has its own
argument declaration. The output is named `xnext`, whose shape is inferred
from the returned expression because its declaration gives no shape. Use `sc.group`
to combine arguments into a tuple when a parameter or output has several parts.

The decorator creates one symbol per declaration, exactly as `sc.sym("x", 2)`
and `sc.sym("u", 1)` did above, runs the body once with them, and records the
returned expression. After decoration, `model` is an `sc.Function` object, rather than
the original Python function. It remains callable:

```python
x1 = model(np.array([1.0, 2.0]), np.array([0.5]))
print(x1)
# [1.2  2.05]
```

A call with numerical arrays evaluates the graph. The first numerical call
generates and compiles C. Subsequent calls reuse the compiled code. The Python
body does not run again. A call with symbolic expressions instead includes
the function in a larger graph, as the trajectory example below demonstrates.

The declarations determine the call structure. `sc.group` introduces a tuple,
whereas a single `sc.arg` takes or returns an array directly. Shape `()` denotes
a scalar, and shape `(1,)` denotes a one-element vector. These are distinct.
The [functions guide](functions.md) covers nested groups and other declarations.

### Body annotations and call types

The `sc.Expr` annotations let a type checker verify operations on `x` and `u`
inside the body, and verify their agreement with the decorator. They are
optional and do not change tracing or evaluation. The decorator defines
runtime behavior and the types of symbolic and numerical calls.

Declaring every input shape is more to write than a plain Python function, and
this guide does it everywhere. The shapes are what let one definition become
one compiled function with fixed buffer sizes, exported as it is. Declarations
can leave input shapes open, but each shape then has to be supplied later, at
the call or before export. The boilerplate moves rather than disappears. The
[functions guide](functions.md#leaving-shapes-out) covers these templates and
when they pay off.

## Derivatives are Functions too

For the model above,

\[
\frac{\partial f}{\partial x}
= \begin{bmatrix} 1 & 0.1 \\ 0 & 1 \end{bmatrix}.
\]

`sc.jacobian` constructs an `sc.Function` for this derivative:

```python
model_jac = sc.jacobian(model, "xnext", "x")
print(model_jac(np.array([1.0, 2.0]), np.array([0.5])))
# [[1.  0.1]
#  [0.  1. ]]
```

The strings select the output and input by their declared names. `model_jac`
keeps the input structure of `model`, even though this particular Jacobian is
constant. For a nonlinear model, the supplied values determine where the
Jacobian is evaluated.

Scaly applies differentiation rules to the expression graph, producing another
graph. It does not approximate derivatives by perturbing numerical inputs.
The resulting function can be composed and exported like any other function.
The [derivatives guide](derivatives.md) covers gradients, Hessians, and
products with derivative matrices. The [sparsity guide](sparsity.md) covers
calculating only entries that may be nonzero.

## Composing a trajectory model

A function can reuse `model` to compute a trajectory and its cost:

\[
x_{k+1} = f(x_k,u_k), \qquad
J(x_0, U) = \sum_{k=0}^{N-1}\left(\lVert x_k\rVert^2 + 0.1 u_k^2\right)
    + 10\lVert x_N\rVert^2,
\qquad U=(u_0,\ldots,u_{N-1}), \quad N=20.
\]

```python
N = 20


@sc.function(
    sc.arg("x0", 2), sc.arg("us", N),
    outputs=sc.group(sc.arg("xN"), sc.arg("cost")),
)
def rollout(x: sc.Expr, us: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    cost = sc.const(0.0)
    for k in range(N):
        u = us[k : k + 1]
        cost = cost + sc.sumsqr(x) + 0.1 * sc.sumsqr(u)
        x = model(x, u)
    return x, cost + 10.0 * sc.sumsqr(x)


x0 = np.array([1.0, 0.0])
us0 = np.zeros(N)
xN, cost = rollout(x0, us0)
print(xN, float(cost))
# [1. 0.] 30.0
```

`sc.const(0.0)` creates a constant expression, and `sc.sumsqr(x)` expresses
\(\lVert x\rVert^2\). The symbolic call `model(x, u)` records a call to the
existing `Function` inside `rollout`. The two declarations in the output group
give the numerical result its tuple structure.

Python runs while the decorator builds the graph. Consequently, this `for`
loop creates `N` successive calls to `model`. It does not record a loop.
The horizon is fixed when `rollout` is defined, while `x0` and `us` can change
on every evaluation. This distinction between Python execution and recorded
operations also matters for repeated independent calculations, covered in
the [bonus section on `vmap`](#bonus-repeated-stages-with-vmap).

## Optimization problems and solvers

The trajectory model gives a single-shooting formulation with controls as
decision variables and the measured initial state \(\bar x\) as a parameter:

\[
\begin{aligned}
\min_U \quad & J(\bar x,U) \\
\text{subject to}\quad
& x_0 = \bar x, \\
& x_{k+1}=f(x_k,u_k), && k=0,\ldots,N-1, \\
& x_N=0, \\
& -2 \le u_k \le 2, && k=0,\ldots,N-1.
\end{aligned}
\]

The recurrence is already built into `rollout`. `sc.problem` adds the objective,
terminal equality, and control bounds:

```python
@sc.problem(vars=sc.arg("us", N), params=sc.arg("x0", 2))
def control_problem(us: sc.Expr, x0: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    xN, cost = rollout(x0, us)
    return sc.ProblemSpec(
        minimize=cost,
        eq=(xN,),
        lb=sc.const(-2.0),
        ub=sc.const(2.0),
    )


solve = sc.solver(control_problem, "ipopt", options={"print_level": 0})
```

Inside `control_problem`, `us` and `x0` are symbolic `sc.Expr` inputs. The cost,
constraint expressions, and bounds returned in `sc.ProblemSpec` are also
`sc.Expr` objects. The builder records their dependence on variables and
parameters without evaluating them numerically.

The decorator creates an `sc.Problem`, which describes the optimization problem
independently of the solver. `vars` declares what the solver may change, and
`params` declares what stays fixed during each solve. Expressions in `eq` are
constrained to zero. The scalar `lb` and `ub` bounds apply to every control.

`sc.solver` selects a backend and returns an `sc.Solver`. Scaly constructs
the objective, constraint, and derivative functions that backend needs.
The numerical call takes the problem parameters:

```python
us_opt, lam_box, lam_eq, lam_ineq = solve(x0)
status = solve.stats().to_solver_status()
print(status.name)  # OK
assert status.ok

xN_opt, cost_opt = rollout(x0, us_opt)
print(np.abs(xN_opt).max() < 1e-6)  # True
```

The result contains the controls and multipliers for variable bounds, equality
constraints, and inequality constraints. `lam_ineq` is empty here because the
problem has only variable bounds and equalities. Solver status is separate
from these arrays and indicates whether the solve succeeded.

Initial variables and multipliers default to zero. The keyword `x0` names the
initial decision variables, not the state, so `solve(x0, x0=us0)` starts from
the control sequence `us0`. `warm=` accepts a previous result for a warm start.
The [solver guide](solvers.md) describes backend-specific warm-start
behavior and the underlying `solve.function`, an `sc.Function` with explicit
inputs for the initial variables and multipliers.

## Generated C for deployment

The same model can be used from Python during development and exported for
integration into a C or C++ application. For example, a ROS node can evaluate
the dynamics or run an optimization solver without calling Python. Standalone
model code can also be compiled with the toolchain used by an embedded target.

```python
from pathlib import Path
from scaly.codegen import write_module

write_module(model, Path("generated"))
```

This writes `generated/model.c` and `generated/model.h`. Passing `solve` instead
exports the solver and its model calculations. That code also needs the native
IPOPT libraries. Exporting files ahead of the application build is the
ahead-of-time (AOT) path, in contrast to just-in-time (JIT) compilation on the first Python call.

The [code generation guide](codegen.md) describes export options, compilation,
and the generated C and C++ APIs, including argument buffers, working memory,
and [sparse output layouts](codegen.md#sparse-output-patterns).

## Bonus: repeated stages with `vmap`

`sc.vmap` records repeated independent calls to a function as one mapped
operation. The compiler can then emit a loop instead of a separate call site
for every stage. A multiple-shooting formulation makes this useful for the
same control problem by including states among the decision variables:

\[
\begin{aligned}
\min_{X,U}\quad & \sum_{k=0}^{N-1}\left(\lVert x_k\rVert^2+0.1u_k^2\right)
                  + 10\lVert x_N\rVert^2 \\
\text{subject to}\quad
& x_0=\bar x, \qquad x_N=0, \\
& f(x_k,u_k)-x_{k+1}=0, && k=0,\ldots,N-1, \\
& -2\le u_k\le2, && k=0,\ldots,N-1.
\end{aligned}
\]

Each dynamics residual, or *defect*, now takes candidate states as inputs.
Its evaluation does not depend on the result of another defect evaluation:

```python
@sc.function(
    sc.arg("x", 2), sc.arg("u", 1), sc.arg("xnext", 2),
    outputs=sc.arg("defect"),
)
def defect(x: sc.Expr, u: sc.Expr, xnext: sc.Expr) -> sc.Expr:
    return model(x, u) - xnext


@sc.problem(vars=sc.arg("w", 3 * N + 2), params=sc.arg("x0", 2))
def multiple_shooting(w: sc.Expr, x0: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    states = w[: 2 * (N + 1)].reshape((N + 1, 2))
    controls = w[2 * (N + 1) :]
    defects = sc.vmap(defect, N)(states[:-1], controls.reshape((N, 1)), states[1:]).vec()
    return sc.ProblemSpec(
        minimize=sc.sumsqr(states[:-1]) + 0.1 * sc.sumsqr(controls)
                 + 10.0 * sc.sumsqr(states[-1]),
        eq=(states[0] - x0, defects, states[-1]),
        ineq=(sc.bounded(controls, lo=-2.0, hi=2.0),),
    )
```

`w` stacks `N + 1` two-element states followed by `N` controls. Reshaping the
states to `(N + 1, 2)` and the controls to `(N, 1)` puts the stage index on the
leading axis, which is the axis `vmap` maps over. The mapped call passes the
three arguments of `defect` separately, and call `k` receives row `k` of each.
`states[:-1]` supplies stages `0` through `N - 1`, and `states[1:]` supplies
stages `1` through `N`. `sc.bounded` expresses the control
limits as inequalities within the larger decision vector.

The difference in generated structure is roughly as follows. These are sketches
with simplified signatures, before any inlining or other compiler optimization:

```c
// Single shooting: the Python loop creates N call sites.
model(x0, u0, x1);
model(x1, u1, x2);
/* ... */
model(x19, u19, x20);
```

```c
// Multiple shooting: vmap keeps one loop body.
for (int k = 0; k < N; ++k) {
    defect(states[k], controls[k], states[k + 1], defects[k]);
}
```

The mapped stage code stays one loop body as the horizon grows, including in
its derivatives. Numerical work and storage still grow with `N`, as can
[sparsity tables in the generated header](codegen.md#sparse-output-patterns).
`vmap` applies when calls can be evaluated independently, so it cannot replace
the sequential recurrence in `rollout`.
The [functions guide](functions.md#regular-repetition-vmap) covers inputs shared
by every stage and overlapping windows.
