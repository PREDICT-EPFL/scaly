# Optimal control

`scaly.ocp` states an optimal control problem over a horizon from a model, costs and constraints,
transcribes it from continuous to discrete time, and solves it by an OCP method. The solver is one
`Function`, `xs, us, point, info = solve(x0, *params, warm)`, which nests in a graph and generates C
like any other. Receding-horizon control is a loop you write around it: solve at the measured
state, apply the first control, move the point the solve reached up one stage with `ocp.shift`, and
start the next solve from it. The same three pieces composed in one `@sc.function` are the control
law, one C function.

## A continuous-time problem

```python
import numpy as np
import scaly as sc
from scaly import integrators as si, ocp


@sc.function(4, 1, output="xdot")
def cartpole(x, u):
  ...


continuous = ocp.ContinuousOCP(
  cartpole, T=2.0,
  stage_cost=ocp.Quadratic(np.diag([2, 20, 0.1, 0.1]), 0.02 * np.eye(1)),
  terminal_cost=ocp.Quadratic(100 * np.eye(4)),
  u_bounds=(-15.0, 15.0),
  constraints=[ocp.Path(position, lo=-1.0, hi=1.0, soft=1e3)],
)
problem = ocp.transcribe(continuous, ocp.MultipleShooting(si.RK4(steps=2)), N=40)
```

A `ContinuousOCP` is the model `f(x, u, *params) -> xdot` over a horizon of length `T`, with what
is to be optimized; it says nothing yet of how. `ocp.transcribe(continuous, transcription, N=...)`
cuts the horizon into `N` intervals of `T / N` and turns each into variables and equality residuals
by a transcription: multiple shooting with any integrator (`MultipleShooting(RK4())` by default),
local collocation, or pseudospectral segments, as in [Integrators](integrators.md#transcriptions).
The result is a `DiscreteOCP`.

## A discrete-time problem

`ocp.DiscreteOCP(step=F, N=...)` takes a discrete-time map `F(x, u, *params) -> x_next` directly:
an integrator's map, `si.affine(A, B)` for a linear model, or `si.zoh`'s exact discretization of one.

```python
A, B = si.zoh(Ac, Bc, 0.1)
problem = ocp.DiscreteOCP(
  step=si.affine(A, B), N=20,
  stage_cost=ocp.Quadratic(Q, R), terminal_cost=ocp.Quadratic(P),
  u_bounds=(-1, 1), x_bounds=(x_lo, x_hi), terminal=ocp.max_invariant_set(A + B @ K, C),
)
```

A `DiscreteOCP` is the multistage form every method solves. Stage `k` has the state `x_k`, the
control `u_k` and the stage's own variables `w_k` (a collocation's internal states, say), dynamics
that tie them to `x_{k+1}`, a stage cost, path constraints and bounds; the last state has a terminal
cost and a terminal set. It keeps that structure explicit, `problem.stage`, `problem.params` and the
layout `ocp.to_problem` gives, for the methods that read it. The initial state is a parameter of
every solve, `x_0 = x0` an equality.

## Costs

A stage cost is a Function `l(x, u, *params)` with one value, or `ocp.Quadratic(Q, R, x_ref, u_ref)`;
a terminal cost is `Vf(x, *params)` or `ocp.Quadratic(P, x_ref=...)`. A reference is an array, or the
name of a parameter of the right size, which the problem then takes.

With a continuous-time model the running cost is `dt * sum_k l(x_k, u_k)` by default
(`cost="points"`), as at the shooting nodes of acados, or the transcription's integral of `l` over
each interval (`cost="integral"`): the integrator's own quadrature for shooting, the collocation
quadrature otherwise. Pseudospectral segments, whose controls vary inside an interval, take the
integral by default, since the points alone would leave those controls out of the cost. With a
discrete-time map the running cost is `sum_k l(x_k, u_k)`.

## Constraints

- `x_bounds=(lo, hi)` bounds every state but the initial one, which is data, internal states
  included. `u_bounds` bounds every control, the pseudospectral controls inside an interval
  included. A side is a number, an array, or `None`.
- `ocp.Path(g, lo, hi)` constrains `g(x_k, u_k, *params)` at every stage `k < N`. With `soft=w`, each
  row gets a slack, zero at the least, and the cost `w` times their sum: an exact penalty, so the
  constraint holds whenever it can, and the problem stays feasible when the plant leaves the model's
  reach.
- `terminal=ocp.TerminalEquality(x_ref)` requires `x_N = x_ref`; a `sc.sets.Polytope` or
  `sc.sets.Ellipsoid` requires `x_N` in it.

## Parameters

A Function's parameters are its inputs after the state and the control (after the state for a
terminal cost), one vector each, matched by name across Functions. A model taking `mass` and a stage
cost `ocp.Quadratic(..., x_ref="r")` give the problem the parameters `mass` and `r`, listed in
`problem.params`. `varying=("r",)` makes `r` take `N + 1` values, one per grid point: stage `k` reads
the `k`-th, the terminal cost the last. A trajectory to track is a varying reference.

## Solving

```python
method = ocp.Direct(sc.opt.IPOPT(options={"tol": 1e-8}))
solve = ocp.solver(problem, method)
xs, us, point, info = solve(x0, mass, r, ocp.initial_guess(problem, method, x0))
```

`ocp.solver(problem, method)` builds the Function that solves `problem` by an OCP method: its
inputs the initial state, the parameters in `problem.params` order (a varying one flat, `N + 1`
values) and the warm start; its outputs the states `(N + 1, nx)`, the controls `(N, nu)`, the point
the solve reached, and an `Info` of `status` (a `sc.Status` code), `iter`, `objective` and
`primal_residual`. The method is an OCP method with its options, its name, or `"auto"`.

`ocp.Direct(method, form="sparse")`, the direct method, formulates the problem as an optimization
problem (`ocp.to_problem`) and solves it with an `sc.opt` method:

| Problem | `sc.opt` method | Warm start |
| --- | --- | --- |
| linear dynamics, quadratic costs, polytopic constraints | `sc.opt.PIQP(sparse=True)`, with the QP proved and extracted as in [Solvers](solvers.md); `sc.opt.PIQP()` for the condensed form | none |
| nonlinear | `sc.opt.IPOPT()` | primal; multipliers with `warm_start_init_point` |
| nonlinear | `sc.opt.SQP()` | primal and multipliers |

The warm start is the whole primal-dual point, flat: the variables, their bounds' multipliers, the
equality and the inequality multipliers, `method.warm_size(problem)` doubles. `method.layout(problem)`
locates each variable leaf in it (`xs`, `us`, a transcription's `zs`, the soft constraints'
`slack`). `ocp.initial_guess(problem, method, x0, u)` is a cold start: every state `x0`, every
control `u` (zeros by default), a transcription's own variables from its `guess`, slacks and
multipliers zero. The solver's own statistics, its time among them, are
`sc.opt.solver_stats(solve)` after a call.

## The other methods

Every method builds the same Function, `xs, us, point, info = solve(x0, *params, warm)`, and has its
own warm start, which `ocp.initial_guess` and `ocp.shift` know. Each one refuses, with its reasons,
a problem it cannot take; `tests/ocp/test_conformance.py` checks each against the direct method on
every problem it takes.

| Method | Takes | Warm start |
| --- | --- | --- |
| `ocp.Direct(method, form)` | any `DiscreteOCP`, through an `sc.opt` method | the primal-dual point |
| `ocp.ILQR()` | a map, costs at the points, no constraints, bounds or terminal set | the controls |
| `ocp.TinyADMM(rho)` | an affine map without parameters, `Quadratic` costs, box bounds | the ADMM's state |
| `ocp.ALTRO()` (experimental) | a map, costs at the points, bounds, hard paths, a terminal equality or set | the controls |
| `ocp.SCvx()` (experimental) | a map, convex `Quadratic` costs, bounds, hard paths, a terminal equality or polytope | the reference `(X, U)` |

- **`ILQR`** is iterative LQR with the regularization and line search of Tassa et al.: a rollout
  `scan`, a backward Riccati `scan` on local models from AD, and a line search `while_loop` inside an
  outer `while_loop`, so the whole trajectory optimizer is one generated C function.
- **`TinyADMM`** is TinyMPC's ADMM: every bound gets a slack copy, so the primal step is an LQR whose
  gains are computed offline, and each iteration is a backward pass for the affine terms, a
  rollout, a clip and a dual update. Its gains come from the problem's own terminal cost
  (`scaly.ocp.tinyadmm.finite_cache`); `admm_solver` with `tinympc_cache` is the library's
  convention, which `examples/tinympc` reproduces to rounding.
- **`ALTRO`** is the augmented-Lagrangian iLQR of Howell et al. following Altro.jl 0.5, with the
  optional projected Newton phase (`projected_newton=True`, which takes diagonal `Quadratic`
  costs). `examples/case_studies/altro` checks it against Altro.jl's iterates.
- **`SCvx`** is sequential convex programming with a penalized trust region: each iteration
  linearizes the map and the paths about the reference and solves a QP (by default `sc.opt.IPM`)
  with virtual control and a proximal term.

## A receding horizon

```python
shift = ocp.shift(problem, method)
x, warm = x0, ocp.initial_guess(problem, method, x0)
for k in range(steps):
  _, us, point, info = solve(x, warm)
  warm = shift(point)
  x = plant(x, us[0])
```

`ocp.shift(problem, method)` is a Function, `point -> warm`: every stage block of the states, the
controls, the internal variables, the slacks and their multipliers takes the next stage's value,
the last one repeated. The plant is whatever the loop steps, for instance an `si.adaptive` map of
the true model. A parameter that changes with the step, a reference window, is passed at each solve.

## The control law as one Function

```python
@sc.function(sc.L("x0", nx), sc.L("warm", method.warm_size(problem)), output=sc.G("u", "warm_next"))
def law(x0, warm):
  _, us, point, _ = solve(x0, warm)
  return us[0], shift(point)
```

The solver and the shift nest like any Function, so the controller is one: `law(x0, *params, warm)
-> (u, warm_next)`. `scaly.codegen.write_module(law, directory)` writes it as C; a C caller keeps one
array of `warm_size` doubles between calls, and reads the solver's statistics through
`<solver>_stats()`.

## Sparse and condensed forms

`ocp.to_problem(problem, form)` is the formulation the direct method solves, an `sc.opt` problem
and its `Layout`. The sparse form keeps every state a variable and the dynamics equality rows; its
KKT matrix is banded. `form="condensed"` eliminates the states, `x_k` a `scan` of the model from
`x0`, for a discrete map or multiple shooting with costs at the points: the controls are the only
variables, the Hessian is dense, and every state bound becomes an inequality row. Both have one
solution; which is faster depends on the horizon and the constraints.

## Terminal ingredients

For a linear model, `ocp.lqr(A, B, Q, R)` returns the LQR gain `K` and the cost to go `P` (the
discrete algebraic Riccati equation). With the terminal cost `x'Px`, an unconstrained problem of any
horizon returns `u_0 = K x`. `ocp.max_invariant_set(A + B K, C)` is the maximal positively
invariant set of the closed loop inside a `sc.sets.Polytope` `C`, and `ocp.largest_ellipsoid(P, C)`
the largest invariant ellipsoid `x'Px <= alpha` inside it. Either as the terminal set, with `P` as the
terminal cost, keeps a problem that can be solved now solvable at the next sample, and its optimal
cost falling by at least the stage cost. The ellipsoid's row is quadratic, so a problem with it is
an NLP, which a QP method refuses with `NotQuadratic`.

The notebooks in `examples/ocp/` run each of these: five transcriptions of a swing-up, closed loops
against a mismatched plant with IPOPT and SQP, reference tracking with a preview, linear MPC against
the LQR, and terminal sets with their regions of attraction.
