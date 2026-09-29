# Model predictive control

`scaly.mpc` builds an optimal control problem over a horizon from a model, costs and constraints,
solves it with a solver backend, and turns it into a control law: one `Function` that takes the
measured state and returns the control, the solver and the warm start included, which generates C
like any other.

## An optimal control problem

```python
import numpy as np
import scaly as sc
from scaly import integrators as si, mpc


@sc.function(4, 1, output="xdot")
def cartpole(x, u):
  ...


ocp = mpc.OCP(
  ode=cartpole, dt=0.05, horizon=40,
  transcription=si.MultipleShooting(si.RK4(steps=2)),
  stage_cost=mpc.Quadratic(np.diag([2, 20, 0.1, 0.1]), 0.02 * np.eye(1)),
  terminal_cost=mpc.Quadratic(100 * np.eye(4)),
  u_bounds=(-15.0, 15.0),
  constraints=[mpc.Path(position, lo=-1.0, hi=1.0, soft=1e3)],
)
```

The dynamics are either `ode=f`, a continuous-time model `f(x, u, *params) -> xdot` with an interval
`dt` and a transcription, or `step=F`, a discrete-time map `F(x, u, *params) -> x_next`. The
transcription is any of those in [Integrators](integrators.md#transcriptions): multiple shooting
with any integrator (RK4 by default), local collocation, or pseudospectral segments.

The variables are the states at the grid points `x_0 .. x_N`, the controls `u_0 .. u_{N-1}`, the
transcription's own variables per interval, and one slack per soft constraint row. The initial
state is a parameter, `x_0 = x0` an equality.

## Costs

A stage cost is a Function `l(x, u, *params)` with one value, or `mpc.Quadratic(Q, R, x_ref, u_ref)`;
a terminal cost is `Vf(x, *params)` or `mpc.Quadratic(P, x_ref=...)`. A reference is an array, or the
name of a parameter of the right size, which the OCP then takes.

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
- `mpc.Path(g, lo, hi)` constrains `g(x_k, u_k, *params)` at every stage `k < N`. With `soft=w`, each
  row gets a slack, zero at the least, and the cost `w` times their sum: an exact penalty, so the
  constraint holds whenever it can, and the problem stays feasible when the plant leaves the model's
  reach.
- `terminal=mpc.TerminalEquality(x_ref)` requires `x_N = x_ref`.

## Parameters

A Function's parameters are its inputs after the state and the control (after the state for a
terminal cost), one vector each, matched by name across Functions. A model taking `mass` and a stage
cost `mpc.Quadratic(..., x_ref="r")` give the OCP the parameters `mass` and `r`, listed in
`ocp.params`. `varying=("r",)` makes `r` take `N + 1` values, one per grid point: stage `k` reads the
`k`-th, the terminal cost the last. A trajectory to track is a varying reference.

## The controller

```python
controller = mpc.MPC(ocp, "ipopt", options={"tol": 1e-8})
u = controller(x, r=reference)          # the first control, warm-started from the last solve
solution = controller.solve(x, r=reference)   # xs, us, zs, slack, times, cost, status
run = mpc.simulate(controller, plant, x0, steps=200, r=reference)
```

| Problem | Solver | Warm start |
| --- | --- | --- |
| linear dynamics, quadratic costs, polytopic constraints | `"piqp"`, with the QP proved and extracted as in [Solvers](solvers.md) | none |
| nonlinear | `"ipopt"` | primal; multipliers with `warm_start_init_point` |
| nonlinear | `"sqp"` | primal and multipliers |

Each call starts from the last solution moved up one interval: every stage block of the states, the
controls, the internal variables, the slacks and their multipliers takes the next stage's value, the
last one repeated. `controller.reset()` forgets it, and `controller.initial_guess(x0, u)` is the cold
start. `controller.status` is the last solve's status.

`controller.law` is the controller as one Function, `law(x0, *params, guess) -> (u, guess_next)`,
with the solver nested in it and the shift done in its code. `scaly.codegen.write_module(
controller.law, directory)` writes it as C; a C caller keeps one array of `controller.guess_size`
doubles between calls, and reads the solver's statistics through `<solver>_stats()`.

`mpc.simulate(controller, plant, x0, steps, **params)` runs the closed loop on `plant(x, u) -> x_next`,
for instance an `si.adaptive` map of the true model, and a parameter may be a function of the step.
