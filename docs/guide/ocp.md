# Optimal control problems

`sc.DiscreteOCP` states a discrete-time optimal control problem by the Functions of one stage: the
dynamics, the costs, the constraints and the bounds. It writes the horizon out as an ordinary
`sc.Problem`, which every solver backend in [Solvers](solvers.md) accepts, so the multiple-shooting
bookkeeping of the [getting-started example](getting_started.md#bonus-repeated-stages-with-vmap) is
done once, by the library.

## The problem

Over a horizon of \(N\) stages, with states \(x_k\in\mathbb{R}^{n_x}\) and controls
\(u_k\in\mathbb{R}^{n_u}\), the builder states

\[
\begin{aligned}
\min_{x_0,u_0,\ldots,u_{N-1},x_N}\quad & \sum_{k=0}^{N-1} \ell(x_k,u_k) + V_f(x_N) \\
\text{subject to}\quad
& x_0=\bar x, \qquad x_{k+1}=F(x_k,u_k), && k=0,\ldots,N-1, \\
& g_{\mathrm{lb}}\le g(x_k,u_k)\le g_{\mathrm{ub}}, \qquad u_{\mathrm{lb}}\le u_k\le u_{\mathrm{ub}}, && k=0,\ldots,N-1, \\
& x_{\mathrm{lb}}\le x_k\le x_{\mathrm{ub}}, && k=1,\ldots,N, \\
& h_{\mathrm{lb}}\le h(x_N)\le h_{\mathrm{ub}}.
\end{aligned}
\]

\(F\) maps one state and control to the next, so a continuous-time model enters through an
integration step written as a Function. The measured initial state \(\bar x\) is a parameter of
every solve. Each Function may also read parameters and global decision variables, which the
sections below introduce.

A pendulum swing-up shows the parts. The step is one explicit Euler step of the pendulum, whose
length is a parameter:

```python
import numpy as np
import scaly as sc

N, dt = 60, 0.05
upright = np.array([np.pi, 0.0])


@sc.function(sc.arg("x", 2), sc.arg("u", 1), sc.arg("length", ()), outputs=sc.arg("x_next", 2))
def pendulum(x: sc.Expr, u: sc.Expr, length: sc.Expr) -> sc.Expr:
    angle, rate = x[0], x[1]
    return x + dt * sc.stack([rate, -9.81 / length * angle.sin() + u[0]])


@sc.function(sc.arg("x", 2), sc.arg("u", 1), outputs=sc.arg("l", ()))
def effort(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    return sc.sumsqr(x - upright) + 0.01 * sc.sumsqr(u)


@sc.function(sc.arg("x", 2), outputs=sc.arg("vf", ()))
def arrival(x: sc.Expr) -> sc.Expr:
    return 1000.0 * sc.sumsqr(x - upright)


swing_up = sc.DiscreteOCP(
    N=N,
    dynamics=pendulum,
    stage_cost=effort,
    terminal_cost=arrival,
    u_bounds=(-8.0, 8.0),
    name="swing_up",
)
```

The dynamics take the state, then the control, and return the next state. The stage cost takes the
same two and the terminal cost the state alone, and each returns one value. The builder reads the
sizes \(n_x=2\) and \(n_u=1\) from the dynamics and checks the other Functions against them.

## Solving it

`swing_up.problem()` writes the horizon out as an `sc.Problem`:

```python
problem = swing_up.problem()
print(problem.vars.names, problem.vars.shapes)  # ('z', 'x_N') ((60, 3), (2,))
print(problem.params.names)                     # ('initial_state', 'length')
```

Row \(k\) of `z` holds \([x_k, u_k]\), and `x_N` holds the last state. The variables are ordered
stage by stage, \(x_0,u_0,x_1,u_1,\ldots,x_N\), so the constraint Jacobian and the Lagrangian
Hessian are banded. The parameters are the initial state \(\bar x\), then the parameters the
Functions name, here the pendulum's length.

`pack` and `unpack` convert between trajectories and these variables. `pack` broadcasts its
arguments, so `pack(x_start, 0.0)` is an initial guess that holds the initial state with zero
controls:

```python
solve = sc.solver(problem, "ipopt", options={"print_level": 0, "sb": "yes"})
x_start = np.zeros(2)
variables, *_ = solve((x_start, np.array(1.0)), x0=swing_up.pack(x_start, 0.0))
assert solve.stats().to_solver_status().ok

xs, us = swing_up.unpack(variables)
print(xs.shape, us.shape)    # (61, 2) (60, 1)
print(np.round(xs[-1], 3))   # [3.135 0.   ]
```

The solver is the one [Solvers](solvers.md) describes, so its multipliers, warm starts and
statistics behave the same. The keyword `x0` names the initial guess for the variables, while the
initial state is the first parameter.

## Constraints and bounds

`sc.StageConstraint(g, lo, hi)` bounds the output of a Function. In `path`, the Function takes the
state and the control and the constraint holds at every stage before the last. In `terminal`, it
takes the state and holds at the last stage. A side is `None`, a number or an array of the output's
size, and equal sides make an equality. Here the rate is limited along the way and the swing must
end upright and at rest:

```python
@sc.function(sc.arg("x", 2), outputs=sc.arg("error", 2))
def at_top(x: sc.Expr) -> sc.Expr:
    return x - upright


@sc.function(sc.arg("x", 2), sc.arg("u", 1), outputs=sc.arg("rate", ()))
def rate(x: sc.Expr, u: sc.Expr) -> sc.Expr:
    return x[1]


constrained = sc.DiscreteOCP(
    N=N,
    dynamics=pendulum,
    stage_cost=effort,
    path=[sc.StageConstraint(rate, lo=-6.0, hi=6.0)],
    terminal=[sc.StageConstraint(at_top, lo=0.0, hi=0.0)],
    u_bounds=(-8.0, 8.0),
    name="constrained_swing_up",
)
problem = constrained.problem()
print(problem.n_eq, problem.n_ineq)  # 122 62
```

The \(2 + 60\cdot 2\) equalities are the initial state and the dynamics. The inequalities are the
60 path rows and the two terminal ones, in stage order, and their multipliers follow that order.
Solved like the first problem, the last state is \([\pi, 0]\) to the solver's tolerance.

`u_bounds` and `x_bounds` are `(lo, hi)` pairs, each side `None`, a number or an array. They become
bounds on the variables. The control bounds hold at every stage. The state bounds hold from
\(x_1\) on, since \(x_0\) is fixed by the measurement.

!!! note "Path constraints at the first stage"
    A path constraint holds at stage 0 too, where \(x_0\) is the measured state. A constraint that
    reads only the state and that the measurement violates makes the problem infeasible, however
    the controls are chosen. Bound such a quantity with `x_bounds`, which leave \(x_0\) out.

## Parameters

Every input of a Function after the state and the control, or after the state for the terminal
Functions, is matched by its name across the Functions. A name the dynamics and the stage cost
both take is one parameter, and the builder checks that both declare the same shape. The problem
takes the parameters after `initial_state`, in the order the Functions first name them: the
dynamics, the stage cost, the terminal cost, the path constraints, then the terminal ones.

A name listed in `stage_params` takes a value per stage, with \(N + 1\) rows. Stage \(k < N\)
reads row \(k\) and the terminal Functions read the last row, which suits a reference trajectory:

```python
@sc.function(sc.arg("x", 2), sc.arg("u", 1), sc.arg("angle_ref", ()), outputs=sc.arg("l", ()))
def follow(x: sc.Expr, u: sc.Expr, angle_ref: sc.Expr) -> sc.Expr:
    return (x[0] - angle_ref) ** 2 + 0.01 * sc.sumsqr(u)


@sc.function(sc.arg("x", 2), sc.arg("angle_ref", ()), outputs=sc.arg("vf", ()))
def settle(x: sc.Expr, angle_ref: sc.Expr) -> sc.Expr:
    return 10.0 * (x[0] - angle_ref) ** 2


tracking = sc.DiscreteOCP(
    N=N,
    dynamics=pendulum,
    stage_cost=follow,
    terminal_cost=settle,
    u_bounds=(-8.0, 8.0),
    stage_params=["angle_ref"],
    name="tracking",
)
problem = tracking.problem()
print(problem.params.names, problem.params.shapes)
# ('initial_state', 'length', 'angle_ref') ((2,), (), (61,))

angle_ref = 0.5 * np.sin(np.linspace(0.0, np.pi, N + 1))
solve = sc.solver(problem, "ipopt", options={"print_level": 0, "sb": "yes"})
variables, *_ = solve((x_start, np.array(1.0), angle_ref), x0=tracking.pack(x_start, 0.0))
```

Inside the problem, the stage Functions are called through [`vmap`](functions.md#regular-repetition-vmap):
a parameter with one value reaches every stage whole, and a stage parameter is sliced along its
first axis. The generated code evaluates each Function in a loop over the stages rather than one
call site per stage.

## Global variables and a free final time

A name listed in `global_vars` is a decision variable that every stage shares, such as the
duration of a manoeuvre. The fastest way to move a cart from rest at 0 to rest at 1 with
\(|u|\le1\) takes the duration \(T\) as such a variable, scales the step by \(T/N\) and minimizes
\(T\) itself:

```python
K = 20


@sc.function(sc.arg("x", 2), sc.arg("u", 1), sc.arg("T", ()), outputs=sc.arg("x_next", 2))
def cart(x: sc.Expr, u: sc.Expr, T: sc.Expr) -> sc.Expr:
    return x + T / K * sc.stack([x[1], u[0]])


@sc.function(sc.arg("x", 2), sc.arg("T", ()), outputs=sc.arg("time", ()))
def duration(x: sc.Expr, T: sc.Expr) -> sc.Expr:
    return T


@sc.function(sc.arg("x", 2), outputs=sc.arg("error", 2))
def at_rest_at_one(x: sc.Expr) -> sc.Expr:
    return x - sc.const(np.array([1.0, 0.0]))


fastest = sc.DiscreteOCP(
    N=K,
    dynamics=cart,
    terminal_cost=duration,
    terminal=[
        sc.StageConstraint(at_rest_at_one, lo=0.0, hi=0.0),
        sc.StageConstraint(duration, lo=0.1),
    ],
    u_bounds=(-1.0, 1.0),
    global_vars=["T"],
    name="fastest",
)
```

`duration` serves twice, as the terminal cost and, with a lower bound, to keep the duration
positive. A global variable has no bounds of its own, so a constraint like this one bounds it.

## The stage form

`problem()` keeps a global variable as one variable after the trajectory. Every stage depends on
it, which a sparse solver such as IPOPT or Scaly's SQP handles well. `stage_problem()` states the
same problem in a form where every stage depends on its own variables only. Each global variable
becomes part of the state, held constant by the dynamics, \(T_{k+1}=T_k\):

```python
general, staged = fastest.problem(), fastest.stage_problem()
print(general.vars.names, general.vars.shapes)  # ('z', 'x_N', 'T') ((20, 3), (2,), ())
print(staged.vars.names, staged.vars.shapes)    # ('z', 'x_N') ((20, 4), (3,))
```

The state of the stage form is \([x_k, T_k]\), so row \(k\) of `z` holds \([x_k, T_k, u_k]\). It
is the form that solvers built on a Riccati recursion over the stages require, and the problem
records the sizes of its stages for them. Without global variables the two forms are the same
problem. Both are ordinary problems and reach the same solution:

```python
options = {"print_level": 0, "sb": "yes"}
start = np.zeros(2)
shared, *_ = sc.solver(general, "ipopt", options=options)(start, x0=fastest.pack(start, 0.0, 3.0))
carried, *_ = sc.solver(staged, "ipopt", options=options)(
    start, x0=fastest.pack(start, 0.0, 3.0, stage_form=True)
)
_, _, T_shared = fastest.unpack(shared)
_, _, T_carried = fastest.unpack(carried, stage_form=True)
print(round(float(T_shared), 4), round(float(T_carried), 4))  # 2.0 2.0
```

`pack` takes one value per global variable after the controls, and `unpack` returns them after the
trajectory. With `stage_form=True`, both convert the variables of `stage_problem()`, and `unpack`
reads each global variable from the last stage.

## Limits

The builder states discrete-time problems whose state and control sizes are the same at every
stage. A continuous-time model enters through an integration step in the dynamics, as the
pendulum's Euler step does above. Every constraint is hard.
