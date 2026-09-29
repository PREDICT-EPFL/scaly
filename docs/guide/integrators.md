# Integrators

`scaly.integrators` turns a continuous-time model into a discrete-time map: a `Function` that takes
the state at the start of an interval and returns the state at its end. The map is an ordinary
graph, so it composes into a shooting constraint, differentiates, and generates C like anything
else.

## The model

A model is a `Function` whose first parameter is the state, one vector, and whose one output is
the state's derivative, shaped like it. Every other parameter (inputs, physical parameters,
references) is held fixed over an interval, which gives a zero-order hold on the inputs.

```python
import numpy as np
import scaly as sc
from scaly import integrators as si


@sc.function(4, 1, output="xdot")
def cartpole(x, u):
  ...


step = si.rk4(cartpole, dt=0.05)            # step(x, u) -> xnext
step(np.zeros(4), np.array([1.0]))
```

The map takes the model's own parameters, in order and under the same names. Its output is the
state's name followed by `next` (`xnext`, or `znext` for a state named `z`). A template model gives
a template map, one instance per argument signature.

## Methods

Every named method is also a class, an integrator method with its options, found by name in
`si.REGISTRY` as `integrators.<name>`. `si.solver(si.ODE(f, dt=...), method)` builds the same map
as the shorthand the sections below describe; a method is what a transcription takes, and what
composes with the methods of other domains.

```python
ode = si.ODE(cartpole, dt=0.05)
step = si.solver(ode, si.RK4(steps=2))                          # = si.rk4(cartpole, dt=0.05, steps=2)
stiff = si.solver(ode, si.RadauIIA(3, newton=sc.roots.Newton(tol=1e-10, rtol=1e-10)))
shooting = sc.ocp.MultipleShooting(si.Tsit5())
```

| Methods | Options | Shorthand |
| --- | --- | --- |
| `si.Euler`, `si.Heun`, `si.Midpoint`, `si.Ralston`, `si.RK3`, `si.SSPRK3`, `si.RK4`, `si.RK38`, `si.BS32`, `si.DOPRI5`, `si.Tsit5` | `steps` | `si.explicit(f, "rk4", ...)` |
| `si.BackwardEuler`, `si.ImplicitMidpoint`, `si.Trapezoidal`, `si.SDIRK2`, `si.SDIRK3` | `steps`, `newton` | `si.implicit(f, "sdirk3", ...)` |
| `si.GaussLegendre(stages)`, `si.RadauIIA(stages)`, `si.LobattoIIIA(stages)`, `si.LobattoIIIC(stages)` | `stages`, `steps`, `newton` | `si.implicit(f, "radau_iia", stages=3, ...)` |
| `si.Adaptive(pair)`, `pair` one of `si.DOPRI5()`, `si.Tsit5()`, `si.BS32()` | `rtol`, `atol`, `max_steps`, `h0` | `si.adaptive(f, "dopri5", ...)` |
| `si.StormerVerlet`, `si.SymplecticEuler` | `split`, `steps` | `si.symplectic(f, "stormer_verlet", ...)` |

An implicit method's `newton` is an `sc.roots.Newton` ([Nonlinear equations](roots.md)): the default
`Newton(tol=None, max_iter=3, simplified=True)` is the shorthand's `newton_iters=3`;
`tol=...` with `rtol` equal to it is the shorthand's `tol`, and `simplified=False` its
`newton="full"`. The ODE's `dt=None` makes the interval an input, and its `name` names the maps
(`{name}_{method}`). A package adds a method by subclassing `si.ExplicitRK` or `si.ImplicitRK` with
its `name` and `table` and declaring it in the `scaly.methods` entry points.

## Explicit Runge-Kutta methods

`si.explicit(f, method, dt=..., steps=1, name=None)` builds the map of an explicit method, and
`si.rk4` is the classical one.

| `method` | Stages | Order |
| --- | --- | --- |
| `"euler"` | 1 | 1 |
| `"heun"`, `"midpoint"`, `"ralston"` | 2 | 2 |
| `"rk3"` (Kutta), `"ssprk3"` (strong-stability preserving) | 3 | 3 |
| `"rk4"`, `"rk38"` (the 3/8 rule) | 4 | 4 |
| `"bs32"` (Bogacki-Shampine) | 4 | 3 |
| `"dopri5"` (Dormand-Prince), `"tsit5"` (Tsitouras) | 7 | 5 |

The three embedded pairs step with their higher-order solution; the last stage, which only feeds the
error estimate, is never computed. A method may also be a `si.Tableau(a, b, c, order)` of your own,
whose declared order is checked against the order conditions when it is built.

- `dt=None` makes the interval a trailing input, `step(x, u, dt)`, for a free final time or a
  sampling time chosen at run time. Its derivative is available like any other.
- `steps=n` divides the interval into `n` equal substeps. Up to `si.UNROLL_STEPS` (4) are unrolled;
  more run as a `scan`, a loop in the generated C whose size does not depend on `n`.
- The default name is `{model}_{method}` (`cartpole_rk4`). Two maps of one model with different
  settings in one graph need distinct names, given with `name=`.

## Implicit Runge-Kutta methods

`si.implicit(f, method, stages=None, dt=..., steps=1, newton_iters=3, tol=None, max_iter=20,
newton="simplified", name=None)` builds the map of an implicit method, for stiff models and for the
accuracy per stage that collocation methods give.

| `method` | Stages | Order | Stability |
| --- | --- | --- | --- |
| `"gauss_legendre"`, with `stages=s` | s | 2s | A-stable, symplectic |
| `"radau_iia"`, with `stages=s` | s | 2s - 1 | L-stable, stiffly accurate |
| `"lobatto_iiia"`, with `stages=s >= 2` | s | 2s - 2 | A-stable, stiffly accurate, first stage explicit |
| `"lobatto_iiic"`, with `stages=s >= 2` | s | 2s - 2 | L-stable, stiffly accurate |
| `"backward_euler"`, `"implicit_midpoint"`, `"trapezoidal"` | 1, 1, 2 | 1, 2, 2 | L, A, A |
| `"sdirk2"`, `"sdirk3"` (Alexander) | 2, 3 | 2, 3 | L-stable, diagonally implicit |

The four families are built from their nodes (`si.radau_iia(3)` is the tableau), and every one is
checked against the order conditions when it is built.

```python
step = si.implicit(cartpole, "radau_iia", stages=3, dt=0.05)            # three Newton iterations
step = si.implicit(cartpole, "sdirk3", dt=0.05, tol=1e-12, max_iter=20)  # to a tolerance
```

**Newton.** The stage equations `G(K) = K - f(x + h (A ⊗ I) K) = 0` are solved from `f(x)` at every
stage, by `sc.roots.Newton` ([Nonlinear equations](roots.md)) with a solve for the stage matrix.

- `newton_iters` fixes the number of iterations, the choice for control, where every call should
  take the same time.
- `tol` iterates in a `while_loop` until the residual is below `tol * (1 + |K|)`, at most `max_iter`
  times.
- `newton="simplified"` factors `I - h A ⊗ J` once per step, `J` the model's Jacobian at the start
  of the step. For the coupled families it splits that matrix by the eigenvalues of `A` into one
  system of the state's size per real eigenvalue and one of twice it per complex pair, as RADAU5
  does. Radau IIA with three stages then factors one system of order `n` and one of `2n`, not one
  of `3n`.
- `newton="full"` refactors the exact stage Jacobian at every iteration.
- A diagonally implicit method solves its stages one after another, each a system of the state's
  size, and simplified Newton factors one matrix for all of them.

**Derivatives.** The derivative of the map does not go through the iterations. At the stages the
step found, the implicit function theorem (`sc.roots.custom_root`) gives `dK = -G_K^{-1} (G_x dx + G_u du + ...)`, with one
factorization of `G_K` shared by every direction; in a Jacobian, the factorization runs once and
each column is one solve. Reverse mode is one transposed solve. Second derivatives are implicit
too, so a solver's Lagrangian Hessian through the map is exact at the stages found. A third
derivative would reach the iterations, and the factorization there refuses it.

## Adaptive steps

`si.adaptive(f, pair="dopri5", dt=None, rtol=1e-6, atol=1e-9, max_steps=10000, h0=None)` chooses
its steps as it integrates, as `solve_ivp` does, with the Dormand-Prince 5(4) or Bogacki-Shampine
3(2) pair. It is the map for a plant model in closed-loop simulation, where the accuracy should not
depend on the sampling time.

```python
plant = si.adaptive(cartpole, rtol=1e-10, atol=1e-12)     # plant(x, u, dt) -> x after dt
x = plant(x, u, np.array(0.05))
```

- A step is accepted when the difference between the pair's two solutions, divided by
  `atol + rtol * max(|x|, |x_next|)` entry by entry, has a root mean square of at most 1. The next
  step is `0.9 err^(-1/(q+1))` times the last, within `[0.2, 5]`.
- The loop is a `while_loop` of at most `max_steps` accepted and rejected steps. When it ends before
  the interval does, the result is NaN rather than a state short of the end.
- Derivatives take the steps as they were chosen: the controller's factor goes through an integer,
  which carries no derivative, so a derivative is that of a fixed-step method on the same steps, and
  the derivative in `dt` is the last step's.

## Symplectic methods

`si.symplectic(f, "stormer_verlet", split=nq, dt=..., steps=1)` integrates a model whose state is
positions then velocities, `x = [q, v]` with `q = x[:nq]`. Störmer-Verlet (order 2) is a half kick, a
drift and a half kick; `"symplectic_euler"` (order 1) is a kick and a drift. When `q'` depends only on
the velocities and `v'` only on the positions and the held inputs, both are symplectic: over long
simulations the energy error stays bounded where a Runge-Kutta method's accumulates.

## Linear systems

```python
Ad, Bd = si.zoh(A, B, dt)            # exact for x' = A x + B u with u held over the interval
Ad, B0, B1 = si.foh(A, B, dt)        # exact with u ramped from u_k to u_{k+1}
A, B = si.linearize(cartpole, x_eq, u_eq)      # Jacobians of a model (or a map) at a point
```

`zoh` and `foh` read their matrices off one matrix exponential (SciPy's `expm`), so they take numbers,
not expressions. `linearize` evaluates the Jacobians of any Function's one output as generated code,
one matrix per input leaf; linearizing a model and then `zoh` gives the exact discretization of the
linearization, and linearizing a map built by `explicit` or `implicit` gives the linearization of that
discretization.

## Transcriptions

A transcription says how one interval of an optimal control horizon becomes variables, equality
constraints and a cost. It is what `sc.ocp.transcribe` builds a discrete OCP from, one interval
Function mapped over the intervals, and it lives in `scaly.ocp` beside it.

```python
interval = sc.ocp.Collocation(3, "radau").interval(cartpole, running_cost, dt=0.05)
interval.fn           # (x, u, z, xnext, *params) -> [residuals; cost]
interval.n_internal   # the size of z: this interval's own variables
interval.guess(x, u)  # a starting point for z
```

The model is `f(x, u, *params)`, the control its second input; the running cost `l(x, u, *params)`
takes the same inputs and gives one value, integrated over the interval.

| Transcription | Interval variables `z` | Residuals | Control |
| --- | --- | --- | --- |
| `sc.ocp.MultipleShooting(si.RK4(steps=2))`, or any integrator method or its name | none | `F(x, u) - xnext` | held |
| `sc.ocp.Collocation(degree, "radau")` | `degree - 1` states | `degree` collocation conditions | held |
| `sc.ocp.Collocation(degree, "legendre")` | `degree` states | `degree` conditions and continuity | held |
| `sc.ocp.Pseudospectral(nodes)` | `nodes - 1` states and `nodes - 1` controls | `nodes` conditions | at each node |

- Multiple shooting integrates the running cost with the same method as one more state, so the step
  and its cost are one call.
- Collocation at Radau or Gauss points is, at a fixed control, the Radau IIA or Gauss-Legendre
  step of the same degree; its cost is the matching quadrature.
- `Pseudospectral` is Radau pseudospectral collocation, as in GPOPS-II: the states and the controls
  at Legendre-Gauss-Radau points with 0 among them, the end an extra node. One interval over the
  whole horizon is the global method, several are segments. For a smooth solution the accuracy grows
  faster than any fixed order as `nodes` does. The node at 0 carries the control a receding horizon
  applies; the Gauss and Lobatto schemes are not offered, because the first has no node there and the
  second collocates at both ends, one condition more than its unknowns at a fixed control.
- `dt=None` makes the interval length the last input, after the parameters.
- `si.lgl(n)` gives the Legendre-Gauss-Lobatto rule of `n` intervals on `[-1, 1]` (its nodes,
  quadrature weights and differentiation matrix), for pseudospectral code of your own.

The derivatives of a horizon of intervals follow the stage structure: when the variables are
slices of one vector, as an NLP's are, each interval's Jacobian is computed once per interval in one
map, colored per variable block, never as a whole-horizon coloring.

## What the generated code does

Every stage calls the model once, as a call node, so the model is built, generated and
differentiated once however many stages and substeps there are. The step spells its coefficients
as a person would: a numerical interval folds into the weights, zero weights drop out, and the most
frequent weight is factored out, so RK4 becomes `x + h/6 (k1 + 2 k2 + 2 k3 + k4)`. A shooting defect
built on `si.rk4` generates the same C as the hand-written RK4 of `examples/nmpc_cartpole.py`.
