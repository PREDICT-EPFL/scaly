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
| `"dopri5"` (Dormand-Prince) | 7 | 5 |

The two embedded pairs step with their higher-order solution; the last stage, which only feeds the
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
stage.

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
step found, the implicit function theorem gives `dK = -G_K^{-1} (G_x dx + G_u du + ...)`, with one
factorization of `G_K` shared by every direction; in a Jacobian, the factorization runs once and
each column is one solve. Reverse mode is one transposed solve. Second derivatives are implicit
too, so a solver's Lagrangian Hessian through the map is exact at the stages found. A third
derivative would reach the iterations, and the factorization there refuses it.

## What the generated code does

Every stage calls the model once, as a call node, so the model is built, generated and
differentiated once however many stages and substeps there are. The step spells its coefficients
as a person would: a numerical interval folds into the weights, zero weights drop out, and the most
frequent weight is factored out, so RK4 becomes `x + h/6 (k1 + 2 k2 + 2 k3 + k4)`. A shooting defect
built on `si.rk4` generates the same C as the hand-written RK4 of `examples/nmpc_cartpole.py`.
