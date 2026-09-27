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

## What the generated code does

Every stage calls the model once, as a call node, so the model is built, generated and
differentiated once however many stages and substeps there are. The step spells its coefficients
as a person would: a numerical interval folds into the weights, zero weights drop out, and the most
frequent weight is factored out, so RK4 becomes `x + h/6 (k1 + 2 k2 + 2 k3 + k4)`. A shooting defect
built on `si.rk4` generates the same C as the hand-written RK4 of `examples/nmpc_cartpole.py`.
