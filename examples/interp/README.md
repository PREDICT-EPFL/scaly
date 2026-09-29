# Interpolation and splines

Notebooks for `scaly.interp` (`from scaly import interp`): lookup tables and interpolating splines,
fits to scattered data with and without shape constraints, and splines whose coefficients are
decision variables. The CasADi pairs in `pairs/` solve five of the problems in both libraries. Every
notebook ends with a cell of assertions against its references, and
`tests/integration/test_notebooks.py` runs each one top to bottom. They are saved with their outputs,
so they read on GitHub without running.

| Notebook | Problem | What it shows |
| --- | --- | --- |
| `interpolation_kinds.ipynb` | All ten kinds on Runge's function and on a staircase with plateaus | each kind against SciPy (or Steffen's paper, or CasADi's `smooth_linear`) to 2e-15; overshoot, monotonicity and the smoothness class measured across the joins; the five `extrap` modes, and CasADi's `bspline` zero outside its grid; derivatives to the fifth by AD and by `derivative()`; `integrate` with a symbolic limit; a thermistor read backwards by `inverse()`; the inverse written as C |
| `lookup_tables_nd.ipynb` | A heat pump's COP map, a motor's efficiency map, a wing's 3-D lift table | tensor-product tables against SciPy's `NdBSpline`, one kind per axis; the searches (`uniform`, `bucket`, `binary`, `count`) and the layouts (`pp`, `basis`) timed on batches of 10⁶ points; C size against layout; the C of a batch independent of its size; a table baked into C, and a table passed in by a C `main` and checked against the JIT |
| `learning_tables.ipynb` | A 12 x 12 table calibrated to 3 000 measurements; a Hammerstein model identified from 2 000 samples, *IPOPT* | `at()` at known points against NumPy's `lstsq` on `basis()`; the calibrated table as an input of compiled code; the Jacobian's pattern at known and at symbolic points (4 against 144 entries a row) and what it costs the solver; a spline with `Expr` coefficients inside a `scan`, against SciPy's `least_squares` and central differences |
| `shape_constrained.ipynb` | A monotone LFP battery OCV curve, a convex fuel-cost curve, a bounded 2-D efficiency map, *PIQP* | `interp.constrained` against least squares, PCHIP and Steffen through noisy data; the KKT conditions checked from outside the solver; state of charge from voltage by `inverse()`, refused for the unconstrained fit; agreement with `trust-constr` |
| `contouring_control.ipynb` | A closed-loop lap of an FSDS track by MPCC, *IPOPT* | the track as one periodic vector-valued cubic in arc length (centre, heading, curvature); the spline in a stage cost and a lateral-acceleration path constraint through `scaly.ocp`; a 34 s lap in 681 solves; the control law written as C |
| `spline_trajectories.ipynb` | A cart-pole swing-up with a spline force; a double integrator's path around obstacles, *IPOPT*, *PIQP* | a force of 13 coefficients read by `at()` against the piecewise-constant transcription; velocity and acceleration limits through `derivative()`; clearance guaranteed by Bézier convex hulls, as a QP |

```bash
uv run --with jupyterlab jupyter lab examples/interp
```

The plots use the shared style in `plotstyle.py`, beside them. Five of the notebooks write their C to
`examples/generated/interp/<name>/` (git-ignored); `shape_constrained` has none to write. The FSDS centre line in `data/` is the one the
race-car benchmark uses (`benchmarks/problems/race_cars/data/tracks/fsds_competition_1`).

## CasADi pairs

Each pair is `<name>_casadi.py` and `<name>_scaly.py`, following `examples/casadi`: `build()` does the
modelling and returns `run`, which returns a dict of arrays. `pairs/_data.py` generates the data both
halves use, from fixed seeds, and `pairs/_common.py` loads the helpers of `examples/casadi/_common.py`.

| Pair | Problem | CasADi | Scaly |
| --- | --- | --- | --- |
| `lut_eval` | a 1-D cubic on 1 000 non-uniform sites, a 64 x 64 bicubic with Hessians, a 20³ trilinear: values and gradients at 10 000 random points, every node, and (trilinear) 1 000 points outside | `interpolant` (`bspline`, `linear`) mapped, `lookup_mode` `binary` or `exact` | `interpolant` called on batches, gradients by reverse mode through the map |
| `table_calibration` | a 12 x 12 bilinear table fitted to 3 000 noisy measurements, IPOPT | parametric `interpolant` with `inline=True` | `interpolant` over an `Expr` table, `at()` at the measurement points |
| `hammerstein_sysid` | a cubic B-spline nonlinearity (20 coefficients) and second-order dynamics, 2 000 samples, single shooting, IPOPT | `bspline` node with MX coefficients, `inline=True`, `mapaccum` | `BSpline` with `Expr` coefficients inside a `scan` |
| `contouring_mpc` | MPCC around an FSDS track, a first solve and 200 closed-loop steps, IPOPT | `interpolant("bspline")` on the lap repeated three times, `Opti` | `interpolant(bc="periodic")`, `scaly.ocp` |
| `heat_pump_mpc` | a day of 15-minute heat-pump MPC with a bicubic COP table and hourly price and temperature forecasts, IPOPT | `interpolant("bspline")` mapped, `interp1d(..., "floor")` on `Opti` parameters | `interpolant(kind="cubic")` on a batch, `kind="zoh"` tables of parameters read by `at()` |

```bash
uv run python examples/interp/pairs/lut_eval_scaly.py         # either half on its own prints its results
uv run python examples/casadi/compare.py --dir examples/interp/pairs
```

`compare.py --dir examples/interp/pairs`, three fresh processes per variant, on an Apple M3 Max
(16 logical CPUs), Darwin 25.6, Python 3.14.3, CasADi 3.8.0, NumPy 2.4.6, Apple clang 21. Setup is
modelling plus the first run less a steady one: for Scaly that includes generating and compiling
its C. The JIT column is CasADi with its oracles compiled (`casadi_jit(expand=False)`). Run is the
median steady-state call.

| Pair | Agree (max rel. diff) | Iterations CasADi / Scaly | Code lines CasADi / Scaly | Setup CasADi · JIT · Scaly | Run CasADi · JIT · Scaly |
| --- | --- | --- | ---: | ---: | ---: |
| `contouring_mpc` | yes (4e-13, loop_us) | 15 / 15; 3641 / 3641 | 78 / 63 | 52.3 ms · 8.22 s · 3.34 s | 9.63 s · 1.59 s · 801.7 ms |
| `hammerstein_sysid` | yes (6e-16, f) | 13 / 13 | 23 / 27 | 1.68 s · 71.44 s · 3.85 s | 5.95 s · 205.8 ms · 15.9 ms |
| `heat_pump_mpc` | yes (1e-14, P) | 25 / 25 | 42 / 46 | 9.4 ms · 2.78 s · 913.4 ms | 12.2 ms · 11.6 ms · 4.8 ms |
| `lut_eval` | yes (2e-10, h_2d) | n/a | 49 / 45 | 25.9 ms · 4.51 s · 4.15 s | 44.3 ms · 27.7 ms · 819 µs |
| `table_calibration` | yes (2e-15, table) | 1 / 1 | 30 / 20 | 14.08 s · 90.76 s · 774.8 ms | 757.7 ms · 371.5 ms · 879 µs |

Every pair agrees, with the same iteration counts on both sides. The one gap above $10^{-13}$ is
`lut_eval`'s bicubic Hessian, $2 \times 10^{-10}$ relative. That is CasADi's fit, not the
evaluation: its coefficients miss SciPy's by rounding amplified by its solver, and a second
derivative on a 1/64 grid multiplies that by about 4 000. Scaly's Hessian is within
$2 \times 10^{-12}$ of SciPy's `NdBSpline`, CasADi's within $2 \times 10^{-9}$.

## What the examples ran into

- **CasADi's parametric interpolants have a zero derivative in their data**, and its `bspline` node
  a zero derivative in symbolic coefficients, unless built with `inline=True`. IPOPT then stops at the
  initial point after 0 iterations and reports `Solve_Succeeded`: `table_calibration_casadi.py`'s
  `default_derivative_run()` shows it.
- **CasADi's `bspline` is zero outside its grid.** In `heat_pump_mpc` the optimal supply temperature
  sits on its 25 °C bound, and IPOPT's bound relaxation puts it a hair below, where CasADi's COP is 0
  and the problem turns infeasible. The COP table therefore reaches 5 K past the bounds.
- **CasADi refuses `lookup_mode="exact"` for a `bspline`** (its not-a-knot knots are not the uniform
  grid), and checks "exact" grids for equal spacing strictly: `linspace` grids fail. The pairs use
  grids that step by powers of two.
- **CasADi's JIT with `expand=True` fails on graphs holding an interpolant** ("`eval_sx` not defined"),
  so three pairs compile unexpanded MX, whose builds take 8 to 91 s.
- **The generated law of `contouring_control` repeats the track's table** in each of the six
  functions that read it, 942 kB of C: the code generator does not share a constant between the
  functions of a module yet (todo C-187).
- **On small tables, `"auto"` is not always the fastest search at random batch points.** It picks
  `binary` up to 32 cells. On the 10 x 11 motor table the `count` search, which it never picks, is
  0.5 ns a point faster, since its compares vectorize in a batch loop. For one point called
  repeatedly, what `auto` was tuned on, count never won.
- **An interpolating cubic's Jacobian in its data is dense even at known points**, since its fit is
  global. To fit a smooth table, make B-spline coefficients the variables (`learning_tables.ipynb`).
