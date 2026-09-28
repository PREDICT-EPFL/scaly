# Interpolation and lookup tables

`scaly.interp` turns tabulated data into functions a model can use: lookup tables, interpolating
and smoothing splines, and fits with a shape the physics requires. Every one is a tensor-product
B-spline, `interp.BSpline`. It is evaluated in generated C like any other graph, differentiates to
any order its degree allows, and takes its data or coefficients as numbers known now or as `Expr`s
decided at run time.

```python
import numpy as np
import scaly as sc
from scaly import interp

soc = np.linspace(0.0, 1.0, 11)
ocv = np.array([3.0, 3.45, 3.55, 3.6, 3.63, 3.66, 3.7, 3.76, 3.85, 3.97, 4.2])
f = interp.interpolant(soc, ocv, kind="pchip")      # a monotone curve through monotone data


@sc.function(sc.L("s", ()), output=sc.G("v", "dv_ds"))
def cell(s):
  v = f(s)
  return v, sc.gradient(v, s)
```

## Kinds

`interp.interpolant(grid, values, kind=...)` builds a curve through data on a grid: a vector of
sites in 1-D, a tuple of vectors in n-D, with `values` shaped `(n_1, ..., n_D, *out_shape)`.
Trailing axes of `values` make the interpolant vector- or matrix-valued. `kind` is one per axis or
one for all.

| `kind` | Continuity | Passes through the data | Same as |
| --- | --- | --- | --- |
| `"nearest"` | a step at each midpoint | yes | `RegularGridInterpolator(method="nearest")` |
| `"zoh"` | a step at each site | yes | `searchsorted(side="right") - 1` |
| `"linear"` | C⁰ | yes | `np.interp`, multilinear in n-D |
| `"cubic"` | C² | yes | `CubicSpline`, with `bc` `"not-a-knot"`, `"natural"`, `"clamped"` or `"periodic"` |
| `"spline"` | C^(k−1) | yes | `make_interp_spline(k=degree)`, degree 1 to 5 |
| `"pchip"` | C¹, monotone data stay monotone | yes | `PchipInterpolator` |
| `"akima"`, `"makima"` | C¹, damped overshoot | yes | `Akima1DInterpolator` |
| `"steffen"` | C¹, monotone, no extremum between sites | yes | Steffen (1990) |
| `"smooth_linear"` | C², linear away from the sites | no | CasADi's `smooth_linear` |

The C² splines ring on steps and plateaus. Where the data have a shape to keep, use the four
shape-preserving kinds, which are 1-D only. A periodic axis (`bc="periodic"`) needs equal first
and last values, and wraps points outside by default. `examples/interp/interpolation_kinds.ipynb`
measures every kind's overshoot, monotonicity and continuity on the same data.

## Beyond the data

Outside the grid, `extrap=` decides, per axis:

- `"linear"`: the default from degree 1 up. The curve continues along its end tangent, so the
  value and the slope are continuous at the end and the slope outside is bounded;
- `"clamp"`: the default for `nearest` and `zoh`. The end value holds;
- `"extend"`: the end polynomial continues, as SciPy's `extrapolate=True`;
- `"periodic"`: the point is wrapped into the base interval;
- `"fill"`: a constant, `fill=`, NaN by default.

A NaN point gives NaN in every mode.

## Evaluating

A spline is called like a function. At one point (a scalar in 1-D, a `(D,)` vector in n-D) it
inlines its evaluation into the caller's graph. At a batch, `(N,)` in 1-D or `(N, D)`, it is one
mapped call of the point's graph, so the generated C is a loop whose size does not depend on `N`.

Evaluation finds each coordinate's cell, then evaluates that cell's polynomial.

- `search=` picks how the cell is found, per axis: `"uniform"` (a floor and one correction, for
  uniform grids), `"bucket"` (a uniform bucket index and a few compares), `"binary"` (branch-free
  halvings) or `"count"`. `"auto"` takes binary up to 32 cells, then the first of bucket, uniform
  and binary the grid allows.
- `strategy=` picks how the polynomial is stored. `"pp"` keeps each cell's Taylor coefficients and
  evaluates them by nested Horner, the fastest. `"basis"` keeps the B-spline coefficients and
  combines each axis's local basis functions, which is smaller. `"auto"` takes `"pp"` while its
  table fits `interp.PP_BUDGET` (4 MB) or is the smaller of the two.
- `f.index(x)` finds the cells once, and `f(x, index=...)` reuses them for several splines on the
  same partition: a table and its derivative, say.

## Derivatives and sparsity

A spline's derivatives with respect to the point come from differentiating its graph, forward or
reverse, to any order. At a knot, a derivative that jumps is the one-sided derivative of the cell
to the right. `f.derivative(nu, axis)` is the derivative as a spline of its own, of degree
`k − nu`. It is for when the derivative must be a table: a curvature table, or a bound on `f'`
through its coefficients.

With data or coefficients that are `Expr`s, the spline also differentiates with respect to them,
and where the points are read decides the Jacobian's pattern:

- `f.at(points)`, with points known when the graph is built, is `f.basis(points)` (the sparse
  design matrix) times the coefficients: one sparse product whose Jacobian has exactly the
  basis's pattern, 4 entries a row for a bilinear table, 16 for a bicubic;
- `f(x)` at symbolic points finds the cells at run time, so every value may depend on every
  coefficient, and the pattern is dense;
- an interpolating cubic's values depend on all of its data, so its pattern is dense even at known
  points. To fit a smooth table, make the B-spline's coefficients the variables.

`examples/interp/learning_tables.ipynb` shows the difference: the same calibration takes about
20 times longer per solve with the points symbolic.

## Where the coefficients come from

- **Interpolation.** `interp.interpolant` fits per axis, as SciPy's `NdBSpline` of per-axis
  `make_interp_spline` fits does.
- **Smoothing.** `interp.smoothing(x, y, ...)` fits noisy data. `method="pspline"` (Eilers and
  Marx) is a least-squares B-spline on `segments` equal intervals with a difference penalty, any
  degree and dimension. `method="cubic"` is SciPy's `make_smoothing_spline`. `lam="gcv"` chooses
  the weight by generalized cross-validation.
- **Shape constraints.** `interp.constrained(x, y, ...)` is least squares under linear conditions
  on the coefficients that make a shape hold everywhere, not only at the data: `monotone`,
  `convex` (per axis), `bounds`, pinned values and derivatives (`equal`), and `periodic` ends. It
  is a quadratic program, solved by PIQP when the graph is built. An active-set step after the
  interior point makes the fit exact to rounding.
- **Your own.** `interp.BSpline(knots, coeffs, degree)` takes coefficients directly, numbers or an
  `Expr`: a spline-parametrized input or trajectory, or a nonlinearity to identify.
- **Data at run time.** `interp.interpolant` with an `Expr` for `values` puts the fit in the graph.
  For the kinds linear in the data it is a constant linear map. Above 40 sites a C² cubic solves
  its tridiagonal system in scans. The shape-preserving slopes are expressions.

`f.integrate(a, b)` integrates exactly, and an `Expr` bound gives an expression.
`f.antiderivative()` is the integral as a spline. `f.inverse()` reads a strictly monotone 1-D spline
backwards, by safeguarded Newton in a loop, with the derivative `1/f'` supplied directly.

## Generated code

- **Tables become constants** in the C source. A spline tabulating more than 2²⁰ values warns: that
  is about 20 MB of source and seconds of compilation. Pass such a table as an input instead.
- **A table as an input** is an `Expr` table, a parameter of the Function. The generated header
  states the buffer's shape and order (`// 9 x 8, row-major (C order)`), and `f.pack(values)`
  gives the flat buffer `f.function()` expects.
- **`dtype="float32"`** stores the tables and evaluates in single precision for embedded targets.
  Values agree with double precision to about 1e-6. Derivatives in single precision wait on the
  core AD.
- **`f.function()`** makes the spline a Function of one point, so that several callers share one
  procedure and one copy of its table. A table read by several generated functions, a stage cost
  and its derivatives say, is currently emitted once per function.

## For CasADi users

- **Data order.** CasADi's `interpolant` takes the values flattened with the first grid axis
  fastest, `np.ravel(values, order="F")`. `interp.interpolant` takes them shaped like the grid.
- **Derivatives in the data.** A parametric CasADi interpolant, and a `bspline` node with symbolic
  coefficients, differentiate to zero with respect to that data unless built with `inline=True`,
  and a solver then stops at its starting point. Here the derivative with respect to data or
  coefficients is always there.
- **Outside the grid.** CasADi's `bspline` is zero outside its grid, a cliff that a relaxed bound
  can reach. Here the default continues linearly.
- **Degrees and lookup.** CasADi's `bspline` refuses degrees 2 and 4, and refuses
  `lookup_mode="exact"`, since its not-a-knot knots are not the grid. Every degree works here, and
  the search is chosen per axis.

`examples/interp/pairs/` solves five problems in both libraries, and `examples/interp/README.md`
compares them.
