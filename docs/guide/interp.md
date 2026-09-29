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
Trailing axes of `values` make the interpolant vector- or matrix-valued. `kind`, `bc`, `extrap` and
`search` take one value for all axes, or a tuple or list with one per axis.

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

- `"linear"` continues along the end tangent, so the value and the slope are continuous at the end
  and the slope outside is bounded. It is the default from degree 1 up.
- `"clamp"` holds the end value. It is the default for `nearest` and `zoh`.
- `"extend"` continues the end polynomial, as SciPy's `extrapolate=True`.
- `"periodic"` wraps the point into the base interval.
- `"fill"` returns a constant, `fill=`, NaN by default.

A NaN point gives NaN in every mode. An infinite point gives the end value for `clamp`, the fill
for `fill`, and NaN for `periodic`. A `linear` continuation (from degree 2), and an `extend` one
under the `basis` strategy, is followed out to where the k-th power of the distance still fits a
float, 10^(300/k) (10^(30/k) in float32), and held beyond: an infinite point gives the
continuation's value there, a large finite number with its sign, and a zero derivative. An
`extend` continuation under `pp` gives the polynomial's own infinity. Far out, a `linear` continuation's value is its slope times the
distance, so a slope that is zero only up to rounding still shows. A periodic point far enough out
that its count of periods is no longer exact has lost its phase to rounding, and is clamped into
the base interval.

## Evaluating

A spline is called like a function. At one point (a scalar in 1-D, a `(D,)` vector in n-D) it
inlines its evaluation into the caller's graph. At a batch, `(N,)` in 1-D or `(N, D)`, it is one
mapped call of the point's graph, so the generated C is a loop whose size does not depend on `N`.

Evaluation finds each coordinate's cell, then evaluates that cell's polynomial.

- `search=` picks how the cell is found, per axis. `"uniform"` takes a floor and one correction,
  for uniform grids. `"bucket"` takes a uniform bucket index and a few compares. `"binary"` makes
  branch-free halvings, and `"count"` counts the edges passed. `"auto"` takes binary up to 32
  cells, then the first of bucket, uniform and binary the grid allows.
- `strategy=` picks how the polynomial is stored. `"pp"` keeps each cell's Taylor coefficients and
  evaluates them by nested Horner, the faster. `"basis"` keeps the B-spline coefficients and
  combines each axis's local basis functions, the smaller. `"auto"` takes `"pp"` while its table
  fits `interp.PP_BUDGET` (4 MB) or is the smaller of the two.
- `f.index(x)` finds the cells once, and `f(x, index=...)` reuses them for other splines on the
  same partition, such as a table and its derivative. A spline on another partition refuses the
  index.
- `dtype="float32"` stores the tables and evaluates in single precision, for embedded targets. The
  tables are expanded about the cell centers exactly as float32 holds them, so values agree with
  double precision to about 1e-6, far from the origin too.

## Derivatives and sparsity

A spline's derivatives with respect to the point come from differentiating its graph, forward or
reverse, to any order. At a knot, a derivative that jumps is the one-sided derivative of the cell
to the right. `f.derivative(nu, axis)` is the derivative as a spline of its own, of degree
`k − nu`. It is for when the derivative must be a table, a curvature table say, or a bound on `f'`
through its coefficients. Derivatives in single precision wait on the core AD.

With data or coefficients that are `Expr`s, the spline also differentiates with respect to them,
and where the points are read decides the Jacobian's pattern.

- `f.at(points)`, with points known when the graph is built, is `f.basis(points)` (the sparse
  design matrix) times the coefficients, one sparse product. Its Jacobian has exactly the basis's
  pattern, 4 entries a row for a bilinear table and 16 for a bicubic.
- `f(x)` at symbolic points finds the cells at run time, so every value may depend on every
  coefficient, and the pattern is dense.
- An interpolating cubic's values depend on all of its data, so its pattern is dense even at known
  points. To fit a smooth table, make the B-spline's coefficients the variables.

`examples/interp/learning_tables.ipynb` shows what the dense pattern costs a solver.

## Where the coefficients come from

- `interp.interpolant` fits per axis, as SciPy's `NdBSpline` of per-axis `make_interp_spline` fits
  does.
- `interp.smoothing(x, y, ...)` fits noisy data. `method="pspline"` (Eilers and Marx) is a
  least-squares B-spline on `segments` equal intervals with a difference penalty, in any degree
  and dimension. `method="cubic"` is SciPy's `make_smoothing_spline`. `lam="gcv"` chooses the
  weight by generalized cross-validation.
- `interp.constrained(x, y, ...)` is least squares under linear conditions on the coefficients
  that make a shape hold on the data's range: `monotone` and `convex` per axis, `bounds`, pinned
  values and derivatives (`equal`), and `periodic` ends. It is a quadratic program, solved by PIQP
  when the graph is built (`qp=` takes another `sc.opt` method that reads the bounds as data at run
  time), then refined by one active-set step, so the fit is exact to rounding
  and the same at any scale of the data. Outside the data's range the default `linear`
  continuation keeps a monotone or convex shape but not the bounds, which `extrap="clamp"` keeps.
- `interp.BSpline(knots, coeffs, degree)` takes coefficients directly, numbers or an `Expr`: a
  spline-parametrized input or trajectory, or a nonlinearity to identify.
- `interp.interpolant` with an `Expr` for `values` puts the fit in the graph. For the kinds linear
  in the data it is a constant linear map, which warns when it is dense and large. Above 40 sites
  a C² cubic solves its tridiagonal system in scans instead. The shape-preserving slopes are
  expressions.

`f.integrate(a, b)` integrates exactly, and an `Expr` bound gives an expression, exact in 1-D in
every extrapolation mode, whose derivative in the bound is the spline there. `f.antiderivative()`
is the integral as a spline. `f.inverse()` reads a strictly monotone 1-D spline backwards, by
safeguarded Newton (`sc.roots.NewtonBisection` on the cell) in a loop that stops relative to the
cell's width, with the derivative `1/f'` supplied directly.

## Methods

Each kind is also a class, an interp method with its options, found by name in `interp.REGISTRY` as
`interp.<kind>`. `interp.solver(interp.Fit(x, y, ...), method)` fits the spline the shorthand fits;
`Fit` holds the data, on a grid or scattered, and how the spline is evaluated (`extrap`, `fill`,
`search`, `strategy`, `dtype`, `name`). `"auto"` interpolates data on a grid linearly and smooths
scattered data.

```python
table = interp.solver(interp.Fit(soc, ocv, extrap="clamp"), interp.PCHIP())
mixed = interp.solver(interp.Fit((t, v), map2d), interp.PerAxis(interp.Linear(), interp.Cubic(bc="natural")))
fit = interp.solver(interp.Fit(x, y), interp.Constrained(knots=12, monotone="increasing", bounds=(0.0, 1.0)))
```

| Methods | Options | Shorthand |
| --- | --- | --- |
| `interp.Nearest`, `interp.ZOH`, `interp.Linear`, `interp.Cubic`, `interp.Spline`, `interp.PCHIP`, `interp.Akima`, `interp.Makima`, `interp.Steffen`, `interp.SmoothLinear` | `period` (`ZOH`), `bc` (`Cubic`, `Spline`), `degree` (`Spline`), `frac` (`SmoothLinear`) | `interp.interpolant(x, y, kind=...)` |
| `interp.PerAxis(method, ...)`, one interpolating method per axis | | `interpolant` with a tuple of kinds |
| `interp.Smoothing` | `degree`, `segments`, `penalty`, `lam`, `method` | `interp.smoothing` |
| `interp.Constrained` | `degree`, `knots`, `weights`, `monotone`, `convex`, `bounds`, `equal`, `periodic`, `penalty`, `lam`, `qp` | `interp.constrained` |

A method refuses the data it cannot fit, naming why: an interpolant scattered points, a
shape-preserving kind a grid of more than one axis, a constrained fit an `Expr`.

## Generated code

- Tables become constants in the C source. A spline tabulating more than 2²⁰ values warns, since
  that is tens of megabytes of source; pass such a table as an input instead.
- A table becomes an input when it is an `Expr` parameter of your Function:
  `interp.interpolant(grid, table)` inside the Function keeps the fit in the graph, and the
  generated header states the buffer's shape and order (`// 9 x 8, row-major (C order)`). For a
  `BSpline` whose coefficients are the input, `f.function()` takes them flat as `c`, and
  `f.pack(coeffs)` builds that buffer.
- `f.function()` makes the spline a Function of one point, so that several callers share one
  procedure. Each derivative of it is a procedure of its own with its own copy of the table, and
  a table read by several generated functions, a stage cost and its derivatives say, is emitted
  once per function.

## For CasADi users

- CasADi's `interpolant` takes the values flattened with the first grid axis fastest,
  `np.ravel(values, order="F")`. `interp.interpolant` takes them shaped like the grid.
- A parametric CasADi interpolant, and a `bspline` node with symbolic coefficients, differentiate
  to zero with respect to that data unless built with `inline=True`, and a solver then stops at its
  starting point. Here the derivative with respect to data or coefficients is always there.
- CasADi's `bspline` is zero outside its grid, a cliff that a relaxed bound can reach. Here the
  default continues linearly.
- CasADi's `bspline` refuses degrees 2 and 4, and refuses `lookup_mode="exact"`, since its
  not-a-knot knots are not the grid. Every degree works here, and the search is chosen per axis.

`examples/interp/pairs/` solves five problems in both libraries, and `examples/interp/README.md`
compares them.
