# Splines and lookup tables: implementation plan (2026-09-28, v1)

Status: **in progress** on `claude/interp`, branched from `claude/integrators-mpc`. The §12 questions
were taken at the proposed answers: `scaly.interp`; `linear` (degree ≥ 1) and `clamp` (degree 0) as
the default extrapolation; the pairs in `examples/interp/pairs/` with `compare.py --dir`; one report
per PR; no scope trimmed; a zero `floor`/`ceil` derivative refused under `nonsmooth="error"`. Todo ids
are this plan's plus 12 (C-170 to API-185), since C-158 to C-161 were taken on
`claude/edge-case-fixes`. This adds a new in-tree sub-package,
`scaly.interp`. It covers:

- interpolants on rectilinear grids in any dimension: lookup tables, shape-preserving cubics,
  interpolating and smoothing splines, and shape-constrained fits;
- vector-valued outputs;
- coefficients that are constants, runtime inputs or decision variables;
- evaluation at symbolic points (a runtime interval search) and at points known when the graph is
  built (a constant sparse basis matrix);
- spline calculus: derivatives, antiderivatives, integrals and the inverse of a monotone curve.

The phase list is §7. Every numeric claim is tested against SciPy, and against CasADi 3.8.0 (in
the `dev` group of `pyproject.toml`) wherever CasADi has the feature (§2, §9).

The design follows the brainstorm of 2026-09-27:

1. **One object for tables and splines.** Every interpolant is a tensor-product spline on a
   rectilinear grid. A lookup table is the degree-0 or degree-1 case.
2. **No new expression op in the first pass.** Evaluation is composed from ops scaly already has:
   `floor`, `cast`, `take`, `where` and arithmetic. The prototype in §1.2 confirms this works,
   derivatives included. A dedicated op is added only if the benchmark gate at the end of SP4
   trips (§10).
3. **Generation-time choices.** The search strategy and the evaluation strategy are picked when
   the code is generated, from the grid's size and uniformity. The user can override both.

## 1. What exists today

### 1.1 In the tree

`src/scaly` has no interpolation module. What the repository does instead:

| Pattern | Where |
| --- | --- |
| Track centre line fitted by a closed cubic spline in NumPy (a dense KKT solve replacing the original's OSQP fit), sampled in arc length, and read off as a time reference outside the NLP | `benchmarks/problems/race_cars/reference.py` (`fit_spline`) |
| Tables indexed by a known integer | `sc.take` / `sc.gather` in `truss_sizing.py`, `optimal_power_flow.py` |
| A reference evaluated in NumPy at the MPC's sample times and passed as a parameter vector `r` | `examples/mpc/reference_tracking.ipynb` |

What the compiler already provides:

- `sc.take(x, idx, in_range=)` with runtime `int64` indices, differentiable in `x` with `put_add`
  as its adjoint;
- `floor`, `ceil`, `cast`, `where`, `minimum`, `maximum` and `isfinite`;
- `sc.scan` and `sc.while_loop` (with `params=`), `sc.vmap` and `sc.custom_derivative`;
- `linalg.SparseMatrix` (`from_scipy`, `matvec`) for constant sparse maps;
- `static const` tables in the generated C;
- `sc.options(nonsmooth=...)`;
- `sc.qp_problem` with the PIQP backend.

SciPy (1.18) is a runtime dependency, so the fits done when the graph is built can use
`scipy.interpolate` directly.

### 1.2 Prototype (2026-09-28, staged copy of `claude/integrators-mpc`, CPython 3.13, gcc)

A 1-D cubic spline written with the existing ops:

1. `i = cast(clip(floor((x - t0)/h), 0, n-1), int64)`;
2. `s = x - t_i`;
3. `c = take(C, [i])` on the `(4, n)` piecewise-polynomial coefficient table taken from SciPy's
   `CubicSpline.c`;
4. Horner in `s`.

What it showed:

- **Values.** They match `CubicSpline` to 1e-12 through the JIT, at knots and at the right end
  included.
- **Derivatives.** `sc.gradient`, `sc.hessian` and forward-mode `sc.jacobian` match
  `CubicSpline(x, 1)` and `(x, 2)` to 1e-15. The index passes through a float-to-int cast, which
  has no derivative, so AD never reaches `floor`.
- **Non-uniform grids.** The branch-free count search (`sum(x >= t[1:-1])`) works.
- **Coefficients as a Function input.** The Jacobian with respect to them is correct: 4 nonzeros
  in the active column.
- **Non-finite input.** `NaN` gives `NaN`. `±inf` gives the end polynomial's limit.

**Two AD gaps found** (the second by the verification pass):

1. **`floor` on a differentiable path.** A periodic wrap written `x - T*floor(x/T)` differentiates
   through `floor`. Today that raises `NotImplementedError: VJP for nonsmooth op floor`
   (`ad/reverse.py:549`, and the JVP at `ad/forward.py:240`). A round trip through `int64`,
   `x - T*cast(cast(floor(x/T), int64), float64)`, works today (gradient `cos(1.3)` exactly), so
   this is not blocking. SP0 still gives `floor` and `ceil` a derivative, so neither users nor the
   library need the trick.
2. **Integer index arithmetic in forward mode.** Build the index vector with broadcast `int64`
   arithmetic, e.g. `arange(K) + stack([base])*K`, or `(N,1)*K + (1,K)` for a batch. Then
   `sc.gradient` works, but `sc.jacobian` and `sc.hessian` fail in `_jvp_many_structural`
   (`ad/forward.py:1449`) with
   `TypeError: mixed-dtype operation not yet supported: float64 vs int64`.
   - Scalar `int64` arithmetic is fine.
   - The workaround is to build the indices in `float64` and cast once.
   - SP0 fixes it: the structural JVP skips nodes that carry no derivative.

**Extended to two dimensions** (the verification pass). On a 9×12 grid, a bicubic built from
`take` and nested Horner matches `NdBSpline`:

| Quantity | Error |
| --- | --- |
| values | 2.4e-15 |
| gradient | 5.6e-15 |
| Hessian | 2.2e-14 |

A batch of 205 points runs as one graph with 205·16 indices. Its Jacobian pattern is exactly two
entries per row. Build such a Jacobian with `sparse_jacobian` (0.03 s): a dense `sc.jacobian` of
the batch took 44 s.

## 2. What CasADi implements (verified on the 3.8.0 wheel, 2026-09-28)

| Feature | CasADi | Verified behaviour |
| --- | --- | --- |
| `interpolant(name, "linear", grid, values)` | multilinear on a tensor grid, vector outputs | equals `np.interp` and `RegularGridInterpolator(method="linear")` exactly. Outside the grid it continues the end cell's multilinear polynomial, which equals `RegularGridInterpolator(fill_value=None)` in 1-D and 2-D |
| `interpolant(name, "bspline", grid, values)` | fitted B-spline, `degree` per axis (default 3), `algorithm` `not_a_knot` or `smooth_linear` | equals SciPy's not-a-knot fit to 1e-14: `CubicSpline` and `make_interp_spline(k)` in 1-D, and the per-axis tensor fit (`NdBSpline`, or `RegularGridInterpolator(method="cubic_legacy")`) in 2-D. It does **not** equal SciPy 1.18's new `method="cubic"` (6e-5 apart) |
| Fitting degrees | 1, 3, 5 | degrees 2 and 4 raise `Not implemented` |
| `bspline` outside the knot span | — | returns **0** |
| `smooth_linear` | smoothed linear, sharpness `smooth_linear_frac` in [0, 0.5] | matches linear at mid-cells but not at the data points (0.8380 against sin(1) = 0.8415) |
| `lookup_mode` | `linear` (≤100 knots), `exact` (uniform), `binary` (>100) | — |
| Parametric variants | grid and/or values as inputs | the parametric `bspline` interpolant fits the given values at run time, so its values equal the non-parametric one |
| Derivatives w.r.t. values or coefficients | — | **zero by default**: `interpolant` (linear and bspline), the `bspline` MX node with MX coefficients, and `blazing_spline`. The `bspline` default keeps structural nonzeros (11 of 11) that are numerically zero; `linear` has none. They are correct only with `inline=True` (`interpolant`, `bspline`: finite-difference error 1e-11 to 3e-10), and then structurally dense |
| `bspline(x, coeffs, knots, degree, m)` | an MX node that evaluates B-spline coefficients | values and `d/dx` match SciPy `BSpline`. `d/dc` is 0 unless `inline=True` |
| `blazing_spline(name, knots)` | cubic, up to 5-D, scalar output, coefficients as an input `C` (knots as an input optionally) | `precompute_coeff` and `precompute_grid`; lookup `linear`, `exact`, `binary` or `auto`; `d/dC` is 0 |
| `interp1d(x, v, xq, mode)` | 1-D, `linear`, `floor` or `ceil` | numeric query points, so a constant map applied to `v` |
| Hessians w.r.t. `x` | — | correct (bspline: −0.73817818692238 against SciPy's −0.73817818692238) |
| Generated C | `casadi_low`, `casadi_de_boor`, `casadi_nd_boor_eval` | 361 lines for a 1-D cubic value and Jacobian |
| Missing | — | PCHIP, Akima, Steffen; monotone, convex or bounded fits; smoothing splines; periodic splines or periodic extrapolation; any extrapolation choice; ZOH or nearest tables in n-D; warm-started search; spline calculus; inverse lookup |

Two consequences for the comparisons in §9:

1. Every CasADi side that differentiates with respect to table data passes `inline=True`, and the
   write-up states that the default gives a zero derivative.
2. Periodic tracks are padded by one lap on the CasADi side. Out-of-range queries on its bspline
   side are kept inside the grid, since it returns 0 there.

## 3. Design

### 3.1 The canonical object

An interpolant is a **tensor-product B-spline** on a rectilinear grid. It is given by:

- per axis `d`, a knot vector `t_d` (multiplicities allowed) and a degree `k_d`;
- a coefficient tensor `c` of shape `(n_1, ..., n_D, *out_shape)`, as an `np.ndarray` or an `Expr`;
- per axis, an extrapolation mode;
- a name, used for Function and C-table names.

$$f(x)=\sum_{\alpha} c_\alpha \prod_{d=1}^{D} B_{\alpha_d,k_d}(x_d;t_d)$$

Every kind of §3.4 becomes this object:

- a linear table is degree 1, and its coefficients are the data;
- ZOH is degree 0;
- nearest is degree 0 on midpoint knots;
- PCHIP, Akima and Steffen are C¹ cubics, i.e. degree 3 with double interior knots;
- periodic splines wrap their coefficients.

The B-form is the canonical form rather than the piecewise-polynomial (pp) form because of
storage. The pp form stores `(k+1)^D` coefficients per cell. For a tricubic table on a 64³ grid
that is 64 times the data, about 130 MB of `static const`. The B-form stores about one
coefficient per data point, as CasADi and SciPy's `NdBSpline` do. The Hermite kinds are the
exception: with their double knots they need `2^D` per point. The pp form is kept as an **evaluation
strategy** (§3.5), and it is the fastest one in 1-D.

### 3.2 Package and import layers

```
src/scaly/interp/
  __init__.py     the public interpolation surface
  grid.py         Axis: knots, degree, uniformity, extrapolation; validation; the interval searches as Expr builders
  spline.py       Spline: the tensor-product B-spline, its evaluation strategies, calculus, basis matrices, .function(), .to_scipy()
  fit.py          the fit rules of §3.4: SciPy at build time for NumPy data, in-graph linear maps or local formulas for Expr data
  constrained.py  least-squares fits with shape constraints, as a QP solved by the PIQP backend
```

The package sits at **import layer 5**, like `integrators` and `mpc`:

- it needs `function/sugar` (4) and `function/api`, `linalg` and `solvers` (5);
- nothing inside the compiler imports it;
- `scaly/__init__.py` gains `from . import interp`, with no flat re-exports;
- `scipy.interpolate` is imported inside the fitting functions, so `import scaly` does not pay for
  it.

Each module gets an `IMPORT_LAYERS` entry, a one-line ownership docstring and a mirrored test file
under `tests/interp/`.

### 3.3 The API

```python
import scaly as sc
from scaly import interp

# NumPy data, fitted when the graph is built: the coefficients are baked into the C as static const tables
f = interp.interpolant(t, y, kind="cubic", bc="not-a-knot", extrap="linear")      # 1-D
g = interp.interpolant((T_out, T_sup), COP, kind=("cubic", "linear"))              # 2-D, one kind per axis
h = interp.interpolant(theta, XYPK, kind="cubic", bc="periodic", extrap="periodic")  # (n, 4) data: vector output
p = interp.interpolant(t_hours, price, kind="zoh", extrap="periodic")              # a table in time

f(x)                    # x a scalar Expr -> scalar
g(sc.stack([To, Ts]))   # a point (2,) -> scalar
g(P)                    # a batch (N, 2) -> (N,), one vectorized evaluation, not N calls
i = f.index(x); f(x, index=i); f.derivative()(x, index=i)   # one search shared by several evaluations
f.derivative(1); f.antiderivative(); f.integrate(a, b); f.inverse()

# Coefficients or data as Exprs: runtime inputs (calibration) or decision variables (identification, learning)
tab = interp.interpolant((gx, gy), vals, kind="linear")          # vals is an Expr; the fit becomes part of the graph
s = interp.BSpline(knots, degree=3, coeffs=c)                    # c is an Expr: the B-coefficients themselves
B = s.basis(t_samples)                                           # scipy.sparse design matrix, for points known now
yhat = s.at(t_samples)                                           # SparseMatrix(B) @ c: exact Jacobian sparsity

# Fitted by optimization, NumPy data
sm = interp.smoothing(t, y, degree=3, lam="gcv")                                  # P-spline or make_smoothing_spline
ocv = interp.constrained(soc, v, degree=3, knots=16, monotone="increasing", bounds=(2.5, 4.3))  # a QP through PIQP

# Choices normally made at generation time, overridable
interp.interpolant(t, y, search="auto" | "uniform" | "count" | "binary" | "bucket",
                   strategy="auto" | "pp" | "basis", name="cop")
F = f.function()        # a ConcreteFunction, so a large table sits in one procedure behind CALLs
f.to_scipy()            # BSpline / NdBSpline / PPoly, for tests and plotting
```

Calling a `Spline` inlines expressions by default, like a math function. The expressions are
hash-consed, so a spline used once per `vmap` body appears once. `.function()` is for large
tables used from several Functions: otherwise each procedure that uses the table gets its own
`static const` copy. `.function()` names follow the integrator rule, derived from `name=` and
the configuration, so two splines never collide at lowering.

### 3.4 The kinds

| `kind` | Degree | Continuity | Local | Linear in the data | SciPy reference | CasADi |
| --- | --- | --- | --- | --- | --- | --- |
| `nearest` | 0 | none | yes | yes | `RegularGridInterpolator(method="nearest")` away from midpoints; explicit tie test | — |
| `zoh` (previous value) | 0 | none, right-continuous | yes | yes | `np.searchsorted(side="right") - 1` | `interp1d(..., "floor")`, static points, 1-D |
| `linear` | 1 | C⁰ | yes | yes | `np.interp`, `RegularGridInterpolator("linear")` | `interpolant("linear")`, `"bspline"` degree 1 |
| `cubic`, `bc` in `not-a-knot`, `natural`, `clamped`, `periodic` | 3 | C² | no | yes | `CubicSpline(bc_type=)`; n-D: per-axis `make_interp_spline` into `NdBSpline` (`= RGI "cubic_legacy"`) | `interpolant("bspline")`, not-a-knot only |
| `spline` with `degree=k`, 1 to 5 | k | C^(k−1) | no | yes | `make_interp_spline(k=k)` | `"bspline"`, k in {1, 3, 5} |
| `pchip` | 3 | C¹ | yes | no | `PchipInterpolator`; RGI `"pchip"` | — |
| `akima`, `makima` | 3 | C¹ | yes | no | `Akima1DInterpolator(method=)` | — |
| `steffen` | 3 | C¹, monotone | yes | no | none: Steffen (1990) formulas, and a monotonicity property test | — |
| `smooth_linear` | 3 | smooth (the exact class is read from CasADi's source in SP2) | to be confirmed | yes | none | `"bspline"`, `algorithm="smooth_linear"` |
| `interp.smoothing` | k | C^(k−1) | no | yes for a fixed λ | P-splines (Eilers–Marx, any degree, any D): a dense penalized least-squares solve. `method="cubic"`: `make_smoothing_spline` (cubic, knots at the data, penalty ∫f″², GCV for `lam=None`), which is a different estimator and is compared only with itself | — |
| `interp.constrained` | k | C^(k−1) | no | no (a QP) | `scipy.optimize.minimize(method="trust-constr")` on small cases | — |

For n-D grids, `kind` is per axis. The exceptions are the shape-preserving kinds (`pchip`, `akima`,
`makima`, `steffen`), which are **1-D only** in this plan:

- They are nonlinear in the data.
- RGI `"pchip"` in 2-D is not a tensor-product spline. With `x` fixed it is not a cubic in `y`
  within a cell (a cubic fit leaves a 3e-2 residual; `linear`, `cubic` and `cubic_legacy` leave
  1e-13), so it cannot serve as the reference for the object of §3.1.
- A tensor-product Hermite definition is deferred (API-185). In n-D, `interp.constrained` is the
  shape-preserving route.

Two conventions are settled in SP2:

- **`nearest` at a midpoint** follows SciPy and takes the lower neighbour, so its midpoint knots
  are left-continuous, unlike the right-continuous convention of §3.5.
- **`smooth_linear`** is matched to CasADi's algorithm, read from its source. If the CasADi source
cannot be matched exactly, `smooth_linear` ships under its own documented definition and the
comparison becomes qualitative (maximum deviation from linear).

### 3.5 Evaluation

Evaluation has four steps. Each is chosen per axis when the graph is built.

**1. Find the interval.** Intervals are `[t_j, t_{j+1})`, with the last one closed. This is
right-continuous, like `searchsorted(side="right") - 1`, and it decides what degree 0 returns and
which one-sided derivative a C⁰ or C¹ kind gets at a knot.

| `search` | Cost | Built from | When `auto` picks it |
| --- | --- | --- | --- |
| `uniform` | a subtract, a multiply, `floor`, one correction compare | `floor`, `cast`, one `take` of `t` | the knots are uniform: `max_j |t_j - (t0 + j h)|` within a few ulp. A plain `floor((t_j - t0)/h) == j` check fails on 842 of 990 `np.linspace` grids, e.g. `0.30000000000000004/0.1 = 3.0000000000000004`, hence the correction compare. The corrected index is checked against `searchsorted` at every knot and at `nextafter` on both sides when the graph is built |
| `count` | `n - 1` compares and a sum, branch-free, vectorizes | `>=` against a constant vector, `sum` | non-uniform with `n ≤ N_count` (set by measurement in SP4; CasADi uses 100) |
| `binary` | `ceil(log2 n)` branch-free halvings `lo += where(x >= t[lo + 2^m], 2^m, 0)`, unrolled, on a copy of `t` padded with `+inf` to a power of two | `take` of `t` at runtime indices | non-uniform, large, clustered |
| `bucket` | a uniform bucket index table, then a count over the bucket's at most `w` knots | two `take`s, `w` compares | non-uniform, large, with at most `w` (default 4) knots per bucket |
| hint (SP4, only if it wins) | a local count within ±`w` of a carried index | `take`, compares | only when asked: `f.index(x, hint=i_prev)` inside a `scan` |

**Non-finite input.** The index is always clamped into `[0, n_cells - 1]` in `float64`, **before**
the cast to `int64`:

- Casting `NaN` or `±inf` to an integer is undefined behaviour in C, and `in_range=True` would then
  read out of bounds.
- `minimum` and `maximum` lower to C `fmin` and `fmax`, which already map `NaN` to a finite bound
  and `±inf` to the end cell. The clamp therefore makes the index safe.
- The local coordinate `s = x - t_j` still carries the `NaN` or `inf` to the output (§3.6).
- The clamp-before-cast order and the `fmin`/`fmax` semantics are pinned by a test (SP0). An
  `isfinite` guard is added only if some backend's lowering differs.

**2. Fetch the coefficients.** They are stored flat and cell-major, so that each cell's block is
contiguous. There is one `take` per point with `K` indices, `base*K + arange(K)`, and a batch of
`N` points takes `N*K` indices in one op. `in_range=True` holds because the index is clamped.

**3. Evaluate locally.** There are two strategies (`strategy=`):

- **`pp`**: per-cell polynomial coefficients in powers of `s_d = x_d - t_{d,j}` (SciPy's `PPoly`
  convention, so the 1-D tables can be copied from SciPy), then nested Horner. This is the fastest.
  It stores `(k+1)^D` values per cell, so `auto` uses it only when that storage fits a budget
  (default 1 MB, measured in SP4) and the coefficients are constants.
- **`basis`**: per axis, take the `(k+1)×(k+1)` local polynomial matrix of the interval's
  non-zero basis functions (one shared matrix for uniform interior knots), evaluate
  `b_d = P_{d,j} s_d^{0..k}` by Horner, fetch the `(k+1)^D` B-coefficients through a constant
  offset table, and contract one axis at a time. This is lean in memory and it is what
  `Expr` coefficients use.

**4. Batch.** Everything above is elementwise over a leading batch axis, so `g(P)` for
`P (N, D)` is one vectorized graph. Under `vmap`, the body sees one point.

### 3.6 Extrapolation and non-finite input

| `extrap` | Outside the grid | Parity |
| --- | --- | --- |
| `extend` | the end cell's polynomial, continued | SciPy `PPoly`/`BSpline(extrapolate=True)`; CasADi `linear` in n-D, i.e. `RGI(fill_value=None)` |
| `linear` | a first-order Taylor expansion at the clamped point, `f(x_c) + ∇f(x_c)·(x - x_c)`: C¹ for kinds of degree ≥ 2, and nonzero gradients outside | Modelica `LinearSegments`-style |
| `clamp` | the value at the clamped point | Modelica `HoldLastPoint` |
| `periodic` | `x` wrapped into `[t0, tN)`; the data must close up (checked) | `CubicSpline(extrapolate="periodic")` |
| `fill=v` | the constant `v` (default `NaN`) | `RGI(bounds_error=False, fill_value=v)`; CasADi `bspline` with `v = 0` |

The proposed default is `linear` for degree ≥ 1 and `clamp` for degree 0 (§12, question 2). It is
safe inside an NLP: `clamp` stalls a solver on a zero gradient, `extend` lets a cubic run away,
and `fill=0` is CasADi's trap.

Non-finite input:

- `NaN` in gives `NaN` out, in every mode.
- `±inf`:
  - `clamp` gives the end value;
  - `fill` gives `v`;
  - `linear` and `extend` give the IEEE result of the formula, which is tested and documented.

Periodic wrapping uses `floor` on a differentiable path once SP0 lands, and the `int64` round trip
of §1.2 before that.

### 3.7 Derivatives and sparsity

- **With respect to `x`.** Derivatives come from AD through the composite graph, since the index
  path carries none. This gives the exact piecewise derivative, one-sided at the knots per §3.5.
  Hessians follow.
- **Sharing the search.** Hash-consing shares the interval search between the value, gradient and
  Hessian graphs. SP1 tests this: one search in the generated program.
- **With respect to `Expr` coefficients or data.** The derivative flows through `take`, whose
  adjoint is `put_add`. At a **symbolic** point the structural Jacobian row is dense in the
  coefficients, because any cell could be selected. That is inherent, and CasADi's `inline=True`
  is dense too.
- **Batches.** A batch evaluation has a block-diagonal Jacobian with respect to its points.
  Build it with `sparse_jacobian`, not a dense `jacobian`: 0.03 s against 44 s for 205 points in
  the verification pass. The guide says so.
- **Exact sparsity at static points.** When the points are known while the graph is built, `.at()`
  (§3.9) gives exact sparsity. This path is for identification, fitting and control
  parametrization.
- **Explicit derivative splines.** `f.derivative(nu)` builds the derivative spline itself: degree
  `k - nu`, coefficients by the de Boor differencing formula, constant or `Expr`. Use it when a
  derivative must be a spline, e.g. a curvature table or a constraint on `f'`. AD remains the way
  to get Jacobians.

### 3.8 Where the coefficients come from

| Provenance | What the C holds | Fit |
| --- | --- | --- |
| NumPy data | `static const` coefficient table | SciPy, when the graph is built |
| Runtime input (a calibration table changed without recompiling) | a Function input; the grid is fixed | in-graph (below); `hoist_invariant` moves it out of a `vmap` when the table is broadcast, which SP3 checks in the generated program |
| Decision variables (identification, learning) | an `Expr` of B-coefficients, usually `interp.BSpline(..., coeffs=c)` with no fit | none |

**In-graph fits for `Expr` data.** They depend on the kind:

- **Kinds linear in the data** (`zoh`, `linear`, `cubic` for every `bc`, `spline`, `smooth_linear`,
  `smoothing` with a fixed λ) give `c = M y` with a constant `M`.
  - Up to `n ≤ 256` points per axis, `M` is a dense constant matrix, applied one axis at a time
    (mode-`d` products) on tensor grids.
  - Above that, the cubic kinds solve their tridiagonal system with a Thomas-algorithm `scan`
    (not-a-knot by the usual row elimination, periodic by Sherman-Morrison). Its derivative is
    another tridiagonal solve, through `custom_derivative`.
- **Local nonlinear kinds** (`pchip`, `akima`, `makima`, `steffen`) compute their slopes from `y`
  with vectorized elementwise ops (`where`, `minimum`, `copysign`). Their derivative with respect
  to `y` follows `sc.options(nonsmooth=...)` at the switching points.

### 3.9 Points known when the graph is built

- **`s.basis(points)`** returns the SciPy sparse design matrix of the B-spline: `BSpline.design_matrix`
  or `NdBSpline.design_matrix`, composed with `M` for fitted `Expr` data.
- **`s.at(points)`** returns `SparseMatrix.from_scipy(B).matvec(c)`, so `jacobian_sparsity` equals
  the pattern of `B`. For NumPy coefficients it is simply a constant.

This is the path for:

- B-spline input parametrizations in an OCP;
- fitting a table to measurements;
- identifying a static nonlinearity at the sample points.

### 3.10 Spline calculus and the inverse

These are built on the B-form when the graph is built, and work for constant or `Expr` coefficients:

- `derivative(nu, axis)` and `antiderivative(axis)`;
- `integrate(a, b)`: over a box with numeric bounds it returns a number or a linear `Expr`; with
  `Expr` bounds it goes through the antiderivative.

`inverse()` exists for 1-D curves that are strictly monotone (checked when the graph is built for
NumPy coefficients; for `Expr` coefficients the user asserts it):

- degree ≤ 1: the exact table with the axes swapped;
- higher degree: safeguarded Newton in a `while_loop`, started from the inverse of the linear
  table, with `custom_derivative` supplying `dx/dy = 1/f'(x)`.

### 3.11 Shape-constrained fitting (`interp.constrained`)

A least-squares B-spline fit with a smoothness penalty and linear constraints on the
coefficients. The conditions are sufficient ones:

- **monotone:** the first differences of the coefficients are ≥ 0 (or ≤ 0);
- **convex or concave:** the knot-weighted second differences of the coefficients are ≥ 0 (or ≤ 0);
- **bounds:** on the coefficients, by the convex-hull property;
- **pointwise equalities:** on values or derivatives, from rows of the basis;
- **periodic:** equalities between coefficients.

It is built as an `sc.qp_problem` and solved by the PIQP backend when the graph is built, for
NumPy data, and returns a `Spline` with constant coefficients. Fitting online in generated C waits
for the generated IPM (deferred API-183).

### 3.12 What the generated C needs

- **Large constant tables.** SP4 measures lowering time, C source size and compile time on a 1-D
  table of 10⁴ points, a 2-D table of 256², and a 3-D table of 64³ in basis form. Above a
  threshold `interpolant` warns and suggests passing the table as an input. The CasADi pairs show
  what CasADi's C does at the same sizes.
- **Tables as inputs for AOT deployment.** The coefficients (or the data, with the fit in the
  graph) are an ordinary Function input with a fixed shape. The header documents their layout,
  and `f.pack(values)` produces the flat buffer.
- **`float32` tables.** For embedded targets: supported where `cast` and `take` already are, and
  tested at 1e-6.
- **Exact constants.** Parity at 1e-14 relies on exact float printing; SP1 checks that `_c_float`
  round-trips.

## 4. Compiler changes (SP0)

Three small changes, all inside the rules of "Where to add things".

1. **A derivative for `FLOOR` and `CEIL`.** It is zero, which is exact everywhere except at the
   jumps, a set of measure zero. The change is:
   - forward and reverse rules in `ad/forward.py` and `ad/reverse.py`;
   - under `sc.options(nonsmooth="error")` they keep raising, since that option means "refuse
     nonsmooth derivatives";
   - `ad/sparsity.py` records no dependence through them, so a Jacobian's pattern does not carry a
     structural entry that is always zero;
   - `docs/guide/options.md` gets one line.

   Periodic extrapolation needs this. So does any user code that wraps an angle.
2. **Forward mode through integer index arithmetic.** `_jvp_many_structural` (`ad/forward.py`)
   returns no tangent for nodes whose type carries no derivative (integer and bool dtypes, and
   `diff=False`) instead of forming a mixed-dtype product. The test is the broadcast `int64` index
   build of §1.2 under `sc.jacobian` and `sc.hessian`, for a single point and for a batch.
3. **Finite indices from floats.** A test pins down the order the interpolation code relies on:
   the clamp by `minimum`/`maximum` (C `fmin`/`fmax`) comes before the cast, so `NaN` and `±inf`
   never reach it (§3.5). The test runs `NaN`, `±inf` and `1e300` through the generated C. `cast`'s
   docstring states that a non-finite float cast to an integer type is undefined behaviour in the C
   it generates. The IR does not change.

SP0 also carries the feasibility spike behind the kill criterion. The composite evaluation of
value and gradient, for a 1-D cubic of 1 000 points and a 2-D bicubic of 64², is timed in C
against CasADi's generated C with the harness in `perf_2026_09_27_integrators/` (`time_entry.c`).

## 5. Examples

### 5.1 Notebooks, `examples/interp/`

These follow the integrators and MPC notebooks:

- each runs top to bottom through `tests/integration/test_notebooks.py` (new entries in `NAMES`,
  and in `SOLVER` where they need a solver);
- each ends with a cell of assertions against its references;
- each writes its C to `examples/generated/interp/`.

| Notebook | Content | References and CasADi |
| --- | --- | --- |
| `interpolation_kinds.ipynb` | Every kind on two 1-D data sets (Runge-like, and steps with a plateau): overshoot, continuity, monotonicity, every extrapolation mode, derivatives up to the degree, calculus (`integrate`, `inverse`) | SciPy for every kind; CasADi `linear`, `bspline` and `smooth_linear` overlaid where they exist, and the `bspline` zero outside shown |
| `lookup_tables_nd.ipynb` | A heat-pump COP(T_out, T_supply) map (2-D), a motor-efficiency map η(ω, τ) (2-D), an aerodynamic C_L(α, Mach, δ) table (3-D): search strategies, `pp` against `basis`, the batch path, C export with the table baked in and with the table as an input (a C `main` checked against the JIT) | SciPy RGI / `NdBSpline`; timings from pair P1 |
| `learning_tables.ipynb` | A table as a runtime input, calibrated from scattered measurements by least squares through the table; a Hammerstein model whose static nonlinearity is a cubic B-spline with decision-variable coefficients; Jacobian sparsity at symbolic against static points | finite differences, SciPy least squares on the design matrix; pairs P2 and P3 |
| `shape_constrained.ipynb` | A monotone battery OCV(SOC) curve from noisy data; a convex fuel-cost curve; a bounded efficiency map; comparison with local shape-preserving kinds (PCHIP, Steffen) | `trust-constr` on small cases; KKT residuals; no CasADi counterpart |
| `contouring_control.ipynb` | MPCC around an FSDS track (data copied from the race-car benchmark): a periodic arc-length cubic giving centre line, heading and curvature as one vector-valued spline, progress θ as a state, a closed-loop lap | pair P4 |
| `spline_trajectories.ipynb` | A B-spline input parametrization of a cart-pole swing-up (static points: `at()`), velocity and acceleration limits through `derivative()`, Bézier convex-hull constraints for obstacle avoidance on a double integrator | a hand-built design matrix; the direct transcription it replaces |

### 5.2 CasADi pairs, `examples/interp/pairs/`

These follow the `examples/casadi` protocol:

- `<name>_casadi.py` and `<name>_scaly.py`, each with `build(verbose=False)` returning `run`;
- `run()` returns a dict of NumPy arrays;
- `compare.py` reports agreement, code lines, setup time and run time, with a CasADi JIT column
  where it applies.

`examples/casadi/compare.py` gains a `--dir` argument and nothing else. The existing README stays
about CasADi's own examples (§12, question 4).

| Pair | Problem | CasADi side | Scaly side | Agreement |
| --- | --- | --- | --- | --- |
| P1 `lut_eval` | 1-D non-uniform cubic (1 000 points), 2-D bicubic 64², 3-D trilinear 20³: value, gradient and (2-D) Hessian at 10⁴ random points, and at knots and outside | `interpolant` (`bspline`, `linear`) with `map`, `lookup_mode` set to match | `interpolant`, batch evaluation | 1e-12, with queries kept inside the grid for `bspline` |
| P2 `table_calibration` | A 2-D 12×12 linear table fitted to 3 000 scattered noisy measurements, IPOPT | parametric `interpolant` with `inline=True`. The default (derivative zero) is also run once and what IPOPT does with it is reported (it is expected to stop at the initial point) | `interpolant` over `Expr` data, the static path | 1e-6 on the table |
| P3 `hammerstein_sysid` | Static nonlinearity as a cubic B-spline (20 coefficients) followed by second-order linear dynamics, 2 000 samples, single shooting, IPOPT | `bspline` MX node, MX coefficients, `inline=True`; `mapaccum` | `BSpline(coeffs=c)` inside a `scan` | 1e-6 on parameters and cost |
| P4 `contouring_mpc` | MPCC horizon of 40 on an FSDS track, then 200 closed-loop steps | `interpolant("bspline")` on the track padded by one lap (CasADi has no periodic spline), `Opti` + IPOPT | `interpolant(bc="periodic", extrap="periodic")` + IPOPT | first solve 1e-6; closed-loop progress to 1e-4 |
| P5 `heat_pump_mpc` | 24 h building MPC at 15 min: bicubic COP(T_out, T_supply) inside the dynamics, hourly price and temperature forecasts as ZOH tables sampled at the MPC's times | `interpolant("bspline")` for COP, `interp1d(..., "floor")` for the forecasts | `interpolant(kind="cubic")`, `kind="zoh"` with `.at()` | 1e-6 |

## 6. Documentation and todo

- A guide page `docs/guide/interp.md` covering:
  - the kinds table;
  - extrapolation;
  - derivatives and sparsity (the symbolic against static point distinction);
  - where coefficients come from;
  - codegen advice;
  - a short section for CasADi users (data order, the zero-derivative default, the zero outside).
- An API page `docs/api/interp.md`, `zensical.toml` entries under Guide and API, and a row in
  `docs/api/index.md`.
- A section "Interpolation and lookup tables" in `examples/README.md`, and
  `examples/interp/README.md`.
- `docs/guide/options.md` gets the `floor`/`ceil` line from SP0.
- **`internal/todo.md`:** one item per PR. "Next id" read 158, but `claude/edge-case-fixes` holds C-158 to
  C-161, so this plan takes C-170 to API-185 and leaves 162 to 169 free.

## 7. Phases

One commit per PR on `claude/interp`. Each PR follows the handoff procedure:

- a todo id;
- tests, then mutation checks: break by hand, see red, restore;
- a benchmark script in `internal/notes/perf_2026_MM_DD_interp/` with a row in its `README.md`;
- an HTML report `internal/notes/interp_spN_report.html`;
- the node-ID baseline regenerated.

| PR | Todo | Content | Gate |
| --- | --- | --- | --- |
| SP0 | C-170 | Zero derivative for `FLOOR`/`CEIL` (forward, reverse, sparsity; still raising under `nonsmooth="error"`); the structural-JVP fix for integer nodes; the clamp-before-cast test; the C-timed feasibility spike (§4) | Periodic wrap `sin(x - T floor(x/T))`: gradient and Hessian match the analytic ones to 1e-14 off the jumps. `nonsmooth="error"` still raises. The broadcast `int64` index build differentiates under `jacobian` and `hessian`. Full suite green. Spike numbers recorded for the gate in §10 |
| SP1 | API-171 | Package skeleton; `grid.py` (validation, uniformity detection, `uniform`/`count`/`binary` search, sanitizing); `spline.py` (the B-form, `pp` and `basis` strategies, batch and vector output, `index=` sharing, the five extrapolation modes, `to_scipy`); `fit.py` for `nearest`, `zoh`, `linear`, `cubic` (all four `bc`) and `spline` k = 1 to 5, NumPy data, 1-D to 4-D | Values against SciPy to 1e-13 relative, gradient and Hessian against SciPy's derivative splines to 1e-11, on random grids, at knots, at ±1 ulp, outside, with `NaN`/`±inf`. Every search gives the same index as `searchsorted` on 10⁶ adversarial points. CasADi parity for `linear` (inside and extended) and `bspline` k = 1, 3, 5 (inside) to 1e-13 in 1-D, 2-D and 3-D. One search node shared by value, gradient and Hessian in the lowered program |
| SP2 | API-172 | `pchip`, `akima`, `makima`, `steffen` (1-D); `smooth_linear`; `interp.smoothing` (P-splines of any degree and dimension with GCV, and `method="cubic"` matching `make_smoothing_spline`); calculus (`derivative`, `antiderivative`, `integrate`); `inverse()` | PCHIP, Akima and makima against SciPy to 1e-13; Steffen against the published formulas, with monotone output on 10⁴ random monotone data sets. `smooth_linear` against CasADi to 1e-12, or the fallback of §3.4 recorded. Integrals against `scipy.integrate.quad` of the SciPy spline. The inverse round-trips to 1e-12, and its derivative is 1/f′ |
| SP3 | API-173 | `interp.BSpline(coeffs=Expr)`; `Expr` data for every kind (dense `M`, the tridiagonal `scan` above the threshold, local slope formulas); `basis()` and `at()`; hoisting of a broadcast table's fit out of a `vmap` | Jacobians with respect to coefficients and data against finite differences to 1e-7 and against the dense `M` to 1e-12. `jacobian_sparsity` of `at()` equals the pattern of `BSpline.design_matrix`. Against CasADi `inline=True`: equal to 1e-12. The fit of a broadcast table runs once per call (checked in the program dialect) |
| SP4 | API-174 | `bucket` search, and hint search if it wins; `auto` thresholds (`N_count`, the pp storage budget) set by measurement; large-table codegen measurements and the warning; table-as-input AOT with a header layout note; `float32` | C timings against CasADi `interpolant` (all `lookup_mode`s), `bspline` and `blazing_spline` across sizes and dimensions (§9.2). `auto` never more than 10 % slower than the best fixed choice on the benchmark grid. The dedicated-op gate (§10) evaluated and recorded |
| SP5 | API-175 | `interp.constrained`: monotone, convex/concave, bounds, pointwise value and derivative equalities, periodic; PIQP backend | Constraints hold on a 10⁴-point sampling. The fit equals the unconstrained one to 1e-10 when no constraint is active. It agrees with `trust-constr` to 1e-6 on small cases. `@pytest.mark.solver("piqp")` |
| SP6 | API-176 | The six notebooks (§5.1), the five pairs (§5.2), `compare.py --dir` | Notebooks pass `test_notebooks.py`. `compare.py --dir examples/interp/pairs` reports "yes" for all five. Results table written into `examples/interp/README.md` |
| SP7 | D-177 | Guide and API pages, nav, `examples/README.md` section, options line | `zensical build` clean; every public name has a docstring (`test_import_boundaries.py`) |
| SPR | API-178 | Review round with five agents: IR and AD (SP0, take-based evaluation), numerics (fits, extrapolation, calculus), performance, API and docs, tests and mutation coverage. Then fixes, a summary report, and the tag `interp-complete` | — |

Dependencies:

- SP0 comes before SP1. Its integer-JVP fix is needed for Jacobians of batch evaluations; the
  periodic mode could use the `int64` round trip without it.
- SP1 comes before SP2, SP3 and SP4.
- SP5 needs only SP1.
- SP6 needs SP1 to SP5. P4 and `contouring_control` use `scaly.mpc` if M1 is on the base branch,
  and `sc.problem` directly otherwise.
- SP7 needs SP1 to SP6, and SPR comes last.

## 8. Tests

`tests/interp/` mirrors the package: `test_grid.py`, `test_spline.py`, `test_fit.py`,
`test_constrained.py`, plus `test_casadi_parity.py`. The last uses
`pytest.importorskip("casadi")`, since CasADi is a dev dependency and not a runtime one. SP0's
tests go into `tests/ad/`.

- **References.** Every numeric claim is differential against a SciPy object (§3.4), CasADi (§2),
  finite differences, `quad`, or a closed form such as a quadratic reproduced exactly by a cubic
  spline.
- **Adversarial points.** Every value test includes:
  - each knot exactly, `nextafter` on both sides of it, both ends;
  - far outside (±10³ widths);
  - `NaN`, `±inf`;
  - batches of 10⁴ random points.
- **Derivatives.**
  - Forward, reverse, `sparse_jacobian` and `sparse_hessian` against each other.
  - Against SciPy's `derivative()` away from knots.
  - At knots, the one-sided convention of §3.5.
- **Codegen.**
  - Every test evaluates through the JIT.
  - One `tests/baseline/c/` snapshot for a 1-D linear table of 5 points, so a change to the
    generated shape is seen in review.
  - An AOT test compiles a C `main` with the table as an input and checks it against the JIT.
- **Solver tests** carry `@pytest.mark.solver("piqp")` (SP5) or `("ipopt")` (an integration test,
  `tests/integration/test_interp_ocp.py`, where a small OCP with a bicubic table in its dynamics
  matches a hand-written piecewise-polynomial formulation to 1e-8).
- **Mutation checks,** per PR and reported. Planned mutants:
  - the `searchsorted` side, i.e. the tie at a knot;
  - the index clamp bound;
  - the uniform correction compare;
  - one not-a-knot end row;
  - the PCHIP harmonic-mean weights;
  - the Steffen limiter;
  - the periodic offset;
  - a transposed local basis matrix;
  - one tensor stride;
  - the `fill` value;
  - a knot multiplicity in the Hermite conversion;
  - the direction of a monotonicity row in the QP;
  - the de Boor differencing factor in `derivative()`.

## 9. CasADi comparisons and benchmarks

### 9.1 What is compared

1. **Values and derivatives.**
   - `tests/interp/test_casadi_parity.py` pins every row of §2 that CasADi shares with scaly, to
     1e-13.
   - It pins CasADi's quirks as expected behaviour, so a CasADi upgrade that changes them is
     noticed: the zero outside for `bspline`, the zero derivative with respect to data by default,
     and the rejection of degrees 2 and 4.
   - Data ordering is handled by one helper: CasADi takes the values flattened in Fortran order.
2. **Whole problems.** The five pairs of §5.2 through `compare.py`.
3. **Generated code.** For the same tables, both sides are compared on:
   - lines of C;
   - `static const` bytes;
   - generation time;
   - compile time at `-O2`.

### 9.2 Benchmarks, `internal/notes/perf_2026_MM_DD_interp/`

The protocol is that of `docs/results/fairness.md` and the integrator reports: C-side timing with
`time_entry.c`, rounds interleaved, the fastest sample kept, values checked before timing.

| Script | Scaly side | Baselines |
| --- | --- | --- |
| `bench_eval.py` | value, value+gradient, value+gradient+Hessian; D = 1, 2, 3; degree 0, 1, 3; n per axis in {8, 32, 128, 1 024, 10⁴ (1-D only)}; uniform and clustered grids; every `search` and `strategy` | CasADi `interpolant` code-generated with each `lookup_mode`; `bspline` MX; `blazing_spline` (with `precompute_coeff`); a hand-written C reference for 1-D linear and cubic |
| `bench_batch.py` | batch evaluation of 10³ to 10⁶ points | CasADi `map` (serial and unrolled), SciPy (called from Python, context only) |
| `bench_param.py` | Jacobian with respect to coefficients: symbolic points (dense rows) against static points (`at()`) | CasADi `inline=True` |
| `bench_codegen.py` | lowering time, C lines and bytes, compile time against table size | CasADi `CodeGenerator` on the same tables |
| `bench_fit.py` | fits when the graph is built (SciPy), the in-graph Thomas `scan` against dense `M`, the constrained QP | SciPy, `trust-constr` |

The numbers stay in the per-PR reports. None reach `docs/` unless they go through the results
pages.

## 10. Risks and kill criteria

- **The composite evaluation is too slow, or its code too large.** Measured in SP0, decided at
  the end of SP4. For the 2-D 64² bicubic value+gradient and the 3-D 20³ trilinear value, the
  dedicated op goes ahead if any one of these holds:
  - scaly's time is more than 1.5 times CasADi's generated C at the matching search;
  - its C is more than three times the lines;
  - lowering takes more than 1 s.

  The op would be `ExprOp.SPLINE_EVAL`: a structural op with its own `@lowers` rule, a sparsity
  rule, and derivatives through `derivative()` splines that share the search. It is opened as
  API-179.
- **Rounding at knots on uniform grids.** `floor((x - t0)/h)` can land one cell off. The one-step
  correction and the adversarial tests of §8 cover it.
- **Non-finite input reaching an integer cast.** Covered by the sanitizing of §3.5 and SP0's test.
- **Large constants.** Covered by:
  - the `basis` strategy;
  - the table-as-input path;
  - the SP4 measurements;
  - the cost of hash-consing a 10⁶-entry constant at trace time, measured in SP4 and interned by
    content hash if it shows up.
- **Dense Jacobian rows** with respect to coefficients at symbolic points. This is inherent, and
  documented; `at()` is the sparse route.
- **Fairness against CasADi.** The pairs state `inline=True`, the padded periodic track, the
  queries kept inside the grid for `bspline`, and the matching `lookup_mode`. Timings follow the
  fairness protocol.
- **SciPy drift.** SciPy 1.18 changed RGI `"cubic"`. The tests reference `make_interp_spline`,
  `NdBSpline` and `CubicSpline` directly, with RGI only where its method is stable.

## 11. Deferred, opened as todo items with the plan

- API-179: `ExprOp.SPLINE_EVAL`, only if the gate in §10 trips.
- API-180: scattered-data interpolants (radial basis functions, the Gaussian-process posterior
  mean) behind the same calling convention.
- API-181: piecewise-affine functions on polyhedral or simplicial partitions (explicit MPC laws,
  PWA models) with point location by a search tree.
- API-182: B-spline input parametrizations as a transcription option in `scaly.mpc`.
- API-183: shape-constrained fitting online, through the generated IPM once Tier 4 #30 lands.
- API-184: knots as `Expr`s (CasADi's parametric grid), free-knot fitting.
- API-185: shape-preserving kinds in n-D, as a tensor-product Hermite spline with a stated
  definition. SciPy's RGI `"pchip"` is not one, so the reference would be the definition itself.

## 12. Questions for sign-off

1. **Name:** `scaly.interp`, or `scaly.interpolate` (SciPy's name), or `scaly.splines`?
2. **Default extrapolation:** `linear` for degree ≥ 1 and `clamp` for degree 0 (§3.6), or
   `extend` everywhere for SciPy parity?
3. **Branch:** `claude/interp` from `claude/integrators-mpc` (so the MPC examples can use
   `scaly.mpc`), started now in parallel with M2 to M4, or after them?
4. **CasADi pairs:** in `examples/interp/pairs/` with `compare.py --dir` (proposed), or added to
   `examples/casadi/` under a separate README section?
5. **Process:** the full per-PR procedure (about nine reports), or one report per two PRs?
6. **Scope trims, if you want them:** hint search (SP4), `inverse()` of cubics (SP2), n-D
   smoothing splines (SP2) and `spline_trajectories.ipynb` are the easiest to defer. SP5 is
   independent and could move after SP6.
7. **`FLOOR`/`CEIL` derivative:** zero almost everywhere under `split` and `first`, and still an
   error under `nonsmooth="error"`. Agreed?

## References

- C. de Boor, *A Practical Guide to Splines*, rev. ed., Springer, 2001.
- F. N. Fritsch, R. E. Carlson, "Monotone piecewise cubic interpolation," *SIAM J. Numer. Anal.*
  17(2):238–246, 1980. https://doi.org/10.1137/0717021
- H. Akima, "A new method of interpolation and smooth curve fitting based on local procedures,"
  *J. ACM* 17(4):589–602, 1970. https://doi.org/10.1145/321607.321609
- M. Steffen, "A simple method for monotonic interpolation in one dimension," *Astron. Astrophys.*
  239:443–450, 1990. https://ui.adsabs.harvard.edu/abs/1990A%26A...239..443S
- P. H. C. Eilers, B. D. Marx, "Flexible smoothing with B-splines and penalties," *Statist. Sci.*
  11(2):89–121, 1996. https://doi.org/10.1214/ss/1038425655
- A. Liniger, A. Domahidi, M. Morari, "Optimization-based autonomous racing of 1:43 scale RC cars,"
  *Optim. Control Appl. Meth.* 36(5):628–647, 2015. https://doi.org/10.1002/oca.2123
- J. A. E. Andersson et al., "CasADi: a software framework for nonlinear optimization and optimal
  control," *Math. Prog. Comp.* 11:1–36, 2019. https://doi.org/10.1007/s12532-018-0139-4
- CasADi interpolant API: https://web.casadi.org/api/html/de/dbe/group__interpolant.html
- SciPy `scipy.interpolate`: https://docs.scipy.org/doc/scipy/reference/interpolate.html
