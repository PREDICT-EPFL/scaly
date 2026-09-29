# Nonlinear equations

`scaly.roots` solves systems of equations `F(z; p) = 0` and nonlinear least-squares problems
`minimize 1/2 |r(z; p)|^2` as generated code. As in [Solvers](solvers.md), a problem names its
unknowns, parameters and residual and no method; `sc.roots.solver(problem, method)` builds the
`Function` that solves one with the other. The iteration is a loop in the generated C, so the solve
nests in a larger graph, is `vmap`-ped over a batch, or ships in a module like any other Function.

## Declare a problem

```python
import numpy as np
import scaly as sc

@sc.roots.root(vars=sc.L("z", 2), params=sc.L("p", 2), name="circle")
def circle(z, p):
    return sc.stack([z[0] ** 2 + z[1] ** 2 - p[0], z[0] - p[1] * z[1]])
```

The body returns the residual: an expression, or a tuple of them, flattened and joined. A `Root`
needs as many residuals as unknowns. `sc.roots.least_squares(vars=..., params=...)` declares a
`LeastSquares` problem the same way, with at least as many residuals as unknowns. `vars` and
`params` are trees as in [Building functions](functions.md), and `params` may be left out when the
residual reads nothing but the unknowns and constants; anything else it reads raises, naming it.

To bound the unknowns, for the bracketing method, return a `RootSpec`:

```python
@sc.roots.root(vars=sc.L("sigma", ()), params=sc.L("quote", 5), name="implied_vol")
def implied_vol(sigma, quote):
    price = call_price(sc.stack([quote[0], quote[1], quote[2], sigma, quote[3]]))  # S, K, T, sigma, r
    return sc.roots.RootSpec(price - quote[4], lb=sc.const(1e-4), ub=sc.const(5.0))
```

`lb` and `ub` have the unknowns' structure and may depend on the parameters, not on the unknowns.

## Solve it

```python
solve = sc.roots.solver(circle)                                  # "auto": Newton for a square system
solve = sc.roots.solver(circle, sc.roots.Newton(tol=1e-12), name="circle_newton")
z, info = solve(np.array([1.0, 1.0]), np.array([4.0, 0.5]))      # warm start, parameters
if not sc.Status(int(info.status)).ok:
    raise RuntimeError(sc.Status(int(info.status)))
```

Every roots solver takes the unknowns' warm start and the parameters (just the warm start when the
problem has none) and returns the solution, with the unknowns' tree, and an `sc.roots.Info`:
`status`, an `sc.Status`; `iter`, the iterations taken; and `residual`, the largest entry of what
the method drives to zero at the returned point, `F` for a root and the gradient `J^T r` for least
squares.

| Method | Problem | Iteration |
| --- | --- | --- |
| `Newton(tol=1e-10, max_iter=50, linear="lu", rtol=0, simplified=False, max_step=None)` | `Root` | `z <- z - F_z^{-1} F`, until `|F|_inf <= tol + rtol |z|_inf` |
| `NewtonBisection(tol=1e-14, max_iter=60, increasing=True)` | `Root`, one bounded unknown | Newton inside the bracket, bisecting whenever a step would leave it |
| `GaussNewton(tol=1e-10, max_iter=50, max_step=None)` | `LeastSquares` | `z <- z - (J^T J)^{-1} J^T r` |
| `LevenbergMarquardt(tol=1e-10, max_iter=100, damping=1e-3)` | `LeastSquares` | `-(J^T J + mu I)^{-1} J^T r`, with Nielsen's update of `mu` |

`"auto"` takes Newton for a square system, the bracketing Newton for one bounded unknown and
Levenberg-Marquardt for least squares; a method is also named by string (`"gauss_newton"`). A
method that cannot solve the problem raises, naming why: Newton refuses bounded unknowns, which it
would ignore.

**Newton's options.** `tol=None` takes exactly `max_iter` iterations as straight-line code, a fixed
amount of work, the choice for control. `linear` names the solve: `"lu"`; `"cholesky"` for a
symmetric positive definite Jacobian, such as the Hessian of a convex energy; `"sparse_ldl"`, the
compact sparse Jacobian factored by `linalg.SparseLDL`, for a large symmetric one. `simplified`
factors once, at the warm start. `max_step` caps each step's largest entry, a damped Newton for a
start far from the root. `rtol` suits a residual measured in the unknowns' units; it is 0 by
default, since a bounded residual would otherwise pass the test as a diverging iterate grows.

**The bracketing Newton** needs a residual that changes sign across the bracket and increases
through its root (`increasing=False` for one that decreases). It stops once a step moves the
unknown by no more than `tol` times the larger of its magnitude and the bracket's first width, and
never leaves the bracket.

## Derivatives

A solver's solution is differentiable in the parameters by the implicit function theorem, never
through the iterations: at the root found, `dz = -F_z^{-1} F_p dp`, one solve in forward mode and
one transposed solve in reverse, exact at the point found, whatever iteration found it. For least
squares the same holds for the stationarity condition `J^T r = 0`, with the full Hessian of
`1/2 |r|^2`. Second derivatives are implicit too; the warm start has none.

```python
p = sc.sym("p", 2)
z, _ = solve(sc.const(np.array([1.0, 1.0])), p)
dz_dp = sc.Function.from_exprs("dz_dp", [p], [sc.jacobian(z, p)], ["p"], ["J"])
```

When Newton's `linear` is `"cholesky"` or `"sparse_ldl"` the rule solves with it, so the derivative
of a large sparse problem stays sparse; otherwise it solves with the dense Jacobian.

`sc.roots.custom_root(residual, z, params, name=...)` gives the same derivative to a root found any
other way: it returns a Function `(params..., zstar) -> z`, the identity on `zstar`, whose rules are
the implicit function theorem's at `residual(zstar, params) = 0`. Call it on the result of your own
loop:

```python
root = sc.roots.custom_root(lambda z, ps: z ** 3 + z - ps[0], sc.sym("z", 3), [sc.sym("p", 3)], name="cubic", names=["p"])
z = root((p, my_loop(p)))
```

## Inside other methods

A method's iteration is also callable on expressions, for code that builds its own problem:
`Newton(...).iterate(residual, z0, params, name=..., linear=...)` and
`NewtonBisection(...).iterate(residual, x0, lo, hi, params, name=..., slope=...)` return the solution
and its `Info` as expressions, and an `Info` the caller never uses costs nothing in the generated
code. `sc.integrators.implicit` solves its stage equations with `Newton.iterate`, passing a `Linear`
that factors its structured stage matrix, and differentiates them with `custom_root`;
`BSpline.inverse()` is `NewtonBisection.iterate` on the spline's cell.
