# Solver backends

Scaly provides interfaces to PIQP, IPOPT, and its own sequential quadratic
programming (SQP) solver. The model declaration and solver input structure stay the
same when you switch backend. Their supported problems, numerical methods,
and warm-start behavior differ.

Options belong to the Python `Solver` and are passed to the generated code on
each call. Solvers for the same problem and with the same name share one compiled
module even when their options differ. Constructing another solver with different
options reuses the compiled wrapper and oracles.

## Problem types and backend selection

| Backend | Supported problem | Installation |
| --- | --- | --- |
| `"piqp"` | convex quadratic objective and linear constraints | `uv add "scaly[piqp]"` |
| `"ipopt"` | smooth nonlinear objective and constraints | `uv add "scaly[ipopt]"` |
| `"sqp"` | smooth nonlinear objective and constraints | `uv add "scaly[sqp]"` |

PIQP is specialized for convex quadratic programs, or QPs. Scaly checks whether
the expressions are quadratic and linear where required, but that check does
not prove convexity. A nonconvex quadratic objective may pass construction and
still be unsuitable for PIQP.

IPOPT and Scaly SQP accept nonlinear programs, or NLPs. They seek a local
solution and do not guarantee a global optimum. Scaly SQP generates its
sequential quadratic programming iteration as C and uses PIQP for the QP
subproblems. IPOPT is a separate native solver.

The examples below share this problem, which all three backends accept:

```python
import numpy as np
import scaly as sc


@sc.problem(vars=sc.arg("x", 2), params=sc.arg("target", 2))
def tracking(x: sc.Expr, target: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    return sc.ProblemSpec(
        minimize=sc.sumsqr(x - target),
        lb=sc.const(0.0),
    )


target = np.array([-1.0, 2.0])
```

The symbolic objective describes \(\min_{x\geq0}\lVert x-\mathrm{target}\rVert^2\).
At the supplied target, the optimum is `[0, 2]`. Selecting a backend builds an
`sc.Solver` from the same `Problem` without running the Python builder again.
The [solver guide](solvers.md) covers call structures and result multipliers.

## PIQP

PIQP uses an interior-point method for convex QPs. Scaly extracts the problem
matrices from the symbolic cost and constraints. By default, it passes dense
matrices. `sparse=True` selects compact sparse matrices:

```python
solve_piqp = sc.solver(tracking, "piqp", options={"sparse": True})
x, *_ = solve_piqp(target)
assert solve_piqp.stats().to_solver_status().ok
print(np.round(x, 6))  # approximately [0. 2.]
```

The sparse option stores only structurally nonzero entries of the Hessian and
constraint matrices. Sparse storage does not necessarily make small problems
faster, so the useful choice depends on the problem and its matrix sizes.

PIQP matrix analysis compiles a probe function during solver construction in
both dense and sparse modes, so it needs a C compiler before the first solve. The QP check
rejects costs, constraints, and bounds that contain another solver call because
it cannot prove the required dependence on the decision variables through that
call. This restriction applies to both dense and sparse PIQP modes.

Other option names correspond to PIQP settings. An unknown name raises a Python
`ValueError` listing the supported settings when `sc.solver` builds the solver.

PIQP ignores `x0` and `warm` because its C interface has no warm-start entry
point. Repeated calls reuse the wrapper's solver workspace and update numerical
data. This workspace reuse is separate from accepting an initial solution.

## IPOPT

IPOPT solves nonlinear problems using an interior-point method. Scaly supplies
the cost, constraints, gradient, sparse constraint Jacobian, and sparse
Lagrangian Hessian. A function supplying one of these quantities is called an
oracle.

Exact second derivatives are the default. IPOPT's `hessian_approximation`
option changes its numerical method to a limited-memory approximation:

```python
solve_ipopt = sc.solver(
    tracking, "ipopt", options={"hessian_approximation": "limited-memory"},
)
x, *_ = solve_ipopt(target, x0=np.ones(2))
assert solve_ipopt.stats().to_solver_status().ok
print(np.round(x, 6))  # approximately [0. 2.]
```

Scaly still constructs the problem's exact Hessian when building the solver.
This option changes IPOPT's use of the Hessian, not which expressions Scaly
must be able to differentiate.

`x0` sets IPOPT's initial variables. `warm` also supplies multipliers, but IPOPT
uses those only with `options={"warm_start_init_point": "yes"}`.
Without `x0` or `warm`, Scaly supplies zeros. The
[warm-start example](solvers.md#warm-starts-and-repeated-solves) shows the
relationship between these inputs.

Options use IPOPT's names and accept strings, integers, or floats. Scaly defaults
to `print_level=0`, which suppresses the convergence log but not the banner
IPOPT prints once per process. `options={"sb": "yes"}` suppresses the banner
too. A positive `print_level` enables IPOPT's convergence log. IPOPT reports rejected options through solver
statistics with native status `Invalid_Option`.

## Scaly SQP

Sequential quadratic programming, abbreviated SQP, repeatedly approximates the
nonlinear problem by a quadratic subproblem. Scaly generates this iteration as
C and calls PIQP to solve each subproblem. The QP backend is always PIQP.
Installing another QP plugin does not make it available inside SQP.

SQP uses the supplied initial variables and all multipliers on every call.
It first clamps the variables to their bounds. Sparse QP subproblems are the
default, and `options={"qp": "dense"}` selects dense ones:

```python
solve_sqp = sc.solver(tracking, "sqp", options={"qp": "dense"})
first = solve_sqp(target, x0=np.ones(2))
assert solve_sqp.stats().to_solver_status().ok
second = solve_sqp(np.array([-0.5, 1.5]), warm=first)
assert solve_sqp.stats().to_solver_status().ok
print(np.round(second[0], 6))  # approximately [0.  1.5]
```

The solver chooses a step length to avoid accepting steps that make convergence
worse. By default, it compares objective reduction and constraint violation
using a filter. The alternative `globalization="l1"` uses an objective plus
constraint-violation penalty. With this alternative, `watchdog=N` permits up to
`N` trial iterations before restoring a checkpoint if progress is inadequate.

Success requires both feasible constraints and the Karush-Kuhn-Tucker, or KKT,
optimality conditions. These conditions measure first-order stationarity and
consistency of the constraint multipliers. A feasible point at the iteration
limit is not reported as success.

With `options={"trace": True}`, the generated solver prints iteration residuals,
QP outcomes, step lengths, and rejected trial counts to standard error. This
option is passed at run time, so enabling it reuses the compiled solver.

| Option | Default | Meaning |
| --- | --- | --- |
| `globalization` | `"filter"` | step acceptance by `"filter"` or `"l1"` penalty |
| `watchdog` | `0` | nonnegative trial-iteration count, only with `"l1"` |
| `tol` | `1e-6` | constraint feasibility tolerance |
| `dual_tol` | `1e-4` | stationarity and complementarity tolerance |
| `max_iter` | `50` | nonlinear iteration limit |
| `hessian` | `"exact"` | `"exact"` Lagrangian or `"objective"` Hessian |
| `regularization` | `1e-6` | diagonal shift added to the QP Hessian, and the pivot floor when repairing it |
| `line_search_beta` | `0.7` | factor used to reduce a rejected step length |
| `merit` | `10` | penalty offset for the `"l1"` mode |
| `qp` | `"sparse"` | `"sparse"` or `"dense"` PIQP subproblems |
| `qp_tol` | `1e-6` | QP tolerance |
| `qp_max_iter` | `50` | QP iteration limit |
| `trace` | `False` | print iteration diagnostics |

Invalid option names, types and combinations are rejected when constructing the
solver. How Scaly SQP repairs an indefinite Hessian and handles a failed subproblem is
described under [Hessian regularization](../how_it_works/solvers.md#hessian-regularization)
and [stopping and failure rules](../how_it_works/solvers.md#stopping-and-failure-rules).

## Native libraries and deployment

The PIQP and IPOPT plugins include native libraries in their wheels. Scaly SQP
uses the PIQP library and has no additional solver binary of its own. The
interfaces call these libraries from generated C, so solver iterations and
function evaluation run without returning to Python for each model value or
derivative.

An ahead-of-time application can link directly to the libraries bundled with the
installed plugins. There is no need for a separate system installation of PIQP
or IPOPT. Deploying with these libraries also avoids accidentally using a
different solver version, because they are the same libraries used when you
test the model through Scaly's JIT path.

The build configuration of the shipped libraries favours speed. PIQP and Blasfeo use CMake's Release build configuration, with Blasfeo enabled
in PIQP. On Linux, IPOPT uses OpenBLAS built with `DYNAMIC_ARCH=1` to select
CPU-specific kernels at runtime. Such choices can outperform a distribution's
more generic library build, depending on its configuration and your workload.
Bundling alone does not guarantee a faster solve.

| Plugin | Solver and bundled dependencies |
| --- | --- |
| `scaly-piqp` | PIQP, Eigen, and Blasfeo |
| `scaly-ipopt` | IPOPT, MUMPS, METIS, GKlib, and platform linear algebra libraries |
| `scaly-sqp` | the library supplied by `scaly-piqp` |

Deployment needs the plugin's complete `lib/` directory, not only its main
solver binary. IPOPT also needs bundled runtime libraries, including the
Fortran runtime.
Each binary plugin includes dependency licenses under `licenses/` and a
`THIRD_PARTY_NOTICES.md` file. Dependency versions are pinned in the plugin's
`build_config.json`.

Platforms without a compatible wheel require a source build, described in
[Contributing](../dev/contributing.md#setup). PIQP requires a C++ compiler.
IPOPT also requires a Fortran compiler.

An exported solver still depends on these native libraries. The
[code generation guide](codegen.md) covers compiling and linking the generated
module into a C or C++ application. Bundled libraries are built for their wheel's
platform, so deploying to another architecture requires compatible builds.

## Plugin architecture and custom backends

Each backend is a separate Python package, so you can install only the
backends you need. Installing a plugin registers a name in
Scaly's `scaly.solvers` entry-point group, which makes that name available to
`sc.solver(problem, name)`. The core library owns the symbolic `Problem`, its
derivatives, and the common solver call structure. The plugin supplies native
library information and generates the C wrapper that drives its solver.

A quadratic-program plugin receives extracted problem matrices. A nonlinear
plugin receives functions for the objective, constraints, and derivatives.
This separation lets you integrate your own solver without waiting for an
official Scaly backend or adding solver-specific logic to the core package.

[Writing solver plugins](../dev/solver_plugins.md) describes the interface and
uses the bundled backends as examples. New solver integrations can be developed
as separate packages. Contributions of new official plugins are welcome and
should follow the same [contribution workflow](../dev/contributing.md) as the
core library.
