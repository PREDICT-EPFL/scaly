# Solver backends

Four methods ship with scaly. Three drive a solver library and are separate distributions under
`plugins/`, discovered by entry point, so installing one makes its name available to
`sc.opt.solver(problem, method)`. The fourth, IPM, is part of scaly: PIQP's algorithm generated as C,
with no library behind it. This page says what each one is, when to pick it, and what it costs to
ship.

[Solvers](solvers.md) covers problem shapes, calling conventions and nesting a solver in a graph.
[How solvers work](../how_it_works/solvers.md) covers the generated wrappers.

## Choosing one

| | PIQP | IPM | IPOPT | scaly-sqp |
| --- | --- | --- | --- | --- |
| Solves | quadratic programs | quadratic programs | nonlinear programs | nonlinear programs |
| Method | `sc.opt.PIQP(...)`, or `"piqp"` | `sc.opt.IPM(...)`, or `"ipm"` | `sc.opt.IPOPT(...)`, or `"ipopt"` | `sc.opt.SQP(...)`, or `"sqp"` |
| Algorithm | proximal interior point | PIQP's, generated as C | primal-dual interior point, filter line search | sequential quadratic programming, PIQP subproblems |
| Sparse data | `sparse=True` for the problem data | specialised to the problem's patterns always; `sparse=False` condenses the KKT system | sparse Jacobian and Hessian, always | sparse oracles always; sparse subproblems by default, `qp="dense"` to switch |
| Exact Lagrangian Hessian | n/a (the Hessian is your `P`) | n/a | yes, default | yes, default; `hessian="objective"` to approximate |
| Warm start | no, upstream has no C API for it | no, as PIQP | primal always; multipliers only if you ask | primal and dual, always |
| Foreign oracles | no | no | no | yes, see [below](#driving-the-sqp-with-foreign-oracles) |
| Written by | the PIQP authors | this project, after PIQP 0.6.2 | the COIN-OR project | this project |

Pick PIQP when the problem is a quadratic program (QP): a convex quadratic objective and linear
constraints, which is what a linear model-predictive-control formulation or an input-affine safety
filter gives you. A nonlinear-program (NLP) solver will also solve a QP, but you pay for generality
you are not using. Not every barrier-function filter qualifies; this repository's own safety-filter
benchmark has a neural model inside its constraints, so it is an NLP.

Pick IPM for the same problems when the solver has to ship without a library: a microcontroller,
or a build that should carry nothing but generated C. It takes the PIQP library's iterations and
returns its solution; its C is larger and slower to compile, and how its solve time compares depends
on the problem. `examples/opt/qp_solvers` measures both on four QP families.

Pick IPOPT when the problem is nonlinear and you want a solver with two decades of use behind it
and its own documentation.

Pick scaly-sqp when you want the solve itself in generated C with no external solver binary beyond
PIQP, when you want to read and modify the solver, or when your oracles come from somewhere other
than scaly. It is the newest of the three and the least tested.

## PIQP

A proximal interior-point QP solver. Select it with `sc.opt.solver(problem, "piqp")`.

By default scaly assembles the problem for PIQP's dense interface. With `sc.opt.PIQP(sparse=True)`
it instead derives the structural patterns of `P`, `A_eq` and `G_ineq` once and bakes them into the
wrapper as static compressed sparse column (CSC) tables, so each solve refills values only. Which is
faster depends on how sparse your data is; there is no measurement in this repository comparing the
two, so try both if it matters.

Two things only the sparse path does:

- It needs a C compiler when the solver is built, not only at the first solve, because deriving the
  pattern JIT-compiles and evaluates a small probe function.
- It refuses matrices computed from another solver's output. `P`, `A_eq` and `G_ineq` are the ones
  whose pattern has to be derived, and a `extern_call` output is opaque to the dependency analysis,
  so that raises `NotImplementedError`. The vectors `c`, the bounds and the right-hand sides are
  unaffected, since no pattern is derived for them. The dense path has no such restriction, which
  is why the [nesting example](solvers.md#nesting-a-solver-in-a-graph) works.

Scaly extracts the objective Hessian before either PIQP path. For `0.5 * x @ P @ x`, both paths
use `0.5 * (P + P.T)`. The sparse path bakes exactly its upper triangle. Pass a symmetric `P` when
that distinction matters.

Options pass straight through to `piqp_settings` field names, unvalidated in Python. An unknown
name is spliced into the generated wrapper and comes back as a `JitError` from the C compiler
saying `piqp_settings` has no such member.

PIQP's C interface has no warm-start entry point. The fixed solver signature still has the
warm-start inputs, and PIQP ignores them. Repeated solves are still cheaper than the first, because
the wrapper keeps a persistent workspace and re-solves after a value update instead of rebuilding.

## IPM

PIQP 0.6.2's proximal interior-point method, written once in scaly as generated code: Ruiz
equilibration, the initial point, the Mehrotra predictor-corrector with PIQP's proximal updates, and
unscaling, each a loop in the solver's graph. Select it with `sc.opt.solver(problem, "ipm")`.

The solver is specialised when it is built. The structural patterns of `P`, `A_eq` and `G_ineq` are
derived as for sparse PIQP, and which bounds are finite is read at one probe of the parameters, so
a bound that is infinite there is left out of the generated code for good, and one that is finite
there must stay finite: an infinite value at run time ends the solve with `Status.ERROR`. A
problem whose bounds are parameters that are sometimes infinite (an `sc.opt.QP` with one-sided rows,
say) belongs to the PIQP library, which takes them as data. The KKT system is
factored whole by `linalg.SparseLDL` by default; `sparse=False` condenses it and factors it by a
dense Cholesky, as PIQP's dense interface does.

Options are PIQP's settings by name (`eps_abs`, `eps_rel`, `max_iter`, ...), checked when the method
is made: an unknown name raises `TypeError` listing the settings. PIQP's `verbose` is accepted and
does nothing, so a PIQP method's options carry over.

There is no statistics struct: `info` is all it reports, and `sc.opt.solver_stats` is for the
library methods.

## IPOPT

The COIN-OR interior-point NLP solver. Select it with `sc.opt.solver(problem, "ipopt")`.

Scaly feeds it a compact sparse constraint Jacobian and a compact sparse Lagrangian Hessian, both
built through `Function.factory` from the same `sc.factory.SpJac` and `sc.factory.SpHess` requests
any user can make. IPOPT consumes the lower triangle. `sc.opt.solver` asks for that triangle when it
builds the descriptor, so the descriptor pattern and oracle values already match and the generated
wrapper writes them directly into IPOPT's value buffer.

Exact Hessians are the default. `sc.opt.IPOPT(options={"hessian_approximation": "limited-memory"})` gives the
quasi-Newton alternative.

Warm starting is split. `x0` is always IPOPT's starting point. The multipliers are always passed
too, `mult_g` from the equality and inequality duals and `mult_x_L`/`mult_x_U` from the sign-split
box duals, but IPOPT ignores them unless you also pass `options={"warm_start_init_point": "yes"}`.

Options are IPOPT's own, passed as strings, integers or floats. Scaly defaults to `print_level=0`
and `sb="yes"` so the solver stays quiet inside a control loop; the method's `options` override or
extend. A rejected option does not crash. It comes back as `ERROR` statistics with a native status
of `Invalid_Option` and defined outputs.

The status mapping is compiled against the vendored header's own enum constants, so an upstream
change breaks the build instead of silently mapping a status onto the wrong scaly code.

## scaly-sqp

A sequential quadratic programming solver written for this project and emitted as generated C. Its
subproblems go to PIQP, so it needs PIQP's library but no solver binary of its own.

It exists so that the whole solve is readable, modifiable C, so that oracles scaly did not generate
can drive a solver, and so that a second NLP implementation can disagree with IPOPT.

The QP backend is PIQP only. If you know CasADi's `sqpmethod`, where any registered QP solver plugs
in as the subproblem solver, do not expect the same here. scaly-sqp does not go through the QP
plugin contract that `sc.opt.solver(problem, "piqp")` uses; its generated C calls PIQP's C API directly
and borrows the vendored library from `scaly-piqp`. A future QP plugin such as OSQP or HPIPM would
be selectable as a standalone solver but not as the SQP subproblem solver. Making that possible
needs a second, narrower contract for in-C QP subproblems, plus the problem-form reconciliation
CasADi's `conic` layer does. It is backlog work.

Warm starting is unconditional. All four initial iterates are used on every solve: `x0` is clamped
into the variable bounds and taken as the starting point, and the equality, inequality and box
multipliers seed the corresponding duals directly. There is no option to turn this on, which makes
it the simplest of the three in a receding-horizon loop where each solve starts from the last one.

Globalization is a bounded objective/violation filter by default, or an l1 merit function
(`globalization="l1"`). Both account for equality, inequality and variable-bound violations, and
accepted primal and signed dual iterates take the same step length.

The l1 path can also take watchdog steps (`watchdog=N`): up to `N` consecutive iterations are
accepted on an Armijo test without requiring the merit function to decrease monotonically, which
lets the solver through a region where a strict decrease would force very short steps. If that does
not pay off, it restores the last complete checkpoint (primal, dual and subproblem state) and
resumes with an ordinary line search.

Convergence requires primal feasibility plus Karush-Kuhn-Tucker (KKT) stationarity and
signed-multiplier complementarity. A feasible result that ran out of iterations is not promoted to
success. Keep this in mind when comparing status codes against a solver that is more generous.

A failing subproblem does not fail the solve. If PIQP hits its own iteration limit or returns an
infeasibility certificate, the SQP continues from its best iterate and lets globalization and the
KKT test decide. Only PIQP's numerical, unsolved and invalid-settings statuses stop it.

Hessian handling. The descriptor hands scaly-sqp PIQP's upper triangle; the wrapper maps each
`(row, column)` to its canonical `(min(row, column), max(row, column))` slot, so foreign oracles
may supply either one triangle or a full symmetric pattern. Equality-constrained problems
regularize in the constraint-normal space so the added curvature does not damp the feasible
direction, escalating until the model is positive definite. If an exact Lagrangian Hessian still
has negative reduced curvature, the model falls back to the objective Hessian and stops escalating
in the constraint-normal space, so that term is not driven up until the problem disappears. The
ordinary regularization diagonal and the repairing factorization still apply to the fallback model.
See [how solvers work](../how_it_works/solvers.md#what-the-generated-wrapper-contains).

Debugging. `options={"trace": True}` prints to stderr as it goes: the KKT state at the start and
after each accepted step, one line per subproblem solve, and one line per iteration with the step
length, whether it was accepted, and how many trial points it rejected. The option is baked into
the generated C, so turning it on recompiles the solver.

Options. Invalid combinations are rejected when the wrapper is generated, not at solve time.

| Option | Default | Meaning |
| --- | --- | --- |
| `globalization` | `"filter"` | `"filter"` or `"l1"` |
| `watchdog` | `0` | non-negative; only with `l1` |
| `tol` | `1e-6` | primal tolerance |
| `dual_tol` | `1e-4` | stationarity and complementarity tolerance |
| `max_iter` | `50` | iteration limit |
| `hessian` | `"exact"` | `"exact"` or `"objective"` |
| `regularization` | `1e-6` | floor for the modified factorization |
| `line_search_beta` | `0.7` | backtracking factor |
| `merit` | `10` | l1 penalty offset |
| `qp` | `"sparse"` | `"sparse"` or `"dense"` |
| `qp_tol` | `1e-6` | subproblem tolerance |
| `qp_max_iter` | `50` | subproblem iteration limit |
| `trace` | `False` | debug-only per-iteration trace to stderr |

### Driving the SQP with foreign oracles

`scaly_sqp.external_nlp` builds the same solver interface around oracles scaly did not generate.
You supply C source defining the oracle symbols `base`, `grad`, `hess` and `bounds`, plus `jac`
once there are constraints, with the same signatures typed problem construction produces, along
with the sparsity patterns, and get back an ordinary typed `Function`:

```python
from scaly_sqp import external_nlp

solver = external_nlp(
    name="my_nlp",
    n=n, n_eq=n_eq, n_ineq=n_ineq,
    params=[("p", (4,))],
    source=my_c_source,
    raw_symbols={
        "base": "my_base",
        "grad": "my_grad",
        "jac": "my_jac",        # only needed when there are constraints
        "hess": "my_hess",
        "bounds": "my_bounds",
    },
    jac_sparsity=jac_pattern,
    hess_sparsity=hess_pattern,
)
```

This is how you put a solver in front of a model you cannot or do not want to rewrite in scaly.
`scaly_sqp.casadi.build_casadi_external_sqp` is a worked instance: hand it CasADi `Function`
objects for the objective, gradient, Jacobian and Hessian and it code-generates them, wraps each in
an adapter, and hands the result to `external_nlp`. The benchmark suite uses it to run the identical
NLP through the identical SQP solver with only the oracle provider changed. The published
[oracle comparisons](../results/index.md) are the IPOPT ones, which get the same control a
different way, through identical iteration counts on both sides.

## What each one vendors

Scaly builds its solver stacks from source, so a plugin carries its own dependencies instead of
expecting system packages. This matters when you ship.

| Plugin | Builds | Also pulls in |
| --- | --- | --- |
| `scaly-piqp` | PIQP v0.6.4 | Eigen 3.4.1, Blasfeo 0.1.4.3 |
| `scaly-ipopt` | IPOPT 3.14.19 | MUMPS 5.8.2 through COIN-OR ThirdParty-Mumps 3.0.12, METIS 5.2.1 with GKlib, and on Linux OpenBLAS v0.3.28; macOS uses Apple's Accelerate framework |
| `scaly-sqp` | nothing of its own | links PIQP's library, so it needs `scaly-piqp` built |
| IPM (in `scaly`) | nothing | nothing: the generated C is the whole solver |

Every dependency is pinned to a tag or release branch in the plugin's `build_config.json`.

Build requirements differ by plugin. `scaly-piqp` needs Git and a C/C++ compiler.
`scaly-ipopt` needs Git, Make and a C/C++/Fortran compiler, and uses `./configure`. Both get CMake
from PyPI as a build requirement. A cold build of both is 5 to 8 minutes; see
[Contributing](../dev/contributing.md#setup). Later syncs reuse the cached artifacts.

The whole `lib/` directory of a plugin has to travel with it. The build bundles the Fortran runtime
next to `libipopt` so it resolves without a system install.

!!! note "Licensing"
    Scaly and the three plugins are BSD-2-Clause. The `scaly-piqp` and `scaly-ipopt` wheels also
    ship other people's binaries, so each carries its dependencies' license texts under
    `licenses/<dep>/` and a `THIRD_PARTY_NOTICES.md` listing name, pinned version, license and
    upstream URL. The build hook generates both from the cloned sources, so they cannot drift from
    the pins.

## Adding your own

A backend is a small distribution: packaging metadata naming the library and its link flags, plus
one `render_wrapper` hook that emits the C body driving your solver. Core hands it a context and
frames the result with scaly's own statistics storage. The contract is
[Solver plugins](../dev/solver_plugins.md).
