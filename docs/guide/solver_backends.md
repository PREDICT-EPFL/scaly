# Solver backends

Three backends ship with alloy. Each is a separate distribution under `plugins/`, discovered by
entry point, so installing one is what makes its name available to `al.solver(problem, backend)`. This page
is what each one is, when to reach for it, and what it costs you to ship.

How to *use* a solver — problem shapes, calling conventions, nesting one in a graph — is
[Solvers](solvers.md). How the machinery works underneath is
[how solvers work](../how_it_works/solvers.md).

## Choosing one

| | **PIQP** | **IPOPT** | **alloy-sqp** |
| --- | --- | --- | --- |
| Solves | quadratic programs | nonlinear programs | nonlinear programs |
| Reached through | `al.solver(problem, "piqp")` | `al.solver(problem, "ipopt")` | `al.solver(problem, "sqp")` |
| Method | proximal interior point | primal-dual interior point, filter line search | sequential quadratic programming, PIQP subproblems |
| Sparse data | `options={"sparse": True}` for the problem data | sparse Jacobian and Hessian, always | sparse oracles always; sparse subproblems by default, `qp="dense"` to switch |
| Exact Lagrangian Hessian | n/a (the Hessian is your `P`) | yes, default | yes, default; `hessian="objective"` to approximate |
| Warm start | no — upstream has no C API for it | primal always; multipliers only if you ask | primal and dual, always |
| Foreign oracles | no | no | yes, see [below](#driving-the-sqp-with-foreign-oracles) |
| Written by | the PIQP authors | the COIN-OR project | this project |

**Reach for PIQP** when your problem is genuinely a QP — a convex quadratic objective and linear
constraints, which is what a linear-MPC formulation and an input-affine safety filter give you.
Handing a QP to an NLP solver works, but you pay for generality you are not using. (Not every
barrier-function filter qualifies: this repository's own safety-filter benchmark has a neural model
inside its constraints, so it is an NLP.)

**Reach for IPOPT** when the problem is nonlinear and you want a solver with two decades of use
behind it and its own extensive documentation. Select it with `al.solver(problem, "ipopt")`.

**Reach for alloy-sqp** when you want the solve itself in generated C with no external solver
binary beyond PIQP, when you want to read and modify the solver, or when your oracles come from
somewhere other than alloy. It is the newest of the three and the least battle-tested.

## PIQP

A proximal interior-point QP solver. Select it with `al.solver(problem, "piqp")`.

**Dense by default, sparse on request.** `al.solver(problem, "piqp")` assembles the problem for PIQP's dense
interface unless you pass `options={"sparse": True}`, which instead derives the structural patterns of `P`,
`A_eq` and `G_ineq` once and bakes them into the wrapper as static CSC tables, so each solve
refills values only. Which is faster depends on how sparse your data actually is; there is no
measurement in this repository comparing the two, so try both if it matters.

Two things only the sparse path does, and they are the reasons to think before switching:

- **It needs a C compiler when the solver is *built*,** not just at first solve. Deriving the
  pattern means JIT-compiling and evaluating a small probe function.
- **It refuses matrices computed from another solver's output.** `P`, `A_eq` and `G_ineq` are the
  ones whose pattern has to be derived, and a `solver_call` output is opaque to the dependency
  analysis, so that raises `NotImplementedError`. The vectors — `c`, the bounds, the
  right-hand sides — are unaffected, since no pattern is derived for them. The dense path has no
  such restriction at all, which is why the
  [nesting example](solvers.md#nesting-a-solver-in-a-graph) works.

**Symmetry.** Alloy extracts the objective Hessian before either PIQP path. For `0.5 * x @ P @ x`,
both paths therefore use `0.5 * (P + P.T)`. The sparse path bakes exactly its upper triangle. Pass a
symmetric `P` when that distinction matters.

**Options** pass straight through to `piqp_settings` field names, unvalidated in Python. An unknown
name is spliced into the generated wrapper and surfaces as a `JitError` from the C compiler saying
`piqp_settings` has no such member.

**No warm start.** PIQP's C interface exposes no warm-start entry point, so the fixed typed solver signature includes
warm-start inputs, which PIQP ignores. Repeated solves are still cheaper than the
first: the wrapper keeps a persistent workspace and re-solves after a value update rather than
rebuilding.

## IPOPT

The COIN-OR interior-point NLP solver. Select it with `al.solver(problem, "ipopt")`.

Alloy feeds it a compact sparse constraint Jacobian and a compact sparse Lagrangian Hessian, both
built through `Function.factory` from the same `al.factory.SpJac` and `al.factory.SpHess` requests
any user can make. IPOPT consumes the lower triangle. `al.solver` asks for that triangle when it builds
the descriptor, so the descriptor pattern and oracle values already match and the generated wrapper
writes them directly into IPOPT's value buffer.

**Exact Hessians are the default**, and worth keeping. On the safety-filter benchmark they roughly
halve IPOPT's iteration count against a limited-memory approximation — see
[the results](../results/index.md#safety-filter-with-a-solver-in-the-loop). Pass
`options={"hessian_approximation": "limited-memory"}` if you want the other behaviour.

**Warm starting** is split. `x0` is always IPOPT's starting point. The *multipliers* are always
passed too — `mult_g` from the equality and inequality duals, `mult_x_L`/`mult_x_U` from the
sign-split box duals — but IPOPT ignores them unless you also pass
`options={"warm_start_init_point": "yes"}`.

**Options** are IPOPT's own, passed as strings, integers or floats. Alloy defaults to
`print_level=0` and `sb="yes"` so the solver stays quiet inside a control loop; pass `options={...}`
to override or extend. A rejected option does not crash — it surfaces as `ERROR` statistics with a
native status of `Invalid_Option` and defined outputs.

**Status mapping** is compiled against the vendored header's own enum constants, so an upstream
change breaks the build rather than silently remapping a status onto the wrong alloy code.

## alloy-sqp

A sequential quadratic programming solver written for this project and emitted as generated C. Its
subproblems go to PIQP, so it needs PIQP's library but no separate solver binary of its own.

It exists for three reasons: the whole solve is readable and modifiable C rather than a vendored
blob; it accepts oracles that alloy did not generate; and it is a control on the other two — a
second NLP implementation to disagree with.

**The QP backend is PIQP, and only PIQP.** If you know CasADi's `sqpmethod`, where any registered
QP solver plugs in as the subproblem solver, do not expect the same here. alloy-sqp does not go
through the QP plugin contract that `al.solver(problem, "piqp")` uses; its generated C calls PIQP's
C API directly and borrows the vendored library from `alloy-piqp`. A future QP plugin such as OSQP
or HPIPM would be selectable as a standalone solver but would not be usable as the SQP subproblem
solver. Making that possible needs a second, narrower contract for in-C QP subproblems, plus the
problem-form reconciliation CasADi's `conic` layer does, and it is tracked as backlog work.

**Warm starting is unconditional.** All four initial iterates are used on every solve: `x0` is
clamped into the variable bounds and taken as the starting point, and the equality, inequality and
box multipliers seed the corresponding duals directly. There is no option to turn this on, so
alloy-sqp has the simplest warm-start semantics of the three in a receding-horizon loop where each
solve starts from the last one.

**Globalization** is a bounded objective/violation filter by default, or an l1 merit function
(`globalization="l1"`). Both account for equality, inequality and variable-bound violations, and
accepted primal and signed dual iterates take the same step length.

The l1 path can also take *watchdog* steps (`watchdog=N`): up to `N` consecutive iterations are
accepted on an Armijo test without insisting the merit function decrease monotonically, which lets
the solver through a region where a strict decrease would force uselessly short steps. If that
gamble does not pay off, it restores the last complete checkpoint — primal, dual and subproblem
state — and resumes with an ordinary line search.

**Convergence** requires primal feasibility plus KKT stationarity and signed-multiplier
complementarity. A merely feasible result that ran out of iterations is *not* promoted to success —
worth knowing if you are comparing status codes against another solver that is more generous.

**A failing subproblem does not fail the solve.** If PIQP hits its own iteration limit or returns
an infeasibility certificate, the SQP continues from its best iterate and lets globalization and
the KKT test decide. Only PIQP's numerical, unsolved and invalid-settings statuses stop it.

**Hessian handling** is where most of the care went. The descriptor hands alloy-sqp PIQP's upper
triangle; the wrapper maps each `(row, column)` to its canonical `(min(row, column), max(row, column))`
slot, so foreign oracles may supply either one triangle or a full symmetric pattern.
Equality-constrained problems regularize in the constraint-normal space so the added curvature does not damp
the feasible direction, escalating until the model is positive definite; if an exact Lagrangian
Hessian still has negative reduced curvature, the model falls back to the objective Hessian and
stops escalating in the constraint-normal space rather than driving that term up until the problem
disappears. The ordinary regularization diagonal and the repairing factorization still apply to the
fallback model. See [how solvers work](../how_it_works/solvers.md#what-the-generated-wrapper-contains).

**Debugging.** `options={"trace": True}` prints to stderr as it goes: the KKT state at the start
and after each accepted step, one line per subproblem solve, and one line per iteration summarizing
the line search — the step length, whether it was accepted, and how many trial points it rejected.
Because the option is baked into the generated C, turning it on recompiles the solver.

**Options.** Invalid combinations are rejected when the wrapper is generated, not at solve time.

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

`alloy_sqp.external_nlp` builds the same solver interface around oracles alloy did not generate.
You supply C source defining the oracle symbols — `base`, `grad`, `hess` and `bounds`, plus `jac`
once there are constraints, with the same signatures typed problem construction produces — along with the
sparsity patterns, and get back an ordinary typed `Function`:

```python
from alloy_sqp import external_nlp

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

This is how you put a solver in front of a model you cannot or do not want to rewrite in alloy.
`alloy_sqp.casadi.build_casadi_external_sqp` is a worked instance of it: hand it CasADi `Function`
objects for the objective, gradient, Jacobian and Hessian and it code-generates them, wraps each in
an adapter, and hands the result to `external_nlp`. The benchmark suite uses it to run the identical
NLP through the identical SQP solver with only the oracle provider changed — a controlled
comparison rather than two different solves. (The published
[oracle comparisons](../results/index.md) happen to be the IPOPT ones, which get the same control a
different way: identical iteration counts on both sides.)

## What each one vendors

Alloy builds its solver stacks from source rather than depending on system packages, so a plugin
carries its own dependencies rather than expecting them installed. This matters when you ship.

| Plugin | Builds | Also pulls in |
| --- | --- | --- |
| `alloy-piqp` | PIQP v0.6.2 | Eigen 3.4.1, Blasfeo (**unpinned** — see below) |
| `alloy-ipopt` | IPOPT 3.14.19 | MUMPS (ThirdParty 3.0.12), METIS (ThirdParty 2.0.1), and on Linux OpenBLAS v0.3.28 — macOS uses Apple's Accelerate framework |
| `alloy-sqp` | nothing of its own | links PIQP's library, so it needs `alloy-piqp` built |

Everything is pinned to a tag except **Blasfeo**, which is cloned from its default branch. Two
builds on different days can therefore pick up different Blasfeo revisions, which is worth knowing
if you need a reproducible artifact.

Build requirements differ by plugin: `alloy-piqp` needs Git, CMake and a C/C++ compiler;
`alloy-ipopt` needs Git, Make and a C/C++/Fortran compiler, and uses `./configure` rather than
CMake. A cold build of both is 5 to 8 minutes; see
[Installation](installation.md#the-solvers). Later syncs reuse the cached artifacts.

The whole `lib/` directory of a plugin is what has to travel, not just the solver library —
the build bundles the Fortran runtime next to `libipopt` so it resolves without a system install.

!!! note "Licensing"
    <!-- TODO: fill in once the project's own licence and the vendored terms are settled. -->
    Alloy's own licence is not yet decided, and neither is what to state here about the vendored
    stacks above. If you are evaluating alloy for a context where the licence of the solver closure
    matters, ask before assuming.

## Adding your own

A backend is a small distribution: packaging metadata naming the library and its link flags, plus
one `render_wrapper` hook that emits the C body driving your solver. Core hands it a context and
frames the result with alloy's own statistics storage. The contract is
[Solver plugins](../dev/solver_plugins.md).
