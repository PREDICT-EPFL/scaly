# How solvers work

A solver in alloy is not a library call wrapped in Python. It is a `Function` whose body happens to
be a `solver_call`, whose data comes from ordinary alloy functions, and whose solve is a piece of
generated C that drives a vendored solver directly. This page is what happens between
`al.nlp(...)` and a number coming back.

The usage side is [Solvers](../guide/solvers.md); writing a backend is
[Solver plugins](../dev/solver_plugins.md).

## Everything except the solve is an ordinary function

The builders assemble the data the backend needs as normal alloy `Function`s. They lower, optimize,
render and compile exactly like anything else — only the final solve is opaque.

**A QP needs one function.** Its inputs are the free parameters discovered in the symbolic
arguments; its outputs are the flattened QP data: `P`, `c`, `A_eq`, `b_eq`, `G_ineq`, `l_ineq`,
`u_ineq`, `x_lb`, `x_ub`. It is evaluated once per solve and the arrays are handed to PIQP.

**An NLP needs four**, and three of them come from `Function.factory` — the same derivative
machinery any user has:

| Function | Built as | Provides |
| --- | --- | --- |
| `nlp_base` | written directly | `(x, *params) -> (f, g_all)`, where `g_all = concat([h_eq, g_ineq])` |
| `nlp_grad` | `al.gradient(nlp_base, x, "f")` | dense objective gradient |
| `nlp_jac` | `al.spjacobian(nlp_base, x, "g")` | compact sparse constraint Jacobian, pattern attached |
| `nlp_hess` | `al.sparse_lagrangian_hessian(nlp_base, x, ["f", "g"])` | compact sparse Lagrangian Hessian |

For the Hessian, `lam:f` is IPOPT's `obj_factor` and `lam:g` is its stacked multiplier vector at
call time. IPOPT wants only the lower triangle, so the symmetric pattern the factory returns is
filtered to `i >= j` when the solver is built, and the matching values are gathered per call.

A fifth small function, `nlp_bounds`, evaluates the parameter-dependent `x_lb`, `x_ub`, `l_ineq` and
`u_ineq` once per solve.

The consequence worth noticing: **the sparse Lagrangian Hessian an NLP needs is the same
`sphess` any user can ask for.** There is no privileged internal path. Improving that construction
improves both.

## One solve path

There is exactly one: the generated C wrapper. Every `SolverFunction` call compiles a
self-contained translation unit whose wrapper calls `piqp_c` or `IpStdCInterface.h` directly against
the generated oracle kernels, and fills the alloy-owned statistics struct. The same artifact serves
a Python call, a nested solve inside a bigger graph, and an ahead-of-time deployment.

The Python-interleaved backends that used to exist — a nanobind PIQP extension, ctypes IPOPT
callbacks — were deleted. The plugin packages now ship only the vendored native library, its
headers, entry-point metadata, and a wrapper template.

## The pieces

| Module | Owns |
| --- | --- |
| `solvers/qp.py`, `solvers/nlp.py` | the builders: oracle and derivative assembly, and the `SolverDescriptor` |
| `solvers/_oracle.py` | `collect_free_inputs` — walks the graph and returns the unique input nodes in deterministic order |
| `solvers/solver_function.py` | `SolverFunction`: a real `Function` whose outputs are `solver_call` nodes; dispatches through the JIT and refreshes `last_stats` |
| `solvers/registry.py` | entry-point discovery and the `SolverBackend` protocol |
| `solvers/graph.py` | the queries over a graph — which functions are solvers, what a descriptor drives, which backends are reachable, what flags they need |
| `solvers/paths.py` | vendored library and header discovery behind those flags |
| `solvers/stats.py` | the versioned statistics layout and the status enum |
| `codegen/solver.py` | frames each plugin-rendered wrapper body with alloy-owned statistics storage and accessor |
| `plugins/alloy-{piqp,ipopt,sqp}` | the vendored libraries and the per-backend C wrapper templates |

## What the generated wrapper contains

**For PIQP**, the wrapper calls the rendered oracle to fill the QP data, then either transposes
`P`, `A` and `G` into PIQP's column-major layout (the dense path) or hands compact CSC-ordered
values to statically baked pattern tables (the sparse path). It lazily sets up a static workspace
and dispatches `piqp_update_{dense,sparse}` followed by `piqp_solve` on every call.

**For IPOPT**, the generated source emits rather more: a static context struct holding parameter
pointers, the caller's workspace and the timing and evaluation counters; `static const int` arrays
for the sparse Jacobian and lower-triangular Hessian patterns; five `eval_*` callbacks bridging
IPOPT into the rendered kernels, each timed, each counted, and each receiving the caller's `w` —
which is required scratch space, not an optional extra; and an intermediate callback recording the
iteration count.

The wrapper body computes bounds through the rendered bounds function, clamps them to IPOPT's
±2e19 infinity convention, creates the problem, applies the descriptor's options, seeds `mult_g`
from the initial equality and inequality multipliers and `mult_x_L`/`mult_x_U` from the sign-split
box multipliers, solves, writes the seven outputs, and maps `ApplicationReturnStatus` onto the alloy
status enum using the vendored header's own enum constants — so an upstream change breaks at compile
time rather than silently remapping a status.

Failures are handled rather than propagated as garbage: a rejected option surfaces as `ERROR`
statistics with a native status of `Invalid_Option` or `Invalid_Problem_Definition`, and defined
outputs (`x = x0`, zeros elsewhere).

**The SQP plugin** assembles its subproblem through PIQP's sparse interface by default
(`qp="dense"` switches to the dense one, which is slower at every benchmark point measured here).
Because the Jacobian and Hessian patterns are fixed across iterations, it bakes the CSC index tables
at codegen time and refills only values, through a permutation mapping each oracle's own buffer
order onto CSC order.

Its Hessian handling is worth recording. Equality-constrained problems regularize in the
constraint-normal space, `H + rho A' A + regularization I`, accumulated row by row; the added
quadratic is constant over the linearized equality manifold, so it does not damp the feasible
direction. `rho` escalates by decades until the model is positive definite, and if an exact
Lagrangian Hessian still has negative reduced curvature the model falls back to the objective
Hessian and stops escalating `rho` — which keeps exact Hessians as the default without letting that
term grow until it manufactures a false stationarity floor. The fallback model is still
regularized and still goes through the repairing factorization below. Inequality-only models keep their exact active-constraint
curvature.

Positive definiteness is tested and repaired in one modified LDL' pass over the assembled `P`,
using an elimination tree and factor column counts computed at codegen time. A pivot below the
regularization floor is raised to its own magnitude; since only the diagonal is touched, the
factorization equals the assembled `P` plus that diagonal shift, and the shift is exactly zero when
the model was already positive definite. The pass costs `O(nnz(L))`, which is what makes escalating
`rho` cheap enough to do inside every iteration.

## Compiling and linking

When a `SolverFunction` is reachable from the function being compiled, the JIT adds the include,
library, run-path and link flags for every plugin reached. A nested safety filter — QP or NLP —
therefore compiles to a single shared library that links directly against the vendored solver and
is callable from C++ through the universal ABI.

Wrapper state lives in per-symbol statics, which is what makes distinct solvers independent inside
one translation unit and what makes any single solver non-reentrant. See
[the ABI](c_abi.md#solver-bearing-modules).

## Open work

- Warm-start handover into PIQP — upstream has no C API for it.
- Differentiating through `solver_call` via the implicit function theorem. Today derivatives
  through a solve are zero.
- Deduplicating identical solver call sites. Several outputs taken from one `solver.call(...)`
  already lower to a single program-dialect call, but two separate call sites with identical
  arguments are not merged.
