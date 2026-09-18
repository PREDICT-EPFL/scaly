# How solvers work

A solver in Scaly is a plain typed `Function` whose outputs are opaque `SOLVER_CALL` nodes sharing
one `SolverDescriptor`. Its data comes from ordinary generated functions, and a plugin-owned C
wrapper drives the native solver. Python is not part of a solve.

The usage side is [Solvers](../guide/solvers.md). The plugin contract is
[Solver plugins](../dev/solver_plugins.md).

## From a Problem to oracles

`@sc.problem` traces a `ProblemSpec` over declared variable and parameter trees. The resulting
`Problem` is independent of a backend. `sc.solver(problem, backend)` selects an entry point and
builds the descriptor family required by its `kind`.

Multi-block variables are concatenated into one internal decision vector for differentiation and
native solver calls. Substitution maps each declared variable symbol to its slice of that vector.
The solver function maps native results back to the declared variable tree.

### Nonlinear-program oracles

IPOPT and scaly-sqp consume the same normalized nonlinear-program (NLP) oracles:

| Oracle | Inputs | Outputs |
| --- | --- | --- |
| `base` | `(x, params)` | objective `f` and stacked constraints `g` |
| `grad` | `(x, params)` | dense objective gradient |
| `jac` | `(x, params)` | compact sparse Jacobian of `g` |
| `hess` | `(x, params, lam:f[, lam:g])` | compact sparse Lagrangian Hessian |
| `bounds` | `params` | variable and inequality bounds |

Equalities come first in `g`, followed by bounded inequalities.

Box-bound leaves have the variables' tree structure. Scalar leaves broadcast, and IEEE negative or
positive infinity represents an absent lower or upper bound until the solver adapter normalizes it.

The backend chooses the Hessian triangle: IPOPT asks for lower and scaly-sqp asks for upper.
A `Problem` caches `base`, `grad`, `jac`, the full Hessian construction and `bounds`, plus one
compact Hessian function per requested triangle. Building two solver artifacts from one problem
therefore shares all compatible machinery without giving the artifacts the same C symbols.

### Quadratic-program proof and extraction

A quadratic-program (QP) backend first proves the specialization structurally:

- the objective Hessian does not depend on the variables;
- each constraint Jacobian does not depend on the variables;
- variable and constraint bounds do not depend on the variables.

The proof uses `_jac_mask` over the real derivative expressions. It does not evaluate at sample
values. A rejected problem raises `NotQuadratic` during solver construction.

After the proof, core substitutes `x = 0` to extract:

```text
P = hessian(f, x)
c = gradient(f, x) at x = 0
A = jacobian(h_eq, x)
b = -h_eq at x = 0
G = jacobian(g_ineq, x)
bounds = declared bounds shifted by g_ineq at x = 0
```

One oracle maps the problem parameters to those QP buffers. `qp_problem` is only a typed
matrix-data declaration; it goes through the same proof and extraction.

With `options={"sparse": True}`, core derives fixed compressed sparse column (CSC) patterns for `P`,
`A` and `G`. The oracle then emits only compact values in those orders.

## One fixed solver signature

Every backend receives the same flattened five-group call:

```text
inputs  = variables, box multipliers, equality multipliers, inequality multipliers, parameters
outputs = variables, box multipliers, equality multipliers, inequality multipliers
```

The descriptor records `n_var_blocks` so a plugin can find the fixed groups and scatter its native
flat solution into variable leaves. Empty multiplier categories remain zero-sized arrays. The
objective and detailed status are reported through `SolverStats`, not through extra function
outputs.

`descriptor_function` creates one `ExprOp.SOLVER_CALL` node per output leaf. All nodes share the
descriptor identity, so lowering emits one wrapper call and distributes its outputs.

## One solve path

There is exactly one solve path: generated C. The same artifact serves

- a numerical call through the just-in-time (JIT) cache;
- a symbolic call nested in a larger graph;
- ahead-of-time (AOT) C deployment.

A plugin package ships a native library, headers, entry-point metadata and `render_wrapper`. It
ships no Python numerical solver.

## The pieces

| Module | Owns |
| --- | --- |
| `solvers/problem.py` | `ProblemSpec`, `Problem`, bounds, and tracing |
| `solvers/nlp.py` | shared NLP oracle construction |
| `solvers/qp.py` | quadratic proof, extraction, sparse patterns, and `qp_problem` |
| `solvers/solver.py` | backend selection |
| `solvers/model.py` | `SolverDescriptor` and its plain `Function` |
| `solvers/registry.py` | entry-point discovery and protocol validation |
| `solvers/graph.py` | solver reachability and link-flag queries |
| `solvers/paths.py` | vendored library and header discovery |
| `solvers/stats.py` | the versioned statistics layout and statuses |
| `codegen/solver.py` | Scaly-owned wrapper framing and statistics accessors |
| `plugins/scaly-{piqp,ipopt,sqp}` | backend metadata and C wrapper generators (a Jinja template for scaly-sqp; PIQP and IPOPT emit C from Python strings) |

## What the generated wrapper contains

The PIQP wrapper calls the QP oracle once per solve. Its dense path transposes row-major matrices
into PIQP's column-major buffers. Its sparse path uses baked CSC tables and updates values only.
PIQP keeps one static workspace and calls `piqp_update_dense` or `piqp_update_sparse` before
`piqp_solve`.

The IPOPT wrapper emits a static context, Jacobian and Hessian index tables, and callbacks into the
rendered oracles. Each callback receives the caller's packed workspace, updates timing and
evaluation counters, and writes directly in the descriptor's sparse order. The wrapper maps the
typed warm start to IPOPT's primal, constraint-dual and sign-split box-dual buffers.

The scaly-sqp wrapper assembles each subproblem for PIQP. It bakes permutations from oracle
sparsity order to CSC order and refills values at each iteration. Its modified LDL factorization
regularizes the quadratic model while preserving the fixed structural pattern.

Every wrapper fills the Scaly statistics structure on success and failure. Native status values are
mapped through constants from the vendored headers, so an upstream enum change fails at compile
time instead of silently changing meaning.

## Compiling and linking

When a solver function is reachable, AOT rendering and the JIT add the headers, libraries, runtime
paths and link flags for every reached backend. Oracles, wrapper and host function occupy one
translation unit.

Wrapper state is stored in per-symbol statics. Distinct solver artifacts in one translation unit
are independent, but one compiled solver is not reentrant. See
[the generated interface](generated_interface.md#solver-bearing-modules).

## Open work

- Warm-start handover into PIQP; its C API does not expose one.
- Differentiation through `SOLVER_CALL` by the implicit function theorem.
- Deduplicating separate solver call sites with identical arguments.
