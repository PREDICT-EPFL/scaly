# How solvers work

A solver in Scaly is a plain typed `Function` whose outputs are opaque `EXTERN_CALL` nodes sharing
one `SolverDescriptor`. Its data comes from ordinary generated functions, and a plugin-owned C
wrapper drives the native solver. Python is not part of a solve.

The usage side is [Solvers](../guide/solvers.md). The plugin contract is
[Solver plugins](../dev/solver_plugins.md).

## From a problem to oracles

`@sc.opt.problem` traces a `ProblemSpec` over declared variable and parameter trees. The resulting
`sc.opt.NLP` names no solver. `sc.opt.solver(problem, method)` resolves the method (an instance, a
name in the `scaly.methods` registry, or `"auto"`), and the method builds the descriptor family its
`kind` requires from one of the problem's two normal forms: `sc.opt.nlp_oracles(problem)` or
`sc.opt.extract_qp(problem)`.

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

The method chooses the Hessian triangle: IPOPT asks for lower and scaly-sqp asks for upper.
A problem caches `base`, `grad`, `jac`, the full Hessian construction and `bounds`, plus one
compact Hessian function per requested triangle. Building two solver artifacts from one problem
therefore shares all compatible machinery without giving the artifacts the same C symbols.

### Quadratic-program proof and extraction

A quadratic-program (QP) method first proves the specialization structurally (`prove_qp`):

- the objective Hessian does not depend on the variables;
- each constraint Jacobian does not depend on the variables;
- variable and constraint bounds do not depend on the variables.

The proof uses structural dependency masks over the real derivative expressions. It does not
evaluate at sample values. A rejected problem raises `NotQuadratic` during solver construction, and
a QP method's `supports` reports it as the reason, which is how `"auto"` passes over PIQP for an
NLP.

After the proof, core substitutes `x = 0` to extract:

```text
P = hessian(f, x)
c = gradient(f, x) at x = 0
A = jacobian(h_eq, x)
b = -h_eq at x = 0
G = jacobian(g_ineq, x)
bounds = declared bounds shifted by g_ineq at x = 0
```

That is the problem's `QPForm`, with the objective's constant `f0`. One oracle maps the problem
parameters to its buffers. An `sc.opt.QP` is only a typed matrix-data declaration; it goes through
the same proof and extraction.

With `sc.opt.PIQP(sparse=True)`, core derives fixed compressed sparse column (CSC) patterns for `P`,
`A` and `G` (`QPForm.patterns`). The oracle then emits only compact values in those orders.

## One fixed solver signature

Every solver `Function` takes the same five arguments and returns the same five results:

```text
arguments = variables, box multipliers, equality multipliers, inequality multipliers, parameters
results   = variables, box multipliers, equality multipliers, inequality multipliers, info
```

The wrapper receives them flattened, in that order. The descriptor records `n_var_blocks` so a
plugin can find the fixed groups and scatter its native flat solution into variable leaves. Empty
multiplier categories remain zero-sized arrays. `info` is an `sc.opt.Info`: status, iterations,
objective and primal residual, four scalar outputs that core's frame around the plugin's wrapper
copies from the statistics it filled. The rest of the statistics, timings and evaluation counts
among them, are read through `SolverStats`.

`descriptor_function` creates one `ExprOp.EXTERN_CALL` node per output leaf. All nodes share the
descriptor identity, so lowering emits one wrapper call and distributes its outputs. The
descriptor is the node's and the Function's `extern` attribute: the compiler reaches a solver only
through the extern-callee protocol (`function/extern.py`), and `opt/external/wrapper.py` implements it.

## One solve path

There is exactly one solve path: generated C. The same artifact serves

- a numerical call through the just-in-time (JIT) cache;
- a symbolic call nested in a larger graph;
- ahead-of-time (AOT) C deployment.

A plugin package ships a native library, headers, and a method class (`opt.external.External`)
with its metadata and `render_wrapper`, declared in the `scaly.methods` entry points. It ships no
Python numerical solver.

`opt.ipm` reaches the same generated C without a library or a wrapper. `sc.opt.IPM` extracts the QP,
fixes its structure (the patterns of `P`, `A` and `G` and which bounds are finite), and traces PIQP's
algorithm on the extracted data as an ordinary graph: Ruiz equilibration and the Mehrotra iterations
are `while_loop`s, and the KKT system is factored by `linalg.SparseLDL` or, condensed, by a dense
Cholesky. The condensed matrix is assembled through the patterns' index tables, except that a
constraint matrix with dense rows enters as a dense product, which the lowering runs in register
tiles. The solver `Function` has no extern callee, so it lowers, fuses and ships like any other.

## The pieces

| Module | Owns |
| --- | --- |
| `opt/problem.py` | `ProblemSpec`, `NLP`, bounds, and tracing |
| `opt/nlp.py` | the NLP normal form, `nlp_oracles` |
| `opt/qp.py` | quadratic proof, the QP normal form `extract_qp`, sparse patterns, and `QP` |
| `opt/method.py` | the method registry, `METHOD_API`, and `Info` |
| `opt/solver.py` | `solver`: resolve the method, build the Function |
| `opt/external/method.py` | `External`, the base of every plugin's method class |
| `opt/external/model.py` | `SolverDescriptor`, the extern callee of its plain `Function` |
| `opt/external/graph.py` | solver reachability and link-flag queries |
| `opt/external/paths.py` | vendored library and header discovery |
| `opt/external/stats.py` | the versioned statistics layout |
| `opt/external/wrapper.py` | the wrapper framing, the `Info` outputs, statistics accessor and build requirements behind that callee |
| `opt/ipm/` | `IPM`: PIQP's algorithm as generated code, the `QPStructure` it is specialised to and its KKT backends |
| `plugins/scaly-{piqp,ipopt,sqp}` | method classes and C wrapper generators (a Jinja template for scaly-sqp; PIQP and IPOPT emit C from Python strings) |

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
- Differentiation through `EXTERN_CALL` by the implicit function theorem.
- Deduplicating separate solver call sites with identical arguments.
