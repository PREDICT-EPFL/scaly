# The codebase

The reference tables for changing scaly: which module owns what, which modules may import which,
what a given kind of change touches, and the rules that keep the tree in this shape. Read
[Architecture](../how_it_works/architecture.md) first for what the pieces do; this page is about
where they are and how they may depend on each other. Agents and contributors alike are expected
to follow it, and two tests enforce most of it.

## Package map

One line of ownership per module. Every module states the same thing in its own docstring, at more
length; if the two disagree, the docstring wins. The `__init__.py` files are not listed; each
one's docstring says what its package owns.

```
src/scaly/
  __init__.py            curated public re-exports, and nothing else
  ext.py                 the extension API: registries, protocols and the library-author Function API, collected

  ir/                    dialect definitions, verification, text, pass infrastructure
    types.py             DType, DeviceSpec, TensorType, SparsityType, ScalarType, backend support
    expr.py              the op registry (OpDef, register_op), ExprOp names, Expr, interning, builders, topo, format_expr
    expr_spec.py         expression-dialect verify rules, verify_expr
    program.py           ProgramOp, RangeKind, ProgramNode, interning, builders
    program_spec.py      program-dialect verify rules, verify_program
    spec.py              the shared Rule/Spec table machinery and the one VerifyError
    match.py             Pattern, PatternMatcher, rewrite: how a pass is defined and applied
    text.py              the stable assembly listings for both dialects, plus format_program

  passes/                every concrete IR-to-IR transformation
    affine.py            the affine structure of a concrete index array, for table-free gathers
    arith.py             shared arithmetic identities (simplify_arith, fold_program)
    expr.py              simplify, constant folding, CSE            (Expr -> Expr)
    lowering.py          lower_function, the per-ExprOp rule registry (Expr -> ProgramNode)
    program/             program optimizations                    (ProgramNode -> ProgramNode)
      __init__.py        explicit PASS_PIPELINE, the slots passes are inserted at, and optimize_program
      _common.py         shared buffer references, loop helpers, names, and reachability
      hoist_invariant.py loop-invariant callee work moved before a mapped loop
      scalarize.py       bounded scalar expansion, folding, and scheduling
      combine_scatter_sums.py  shared accumulation for sums of scatters
      fuse_elementwise.py     elementwise producer fusion
      fold_arith.py           constant reads and shared arithmetic identities in loop bodies
      unroll_unit_loops.py    empty- and single-iteration loop removal
      delinearize_loops.py    flat loops whose indices divide the loop variable split into nested loops
      pack_workspace.py      buffer lifetime packing
      coalesce_stores.py     alias-safe adjacent store pairing
      scheduling.py          shared scalar value scheduling
      prepare_scalar.py      statement-local depth bounds before rendering

  function/              the frontend
    model.py             Function (a body instantiated per argument signature) and ConcreteFunction, call composition, graph validation
    tree.py              the typed pytree declarations (Tree, L, G, Record), holes and parameter lists
    factory.py           the typed derivative specs and the AD each dispatches to
    api.py               the @function decorator and the convenience derivative wrappers
    sugar.py             expression builders that need a Function: vmap, scan, while_loop, custom_derivative
    extern.py            the extern-callee protocol: a Function whose C body comes from elsewhere
    method.py            the method interface: Method, MethodRegistry over the scaly.methods entry points, Status, Info

  ad/                    derivative construction, all of it inside the expression dialect
    forward.py           jvp, jvp_many
    reverse.py           vjp, vjp_many, and the per-op local adjoint rules
    derivatives.py       jacobian, gradient, hessian, basis, finite_difference
    sparsity.py          structural sparsity patterns and greedy coloring; no AD in it
    sparse.py            sparse_jacobian, sparse_hessian: AD driven by a structural pattern

  linalg/                sparse and dense linear algebra as generated code
    ops/                 the expression ops linalg registers, each with all its rules (scaly.ext)
      dense.py           cholesky, ldl, lu
      trisolve.py        solve_triangular, and what the dense factorizations share with it
      sparse_ldl.py      sparse_ldl_factor and sparse_ldl_solve over linalg.symbolic's tables
      ragged.py          ragged_add and ragged_dot, the run-time ranges of a sparse column update
    options.py           the linalg option namespace (dense_unroll, sparse_unroll)
    dense.py             the solves built from cholesky, ldl, lu and solve_triangular
    banded.py            tridiagonal and cyclic tridiagonal solves (Thomas) for a matrix known at build time
    stagewise.py         Riccati: the factorization, solve and implicit derivative of a stage-structured LQ problem
    sparse.py            SparseMatrix: a static CSC pattern with Expr values
    symbolic.py          orderings, elimination tree, the pattern of L, left-looking tables, segments
    sparse_factor.py     SparseLDL: the generated left-looking factorization, its solves, implicit derivatives

  interp/                interpolation and lookup tables: tensor-product B-splines as generated code
    grid.py              an axis: knots, the partition and its searches, extrapolation, B-spline tables
    spline.py            BSpline: evaluation (per-cell polynomials or local bases), calculus, the inverse
    fit.py               interpolant and smoothing, from NumPy data or in the graph from Expr data
    constrained.py       least squares under shape constraints, a QP solved by PIQP

  integrators/           discretization of continuous-time models: Runge-Kutta maps over the model's own signature
    model.py             the model contract f(x, ...) -> xdot, and the discrete-time maps built over its signature
    tableau.py           Butcher tableaus: the named Runge-Kutta families and their order conditions
    polynomial.py        Gauss, Radau and Lobatto nodes on [0, 1] and the Lagrange basis over them
    explicit.py          explicit Runge-Kutta steps: fixed, adaptive and symplectic
    implicit.py          implicit Runge-Kutta steps: Newton on the stage equations, implicit-function derivatives
    linear.py            exact discretization of LTI systems (ZOH, FOH), and linearization at a point
    transcription.py     how one interval of a horizon becomes variables, equality constraints and a cost

  mpc/                   model predictive control: OCPs over a horizon, their solvers and control laws
    ocp.py               the OCP: dynamics, costs, constraints, horizon, transcription, and the sc.problem it builds
    controller.py        MPC: the OCP's solver, its control law with the shifted warm start, closed-loop simulation
    polytope.py          polytopes in halfspace form and the linear programs on them (SciPy's HiGHS)
    terminal.py          terminal ingredients: the LQR gain and cost, ellipsoidal and maximal invariant sets

  codegen/
    abi.py               the pointer ABI: signature, status codes, mangling
    c.py                 ProgramNode -> standalone scalar C; no lowering policy of its own
    adapter.py           output adapters: named layers over a rendered module, their hooks and registry
    cpp.py               the cpp adapter: the C++ header, the Buffer template and a namespace per function
    casadi.py            the casadi adapter: CasADi 3.8 query functions, CSC encoding, the gather
    __main__.py          compatibility shim for `python -m scaly.codegen`
    aot.py               one lowering -> CModule, the extern callees' requirements merged, the C header, the file-writing driver, the CLI
    jit.py               CModule -> compile, cache, dlopen, ctypes dispatch
    toolchain.py         C compiler discovery, cache root, the diagnostics report

  solvers/
    model.py             SolverDescriptor, the extern callee of its opaque plain Function
    wrapper.py           that callee's C: the plugin-rendered wrapper, its stats accessor, its build requirements
    problem.py           typed backend-free Problem declarations
    solver.py            backend selection
    graph.py             the solver queries over a Function graph
    registry.py          plugin discovery and protocol validation
    paths.py             vendored solver library and header discovery
    stats.py             the versioned solver-statistics ABI and SolverStatus
    qp.py nlp.py         quadratic proof/extraction and NLP oracle construction
    _oracle.py           shared oracle-assembly helpers
    ipm/                 PIQP's interior-point method written once as generated code
      structure.py       QPStructure (the patterns and which bounds exist) and QPValues
      ruiz.py            Ruiz equilibration of the problem data, as a while_loop
      kkt.py             the KKT system: dense and sparse backends, retries, refinement
      algorithm.py       Solver: the initial point, one iteration, the loop and the result

  viz/
    graph.py             graph JSON, colors and labels; presentation, not compiler text
    recording.py         records a render by registering into codegen's observer hook
    serve.py             the tiny recording browser

  utils/
    env.py               the environment variables and platform facts scaly reads
    ext_api.py           EXT_API_VERSION and the check a package runs against it
    names.py             C identifier spelling shared by passes and code generation
    options.py           sc.options and sc.set_options: user conventions read when a graph is built
    torch_state_dict.py  reading PyTorch checkpoints without depending on torch
```

Solver backends are not in this tree. Each is a separate distribution under `plugins/`
(`scaly-piqp`, `scaly-ipopt`, `scaly-sqp`) discovered through an entry point; see
[Solver plugins](solver_plugins.md). `tests/` mirrors this layout directory for directory.

## Import layers

Every module has an import layer. A module may import modules in its own import layer or a lower
one, never a higher one.

| Import layer | Modules | Why here |
| --- | --- | --- |
| 0 | `utils/*` | Leaves. Environment, identifier spelling and file parsing; no scaly concepts. |
| 1 | `ir/*` | The vocabulary. Both dialects, their verifiers, their text, and the machinery for defining passes. |
| 2 | `passes/affine`, `passes/arith`, `passes/expr`, `ad/sparsity`, `solvers/stats` | Above import layer 1 but below the frontend: index-map recovery, shared arithmetic identities, expression rewrites, structural sparsity, and the solver-statistics layout (which needs nothing from the IR). Nothing here knows what a `Function` is. |
| 3 | `function/{model,tree,extern}` | `Function` itself, a named graph boundary over import layer 1, the pytree declarations, and the protocol a Function with an extern body implements. |
| 4 | `ad/{forward,reverse,derivatives,sparse}`, `function/sugar` | Differentiation, which has to look inside a callee, and the builders that do too (`vmap`, `scan`, `while_loop`, `custom_derivative`). |
| 5 | `function/{factory,api}`, the rest of `solvers/`, `linalg/*`, `interp/*`, `integrators/*`, `mpc/*` | The user-facing request layer: typed derivative specs, the decorator, the solver builders, linear algebra built from expressions and loops, splines, integrators and MPC. |
| 6 | `passes/lowering`, `passes/program/*` | Lower whole Functions, including the Functions extern callees call, and optimize the program dialect. |
| 7 | `codegen/*` | The backend: render, compile, load, dispatch. |
| 8 | `viz/*` | Observes the backend. Nothing in the compiler depends on it. |
| 9 | `scaly/__init__`, `scaly/ext` | The public names sit above everything they re-export, the extension API with them. |

`passes/` straddles the frontend: its expression rewrites are below `Function` (import layer 2)
and its lowering is above it (import layer 6). Enforcement is per module, not per package, so
this is legal. Package `__init__` files carry their own entry, set by what they re-export:
`scaly.function` re-exports names from import layer 3 and is import layer 3, while `scaly.passes`
re-exports nothing and sits at import layer 1, below both of its modules.

`tests/test_import_layering.py` holds this table and reads imports with `ast`, function-local ones
included, since a deferred import inside a function is how the old cycles stayed alive. It also
checks that the graph is acyclic outside the recorded exceptions; that `IMPORT_LAYERS` names
exactly the modules that exist, so a new file cannot slip in unplaced; that every recorded
exception is still a real edge and still a real violation; and that each module imports cleanly
first in a fresh interpreter. That last check catches what a static check cannot: moving a file
can leave a package `__init__` deadlocking on a half-initialized module while every individual
import still looks fine.

The test skips `if TYPE_CHECKING:` imports and does not model dynamic loading (`EntryPoint.load`
in `solvers/registry.py`). Both are ways a dependency can exist without the test knowing;
[one of them](#one-dependency-the-table-cannot-see) matters and is written down below.

### The two sanctioned exceptions

Two places where the call graph and the import-layer order disagree. The first is a recorded
upward import. The second is no import at all.

1. Calling a `Function` compiles it. `function/model.py` (import layer 3) reaches `codegen/jit`
   (import layer 7) through a single deferred import in `_jit()`. Every backend use in the frontend
   (`_flat_numerical_call`, `recompile`, `callee_state`) goes through that one function. This is the
   only entry in `SEAM`, and `test_import_layering.py` asserts it stays one import statement.

2. `viz` observes; `codegen` does not know it exists. The naive wiring would be a `codegen -> viz`
   import, downhill in the call graph and uphill in the import-layer order. Instead
   `codegen/aot.py` owns a `RenderObserver` protocol and `register_render_observer`, and
   `viz/recording.py` registers itself at import time. The only import is `viz -> codegen`, which
   is downward and legal, so nothing needs recording in `SEAM`. Importing `scaly.viz` is what arms
   recording, and marking a target already requires it. For the same reason `scaly/__init__.py`
   re-exports `expr_graph` and `program_graph` lazily through a module `__getattr__`: an eager
   re-export would run `viz/__init__.py` on every `import scaly` and arm the observer for users who
   never asked.

### One dependency the table cannot see

`ir/text.py` renders a `Function`. It reads `.concrete`, `.name`, `.inputs`, `.outputs`,
`.input_names` and `.output_names` through a `TYPE_CHECKING`-only import. The static edge is gone; the structural
dependency is not. Import layer 1 is therefore not free of the frontend contract, and changing
those attributes means changing `ir/text.py` with them. `tests/viz/test_assembly.py` would fail if
the rendering broke, but nothing enforces the direction; only this paragraph records that import
layer 1 knows what a `Function` looks like.

## Where to add things

A scalar math op touches seven files, plus `fuse_elementwise.py` when the op is expensive.

| To add | Touch |
| --- | --- |
| A scalar math op | an `ExprOp` name and a `_BUILTIN_OPS` row (the op's registration) in `ir/expr.py`, with its traits there: `elementwise` naming the `ProgramOp` it computes (which is its lowering), and `expensive` if that is a libm call; its rules, each a row in the table of the module that owns the kind: a verify rule in `_BUILTIN_RULES` in `ir/expr_spec.py`, `jvp` and `jvp_many` in `ad/forward.py`, `vjp` in `ad/reverse.py`, a pattern in `ad/sparsity.py`; a matching `ProgramOp` in `ir/program.py` and its category set; and the C spelling in `codegen/c.py` |
| A structural expression op | the same, minus the `elementwise` trait, plus its own `@lowers` rule in `passes/lowering.py` and the traits that fit (`runtime_index`, `exact_reads`, `update`, `reads`) |
| An op outside the compiler (a library's) | `register_op(name, arity=..., jvp=..., vjp=..., sparsity=..., verify=..., lower=...)` in the module that provides its builder; `OpDef` in `ir/expr.py` gives each rule's signature and its default. Nothing in the compiler changes |
| A linear-algebra op | the same, in a module of `linalg/ops/`, which takes only public names from the compiler (`tests/test_import_boundaries.py`) |
| An expression rewrite | a pattern in `passes/expr.py` |
| An arithmetic identity | a rule in `simplify_arith` in `passes/arith.py`; it reaches expression graphs, scalarized code and loop bodies through their adapters |
| A program-dialect optimization | a module in `passes/program/` and an explicit entry in its `__init__.py` pipeline |
| A program-dialect pass from outside the compiler | `insert_after(anchor, name, fn)` or `insert_before` in `passes/program/__init__.py` |
| A program op | `ProgramOp`, its builder, and the right op-category set (`SCALAR_OPS`, `UNARY_FN_OPS`, ...) in `ir/program.py`; a rule in `ir/program_spec.py`; a branch in `ir/text.py` for a statement op (scalars need none); the C spelling in `codegen/c.py` |
| A derivative kind | a frozen `DerivSpec` subclass in `function/factory.py`, plus a wrapper in `function/api.py` |
| A method of a problem class | a frozen dataclass of its options with `name`, `problem`, `api`, `supports` and `build` (`Method` in `function/method.py`), and an entry point `<domain>.<name>` in the `scaly.methods` group of its distribution's manifest; nothing in the domain changes |
| A solver backend | a distribution under `plugins/`, an entry point, and a `render_wrapper` hook; see [Solver plugins](solver_plugins.md) |
| A Function with a hand-written C body | an object implementing `ExternCallee` in `function/extern.py`, passed to `extern_function`; nothing in the compiler changes |
| A public name | the re-export and `__all__` entry in `scaly/__init__.py` |
| An output adapter (another header language, another consumer's symbols) | a module calling `register_adapter` in `codegen/adapter.py`, and an entry point under `scaly.adapters` naming it |
| A module | an entry in `IMPORT_LAYERS` in `tests/test_import_layering.py`, a one-line ownership docstring, and a test file in the mirrored place under `tests/` |

## The rules that keep it this way

1. Imports go down. The import-layer table above is enforced by `tests/test_import_layering.py`.
   A new upward edge is a design decision: a permanent one goes in `SEAM` with the reason, and one
   being carried across a migration goes in `TOLERATED`, which is currently empty and meant to be
   emptied again whenever it fills.
2. One home per concept, named in its docstring. Directory names do not stop a package from going
   heterogeneous; the one-line ownership statement at the top of each module makes drift visible.
3. Never load an IR class under two module paths. Both node types are hash-consed through a weakref
   intern table keyed by op, args and attrs. Two copies of the class means two tables, and the
   structural identity that AD, CSE and the pass pipeline rely on silently stops holding. This is
   why the restructure shipped no compatibility shim modules for `ir/` types.
4. Decide in one place, spell in another. `passes/lowering.py` chooses the implementation;
   `codegen/c.py` only writes it down. Anything the renderer decides for itself is invisible to the
   pass pipeline and to the visualizer.
5. One lowering per render. Header, source, workspace size and link flags all come off the same
   `_RenderCtx`; two lowerings is how they drift apart.
6. `viz` observes, never participates. The dependency runs backend-to-frontend through a registered
   hook, and importing `scaly.viz` is the only thing that arms it.
7. No silent fallbacks. Unsupported ops, missing compilers and unlowerable cases raise. There is no
   interpreter to fall back to, so generated C is the only semantics.
8. Pre-1.0, breaks are unshimmed. When a name moves it moves; see
   [Versioning](versioning.md).

Two tests carry most of this. `tests/test_import_layering.py` holds the import-layer table and the
exceptions; `tests/test_import_boundaries.py` pins the public names (`sc.Expr is ir.expr.Expr`,
both dialects verify through the same `Spec` type, retired module paths and vocabulary stay gone).

