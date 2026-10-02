# The codebase

Use this page to find the files involved in a compiler change and check their allowed dependencies.
The [architecture guide](../how_it_works/architecture.md) explains the compilation process, and
[contributing](contributing.md) covers setup and validation.

## Package map

One line of ownership per module. Every module states the same thing in its own docstring, at more
length. If the two disagree, the docstring wins. The `__init__.py` files are not listed, and
each one's docstring says what its package owns.

```
src/scaly/
  __init__.py            curated public re-exports, and nothing else

  ir/                    dialect definitions, verification, text, pass infrastructure
    types.py             DType, DeviceSpec, TensorType, SparsityPattern, backend support
    expr.py              ExprOp, OP_INFO, Expr, interning, builders, topo, format_expr
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
    lowering/           expression-to-program lowering
      ctx.py            lowering state, rule registry, flat indices, lower_function
      elementwise.py    unary and binary arithmetic with broadcasting
      movement.py       constants, reshapes, slices, transposes, stacks, concatenations
      reduce.py         full tensor reductions
      contraction.py    vector and matrix products
      calls.py          Function calls, mapped calls, solver oracle dependencies
      gather.py         gathers and scatters
    program/             program optimizations                    (ProgramNode -> ProgramNode)
      __init__.py        explicit PASS_PIPELINE and optimize_program
      _common.py         shared buffer references, loop helpers, names, and reachability
      hoist_invariant.py loop-invariant callee work moved before a mapped loop
      scalarize.py       bounded scalar expansion, folding, and scheduling
      fold_tiles.py      exact periodic constant-table compression
      fuse_ranges.py     mapped scalar bodies and their derivative assembly in one range
      combine_scatter_sums.py  shared accumulation for sums of scatters
      fuse_elementwise.py     elementwise producer fusion
      fold_arith.py           constant reads and shared arithmetic identities in loop bodies
      hoist_reciprocals.py     optional invariant divisor reciprocals with scalar-definition checks
      widen_ranges.py         explicit lanes, access layouts, and ordered reductions
      unroll_unit_loops.py    empty- and single-iteration loop removal
      pack_workspace.py      buffer lifetime packing
      coalesce_stores.py     alias-safe adjacent store pairing
      scheduling.py          shared scalar value scheduling
      prepare_scalar.py      statement-local depth bounds before rendering

  function/              the frontend
    model.py             Function declarations, instance registry, lifted transforms
    concrete.py          concrete expression graphs, graph validation, calls and factory
    tree.py              the typed pytree declarations (Tree, arg, group)
    factory.py           the typed derivative specs and the AD each dispatches to
    api.py               the @function decorator and the convenience derivative wrappers
    sugar.py             typed mapped callables, markers, and the private VMAP node builder

  ad/                    derivative construction, all of it inside the expression dialect
    forward.py           jvp, jvp_many
    reverse.py           vjp, vjp_many, and the per-op local adjoint rules
    derivatives.py       jacobian, gradient, hessian, finite_difference
    sparsity.py          structural sparsity patterns and greedy coloring; no AD in it
    sparse.py            sparse_jacobian, sparse_hessian: AD driven by a structural pattern

  codegen/
    abi.py               the pointer ABI: signature, status codes, mangling
    c.py                 ProgramNode -> GNU vector C or C99 lane loops; no lowering policy of its own
    cpp.py               the C++ header: the Buffer template and a namespace per function
    casadi.py            the CasADi 3.8 layer: query functions, CSC encoding, the gather
    __main__.py          compatibility shim for `python -m scaly.codegen`
    solver.py            solver-wrapper framing around a plugin-rendered body
    aot.py               one lowering -> CModule, the C header, the file-writing driver, the CLI
    jit.py               CModule -> compile, cache, dlopen, ctypes dispatch
    toolchain.py         CPU build recipes, native feature detection, compiler discovery, cache root

  solvers/
    model.py             SolverDescriptor and its opaque plain Function
    problem.py           typed backend-free Problem declarations
    solver.py            backend selection and the parameter-based Solver call
    graph.py             the solver queries over a Function graph
    registry.py          plugin discovery and protocol validation
    paths.py             vendored solver library and header discovery
    stats.py             the versioned solver-statistics ABI and SolverStatus
    qp.py nlp.py         quadratic proof/extraction and NLP oracle construction
    _oracle.py           shared oracle-assembly helpers

  viz/
    graph.py             graph JSON, colors and labels; presentation, not compiler text
    recording.py         records a render by registering into codegen's observer hook
    serve.py             the tiny recording browser

  utils/
    env.py               the environment variables and platform facts scaly reads
    names.py             C identifier spelling shared by passes and code generation
    torch_state_dict.py  reading PyTorch checkpoints without depending on torch
```

Solver backends are not in this tree. Each is a separate distribution under `plugins/`
(`scaly-piqp`, `scaly-ipopt`, `scaly-sqp`) discovered through an entry point. See
[Solver plugins](solver_plugins.md). `tests/` mirrors this layout directory for directory.

## Import layers

Every module has an import layer. A module may import modules in its own import layer or a lower
one, never a higher one.

| Import layer | Modules | Why here |
| --- | --- | --- |
| 0 | `utils/*` | Leaves. Environment, identifier spelling and file parsing; no scaly concepts. |
| 1 | `ir/*` | The vocabulary. Both dialects, their verifiers, their text, and the machinery for defining passes. |
| 2 | `passes/affine`, `passes/arith`, `passes/expr`, `ad/sparsity`, `solvers/stats` | Above import layer 1 but below the frontend: index-map recovery, shared arithmetic identities, expression rewrites, structural sparsity, and the solver-statistics layout (which needs nothing from the IR). Nothing here knows what a `Function` is. |
| 3 | `function/{model,concrete,tree}` | Function declarations, concrete graph instances, and typed trees over import layer 1. |
| 4 | `ad/{forward,reverse,derivatives,sparse}`, `function/sugar` | Differentiation, which has to look inside a callee, and the one builder that does too (`vmap`). |
| 5 | `function/{factory,api}`, the rest of `solvers/` | The user-facing request layer: typed derivative specs, the decorator, the solver builders. |
| 6 | `passes/lowering/*`, `passes/program/*` | Lower whole Functions, including their solver callees, and optimize the program dialect. |
| 7 | `codegen/*` | The backend: render, compile, load, dispatch. |
| 8 | `viz/*` | Observes the backend. Nothing in the compiler depends on it. |
| 9 | `scaly/__init__` | The public names sit above everything they re-export. |

`passes/` straddles the frontend: its expression rewrites are below `Function` (import layer 2)
and its lowering is above it (import layer 6). Enforcement is per module, not per package, so
this is legal. Package `__init__` files carry their own entry, set by what they re-export:
`scaly.function` re-exports names from import layer 3 and is import layer 3, while `scaly.passes`
re-exports nothing and sits at import layer 1, below both of its modules.

`tests/test_import_layering.py` checks this table against every Python module. It checks imports
inside functions as well as module-level imports, rejects cycles outside the recorded exceptions,
and verifies that each exception is still needed. It also imports each module first in a fresh
interpreter to catch failures caused by partially initialized packages.

The test excludes `if TYPE_CHECKING:` imports and dynamic plugin loading through `EntryPoint.load`.
These can still create dependencies, as the [text renderer example](#one-dependency-the-table-cannot-see) shows.

### The two sanctioned exceptions

Numerical calls and visualization need connections that do not fit a simple import hierarchy.
They use the following arrangements:

1. `function/concrete.py`, at layer 3, imports `codegen/jit`, at layer 7, inside `_jit()`.
   Explicit compilation, numerical evaluation, recompilation, and solver statistics all use this helper. It is the only
   upward import recorded in `SEAM`, and the import-layer test checks that it remains one statement.
2. Visualization registers a hook with code generation. `codegen/aot.py` defines `RenderObserver`
   and `register_render_observer`, and `viz/recording.py` registers its observer when imported.
   This allows code generation to notify the visualizer without importing it. The import still
   runs from layer 8 to layer 7, so it needs no `SEAM` entry.

`scaly/__init__.py` does not import `scaly.viz`. An ordinary `import scaly` must not enable
recording for an application that never uses visualization.

### One dependency the table cannot see

`ir/text.py` resolves a declared `Function` and renders its concrete instance by reading its `name`, `inputs`, `outputs`, `input_names`, and
`output_names` attributes. Its import exists only under `TYPE_CHECKING`, so the import-layer test
does not see this dependency. If you change those attributes, update the renderer and run
`tests/viz/test_assembly.py` too.

## Where to add things

A scalar math op touches seven files, plus `fuse_elementwise.py` when the op is expensive.

| To add | Touch |
| --- | --- |
| A scalar math op | `ExprOp` and `OP_INFO` in `ir/expr.py`; a verify rule in `ir/expr_spec.py`; AD rules in `ad/forward.py` and `ad/reverse.py`; a matching `ProgramOp` in `ir/program.py` and its category set; an entry in `_UNARY`/`_BINARY` in `passes/lowering/elementwise.py` (the elementwise `@lowers` rule is shared, so no new rule); the C spelling in `codegen/c.py`; and `_EXPENSIVE_OPS` in `passes/program/fuse_elementwise.py` if it lowers to a libm call |
| A structural expression op | the same, minus the elementwise maps, plus its own `@lowers` rule in `passes/lowering/` and a structural rule in `ad/sparsity.py` |
| An expression rewrite | a pattern in `passes/expr.py` |
| An arithmetic identity | a rule in `simplify_arith` in `passes/arith.py`; it reaches expression graphs, scalarized code and loop bodies through their adapters |
| A program-dialect optimization | a module in `passes/program/` and an explicit entry in its `__init__.py` pipeline |
| A program op | `ProgramOp`, its builder, and the right op-category set (`SCALAR_OPS`, `UNARY_FN_OPS`, ...) in `ir/program.py`; a rule in `ir/program_spec.py`; a branch in `ir/text.py` for a statement op (scalars need none); the C spelling in `codegen/c.py` |
| A derivative kind | a frozen `DerivSpec` subclass in `function/factory.py`, plus a wrapper in `function/api.py` |
| A solver backend | a distribution under `plugins/`, an entry point, and a `render_wrapper` hook; see [Solver plugins](solver_plugins.md) |
| A public name | the re-export and `__all__` entry in `scaly/__init__.py`, or a subpackage's `__init__.py` for a lower-level name, and a `:::` entry on an [API page](../api/index.md). Only package `__init__.py` files and `function/factory.py`, which is `sc.factory`, define `__all__` |
| A module | an entry in `IMPORT_LAYERS` in `tests/test_import_layering.py`, a one-line ownership docstring, and a test file in the mirrored place under `tests/` |

## The rules that keep it this way

1. Keep imports within the allowed layers. Record a justified permanent exception in `SEAM`.
   Temporary migration exceptions go in `TOLERATED` and must be removed when the migration ends.
2. Give each concept one owning module. State that responsibility in the module docstring.
3. Do not load an intermediate-representation class under two module paths. Each node class shares
   structurally identical nodes through its own intern table. Loading a second copy creates a second
   table and breaks the identity assumptions used by differentiation and expression reuse.
4. Choose implementations in `passes/lowering/`. `codegen/c.py` renders those choices as C.
   An optimization implemented only in the renderer cannot be inspected or reused by compiler passes.
5. Lower once per render. The header, source, workspace size, and link flags must come from the
   same `_RenderCtx` so that all generated artifacts agree.
6. Keep visualization optional. Register the observer when `scaly.viz` is imported.
7. Raise for unsupported operations or a missing compiler. Scaly has no interpreter fallback.
8. Follow the [versioning policy](versioning.md) when changing public names or generated interfaces.

Two tests carry most of this. `tests/test_import_layering.py` holds the import-layer table and the
exceptions, and `tests/test_import_boundaries.py` pins the public names (`sc.Expr is ir.expr.Expr`,
both dialects verify through the same `Spec` type, retired module paths and vocabulary stay gone).

