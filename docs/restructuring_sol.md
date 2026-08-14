# Alloy core restructuring — Sol analysis and plan

- **Author:** Sol
- **Date:** 2026-08-14
- **Profile:** design and architecture; this document does not authorize the restructuring itself

## Executive opinion

The inconsistency is real, and the Program IR migration is the point at which it became visible,
but Program IR is not itself the mistake. The migration introduced a second real language and a
real compiler pipeline into a package whose original layout assumed one small semantic graph. The
new concepts were added mostly as root modules, so the filesystem no longer communicates the
pipeline that the code actually implements.

The best fix is not merely to move files into the rough categories in the prompt. Alloy is missing
one category: **compiler**. `lowering.py` and `passes.py` are neither IR definitions nor C rendering;
they implement the semantic-IR-to-Program-IR compiler. Giving that layer a name makes the remaining
boundaries much simpler.

I recommend:

1. Make the two IR dialects structurally and lexically parallel inside `alloy.core`.
2. Separate the `Function` model from derivative-factory construction and convenience APIs.
3. Put dense and sparse automatic differentiation under one `alloy.ad` package.
4. Put lowering and Program IR optimization in `alloy.compiler`.
5. Make `alloy.codegen.aot` the explicit generated-module path and make
   `alloy.codegen.jit` a thin consumer of it, rather than two render paths.
6. Keep textual IR inspection in the core, but move graph serialization, colors, recordings, and
   serving into `alloy.viz`; `assembly.py` should be split rather than moved wholesale.
7. Rebuild `tests/` around the same subsystem boundaries, with no redundant `tests/alloy/` level.
8. Design the documentation taxonomy now, update paths during each source move, and perform the
   full documentation rewrite after the source layout stabilizes.

This is a reorganization, not an opportunity to redesign IR semantics, automatic differentiation,
the solver protocol, or generated C. Those should remain behaviorally unchanged until the moves
and dependency boundaries are complete.

## What exists today

The core package currently has 17 Python modules and roughly 7,940 lines directly under
`src/alloy/`, before counting the existing `codegen`, `solvers`, `utils`, and `viz` packages. File
size alone is not the problem; several large modules are cohesive. The problem is that root-level
location no longer predicts responsibility or dependency direction.

### The two IRs are conceptually parallel but packaged differently

The semantic IR is spread across:

- `ops.py`: `Ops`, `OpInfo`, operation groups, and NumPy reference functions;
- `expr.py`: `Expr`, hash-consing, builders, traversal, and one textual printer;
- `spec.py`: semantic verifier infrastructure and rules;
- `rewrite.py`: pattern matching, constant folding, simplification, and common-subexpression
  elimination.

Program IR is arranged differently:

- `program.py`: `POps`, `PNode`, hash-consing, builders, verifier infrastructure and rules, and a
  textual printer;
- `passes.py`: Program IR transformations;
- `assembly.py`: a second Program IR textual representation and graph serialization.

`Expr` and `PNode` do share an important design: each is a frozen, hash-consed node with an opcode,
node arguments, and attributes. Their vocabularies should therefore use parallel names and their
supporting facilities should be easy to find in parallel files.

They should **not** be forced behind one base node class. `Expr` is a typed mathematical value with
semantic shape, differentiability, constants, and lowering policy. `PNode` also represents
declarations and statements and carries scalar dtypes, address spaces, and control structure. A
shared inheritance hierarchy would hide those differences for little gain. Consistent names and
shared small mechanisms are enough.

### Root modules cross several architectural layers

Examples of mixed ownership include:

- `function.py` owns the function/port model, validates graphs, parses the derivative factory
  language, invokes dense and sparse AD, and lazily dispatches into JIT compilation.
- `assembly.py` owns stable semantic and Program IR text, visualization colors, and JSON graph
  serialization.
- `toolchain.py` owns environment parsing, C compiler discovery, JIT cache location, platform
  shared-library details, solver-plugin include/library discovery, link flags, loadability checks,
  diagnostics, and a command-line interface.
- `codegen/c.py`, described as an orchestrator, owns public C header generation, typed C++ wrappers,
  sparse metadata tables, generated-module packaging, solver/non-solver traversal, visualization
  recording hooks, and the universal entry point orchestration.
- `lowering.py` imports solver knowledge from `codegen/solver_c.py`, making a compiler stage depend
  on a renderer in order to identify opaque solver calls and their oracles.
- `codegen/c.py` imports `viz._recording`, so a production backend depends upward on an optional
  presentation subsystem.

The many function-local imports are useful evidence here. Some are legitimate lazy runtime
boundaries, but others are holding cycles together: `Function` to AD, AD back to helper Functions,
sparsity back to AD, lowering to solver codegen, toolchain to the solver registry, and solver stats
back to solver models.

### JIT and AOT are one path with asymmetric names

There is already an AOT path; it is just not named as one. The current flow is:

```text
Function
  -> lower_function
  -> optimize_program
  -> Program IR C renderer
  -> render_c_source / render_c_module       (current implicit AOT boundary)
  -> JIT compile + cache + ctypes dispatch   (optional final runtime steps)
```

`render_c_module` is the ahead-of-time product. `jit.py` consumes the same generated source, adds
compiler invocation and caching, dynamically loads the shared library, and marshals NumPy arrays.
Creating an unrelated second AOT implementation would be wrong. The layout should make it obvious
that JIT wraps the AOT module generator.

There is also avoidable repeated work: public header generation asks for workspace size by lowering
the function, while source generation lowers it again. A single internal render context should
produce the header, source, workspace metadata, and solver link flags once; both AOT and JIT should
consume that result.

### Tests mirror historical files rather than current responsibilities

All core tests live under the redundant `tests/alloy/` directory and are effectively flat. Several
files demonstrate why directory moves alone will not be enough:

- `test_core.py` mixes semantic expressions, types, rewrites, function calls, JIT errors, C headers,
  compiled C, sparse ABI metadata, and C++ wrappers in 718 lines.
- `test_map.py` mixes semantic construction, AD, sparsity, lowering, C source-size properties, ABI
  metadata, compilation, and workload-shaped integration tests in 778 lines.
- `test_program_migration.py` is now a permanent lowering/codegen regression suite despite its
  migration name.
- benchmark-harness tests live beside library tests even though they test `benchmarks/`, not
  `src/alloy`.

The target should group tests by the behavior they protect, not preserve a one-test-file-per-old-
source-file mapping.

### Documentation has the same accumulated-history problem

The current documentation is approximately 4,500 lines, with `docs/roadmap.md` at 759 lines,
`docs/spec.md` at 428 lines, and a separate 805-line root `ROADMAP.md`. `docs/spec.md` mixes IR
semantics, visualization, compiler limitations, benchmark fixtures, ABI details, and solvers.
`docs/roadmap.md` mixes a current roadmap with a detailed chronological implementation log.
`program_ir_migration.md` is valuable history but describes a completed migration as a north star.
Many documents also contain now-stale paths to `src/alloy/*.py` and `tests/alloy/*.py`.

This is worth accounting for now because package and concept names should be chosen once. The full
documentation move should wait until source paths stabilize so it does not double the churn.

## Target source architecture

The following tree is intentionally modest. It adds one new top-level subsystem (`compiler`) and
does not create a directory for every class.

```text
src/alloy/
  __init__.py                 # curated public API only

  core/
    __init__.py               # advanced IR API
    types.py                  # DType, TensorType, SparsityType, DeviceSpec
    spec.py                   # small shared rule-table machinery
    expr.py                   # ExprOp, Expr, semantic builders, traversal
    expr_spec.py              # semantic verification rules
    expr_rewrite.py           # PatternMatcher, simplify, CSE
    program.py                # ProgramOp, ProgramNode, builders, traversal
    program_spec.py           # Program IR verification rules
    text.py                   # canonical Expr/ProgramNode textual forms

  function/
    __init__.py               # Function, Port, decorator and factory-facing API
    model.py                  # Function/Port graph model and call composition
    factory.py                # jac:/grad:/hess:/lam: request parsing and construction
    api.py                    # @function and human convenience wrappers
    composition.py            # map_ and other Function-aware graph builders
    text.py                   # transitive Function-module textual form

  ad/
    __init__.py               # public dense and sparse AD surface
    forward.py                # JVP and multi-seed JVP, CALL/MAP derivative helpers
    reverse.py                # VJP and reverse local rules
    derivatives.py            # jacobian, gradient, hessian, linear combinations
    sparse.py                 # sparsity propagation, coloring, sparse Jacobian/Hessian
    finite_difference.py      # numerical validation helper, if retained as library API

  compiler/
    __init__.py               # lower_function and compiler observer types
    lowering.py               # Expr/Function -> ProgramNode
    passes.py                 # ProgramNode -> ProgramNode optimization pipeline

  codegen/
    __init__.py               # public generated-module and JIT entry points
    abi.py                    # universal ABI definitions
    c.py                      # ProgramNode -> C backend renderer
    solver.py                 # plugin wrapper framing and solver emission helpers
    aot.py                    # Function -> complete C module/header/source/link metadata
    jit.py                    # AOT module -> cached shared library -> ctypes/NumPy dispatch
    toolchain.py              # C compiler and platform compilation mechanics

  solvers/
    __init__.py
    model.py                  # SolverFunction, descriptors, external oracles, statuses
    graph.py                  # solver/callee/oracle graph inspection shared by compiler/codegen
    registry.py               # plugin discovery and protocol
    toolchain.py              # solver include/lib discovery, loadability, link flags
    stats.py                  # solver statistics model and C ABI fragment
    qp.py
    nlp.py

  viz/
    __init__.py
    graph.py                  # graph JSON and visual styling only
    recording.py             # opt-in compilation recordings
    serve.py

  utils/
    __init__.py
    env.py                    # generic environment parsing used outside codegen
    torch_state_dict.py
```

This tree is a target, not a mandate to split a module mechanically at every suggested seam. In
particular, `ad/derivatives.py` may remain in `forward.py` or `reverse.py` if moving it would create
more coupling than it removes. The ownership rules below matter more than exact file count.

### Naming decision

Use singular, dialect-qualified names:

| Current | Target |
| --- | --- |
| `Ops` | `ExprOp` |
| `Expr` | `Expr` |
| `POps` | `ProgramOp` |
| `PNode` | `ProgramNode` |
| `VerifyRule` / `PRule` | shared generic `Rule`, with dialect-specific verifier functions |
| `Spec` / `PSpec` | shared generic `Spec` |

`Expr` should stay `Expr`: it is central to the public modeling API and is more meaningful than
`ExprNode`. `ProgramNode` should be spelled out because `PNode` saves few characters and obscures
the dialect at every cross-layer call site. `ExprOp` and `ProgramOp` make imports unambiguous and
remove the plural-enum/single-node mismatch.

This project is pre-1.0 and these advanced IR names are documented but still evolving. I would do
the rename atomically and not retain permanent `Ops`/`POps`/`PNode` aliases. If there are known
external users of those advanced imports, keep aliases for exactly one pre-1.0 compatibility line,
mark their removal in the active roadmap, and delete them on schedule. Do not add aliases merely
for hypothetical consumers.

### Dependency rules

The package layout should be backed by a small set of enforceable rules:

1. `core` imports only Python, NumPy, and genuinely leaf-level `utils`; it never imports
   `Function`, AD, solvers, compiler, codegen, or visualization.
2. `function/model.py` depends on `core`. Function-aware builders such as `map_` live outside
   `core` so `core.expr` does not import `Function` just to perform an `isinstance` check.
3. AD depends on `core` and `function.model`. `function.factory` depends on AD. A lazy delegation
   from `Function.factory` to `function.factory` is the deliberate bridge that avoids a module
   cycle.
4. `compiler` depends on `core`, `function.model`, and solver graph metadata, but never on a C
   renderer. Solver inspection currently in `codegen/solver_c.py` moves to `solvers/graph.py`.
5. `codegen` depends on `compiler`, `core`, `function`, and `solvers`. The Program IR C renderer
   receives a `ProgramNode`; it does not initiate lowering itself.
6. JIT consumes the same complete generated module as AOT. It may add caching, compilation,
   loading, and dispatch, but no lowering or rendering decisions.
7. `viz` is a terminal observer of the other layers. Compiler/codegen must not import `viz`.
   `Function` can carry neutral compilation observers, or JIT/AOT can accept observers explicitly;
   visualization registers one of those observers.
8. `utils` is a leaf collection, not a dumping ground. Compiler discovery, solver discovery, ABI
   logic, IR text, and numerical AD checks retain their domain owners.

These rules can later be guarded by a small import-boundary test using the standard-library AST;
there is no need to add a dependency for that check.

## Explicit AOT/JIT design

The human-readable relationship should be:

```text
Function
  -> compiler.lowering
  -> compiler.passes
  -> codegen.c
  -> codegen.aot.render_module() -> CModule
                                  |-> caller writes/builds it (AOT)
                                  `-> codegen.jit builds, caches, loads, dispatches
```

`CModule` should be the one product shared by both paths. It should carry at least the header and
source names/text and the compiler/link flags required by reached solver plugins. An internal
render context may also carry the already-lowered Program IR, workspace size, symbol, and solver
order so the header and source are generated from one compilation rather than lowering twice.

Public entry points should be unsurprising:

- `alloy.codegen.aot.render_module(fun, ...) -> CModule`
- `alloy.codegen.aot.render_source(fun, ...) -> str` as convenience
- `alloy.codegen.jit.compile(fun) -> CompiledFunction`
- `Function.__call__` lazily delegates to `alloy.codegen.jit`

The top-level convenience exports `alloy.render_c_module` and similar can remain if they are part of
the intended public API, but their implementation should delegate to `codegen.aot`. Do not create
an `aot.py` file that merely re-exports a second renderer; the point is to name the existing common
boundary.

## What to do with `assembly.py`

Do not fold it wholesale into `viz`.

- Stable textual IR is compiler infrastructure: it is used for diagnostics, tests, diffs, and
  potentially future parsing. Move that part, together with the existing `format_expr` and
  `format_program`, into `core/text.py` and `function/text.py`.
- Graph JSON, node colors, and display labels exist solely for presentation. Move them into
  `viz/graph.py`.
- Recording and serving stay in `viz`.

There is no general IR parser today; the current code has printers only. Do not add a parser just to
complete a folder taxonomy. If round-tripping textual IR becomes a requirement, add it beside the
canonical text format as a separate feature with round-trip tests.

## Target test architecture

Tests should live directly under subsystem directories below `tests/`, with no `tests/alloy/`
level and no `__init__.py` files unless a real shared test package is needed.

```text
tests/
  core/
    test_expr.py
    test_types.py
    test_expr_rewrite.py
    test_expr_spec.py
    test_program.py
    test_program_spec.py
    test_text.py
  function/
    test_model.py
    test_api.py
    test_factory.py
    test_map.py
  ad/
    test_forward.py
    test_reverse.py
    test_sparse.py
    test_casadi.py
  compiler/
    test_lowering.py
    test_passes.py
  codegen/
    test_c.py
    test_aot.py
    test_jit.py
    test_toolchain.py
  solvers/
    test_model.py
    test_nesting.py
    test_registry.py
    test_codegen.py
    test_stats.py
  viz/
    test_graph.py
    test_recording.py
  utils/
    test_torch_state_dict.py
  integration/
    test_stage_transcription.py
  benchmarks/
    test_closed_loop.py
    test_recording.py
```

Important existing-file treatment:

- Split `test_core.py`; do not rename it wholesale. Semantic construction belongs in `core`,
  function calls in `function`, generated header/source tests in `codegen`.
- Split `test_map.py` by contract. MAP construction/composition stays in `function`; derivative
  rules move to `ad`; constant-source and lowering checks move to `compiler`/`codegen`; the small
  end-to-end workload stays in `integration` if it cannot be made local to one subsystem.
- Rename `test_program_migration.py` to lowering/codegen tests. The migration is complete and the
  permanent behavior should no longer be described as transitional.
- Keep plugin-specific numerical tests under each `plugins/*/tests/`; core solver tests should test
  the descriptor, registry, generated-wrapper protocol, and nesting without owning backend tests.
- Keep benchmark correctness gates in `benchmarks/problems/*/checks.py` as required by the existing
  project rule. `tests/benchmarks` is only for the reusable benchmark harness.
- Prefer the public `import alloy as al` surface in black-box tests. Import private subsystem names
  only in tests that intentionally verify those internals.

## Documentation restructuring

The documentation taxonomy should reflect reader intent rather than source-file history:

```text
docs/
  index.md
  guides/
    getting_started.md
    functions_and_derivatives.md
    solvers.md
    visualization.md
  architecture/
    overview.md
    semantic_ir.md
    program_ir.md
    compiler.md
    codegen_jit_aot.md
    solver_plugins.md
  reference/
    python_api.md
    c_abi.md
    configuration.md
  development/
    roadmap.md
    testing_and_fuzzing.md
    versioning.md
  benchmarks/
    methodology.md
    scalability.md
  workloads/
    safety_filter.md
  history/
    program_ir_migration.md
    native_toolchain_exploration.md
    macos_clang_call_miscompile.md
    naming.md
    vendored_solver_notes.md
```

The exact moves can change, but the following content changes matter:

1. Write a short architecture overview early, using the final subsystem names and the one
   semantic-IR -> Program-IR -> CModule -> AOT/JIT diagram.
2. Split `docs/spec.md` into semantic IR, Program IR, compiler, C ABI, and solver-facing reference.
   A specification should not contain benchmark history or test-fixture narratives.
3. Turn `docs/roadmap.md` into a current library roadmap. Move completed phase narratives to
   history rather than maintaining a permanent 700-line status log.
4. Reconcile the library roadmap with root `ROADMAP.md`, which currently owns benchmarks, plugins,
   and the paper plan. Prefer one obvious current project roadmap with linked subsystem plans over
   two documents named roadmap.
5. Archive `program_ir_migration.md` as a completed design record. Keep it because it explains
   consequential decisions, but stop presenting it as the active north star.
6. Split `docs/solvers.md` into a user guide and internal/plugin architecture.
7. Add a real documentation navigation configuration for the existing Zensical dependency and a
   link/build check. At present the dependency exists but there is no checked-in site navigation
   configuration.
8. Update source paths, import examples, and test paths during the source/test phase that changes
   them; do not leave all broken links for the final documentation pass.

The source architecture and documentation taxonomy should be agreed together now. The bulk content
rewrite should be a later phase after module paths stop moving.

## Implementation phases

Each phase ends with formatting, linting, type checking, and the relevant test subset; broader tests
run before the next phase. Temporary compatibility shims, if any are proven necessary, must be
listed in the active plan with an explicit removal phase.

### Phase 0 — Freeze behavior and public boundaries

- Record the intended stable public surface in `alloy/__init__.py`, `alloy.function`, `alloy.ad`,
  `alloy.codegen`, and `alloy.solvers`.
- Record advanced imports that are intentionally allowed to break before 1.0.
- Add a small import-boundary test for the target rules before moving central modules.
- Capture baseline test collection, full-suite result, generated C snapshots or hashes for a few
  representative functions, and JIT/AOT cache behavior.
- Make no semantic or generated-code changes.

Exit: the move can prove that behavior and generated artifacts did not change accidentally.

### Phase 1 — Normalize the IR core

- Create `alloy/core`.
- Move types and both IR vocabularies into it.
- Rename `Ops`/`POps`/`PNode` to `ExprOp`/`ProgramOp`/`ProgramNode` atomically.
- Extract Program IR verification and printing from `program.py`; move semantic verification and
  rewriting beside their dialect.
- Share only the small `Rule`/`Spec` table mechanism. Do not introduce a shared node base class.
- Move Function-aware MAP construction out of `core.expr`.
- Split textual IR from visualization graph data.
- Move and split the corresponding `tests/core` and `tests/viz` tests.
- Update documentation and benchmark imports that name the old IR modules.

Exit: `core` has no imports from higher layers; both dialects have parallel names and support files;
IR verification, rewriting, and text tests pass.

### Phase 2 — Separate Function frontend from AD

- Create `alloy/function` and `alloy/ad`.
- Move `Function`/`Port` to `function/model.py`.
- Move factory request parsing and construction out of the model.
- Move convenience decorators/wrappers to `function/api.py` and Function-aware MAP construction to
  `function/composition.py`.
- Split forward, reverse, dense derivative construction, and sparse AD according to the target
  tree, adjusting the split if it creates artificial cycles.
- Replace module-level `Function -> AD` imports with lazy factory delegation.
- Move and split `tests/function` and `tests/ad`, retaining self-contained workload reproductions.

Exit: importing the Function model does not import AD or codegen; the public top-level modeling API
and derivative numerics remain unchanged.

### Phase 3 — Establish the compiler boundary

- Create `alloy/compiler` and move lowering and Program IR passes there.
- Move solver/callee/oracle discovery needed by lowering from `codegen/solver_c.py` to
  `solvers/graph.py`.
- Make lowering return Program IR without importing a renderer.
- Keep pass order and generated Program IR identical.
- Rename permanent migration tests to `tests/compiler/test_lowering.py`; move pass tests beside them.

Exit: dependency direction is Function/core/solver metadata -> compiler -> Program IR, with no
compiler-to-codegen edge.

### Phase 4 — Make AOT and JIT explicit

- Move ABI definitions under `codegen`.
- Rename the Program IR C renderer to `codegen/c.py` and solver wrapper framing to
  `codegen/solver.py`.
- Turn the current `codegen/c.py` orchestration and `CModule` packaging into `codegen/aot.py`.
- Move `jit.py` under `codegen` and make it consume the same `CModule` result.
- Produce header, source, workspace metadata, and solver compile/link flags from one internal
  compilation/render context rather than lowering separately.
- Separate generic compiler mechanics from solver discovery: `codegen/toolchain.py` and
  `solvers/toolchain.py`.
- Move generic environment parsing needed by AD to `utils/env.py` so AD no longer imports a native
  toolchain module.
- Invert visualization recording so codegen emits neutral observer events and never imports `viz`.
- Update official plugins atomically for moved protocol type imports. Bump the solver protocol only
  if the runtime contract changes, not for a type-only module path move.
- Split codegen, JIT, AOT, toolchain, and solver tests into their target directories.

Exit: AOT and JIT visibly share one generated module; generated C and cache keys remain stable
unless an intentional cache-version bump is documented; compiler/codegen/viz dependencies point in
one direction.

### Phase 5 — Remove migration scaffolding and validate the repository

- Delete temporary import shims and old empty modules/directories.
- Move all remaining `tests/alloy/*` files, leaving no redundant directory.
- Update `pyproject.toml`, continuous integration commands, root `conftest.py`, benchmark imports,
  plugin imports, console-script paths, and all source/test path references.
- Run Ruff format/check, ty, the full parallel test suite, plugin tests, and benchmark correctness
  gates appropriate to changed IR/AD/codegen paths.
- Compare representative generated C, workspace sizes, numerical outputs, JIT cache reuse, solver
  nesting, and AOT consumer compilation with the Phase 0 baseline.

Exit: no old module path is used inside the repository, the full suite is green, and the source
tree alone communicates the compiler pipeline.

### Phase 6 — Documentation information architecture

- Add the documentation navigation/build configuration and architecture overview.
- Split the spec and solver documents by reader intent.
- Consolidate current roadmaps and archive completed plans/investigations.
- Update all links, imports, diagrams, and test/source references.
- Add a documentation build/link check to continuous integration.
- Remove this temporary restructuring plan once every phase is complete, unless it is deliberately
  retained as a historical design record.

Exit: a new contributor can find the public guide, the two IR specifications, the compiler/AOT/JIT
architecture, current roadmap, and historical records without reading chronological migration logs.

## Risks and controls

- **Large import-only diffs can hide semantic edits.** Keep file moves and behavior changes in
  separate phases; compare generated C and numerical results at every compiler-facing phase.
- **Hash-consing is identity-sensitive.** Do not duplicate node classes through compatibility
  modules or load the same implementation under two module names.
- **Official plugins are separate distributions.** Update their type imports, supported core range,
  tests, and protocol only as required in the same phase as codegen/solver moves.
- **JIT cache compatibility can mask or invalidate results.** Preserve the cache key inputs when
  output is unchanged; bump the cache version only when ABI/codegen/cache layout truly changes.
- **Moving tests can silently change collection.** Compare collected node IDs/counts before and
  after, including plugin tests and solver markers.
- **Documentation churn can obscure source review.** Update broken paths locally during source
  phases, but keep the large content/taxonomy rewrite in its own final phase.
- **A package can remain heterogeneous after a move.** Enforce dependency rules and ownership, not
  just directory names.

## Definition of done

- The semantic and Program IRs use `ExprOp`/`Expr` and `ProgramOp`/`ProgramNode`, with parallel
  locations for their vocabularies, verifiers, and text forms.
- `alloy.core` has no imports from Function, AD, compiler, codegen, solvers, or visualization.
- `alloy.compiler` owns all semantic-to-Program lowering and Program IR optimization.
- `alloy.codegen.aot` produces the single module consumed by both external AOT users and
  `alloy.codegen.jit`.
- Codegen does not import visualization, and compiler does not import codegen.
- Dense and sparse AD are under `alloy.ad`; the Function model no longer owns their implementation.
- Solver graph inspection, plugin discovery, toolchain discovery, and wrapper rendering have clear,
  separate owners.
- `assembly.py`, root `jit.py`, root `lowering.py`, root `passes.py`, root `abi.py`, root
  `toolchain.py`, and `tests/alloy/` no longer exist.
- The full source, plugin, and benchmark correctness checks pass with no unintended generated-C,
  ABI, numerical, workspace, or cache regressions.
- Documentation has one current roadmap, a short architecture overview, separate semantic and
  Program IR specifications, an explicit AOT/JIT explanation, and archived migration history.
