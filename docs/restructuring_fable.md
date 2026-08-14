# Restructuring alloy — distilled plan (Fable)

Design note, 2026-08-14. **Status: agreed plan — implementation not yet started.**

This document supersedes `restructuring_opus.md` and `restructuring_sol.md`, and folds in
`restructuring_factory.md` (typed derivative specs). It keeps Sol's destination (the conceptual
layout and naming) and Opus's process (the migration mechanics and gates), with the
disagreements between them resolved by Ted on 2026-08-14. The three source documents move to
`docs/history/` in phase 8.

## Diagnosis (shared by both plans, verified)

The Program IR migration (`docs/program_ir_migration.md`) added a second dialect and a real
compiler pipeline as parallel root modules and stopped before the consolidation pass: the new
stack never adopted the old one's vocabulary, the old modules were never re-cut around the new
architecture, and `codegen/c.py` was left holding whatever fit neither. Symptoms, all verified:

- Two dialects, two of everything, no shared vocabulary: `Ops`/`Expr` across four files vs
  `POps`/`PNode` in one 814-line `program.py`; the `P` prefix applied inconsistently
  (`PNode`, `PSpec`, but `ProgramVerifyError`, `format_program`, no `POpInfo`).
- Seven import cycles held together by ~50 function-local imports (Opus §1.3 has the full edge
  table): frontend→backend via JIT, `toolchain ↔ solvers.registry`, `ad ↔ sparsity`,
  `codegen.solver_c ↔ solvers.solver_function`, `lowering → codegen.solver_c`,
  `codegen.c → viz._recording`, `solvers.stats ↔ solvers.solver_function`.
- Heterogeneous modules: `codegen/c.py` (ABI defines + header + AOT module + graph walks),
  `toolchain.py` (compiler discovery + solver-library discovery + CLI), `sparsity.py`
  (structural analysis + AD-driven construction), `expr.py` (node core + user-facing math sugar),
  `assembly.py` (compiler text + viz JSON).
- AOT exists but is unnamed: `render_c_module` at the bottom of `codegen/c.py`;
  `benchmarks/harness/sweep.py:14` imports the private `_workspace_size`; header and source
  generation each lower the function separately.
- `tests/alloy/` is flat, with grab bags (`test_core.py`, `test_map.py`) and historical names
  (`test_program_migration.py`).
- Docs mix reference, history, results, and plans; two files named roadmap; no architecture
  document — which is why the layout drifted in the first place.

## Decisions (ratified)

1. **Vocabulary.** `Ops → ExprOp`, `POps → ProgramOp`, `PNode → ProgramNode`; `Expr` stays.
   Atomic rename, no permanent aliases. `ProgramNode` (not `Program`): most nodes are fragments
   inside a program — index arithmetic, loads, buffer declarations — and `coords: list[Program]`
   misreads; `Expr` works because every node genuinely is an expression.
2. **Dialect names.** "Semantic dialect" is retired: the two dialects are the **expression
   (expr) dialect** and the **program dialect**. Assembly prefixes become `expr.*` / `prog.*`
   (today `sem.*` / `prog.*`), `sem.module → expr.module`. Docs and printers follow.
3. **IR layout.** One flat `ir/` package with dialect-prefixed files — not `core/` (a name with
   no exclusion criterion accretes) and not mirrored subpackages (duplicate filenames force
   import aliasing). `ir/` owns definitions, verification, text, and the *infrastructure* to
   define and apply passes (`PatternMatcher` and company).
4. **`passes/` package.** All concrete IR→IR transformations live in one middle package named
   `passes/` (the standard compiler term, already the repo's vocabulary): expr-dialect rewrites
   (simplify, CSE), lowering (expr → program), and program-dialect passes. Not `compiler/` —
   the compiler is the whole system, `ir/ + passes/ + codegen/` together.
5. **Sequencing.** Opus's discipline: break cycles in place first, then one pure-move phase,
   then local re-cuts — every diff is either "seam change, layout unchanged" or "moves, zero
   logic change", gated by a byte-for-byte generated-C snapshot.
6. **Typed derivative specs replace the factory string grammar**
   (`restructuring_factory.md`). `"jac:eq:z"` → `al.jac("eq", "z")`; the `_factory_output`
   string parser is deleted outright, no compatibility shim (nothing external consumes the
   strings). Scope boundaries: input names stay strings (including the auto-created
   `"lam:<output>"` / `"fwd:<input>"` conventions), `aux` is unchanged, and derived output
   names stay `jac_eq_z`-shaped so generated C symbols do not move. The `api.py` wrappers
   become the canonical convenience layer and gain pass-through inputs so `solvers/nlp.py`
   stops hand-rolling `"lam:f"` factory strings. Defaults on the note's open decisions: keep
   the name `factory`, lowercase constructors (`al.jac`), spec types in `function/factory.py`.

Adjudicated from the two plans (rationale inline):

- **`assembly.py` splits** (Sol): stable IR text is compiler infrastructure → `ir/text.py`;
  graph JSON, colors, labels are presentation → `viz/graph.py`. Not a wholesale move to viz.
- **No solver-protocol bump for path-only moves** (Sol): all plugin imports of
  `SolverWrapperCtx` are `TYPE_CHECKING`-only (verified); the runtime contract does not change.
  Plugins are in-repo and update atomically in the same diff.
- **Single render context** (Sol): header, source, workspace size, and solver link flags come
  from one lowering, packaged as a `CModule` consumed by both AOT and JIT.
- **Tests get `__init__.py`** (Opus): without it, same-named files in different directories
  collide under `pytest -n=auto`.
- **Matcher unification is optional and last** (both, reconciled): lands only if it shrinks
  the code, gated by the C snapshot.
- **`function/` as the frontend package name** (over `frontend/`): concrete, and converting
  `function.py` to a package keeps `from alloy.function import Function` — which every plugin
  uses — working verbatim.
- **Docs**: taxonomy and names agreed now, stale paths fixed in the phase that breaks them,
  full content rewrite last, on its own branch (both plans agree).

## Target layout

```
src/alloy/
  __init__.py              curated public re-exports only

  ir/                      definitions, verification, text, pass infrastructure
    types.py               DType, DeviceSpec, TensorType, SparsityType, ScalarType
    expr.py                ExprOp, OpInfo, OP_INFO, Expr, builders, interning, topo
    expr_spec.py           expr-dialect verify rules, verify_expr
    program.py             ProgramOp, RangeKind, ProgramNode, builders, interning
    program_spec.py        program-dialect verify rules, verify_program
    spec.py                shared Rule/Spec table machinery, one VerifyError
    match.py               Pattern, PatternMatcher, walk/rebuild — how passes are defined/applied
    text.py                format_expr, format_program, render_expr_assembly, render_program_assembly

  passes/                  every concrete IR→IR transformation
    expr.py                simplify, CSE, constant folding            (Expr → Expr)
    lowering.py            lower_function                             (Function/Expr → ProgramNode)
    program.py             PASS_PIPELINE, optimize_program, the passes (ProgramNode → ProgramNode)

  function/                frontend; __init__ re-exports Function, Port
    model.py               Function, Port, call composition, graph validation
    factory.py             typed derivative specs (jac, grad, hess, spjac, …) and their
                           graph construction → AD dispatch — no string parsing
    api.py                 @function decorator, convenience wrappers
    sugar.py               dot, norm_2, sumsqr, vec, split, stack, gather, map_, …

  ad/                      target split; merge legs if it creates artificial coupling
    forward.py             jvp, jvp_many
    reverse.py             vjp, vjp_many, local VJP rules
    derivatives.py         jacobian, gradient, hessian, basis, linear_combination
    sparsity.py            structural masks + coloring (no AD import)
    sparse.py              sparse_jacobian, sparse_hessian construction

  codegen/
    abi.py                 C ABI signature, BufferType, status defines
    c.py                   ProgramNode → C renderer (today's program_c.py)
    solver.py              SolverWrapperCtx + wrapper framing — rendering only
    aot.py                 render context → CModule (header, source, workspace, link flags),
                           file-writing driver, `python -m alloy.codegen.aot`,
                           public workspace-size accessor
    jit.py                 CModule → compile, cache, dlopen, ctypes dispatch
    toolchain.py           C compiler discovery, cache dir, shared-lib extension, diagnostics CLI

  solvers/
    solver_function.py     SolverFunction, SolverDescriptor (name kept: plugin import stability)
    graph.py               is_solver_function, solver_callees, backends_used, external_oracles
    registry.py            plugin discovery, SOLVER_PLUGIN_PROTOCOL_VERSION
    paths.py               vendored solver lib/header discovery (split out of toolchain.py)
    stats.py               solver statistics + SolverStatus (moves in from solver_function.py)
    qp.py  nlp.py  _oracle.py

  viz/
    graph.py               graph JSON, colors, labels (from assembly.py)
    recording.py           registers into a codegen-owned observer hook
    serve.py

  utils/
    env.py                 generic env-var parsing (leaf; what ad needs from today's toolchain)
    torch_state_dict.py
```

### Layering

Each module may import strictly lower layers. `passes/` deliberately spans layers (its expr
rewrites sit below the frontend, its lowering above it); enforcement is per-module, not
per-package.

```
0  utils/*
1  ir/*
2  passes/expr, ad/sparsity, solvers/stats
3  function/model
4  ad/{forward,reverse,derivatives,sparse}
5  function/{factory,api,sugar}, solvers/{solver_function,graph,qp,nlp,registry,paths}
6  passes/{lowering,program}
7  codegen/*
8  viz/*
```

Exactly two sanctioned exceptions, both written down in `docs/architecture.md`:

- `function/model.py` reaches layer 7 through **one** named seam (calling a `Function`
  JIT-compiles it) — replacing today's three scattered local imports;
- `viz` registers into an observer hook that `codegen` owns — no `codegen → viz` import.

`tests/test_layering.py` (stdlib `ast`, ~40 lines) encodes this table with an explicit
allow-list for the two exceptions, and — during the migration — a tolerated-violations list
that each phase shrinks. This test is permanent; it is what stops the drift from recurring.

## Gates

Run at the end of every phase: `uv run pytest -n=auto`, `uv run ruff check`,
`uv run ruff format`, `uv run ty check`, and the benchmark correctness checks.

**Refactor gate (phases 1–6): generated C must not change.** Phase 0 snapshots
`render_c_source` output for a fixed corpus — a forward, a `jac`, a `spjac`, a MAP workload,
a solver-bearing function. Every phase re-renders and diffs byte-for-byte. This is a pure
refactor; if the C moves, something semantic moved with it. It also keeps the JIT cache key
(a hash of source text) stable. The `sem.* → expr.*` assembly rename touches viz text only,
never C.

## Phases

```
0 → 1 → 2 → 3 → { 4 , 5 } → 6 → 7 → 8
                              ↑
                              9 (optional, after 4)
```

One agent per phase. 4 and 5 run in parallel with disjoint ownership (4: `ir/`, `passes/`,
`function/`, `ad/`; 5: `codegen/`). 2, 3, and 6 are large but mechanical; 1, 4, and 5 need care.

### Phase 0 — instrument (no behavior change)

- Add `tests/test_layering.py` with the target edge table plus the recorded
  tolerated-violations list.
- Snapshot the generated-C corpus and add the diff helper (scaffolding; deleted in phase 7).
- Baseline: collected pytest node IDs, full suite green.

### Phase 1 — break the cycles in place (design-bearing; no file moves)

1. `SolverStatus` → `solvers/stats.py`.
2. Solver graph queries (`is_solver_function`, `solver_callees`, `solver_backends_used`,
   `external_oracles`, `solver_compile_flags`) out of `codegen/solver_c.py` into a solvers
   module (becomes `solvers/graph.py` after phase 2); repoint `lowering.py`.
3. Split `toolchain.py`: compiler/env mechanics vs solver-library discovery (the half that
   imports `solvers.registry`).
4. Split `sparsity.py`: structural analysis (no AD import) vs AD-driven construction.
5. Invert `codegen → viz`: codegen declares an observer hook it owns (the `ProgramObserver`
   pattern); viz registers into it.
6. Collapse `Function`'s three local `from .jit import …` into one named seam.

Exit: import graph is a DAG modulo the one documented seam; the violations list is empty for
cycles; no files have moved.

### Phase 2 — pure moves (mechanical; zero logic change)

`git mv` into the target tree, rewrite imports, update `alloy/__init__.py` re-exports, and
repoint in-repo consumers: `benchmarks/` (`sweep.py`'s private `_workspace_size` import is
repointed here and made public in phase 5), `plugins/*` (type-only `SolverWrapperCtx` paths;
`from alloy.function import Function` keeps working via the package `__init__`), and `tests/`
(imports only; the test tree itself moves in phase 6). No function bodies change. No protocol
bump; note the new module paths in `docs/solver_plugins.md`.

### Phase 3 — vocabulary rename (atomic; no aliases)

- `Ops → ExprOp`, `POps → ProgramOp`, `PNode → ProgramNode`.
- `PSpec`/`PRule` and `Spec`/`VerifyRule` collapse onto the shared machinery in `ir/spec.py`
  (one `Rule`, one `Spec`, one `VerifyError`), with dialect-specific rule tables and
  `verify_expr` / `verify_program` entry points.
- Retire "semantic": assembly prefix `sem.* → expr.*`, `sem.module → expr.module`; update the
  tests and doc passages pinned to that text in the same diff.

### Phase 4 — IR and passes re-cuts

- Fold each dialect's vocabulary and node into one module: `ops.py + expr.py → ir/expr.py`
  (minus the sugar), `program.py` split into `ir/program.py` + `ir/program_spec.py` + its
  printer into `ir/text.py`.
- Extract the user-facing math surface from `expr.py` into `function/sugar.py` (with `map_`,
  so `ir/expr.py` no longer imports `Function`).
- Split `assembly.py`: stable text → `ir/text.py`, graph JSON/colors → `viz/graph.py`.
- Concrete rewrites → `passes/expr.py`; `Pattern`/`PatternMatcher` stay in `ir/match.py`;
  `passes.py` → `passes/program.py`; `lowering.py` → `passes/lowering.py`.
- Split `ad.py` per the target tree, merging legs where the split would create coupling.
- **Typed derivative specs** (own commit, full suite as gate): when carving
  `function/factory.py` out of the model, implement the spec types directly and delete the
  `_factory_output` string parser in the same diff — it is never moved just to be deleted.
  `factory` accepts `Sequence[str | DerivSpec]` (bare strings remain output passthroughs);
  each spec type carries its own build method, replacing the `split(":")` if-chain with
  dispatch on type. Update the `api.py` wrappers (now canonical, with pass-through inputs),
  the 4 `solvers/nlp.py` call sites, the ~6 benchmark call sites
  (`problems/chain/`, `problems/race_cars/checks.py`, `harness/sweep.py`), and the ~9 test
  files with raw `.factory(` strings. Derived graphs, output names, sparsity handling, and
  generated C are unchanged — the C snapshot must stay byte-identical through this step.

### Phase 5 — AOT/JIT symmetry

- `codegen/aot.py`: one internal render context produces header, source, workspace size, and
  solver link flags from a single lowering, packaged as `CModule`; a file-writing driver;
  `python -m alloy.codegen.aot`; a public workspace-size accessor replacing `_workspace_size`
  (fix `benchmarks/harness/sweep.py`).
- `codegen/jit.py` consumes the same `CModule` — caching, compilation, loading, dispatch only;
  no rendering decisions. Factor the compile-flag logic shared by both into one function.
- `codegen/c.py` is left as rendering only. Cache-key inputs preserved; any bump is documented.

### Phase 6 — tests mirror the source

```
tests/
  conftest.py  test_layering.py
  ir/  passes/  function/  ad/  codegen/  solvers/  viz/  utils/  integration/  benchmarks/
```

- Drop `tests/alloy/`; each subdirectory gets an `__init__.py`.
- Split `test_core.py` and `test_map.py` by the contract they protect (expr construction → `ir/`,
  derivative rules → `ad/`, lowering/C checks → `passes/`+`codegen/`, workload-shaped
  end-to-end → `integration/`).
- `test_program_migration.py` → `tests/passes/test_lowering.py` (content kept — it is the only
  non-benchmark coverage of several lowering paths).
- `test_factory_casadi.py` → `tests/function/test_factory.py` (the CasADi cross-checks keep
  their value: same math, new request encoding). The malformed-string cases in `test_api.py`
  disappeared in phase 4 (unrepresentable); the unknown-name and aux-shadowing error tests stay.
- `test_benchmark_recording.py` → `tests/viz/`; `test_benchmark_closed_loop.py` →
  `tests/benchmarks/` (harness only; problem gates stay in `benchmarks/problems/*/checks.py`).
- Compare collected node IDs against the phase-0 baseline.

### Phase 7 — write the architecture down

`docs/architecture.md`: the pipeline (expr dialect → `passes` → program dialect → `CModule` →
AOT/JIT), the layer table, package ownership, the two sanctioned exceptions, and the rule
`test_layering.py` enforces. One-line ownership docstring per module. Delete the phase-0 C
corpus scaffolding. This lands with the restructure, not after it.

### Phase 8 — documentation reorganization (separate branch)

```
docs/
  architecture.md          (from phase 7)
  guides/                  getting_started.md, solvers.md (user-facing half), benchmarks.md, fuzzing.md
  reference/               expr_ir.md + program_ir.md (split of spec.md), c_abi.md,
                           solvers.md (internal/plugin half), solver_plugins.md, versioning.md,
                           conventions.md (naming rules, linked from AGENTS.md)
  results/                 scalability.md
  workloads/               safety_filter.md
  history/                 program_ir_migration.md, naming.md, native_toolchain_exploration.md,
                           macos_clang_call_miscompile.md, vendored_solvers.md,
                           restructuring_{opus,sol,factory,fable}.md  (frozen, dated)
  roadmap.md
```

- One roadmap: `docs/roadmap.md` becomes the current library roadmap with completed phase
  narratives moved to history; the root `ROADMAP.md` (benchmarks/paper) is renamed so the two
  no longer collide (e.g. `BENCHMARKS.md`), with cross-links.
- Fix `README.md`'s stale status paragraph; retitle `program_ir_migration.md` as completed
  history and repoint `lowering.py`'s docstring at `docs/architecture.md`.
- Retire the "CasADi-style derivative factories (`jac:*` / `grad:*` / …)" identity phrasing in
  `AGENTS.md`, `README.md`, `docs/spec.md`'s "Canonical factory API" section (and its
  CasADi-comparison table row), and the `docs/roadmap.md` echoes — the request encoding is now
  typed specs, and the break with CasADi is deliberate.
- Add the Zensical navigation config (the dependency exists in `pyproject.toml`; no site config
  does) and a docs build/link check in CI.

### Phase 9 — one matcher (optional)

Unify `PatternMatcher`, the spec rule tables, and `passes/program.py`'s hand-rolled
walk/rebuild onto `ir/match.py`. The only phase that can change behavior by accident (pass
ordering, match semantics), so it runs after the moves, gated by the C snapshot — and it lands
only if it shrinks the code.

## Risks

- **Big mechanical diffs can hide semantic edits** — hence the strict phase separation and the
  byte-for-byte C gate.
- **Hash-consing is identity-sensitive** — never load the same node class under two module
  paths (no compatibility shim modules for `ir/` types).
- **JIT cache** — key inputs preserved through phases 1–6; a bump is an explicit, documented
  event.
- **Test collection can change silently under moves** — compare collected node IDs, including
  plugin tests and solver markers.
- **A package can stay heterogeneous after a move** — the layering test and the ownership
  docstrings are the enforcement, not the directory names.
- **Two deliberate API breaks, no others.** The vocabulary rename (phase 3) and the factory
  spec encoding (phase 4) are the only user-visible breaks in the plan, both pre-1.0 and both
  without compatibility shims. Anything else that surfaces as a break is a bug in the refactor.
