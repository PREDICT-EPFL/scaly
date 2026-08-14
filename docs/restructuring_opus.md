# Restructuring alloy's source layout

Design note, 2026-08-14. **Status: proposal — authorizes no code changes yet.**

This note diagnoses the current organisation of `src/alloy`, `tests/`, and `docs/`, and proposes an
ordered restructuring plan. It is deliberately a plan and not a patch: phase 1 contains real design
decisions (where the seams go), and those should be agreed before any file moves.

---

## 1. Diagnosis

### 1.1 The root cause is an unfinished migration, not a bad design

The Program IR was added as a **parallel stack** (`program.py`, `lowering.py`, `passes.py`,
`codegen/program_c.py`) alongside the existing stack (`ops.py`, `expr.py`, `spec.py`, `rewrite.py`,
`codegen/c.py`). The migration (`docs/program_ir_migration.md`) ended at *"legacy renderer deleted,
suite green"*; its own Step 7 is "docs + merge". What was never done is the **consolidation pass**:
the new stack never adopted the old one's vocabulary, the old modules were never re-cut around the
new architecture, and `codegen/c.py` was left holding whatever did not belong to either
(ABI defines, the public header, the AOT module, function ordering, workspace sizing).

So the disorder is real, and the fix is best framed as *finishing the migration*, not as a
speculative tidy-up.

### 1.2 Two IRs, two of everything, no shared vocabulary

| concept | semantic IR | program IR |
| --- | --- | --- |
| op enum | `Ops` (`ops.py`) | `POps` (`program.py`) |
| node | `Expr` (`expr.py`) | `PNode` (`program.py`) |
| op metadata | `OpInfo` / `OP_INFO` | — (ad-hoc `SCALAR_OPS` frozensets) |
| verifier rule / spec | `VerifyRule` / `Spec` (`spec.py`) | `PRule` / `PSpec` (`program.py`) |
| verifier entry | `verify_expr` | `verify_program` |
| verifier error | `VerifyError` | `ProgramVerifyError` |
| printer | `format_expr` (`expr.py`) | `format_program` (`program.py`) |
| assembly printer | `render_expr_assembly` (`assembly.py`) | `render_program_assembly` (`assembly.py`) |
| self-rewrite leg | `rewrite.py` (`Pattern`, `PatternMatcher`, `rewrite`) | `passes.py` (`register_pass`, hand-rolled `_walk`/`_transform`) |

Three concrete consequences:

- **Split is inconsistent.** The semantic IR is spread over `ops.py` + `expr.py` + `spec.py` +
  `rewrite.py`; the program IR is one 814-line `program.py` (vocabulary + node + verifier + printer)
  plus a detached `passes.py`. Neither is wrong on its own; having both is what costs a reader.
- **Two printers per IR.** `format_*` (debug) and `render_*_assembly` (MLIR-ish, for viz) exist for
  both IRs, in different modules, with no stated relationship.
- **The rewrite substrate is duplicated.** `rewrite.PatternMatcher` matches `Expr`; `passes.py`
  re-implements traversal and rebuild by hand for `PNode`. `program.PSpec`/`PRule` is a third,
  verifier-only matcher. One matcher over "node with `op`, `args`, `attrs`" would serve all three —
  both node types are already the same shape (frozen, hash-consed, weakref-interned).

Naming: `Expr`/`PNode` and `Ops`/`POps` are fine as a pair *if applied systematically*. Today the
`P` prefix is applied to some things (`PNode`, `POps`, `PRule`, `PSpec`) and not others
(`ProgramVerifyError`, `format_program`, no `POpInfo`).

### 1.3 The module layer graph has cycles, papered over by ~50 function-local imports

This is the most objective symptom, and the best gate for the work. Current runtime import edges
(top-level imports only):

```
alloy.expr        -> function, ops, types          # expr imports function (for map_)
alloy.function    -> ad, expr, jit, ops, rewrite, solvers.stats, sparsity, types
alloy.jit         -> abi, codegen.c, codegen.solver_c, function, solvers.stats, toolchain
alloy.codegen.c   -> abi, codegen.program_c, codegen.solver_c, expr, function, lowering, ops,
                     passes, program, solvers.stats, viz._recording
alloy.lowering    -> codegen.solver_c, expr, function, ops, passes, program,
                     solvers.solver_function, types
alloy.toolchain   -> solvers.registry
alloy.ad          -> expr, function, ops, rewrite, sparsity, toolchain, types
alloy.sparsity    -> ad, expr, ops, rewrite, types
```

The cycles, and what each really means:

1. **`function → jit → codegen.c → lowering → function`.** The frontend depends on the entire
   backend because `Function.__call__` JIT-compiles. `Function.eval_list`, `recompile`, and
   `solver_stats` each open their own local `from .jit import …` to survive.
2. **`toolchain → solvers.registry → codegen.solver_c → toolchain`.** `toolchain.py` mixes two
   unrelated jobs: environment/cache/compiler discovery (a leaf concern) and *solver library*
   discovery (a solver concern). Splitting them removes the cycle outright.
3. **`solvers.stats ↔ solvers.solver_function`.** `SolverStatus` is a stats concept that lives in
   `solver_function.py`.
4. **`ad ↔ sparsity`.** `sparsity.py` holds two different things: structural mask propagation and
   colouring (pure analysis, no AD), and sparse Jacobian/Hessian *construction* (needs `jvp_many`,
   `gradient`). Only the second half creates the cycle.
5. **`codegen.solver_c ↔ solvers.solver_function`.** `is_solver_function`, `solver_callees`,
   `solver_backends_used`, `external_oracles`, `solver_compile_flags` are *model queries about
   solvers* that happen to live in a codegen module. Only `render_solver_raw` and the wrapper
   orchestration are codegen.
6. **`lowering → codegen.solver_c`.** A downward import: the middle of the pipeline reaches into the
   backend for the queries from (5). Disappears once (5) is fixed.
7. **`codegen.c → viz._recording`.** Codegen depends on visualisation, for one `begin_recording`
   call. Should be inverted through a hook — `passes.ProgramObserver` already establishes the
   pattern.

Cycles 2–7 are accidental and dissolve with the splits named above. Cycle 1 is essential in one
direction (calling a `Function` must reach the backend) and should become **one documented seam**
rather than four scattered local imports.

Deferred imports are not inherently bad — `ad.py`'s local `simplify_cse_fixpoint` imports are
mostly cycle-driven too, and some are genuine lazy-loading (`viz.serve`). But today there is no way
to tell the two apart, and no check that stops a new cycle from appearing.

### 1.4 Module responsibilities are heterogeneous

- `codegen/c.py` (379) — ABI status defines, typed C++ wrapper, public header, source orchestration,
  the AOT `render_c_module`, `_function_order`/`_callees` graph walks, and a `_workspace_size` shim.
  Four responsibilities under one name.
- `toolchain.py` (386) — env vars, cache dir, shared-library extension, compiler discovery, solver
  library/header discovery, compile flags, a diagnostic report, and a `main()` CLI.
- `sparsity.py` (416) — structural analysis *and* AD-driven construction (see cycle 4).
- `expr.py` (650) — the `Expr` node and interning (core) *and* the whole user-facing math surface
  (`dot`, `norm_2`, `sumsqr`, `vec`, `split`, `stack`, `gather`, `map_`, …), which is frontend.
- `assembly.py` (357) — printers for both IRs plus graph JSON, used only by `viz` and re-exported
  from `alloy/__init__`. It is viz infrastructure sitting at top level.
- `abi.py` (34) — a signature string and `BufferType`. Too small to be a top-level module; it is
  codegen's contract.

### 1.5 JIT and AOT are not symmetric

`jit.py` is a real module: cache key, compile, `dlopen`, ctypes dispatch. AOT is a single function,
`render_c_module`, buried at the bottom of `codegen/c.py` — no driver, no CLI, no file-writing
helper. Benchmarks reach for it directly, and `benchmarks/harness/sweep.py` imports the *private*
`alloy.codegen.c._workspace_size`, which is the tell that the AOT surface is under-specified. The
two paths share source rendering, ABI, workspace sizing, and solver compile flags; none of that
sharing is visible from the module names.

### 1.6 Tests

`tests/alloy/` is flat (21 files) and its names do not track the source:

- redundant prefixes: `test_alloy_jit.py`, `test_alloy_sparsity.py` (inside `tests/alloy/`);
- grab bags: `test_core.py` (718 lines), `test_map.py` (778) span IR, AD, and codegen;
- historical framing: `test_program_migration.py` (395) is the migration self-certification harness
  — valuable content, obsolete name;
- misfiled by subject: `test_benchmark_recording.py` (518) and `test_benchmark_closed_loop.py` test
  `alloy.viz` recording and the benchmark harness, not "benchmarks".

The `tests/alloy/` nesting buys nothing: `conftest.py` already puts the repo root on `sys.path`, and
`pyproject.toml` sets `testpaths = ["tests", "plugins"]`.

### 1.7 Docs have the same disease

13 files in `docs/` with no separation between **reference** (`spec.md`, `solvers.md`,
`solver_plugins.md`, `versioning.md`), **history/incident notes** (`program_ir_migration.md`,
`naming.md`, `native_toolchain_exploration.md`, `macos_clang_call_miscompile.md`), **results**
(`scalability.md`), and **plans** (`roadmap.md`). Plus:

- two roadmaps named alike — `ROADMAP.md` (benchmarks/paper) at the root and `docs/roadmap.md`
  (library phases);
- `README.md`'s status paragraph ("Phases 0-4 complete, Phase 5 in progress") is stale;
- `docs/program_ir_migration.md` still reads as an active north star and is cited as such by
  `lowering.py`'s module docstring, though the migration is finished;
- **there is no architecture document** — nothing states which module owns what, or which way
  imports are allowed to point. That absence is precisely why the layout drifted.

---

## 2. Target layout

Two naming principles, both aimed at the failure mode above:

- **No module named `core` or `utils` at a layer boundary.** Neither has an exclusion criterion, so
  both accrete. `utils/` survives only as the existing single-purpose `utils/torch_state_dict.py`.
- **Mirror-symmetric file names across the two IRs**, so learning one teaches the other.

```
src/alloy/
  __init__.py             public re-exports only
  types.py                DType, DeviceSpec, TensorType, SparsityType, ScalarType     [layer 0]
  env.py                  env vars, cache dir, compiler discovery, shared-lib ext     [layer 0]

  ir/
    matcher.py            one PatternMatcher over (op, args, attrs); serves both IRs
    expr/
      ops.py              Ops, OpInfo, OP_INFO, op-set groupings
      node.py             Expr, interning, topo
      spec.py             Spec, VerifyRule, verify_expr
      fmt.py              format_expr
      rewrite.py          simplify, cse, algebraic rules
    program/
      ops.py              POps, RangeKind, SCALAR_OPS, (POpInfo if it earns its place)
      node.py             PNode + constructors
      spec.py             PSpec, PRule, verify_program
      fmt.py              format_program
      passes.py           PASS_PIPELINE, optimize_program, the passes

  frontend/
    function.py           Function, Port, factory
    api.py                @function, jacobian/gradient/hessian/spjacobian/…
    sugar.py              dot, norm_2, sumsqr, vec, split, stack, concat, gather, map_, …

  ad/
    forward.py            jvp, jvp_many
    reverse.py            vjp, vjp_many, local VJP rules
    derivatives.py        jacobian, gradient, hessian, basis, linear_combination
    sparsity.py           masks, colouring — structural only, no AD import
    sparse.py             sparse_jacobian, sparse_hessian, colored/structured construction

  codegen/
    abi.py                C ABI signature, BufferType, status defines
    lowering.py           Expr -> PNode  (leg 1)
    c.py                  PNode -> C source + public header  (leg 3; leg 2 is ir/program/passes.py)
    solver_c.py           solver wrapper orchestration only
    jit.py                source -> .so -> callable, cache
    aot.py                source -> files on disk, workspace size, compile flags, CLI

  solvers/
    qp.py  nlp.py  solver_function.py  registry.py  stats.py
    model.py              is_solver_function, solver_callees, backends_used, external_oracles
    paths.py              vendored solver library/header discovery (split out of toolchain.py)

  viz/
    assembly.py           MLIR-ish printers + graph JSON  (moved from top level)
    recording.py  serve.py

  utils/torch_state_dict.py
```

Allowed import direction (each layer may import strictly lower ones):

```
0  types, env
1  ir/            (matcher, expr/*, program/*)
2  ad/            solvers/{stats,paths,registry}
3  frontend/      solvers/{solver_function,model,qp,nlp}
4  codegen/       (lowering, passes-consumer, c, solver_c, jit, aot)
5  viz/
```

with exactly two sanctioned exceptions, both written down:

- `frontend/function.py` reaches layer 4 through **one** seam (calling a `Function` executes it);
- `viz` installs an observer that `codegen` consults through a protocol it owns — no `codegen → viz`
  import.

**Gate:** a `tests/test_layering.py` that parses the package with `ast` and asserts the edge table
above, with an explicit allow-list for the two exceptions. ~40 lines, and it is what stops this
from happening again.

### Choices worth arguing about

1. **Where `passes.py` lives.** Placed above under `ir/program/` for symmetry with
   `ir/expr/rewrite.py` (each IR owns its self-rewrite leg). The counter-argument is the migration
   doc's framing of lowering/passes/renderer as three legs of one pipeline, which would put it in
   `codegen/`. Symmetry wins in this proposal because it is the property being asked for; the
   pipeline story is recovered by the architecture doc.
2. **`ir/expr/` + `ir/program/` subpackages vs. flat `ir/{ops,expr,spec,fmt,rewrite}.py` +
   `ir/{pops,pnode,pspec,pfmt,passes}.py`.** Subpackages keep names identical across the two IRs,
   which is the whole point; the cost is `from alloy.ir.program import ops as pops` in
   `lowering.py`, the one module that speaks both.
3. **`env.py` vs keeping the name `toolchain.py`** for the layer-0 half. `toolchain` currently means
   "compilers *and* solver libraries"; the split needs two names and `env` is the honest one for the
   remainder, but `toolchain.py`/`toolchain/solver_paths.py` is a smaller diff.
4. **Renaming `ProgramVerifyError` → `PVerifyError`** and adding `POpInfo`. Cheap, and completes the
   `P`-prefix rule; only worth it if the prefix rule is adopted deliberately.

---

## 3. Plan

Nine phases. Phase 1 is design-bearing and should be reviewed alone; phase 2 is mechanical and
large; everything after is local.

**Global gates, run at the end of every phase:** `uv run pytest -n=auto`, `uv run ruff check`,
`uv run ruff format`, `uv run ty check`, and the benchmark correctness checks.

**Refactor-specific gate (phases 1–6): generated C must not change.** Before phase 1, snapshot
`render_c_source` output for a fixed corpus (a forward, a `jac`, a `spjac`, a MAP workload, a
solver-bearing function). Every phase re-renders and diffs byte-for-byte. This is a pure refactor;
if the C moves, something semantic moved with it. It also protects the JIT cache key, which hashes
source text.

### Phase 0 — instrument (small, no behaviour change)

- Add `tests/test_layering.py` encoding the **target** edge table, plus a recorded list of current
  violations it tolerates. Each later phase deletes entries from that list.
- Add the C-snapshot corpus and diff helper used by the gate above (throwaway; delete at phase 9).

### Phase 1 — break the cycles in place (design-bearing; no file moves)

Do the thinking with the layout unchanged, so the diff is readable:

1. Move `SolverStatus` into `solvers/stats.py`; drop the two deferred imports. *(cycle 3)*
2. Split solver **model queries** out of `codegen/solver_c.py` into `solvers/model.py`; repoint
   `lowering.py` at it. *(cycles 5, 6)*
3. Split `toolchain.py` into the layer-0 half (env, cache dir, compiler, lib extension) and the
   solver-discovery half; the latter imports `solvers.registry`. *(cycle 2)*
4. Split `sparsity.py` into structural analysis (no `ad` import) and AD-driven construction.
   *(cycle 4)*
5. Invert `codegen → viz`: `codegen` declares an observer hook it owns; `viz` registers into it.
   *(cycle 7)*
6. Collapse `Function`'s three local `from .jit import …` into one named seam
   (`frontend/function.py` calls a single `execute`/`compiled` entry point) and document it. *(cycle 1)*

Exit: import graph is a DAG modulo the one documented seam; `tests/test_layering.py`'s violation
list is empty for cycles; **no files have moved yet.**

### Phase 2 — pure moves (mechanical, large diff, zero logic change)

`git mv` into the tree of §2, rewrite imports, update `alloy/__init__.py` re-exports, and update the
in-repo consumers: `benchmarks/` (notably `harness/sweep.py`'s private `_workspace_size` import),
`plugins/*` (`SolverWrapperCtx`, `solver_function`, `function`, `types`), and `tests/`. **No
function bodies change in this phase** — that is what makes it reviewable.

Note: the solver plugin protocol (v4, `docs/solver_plugins.md`) is a real external contract; moving
`SolverWrapperCtx` means bumping `SOLVER_PLUGIN_PROTOCOL_VERSION` and saying so in that doc.

### Phase 3 — IR symmetry

- Split `program.py` into `ops.py` / `node.py` / `spec.py` / `fmt.py`; split `ops.py`+`expr.py`+
  `spec.py` to match.
- Move the user-facing math sugar out of `expr.py` into `frontend/sugar.py`.
- Apply the `P`-prefix rule consistently if adopted (choice 4 above).

### Phase 4 — one matcher

Unify `rewrite.PatternMatcher`, `program.PSpec`/`PRule`, and `passes.py`'s hand-rolled
`_walk`/`_transform` onto `ir/matcher.py`. This is the only phase that can change behaviour by
accident (pass ordering, match semantics), so it is deliberately after the moves and gated by the
C-snapshot diff. It is also the only optional phase: if the unification does not shrink the code, do
not land it.

### Phase 5 — JIT / AOT symmetry

- Extract `codegen/aot.py`: `CModule`, `render_c_module`, a file-writing driver, `python -m
  alloy.aot`, and a **public** workspace-size accessor to replace `_workspace_size`.
- Factor the compile-flag/solver-flag logic shared by `jit` and `aot` into one function.
- Leave `codegen/c.py` as rendering only.

### Phase 6 — tests mirror the source

```
tests/
  conftest.py
  ir/  frontend/  ad/  codegen/  solvers/  viz/  test_layering.py
```

- Drop `tests/alloy/`; strip the redundant `alloy` prefix from file names.
- Split `test_core.py` and `test_map.py` by subject; rehome `test_benchmark_recording.py` to
  `viz/`, `test_benchmark_closed_loop.py` to the harness tests; rename
  `test_program_migration.py` to `codegen/test_lowering.py` keeping its content (per `AGENTS.md`,
  it is the only non-benchmark coverage of several lowering paths).
- Add `__init__.py` to each test subpackage. Without it, same-named files in different directories
  collide under `pytest -n=auto`.

### Phase 7 — write the architecture down

`docs/architecture.md`: the layer table, what each package owns, the two sanctioned exceptions, the
three pipeline legs, and the rule that `tests/test_layering.py` enforces. Refresh every module
docstring to a single line naming what the module owns. Delete the phase-0 scaffolding.

**This should land with the restructure, not after it** — the layering rules have to be written at
the moment they are established, or the next agent re-derives them wrong.

### Phase 8 — documentation reorganisation (separate branch)

Same disease, same treatment, but it touches every file and would drown the code review:

```
docs/
  architecture.md              (from phase 7)
  reference/   spec.md, abi.md, solvers.md, solver_plugins.md, versioning.md, conventions.md
  guides/      getting-started.md, benchmarks.md, fuzzing.md
  results/     scalability.md
  notes/       program_ir_migration.md, naming.md, native_toolchain_exploration.md,
               macos_clang_call_miscompile.md, vendored_solvers.md      (frozen, dated)
  roadmap.md
```

Plus: rename the root `ROADMAP.md` so it does not collide with `docs/roadmap.md` (e.g.
`docs/roadmap-paper.md` or root `BENCHMARKS.md`); fix `README.md`'s stale status paragraph; retitle
`docs/program_ir_migration.md` as completed history and repoint `lowering.py`'s docstring at
`docs/architecture.md`; fold the naming conventions from `AGENTS.md` into a reference page that
`AGENTS.md` links to.

### Dependencies and parallelism

```
0 → 1 → 2 → { 3 , 5 } → 6 → 7 → 8
             ↑
             4 (optional, after 3)
```

- 1 must precede 2 (do not move files and change seams in the same diff).
- 3 and 5 can run in parallel after 2 with disjoint ownership: 3 owns `ir/` + `frontend/`, 5 owns
  `codegen/` + `jit`/`aot`.
- 6 comes after both, because test file placement follows the final source tree.
- 8 can start any time after 7 and is otherwise independent.

One agent per phase. Phases 2 and 6 are the large mechanical ones; 1 and 4 are the ones that need
care.
