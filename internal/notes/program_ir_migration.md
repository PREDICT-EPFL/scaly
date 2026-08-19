# Program IR migration roadmap

> **Frozen note.** Kept for the record, not maintained. For how alloy works now, see
> [`docs/how_it_works/architecture.md`](../../docs/how_it_works/architecture.md).

> **Status (2026-06-08):** Steps 0–5b done — **functional *and* performance parity reached**. Every
> workload compute function (forward + `jac` + `spjac` for tracking and unbumpercars) lowers,
> optimizes, and renders through Program IR as the sole path and is **bit-identical** to the legacy
> renderer. The Step 5b perf/scale gaps are **closed**: a third architectural leg —
> Program-IR → Program-IR **optimization passes** (`src/alloy/passes/program.py`) — now ports the legacy
> renderer's custom optimizations as explicit, individually-testable passes. Re-measured
> head-to-head (same box, raw C-entry, best-of-5, output bit-identical):
>
> | workload | before 5b | after 5b | `sz_w` (legacy → PIR) |
> | --- | ---: | ---: | --- |
> | tracking ms spjac N=50 | 1.05× | **1.03×** | 3154 → 2100 |
> | unbumpercars spjac C=2 | 1.25× | **1.00×** | 0 → 0 |
> | unbumpercars spjac C=8 | 1.22× | **0.98×** | 6664 → 6664 |
>
> **Update (Step 5c done):** Program IR is now the **default** CPU renderer for host functions with
> no solver in their call graph; solver-bearing functions stay on `codegen/solver`/legacy via a guard. The
> full suite is green on a cold cache in parallel (228 passed). A pre-existing cold-cache solver
> link-order bug surfaced and was fixed (libs now follow the source object).
>
> **Update (Step 6 done):** the legacy tape-based scalar renderer in `codegen/c.py` is **deleted**.
> Program IR is now the **sole** CPU path: solver oracles and the host functions that call solvers
> all lower through `passes/lowering.py`/`codegen/c.py`; a `SolverFunction` CALL is lowered *opaquely* (its
> body is not lowered) and the `codegen/solver` wrapper template — the one sanctioned non-Program-IR path
> (rule 6) — is the only hand-written C left. `render_c_source` is now a thin orchestrator that emits
> the Program-IR `_raw` callees, splices in the solver wrappers, and reuses the Program-IR ABI entry.
> The `ALLOY_USE_PROGRAM_IR_C` / `ALLOY_PROGRAM_IR_FALLBACK` flags are gone; mixed-device CALL is a
> hard `LoweringError` (no fallback). `_raw` input params are now `const`-qualified so solver wrappers
> pass `const double*` arguments without discarding qualifiers. Full suite green on a **cold cache, in
> parallel** (228 passed, 6 skipped); `ruff`/`ty` clean. Remaining: **merge (Step 7)**.
>
> **Update (2026-06-10):** the Program IR pass pipeline now includes `unroll_unit_loops` after
> `fuse_elementwise` and before `pack_workspace`. It removes static zero-trip loops and substitutes
> the loop variable in static one-trip loops, keeping canonical loop-shaped producers available for
> fusion first and then erasing scalar-loop noise before workspace packing/rendering. On the tracking
> NMPC structured equality sparse-Jacobian C for N=10, this removed 132 `for (... < 1)` loops and
> reduced source from 29.5 KB / 652 LOC to 17.6 KB / 388 LOC; runtime stayed within benchmark noise.
>
> **Goal:** make the expression-IR → Program-IR → backend-renderer architecture the *sole* CPU
> compilation path, at full **feature *and* performance** parity with `main`, with the legacy
> tape-based C renderer removed — **without** introducing new ops, GPU renderers, or solver changes
> along the way. No parity gates remain open; the legacy renderer is gone.

This document is the north star for the migration. It records *why* we restarted from `main`
instead of finishing the previous attempt, *what* we harvested, *what* we deliberately deferred
(and where to find it), and the *discipline* that keeps every commit green. A future agent should
be able to pick this up cold and either continue the migration or re-introduce the deferred work.

---

## 1. Why this branch exists (the rationale)

A previous branch, **`roadmap-semantic-program-ir`**, executed an ambitious 12-phase plan
(`docs/roadmap.md` *on that branch* has the full design, tinygrad references, and per-phase
exit criteria — read it for the deep context). It proved a great deal:

- the Program IR design (`program.py`),
- a PatternMatcher-backed verifier (`spec.py`),
- an interned dtype + device-placement model (`types.py`),
- **two working GPU backends** (CUDA and Metal, end-to-end),
- axis-aware reductions and batched matmul,
- structured sparsity descriptors.

But it optimized for **breadth over depth**: a "first slice" of every phase, all wired together
behind a feature flag (`ALLOY_USE_PROGRAM_IR_C`) with a **silent fallback** to the legacy
renderer. The fallback masked the truth. An empirical probe (2026-06-08) showed:

> The Program IR path could render **none** of the production benchmark workloads
> (tracking eq-Jacobian, unbumpercars ineq-Jacobian, safety filter). Every one hit a
> `LoweringError` and survived only because `codegen/c` silently fell back to `codegen/c.py`.
> The first blockers were pervasive and basic: **`SLICE` could not do integer/scalar component
> indexing** (`z[3]`, `p[0]`) — which appears in the *forward* pass of every workload — and
> **`CONST` was capped at 16 elements**, blocking every AD-seed constant and all MLP weights.

So "finish the old branch" is deceptively expensive, for three reasons:

1. **Every "✓" needs re-verification.** Green-via-fallback means the maturity markers can't be
   trusted; you'd re-audit each phase under a strict no-fallback mode anyway.
2. **Two renderers coexist for the whole transition.** The fallback glue is throwaway complexity
   you carry until the very end.
3. **Unfinished breadth is entangled with finished breadth.** GPU, new ops, and structured
   sparsity were all layered on the same toy lowerer, so a CPU-parity push has to work around them.

The decision: **reset the trunk, harvest the branch.** Start from `main`, lift the decoupled
foundations as-is, and redo only the lowering migration — this time **depth-first, one op at a
time, with no fallback**. That migration is *self-certifying*: if the suite is green with Program
IR as the *only* path for the ops migrated so far, the work is actually done — no hidden state.

The old branch was not wasted. It was a de-risking spike whose real output is design-validated
modules + the exact knowledge of where the depth gaps are. We keep it intact as the reference and
cherry-pick source.

---

## 2. The reference branch: `roadmap-semantic-program-ir`

Keep that branch — **do not delete or rebase it.** It is both the design reference and the
cherry-pick source for all deferred work. Its `docs/roadmap.md` has the full 12-phase plan.

### 2a. Already harvested onto this branch (clean, frozen foundations)

| What | Source commit | Files | Status here |
| --- | --- | --- | --- |
| Phase 1 — dtype + device model | `2564b7b` | `types.py`, `expr.py`, `function.py`, `abi.py`, `tape.py`, `__init__.py`, `test_dtype_device.py` | ✅ cherry-picked |
| Phase 2 — verifier specs (`Spec`/`verify_expr`, PatternMatcher) | `7537da8` | `spec.py`, `test_verifier.py` | ✅ cherry-picked |
| Phase 4 — Program IR vocabulary + verifier + printer | `80430a7` | `program.py`, `test_program_ir.py` | ✅ cherry-picked |

These three are **decoupled** — they touch no half-finished lowering code. Treat them as stable:
extend by adding rules/ops, don't refactor.

### 2b. Reference-only — port the *recipes*, rewrite the *dispatch*

The old `passes/lowering.py` (930 lines) and `codegen/c.py` (322 lines) contain **correct per-op
lowering recipes** for the ops they cover (elementwise, MAP, CALL, matmul, SUM, transpose,
axis-0 stack/concat). But the dispatch is a hand-rolled `if/elif node.op == …` chain, and the
depth gaps (integer-index SLICE, large CONST) are hard-coded limits in helpers. **Copy the recipe
math; do not copy the dispatch structure** — we want an `ExprOp`-keyed registry (one self-contained
rule per op) so adding/deepening an op is local. Reference commits for the recipes:

`88b6daf` elementwise · `c042508` program-C renderer · `4768858` SUM · `83736a0` MATMUL ·
`31c4311` GATHER/SCATTER · `653df0a` STACK/CONCAT · `9f965ad` SLICE (rank-1) + elementwise parity ·
`e40f962` CALL · `b925eb4` MAP · `4ab8e90` TRANSPOSE · `0243dae` inf/nan CONST.

### 2c. Deferred — re-introduce as separate PRs *after* CPU parity merges

| Deferred capability | Reference commits (on `roadmap-semantic-program-ir`) |
| --- | --- |
| CUDA backend (renderer + nvcc runtime) | `d54f2d5`, `8fdd9a0`, `9a54577` |
| Metal backend (MSL + libobjc runtime) | `18743d8`, `342dac4`, `8d6a5f0`, `bcd8862`, `271d176` |
| GPU schedule pass (kernelize, thread-bind) | `33b7071`, `8fdd9a0` |
| Structured sparsity descriptors | `ff67ef4`, `7ff18f1`, `e1605b7`, `6c755c3` |
| `ExprOp.SUM_AXIS` (axis-aware reduction) | `b83fc63` |
| Batched matmul (fwd + AD + lowering) | `1eb582e`, `61cd39b`, `1009b5c`, `b4c4e0f` |
| `mean(axis=…)` sugar | `938509a` |
| `ExprOp.SCAN` reservation | `35ab83d` |
| Mixed-device CALL | `cb53a5c`, `ba07b57` |
| Solver `DerivativePolicy` framework (Phase 11) | `ebfa441` |
| IPOPT-on-Linux build fix + JIT solver linkage | `56739da` (may already be on `main`; verify) |

---

## 3. Architecture & invariants (the discipline)

These rules are *why* the migration converges. Hold them on every commit.

1. **No silent fallback.** Migrating an op means Program IR becomes its *only* renderer, and the
   legacy handler for that op is deleted in the same change. A strict no-fallback env flag may
   exist as temporary scaffolding *during* the migration, but the end state has **no flag and no
   legacy renderer**.
2. **Registry dispatch.** Lowerer (expression `ExprOp` → Program IR) and renderer (`ProgramOp` → C) each
   route through a table keyed by op, with one self-contained, individually-testable rule per op.
   This is the "easy to add on top" property — new ops and GPU backends later are new rules, not
   edits to a monolith. **Optimizations are the third leg:** Program-IR → Program-IR passes in
   `passes/program.py`, registered in an ordered `PASS_PIPELINE` and run by `optimize_program` at the tail
   of `lower_function`. Each pass is a pure `PROGRAM → PROGRAM` rewrite — a new optimization is a
   new pass, not a special case threaded into the lowerer or renderer. (The legacy renderer baked
   inlining/lifetime-packing into one 1200-line module; the migration splits these into discrete
   passes so each is testable and composable, and so GPU schedule passes can slot in later.)
3. **Green at every commit.** `uv run pytest -n=auto tests/` passes, and the benchmark correctness
   checks pass. The benchmark suite (`benchmarks/scalability_sweep.py`, tracking eq-Jac,
   unbumpercars ineq-Jac, safety filter) is a *regression guardrail*, not just a perf demo.
4. **Foundations frozen.** `program.py`, `spec.py`, and the dtype model are stable. Extend them via
   new rules/ops; don't restructure them mid-migration.
5. **No new ops, no GPU renderers, no solver changes** until CPU parity lands and merges. The
   migration's scope is *exactly* the op set the existing tests + benchmarks already use.
6. **No Python execution shadow path.** Python calls and AOT both go through expression IR → Program
   IR → generated C. The old tape interpreter / `Expr.eval` reference path was deleted after CPU
   parity to keep semantics concentrated in one backend. The solver-wrapper codegen is still
   retained as the sanctioned non-Program-IR path (oracles flow through the Program IR lowerer);
   since 2026-07-15 the wrapper templates live in the solver plugins and `codegen/solver.py` only
   orchestrates them (`docs/solver_plugins.md`) — the sanction extends to plugin-provided
   templates, which must drive Program-IR-rendered oracle kernels and never hand-write oracle math.

---

## 4. Migration plan (depth-first, benchmark-driven)

Each step ends green and makes Program IR the **sole** path for the ops it migrates.

- **Step 0 — Foundations.** ✅ Done. dtype/device model, verifier, Program IR vocabulary +
  verifier + printer harvested; suite green (155 passed, 6 skipped).

- **Step 1 — Lowerer + renderer skeleton.** ✅ Done. `passes/lowering.py` (registry dispatch via
  `@lowers(...)`, keyed by expression `ExprOp`) + `codegen/c.py` (compact per-op maps; emits the
  full universal-ABI TU so `codegen.jit.CompiledFunction` dispatches it unchanged). Covered as sole path:
  all elementwise unary/binary with identical operand shapes, `RESHAPE` (alias), small `CONST`.
  Selection is opt-in via `ALLOY_USE_PROGRAM_IR_C=1` and **strict** — an uncovered op raises
  `LoweringError` (no silent fallback) unless the explicit escape hatch `ALLOY_PROGRAM_IR_FALLBACK=1`
  is set. `tests/alloy/test_program_migration.py` is the self-certifying harness: it renders the
  Program IR source directly (loud on gaps), confirms `render_c_source` selects that exact source
  under the flag, and matched the then-existing interpreter oracle. Temporaries are stack-local arrays (`sz_w=0`);
  workspace packing is deferred. Not yet covered (raise loudly): broadcasting, `SLICE`, large
  `CONST`, `SUM`, `MATMUL`, `TRANSPOSE`, `GATHER`/`SCATTER`, `STACK`/`CONCAT`, `CALL`/`MAP`.

- **Step 2 — The day-one blockers.** ✅ Done. Generalized `SLICE` (integer index drops a dim,
  slices keep one; multi-dim and strided via flat-index arithmetic built as scalar PNodes) and
  added a `constant`-address-space `const_buffer` (rendered `static const`) so constants of any
  size lower — these were the first two probe blockers. Re-probing the forwards confirms `SLICE`
  and `CONST` are cleared; they now block on `CALL`/`MAP`/`GATHER`, which are Step 3/4 ops — so the
  "forwards render end-to-end" milestone lands after Step 3.

- **Step 3 — Core op recipes.** ✅ Done. `SUM` (REDUCE loop), `MATMUL` (dot / matvec / vecmat /
  matmat; rank-3 batched deferred), `TRANSPOSE` (rank ≤ 4, permuted-index copy), `CALL` (callee
  lowered once per name into the shared registry, deduped per invocation, rendered `static inline
  <name>_raw`), and `MAP` (a `length` loop calling the callee with pointer-offset VIEW args).
  Mixed-device CALL raises loudly (deferred). Migration corpus +8. Re-probing the forwards: matmul/
  sum/transpose/call all matched the then-existing interpreter; the forwards then blocked only on `STACK` (tracking)
  and `GATHER` (unbumpercars) — Step 4 ops — so "forwards render end-to-end" lands with Step 4.

- **Step 4 — Sparse + assembly + broadcasting.** ✅ Done. `GATHER`/`SCATTER` (any size, via a
  `static const` int64 index table + one loop — no source blow-up), `STACK`/`CONCAT` (axis 0),
  and elementwise **broadcasting** (numpy right-aligned; size-1 and missing leading dims read index
  0 — covers bias-add and scalar consts). **Milestone reached:** the forward tracking (h=2/5) and
  unbumpercars (n=2/4/8) functions render through Program IR as the sole path and match the
  interpreter oracle; `test_program_migration.py` locked both forwards.
- **Step 4b — General-axis STACK/CONCAT + identifier sanitization (derivatives render).** ✅ Done.
  The derivative/sparse-Jacobian functions blocked on axis-1 `CONCAT`/`STACK` (gradient-column
  assembly) and then on a renderer bug: buffer/var/callee names carry `:` (e.g. `fwd:eq:z`) which
  is not a valid C identifier, and the entry symbol must match `jit`'s `_c_ident(fun.name)`.
  Generalized `STACK`/`CONCAT` to any axis (decompose / shift-or-insert / recombine via affine
  index PNodes) and route every emitted identifier through `_c_ident`. **Full workload parity:**
  forward + `jac` + `spjac` for both tracking (h=2/5) and unbumpercars (n=2/4/8) now render through
  Program IR and matched the then-existing interpreter; the tracking lock test covered all three kinds.

- **Step 5a — Benchmark validation (head-to-head).** ✅ Done — and it found real gaps. Measured
  the legacy vs Program IR renderers on the same Linux box (raw C-entry timing, pre-allocated
  buffers, best-of-5; output compared element-wise). *Numbers are this workstation, not the
  Apple-M-series figures in `docs/scalability.md` — the valid comparison is legacy-vs-Program-IR
  on one machine.*

  | workload | legacy | Program IR | runtime | source LOC | source bytes | `sz_w` | out diff |
  | --- | ---: | ---: | ---: | --- | --- | --- | ---: |
  | tracking spjac N=10 | 1.99 µs | 2.05 µs | 1.03× | 472 → 2483 | 17.8K → 77.4K | 0 → 0 | **0.0** |
  | tracking spjac N=50 | 8.23 µs | 8.60 µs | 1.05× | 473 → 2483 | 22.2K → 85.8K | 3154 → 0 | **0.0** |
  | unbumpercars spjac C=2 | 67.7 µs | 84.7 µs | 1.25× | 1588 → 5767 | 46K → 174K | 0 → 0 | **0.0** |
  | unbumpercars spjac C=8 | 279 µs | 340 µs | 1.22× | 1714 → 4485 | 89K → 204K | 6664 → 0 | **0.0** |

  Output is **bit-identical** to legacy (`0.0`), so functional parity holds. Tracking runtime is at
  parity (≤5%) and its LOC is constant in N. But three regressions surfaced (all expected from
  optimizations the migration deferred):

- **Step 5b — Close the perf/scale gaps.** ✅ Done. Added `src/alloy/passes/program.py` — the optimization
  leg (rule 2) — with an ordered `PASS_PIPELINE` run by `optimize_program` inside `lower_function`.
  The legacy renderer's custom optimizations are ported as discrete, individually-tested passes
  (`tests/alloy/test_passes.py`):
  1. **Workspace spilling — `pack_workspace`.** Lifetime-packs `private` BUFFERs into shared slots
     (greedy left-edge over statement order) and spills float64 slots ≥ 1024 doubles to the
     caller's `w[]` (`double* sN = w + offset;`), with a real `sz_w = own_spill + max(callee sz_w)`.
     Callee `_raw` functions now take a `double* w` tail and the caller passes `w + own_spill`.
     Port of `codegen/c.py::_compute_lifetimes` / `_pack_slots` / `_spill_plan`. Measured `sz_w`
     now **matches or beats** legacy (unbumpercars C=8: 6664 = 6664; tracking ms N=50: 3154 → 2100),
     so the largest cells render with bounded stack — the segfault gate is closed.
  2. **Runtime parity — `fuse_elementwise` + contiguous-slice aliasing.** `fuse_elementwise` inlines
     single-use `private` elementwise/slice/gather producers (a single-`STORE` `FOR` whose store
     index is the loop var) into their one consumer by substituting the producer's scalar RHS at the
     load site, then drops the producer loop+buffer — collapsing chains into one loop and fusing
     elementwise work into reductions. The recompute guard (`_max_load_executions ≤ producer_size`)
     is what keeps a producer *out of* a matmul/contraction operand position (where each element is
     read `m·n·k` times) — the critical correctness detail the legacy encodes by only inlining into
     elementwise consumers. The dominant unbumpercars cost turned out to be a **contiguous slice of
     the 35 k-element `p` input being copied every MAP call**; porting the legacy contiguous-slice /
     reshape **pointer aliasing** (`const T* tN = src + offset;`, in the SLICE lowering rule + an
     alias-aware `pack_workspace`) removed that copy. Result: unbumpercars **1.00–1.02×**, tracking
     ms **1.01–1.03×** (was 1.22–1.25× / 1.03–1.05×).
  3. **Source size.** Fusion cut the 2–4× blow-up to ~1.3–1.5× (tracking ms LOC is again *constant*
     in N: 2483 → 649). A later `unroll_unit_loops` pass erases the leftover scalar-loop boilerplate
     after fusion: for the tracking NMPC structured equality sparse-Jacobian, N=10 source dropped
     29.5 KB / 652 LOC → 17.6 KB / 388 LOC and `for (... < 1)` loops dropped 132 → 0. Runtime was
     unchanged within measurement noise, which is expected because `-O3` already optimizes most
     fixed-trip-count loop overhead. The remaining growth at large N is the gather/scatter **index
     tables** (`static const int64_t kN[...]`) that the legacy compacts via its tile-pattern peephole
     (`_detect_tile` / `_gather_tile_pattern`) and gather-of-transposed-concat peephole
     (`_gather_transposed_concat`). These are **source-size-only** (runtime is already at parity, the
     tables are `static const`) and are left as a follow-up — see "remaining" below.

  Also done: the ABI **header** `SZ_W` macro now agrees with the rendered source under the flag
  (`_abi_workspace_size` routes to the Program IR `sz_w`), so `render_c_module` is sound for AOT /
  benchmark consumers. The JIT now compiles to a process-unique temp `.so` then atomically renames,
  closing a cold-cache parallel-build corruption window (orthogonal robustness fix).

  **Remaining (deferred, non-gating):** (a) gather index-table compaction (the two legacy gather
  peepholes above) — purely source size at large N; (b) the sparse-const matvec path
  (`_sparse_const`) — not needed, unbumpercars matmuls live in callees and are already at parity;
  (c) re-running the **safety filter** sweep cells under the flag (blocked on an unrelated
  pre-existing cold-cache JIT bug, below). None gate the merge.

  **Pre-existing issue discovered (out of scope):** on a *cold* JIT cache, `test_solver_nesting`
  fails with `undefined symbol: piqp_…` because the JIT cache key hashes only the C source, not the
  link flags — so a `safety_filter` `.so` can get cached without `-lpiqpc`. Confirmed identical on
  clean `HEAD` with no flag (clean HEAD fails *more*: 12 vs 7 on this branch, since the atomic-rename
  fix helps). Flag-independent; touches the solver compile path (rule 5: no solver changes), so left
  for a separate fix. Warm-cache runs (the normal dev/CI loop) are green: full suite **219 passed,
  6 skipped** under the flag, and the migration self-cert (`test_program_migration.py`, 64) + pass
  tests (`test_passes.py`, 9) are green.

- **Step 5c — Flip default.** ✅ Done. `use_program_ir_renderer()` now defaults to true; Program IR
  is the CPU renderer for any host function with **no solver in its call graph**. The routing guard
  `_renders_through_program_ir` = `host AND not _uses_solver(fun)`, where `_uses_solver` uses
  `uses_piqp`/`uses_ipopt` (which traverse callees *and* `SOLVER_CALL` nodes — `_function_order`
  does not, so a plain function that merely *calls* a solver is correctly kept on legacy/`codegen/solver`).
  `ALLOY_USE_PROGRAM_IR_C=0` is a transitional escape hatch to force the legacy renderer (removed in
  Step 6). The two `test_map.py` source-structure asserts were already reconciled in 5b.

  **Fixed the cold-cache solver bug here (it was a link-order bug, not a flag bug).** Root cause:
  Linux `ld` defaults to `--as-needed`, and the JIT compile command placed `-lpiqpc`/`-lipopt`
  (from `solver_compile_flags`) *before* the source object — so the linker dropped the library (no
  `DT_NEEDED`) and the `.so` failed to `dlopen` with `undefined symbol: piqp_…`. Reordered the
  command to put link libraries after the source (`jit.py`). With this, the full suite is green on a
  **cold cache, in parallel** (`228 passed, 6 skipped`), with solvers actually JIT-compiling (they
  previously masked the bug by silently falling back to the old interpreter path under strict mode).

- **Step 6 — Delete legacy. ✅ Done — a deliberate solver-codegen refactor, not a mechanical
  cleanup.** Pre-6 the legacy scalar renderer still rendered solver **oracles** *and* the host
  functions that call solvers. Deleting it required routing all of them through Program IR while
  keeping the `codegen/solver` wrapper as the one sanctioned non-Program-IR path (rule 6). What landed, in
  order:
  1. **Const-qualified PIR `_raw` inputs.** Each PROC carries an `input_count` attr (emit_inputs runs
     before register_outputs, so inputs are the leading params); `codegen/c._render_raw_callee`
     `const`-qualifies the first `input_count` params. A solver wrapper passing `const double* in{i}`
     no longer discards qualifiers, and PIR→PIR calls passing input args (`arg[i]`) lost their
     warnings too. Inputs are read-only by construction, so `const` is sound.
  2. **Opaque solver lowering + solver workspace on PIR.** `lowering._ensure_callee` detects a
     `SolverFunction` callee: it does **not** lower the solver's `SOLVER_CALL` body (no rule), but
     it *does* lower the oracle Functions to PROCs and records the solver→oracle-name map on the
     PROGRAM (`solver_oracles` attr). `passes.pack_workspace` reads that map to size a solver call's
     workspace as `max(oracle sz_w)` (the wrapper passes its `w` straight through), so the caller
     spills correctly and a CALL-to-solver gets `callee_needs_w` right. So `program_ir_sz_w(fun)` is
     now the single source of `sz_w` for *every* host function, solver-bearing or not.
  3. **Rendered solver oracles + callers via PIR.** `render_c_source` is now a thin orchestrator:
     for a solver-bearing graph it lowers the whole thing once, then walks `_function_order`
     (topological — a solver sits after its oracle PROCs and before its caller) emitting each
     non-solver Function as a Program-IR `_render_raw_callee` and each `SolverFunction` as
     `render_solver_raw`, then the top function's Program-IR ABI entry (`codegen/c._render_entry`,
     reused). No forward references, single translation unit, solver includes spliced in.
  4. **Mixed-device CALL** is a hard `LoweringError` (a host→GPU call is meaningless on a CPU build);
     `test_uncovered_case_raises_loudly` asserts both `render_program_c_source` and `render_c_source`
     raise, no fallback.
  5. **Deleted** the scalar renderer (`_render_c_raw_function` / `_render_instruction` /
     `_elementwise_*` / `_matmul` / `_inline_scalar_table` / `_skipped_instructions` / `_detect_tile`
     / `_compute_lifetimes` / `_pack_slots` / `_spill_plan` / …) and the `ALLOY_PROGRAM_IR_FALLBACK` /
     `ALLOY_USE_PROGRAM_IR_C` flag glue. `c.py` shrank from ~1240 to ~300 lines. Kept
     `render_c_api_header` / `render_c_module` / `_c_ident` / `CModule`, `_function_order` / `_callees`,
     and a `_workspace_size` shim (now `= program_ir_sz_w`) so the benchmark imports keep resolving.
     The `codegen/solver` wrapper codegen is retained (rule 6). Three now-dead solver helpers
     (`uses_any_solver` / `_walk_tape_callees` / `solver_workspace`) were removed.
  6. **No golden-source rebaseline needed** — the suite's source checks are structural / numeric, and
     all pass.

  Verified: full suite **228 passed, 6 skipped on a cold cache, in parallel**, with PIQP/IPOPT
  nesting tests JIT-compiling through the new path; `ruff` / `ty` clean. Source compiles warning-free
  under `-Wall` (the lone `-Wextra` `unused-parameter` on an empty solver output predates this work).

- **Step 7 — Docs + merge.** Update `docs/spec.md` and this file; merge to `main`. Now unblocked.

---

## 5. Definition of done (merge bar)

- Every function exercised by `tests/` and `benchmarks/` renders through Program IR as the **sole**
  CPU renderer — verified with the no-fallback strict mode in CI. ✅ (functional parity reached)
- **Workspace spilling** lands so the largest sweep cells (tracking N=500, unbumpercars C=32)
  render and run without overflowing the stack. ✅ — `pack_workspace` (Step 5b.1); `sz_w` matches
  or beats legacy.
- **No material runtime regression vs the legacy renderer** on the benchmark workloads (target:
  within a few %). ✅ — `fuse_elementwise` + contiguous-slice aliasing (Step 5b.2); unbumpercars
  1.00–1.02×, tracking ms 1.01–1.03×.
- Source size does not regress materially. ✅ — fusion cut the blow-up to ~1.3–1.5× (tracking ms LOC
  constant in N); gather index-table compaction is a deferred source-size-only follow-up.
- `codegen/c.py`'s legacy scalar renderer is deleted; no `ALLOY_USE_PROGRAM_IR_C` flag remains. ✅ Step 6.
- `codegen/solver.py` host-wrapper path retained; solver oracles lower through Program IR. ✅
- `uv run pytest -n=auto tests/`, `uv run ruff check`, `uv run ty check` all clean. ✅ — now green on a
  **cold cache, in parallel** (228 passed, 6 skipped); the Step 5b cold-cache solver link-flag bug was
  fixed in 5c (link libraries follow the source object).

**Feature *and* performance parity is primordial.** Both halves are done (the functional half is
bit-identical, the Step 5b perf/scale gaps are closed) and the legacy renderer is gone. All that
remains is **Step 7 — merge to `main`** (and the `docs/spec.md` refresh that lands with it).

Explicitly **out of scope** for this merge (and tracked for follow-up PRs from the reference
branch): OpenCL, GPU renderers, a tinygrad-style general range scheduler, full einsum, new
expression ops (`SUM_AXIS`, batched matmul, `SCAN`), structured sparsity, and Phase 11 differentiable
solvers. The architecture leaves the seams for these (device on `Function`, `RangeKind` in Program
IR, dtype everywhere) — they re-land as additive rules, not rewrites.
