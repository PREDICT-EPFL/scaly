# Program IR migration roadmap

> **Status:** foundations harvested onto this branch (`program-ir-migration`), suite green.
> **Goal:** make the semantic-IR → Program-IR → backend-renderer architecture the *sole* CPU
> compilation path, at full parity with `main`, with the legacy tape-based C renderer removed —
> **without** introducing new ops, GPU renderers, or solver changes along the way.

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
> `LoweringError` and survived only because `program_c` silently fell back to `codegen/c.py`.
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

The old `lowering.py` (930 lines) and `codegen/program_c.py` (322 lines) contain **correct per-op
lowering recipes** for the ops they cover (elementwise, MAP, CALL, matmul, SUM, transpose,
axis-0 stack/concat). But the dispatch is a hand-rolled `if/elif node.op == …` chain, and the
depth gaps (integer-index SLICE, large CONST) are hard-coded limits in helpers. **Copy the recipe
math; do not copy the dispatch structure** — we want an `Ops`-keyed registry (one self-contained
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
| `Ops.SUM_AXIS` (axis-aware reduction) | `b83fc63` |
| Batched matmul (fwd + AD + lowering) | `1eb582e`, `61cd39b`, `1009b5c`, `b4c4e0f` |
| `mean(axis=…)` sugar | `938509a` |
| `Ops.SCAN` reservation | `35ab83d` |
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
2. **Registry dispatch.** Lowerer (semantic `Ops` → Program IR) and renderer (`POps` → C) each
   route through a table keyed by op, with one self-contained, individually-testable rule per op.
   This is the "easy to add on top" property — new ops and GPU backends later are new rules, not
   edits to a monolith.
3. **Green at every commit.** `uv run pytest -n=auto tests/` passes, and the benchmark correctness
   checks pass. The benchmark suite (`benchmarks/scalability_sweep.py`, tracking eq-Jac,
   unbumpercars ineq-Jac, safety filter) is a *regression guardrail*, not just a perf demo.
4. **Foundations frozen.** `program.py`, `spec.py`, and the dtype model are stable. Extend them via
   new rules/ops; don't restructure them mid-migration.
5. **No new ops, no GPU renderers, no solver changes** until CPU parity lands and merges. The
   migration's scope is *exactly* the op set the existing tests + benchmarks already use.
6. **Interpreter + tape stay** as the explicit reference oracle throughout. ~50 tests, the
   benchmark correctness checks, and `rewrite.py` constant folding depend on `eval_interpreter` /
   `Expr.eval`. Their removal (if ever) is a *post-parity* decision, separate from deleting the
   legacy C renderer. `solver_c.py`'s host-wrapper codegen is likewise retained as a sanctioned
   non-Program-IR path (oracles flow through the new lowerer).

---

## 4. Migration plan (depth-first, benchmark-driven)

Each step ends green and makes Program IR the **sole** path for the ops it migrates.

- **Step 0 — Foundations.** ✅ Done. dtype/device model, verifier, Program IR vocabulary +
  verifier + printer harvested; suite green (155 passed, 6 skipped).

- **Step 1 — Lowerer + renderer skeleton.** ✅ Done. `lowering.py` (registry dispatch via
  `@lowers(...)`, keyed by semantic `Ops`) + `codegen/program_c.py` (compact per-op maps; emits the
  full universal-ABI TU so `jit.CompiledFunction` dispatches it unchanged). Covered as sole path:
  all elementwise unary/binary with identical operand shapes, `RESHAPE` (alias), small `CONST`.
  Selection is opt-in via `ALLOY_USE_PROGRAM_IR_C=1` and **strict** — an uncovered op raises
  `LoweringError` (no silent fallback) unless the explicit escape hatch `ALLOY_PROGRAM_IR_FALLBACK=1`
  is set. `tests/alloy/test_program_migration.py` is the self-certifying harness: it renders the
  Program IR source directly (loud on gaps), confirms `render_c_source` selects that exact source
  under the flag, and matches the interpreter oracle. Temporaries are stack-local arrays (`sz_w=0`);
  workspace packing is deferred. Not yet covered (raise loudly): broadcasting, `SLICE`, large
  `CONST`, `SUM`, `MATMUL`, `TRANSPOSE`, `GATHER`/`SCATTER`, `STACK`/`CONCAT`, `CALL`/`MAP`.

- **Step 2 — The day-one blockers.** Generalize `SLICE` (integer index, multi-dim, strided) and
  add a `CONST_BUFFER` Program IR op for large constants. **Exit:** the *forward* tracking and
  unbumpercars functions render through Program IR as sole path.

- **Step 3 — Core op recipes.** Port `MATMUL`, `SUM`, `TRANSPOSE`, `CALL`, `MAP` from the
  reference branch into registry rules; delete their legacy handlers. **Exit:** forward safety
  filter (incl. dense MLP) renders through Program IR.

- **Step 4 — Sparse + derivative assembly.** Large `GATHER`/`SCATTER` via constant index tables,
  elementwise **broadcasting** (bias-add etc.), multi-axis `STACK`/`CONCAT` as needed. **Exit:**
  the eq/ineq-Jacobian and sparse-Jacobian functions for all three workloads render through
  Program IR.

- **Step 5 — Flip default + benchmark validation.** Make Program IR the default (and only) CPU
  renderer. Run `benchmarks/` and confirm **no material regression** in generated source size or
  runtime vs `main` — in particular that the loop-based matmul lowering is competitive with the
  legacy "block" path (this is the "performant on the benchmarks" bar).

- **Step 6 — Delete legacy.** Remove `codegen/c.py`'s scalar renderer and the
  `program_c → legacy` fallback glue. Keep the interpreter/tape as the documented debug oracle
  (rule 6). Establish a fresh source baseline against the Program IR renderer.

- **Step 7 — Docs + merge.** Update `docs/spec.md` and this file; merge to `main`.

---

## 5. Definition of done (merge bar)

- Every function exercised by `tests/` and `benchmarks/` renders through Program IR as the **sole**
  CPU renderer — verified with the no-fallback strict mode in CI.
- `codegen/c.py`'s legacy scalar renderer is deleted; no `ALLOY_USE_PROGRAM_IR_C` flag remains.
- Benchmarks show no material source-size or runtime regression vs `main`.
- `solver_c.py` host-wrapper path retained; solver oracles lower through Program IR.
- `uv run pytest -n=auto tests/`, `uv run ruff check`, `uv run ty check` all clean.

Explicitly **out of scope** for this merge (and tracked for follow-up PRs from the reference
branch): OpenCL, GPU renderers, a tinygrad-style general range scheduler, full einsum, new
semantic ops (`SUM_AXIS`, batched matmul, `SCAN`), structured sparsity, and Phase 11 differentiable
solvers. The architecture leaves the seams for these (device on `Function`, `RangeKind` in Program
IR, dtype everywhere) — they re-land as additive rules, not rewrites.
