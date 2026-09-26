# Handoff: the PIQP-in-Scaly work, for Claude Code sessions

This work started in Claude Cowork sessions (Tiers 1–2, 2026-09-25/26) and moves to Claude Code
here. Read this first, then the documents it points to. It holds how the work is done; the plan
holds what is to be done.

## Where things stand
- **Plan:** `internal/notes/piqp_plan.md` (v4.8), the living plan: goal, principles, Tiers 1–6, gates, kill criteria, decisions, change log.
  - Its twin, the claude.ai project doc `claude/piqp-plan.md`, can't be reached from Claude Code. From now on, the repo copy is authoritative.
- **Branches and tags:**
  - Tier 1 is done: tag `tier1-complete`, branch `claude/tier1-primitives`.
  - Tier 2 is done: tag `tier2-complete`, branch `claude/tier2-sparse`.
  - Tier 3 is done: tag `tier3-complete`, branch `claude/tier3-ipm` (from `claude/tier2-sparse`).
  - The chain is `main` → `t3code/plan-compiler-performance-passes` → `claude/tier1-primitives` → `claude/tier2-sparse` → `claude/tier3-ipm`. None of it is merged or pushed.
  - Start Tier 4 on a new branch from `claude/tier3-ipm`.
- **Status summaries:** `tier1_implementation_status.md`, `tier2_implementation_status.md` and `tier3_implementation_status.md`.
  - Full reports: `tier1_pr*_report.html`, `tier1_summary_report.html`, `tier2_pr1..9_report.html`, `tier2_review_report.html`, `tier3_pr0a..5_report.html` and `tier3_review_report.html` (the Tier 3 summary and hand-off).
  - Historical gap analysis: `programmatic_branch_analysis.md`.
- **Open items:** `internal/todo.md`, next id C-135. Check the list for an id before taking it: a parallel session's entries once collided with this work's (C-125, C-126).
  - Tier 1 and 2 leftovers: C-89 … C-99, C-111, C-114 … C-117, C-122.
  - Relevant to Tier 4: **C-114** (unpadded factor updates: the factorization is 40–60% of a sparse IPM step), **C-117** (the dense Cholesky kernel, 80% of a dense step), **C-130** (QRECIPE's sparse iteration count), **C-131** and **C-134** (step overheads the T3-R review measured).
  - C-126 (TinyMPC) is checked off in the committed list, but its files (`examples/tinympc/`, `tests/integration/test_tinympc.py`, `notes/tinympc_benchmark_report.html`) are Colin's and not committed.
- **Suite:** 2 218 passed and 42 skipped on the Mac (Apple clang 21) at the end of Tier 3, with Colin's untracked test files left out (`--ignore` them, or the node-ID baseline check fails on their ids).
  - The IPM tests also pass with `-ffp-contract=fast` and `off` (a `SCALY_CC` wrapper), which bracket what GCC does.

## How Colin wants each PR done
1. **A todo item.** Each PR gets its id (`C-1xx`) in `internal/todo.md`, checked off with a one-paragraph summary when done.
2. **Thorough tests, then mutation checks.** For each fix or feature:
   - break it by hand;
   - check that a test fails;
   - restore it.
   A mutant that survives gets a new test, or the dead code goes. Report the mutants and the result.
3. **A benchmark script.** It goes in a per-tier timing folder (for Tier 3, e.g. `internal/notes/perf_2026_MM_DD_tier3/`), with a row in that folder's `README.md`. Measure the Scaly side against a baseline whenever one exists (vendored PIQP, SciPy, a C reference).
4. **An HTML report.**
   - Name: `internal/notes/tier3_prN_report.html`.
   - Style: copy the `<head>` and CSS of an existing report such as `tier2_pr9_report.html`.
   - Sections: summary box, what was built, tests and mutation checks, benchmarks, gates status, next.
   - Colin prefers HTML reports and brief, high-signal prose.
5. **The node-ID baseline.** Regenerate `tests/baseline/pytest_nodeids.txt` from the full collection with the recipe in `docs/dev/contributing.md`. The root `conftest.py` refuses extra ids as well as missing ones, so a union of old and new ids fails once a test is renamed.
6. **One commit per PR** on the tier branch. Don't cite branch commit hashes anywhere (AGENTS.md).
7. **Close each tier** with:
   - an agent review round: parallel reviewers for lowering/IR, AD, the library, performance, and docs/tests, then fixes and optimizations;
   - a summary report;
   - the plan updated (status line, results, change log);
   - a `tierN-complete` tag.
8. **Leave Colin's untracked or local files alone:** `generated/`, `pla/`, `mymodule.py`, `docs/.obsidian`, `.DS_Store` files, `internal/notes/pipeline_analysis_2026_09_22.html`, and the modified `.python-version`.

## Decisions already taken (don't reopen)
- **Generated code only.** No BLAS/LAPACK/sparse libraries in shipped code; external kernels are benchmark baselines only.
- **Generic sparse before multistage/Riccati structure** (Tier 5).
- **Tier 2 is general-purpose linear algebra.** PIQP-specific choices belong to Tiers 3–4.
- **Loops stay loops.** Unrolling only below the `dense_unroll`/`sparse_unroll` thresholds.
- **The JIT compiles at `-O2`,** adding `-ftree-vectorize` only for GCC before 12 (C-107; T3-R narrowed it).
- **The generated IPM is held to PIQP's run of the same backend,** by decision trace, wherever PIQP's own two backends agree; the NumPy reference is the step-by-step oracle.
- **The dense backend is a condensed Cholesky,** the sparse one `SparseLDL` on the whole KKT matrix.

## Decisions to take with Colin at the start of Tier 4 (from the plan)
1. **Parity semantics:** decision traces plus tolerances, as Tier 3's gate uses; and whether the drop-in keeps PIQP's quirks (`piqp_plan.md`, Tier 3 notes) and the dense backend's noise-pivot rule.
2. **Baselines:** Mac-only vendored baselines and the SQP corpus dump.
3. **Speed levers:** C-114 before or after the drop-in; a blocked dense Cholesky (C-117).

## Tier 2 API to build on
```python
import scaly as sc
from scaly import linalg
K = linalg.SparseMatrix.block([[P.add_diagonal(rho), None], [A, linalg.SparseMatrix.identity(m) * -delta]])  # lower triangle
fact = linalg.SparseLDL(K)                  # analysis now; schedule="auto" | "scan" | "unroll"
x = fact.solve(b, refine=2)                 # or refine=5, tol=1e-12 (adaptive, a while_loop)
fact.inertia(); fact.health(signs=s, pivot_tol=0.0, x=x)   # bool / float counts in generated code
L = linalg.cholesky(Cdense); y = linalg.cho_solve(L, r)
sc.scan(body, init, xs, length=N, index=True); sc.while_loop(cond, body, init, max_iter=M)
sc.take / sc.put / sc.put_add; sc.custom_derivative(fn, jvp=, vjp=, sparsity=)
with sc.options(sparse_unroll=..., dense_unroll=..., max_trajectory=...): ...
```
Worked uses: `examples/sqp_newton_sparse.py`, `examples/kalman_update.py`, `tests/linalg/`.

## Traps learned the hard way
- **`==` on an `Expr` is identity** (hash-consing needs it). Use `sc.equal`. `sc.where` refuses a Python bool.
- **`SparseLDL` needs every diagonal entry stored.** A zero block of `SparseMatrix.block` stores nothing; use `add_diagonal`, even with zeros.
- **One `name=` per factorization in a graph.** Two different Functions sharing a name are refused at lowering.
- **Loop invariants go in `while_loop(..., params=...)`** (C-112), not the carry. Write carry updates as a chain of `put`/`put_add` with constant indices where no update reads an entry it, or a later update, writes; then the loop updates the carry in place. `SparseLDL._refined_solve` is the pattern: the factor, K and b are params, `put_add` updates `x`, then `put` updates `r` reading the updated link.
- **Derivatives in a slice of the carry** work since C-120 (they silently returned zero before).
- **Reverse mode through a loop stores every carry.** `max_trajectory` refuses anything too large. Give solver-like loops an implicit rule with `sc.custom_derivative`, as `SparseLDL.solve` does, so the loops are never differentiated.
- **Straight-line code costs about 1.2 ms of generation per scalar operation.** Keep unrolling for tiny kernels.
- **For KKT systems whose `H` may be indefinite (SQP), check `inertia() == (n, m, 0)`,** not `health(signs=...)`. The per-row sign check is for quasi-definite matrices, which PIQP's regularized KKT systems are.
- **Integers and bools cross the ABI as doubles.**
- **Nested loops need their body as a Function over symbols.** Build it once per structure and call it everywhere (`scaly.solvers.ipm.Kernels`): the IPM's initial point and loop then share one factorization, and no two Functions share a name.
- **Condensed Cholesky pivots are rounding noise late in an IPM.** Their sign decides PIQP's retries, and no two kernels agree on it: compare traces only where PIQP's own two backends agree (`piqp_trace.backends_agree`).
- **Time on the Mac with an A/B, not against old numbers.** Spotlight indexing can double a timing; interleave old (a `git worktree` of the last commit) and new runs and take minima.
- **Warm both sides up.** Apple Silicon runs the first calls of a burst up to 1.6x slower; a PIQP run is one cold solve unless `piqp_trace.run(..., repeat=k)` repeats it in-process. Compare minima over equally many samples.
- **Profile generated code in C, not through Python.** A call through the Function layer costs ~3 µs plus ~1 µs per input; run the piece K times in a `while_loop` whose carry depends on it through a run-time zero (Scaly folds `0.0 * x`) and difference two K.
- **Run the whole suite for any IR, AD or codegen change** (AGENTS.md). Those paths break subtly.
