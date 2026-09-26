# Handoff: the PIQP-in-Scaly work, for Claude Code sessions

This work started in Claude Cowork sessions (Tiers 1–2, 2026-09-25/26) and moves to Claude Code
here. Read this first, then the documents it points to. It holds how the work is done; the plan
holds what is to be done.

## Where things stand
- **Plan:** `internal/notes/piqp_plan.md` (v4.3), the living plan: goal, principles, Tiers 1–6, gates, kill criteria, decisions, change log.
  - Its twin, the claude.ai project doc `claude/piqp-plan.md`, can't be reached from Claude Code. From now on, the repo copy is authoritative.
- **Branches and tags:**
  - Tier 1 is done: tag `tier1-complete`, branch `claude/tier1-primitives`.
  - Tier 2 is done: tag `tier2-complete`, branch `claude/tier2-sparse`.
  - The chain is `main` → `t3code/plan-compiler-performance-passes` → `claude/tier1-primitives` → `claude/tier2-sparse`. None of it is merged, and `claude/tier2-sparse` is not pushed.
  - Start Tier 3 on a new branch `claude/tier3-ipm` from `claude/tier2-sparse`.
- **Status summaries:** `tier1_implementation_status.md` and `tier2_implementation_status.md`.
  - Full reports: `tier1_pr*_report.html`, `tier1_summary_report.html`, `tier2_pr1..9_report.html` and `tier2_review_report.html` (the Tier 2 summary and hand-off).
  - Historical gap analysis: `programmatic_branch_analysis.md`.
- **Open items:** `internal/todo.md`, next id C-127.
  - Tier 1 leftovers: C-89 … C-100.
  - Tier 2 leftovers: C-107 (a decision for Colin: `-O3` by default), C-111, C-112, and C-114 … C-117.
  - Relevant to Tier 3: **C-112**, loop-invariant `while_loop` inputs for the IPM outer loop, and **C-114**, unpadded factor updates, since the factorization dominates IPM time.
- **Suite:** 1 460 passed and 88 skipped in the Linux VM.
  - Four failures there come only from the VM's environment: 3 in `tests/benchmarks/test_sweep.py`, and `plugins/scaly-sqp/tests/test_nlp_sqp.py::test_casadi_external_sqp_validates_oracle_and_bound_shapes`.
  - **Tier 2 has never been run on the Mac. Run the whole suite there first** (`uv run pytest -n=auto`). Watch for Apple clang differences, e.g. `internal/notes/macos_clang_call_miscompile.md`.

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

## Decisions to take with Colin at the start of Tier 3 (from the plan)
1. **T3-0:** a NumPy reference IPM as the test oracle, plus a curated Maros–Mészáros subset (and infeasible LPs, SQP/MPC sets; Apache-2.0) in `tests/data`. Also a PIQP trace harness built from the vendored headers (`plugins/scaly-piqp`).
   - Gate: the NumPy reference reproduces PIQP decision traces on ≥ 90% of the small MM subset.
2. **The dense backend:** condensed Cholesky `C = P + reg + AᵀA/δ + GᵀW⁻¹G` on the Tier 2 dense kernels, next to the sparse backend (full KKT with `SparseLDL`).
3. **The outer loop:** `while_loop` whose carry packs the IPM state, or C-112 first (loop-invariant inputs).

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
- **Run the whole suite for any IR, AD or codegen change** (AGENTS.md). Those paths break subtly.
