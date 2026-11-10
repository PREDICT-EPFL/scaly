# Sparse kernels, issue #46

This is layer 2 of #45 -> #46 -> #47 in the unattended compiler run. Follow
`~/.scaly-suite/RULES.md` and `internal/notes/compiler_run_2026_11_10/plan.md`
(the coordinator checkout supplies the latter until #291 merges). The roadmap's
decision 5 and Sparse tensors section decide the design. Read section 1's packet
from `origin/compiler-run/sparse-values-packet` until #308 merges.

Base: `origin/t3code/45-accumulators`. Before the first push, if #315 merged,
rebase only this task's commits onto `origin/main`. Do not change #45's branch.
Use `gh stack`. Existing overlapping work is listed in the dispatch brief.

## Forward kernels, first pull request

- [x] Read the run rules, plan, issue discussion, architecture, codebase,
  conventions, maintenance context and sparse boundary implementation.
- [x] Add failing differential tests for sparse movements, compact maps, mixed
  products, direct boundary writes, structural zeros and loop contracts.
- [x] Implement `passes/lowering/plans.py`: compact regions, four forms and
  shared emission through the loop compiler. Choose sparse traversal order in
  lowering and fold boundary permutations into access tables.
- [x] Replace expanded mixed-product contribution tables with loaded-bound CSR
  row traversal and its transposed counterpart. Reduce stored entries.
- [x] Verify focused checks and perturb the new gates. Preserve dense snapshots.
- [ ] Run the detached full pre-merge hook through the suite queue, open a PR,
  cross-review it, confirm fixes and queue it. Reference `Part of #46`.

## Stored-coordinate derivatives and workload gate, second pull request

- [x] Add failing tests for sparse tangent patterns and shared multiseeds,
  sampled mixed-product adjoints, and banded sparse-leaf gradient/Hessian storage.
- [x] Remove mixed-product AD densification. Keep the sparse edits local in
  forward/reverse/sparsity; evaluate elementwise partials on stored coordinates.
- [x] Port the Maros-Meszaros problems and independently recomputed references
  from devrush, retaining source and license. Gate KKT assembly and products,
  Jacobian patterns and absence of full matrix buffers.
- [x] Exercise finite differences, every kernel and every region form. Show each
  new gate failing under a temporary perturbation.
- [ ] Run focused checks then the detached full pre-merge hook. Open the final
  PR, cross-review, confirm fixes, publish the verdict line and queue it.

No sparse-sparse lowering, seed-axis compaction, fusion, loaded-bound lanes,
sparse derivative output leaves, pattern holes or sparse user documentation.
Record structural-zero semantics in internal notes and tests for #307 to publish.
Run-time measurements wait for the milestone. Evidence here uses operation
counts, generated C size and compile time. Delete this temporary plan when done.

## Current handoff

Forward PR #349 is draft, based on #315. Its remote head predates the local review fixes.
Local forward commits contain the normalization fixes.
The final detached pre-merge hook passed with 3,031 tests and 55 skips, smoke and docs.
The sol correctness review found three normalization bugs, all fixed with seven
regressions that fail on the original head. The low-effort delta check approved all three findings.
The main Opus review failed on its provider rate limit and remains missing; do not
queue #349 or claim its main review passed. Two Opus attempts failed on the provider rate limit. Retry the same reviewer after
the provider recovers, with a new delegate task. Keep its findings with the one allowed
combined review-fix push. No review-fix push has happened.

The coordinator moved max/min stored reductions to #66, because REDUCE with kind
arrives there. Roadmap edits and the note are in the local forward commits. The PR
description and #66 comment are updated. No other scope change.

The AD implementation and 48 problem/reference fixtures are in the checkout.
The references were computed from devrush through a git-backed import loader,
not copied code. All 48 KKT cases and 24 AD cases pass. A focused compiler run
passed 1,365 tests, and the perturbation evidence is in the internal AD note.
The AD full pre-merge run is next. SELECT remains part of this PR and #46: it
needs #320 to land, then integration in the compact map and stored partials.
The current base has no SELECT operation. Do not close #46 before that is tested.
