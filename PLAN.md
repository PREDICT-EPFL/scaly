# Sparse kernels, issue #46

This is layer 2 of #45 -> #46 -> #47 in the unattended compiler run. Follow
`~/.scaly-suite/RULES.md` and `internal/notes/compiler_run_2026_11_10/plan.md`
on main. The roadmap's decision 5 and Sparse tensors section decide the design.
Section 1's packet is on main under the same run folder in `sparse_values/`.

Base: `origin/t3code/45-accumulators`. Follow the coordinator's rebase handoff
below before the first AD push. Do not change #45's branch. Use `gh stack`.
Existing overlapping work is listed in the dispatch brief.

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
- [x] Run the detached full pre-merge hook through the suite queue, open a PR,
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

Forward PR #349 is merged into the stack's parent branch, not main. Both reviewers
approved after their delta checks. All fixes were pushed together once. The full
hook passed 3,042 tests with 55 solver skips plus types, format, lint, smoke,
docs and the private-name check. Its source and compile counts are in the forward
note. All its delegate threads are settled.

The coordinator decided #315 now lands #45 and the forward layer together. The
AD branch is based on its published parent 0758bf13. The #45 owner is rebasing that
parent onto main and owns its conflict resolution, Opus check and lease push.
Do not push to the parent's branch. Do not push AD or open its PR until the
coordinator sends the rebased parent head. Then rebase only the AD commits with
`git rebase --onto <new head> 0758bf13`, run focused checks and one detached full
hook, and open the AD PR. Queue AD only after #315 merges and the AD PR targets
main. Never enable auto-merge for a pull request targeting another feature branch.

The AD branch includes the forward review fixes and the stored-index sharing,
plus the exact IPM upper-triangle workload gate and direct boundary writes. The
48 problem archives are unchanged from devrush; its implementation was imported
through a git-backed loader only to produce references and was not copied.
There are 36 AD cases; the duplicate scalar/block infinity cases now live in the
forward layer. The previous AD hook passed 3,124 tests and 55 skips before the
parent review fixes. The combined local branch passed 137 sparse-plan, AD and 48-problem workload
cases with two workers, plus type checking. Refresh source/compile counts if the
incoming parent changes generated C. Do the final full hook only after the
rebased parent arrives.

The section 1 packet's banded quadratic reproduction runs all stages. At n=100
and n=400, largest gradient buffers are 100 and 400, and sparse Hessian buffers
are 300 and 1,200. Perturbation evidence is in the AD note.

Sparse SELECT moved to #359, blocked by #30 and #46, after #320 left the merge
queue. The roadmap departure is in this AD layer. The full prepared test file is
preserved at https://github.com/PREDICT-EPFL/scaly/issues/359#issuecomment-6779722333.
Do not rebase onto main for SELECT or implement it in this branch. The second
PR's visible description must state the departure. Its cross-review is still
outstanding; no full #46 verdict for #47 has been issued.
