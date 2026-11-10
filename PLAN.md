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
- [ ] Add failing differential tests for sparse movements, compact maps, mixed
  products, direct boundary writes, structural zeros and loop contracts.
- [ ] Implement `passes/lowering/plans.py`: compact regions, four forms and
  shared emission through the loop compiler. Choose sparse traversal order in
  lowering and fold boundary permutations into access tables.
- [ ] Replace expanded mixed-product contribution tables with loaded-bound CSR
  row traversal and its transposed counterpart. Reduce stored entries.
- [ ] Verify focused checks and perturb the new gates. Preserve dense snapshots.
- [ ] Run the detached full pre-merge hook through the suite queue, open a PR,
  cross-review it, confirm fixes and queue it. Reference `Part of #46`.

## Stored-coordinate derivatives and workload gate, second pull request

- [ ] Add failing tests for sparse tangent patterns and shared multiseeds,
  sampled mixed-product adjoints, and banded sparse-leaf gradient/Hessian storage.
- [ ] Remove mixed-product AD densification. Keep the sparse edits local in
  forward/reverse/sparsity; evaluate elementwise partials on stored coordinates.
- [ ] Port the Maros-Meszaros problems and independently recomputed references
  from devrush, retaining source and license. Gate KKT assembly and products,
  Jacobian patterns and absence of full matrix buffers.
- [ ] Exercise finite differences, every kernel and every region form. Show each
  new gate failing under a temporary perturbation.
- [ ] Run focused checks then the detached full pre-merge hook. Open the final
  PR, cross-review, confirm fixes, publish the verdict line and queue it.

No sparse-sparse lowering, seed-axis compaction, fusion, loaded-bound lanes,
sparse derivative output leaves, pattern holes or sparse user documentation.
Record structural-zero semantics in internal notes and tests for #307 to publish.
Run-time measurements wait for the milestone. Evidence here uses operation
counts, generated C size and compile time. Delete this temporary plan when done.
