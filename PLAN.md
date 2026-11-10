# Sparse generated interfaces, issue #43

Stack layer 5 on `t3code/42-sparse-boundary`, PR #287. The PR base stays that branch.
The decided scope is the Sparse tensors section, decision 4 and Rules for every item in
`internal/notes/core_compiler_roadmap.md`. Also read #94, the #37 coordination and #39 to #42
handoffs, the architecture/codebase/conventions/contributing pages, versioning and compiler
maintenance context. No comments currently exist on #43 itself.

The kernel already uses declared sparse layouts. Replace the temporary typed-header refusal
with stored-size buffers and local layout metadata for inputs and outputs. Keep arbitrary-order
derivative metadata and dense snapshots unchanged. Do not publish sparse names in the guide or
API. CasADi requires CSC and two-dimensional leaves; the question about non-CSC leaves is pending.

- [x] Inspect the clean linked worktree, prerequisites, tracker and existing interfaces.
- [ ] Add differential C/C++ and CasADi tests and observe the current failures.
  Read frozen devrush sparse-signature/export tests for useful problem cases.
- [ ] Publish declared-layout metadata and correct typed buffer sizes, handling shared names,
  batches and empty patterns. Build CasADi sparsity from leaf types.
- [ ] Prove tests detect wrong tables/value ordering by temporary perturbations.
  Run focused tests, dense snapshots, #41 scale gate and custom derivative regression.
- [ ] Run the full pre-merge hook, commit, push and open a draft PR against #42.
- [ ] Apply cross-review with Opus and Sol, fix findings and run short delta checks.
- [ ] Remove this plan, finish the PR evidence and tracker status, mark ready and watch CI.

Phases are sequential. Reviews can run in parallel after all checks pass. Keep an inventory of
header names, table ordering, batch metadata and CasADi choices for the final report, plus any
reviewer disagreements. No merges, tags, main pushes or `gh stack` commands.
