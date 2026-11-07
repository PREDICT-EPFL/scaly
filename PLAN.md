# Stored sparse patterns, issue #39

Branch `t3code/39-sparse-pattern`, base main, first layer of #39 through #43.
The decided contract is in `internal/notes/core_compiler_roadmap.md`, "Sparse tensors" and
"Rules for every item". Read the issue, parent #37, closed prerequisites #11 and #38,
architecture, codebase, conventions, contributing and compiler maintenance context.

- [x] Confirm the linked worktree, clean base, prerequisites and tracker status.
- [ ] Add failing tests for canonicalization, identity, storage, fixed rules, refusal,
  structural keys, substitution and verifier enforcement. Use NumPy differential proof.
- [ ] Implement canonical patterns in `ir/types.py` and shared pattern algebra, with focused
  expression construction and verification changes. Keep arbitrary derivative output ordering.
- [ ] Audit mathematical sizes versus stored buffer lengths. List each affected owning area
  in the pull request and defer lowering and boundary work to #42.
- [ ] Prove gates fail under perturbation; run all pre-merge checks and byte-identical C snapshots.
- [ ] Push a draft PR against main, cross-review under the skill's model rules, fix findings,
  confirm fixes, remove this plan, and mark ready with tracker evidence.

No edits in `src/scaly/passes/lowering/`, no support analysis, sparse leaves, boundary or
lowering implementation. No C snapshot regeneration. Sparse API names remain filtered from
published pages. #25 shares `ir/expr.py`; changes there stay focused.

Implementation phases are sequential. Independent validation commands may run together.
The review uses fresh read-only reviewers after checks pass. The coordinator owns stack links.
