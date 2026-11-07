# Stored sparse patterns, issue #39

Branch `t3code/39-sparse-pattern`, base main, first layer of #39 through #43.
The decided contract is in `internal/notes/core_compiler_roadmap.md`, "Sparse tensors" and
"Rules for every item". Read the issue, parent #37, closed prerequisites #11 and #38,
architecture, codebase, conventions, contributing and compiler maintenance context.

- [x] Confirm the linked worktree, clean base, prerequisites and tracker status.
- [x] Add failing tests for canonicalization, identity, storage, fixed rules, refusal,
  structural keys, substitution and verifier enforcement. Use NumPy differential proof.
- [x] Implement canonical patterns in `ir/types.py` and shared pattern algebra, with focused
  expression construction and verification changes. Keep arbitrary derivative output ordering.
- [x] Audit mathematical sizes versus stored buffer lengths. List each affected owning area
  in the pull request and defer lowering and boundary work to #42.
- [x] Prove ten invariant groups fail under perturbation. The full suite passed, 2514 passed and
  91 solver skips, including byte-identical C snapshots. Types, Ruff and docs passed.
  The pre-merge hook is still running benchmark smoke before its remaining checks.
- [ ] Push a draft PR against main, cross-review under the skill's model rules, fix findings,
  confirm fixes, remove this plan, and mark ready with tracker evidence.

No edits in `src/scaly/passes/lowering/`, no support analysis, sparse leaves, boundary or
lowering implementation. No C snapshot regeneration. Sparse API names remain filtered from
published pages. #25 shares `ir/expr.py`; changes there stay focused.

Implementation phases are sequential. Independent validation commands may run together.
The review uses fresh read-only reviewers after checks pass. The coordinator owns stack links.

Boundary protection: `ConcreteFunction.__post_init__` rejects stored-pattern inputs or outputs
until #42 implements their buffers. This is a refusal guard, not sparse leaf support.
The new API fields and canonicalization are filtered, and class source listings are hidden
because they otherwise publish those names despite the member filters.

Audit findings for the PR: JIT shapes/allocation, C/C++ wrapper lengths, CasADi input/output
layouts; VMAP windows and scratch; lowering input/output/private buffers, copy loops,
elementwise/gather/scatter domains and movement/contraction offsets. All remain unchanged for #42
and #43. Logical indexing and derivative dimensions retain mathematical size; their full-size
allocations belong to #41. Solver decision-vector dimensions and tree leaf counts retain size.
