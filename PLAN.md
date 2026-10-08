# Issue #61

Implement the decided Foundations item in `internal/notes/core_compiler_roadmap.md` against main. Read architecture, codebase, conventions, contributing, compiler maintenance context, the issue and lane brief. Preserve C snapshots and avoid #26/#38 implementation areas.

- [x] Inspect issue, dependencies, worktree and overlapping PRs.
- [x] Add failing tests for non-float64 Function leaves, active solver derivatives and NaN extrema folding. Port the NaN case from devrush's `tests/core/codegen/test_elementwise_edge_cases.py`.
- [x] Add focused refusals and fmin/fmax folds, then prove each guard fails when removed.
- [ ] Run pre-merge hook, verify snapshots, commit and open draft PR against main.
- [ ] Obtain one independent compiler review, address findings and check fixes.
- [ ] Remove plan, mark ready, update issue status and watch PR.

Tests precede implementation. Validation precedes review. No parallel implementation is needed. No merges, main pushes or tags.
