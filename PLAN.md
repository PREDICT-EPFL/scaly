# Issue #61

Implement the decided Foundations item in `internal/notes/core_compiler_roadmap.md` against main. Read architecture, codebase, conventions, contributing, compiler maintenance context, the issue and lane brief. Preserve C snapshots and avoid #26/#38 implementation areas.

- [x] Inspect issue, dependencies, worktree and overlapping PRs.
- [x] Add failing tests for non-float64 Function leaves, active solver derivatives and NaN extrema folding. Port the NaN case from devrush's `tests/core/codegen/test_elementwise_edge_cases.py`.
- [x] Add focused refusals and fmin/fmax folds, then prove each guard fails when removed.
- [x] Run pre-merge hook, verify snapshots, commit and open draft PR against main.
- [ ] Obtain one independent compiler review, address findings and check fixes.
- [ ] Remove plan, mark ready, update issue status and watch PR.

Tests precede implementation. Validation precedes review. No parallel implementation is needed. No merges, main pushes or tags.

Review by Fable 5.1 at high effort requested documentation corrections and removal of the redundant JIT check. Fixed the pages, rely on artifact lowering for JIT refusal, delete duplicate typed refusal tests, and pin the bare-node zero-cotangent case. Run final checks and one delta review, then remove this plan and mark ready.
