# Preserve C wrapper typedef visibility

Issue #344 was assigned after the review of #325. This branch starts from #325, which has both approvals and entered the merge queue. The C++ workspace issue #343 has a separate branch and checkout. The compiler-run plan and `~/.scaly-suite/RULES.md` apply.

The C wrapper renderer currently uses buffer names as parameter names. A parameter can hide a complete C typedef used by a later parameter. Allocate the parameter names against all of the actual typedefs while preserving the exported entry and the existing buffer typedef spellings. Keep the fix in `codegen/aot.py`; do not change the shared C++ buffer aliases unnecessarily.

- [x] Read the task, project conventions, generated naming design and implementation; confirm there are no blockers or parent.
- [ ] Add failing C compile-and-call regressions for input and output parameters hiding later buffer or workspace typedefs.
- [ ] Allocate C wrapper parameters through the existing `NameScope`, prove the test fails with the fix reverted, and keep snapshots unchanged. Regenerate collected test IDs.
- [ ] Run focused checks, then the detached full pre-merge runner. Remove this plan before the first push. Push one complete state for review.
- [ ] Obtain the prescribed cross-review, batch any fixes, and queue once reviewed and green. Retarget to main after #325 merges without rebasing merely because main moved.
