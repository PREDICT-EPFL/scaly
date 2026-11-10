# Safe JIT cache invalidation

Implements #193 under section 6 (#295) of
`internal/notes/compiler_run_2026_11_10/plan.md`. The issue body and #154 decide the design.
Base and pull request target are `origin/main` and `main`. This branch is not a stack.

Invalidation will leave artifact directories untouched and always advance the existing
process-local epoch. A later build in this process gets a fresh path. Other processes keep
their source, temporary output and published library throughout compilation and loading.
Old artifacts need manual cleanup when no process uses the cache.
After one process recompiles, a fresh process still reuses the original shared artifact.
Atomic publication ensures that artifact is complete. The cross-process reuse test pins this.

## Work

- [x] Read the run plan, issue, Windows contract, project rules and surrounding code/tests.
- [x] Add deterministic two-process coverage at linking, publication and loading. See it fail
      on the old implementation. Both processes compare numerical results with NumPy.
- [x] Retain artifact directories and advance the epoch unconditionally. Update recompilation
      tests and affected documentation. Preserve cache reuse and missing-entry invalidation.
- [ ] Run focused tests, the full pre-merge hook and byte-identical snapshot checks.
- [ ] Commit, push and open a draft pull request, then get the cross-review and any delta check.
- [ ] Remove this temporary plan, mark ready, and queue with `gh pr merge --squash` once checks
      and the review pass. Report evidence and cleanup limitations to the coordinator.

The phases depend on the preceding phase. #64 and #269 share `codegen/jit.py`; #64 must
preserve unconditional epoch advancement without deleting artifact directories. The sparse
stack also edits `function/concrete.py` and `codegen/jit.py`; rebase as necessary. No changes
to the graph digest, solver fingerprints or hash-seed determinism belong here.

## Evidence so far

- All three two-process regressions failed on the original implementation for the intended
  missing-file/directory errors. With the fix, 36 JIT tests and 8 C snapshot tests passed.
- Ruff and ty passed. The full pre-merge run was interrupted by the provider stream closing.
  Repeat it detached and inspect `/tmp/scaly-193-pre-merge.log` and `.exit`.
- Commit, push, pull request creation and cross-review remain. Only the initial plan is committed.
