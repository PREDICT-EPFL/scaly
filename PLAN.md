# Stored patterns in listings and diagnostics

Implements PREDICT-EPFL/scaly#302 under the compiler run plan in
`internal/notes/compiler_run_2026_11_10/plan.md`, currently read from the coordinator worktree.
The issue and the roadmap's #39 paragraph fix the scope: print existing pattern information,
without changing patterns, dense text, generated C, or public interfaces.

- [x] Read the issue, parent, tracker, compiler roadmap, architecture and contribution rules.
- [x] Confirm a clean linked worktree, mark #302 In progress, inspect overlapping work.
- [x] Add failing checks for sparse listings, pattern-only diagnostics and exact dense text.
      Nine sparse checks failed before the fix; the dense check passed unchanged.
- [x] Add a compact suffix using stored-entry count and 12 hex digits of the existing digest,
      shared across printers. No pattern or type semantics change.
- [x] Run focused tests and perturb the formatter to prove the tests fail.
      137 tests passed. Erasing the suffix caused the nine sparse checks to fail again.
- [ ] Run pre-merge checks through `~/.scaly-suite/run`, detached. Final-candidate logs are
      `/tmp/scaly-302.YMUXBc/premerge-main.log` and `.exit`.
      The original unbounded run was killed during memory exhaustion. One detached attempt was
      terminated while waiting for the old lock. Its retry was deliberately stopped when #289
      merged so the rebased candidate can be checked. Neither is a full-suite result.
- [x] Once #289 merges, fetch and rebase only this task's commits onto origin/main, using the
      merge base with `origin/t3code/43-sparse-interfaces`. Exactly two own commits replayed.
      `tests/baseline/c/` has no diff. Re-run checks on this rebased state.
- [ ] Commit, push and open a draft PR against main. Remove this temporary plan before ready.
- [ ] Get the cross-review verdict, fix findings and confirm any fixes with one delta check.
- [ ] With green CI and an approving verdict, queue with gh pr merge --squash and report evidence.

Independent reads and checks can run concurrently. Rebase, edits, review fixes, pushes and queueing
are sequential. Sparse names stay out of published guides and API entries until #46 and #47.

The frozen devrush assembly tests were inspected. They exercise the older dense-only listing and
provide no additional stored-pattern cases to port.

Read `~/.scaly-suite/RULES.md` for the current run-wide rules. Full suites use
`~/.scaly-suite/run`, detached with `nohup setsid`, with the log and exit file under the task's
temporary directory. Poll the exit file inside the turn. Focused tests use at most two workers.
GitHub reads use REST, PR checks are read at most once per five minutes, and the merge request is
`gh pr merge N --squash --auto`, once after the verdict.
