# Stored patterns in listings and diagnostics

Implements PREDICT-EPFL/scaly#302 under the compiler run plan in
`internal/notes/compiler_run_2026_11_10/plan.md`, currently read from the coordinator worktree.
The issue and the roadmap's #39 paragraph fix the scope: print existing pattern information,
without changing patterns, dense text, generated C, or public interfaces.

- [x] Read the issue, parent, tracker, compiler roadmap, architecture and contribution rules.
- [x] Confirm a clean linked worktree, mark #302 In progress, inspect overlapping work.
- [ ] Add failing checks for sparse listings, pattern-only diagnostics and exact dense text.
- [ ] Add a compact suffix using stored-entry count and the existing digest, shared across printers.
- [ ] Run focused tests, perturb the formatter to prove the tests fail, then run pre-merge checks.
- [ ] Once #289 merges, fetch and rebase onto origin/main above the supplied sparse-stack base.
      Re-run the checks and confirm C snapshots remain unchanged.
- [ ] Commit, push and open a draft PR against main. Remove this temporary plan before ready.
- [ ] Get the cross-review verdict, fix findings and confirm any fixes with one delta check.
- [ ] With green CI and an approving verdict, queue with gh pr merge --squash and report evidence.

Independent reads and checks can run concurrently. Rebase, edits, review fixes, pushes and queueing
are sequential. Sparse names stay out of published guides and API entries until #46 and #47.

The frozen devrush assembly tests were inspected. They exercise the older dense-only listing and
provide no additional stored-pattern cases to port.
