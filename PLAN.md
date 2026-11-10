# Hash-seed-independent generated C

Implement #163 under section 6 (#295) of the compiler run plan. Base and PR base are main.
The compiler run plan overrides the implement skill, including authorization to queue the PR
after cross-review and green checks. No JIT key, index, or invalidation changes.

Read: compiler run plan, architecture, codebase, conventions, contributing, issue tracking,
compiler maintenance context, core compiler roadmap Foundations #64 and Rules for every item.

- [x] Inspect issue, dependencies, checkout, and overlapping PRs. No blockers; sparse stack
      overlap and snapshot regeneration are allowed by the run plan.
- [x] Add a subprocess corpus test covering mapped stage, nested call/map derivatives, and
      other representative Functions. Confirm failure on the original implementation.
- [x] Locate the first unstable compiler pass and fix iteration using graph or statement order.
      Audit other seed-dependent iteration in lowering, program passes, and rendering.
- [ ] Verify differential behavior, mutation detection, snapshots, full suite, and merge hooks.
      If snapshots move, review why and collect the run plan's required evidence.
- [ ] Commit, push, open and link draft PR. Obtain one correctness cross-review and a delta
      check if needed. Remove this temporary plan before marking ready.
- [ ] Update issue status, handle CI/rebase, queue squash merge with green checks, and report
      evidence, remaining points, and any source:agent follow-ups to the coordinator.

Each phase depends on the previous one, except the iteration audit can accompany diagnosis.

Run-wide validation rule from the section coordinator:

- One full suite across the run at a time. Every new full hook uses
  `flock ~/.scaly-suite.lock env PYTEST_XDIST_AUTO_NUM_WORKERS=8 wt hook pre-merge --yes`.
  A bare full `uv run pytest -n=auto` takes the same lock. Run detached and poll its exit file
  with short commands, since acquiring the lock may take longer than a tool call.
- Focused tests stay unlocked with at most two workers. Never use `-n=auto` outside the lock.
- Run full pre-merge only on a state intended for push. Retry a killed worker or timeout once
  before treating it as a real failure.
- The currently running detached two-worker service is explicitly allowed to finish. Any
  replacement or later full run must take the lock and use eight workers.

Evidence so far:

- The new subprocess test failed before the fix on map_of_call_1 at seed 1. An aggregate probe
  also showed the mapped stage differing at seeds 4, 5, and 42. Both map_of_call seed counts
  differed at seeds 1, 4, 5, 7, and 42 relative to seed 0.
- Fusion now follows the declared output order. Eleven Functions and their C headers match
  across seeds 0, 1, 4, 5, 7, and 42. Reverting just this change made the regression fail again.
- The audit found workspace packing updating dependencies while iterating a set of writes.
  A direct Program regression, swapping two buffer names, failed before its fix. Dependency
  closures now use values from before the statement. An in-memory mutation of this fix failed
  the regression again. The corresponding frozen-devrush test covered scan, which main lacks;
  the small Program reproduction covers the same packing hazard on main's IR.
- The first narrow run passed 56 tests, including the existing NumPy references for both
  reported examples and all C snapshots. No snapshot regeneration so far.
- Foreground merge hooks were interrupted by two provider-stream closures and have no final
  result. A fresh final run is detached with nohup/setsid. Poll
  /tmp/scaly-163-pre-merge.exit for its exit status and /tmp/scaly-163-pre-merge.log for evidence.
  Check its process before restarting. Do not start another suite alongside this run.
- The detached 16-worker suite lost a worker during severe machine memory pressure, with RAM
  nearly full and swap exhausted. It was stopped after that failure; its log is preserved in
  /tmp/scaly-163-pre-merge-16-workers.log. The replacement hook is detached with
  PYTEST_XDIST_AUTO_NUM_WORKERS=2, using the same final log and exit-file paths above.
- That nohup/setsid run also vanished during the next all-Codex stream closure without an exit
  result. Its partial log is /tmp/scaly-163-pre-merge-2-workers-interrupted.log. Validation now
  runs outside Codex's process tree as the transient systemd user service
  scaly-163-pre-merge.service, still with two workers and the same final log/exit paths. Check
  systemctl --user show scaly-163-pre-merge.service before restarting it.
- The systemd run passed lint, formatting, types, and the final full suite: 2606 passed,
  52 skipped. Benchmark smoke has passed its four problem-oracle checks and is still running
  the remaining cells. Draft PR and cross-review can proceed with the review prerequisites
  green; the remaining hooks must finish before the PR is ready to queue.
