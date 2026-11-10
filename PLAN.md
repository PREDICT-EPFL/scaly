# Short producer range fusion

Issue #223, based on main. The compiler run plan overrides snapshot serialization,
defers run-time measurements to milestone checks, and authorizes queuing this PR.
Scope is region selection in `fuse_ranges`, its tests and regenerated snapshots.
Cell simulation belongs to #183, lowering-time fusion to #69.

- [x] Read the task, run plan, compiler roadmap and repository conventions.
- [x] Capture chain M=3 and M=5 operation counts, source size and C compile time.
- [x] Add a numerical and structural regression that fails under the old threshold.
- [x] Select producers by their relationship to buffers already in the region.
- [x] Confirm independent upstream work stays outside, and perturb the new gate.
- [x] Regenerate C snapshots and the test collection baseline.
- [ ] Run the shared-lock detached pre-merge checks on the intended push state.
- [ ] Open a draft PR, obtain Opus high and independent sol correctness reviews,
      address findings and confirm any fixes with a focused delta check.
- [ ] Remove this plan, publish the evidence and verdict, mark ready, and queue.

The PR records each criterion, deferred timing, M=5's operation counts and the
snapshot diff the reviewers read. Rebase and regenerate if another task changes
the snapshots first. Never push to main or edit another task's implementation.

Tracker Project access currently fails with `unknown owner type`; issue records
remain accessible through the REST API.

Evidence is in `/tmp/scaly-223-evidence/` until it is copied into the PR. The
14 regenerated snapshots are unchanged. Focused fusion tests passed 21/21;
515 compiler, integration, snapshot and import checks passed. Ruff and ty passed.
The first detached lock wait ended with exit 143; its required retry is queued.
Both original and changed generated chain kernels pass the dense lower-triangle
reference, including omitted-zero positions, with errors below 5e-16.
