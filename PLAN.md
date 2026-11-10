# Short producer range fusion

Issue #223, based on main. The compiler run plan overrides snapshot serialization,
defers run-time measurements to milestone checks, and authorizes queuing this PR.
Scope is region selection in `fuse_ranges`, its tests and regenerated snapshots.
Cell simulation belongs to #183, lowering-time fusion to #69.

- [x] Read the task, run plan, compiler roadmap and repository conventions.
- [x] Capture chain M=3 and M=5 operation counts, source size and C compile time.
- [x] Add a numerical and structural regression that fails under the old threshold.
- [x] Select cheap upstream assembly producers without an iteration threshold.
- [x] Confirm independent upstream work stays outside, and perturb the new gate.
- [x] Regenerate C snapshots and the test collection baseline.
- [ ] Run the detached pre-merge checks through the shared suite runner.
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
The first full pre-merge passed with 2,605 tests and 52 skips plus smoke and docs.
The current fix needs its final full check through `~/.scaly-suite/run`.
Both original and changed generated chain kernels pass the dense lower-triangle
reference, including omitted-zero positions, with errors below 5e-16.

The independent sol review found that the already-produced-buffer exception
could repeat an expensive initializer in every mapped stage. Removed that
exception and added a verified shared-assembly regression, red before the fix
and green afterward. All 22 fusion tests pass; refreshed chain M=3/M=5 source
hashes and counts match the measured after-state, and all snapshots are unchanged.
The sol delta check approved the expensive-initializer fix. Opus reproduced
the same issue and identified unbounded repeated cheap work. Region selection
now caps each upstream scalar expression at eight operations, counting through
intermediate buffers and keeping larger producers materialized. Four new
structural cases cover one expression and split chains at two sizes; all fail
with the bound removed, and all 26 fusion tests pass with it.

The final chain M=3/M=5 source hashes, byte counts and operation counts exactly
match the measured after-state. All 14 snapshots remain unchanged. Final
generation samples are recorded in /tmp/scaly-223-evidence/final/. The final
full suite needs to run on this state, followed by the one Opus delta check
covering both act-on findings. The superseded check was stopped. The PR must
report the M=3 four-to-two lane change on 256-register-slot targets and the
generation-time samples as well as the already recorded static evidence.
