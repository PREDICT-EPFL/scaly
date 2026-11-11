# Signed wrapping integration with float32 lowering

Task #340, PR #361. The coordinator requires #360 to land first, then #361, then #358.
Do not push until the coordinator reports #360 merged. The compiler run plan and
`~/.scaly-suite/RULES.md` still govern tests, reviews and queueing.

- [x] Rebase the task branch onto main after #320's squash merge, dropping its old commits.
- [ ] Trial-merge `pr360` into `t3code/340-trial-360`, preserving main's newer compiler work.
- [ ] Resolve the shared integer C spelling. This task owns wrapping add/subtract/multiply
  and negation. #55 owns literals, libm, integer abs/min/max and accumulators. Apply the
  wrapping rule to integer code added by #55 without changing its other decisions.
- [ ] Run formatting, lint, types and focused tests with at most two workers. Verify the
  signed-overflow sanitizer and unchanged float64 snapshots. Record all integration fixes.
- [ ] Have Opus check only changed logic introduced by this integration.
- [ ] Wait for the coordinator's #360 merge notification. Rebase onto main, retain the
  prepared resolutions, remove this temporary plan and run the full hook detached through
  the shared suite runner. Check #361 targets main, push once, queue and watch until merged.

The main rebase preserved carried-loop exit-variable declarations and now passes the
integer-data setting through BLOCK statements, as it already did through FOR bodies.
PR #360's fetched head is `pr360`. The task branch before its scratch merge is
`t3code/340-wrap-signed-overflow`. The published PR remains untouched during preparation.

Existing approvals cover the signed-wrap implementation and its two fixed findings.
Review follow-ups are #363 and #365. No new full review is needed for unchanged logic.
