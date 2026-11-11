# Select masks: approved, final rebase checks

PR #434 closes #364 and folded-in #414 under compiler run section 4 (#24).
The coordinator explicitly excluded #420, #432 and #435; they remain open and
unstarted. No LOOP/COND protocol or nonsmooth convention changes. Rule and open
boundaries are identical in PR #434 and comment on #24:
https://github.com/PREDICT-EPFL/scaly/issues/24#issuecomment-6794728245

## Complete

- Sol xhigh correctness review and one low delta, approved. Threads settled.
- Fable high main review and one low delta, approved with no remaining actionable
  point. The low delta resumed automatically after quota reset. All threads settled.
- All three main-review findings fixed: stored-coordinate mask conjunctions,
  recursive equal-condition accumulation, and reverse-only doc qualifiers.
- Cheap notes fixed; #443 tracks redundant boolean-mask simplification. Per-map
  simplification cost and the structural-zero caveat are recorded.
- Prior full hook passed: 6,317 tests, 56 skips; all other gates passed.
  Log /tmp/scaly-434-fable-suite-t53Kwr/final.log.
- Conflict-driven rebase after review preserves main's common-dtype casts in mapped
  adjoints, also casting returned flags into the common buffer and restoring each
  input's dtype after assembly. Upstream seed-axis lowering is retained.
- New RULES.md Tests rule 6 applied: shrink the two repeated grids to three cases
  each, keep feature/differential coverage, add one mixed-formal integration check.
- Rebased focused checks: 94 select/snapshot/coverage checks plus the new mixed-formal
  regression pass. Ruff and ty pass. Known one-ULP assertion left unchanged.
- Ten C/header comparisons match current main byte for byte. Proof path is in
  /tmp/scaly-434-final-c-proof.path. Fable also confirmed the prior ten-file proof.

## Finish

- [ ] Run the full hook detached through ~/.scaly-suite/run on the rebased state and
      keep the turn alive until its exit file exists. Do not start another review.
- [ ] Verify the identical semantic text; finalize the verdict line with both models,
      confirmations and the post-review rebase evidence. Keep #443 as a follow-up.
- [ ] Delete this plan in the last commit before ready, then push once (force-with-lease
      against the old pushed head). Link #434 as the next call after pushing.
- [ ] Mark ready, set #364/#414 to In review, queue gh pr merge 434 --squash --auto once.
- [ ] Keep the existing watch, handle failures/ejection/conflicts, report actual merge,
      and unwatch only at final handback. Do not push to main or create tags/releases.
