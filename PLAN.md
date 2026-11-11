# Select masks: flush hold

PR #434 covers #364 and folded-in #414 in compiler run section 4 (#24).
The latest user instruction and ~/.scaly-suite/super/flush-common.md override
all previous plan steps about pushing, readiness and queueing.

## Current instructions

- Push nothing, mark nothing ready, queue nothing, open no PR and start no Claude
  review. The coordinator owns the flush order and gives the next push instruction.
- Keep this plan through the push. Finish the local checks and report, then stop.
- Current base is origin/main at 589a2e8f, including #433 and the one-ULP assertion
  fix #445. Rebase merged cleanly. The assertion was not modified by this task.
- The live PR's green checks apply to its older pushed head. They do not validate
  the newer local branch; local source must be pushed before the PR can be ready.

## Complete

- Sol xhigh full correctness review and one low delta: approved. Threads settled.
- Fable high main review and one low delta: approved, no remaining actionable point.
  Threads settled. No further review is authorized by this flush.
- All review fixes are committed. Typed mapped-adjoint casts from main preserve
  numerical parts and activity flags in the common buffer, then restore input dtype.
- Updated test-cost rule applied to repeated grids. One mixed-formal integration
  regression checks the rebase. Focused checks: 95 passed, including snapshots and
  operation coverage. Log /tmp/scaly-434-flush-focused.log. Ruff and ty pass.
- No C snapshots changed or regenerated. Ten extra C/header comparisons against
  current main are byte-identical; proof path /tmp/scaly-434-flush-c-proof.path.
- The prior full hook on the preceding base passed: 6,392 tests, 56 skipped, exit 0.
- The latest section 4 decision is canonical:
  https://github.com/PREDICT-EPFL/scaly/issues/24#issuecomment-6808554341
  The prepared PR body copies its rule paragraph and boundary table verbatim.
  #420 transport is now on main through #433; #442 records its producer limitation.
  #432 and the select half of #435 are coordinated follow-up work after this PR.
  The structural-zero half of #435 is a maintainer decision. #443 is code-size work.
  COND/LOOP transport is section 9. No code was changed to apply this decision.
- Proposed PR body is local only: /tmp/scaly-364-pr-body.md. Semantic section:
  /tmp/scaly-364-zero-rule.md. No PR or #24 comment was updated during the flush.

## Finish before reporting

- [ ] Full hook detached through ~/.scaly-suite/run on this rebased state.
      Log /tmp/scaly-434-flush-suite-nU1reR/final.log; completion /tmp/scaly-434-flush-suite-nU1reR/final.exit.
- [ ] Recheck exact semantic paragraph/table equality against the latest #24 decision.
- [ ] Commit the final local evidence in this plan, keep the plan, and report branch,
      head, ahead/behind main, hook counts, changed snapshot files and any open point.
- [ ] Stop after reporting. Do not wait, poll, watch, push, mark ready or queue.
