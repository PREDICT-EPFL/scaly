# Signed wrapping integration with float32 lowering

Task #340, PR #361. The coordinator requires #360 to land first, then #361, then #358.
Do not push until the coordinator reports #360 merged. The compiler run plan and
`~/.scaly-suite/RULES.md` still govern tests, reviews and queueing.

- [x] Rebase the task branch onto main after #320's squash merge, dropping its old commits.
- [x] Trial-merge `pr360` into `t3code/340-trial-360`, preserving main's newer compiler work.
- [x] Resolve the shared integer C spelling. This task owns wrapping add/subtract/multiply
  and negation. #55 owns literals, libm, integer abs/min/max and accumulators. Apply the
  wrapping rule to integer code added by #55 without changing its other decisions.
- [x] Run formatting, lint, types and focused tests with at most two workers. Verify the
  signed-overflow sanitizer and unchanged float64 snapshots. Record all integration fixes.
- [x] Have Opus check only changed logic introduced by this integration.
- [ ] Wait for the coordinator's #360 merge notification. Rebase onto main, retain the
  prepared resolutions, remove this temporary plan and run the full hook detached through
  the shared suite runner. Check #361 targets main, push once, queue and watch until merged.

The main rebase preserved carried-loop exit-variable declarations and now passes the
integer-data setting through BLOCK statements, as it already did through FOR bodies.
PR #360's fetched head is `pr360`. The task branch before its scratch merge is
`t3code/340-wrap-signed-overflow`. The published PR remains untouched during preparation.

Existing approvals cover the signed-wrap implementation and its two fixed findings.
Review follow-ups are #363 and #365. No new full review is needed for unchanged logic.

The fetched #360 still includes the old #30 history. Its trial merge required 16 file
resolutions. Preserve main's sparse support, carried-loop verification and array-based
fusion, combine float32 folding/libm/literals with the signed wrapping rule, and migrate
newer ProgramNode call sites to explicit keyword dtypes. The shared reduction emitter
now creates its identity with constant(0, value.dtype), so integer accumulators start with
an integer constant. No floating reduction order changed.

Focused integration: 635 tests passed, covering signed arithmetic, float32 operations,
both C dialects, carried loops, reductions, fusion, masks and C snapshots. The 104 import
checks also passed. Ruff and ty pass; complete collection contains 3950 IDs. New sanitizer
cases cover sums at both signed limits and nested-block arithmetic at both widths. Both
new gates were shown to fail by removing wrapping or its block propagation.

After #360 merges, return to the task branch and rebase that branch onto main. Do not
push or rebase the scratch merge history. Use the scratch branch to resolve shared code,
retain its new tests/codegen/test_integer_arithmetic.py cases, and check the reduction
identity on the final main base. Regenerate node IDs after moving the new tests. The
scratch branch's other migration changes belong to #55's integration and must not replace
unrelated changes that land on main meanwhile.

Opus approved the new integration logic at xhigh effort, ran all 20 integer arithmetic
tests successfully, and independently reproduced the block-propagation and sum-wrapping
failures with scratch pytest plugins. No action remains from that check. Its review
thread has been settled. The original task branch is checked out again; the reviewed
trial and all prepared code/test changes remain on t3code/340-trial-360. No push was made.
